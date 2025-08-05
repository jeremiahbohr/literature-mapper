"""
mapper.py – core logic for Literature Mapper
Clean architecture focused on reliability and predictable behavior.
"""

import os
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import google.generativeai as genai
import pandas as pd
import pypdf
from tqdm import tqdm

from .ai_prompts import get_analysis_prompt
from .config import DEFAULT_MAX_FILE_SIZE, DEFAULT_MAX_RETRIES, DEFAULT_RETRY_DELAY
from .database import (
    Author,
    Concept,
    Paper,
    PaperAuthor,
    PaperConcept,
    get_db_session,
)
from .exceptions import APIError, DatabaseError, PDFProcessingError, ValidationError
from .validation import validate_api_key, validate_json_response, validate_pdf_file

logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    processed: int = 0
    failed: int = 0
    skipped: int = 0

    @property
    def total(self) -> int:
        return self.processed + self.failed + self.skipped

@dataclass
class CorpusStatistics:
    total_papers: int
    total_authors: int
    total_concepts: int

class PDFProcessor:
    """Handles PDF text extraction with comprehensive error handling."""

    def __init__(self, max_file_size: int = DEFAULT_MAX_FILE_SIZE):
        self.max_file_size = max_file_size

    def extract_text(self, pdf_path: Path) -> str:
        """Extract text from a PDF or raise PDFProcessingError."""
        if not validate_pdf_file(pdf_path, self.max_file_size):
            raise PDFProcessingError(
                "PDF validation failed", pdf_path=pdf_path, error_type="validation"
            )

        try:
            with open(pdf_path, "rb") as f:
                reader = pypdf.PdfReader(f)

                if reader.is_encrypted:
                    raise PDFProcessingError(
                        "PDF is encrypted", pdf_path=pdf_path, error_type="encryption"
                    )

                text_parts = []
                for page in reader.pages:
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text_parts.append(page_text)
                    except Exception:
                        continue  # skip bad page but don't abort whole file

                full_text = "\n".join(text_parts).strip()
                if len(full_text) < 100:
                    raise PDFProcessingError(
                        "Insufficient text extracted",
                        pdf_path=pdf_path,
                        error_type="extraction",
                    )

                # normalize whitespace
                return re.sub(r"\s+", " ", full_text)

        except pypdf.errors.PdfReadError as e:
            raise PDFProcessingError(
                "PDF read error",
                pdf_path=pdf_path,
                error_type="corruption",
            ) from e
        except PDFProcessingError:
            raise  # re-raise our own exceptions
        except Exception as e:
            raise PDFProcessingError(
                f"Unexpected PDF processing error: {e}",
                pdf_path=pdf_path,
                error_type="unknown",
            ) from e

class AIAnalyzer:
    """Handle AI analysis with robust retry logic."""

    def __init__(self, model_name: str, max_retries: int = DEFAULT_MAX_RETRIES, retry_delay: int = DEFAULT_RETRY_DELAY):
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        # Don't create model instance here - create fresh for each analysis

    def analyze(self, text: str) -> dict:
        """Analyze text and return validated JSON response."""
        prompt = get_analysis_prompt().format(text=text[:50000])  # Reasonable text limit
        
        config = genai.types.GenerationConfig(
            max_output_tokens=4096,
            temperature=0.1,
            top_p=0.8,
        )

        for attempt in range(self.max_retries):
            try:
                # Create fresh model instance for each analysis to avoid memory accumulation
                model = genai.GenerativeModel(self.model_name)
                response = model.generate_content(prompt, generation_config=config)
                if not response.text:
                    raise APIError("Empty response from AI model")

                # Clean response text
                cleaned = re.sub(r"```json\s*|\s*```", "", response.text.strip())
                data = json.loads(cleaned)
                result = validate_json_response(data)
                
                # Explicitly clear references
                del model, response, cleaned, data
                return result

            except json.JSONDecodeError as e:
                if attempt < self.max_retries - 1:
                    logger.warning("JSON decode error, retry %d/%d", attempt + 1, self.max_retries)
                    time.sleep(self.retry_delay)
                    continue
                raise APIError("Failed to parse AI response as JSON after retries") from e
            
            except ValidationError:
                # Don't retry validation errors - they won't improve
                raise
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning("AI call failed, retry %d/%d: %s", attempt + 1, self.max_retries, e)
                    time.sleep(self.retry_delay)
                    continue
                raise APIError(f"AI analysis failed after retries: {e}") from e


class LiteratureMapper:
    """High-level interface for corpus management."""

    def __init__(
        self,
        corpus_path: str,
        model_name: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
    ):
        self.corpus_path = Path(corpus_path).resolve()
        self.corpus_path.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        
        self._setup_api(api_key)
        self.pdf_processor = PDFProcessor()
        self.ai_analyzer = AIAnalyzer(model_name)

    def _setup_api(self, api_key: Optional[str]) -> None:
        """Setup and validate API configuration."""
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValidationError("Gemini API key missing", field="api_key")

        if not validate_api_key(key):
            raise ValidationError("Invalid API key format", field="api_key")

        try:
            genai.configure(api_key=key)
            # Quick validation test with minimal resource usage
            test_model = genai.GenerativeModel(self.model_name)
            response = test_model.generate_content(
                "test", generation_config=genai.types.GenerationConfig(max_output_tokens=1)
            )
            # Clean up test objects
            del test_model, response
            logger.info("API and model '%s' validated", self.model_name)
        except Exception as e:
            raise APIError(f"Failed to configure or validate Gemini API: {e}") from e

    def _get_existing_pdf_paths(self, session) -> set[str]:
        """Return absolute paths of already-processed PDFs."""
        existing_paths = set()
        for path in session.query(Paper.pdf_path).scalars().all():
            if path is not None:
                # Normalize to absolute path for consistent comparison
                abs_path = str(Path(path).resolve())
                existing_paths.add(abs_path)
        return existing_paths

    def _save_paper_to_db(self, session, pdf_path: Optional[Path], analysis: dict) -> None:
        """Insert Paper plus authors/concepts."""
        # Store absolute path for consistency
        stored_path = str(pdf_path.resolve()) if pdf_path else None
        
        paper = Paper(
            pdf_path=stored_path,
            title=analysis["title"],
            year=analysis["year"],
            journal=analysis.get("journal"),
            abstract_short=analysis.get("abstract_short"),
            core_argument=analysis["core_argument"],
            methodology=analysis["methodology"],
            theoretical_framework=analysis["theoretical_framework"],
            contribution_to_field=analysis["contribution_to_field"],
            doi=analysis.get("doi"),
            citation_count=analysis.get("citation_count"),
        )
        session.add(paper)
        session.flush()

        # Add authors
        for author_name in analysis.get("authors", []):
            if not author_name.strip():
                continue
            author = session.query(Author).filter_by(name=author_name.strip()).first()
            if not author:
                author = Author(name=author_name.strip())
                session.add(author)
                session.flush()
            session.add(PaperAuthor(paper_id=paper.id, author_id=author.id))

        # Add concepts
        for concept_name in analysis.get("key_concepts", []):
            if not concept_name.strip():
                continue
            concept = session.query(Concept).filter_by(name=concept_name.strip()).first()
            if not concept:
                concept = Concept(name=concept_name.strip())
                session.add(concept)
                session.flush()
            session.add(PaperConcept(paper_id=paper.id, concept_id=concept.id))

        logger.info("Saved paper: %s", analysis["title"])

    def process_new_papers(self, recursive: bool = False) -> ProcessingResult:
        pattern = "**/*.pdf" if recursive else "*.pdf"
        all_pdfs = list(self.corpus_path.glob(pattern))
        
        with get_db_session(self.corpus_path) as session:
            existing_paths = self._get_existing_pdf_paths(session)
            # Compare absolute paths consistently
            new_pdfs = [p for p in all_pdfs if str(p.resolve()) not in existing_paths]

            if not new_pdfs:
                logger.info("No new papers to process")
                return ProcessingResult()

            logger.info("Processing %d new PDFs", len(new_pdfs))
            result = ProcessingResult()

            for pdf_path in tqdm(new_pdfs, desc="Processing papers", unit="pdf"):
                try:
                    text = self.pdf_processor.extract_text(pdf_path)
                    analysis = self.ai_analyzer.analyze(text)
                    self._save_paper_to_db(session, pdf_path, analysis)
                    session.commit()  # Commit after each successful paper
                    result.processed += 1
                    
                except PDFProcessingError as e:
                    logger.warning("Skipped %s: %s", pdf_path.name, e.user_message)
                    result.skipped += 1
                    
                except (APIError, ValidationError) as e:
                    logger.error("Failed %s: %s", pdf_path.name, e.user_message)
                    result.failed += 1
                    
                except DatabaseError as e:
                    if 'UNIQUE constraint failed' in str(e):
                        logger.warning("Duplicate paper skipped: %s", pdf_path.name)
                        result.skipped += 1
                    else:
                        logger.error("Database error for %s: %s", pdf_path.name, e.user_message)
                        result.failed += 1

            logger.info(
                "Processing complete: processed=%d failed=%d skipped=%d",
                result.processed, result.failed, result.skipped,
            )
            return result

    def get_all_analyses(self) -> pd.DataFrame:
        """Return full joined view of papers + authors + concepts."""
        query = """
        SELECT
            p.id,
            p.pdf_path,
            p.title,
            p.year,
            p.journal,
            p.abstract_short,
            p.core_argument,
            p.methodology,
            p.theoretical_framework,
            p.contribution_to_field,
            p.doi,
            p.citation_count,
            GROUP_CONCAT(DISTINCT a.name)  AS authors,
            GROUP_CONCAT(DISTINCT c.name)  AS key_concepts
        FROM papers              p
        LEFT JOIN paper_authors  pa ON p.id = pa.paper_id
        LEFT JOIN authors        a  ON pa.author_id = a.id
        LEFT JOIN paper_concepts pc ON p.id = pc.paper_id
        LEFT JOIN concepts       c  ON pc.concept_id = c.id
        GROUP BY p.id
        ORDER BY p.year DESC, p.title
        """
        with get_db_session(self.corpus_path) as session:
            return pd.read_sql(query, session.bind)

    def export_to_csv(self, output_path: str) -> None:
        """Export current corpus to CSV."""
        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        self.get_all_analyses().to_csv(out, index=False)
        logger.info("Corpus exported to %s", out)

    def add_manual_entry(self, title: str, authors: list[str], year: int, **kwargs) -> None:
        """Insert a paper without a PDF file."""
        if not title.strip():
            raise ValidationError("Title cannot be empty", field="title")
        if not 1900 <= year <= 2030:
            raise ValidationError("Year must be between 1900 and 2030", field="year")
        if not authors or not any(a.strip() for a in authors):
            raise ValidationError("At least one author required", field="authors")

        analysis = {
            "title": title.strip(),
            "authors": [a.strip() for a in authors if a.strip()],
            "year": year,
            "journal": kwargs.get("journal"),
            "abstract_short": kwargs.get("abstract_short"),
            "core_argument": kwargs.get(
                "core_argument", "Manually entered – no automated analysis available"
            ),
            "methodology": kwargs.get("methodology", "Not specified"),
            "theoretical_framework": kwargs.get("theoretical_framework", "Not specified"),
            "contribution_to_field": kwargs.get("contribution_to_field", "Not specified"),
            "key_concepts": kwargs.get("key_concepts", []),
            "doi": kwargs.get("doi"),
            "citation_count": kwargs.get("citation_count"),
        }
        
        with get_db_session(self.corpus_path) as session:
            self._save_paper_to_db(session, None, analysis)
            session.commit()

    def update_papers(self, paper_ids: list[int], updates: dict) -> None:
        """Bulk update allowed columns for given paper IDs."""
        if not paper_ids or not updates:
            raise ValidationError("No paper IDs or updates provided")

        allowed = {
            "title", "year", "journal", "abstract_short", "core_argument",
            "methodology", "theoretical_framework", "contribution_to_field",
            "doi", "citation_count",
        }
        if bad := (set(updates) - allowed):
            raise ValidationError(f"Invalid fields: {', '.join(bad)}")

        with get_db_session(self.corpus_path) as session:
            count = session.query(Paper).filter(Paper.id.in_(paper_ids)).count()
            if count != len(paper_ids):
                raise ValidationError("Some paper IDs do not exist")

            session.query(Paper).filter(Paper.id.in_(paper_ids)).update(
                updates, synchronize_session=False
            )
            session.commit()
            logger.info("Updated %d papers", len(paper_ids))

    def search_papers(self, column: str, query: str) -> pd.DataFrame:
        """Case-insensitive LIKE search over a whitelisted column."""
        searchable = {
            "title", "core_argument", "methodology", "theoretical_framework",
            "contribution_to_field", "journal", "abstract_short",
        }
        if column not in searchable:
            raise ValidationError(
                f"Column '{column}' is not searchable. Valid: {', '.join(searchable)}"
            )
        if not query.strip():
            raise ValidationError("Search query cannot be empty")

        with get_db_session(self.corpus_path) as session:
            ilike_filter = getattr(Paper, column).ilike(f"%{query.strip()}%")
            matching = session.query(Paper).filter(ilike_filter).all()
            if not matching:
                return pd.DataFrame()

            ids = [p.id for p in matching]
            select_cols = [
                "p.id", "p.title", "p.year",
                "GROUP_CONCAT(DISTINCT a.name) AS authors",
                "GROUP_CONCAT(DISTINCT c.name) AS key_concepts",
            ]
            if column != "title":
                select_cols.insert(2, f"p.{column}")

            sql = f"""
            SELECT {', '.join(select_cols)}
            FROM papers p
            LEFT JOIN paper_authors pa ON p.id = pa.paper_id
            LEFT JOIN authors a        ON pa.author_id = a.id
            LEFT JOIN paper_concepts pc ON p.id = pc.paper_id
            LEFT JOIN concepts c       ON pc.concept_id = c.id
            WHERE p.id IN ({', '.join(map(str, ids))})
            GROUP BY p.id
            ORDER BY p.year DESC
            """
            return pd.read_sql(sql, session.bind)

    def get_statistics(self) -> CorpusStatistics:
        """Get corpus statistics."""
        with get_db_session(self.corpus_path) as session:
            return CorpusStatistics(
                total_papers=session.query(Paper).count(),
                total_authors=session.query(Author).count(),
                total_concepts=session.query(Concept).count(),
            )