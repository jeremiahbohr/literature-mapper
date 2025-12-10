"""
mapper.py -- Core logic for Literature Mapper
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
import sqlalchemy as sa
from tqdm import tqdm

from .ai_prompts import (
    get_analysis_prompt, 
    get_kg_prompt,
)
from .config import DEFAULT_MAX_FILE_SIZE, DEFAULT_MAX_RETRIES, DEFAULT_RETRY_DELAY, load_config
from .database import (
    Author,
    Concept,
    Paper,
    PaperAuthor,
    PaperConcept,
    KGNode,
    KGEdge,
    get_db_session,
)
from .exceptions import APIError, DatabaseError, PDFProcessingError, ValidationError
from .validation import validate_api_key, validate_json_response, validate_pdf_file, validate_kg_response
from .embeddings import EmbeddingGenerator, cosine_similarity
from .agents import ArgumentAgent, ValidationAgent

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

            # normalize whitespace but preserve newlines
            # Replace multiple newlines with double newline (paragraph break)
            text = re.sub(r"\n\s*\n", "\n\n", full_text)
            return text.strip()

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
    
    def extract_doi(self, pdf_path: Path) -> Optional[str]:
        """
        Extract DOI from PDF first page.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            DOI string (normalized) or None if not found
        """
        try:
            with open(pdf_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                
                if reader.is_encrypted or len(reader.pages) == 0:
                    return None
                
                # Extract first page text
                first_page = reader.pages[0].extract_text()
                
                if not first_page:
                    return None
                
                # DOI pattern: 10.NNNN/... (where NNNN is at least 4 digits)
                # Common formats:
                # - doi:10.1234/abcd
                # - DOI: 10.1234/abcd
                # - https://doi.org/10.1234/abcd
                # - https://dx.doi.org/10.1234/abcd
                doi_pattern = r'(?:doi\.org/|doi:|DOI:?\s*)?(10\.\d{4,}/[^\s\]},;"\'<>]+)'
                
                match = re.search(doi_pattern, first_page, re.IGNORECASE)
                
                if match:
                    doi = match.group(1)
                    # Clean up common trailing characters
                    doi = doi.rstrip('.,;')
                    logger.debug(f"Extracted DOI: {doi} from {pdf_path.name}")
                    return doi
                
                return None
                
        except Exception as e:
            logger.warning(f"Could not extract DOI from {pdf_path.name}: {e}")
            return None


class AIAnalyzer:
    """Handle AI analysis with robust retry logic."""

    def __init__(self, model_name: str, max_retries: int = DEFAULT_MAX_RETRIES, retry_delay: int = DEFAULT_RETRY_DELAY):
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        # Don't create model instance here - create fresh for each analysis

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the model's tokenizer."""
        try:
            model = genai.GenerativeModel(self.model_name)
            return model.count_tokens(text).total_tokens
        except Exception as e:
            logger.warning(f"Token counting failed: {e}")
            # Fallback estimate: ~4 chars per token
            return len(text) // 4

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
            
            except (ValidationError, ValueError):
                # Don't retry validation errors - they won't improve
                raise
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning("AI call failed, retry %d/%d: %s", attempt + 1, self.max_retries, e)
                    time.sleep(self.retry_delay)
                    continue
                raise APIError(f"AI analysis failed after retries: {e}") from e

    def extract_kg(self, text: str, title: str) -> dict:
        """
        Extract Knowledge Graph from text with robust JSON handling.
        
        Steps:
        1. Ask Gemini for JSON in JSON mode.
        2. On parse failure, run a repair pass via get_kg_json_repair_prompt.
        3. If still failing after retries, raise ValidationError (fail loudly).
        """
        from .ai_prompts import get_kg_json_repair_prompt
        
        prompt = get_kg_prompt(title, text=text[:50000])
        
        # Use JSON mode to reduce malformed output
        config = genai.types.GenerationConfig(
            max_output_tokens=8192,
            temperature=0.1,
            top_p=0.8,
            response_mime_type="application/json",
        )
        
        # Permissive safety settings to prevent blocking scientific content
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                model = genai.GenerativeModel(self.model_name)
                response = model.generate_content(
                    prompt,
                    generation_config=config,
                    safety_settings=safety_settings,
                )
                
                if not response.text:
                    raise APIError("Empty response from AI model during KG extraction")

                raw_text = response.text
                logger.debug("Raw KG response (first 500 chars): %s", raw_text[:500])

                try:
                    # With JSON mode, we shouldn't need to strip markdown fences,
                    # but do it anyway for safety
                    cleaned = re.sub(r"```json\s*|\s*```", "", raw_text.strip())
                    kg_raw = json.loads(cleaned)
                except json.JSONDecodeError as parse_error:
                    # Attempt one repair pass
                    logger.warning(
                        "KG JSON parse failed on attempt %d: %s; trying repair",
                        attempt + 1, parse_error,
                    )
                    
                    repair_prompt = get_kg_json_repair_prompt(raw_text)
                    repair_config = genai.types.GenerationConfig(
                        max_output_tokens=8192,
                        temperature=0.0,  # Deterministic for repair
                        top_p=0.9,
                        response_mime_type="application/json",
                    )
                    
                    repair_response = model.generate_content(
                        repair_prompt,
                        generation_config=repair_config,
                        safety_settings=safety_settings,
                    )
                    
                    if not repair_response.text:
                        raise APIError("Empty response from repair attempt")
                    
                    repair_text = repair_response.text.strip()
                    logger.debug("Repaired KG response (first 500 chars): %s", repair_text[:500])
                    
                    # Try parsing the repaired response
                    cleaned_repair = re.sub(r"```json\s*|\s*```", "", repair_text)
                    kg_raw = json.loads(cleaned_repair)

                # Validate and return
                return validate_kg_response(kg_raw)

            except json.JSONDecodeError as e:
                last_error = e
                logger.warning("KG JSON still invalid after repair on attempt %d: %s", attempt + 1, e)
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    
            except Exception as e:
                last_error = e
                logger.warning("KG extraction attempt %d failed: %s", attempt + 1, e)
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

        # All attempts failed - raise instead of returning empty KG
        msg = f"KG extraction failed after {self.max_retries} attempts: {last_error}"
        logger.error(msg)
        raise ValidationError(msg, field="kg")




class LiteratureMapper:
    """High-level interface for corpus management."""

    def __init__(
        self,
        corpus_path: str,
        model_name: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        validate_api: bool = True,
    ):
        self.corpus_path = Path(corpus_path).resolve()
        self.corpus_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize configuration
        self.config = load_config(model_name=model_name, api_key=api_key)
        self.model_name = self.config.model_name
        
        if validate_api:
            self._setup_api(self.config.api_key)
            self.embedding_generator = EmbeddingGenerator(api_key or os.getenv("GEMINI_API_KEY"))
            self.argument_agent = ArgumentAgent(api_key or os.getenv("GEMINI_API_KEY"), model_name)
            self.validation_agent = ValidationAgent(api_key or os.getenv("GEMINI_API_KEY"), model_name)
        else:
            self.embedding_generator = None
            self.argument_agent = None
            self.validation_agent = None
            
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
                "test", 
                generation_config=genai.types.GenerationConfig(max_output_tokens=1),
                request_options={'timeout': 10}
            )
            # Clean up test objects
            del test_model, response
            logger.info("API and model '%s' validated", self.model_name)
        except Exception as e:
            # If we hit a rate limit during validation, just warn and proceed.
            # The key is likely valid, just exhausted.
            if "429" in str(e) or "ResourceExhausted" in str(e):
                logger.warning("API validation hit rate limit (429). Proceeding with caution.")
                return
            raise APIError(f"Failed to configure or validate Gemini API: {e}") from e

    def _get_existing_pdf_paths(self, session) -> set[str]:
        """Return absolute paths of already-processed PDFs."""
        existing_paths = set()
        for (path,) in session.query(Paper.pdf_path).all():
            if path is not None:
                # Normalize to absolute path for consistent comparison
                abs_path = str(Path(path).resolve())
                existing_paths.add(abs_path)
        return existing_paths

    def _save_paper_to_db(self, session, pdf_path: Optional[Path], analysis: dict) -> Paper:
        """Insert Paper plus authors/concepts."""
        # Store absolute path for consistency
        stored_path = str(pdf_path.resolve()) if pdf_path else None
        
        # Check for existing paper to avoid duplicates
        existing_paper = session.query(Paper).filter_by(
            title=analysis["title"],
            year=analysis["year"]
        ).first()

        if existing_paper:
            logger.warning(f"Duplicate paper found: '{analysis['title']}' ({analysis['year']}). Skipping insertion.")
            # Update PDF path if it was missing (e.g. manual entry)
            if stored_path and not existing_paper.pdf_path:
                 existing_paper.pdf_path = stored_path
                 session.add(existing_paper)
                 session.flush()
            return existing_paper
        
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
            arxiv_id=analysis.get("arxiv_id"),
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
        return paper

    def _save_kg_to_db(self, session, paper_id: int, kg_data: dict) -> None:
        """Save Knowledge Graph nodes and edges."""
        nodes = kg_data.get('nodes', [])
        edges = kg_data.get('edges', [])
        
        if not nodes:
            return

        # Map LLM node ID -> DB node ID
        node_id_map = {}
        
        # 1. Upsert Nodes
        for node in nodes:
            # Check if node exists (by type and label)
            existing_node = session.query(KGNode).filter_by(
                type=node['type'], 
                label=node['label']
            ).first()
            
            if existing_node:
                node_id_map[node['id']] = existing_node.id
            else:
                # Generate embedding
                vector = None
                if self.embedding_generator:
                    # Embed the label + type for context
                    text_to_embed = f"{node['label']} ({node['type']})"
                    vector = self.embedding_generator.generate_embedding(text_to_embed)

                new_node = KGNode(
                    type=node['type'],
                    label=node['label'],
                    source_paper_id=paper_id,
                    vector=vector,
                    embedding_model=self.embedding_generator.model_name if self.embedding_generator else None,
                    claim_confidence=node.get('confidence'),
                    claim_type=node.get('subtype')
                )
                session.add(new_node)
                session.flush() # Get ID
                node_id_map[node['id']] = new_node.id
                
        # 2. Insert Edges
        for edge in edges:
            source_db_id = node_id_map.get(edge['source'])
            target_db_id = node_id_map.get(edge['target'])
            
            if source_db_id and target_db_id:
                new_edge = KGEdge(
                    source_id=source_db_id,
                    target_id=target_db_id,
                    type=edge['type'],
                    source_paper_id=paper_id
                )
                session.add(new_edge)
        
        logger.info("Saved KG: %d nodes, %d edges", len(nodes), len(edges))

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
                    
                    # Extract DOI from PDF if not already in analysis
                    if not analysis.get('doi'):
                        extracted_doi = self.pdf_processor.extract_doi(pdf_path)
                        if extracted_doi:
                            analysis['doi'] = extracted_doi
                    
                    # Extract arXiv ID from PDF text (first page typically)
                    from .arxiv_api import extract_arxiv_id_from_pdf_text
                    arxiv_id = extract_arxiv_id_from_pdf_text(text[:5000])  # First 5000 chars
                    if arxiv_id:
                        analysis['arxiv_id'] = arxiv_id
                    
                    paper = self._save_paper_to_db(session, pdf_path, analysis)
                    
                    # KG Extraction
                    kg_data = self.ai_analyzer.extract_kg(text, analysis['title'])
                    self._save_kg_to_db(session, paper.id, kg_data)
                    
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
            p.arxiv_id,
            p.citation_count,
            p.citations_per_year,
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
                "core_argument", "Manually entered ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“ no automated analysis available"
            ),
            "methodology": kwargs.get("methodology", "Not specified"),
            "theoretical_framework": kwargs.get("theoretical_framework", "Not specified"),
            "contribution_to_field": kwargs.get("contribution_to_field", "Not specified"),
            "key_concepts": kwargs.get("key_concepts", []),
            "doi": kwargs.get("doi"),
            "citation_count": kwargs.get("citation_count"),
        }
        
        with get_db_session(self.corpus_path) as session:
            paper = self._save_paper_to_db(session, None, analysis)
            
            # Create KG Nodes for manual entry to ensure Agent visibility
            nodes = []
            
            # 1. Paper Node
            paper_node = KGNode(
                type="paper",
                label=analysis["title"],
                source_paper_id=paper.id
            )
            nodes.append(paper_node)
            
            # 2. Concept Nodes
            for concept in analysis.get("key_concepts", []):
                if concept.strip():
                    nodes.append(KGNode(
                        type="concept",
                        label=concept.strip(),
                        source_paper_id=paper.id
                    ))
                    
            # 3. Core Argument Node (as Finding)
            if analysis.get("core_argument"):
                 nodes.append(KGNode(
                    type="finding",
                    label=analysis["core_argument"][:200], # Truncate for label
                    source_paper_id=paper.id
                ))

            # Generate embeddings for new nodes
            for node in nodes:
                try:
                    if self.embedding_generator:
                        vector = self.embedding_generator.generate_embedding(node.label)
                        node.vector = vector
                        node.embedding_model = self.embedding_generator.model_name
                    session.add(node)
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for manual node '{node.label}': {e}")
            
            session.commit()
            logger.info(f"Added manual entry with {len(nodes)} KG nodes")

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

    def get_concept_timeline(self, concept: str = None, top_n: int = 20) -> pd.DataFrame:
        """
        Get temporal analysis of concepts in the corpus.
        
        Tracks when concepts first appeared, their peak usage year, and
        paper counts across decades. Useful for understanding how terminology
        and ideas evolved over time.
        
        Args:
            concept: Specific concept name to analyze (case-insensitive). 
                     If None, returns top_n most frequent concepts.
            top_n: Number of top concepts to return if concept is None.
            
        Returns:
            DataFrame with columns:
                - concept: Concept name
                - first_year: Year the concept first appeared
                - total_papers: Total papers mentioning this concept
                - peak_year: Year with most papers mentioning this concept
                - introduced_by: Title of paper that first used this concept
        """
        from sqlalchemy import func, text
        
        with get_db_session(self.corpus_path) as session:
            if concept:
                # Get specific concept analysis
                concepts = session.query(Concept).filter(
                    Concept.name.ilike(f"%{concept}%")
                ).all()
            else:
                # Get top N concepts by paper count
                concept_counts = (
                    session.query(Concept.id, func.count(PaperConcept.paper_id).label('count'))
                    .join(PaperConcept, Concept.id == PaperConcept.concept_id)
                    .group_by(Concept.id)
                    .order_by(func.count(PaperConcept.paper_id).desc())
                    .limit(top_n)
                    .all()
                )
                concept_ids = [c.id for c in concept_counts]
                concepts = session.query(Concept).filter(Concept.id.in_(concept_ids)).all()
            
            results = []
            for c in concepts:
                # Get paper years for this concept
                paper_years = (
                    session.query(Paper.id, Paper.year, Paper.title)
                    .join(PaperConcept, Paper.id == PaperConcept.paper_id)
                    .filter(PaperConcept.concept_id == c.id)
                    .order_by(Paper.year)
                    .all()
                )
                
                if not paper_years:
                    continue
                
                # Calculate metrics
                years = [p.year for p in paper_years if p.year]
                first_year = min(years) if years else None
                total_papers = len(paper_years)
                
                # Find peak year
                from collections import Counter
                year_counts = Counter(years)
                peak_year = year_counts.most_common(1)[0][0] if year_counts else None
                
                # Find introducing paper
                first_paper = next((p for p in paper_years if p.year == first_year), None)
                introduced_by = first_paper.title if first_paper else "Unknown"
                
                results.append({
                    "concept": c.name,
                    "first_year": first_year,
                    "total_papers": total_papers,
                    "peak_year": peak_year,
                    "introduced_by": introduced_by[:60] + "..." if len(introduced_by) > 60 else introduced_by
                })
            
            df = pd.DataFrame(results)
            if not df.empty:
                df = df.sort_values("first_year").reset_index(drop=True)
            return df

    def search_corpus(self, query: str, column: str = 'core_argument', semantic: bool = False, limit: int = 10) -> list[dict]:
        """
        Search the corpus for papers matching the query.
        
        Args:
            query: Search term
            column: Column to search (ignored if semantic=True)
            semantic: If True, use vector search on KG nodes
            limit: Max results
            
        Returns:
            List of matching papers/nodes
        """
        with get_db_session(self.corpus_path) as session:
            if semantic and self.embedding_generator:
                # Semantic Search
                query_vector = self.embedding_generator.generate_query_embedding(query)
                if query_vector is None:
                    logger.warning("Failed to generate query embedding")
                    return []
                
                # Fetch all nodes with vectors
                nodes = session.query(KGNode).filter(KGNode.vector.isnot(None)).all()
                
                results = []
                for node in nodes:
                    sim = cosine_similarity(query_vector, node.vector)
                    if sim > self.config.search_threshold:
                        # Confidence Weighting
                        # If node has confidence (0-1), multiply it into the score.
                        # Default to 1.0 if not present.
                        confidence = 1.0
                        if hasattr(node, 'claim_confidence') and node.claim_confidence is not None:
                            confidence = float(node.claim_confidence)
                        
                        # Apply boost for citations per year if paper exists
                        # (This is subtle: we don't want to over-bias to old popular papers, 
                        #  but we do want to favor influential ones)
                        # For now, we'll stick to semantic + claim confidence.
                        
                        final_score = sim * confidence
                        
                        results.append((final_score, node))
                
                # Sort by final score
                results.sort(key=lambda x: x[0], reverse=True)
                
                # Deduplicate by paper ID, keeping the best match
                seen_papers = set()
                unique_results = []
                
                for score, node in results:
                    if node.source_paper_id in seen_papers:
                        continue
                    seen_papers.add(node.source_paper_id)
                    unique_results.append((score, node))
                    
                    if len(unique_results) >= limit:
                        break
                
                # Format output
                output = []
                
                for score, node in unique_results:
                    paper = session.get(Paper, node.source_paper_id)
                    
                    # Format citation: (Author et al., Year)
                    citation = "Unknown"
                    if paper:
                        authors = paper.authors
                        year = paper.year
                        if authors:
                            first_author = authors[0].name.split()[-1] # Last name
                            if len(authors) > 1:
                                citation = f"{first_author} et al., {year}"
                            else:
                                citation = f"{first_author}, {year}"
                        else:
                            citation = f"Unknown, {year}"
                    
                    match_context = f"[{citation}: {node.label}] ({node.type})"
                    if hasattr(node, 'claim_type') and node.claim_type:
                        match_context += f" [Type: {node.claim_type}]"
                    if hasattr(node, 'claim_confidence') and node.claim_confidence is not None:
                         match_context += f" [Conf: {node.claim_confidence}]"

                    output.append({
                        "id": paper.id if paper else None,
                        "title": paper.title if paper else "Unknown",
                        "year": paper.year if paper else None,
                        "citations_per_year": paper.citations_per_year if paper else None,
                        "match_type": "semantic",
                        "match_score": round(float(score), 3),
                        "match_context": match_context
                    })
                return output
                
            else:
                # Keyword Search (Legacy)
                from .validation import validate_search_params
                column, query = validate_search_params(column, query)
                
                # Dynamic column selection
                target_col = getattr(Paper, column)
                
                papers = session.query(Paper).filter(
                    target_col.ilike(f"%{query}%")
                ).limit(limit).all()
                
                return [{
                    "id": p.id,
                    "title": p.title,
                    "year": p.year,
                    "citations_per_year": p.citations_per_year,
                    "match_type": "keyword",
                    "match_score": 1.0,
                    "match_context": f"Found in {column}"
                } for p in papers]

    def get_paper_by_id(self, paper_id: int) -> dict | None:
        """Get full paper details by ID for validation agents."""
        with get_db_session(self.corpus_path) as session:
            p = session.query(Paper).filter_by(id=paper_id).first()
            if not p:
                return None
                
            return {
                "id": p.id,
                "title": p.title,
                "year": p.year,
                "journal": p.journal,
                "abstract_short": p.abstract_short,
                "core_argument": p.core_argument,
                "methodology": p.methodology,
                "theoretical_framework": p.theoretical_framework,
                "contribution_to_field": p.contribution_to_field,
                "authors": [a.name for a in p.authors],
                "key_concepts": [c.name for c in p.concepts],
                "doi": p.doi,
                "citation_count": p.citation_count
            }

    def synthesize_answer(
        self, 
        query: str, 
        limit: int = 10,
        year_range: tuple[int, int] = None,
    ) -> str:
        """
        Synthesize an answer to a research question using the Argument Agent.
        
        Args:
            query: Research question
            limit: Max number of context nodes to retrieve
            year_range: Optional (start_year, end_year) tuple to filter papers.
                        Only papers published in [start_year, end_year] will be used.
            
        Returns:
            Synthesized answer
        """
        if not self.argument_agent:
            return "Argument Agent not initialized (missing API key)."
        
        # Build contextual query with temporal scope
        effective_query = query
        if year_range:
            effective_query = f"{query} (focusing on research from {year_range[0]} to {year_range[1]})"
            
        # 1. Retrieve context using semantic search
        context_nodes = self.search_corpus(query, semantic=True, limit=limit * 2)  # Get extra for filtering
        
        # 2. Filter by year range if specified
        if year_range:
            start_year, end_year = year_range
            context_nodes = [
                node for node in context_nodes 
                if node.get('year') and start_year <= node['year'] <= end_year
            ][:limit]  # Re-apply limit after filtering
        else:
            context_nodes = context_nodes[:limit]
        
        # 3. Synthesize with temporal context
        return self.argument_agent.synthesize(effective_query, context_nodes)

    def validate_hypothesis(
        self, 
        hypothesis: str,
        year_range: tuple[int, int] = None,
    ) -> dict:
        """
        Validate a hypothesis using the Validation Agent.
        
        Args:
            hypothesis: User hypothesis
            year_range: Optional (start_year, end_year) tuple to filter papers.
                        Only papers published in [start_year, end_year] will be
                        used for validation.
            
        Returns:
            Validation result dict
        """
        if not self.validation_agent:
            return {"verdict": "ERROR", "explanation": "Validation Agent not initialized."}
        
        # Build contextual hypothesis with temporal scope
        effective_hypothesis = hypothesis
        if year_range:
            effective_hypothesis = f"{hypothesis} (as evaluated by research from {year_range[0]} to {year_range[1]})"
            
        # 1. Retrieve context (findings/limitations/hypotheses)
        context_nodes = self.search_corpus(hypothesis, semantic=True, limit=30)  # Get extra for filtering
        
        # 2. Filter by year range if specified
        if year_range:
            start_year, end_year = year_range
            context_nodes = [
                node for node in context_nodes 
                if node.get('year') and start_year <= node['year'] <= end_year
            ][:15]  # Re-apply limit after filtering
        else:
            context_nodes = context_nodes[:15]
        
        # 3. Validate with temporal context
        return self.validation_agent.validate_hypothesis(effective_hypothesis, context_nodes)
    
    def update_citations(self, email: Optional[str] = None, verbose: bool = True) -> None:
        """Update citation counts and references from OpenAlex."""
        from .openalex import fetch_citations_for_corpus

        stats = fetch_citations_for_corpus(str(self.corpus_path), email=email)

        if verbose:
            print(f"Updated: {stats['found']} found, {stats['not_found']} missing, "
                  f"{stats['citations']} references inserted.")

    def build_genealogy(self, verbose: bool = True) -> dict:
        """
        Extract intellectual relationships between papers in the corpus (on-demand).
        
        This uses the LLM to identify EXTENDS, CHALLENGES, and SYNTHESIZES
        relationships between papers. Run this once after processing papers
        to enable get_argument_evolution() queries.
        
        Args:
            verbose: Print progress updates
            
        Returns:
            Dict with counts: {'analyzed': N, 'relationships': N}
        """
        import json
        import re
        from .database import get_db_session, Paper, IntellectualEdge
        from .ai_prompts import get_genealogy_prompt
        
        stats = {'analyzed': 0, 'relationships': 0, 'errors': 0}
        
        with get_db_session(self.corpus_path) as session:
            papers = session.query(Paper).order_by(Paper.year.desc()).all()
            
            if len(papers) < 2:
                if verbose:
                    print("Need at least 2 papers to build genealogy.")
                return stats
            
            # Prepare corpus context (limited to save tokens)
            def format_corpus_list(papers, exclude_id=None):
                lines = []
                for p in papers:
                    if p.id == exclude_id:
                        continue
                    authors = ", ".join([a.name for a in p.authors[:2]]) if p.authors else "Unknown"
                    lines.append(f"ID:{p.id} | {authors} ({p.year}): {p.title}")
                return "\n".join(lines[:30])  # Limit context size
            
            if verbose:
                print(f"Analyzing {len(papers)} papers for intellectual relationships...")
            
            for paper in papers:
                try:
                    # Build prompt
                    corpus_context = format_corpus_list(papers, exclude_id=paper.id)
                    abstract = paper.core_argument or paper.abstract_short or "No abstract available"
                    
                    prompt = get_genealogy_prompt(paper.title, abstract[:500], corpus_context)
                    
                    # Call LLM
                    model = genai.GenerativeModel(self.model_name)
                    response = model.generate_content(prompt)
                    
                    if not response.text:
                        continue
                    
                    # Parse response
                    cleaned = re.sub(r"```json\s*|\s*```", "", response.text.strip())
                    data = json.loads(cleaned)
                    
                    relationships = data.get('relationships', [])
                    
                    for rel in relationships:
                        target_id = rel.get('target_id')
                        rel_type = rel.get('type', 'BUILDS_ON')
                        confidence = rel.get('confidence', 0.5)
                        evidence = rel.get('evidence', '')
                        
                        # Validate target exists
                        if not target_id or not session.get(Paper, target_id):
                            continue
                        
                        # Check for existing edge
                        existing = session.query(IntellectualEdge).filter_by(
                            source_paper_id=paper.id,
                            target_paper_id=target_id,
                            relation_type=rel_type
                        ).first()
                        
                        if not existing:
                            edge = IntellectualEdge(
                                source_paper_id=paper.id,
                                target_paper_id=target_id,
                                relation_type=rel_type,
                                confidence=confidence,
                                evidence=evidence[:500] if evidence else None
                            )
                            session.add(edge)
                            stats['relationships'] += 1
                    
                    stats['analyzed'] += 1
                    session.commit()
                    
                    if verbose and stats['analyzed'] % 5 == 0:
                        print(f"  Analyzed {stats['analyzed']}/{len(papers)} papers...")
                        
                except Exception as e:
                    logger.warning(f"Genealogy extraction failed for '{paper.title[:40]}': {e}")
                    stats['errors'] += 1
                    continue
        
        if verbose:
            print(f"Genealogy complete: {stats['relationships']} relationships found "
                  f"across {stats['analyzed']} papers.")
        return stats

    def get_argument_evolution(self, concept: str = None, paper_id: int = None) -> pd.DataFrame:
        """
        Trace intellectual lineage for a concept or paper.
        
        Shows how ideas extended, challenged, or synthesized each other over time.
        Run build_genealogy() first to populate the relationships.
        
        Args:
            concept: Search for papers mentioning this concept and trace their lineage
            paper_id: Trace lineage for a specific paper
            
        Returns:
            DataFrame with columns: source_title, source_year, relation_type, 
                                   target_title, target_year, confidence, evidence
        """
        from .database import get_db_session, Paper, IntellectualEdge
        
        with get_db_session(self.corpus_path) as session:
            # Build base query
            query = (
                session.query(
                    IntellectualEdge,
                    Paper
                )
                .join(Paper, IntellectualEdge.source_paper_id == Paper.id)
            )
            
            if paper_id:
                # Get all relationships for this paper (both directions)
                outgoing = (
                    session.query(IntellectualEdge)
                    .filter(IntellectualEdge.source_paper_id == paper_id)
                    .all()
                )
                incoming = (
                    session.query(IntellectualEdge)
                    .filter(IntellectualEdge.target_paper_id == paper_id)
                    .all()
                )
                edges = outgoing + incoming
                
            elif concept:
                # Find papers with this concept and get their relationships
                from .database import Concept, PaperConcept
                paper_ids = (
                    session.query(PaperConcept.paper_id)
                    .join(Concept, PaperConcept.concept_id == Concept.id)
                    .filter(Concept.name.ilike(f"%{concept}%"))
                    .all()
                )
                paper_ids = [p[0] for p in paper_ids]
                
                edges = (
                    session.query(IntellectualEdge)
                    .filter(
                        (IntellectualEdge.source_paper_id.in_(paper_ids)) |
                        (IntellectualEdge.target_paper_id.in_(paper_ids))
                    )
                    .all()
                )
            else:
                # Get all relationships
                edges = session.query(IntellectualEdge).all()
            
            # Format results
            results = []
            for edge in edges:
                source = session.get(Paper, edge.source_paper_id)
                target = session.get(Paper, edge.target_paper_id)
                
                if source and target:
                    results.append({
                        "source_title": source.title[:50] + "..." if len(source.title) > 50 else source.title,
                        "source_year": source.year,
                        "relation": edge.relation_type,
                        "target_title": target.title[:50] + "..." if len(target.title) > 50 else target.title,
                        "target_year": target.year,
                        "confidence": edge.confidence,
                        "evidence": edge.evidence[:100] + "..." if edge.evidence and len(edge.evidence) > 100 else edge.evidence
                    })
            
            df = pd.DataFrame(results)
            if not df.empty:
                df = df.sort_values(["source_year", "target_year"]).reset_index(drop=True)
            return df

    def normalize_concepts(self, mappings: dict[str, str], verbose: bool = True) -> int:
        """
        Merge synonym concepts into canonical forms.
        
        Args:
            mappings: {"alias": "canonical_name"} e.g., {"SNA": "Social Network Analysis"}
            verbose: Print progress
            
        Returns:
            Number of concepts normalized
        """
        from .database import get_db_session, Concept, ConceptAlias
        
        normalized = 0
        
        with get_db_session(self.corpus_path) as session:
            for alias_name, canonical_name in mappings.items():
                # Find or create canonical concept
                canonical = session.query(Concept).filter(
                    Concept.name.ilike(canonical_name)
                ).first()
                
                if not canonical:
                    # Check if it's an alias
                    canonical = session.query(Concept).filter(
                        Concept.canonical_name.ilike(canonical_name)
                    ).first()
                
                if not canonical:
                    if verbose:
                        print(f"  Canonical concept '{canonical_name}' not found, skipping.")
                    continue
                
                # Set canonical name on the canonical concept
                canonical.canonical_name = canonical_name
                
                # Find alias concept
                alias_concept = session.query(Concept).filter(
                    Concept.name.ilike(alias_name)
                ).first()
                
                if alias_concept and alias_concept.id != canonical.id:
                    # Set its canonical name to point to the main one
                    alias_concept.canonical_name = canonical_name
                    normalized += 1
                    
                    if verbose:
                        print(f"  '{alias_name}' Ã¢â€ â€™ '{canonical_name}'")
                
                # Also add to ConceptAlias table for explicit lookup
                existing_alias = session.query(ConceptAlias).filter(
                    ConceptAlias.alias.ilike(alias_name)
                ).first()
                
                if not existing_alias:
                    new_alias = ConceptAlias(
                        alias=alias_name,
                        canonical_id=canonical.id
                    )
                    session.add(new_alias)
            
            session.commit()
        
        if verbose:
            print(f"Normalized {normalized} concepts.")
        return normalized

    def get_canonical_concept(self, name: str) -> str:
        """
        Get the canonical form of a concept name.
        
        Args:
            name: Concept name (may be an alias)
            
        Returns:
            Canonical name, or original name if no mapping exists
        """
        from .database import get_db_session, Concept, ConceptAlias
        
        with get_db_session(self.corpus_path) as session:
            # Check alias table first
            alias = session.query(ConceptAlias).filter(
                ConceptAlias.alias.ilike(name)
            ).first()
            
            if alias:
                canonical = session.get(Concept, alias.canonical_id)
                return canonical.canonical_name or canonical.name
            
            # Check if concept exists and has canonical_name
            concept = session.query(Concept).filter(
                Concept.name.ilike(name)
            ).first()
            
            if concept and concept.canonical_name:
                return concept.canonical_name
            
            return name

    def find_contradictions(self, concept: str = None) -> pd.DataFrame:
        """
        Find intellectual contradictions and debates in the corpus.
        
        Identifies 'CHALLENGES' relationships between papers. If a concept is provided,
        filters for debates involving that concept.
        
        Args:
            concept: Optional concept to filter by (e.g., "social capital")
            
        Returns:
            DataFrame with columns: paper_a, paper_b, relation, confidence, evidence
        """
        from .database import get_db_session, Paper, IntellectualEdge, Concept, PaperConcept
        
        with get_db_session(self.corpus_path) as session:
            # Base query for CHALLENGES edges
            query = session.query(IntellectualEdge).filter(
                IntellectualEdge.relation_type.ilike("%CHALLENGE%")
            )
            
            # Filter by concept if provided
            if concept:
                # Get relevant paper IDs
                paper_ids = (
                    session.query(PaperConcept.paper_id)
                    .join(Concept, PaperConcept.concept_id == Concept.id)
                    .filter(
                        (Concept.name.ilike(f"%{concept}%")) |
                        (Concept.canonical_name.ilike(f"%{concept}%"))
                    )
                    .all()
                )
                paper_ids = [p[0] for p in paper_ids]
                
                # Filter edges where either source or target is relevant
                query = query.filter(
                    (IntellectualEdge.source_paper_id.in_(paper_ids)) |
                    (IntellectualEdge.target_paper_id.in_(paper_ids))
                )
            
            edges = query.all()
            
            results = []
            for edge in edges:
                source = session.get(Paper, edge.source_paper_id)
                target = session.get(Paper, edge.target_paper_id)
                
                if source and target:
                    results.append({
                        "paper_a": f"{source.title[:40]}... ({source.year})",
                        "relation": "CHALLENGES",
                        "paper_b": f"{target.title[:40]}... ({target.year})",
                        "confidence": edge.confidence,
                        "evidence": edge.evidence[:150] + "..." if edge.evidence and len(edge.evidence) > 150 else edge.evidence
                    })
            
            df = pd.DataFrame(results)
            if not df.empty:
                df = df.sort_values("confidence", ascending=False).reset_index(drop=True)
            return df

    def normalize_authors(self, mappings: dict[str, str], verbose: bool = True) -> int:
        """
        Merge synonym authors into canonical forms.
        
        Args:
            mappings: {"alias": "canonical_name"} e.g., {"Granovetter, M.": "Granovetter, Mark"}
            verbose: Print progress
            
        Returns:
            Number of authors normalized
        """
        from .database import get_db_session, Author, AuthorAlias
        
        normalized = 0
        
        with get_db_session(self.corpus_path) as session:
            for alias_name, canonical_name in mappings.items():
                # Find or create canonical author
                canonical = session.query(Author).filter(
                    Author.name.ilike(canonical_name)
                ).first()
                
                if not canonical:
                    if verbose:
                        print(f"  Canonical author '{canonical_name}' not found, skipping.")
                    continue
                
                # Set canonical name on the canonical author
                canonical.canonical_name = canonical_name
                
                # Find alias author
                alias_author = session.query(Author).filter(
                    Author.name.ilike(alias_name)
                ).first()
                
                if alias_author and alias_author.id != canonical.id:
                    # Set its canonical name to point to the main one
                    alias_author.canonical_name = canonical_name
                    normalized += 1
                    
                    if verbose:
                        print(f"  '{alias_name}' Ã¢â€ â€™ '{canonical_name}'")
                
                # Also add to AuthorAlias table for explicit lookup
                existing_alias = session.query(AuthorAlias).filter(
                    AuthorAlias.alias.ilike(alias_name)
                ).first()
                
                if not existing_alias:
                    new_alias = AuthorAlias(
                        alias=alias_name,
                        canonical_id=canonical.id
                    )
                    session.add(new_alias)
            
            session.commit()
        
        if verbose:
            print(f"Normalized {normalized} authors.")
        return normalized

    def get_canonical_author(self, name: str) -> str:
        """
        Get the canonical form of an author name.
        
        Args:
            name: Author name (may be an alias)
            
        Returns:
            Canonical name, or original name if no mapping exists
        """
        from .database import get_db_session, Author, AuthorAlias
        
        with get_db_session(self.corpus_path) as session:
            # Check alias table first
            alias = session.query(AuthorAlias).filter(
                AuthorAlias.alias.ilike(name)
            ).first()
            
            if alias:
                canonical = session.get(Author, alias.canonical_id)
                return canonical.canonical_name or canonical.name
            
            # Check if author exists and has canonical_name
            author = session.query(Author).filter(
                Author.name.ilike(name)
            ).first()
            
            if author and author.canonical_name:
                return author.canonical_name
            
            return name