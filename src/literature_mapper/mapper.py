"""
mapper.py – core logic for Literature Mapper
Fully patched version (2025-08-03).

Fixes applied
-------------
1.   _get_existing_papers   → uses `.scalars()` so rows aren’t tuples.  
2.  PDFProcessor.extract_text → guards against None from `page.extract_text()`.  
3.  AIAnalyzer._prepare_analysis → detects model type and passes it to `get_analysis_prompt()`.  
4.  search_papers            → drops relationship fields from whitelist and keeps
   column order deterministic.
"""

from __future__ import annotations

import json
import logging
import os
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
from .database import (
    Author,
    Concept,
    Paper,
    PaperAuthor,
    PaperConcept,
    get_db_session,
)
from .exceptions import APIError, PDFProcessingError, ValidationError
from .validation import validate_api_key, validate_json_response, validate_pdf_file

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#                              Helper dataclasses                             #
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
#                              PDF text extractor                             #
# --------------------------------------------------------------------------- #


class PDFProcessor:
    """Handles PDF text extraction with comprehensive error handling."""

    def __init__(self, max_file_size: int = 50 * 1024 * 1024):
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

                text_parts: list[str] = []
                for page in reader.pages:
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():  # ← guard against None
                            text_parts.append(page_text)
                    except Exception:  # noqa: BLE001
                        continue  # skip bad page but don’t abort whole file

                full_text = "\n".join(text_parts).strip()
                if len(full_text) < 100:
                    raise PDFProcessingError(
                        "Insufficient text extracted",
                        pdf_path=pdf_path,
                        error_type="extraction",
                    )

                # normalise whitespace
                return re.sub(r"\s+", " ", full_text)

        except pypdf.errors.PdfReadError as e:  # type: ignore[attr-defined]
            raise PDFProcessingError(
                "PDF read error",
                pdf_path=pdf_path,
                error_type="corruption",
                original_exception=e,
            ) from e
        except Exception as e:  # noqa: BLE001
            raise PDFProcessingError(
                "Unexpected error",
                pdf_path=pdf_path,
                error_type="unknown",
                original_exception=e,
            ) from e


# --------------------------------------------------------------------------- #
#                               AI analysis client                            #
# --------------------------------------------------------------------------- #


class AIAnalyzer:
    """Call Gemini with model-aware prompt + robust retry logic."""

    def __init__(self, model_name: str, max_retries: int = 3, retry_delay: int = 2):
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    # --------------------------------------------------------------------- #
    #                             public API                                 #
    # --------------------------------------------------------------------- #

    def analyze(self, text: str) -> dict:
        prompt, cfg = self._prepare_analysis(text)

        for attempt in range(self.max_retries):
            try:
                resp = self.model.generate_content(prompt, generation_config=cfg)
                if not resp.text:
                    raise APIError("Empty response from AI model")

                cleaned = re.sub(r"```json\s*|\s*```", "", resp.text.strip())
                return validate_json_response(json.loads(cleaned))

            except json.JSONDecodeError as e:
                if attempt < self.max_retries - 1:
                    logger.warning("JSON decode error – retry %d/ %d", attempt + 1, self.max_retries)
                    time.sleep(self.retry_delay)
                    continue
                raise APIError("Failed to parse AI response as JSON after retries") from e
            except Exception as e:  # noqa: BLE001
                if attempt < self.max_retries - 1:
                    logger.warning("AI call failed – retry %d/ %d: %s", attempt + 1, self.max_retries, e)
                    time.sleep(self.retry_delay)
                    continue
                raise APIError("AI analysis failed after retries", original_exception=e) from e

    # --------------------------------------------------------------------- #
    #                          internal helpers                              #
    # --------------------------------------------------------------------- #

    def _prepare_analysis(self, text: str) -> tuple[str, genai.types.GenerationConfig]:
        """Tailor prompt & generation config to the specific Gemini tier."""
        name = self.model_name.lower()
        if "flash" in name:
            model_type, max_text, max_tokens, temperature = ("flash", 30_000, 2048, 0.2)
        elif "pro" in name:
            model_type, max_text, max_tokens, temperature = ("pro", 80_000, 4096, 0.1)
        elif "ultra" in name:
            model_type, max_text, max_tokens, temperature = ("ultra", 100_000, 8192, 0.1)
        else:
            model_type, max_text, max_tokens, temperature = ("unknown", 50_000, 2048, 0.1)

        truncated = text[:max_text]
        prompt = get_analysis_prompt(model_type).format(text=truncated)

        cfg = genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            top_p=0.8,
        )
        return prompt, cfg


# --------------------------------------------------------------------------- #
#                               main entry point                              #
# --------------------------------------------------------------------------- #


class LiteratureMapper:
    """
    High-level façade for corpus management.
    """

    # --------------------------------------------------------------------- #
    #                              lifecycle                                 #
    # --------------------------------------------------------------------- #

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

        self.db_session = get_db_session(self.corpus_path)
        self.pdf_processor = PDFProcessor()
        self.ai_analyzer = AIAnalyzer(model_name)

    def __del__(self) -> None:
        if hasattr(self, "db_session"):
            self.db_session.close()

    # --------------------------------------------------------------------- #
    #                           internal utilities                           #
    # --------------------------------------------------------------------- #

    def _setup_api(self, api_key: Optional[str]) -> None:
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValidationError("Gemini API key missing – set GEMINI_API_KEY")

        if not validate_api_key(key):
            raise ValidationError("Invalid API key format")

        try:
            genai.configure(api_key=key)
            genai.GenerativeModel(self.model_name).generate_content(
                "ping", generation_config=genai.types.GenerationConfig(max_output_tokens=1)
            )
            logger.info("API and model '%s' validated", self.model_name)
        except Exception as e:  # noqa: BLE001
            raise APIError("Failed to configure or validate Gemini API", original_exception=e) from e

    def _get_existing_papers(self) -> set[str]:
        """Return absolute paths of already-ingested PDFs."""
        try:
            return set(self.db_session.query(Paper.pdf_path).scalars().all())  # fix
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to query existing papers: %s", e)
            return set()

    # --------------------------------------------------------------------- #
    #                           database helpers                             #
    # --------------------------------------------------------------------- #

    def _save_paper_to_db(self, pdf_path: Optional[Path], analysis: dict) -> None:
        """Insert Paper plus authors/concepts in a single transaction."""
        try:
            paper = Paper(
                pdf_path=str(pdf_path) if pdf_path else None,
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
            self.db_session.add(paper)
            self.db_session.flush()

            # authors
            for author_name in analysis.get("authors", []):
                if not author_name.strip():
                    continue
                author = (
                    self.db_session.query(Author)
                    .filter_by(name=author_name.strip())
                    .first()
                )
                if not author:
                    author = Author(name=author_name.strip())
                    self.db_session.add(author)
                    self.db_session.flush()
                self.db_session.add(PaperAuthor(paper_id=paper.id, author_id=author.id))

            # concepts
            for concept_name in analysis.get("key_concepts", []):
                if not concept_name.strip():
                    continue
                concept = (
                    self.db_session.query(Concept)
                    .filter_by(name=concept_name.strip())
                    .first()
                )
                if not concept:
                    concept = Concept(name=concept_name.strip())
                    self.db_session.add(concept)
                    self.db_session.flush()
                self.db_session.add(
                    PaperConcept(paper_id=paper.id, concept_id=concept.id)
                )

            self.db_session.commit()
            logger.info("Saved paper: %s", analysis["title"])
        except Exception as e:  # noqa: BLE001
            self.db_session.rollback()
            logger.error("Failed to save paper %s: %s", pdf_path, e)
            raise

    # --------------------------------------------------------------------- #
    #                             public API                                 #
    # --------------------------------------------------------------------- #

    def process_new_papers(self) -> ProcessingResult:
        """Scan directory, process unseen PDFs, and persist analyses."""
        existing = self._get_existing_papers()
        all_pdfs = list(self.corpus_path.glob("*.pdf"))
        new_pdfs = [p for p in all_pdfs if str(p) not in existing]

        if not new_pdfs:
            logger.info("No new papers to process")
            return ProcessingResult()

        logger.info("Processing %d new PDFs", len(new_pdfs))
        result = ProcessingResult()

        for pdf_path in tqdm(new_pdfs, desc="Processing papers", unit="pdf"):
            try:
                text = self.pdf_processor.extract_text(pdf_path)
                analysis = self.ai_analyzer.analyze(text)
                self._save_paper_to_db(pdf_path, analysis)
                result.processed += 1
            except PDFProcessingError as e:
                logger.warning("Skipped %s: %s", pdf_path.name, e.user_message)
                result.skipped += 1
            except (APIError, ValidationError) as e:
                logger.error("Failed %s: %s", pdf_path.name, e)
                result.failed += 1
            except Exception as e:  # noqa: BLE001
                logger.exception("Unexpected error on %s: %s", pdf_path.name, e)
                result.failed += 1

        logger.info(
            "Processing done – processed=%d failed=%d skipped=%d",
            result.processed,
            result.failed,
            result.skipped,
        )
        return result

    # ------------------------------------------------------------------ #
    #                      query & maintenance helpers                    #
    # ------------------------------------------------------------------ #

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
        return pd.read_sql(query, self.db_session.bind)

    def export_to_csv(self, output_path: str) -> None:
        """Dump current corpus to CSV (utf-8, no index)."""
        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        self.get_all_analyses().to_csv(out, index=False)
        logger.info("Corpus exported to %s", out)

    # ------------------------------------------------------------------ #
    #                       manual entry / editing                        #
    # ------------------------------------------------------------------ #

    def add_manual_entry(self, title: str, authors: list[str], year: int, **kw) -> None:
        """Insert a paper without a PDF file."""
        if not title.strip():
            raise ValidationError("Title cannot be empty")
        if not 1900 <= year <= 2030:
            raise ValidationError("Year must be between 1900 and 2030")
        if not authors or not any(a.strip() for a in authors):
            raise ValidationError("At least one author required")

        if self.db_session.query(Paper).filter_by(title=title.strip(), year=year).first():
            raise ValidationError(f"Paper '{title}' ({year}) already exists")

        analysis = {
            "title": title.strip(),
            "authors": [a.strip() for a in authors if a.strip()],
            "year": year,
            "journal": kw.get("journal"),
            "abstract_short": kw.get("abstract_short"),
            "core_argument": kw.get(
                "core_argument", "Manually entered – no automated analysis available"
            ),
            "methodology": kw.get("methodology", "Not specified"),
            "theoretical_framework": kw.get("theoretical_framework", "Not specified"),
            "contribution_to_field": kw.get("contribution_to_field", "Not specified"),
            "key_concepts": kw.get("key_concepts", []),
            "doi": kw.get("doi"),
            "citation_count": kw.get("citation_count"),
        }
        self._save_paper_to_db(None, analysis)

    def update_papers(self, paper_ids: list[int], updates: dict) -> None:
        """Bulk update allowed columns for given paper IDs."""
        if not paper_ids or not updates:
            raise ValidationError("No paper IDs or updates provided")

        allowed = {
            "title",
            "year",
            "journal",
            "abstract_short",
            "core_argument",
            "methodology",
            "theoretical_framework",
            "contribution_to_field",
            "doi",
            "citation_count",
        }
        if bad := (set(updates) - allowed):
            raise ValidationError(f"Invalid fields: {', '.join(bad)}")

        count = (
            self.db_session.query(Paper).filter(Paper.id.in_(paper_ids)).count()
        )
        if count != len(paper_ids):
            raise ValidationError("Some paper IDs do not exist")

        self.db_session.query(Paper).filter(Paper.id.in_(paper_ids)).update(
            updates, synchronize_session=False
        )
        self.db_session.commit()
        logger.info("Updated %d papers", len(paper_ids))

    # ------------------------------------------------------------------ #
    #                          search utility                             #
    # ------------------------------------------------------------------ #

    def search_papers(self, column: str, query: str) -> pd.DataFrame:
        """
        Case-insensitive LIKE search over a whitelisted column of `papers`.
        Relationship fields (authors, key_concepts) require joins and are excluded
        to avoid SQLAlchemy errors.
        """
        searchable = {
            "title",
            "core_argument",
            "methodology",
            "theoretical_framework",
            "contribution_to_field",
            "journal",
            "abstract_short",
        }
        if column not in searchable:
            raise ValidationError(
                f"Column '{column}' is not searchable. Valid: {', '.join(searchable)}"
            )
        if not query.strip():
            raise ValidationError("Search query cannot be empty")

        ilike_filter = getattr(Paper, column).ilike(f"%{query.strip()}%")
        matching = self.db_session.query(Paper).filter(ilike_filter).all()
        if not matching:
            return pd.DataFrame()

        ids = [p.id for p in matching]
        # deterministic column order – title always second
        select_cols = [
            "p.id",
            "p.title",
            "p.year",
            "GROUP_CONCAT(DISTINCT a.name) AS authors",
            "GROUP_CONCAT(DISTINCT c.name) AS key_concepts",
        ]
        if column != "title":
            # insert searched column just after title
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
        return pd.read_sql(sql, self.db_session.bind)

    # ------------------------------------------------------------------ #
    #                           simple stats                              #
    # ------------------------------------------------------------------ #

    def get_statistics(self) -> CorpusStatistics:
        return CorpusStatistics(
            total_papers=self.db_session.query(Paper).count(),
            total_authors=self.db_session.query(Author).count(),
            total_concepts=self.db_session.query(Concept).count(),
        )
