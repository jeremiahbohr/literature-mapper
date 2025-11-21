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
import sqlalchemy as sa
from tqdm import tqdm

from .ai_prompts import get_analysis_prompt, get_kg_prompt
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
        """Extract Knowledge Graph from text."""
        prompt = get_kg_prompt(title, text=text[:50000])
        
        config = genai.types.GenerationConfig(
            max_output_tokens=8192,
            temperature=0.1,
            top_p=0.8,
        )
        
        # Permissive safety settings to prevent blocking scientific content
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        for attempt in range(self.max_retries):
            try:
                model = genai.GenerativeModel(self.model_name)
                response = model.generate_content(prompt, generation_config=config, safety_settings=safety_settings)
                
                # Handle potential blocking or empty responses safely
                if not response.candidates:
                    raise APIError("No candidates returned from AI model")
                    
                candidate = response.candidates[0]
                if candidate.finish_reason != 1: # 1 = STOP
                    logger.warning("KG extraction finished with reason: %s", candidate.finish_reason)
                
                # Safe text access
                if not candidate.content.parts:
                     # If MAX_TOKENS (2), we might still have partial text in some versions, 
                     # but usually 'parts' should exist. If not, it's a failure.
                     raise APIError(f"No content parts in response (Finish Reason: {candidate.finish_reason})")
                     
                text_content = candidate.content.parts[0].text

                cleaned = re.sub(r"```json\s*|\s*```", "", text_content.strip())
                try:
                    data = json.loads(cleaned)
                except json.JSONDecodeError:
                    # Attempt to repair truncated JSON
                    logger.warning("JSON truncated, attempting repair...")
                    try:
                        # Try closing lists/objects
                        if cleaned.strip().endswith(","):
                            cleaned = cleaned.strip()[:-1]
                        
                        # Naive repair: close open brackets
                        open_braces = cleaned.count("{") - cleaned.count("}")
                        open_brackets = cleaned.count("[") - cleaned.count("]")
                        
                        repaired = cleaned + ("}" * open_braces) + ("]" * open_brackets) + "}" # Extra closing brace for safety
                        
                        # If that fails, try just closing the main object
                        if not repaired.endswith("}"):
                             repaired += "]}"
                        
                        data = json.loads(repaired)
                        logger.info("JSON repair successful")
                    except json.JSONDecodeError:
                         # If repair fails, try to extract just the valid nodes list?
                         # For now, just fail gracefully but log it.
                         logger.error("JSON repair failed")
                         raise 
                
                result = validate_kg_response(data)
                
                del model, response, cleaned, data
                return result

            except (ValidationError, ValueError) as e:
                logger.warning("KG validation failed: %s", e)
                return {"nodes": [], "edges": []}

            except json.JSONDecodeError as e:
                if attempt < self.max_retries - 1:
                    logger.warning("KG JSON decode error, retry %d/%d", attempt + 1, self.max_retries)
                    time.sleep(self.retry_delay)
                    continue
                # Return empty graph on failure rather than crashing
                logger.error("Failed to parse KG response: %s", e)
                return {"nodes": [], "edges": []}
            
            except Exception as e:
                if "429" in str(e) or "ResourceExhausted" in str(e):
                    if attempt < self.max_retries - 1:
                        logger.warning("KG extraction hit rate limit (429), retry %d/%d", attempt + 1, self.max_retries)
                        time.sleep(self.retry_delay * (2 ** attempt)) # Exponential backoff
                        continue

                if attempt < self.max_retries - 1:
                    logger.warning("KG extraction failed, retry %d/%d: %s", attempt + 1, self.max_retries, e)
                    time.sleep(self.retry_delay)
                    continue
                logger.error("KG extraction failed: %s", e)
                return {"nodes": [], "edges": []}


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
        
        # Initialize configuration
        self.config = load_config(model_name=model_name, api_key=api_key)
        self.model_name = self.config.model_name
        
        self._setup_api(self.config.api_key)
        self.pdf_processor = PDFProcessor()
        self.ai_analyzer = AIAnalyzer(model_name)
        self.embedding_generator = EmbeddingGenerator(api_key or os.getenv("GEMINI_API_KEY"))
        
        # Initialize Agents
        self.argument_agent = ArgumentAgent(api_key or os.getenv("GEMINI_API_KEY"), model_name)
        self.validation_agent = ValidationAgent(api_key or os.getenv("GEMINI_API_KEY"), model_name)

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
                    embedding_model=self.embedding_generator.model_name if self.embedding_generator else None
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
                        results.append((sim, node))
                
                # Sort by similarity
                results.sort(key=lambda x: x[0], reverse=True)
                
                # Format output
                output = []
                
                for score, node in results[:limit]:
                    paper = session.query(Paper).get(node.source_paper_id)
                    
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
                    
                    output.append({
                        "id": paper.id if paper else None,
                        "title": paper.title if paper else "Unknown",
                        "year": paper.year if paper else None,
                        "match_type": "semantic",
                        "match_score": round(float(score), 3),
                        "match_context": f"[{citation}: {node.label}] ({node.type})"
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

    def synthesize_answer(self, query: str, limit: int = 10) -> str:
        """
        Synthesize an answer to a research question using the Argument Agent.
        
        Args:
            query: Research question
            limit: Max number of context nodes to retrieve
            
        Returns:
            Synthesized answer
        """
        if not self.argument_agent:
            return "Argument Agent not initialized (missing API key)."
            
        # 1. Retrieve context using semantic search
        context_nodes = self.search_corpus(query, semantic=True, limit=limit)
        
        # 2. Synthesize
        return self.argument_agent.synthesize(query, context_nodes)

    def validate_hypothesis(self, hypothesis: str) -> dict:
        """
        Validate a hypothesis using the Validation Agent.
        
        Args:
            hypothesis: User hypothesis
            
        Returns:
            Validation result dict
        """
        if not self.validation_agent:
            return {"verdict": "ERROR", "explanation": "Validation Agent not initialized."}
            
        # 1. Retrieve context (findings/limitations/hypotheses)
        # We search for the hypothesis itself to find semantically similar claims
        context_nodes = self.search_corpus(hypothesis, semantic=True, limit=15)
        
        # 2. Validate
        return self.validation_agent.validate_hypothesis(hypothesis, context_nodes)