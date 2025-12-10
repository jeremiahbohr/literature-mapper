"""
analysis.py - Corpus Analysis Tools
"""

import logging
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass

import pandas as pd
from rich.console import Console

from .database import get_db_session, Paper, Author, Concept

console = Console()
logger = logging.getLogger(__name__)


@dataclass
class CorpusStatistics:
    """Statistics about the corpus."""
    total_papers: int
    total_authors: int
    total_concepts: int
    papers_with_citations: int
    total_citations: int
    year_range: tuple[int, int] | None
    top_journals: list[tuple[str, int]]


class CorpusAnalyzer:
    """
    Analyzes corpus composition and identifies influential works.
    """

    def __init__(self, corpus_path: str | Path):
        self.db_path = Path(corpus_path)

    def find_hub_papers(self, limit: int = 20) -> pd.DataFrame:
        """
        Find the most influential papers in the corpus by citation count.
        
        These are the field-defining works that everyone builds on.
        
        Args:
            limit: Maximum number of papers to return
            
        Returns:
            DataFrame with columns: title, authors, year, citation_count, journal
        """
        with get_db_session(self.db_path) as session:
            papers = session.query(Paper).filter(
                Paper.citation_count.isnot(None)
            ).order_by(
                Paper.citation_count.desc()
            ).limit(limit).all()
            
            if not papers:
                console.print("[yellow]No citation counts available. Run 'literature-mapper citations' first.[/yellow]")
                return pd.DataFrame()
            
            results = []
            for p in papers:
                # Format authors
                if p.authors:
                    if len(p.authors) > 3:
                        authors_str = f"{p.authors[0].name} et al."
                    else:
                        authors_str = ', '.join(a.name for a in p.authors)
                else:
                    authors_str = 'Unknown'
                
                results.append({
                    'title': p.title,
                    'authors': authors_str,
                    'year': p.year,
                    'citation_count': p.citation_count,
                    'journal': p.journal or 'Unknown',
                })
            
            return pd.DataFrame(results)

    def get_statistics(self) -> CorpusStatistics:
        """
        Get comprehensive statistics about the corpus.
        
        Returns:
            CorpusStatistics dataclass with counts and distributions
        """
        with get_db_session(self.db_path) as session:
            total_papers = session.query(Paper).count()
            total_authors = session.query(Author).count()
            total_concepts = session.query(Concept).count()
            
            # Papers with citation data
            papers_with_citations = session.query(Paper).filter(
                Paper.citation_count.isnot(None)
            ).count()
            
            # Total citations (sum of all citation_counts)
            from sqlalchemy import func
            total_citations = session.query(
                func.sum(Paper.citation_count)
            ).scalar() or 0
            
            # Year range
            min_year = session.query(func.min(Paper.year)).scalar()
            max_year = session.query(func.max(Paper.year)).scalar()
            year_range = (min_year, max_year) if min_year and max_year else None
            
            # Top journals
            journal_counts = session.query(
                Paper.journal,
                func.count(Paper.id).label('count')
            ).filter(
                Paper.journal.isnot(None)
            ).group_by(
                Paper.journal
            ).order_by(
                func.count(Paper.id).desc()
            ).limit(10).all()
            
            top_journals = [(j.journal, j.count) for j in journal_counts]
            
            return CorpusStatistics(
                total_papers=total_papers,
                total_authors=total_authors,
                total_concepts=total_concepts,
                papers_with_citations=papers_with_citations,
                total_citations=total_citations,
                year_range=year_range,
                top_journals=top_journals,
            )

    def get_year_distribution(self) -> pd.DataFrame:
        """
        Get paper count by year.
        
        Returns:
            DataFrame with columns: year, count
        """
        with get_db_session(self.db_path) as session:
            from sqlalchemy import func
            
            results = session.query(
                Paper.year,
                func.count(Paper.id).label('count')
            ).filter(
                Paper.year.isnot(None)
            ).group_by(
                Paper.year
            ).order_by(
                Paper.year
            ).all()
            
            return pd.DataFrame([
                {'year': r.year, 'count': r.count}
                for r in results
            ])

    def get_top_authors(self, limit: int = 20) -> pd.DataFrame:
        """
        Get authors with most papers in corpus.
        
        Args:
            limit: Maximum number of authors to return
            
        Returns:
            DataFrame with columns: author, paper_count, total_citations
        """
        with get_db_session(self.db_path) as session:
            from sqlalchemy import func
            from .database import PaperAuthor
            
            # Count papers per author
            results = session.query(
                Author.name,
                func.count(PaperAuthor.paper_id).label('paper_count'),
            ).join(
                PaperAuthor, Author.id == PaperAuthor.author_id
            ).group_by(
                Author.id
            ).order_by(
                func.count(PaperAuthor.paper_id).desc()
            ).limit(limit).all()
            
            return pd.DataFrame([
                {'author': r.name, 'paper_count': r.paper_count}
                for r in results
            ])

    def get_top_concepts(self, limit: int = 30) -> pd.DataFrame:
        """
        Get most frequent concepts in corpus.
        
        Args:
            limit: Maximum number of concepts to return
            
        Returns:
            DataFrame with columns: concept, paper_count
        """
        with get_db_session(self.db_path) as session:
            from sqlalchemy import func
            from .database import PaperConcept
            
            results = session.query(
                Concept.name,
                func.count(PaperConcept.paper_id).label('paper_count'),
            ).join(
                PaperConcept, Concept.id == PaperConcept.concept_id
            ).group_by(
                Concept.id
            ).order_by(
                func.count(PaperConcept.paper_id).desc()
            ).limit(limit).all()
            
            return pd.DataFrame([
                {'concept': r.name, 'paper_count': r.paper_count}
                for r in results
            ])