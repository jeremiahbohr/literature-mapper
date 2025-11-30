"""
ghosts.py - Ghost Node Detection Engine

Identifies structural gaps in a research corpus using citation data.

Modes:
- bibliographic: Papers frequently cited by corpus but not in corpus
- authors: Authors frequently cited but not represented in corpus
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict

import pandas as pd
from rich.console import Console

from .database import get_db_session, Paper, Citation, Author
from .mapper import LiteratureMapper

console = Console()
logger = logging.getLogger(__name__)


class GhostHunter:
    """
    Identifies structural gaps and key nodes in a research corpus.
    
    All methods rely on citation data from OpenAlex - no speculation,
    no LLM-generated concepts, just bibliometric analysis.
    """

    def __init__(self, mapper: LiteratureMapper):
        self.mapper = mapper
        self.db_path = Path(mapper.corpus_path)

    # -------------------------------------------------------------------------
    # MODE: BIBLIOGRAPHIC GHOSTS
    # Papers cited frequently by corpus but missing from corpus
    # -------------------------------------------------------------------------

    def find_bibliographic_ghosts(self, threshold: int = 3) -> pd.DataFrame:
        """
        Find papers that are frequently cited by the corpus but not in it.
        
        These are candidates for inclusion - the corpus clearly builds on
        this work but doesn't include it.
        
        Args:
            threshold: Minimum number of citing papers to be considered a ghost
            
        Returns:
            DataFrame with columns: title, author, year, citation_count
        """
        with get_db_session(self.db_path) as session:
            # Get all citations
            citations = session.query(Citation).all()
            
            if not citations:
                console.print("[yellow]No citations in database. Run 'literature-mapper citations' first.[/yellow]")
                return pd.DataFrame()
            
            # Get existing paper titles for matching
            existing_papers = session.query(Paper).all()
            existing_titles = self._build_title_index(existing_papers)
            
            # Count citation frequency
            citation_counts = defaultdict(list)
            for cit in citations:
                if not cit.title:
                    continue
                # Use normalized title as key
                key = self._normalize_title(cit.title)
                citation_counts[key].append(cit)
            
            # Find ghosts (cited but not in corpus)
            ghosts = []
            for title_key, cits in citation_counts.items():
                if len(cits) < threshold:
                    continue
                
                # Check if this paper is in corpus
                if self._is_in_corpus(title_key, existing_titles):
                    continue
                
                # Use the citation with most complete metadata
                best_cit = max(cits, key=lambda c: (
                    (1 if c.year else 0) + 
                    (1 if c.author else 0) +
                    len(c.title or '')
                ))
                
                ghosts.append({
                    'title': best_cit.title,
                    'author': best_cit.author or 'Unknown',
                    'year': best_cit.year,
                    'citation_count': len(cits),
                })
            
            if not ghosts:
                return pd.DataFrame()
            
            df = pd.DataFrame(ghosts)
            df = df.sort_values('citation_count', ascending=False)
            return df

    # -------------------------------------------------------------------------
    # MODE: MISSING AUTHORS
    # Authors who appear frequently in citations but have no papers in corpus
    # -------------------------------------------------------------------------

    def find_missing_authors(self, threshold: int = 3) -> pd.DataFrame:
        """
        Find authors frequently cited by the corpus but not represented in it.
        
        These are voices that the field considers important but your corpus
        doesn't include directly.
        
        Args:
            threshold: Minimum papers citing this author to be considered
            
        Returns:
            DataFrame with columns: author, paper_count, cited_by_papers, sample_works
        """
        with get_db_session(self.db_path) as session:
            # Get all citations
            citations = session.query(Citation).all()
            
            if not citations:
                console.print("[yellow]No citations in database. Run 'literature-mapper citations' first.[/yellow]")
                return pd.DataFrame()
            
            # Get authors already in corpus
            corpus_authors = set()
            for author in session.query(Author).all():
                corpus_authors.add(self._normalize_author(author.name))
            
            # Count author appearances in citations
            # Track which papers cite them and sample works
            author_data = defaultdict(lambda: {
                'citing_papers': set(),
                'works': []
            })
            
            for cit in citations:
                if not cit.author:
                    continue
                
                # Extract first author (most citations list first author first)
                first_author = self._extract_first_author(cit.author)
                if not first_author:
                    continue
                
                author_key = self._normalize_author(first_author)
                
                # Skip if in corpus
                if author_key in corpus_authors:
                    continue
                
                author_data[author_key]['citing_papers'].add(cit.source_paper_id)
                
                # Keep sample of their works
                if len(author_data[author_key]['works']) < 5:
                    work_info = f"{cit.title[:60]}..." if len(cit.title) > 60 else cit.title
                    if cit.year:
                        work_info += f" ({cit.year})"
                    if work_info not in author_data[author_key]['works']:
                        author_data[author_key]['works'].append(work_info)
            
            # Filter by threshold and build results
            results = []
            for author_key, data in author_data.items():
                citing_count = len(data['citing_papers'])
                if citing_count < threshold:
                    continue
                
                results.append({
                    'author': author_key.title(),  # Capitalize nicely
                    'cited_by_papers': citing_count,
                    'paper_count': len(data['works']),
                    'sample_works': '; '.join(data['works'][:3]),
                })
            
            if not results:
                return pd.DataFrame()
            
            df = pd.DataFrame(results)
            df = df.sort_values('cited_by_papers', ascending=False)
            return df

    # -------------------------------------------------------------------------
    # HELPER METHODS
    # -------------------------------------------------------------------------

    def _normalize_title(self, title: str) -> str:
        """Normalize title for matching."""
        if not title:
            return ''
        import re
        # Lowercase, remove punctuation, collapse whitespace
        title = title.lower()
        title = re.sub(r'[^\w\s]', '', title)
        title = re.sub(r'\s+', ' ', title)
        return title.strip()

    def _normalize_author(self, author: str) -> str:
        """Normalize author name for matching."""
        if not author:
            return ''
        import re
        # Lowercase, remove punctuation except hyphens
        author = author.lower()
        author = re.sub(r'[^\w\s\-]', '', author)
        author = re.sub(r'\s+', ' ', author)
        return author.strip()

    def _extract_first_author(self, author_str: str) -> Optional[str]:
        """Extract first author from author string."""
        if not author_str:
            return None
        
        # Handle "et al." 
        import re
        author_str = re.sub(r'\s+et\.?\s*al\.?.*$', '', author_str, flags=re.IGNORECASE)
        
        # Split by comma, ampersand, or "and"
        parts = re.split(r',|&|\band\b', author_str)
        if parts:
            first = parts[0].strip()
            if len(first) > 2:
                return first
        
        return None

    def _build_title_index(self, papers: List[Paper]) -> Set[str]:
        """Build set of normalized titles from papers."""
        titles = set()
        for p in papers:
            titles.add(self._normalize_title(p.title))
        return titles

    def _is_in_corpus(self, title_key: str, existing_titles: Set[str]) -> bool:
        """Check if a normalized title is in the corpus."""
        if title_key in existing_titles:
            return True
        
        # Fuzzy match: check if title is substring or vice versa
        for existing in existing_titles:
            if len(title_key) > 20 and len(existing) > 20:
                if title_key in existing or existing in title_key:
                    return True
        
        return False