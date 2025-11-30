"""
openalex.py - OpenAlex API client for citation data.

Usage:
    client = OpenAlexClient()
    paper = client.find_paper(title="Attention Is All You Need", year=2017)
    references = client.get_references(paper['id'])
"""

import re
import time
import logging
import requests
from typing import List, Dict, Optional, Any
from urllib.parse import quote

logger = logging.getLogger(__name__)


class OpenAlexClient:
    """
    Client for OpenAlex API.
    
    Docs: https://docs.openalex.org/
    
    No API key required. Be polite with rate limiting.
    They ask for an email in the User-Agent for polite pool (faster).
    """
    
    BASE_URL = "https://api.openalex.org"
    
    def __init__(self, email: str = None):
        """
        Initialize client.
        
        Args:
            email: Optional email for OpenAlex polite pool (faster rate limits)
        """
        self.session = requests.Session()
        
        # Set user agent - OpenAlex gives better rate limits if you identify yourself
        if email:
            self.session.headers['User-Agent'] = f'LiteratureMapper/1.0 (mailto:{email})'
        else:
            self.session.headers['User-Agent'] = 'LiteratureMapper/1.0'
        
        # Simple cache to avoid repeated lookups
        self._cache: Dict[str, Any] = {}
        
        # Rate limiting
        self._last_request = 0
        self._min_interval = 0.1  # 100ms between requests (10/sec)
    
    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request = time.time()
    
    def _get(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make GET request to OpenAlex API."""
        self._rate_limit()
        
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                logger.debug(f"Not found: {url}")
                return None
            logger.error(f"HTTP error: {e}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return None
    
    def find_paper_by_doi(self, doi: str) -> Optional[Dict]:
        """
        Find paper by DOI (exact match).
        
        Args:
            doi: DOI string (with or without https://doi.org/ prefix)
            
        Returns:
            Paper dict or None if not found
        """
        # Normalize DOI
        doi = doi.strip()
        doi = re.sub(r'^https?://(dx\.)?doi\.org/', '', doi)
        
        cache_key = f"doi:{doi}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # OpenAlex uses DOI as ID with prefix
        result = self._get(f"works/https://doi.org/{doi}")
        
        if result:
            paper = self._parse_work(result)
            self._cache[cache_key] = paper
            return paper
        
        return None
    
    def find_paper_by_title(
        self, 
        title: str, 
        year: int = None, 
        first_author: str = None
    ) -> Optional[Dict]:
        """
        Find paper by title (fuzzy match).
        
        Args:
            title: Paper title
            year: Publication year (helps disambiguation)
            first_author: First author surname (helps disambiguation)
            
        Returns:
            Best matching paper dict or None
        """
        cache_key = f"title:{title}:{year}:{first_author}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Build search query
        params = {
            'search': title,
            'per_page': 5,
        }
        
        # Add filters if we have them
        filters = []
        if year:
            # Allow ±1 year for publication date variance
            filters.append(f"publication_year:{year-1}-{year+1}")
        
        if filters:
            params['filter'] = ','.join(filters)
        
        result = self._get("works", params)
        
        if not result or not result.get('results'):
            self._cache[cache_key] = None
            return None
        
        # Find best match
        best_match = self._find_best_match(
            result['results'], 
            title, 
            year, 
            first_author
        )
        
        if best_match:
            paper = self._parse_work(best_match)
            self._cache[cache_key] = paper
            return paper
        
        self._cache[cache_key] = None
        return None
    
    def find_paper(
        self,
        title: str,
        year: int = None,
        first_author: str = None,
        doi: str = None
    ) -> Optional[Dict]:
        """
        Find paper using best available identifier.
        
        Tries DOI first (exact), then falls back to title search (fuzzy).
        
        Args:
            title: Paper title
            year: Publication year
            first_author: First author surname  
            doi: DOI if available
            
        Returns:
            Paper dict or None
        """
        # Try DOI first (exact match)
        if doi:
            paper = self.find_paper_by_doi(doi)
            if paper:
                logger.debug(f"Found by DOI: {title[:50]}")
                return paper
        
        # Fall back to title search
        paper = self.find_paper_by_title(title, year, first_author)
        if paper:
            logger.debug(f"Found by title: {title[:50]}")
        else:
            logger.warning(f"Not found in OpenAlex: {title[:50]}")
        
        return paper
    
    def get_references(self, openalex_id: str) -> List[Dict]:
        """
        Get papers referenced by a given paper.
        
        Args:
            openalex_id: OpenAlex work ID (e.g., 'W2963403868')
            
        Returns:
            List of paper dicts for each reference
        """
        cache_key = f"refs:{openalex_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Get the work's referenced_works
        result = self._get(f"works/{openalex_id}")
        
        if not result:
            self._cache[cache_key] = []
            return []
        
        referenced_ids = result.get('referenced_works', [])
        
        if not referenced_ids:
            self._cache[cache_key] = []
            return []
        
        # Fetch details for each reference (batched)
        references = self._fetch_works_batch(referenced_ids)
        
        self._cache[cache_key] = references
        return references
    
    def _fetch_works_batch(self, work_ids: List[str], batch_size: int = 50) -> List[Dict]:
        """Fetch multiple works by ID in batches."""
        all_works = []
        
        for i in range(0, len(work_ids), batch_size):
            batch = work_ids[i:i + batch_size]
            
            # Use filter to get multiple works at once
            # IDs are like "https://openalex.org/W2963403868"
            # Extract just the W... part
            clean_ids = []
            for wid in batch:
                if isinstance(wid, str):
                    match = re.search(r'W\d+', wid)
                    if match:
                        clean_ids.append(match.group())
            
            if not clean_ids:
                continue
            
            # Query with OR filter
            filter_str = '|'.join(clean_ids)
            params = {
                'filter': f'openalex_id:{filter_str}',
                'per_page': batch_size,
            }
            
            result = self._get("works", params)
            
            if result and result.get('results'):
                for work in result['results']:
                    parsed = self._parse_work(work)
                    if parsed:
                        all_works.append(parsed)
        
        return all_works
    
    def _parse_work(self, work: Dict) -> Optional[Dict]:
        """Parse OpenAlex work into our format."""
        if not work:
            return None
        
        # Extract authors
        authors = []
        authorships = work.get('authorships', [])
        for authorship in authorships[:10]:  # Limit to first 10
            author = authorship.get('author', {})
            name = author.get('display_name')
            if name:
                authors.append(name)
        
        # Extract title
        title = work.get('title') or work.get('display_name') or ''
        
        # Extract year
        year = work.get('publication_year')
        
        # Extract DOI
        doi = work.get('doi')
        if doi:
            doi = re.sub(r'^https?://doi\.org/', '', doi)
        
        # Extract venue/journal
        venue = None
        primary_location = work.get('primary_location', {})
        if primary_location:
            source = primary_location.get('source', {})
            if source:
                venue = source.get('display_name')
        
        # Citation count
        cited_by_count = work.get('cited_by_count', 0)
        
        # OpenAlex ID
        openalex_id = work.get('id', '')
        if openalex_id:
            match = re.search(r'W\d+', openalex_id)
            openalex_id = match.group() if match else openalex_id
        
        return {
            'openalex_id': openalex_id,
            'title': title,
            'authors': authors,
            'author': ', '.join(authors),  # Convenience field
            'year': year,
            'doi': doi,
            'venue': venue,
            'cited_by_count': cited_by_count,
        }
    
    def _find_best_match(
        self, 
        candidates: List[Dict], 
        title: str, 
        year: int = None, 
        first_author: str = None
    ) -> Optional[Dict]:
        """Find best matching paper from search results."""
        
        if not candidates:
            return None
        
        title_lower = title.lower().strip()
        title_words = set(re.findall(r'\w+', title_lower))
        
        best_score = 0
        best_match = None
        
        for candidate in candidates:
            score = 0
            
            # Title similarity (most important)
            cand_title = (candidate.get('title') or '').lower().strip()
            cand_words = set(re.findall(r'\w+', cand_title))
            
            if title_words and cand_words:
                # Jaccard similarity
                intersection = len(title_words & cand_words)
                union = len(title_words | cand_words)
                title_sim = intersection / union if union > 0 else 0
                score += title_sim * 100
            
            # Exact title match bonus
            if cand_title == title_lower:
                score += 50
            
            # Year match
            cand_year = candidate.get('publication_year')
            if year and cand_year:
                if cand_year == year:
                    score += 20
                elif abs(cand_year - year) == 1:
                    score += 10
            
            # Author match
            if first_author:
                first_author_lower = first_author.lower()
                authorships = candidate.get('authorships', [])
                for authorship in authorships[:3]:
                    author = authorship.get('author', {})
                    author_name = (author.get('display_name') or '').lower()
                    if first_author_lower in author_name:
                        score += 15
                        break
            
            if score > best_score:
                best_score = score
                best_match = candidate
        
        # Require minimum score to accept match
        if best_score < 50:
            logger.debug(f"Best match score {best_score} below threshold for: {title[:50]}")
        
        # Fall back to title search
        paper = self.find_paper_by_title(title, year, first_author)
        if paper:
            logger.debug(f"Found by title: {title[:50]}")
        else:
            logger.warning(f"Not found in OpenAlex: {title[:50]}")
        
        return paper
    
    def get_references(self, openalex_id: str) -> List[Dict]:
        """
        Get papers referenced by a given paper.
        
        Args:
            openalex_id: OpenAlex work ID (e.g., 'W2963403868')
            
        Returns:
            List of paper dicts for each reference
        """
        cache_key = f"refs:{openalex_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Get the work's referenced_works
        result = self._get(f"works/{openalex_id}")
        
        if not result:
            self._cache[cache_key] = []
            return []
        
        referenced_ids = result.get('referenced_works', [])
        
        if not referenced_ids:
            self._cache[cache_key] = []
            return []
        
        # Fetch details for each reference (batched)
        references = self._fetch_works_batch(referenced_ids)
        
        self._cache[cache_key] = references
        return references
    
    def _fetch_works_batch(self, work_ids: List[str], batch_size: int = 50) -> List[Dict]:
        """Fetch multiple works by ID in batches."""
        all_works = []
        
        for i in range(0, len(work_ids), batch_size):
            batch = work_ids[i:i + batch_size]
            
            # Use filter to get multiple works at once
            # IDs are like "https://openalex.org/W2963403868"
            # Extract just the W... part
            clean_ids = []
            for wid in batch:
                if isinstance(wid, str):
                    match = re.search(r'W\d+', wid)
                    if match:
                        clean_ids.append(match.group())
            
            if not clean_ids:
                continue
            
            # Query with OR filter
            filter_str = '|'.join(clean_ids)
            params = {
                'filter': f'openalex_id:{filter_str}',
                'per_page': batch_size,
            }
            
            result = self._get("works", params)
            
            if result and result.get('results'):
                for work in result['results']:
                    parsed = self._parse_work(work)
                    if parsed:
                        all_works.append(parsed)
        
        return all_works
    
    def _parse_work(self, work: Dict) -> Optional[Dict]:
        """Parse OpenAlex work into our format."""
        if not work:
            return None
        
        # Extract authors
        authors = []
        authorships = work.get('authorships', [])
        for authorship in authorships[:10]:  # Limit to first 10
            author = authorship.get('author', {})
            name = author.get('display_name')
            if name:
                authors.append(name)
        
        # Extract title
        title = work.get('title') or work.get('display_name') or ''
        
        # Extract year
        year = work.get('publication_year')
        
        # Extract DOI
        doi = work.get('doi')
        if doi:
            doi = re.sub(r'^https?://doi\.org/', '', doi)
        
        # Extract venue/journal
        venue = None
        primary_location = work.get('primary_location', {})
        if primary_location:
            source = primary_location.get('source', {})
            if source:
                venue = source.get('display_name')
        
        # Citation count
        cited_by_count = work.get('cited_by_count', 0)
        
        # OpenAlex ID
        openalex_id = work.get('id', '')
        if openalex_id:
            match = re.search(r'W\d+', openalex_id)
            openalex_id = match.group() if match else openalex_id
        
        return {
            'openalex_id': openalex_id,
            'title': title,
            'authors': authors,
            'author': ', '.join(authors),  # Convenience field
            'year': year,
            'doi': doi,
            'venue': venue,
            'cited_by_count': cited_by_count,
        }
    
    def _find_best_match(
        self, 
        candidates: List[Dict], 
        title: str, 
        year: int = None, 
        first_author: str = None
    ) -> Optional[Dict]:
        """Find best matching paper from search results."""
        
        if not candidates:
            return None
        
        title_lower = title.lower().strip()
        title_words = set(re.findall(r'\w+', title_lower))
        
        best_score = 0
        best_match = None
        
        for candidate in candidates:
            score = 0
            
            # Title similarity (most important)
            cand_title = (candidate.get('title') or '').lower().strip()
            cand_words = set(re.findall(r'\w+', cand_title))
            
            if title_words and cand_words:
                # Jaccard similarity
                intersection = len(title_words & cand_words)
                union = len(title_words | cand_words)
                title_sim = intersection / union if union > 0 else 0
                score += title_sim * 100
            
            # Exact title match bonus
            if cand_title == title_lower:
                score += 50
            
            # Year match
            cand_year = candidate.get('publication_year')
            if year and cand_year:
                if cand_year == year:
                    score += 20
                elif abs(cand_year - year) == 1:
                    score += 10
            
            # Author match
            if first_author:
                first_author_lower = first_author.lower()
                authorships = candidate.get('authorships', [])
                for authorship in authorships[:3]:
                    author = authorship.get('author', {})
                    author_name = (author.get('display_name') or '').lower()
                    if first_author_lower in author_name:
                        score += 15
                        break
            
            if score > best_score:
                best_score = score
                best_match = candidate
        
        # Require minimum score to accept match
        if best_score < 50:
            logger.debug(f"Best match score {best_score} below threshold for: {title[:50]}")
            return None
        
        return best_match


def fetch_citations_for_corpus(corpus_path: str, email: str = None) -> Dict[str, int]:
    """
    Fetch citations from OpenAlex for all papers in corpus.
    
    Args:
        corpus_path: Path to corpus directory
        email: Optional email for OpenAlex polite pool
        
    Returns:
        Dict with counts: {'found': N, 'not_found': N, 'citations': N}
    """
    from pathlib import Path
    from .database import get_db_session, Paper, Citation
    from sqlalchemy import func
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    
    console = Console()
    corpus_path = Path(corpus_path)
    client = OpenAlexClient(email=email)
    
    stats = {'found': 0, 'not_found': 0, 'citations': 0, 'updated': 0, 'arxiv_fallback': 0}
    
    with get_db_session(corpus_path) as session:
        papers = session.query(Paper).all()
        
        if not papers:
            console.print("[yellow]No papers in corpus[/yellow]")
            return stats
        
        console.print(f"[blue]Fetching citations for {len(papers)} papers from OpenAlex...[/blue]")
        
        for paper in papers:
            try:
                # Get first author surname
                first_author = None
                if paper.authors:
                    first_author_name = paper.authors[0].name
                    # Extract surname (last word, or word before comma)
                    if ',' in first_author_name:
                        first_author = first_author_name.split(',')[0].strip()
                    else:
                        first_author = first_author_name.split()[-1].strip()
                
                console.print(f"[dim]Looking up: {paper.title[:50]}...[/dim]")
                
                # Find paper in OpenAlex
                oa_paper = client.find_paper(
                    title=paper.title,
                    year=paper.year,
                    first_author=first_author,
                    doi=paper.doi
                )
                
                # If not found in OpenAlex and we have arXiv ID, try arXiv
                if not oa_paper and paper.arxiv_id:
                    from .arxiv_api import ArxivClient
                    arxiv_client = ArxivClient()
                    console.print(f"[dim]  - Not in OpenAlex, trying arXiv...[/dim]")
                    
                    arxiv_paper = arxiv_client.find_paper_by_arxiv_id(paper.arxiv_id)
                    
                    if arxiv_paper:
                        console.print(f"[green]  - Found on arXiv[/green]")
                        stats['arxiv_fallback'] += 1
                        
                        # Update paper DOI if we got it from arXiv
                        if arxiv_paper.get('doi') and not paper.doi:
                            paper.doi = arxiv_paper['doi']
                        
                        # arXiv doesn't provide references, so we can't extract citations
                        # But we mark it as found
                        continue
                
                if not oa_paper:
                    stats['not_found'] += 1
                    console.print(f"[yellow]  - Not found in OpenAlex or arXiv[/yellow]")
                    continue
                
                stats['found'] += 1
                
                # Update citation count on paper
                if oa_paper.get('cited_by_count'):
                    paper.citation_count = oa_paper['cited_by_count']
                    stats['updated'] += 1
                
                # Update DOI if we didn't have it
                if oa_paper.get('doi') and not paper.doi:
                    paper.doi = oa_paper['doi']
                
                # Update arXiv ID if we didn't have it
                # (OpenAlex sometimes includes arXiv IDs)
                if not paper.arxiv_id:
                    # Check if DOI is an arXiv DOI
                    openalex_id = oa_paper.get('openalex_id', '')
                    if 'arxiv' in openalex_id.lower():
                        from .arxiv_api import ArxivClient
                        arxiv_client = ArxivClient()
                        arxiv_id = arxiv_client.extract_arxiv_id(openalex_id)
                        if arxiv_id:
                            paper.arxiv_id = arxiv_id
                
                # Get references
                references = client.get_references(oa_paper['openalex_id'])
                
                if not references:
                    console.print(f"[dim]  - Found paper, but no references listed[/dim]")
                    continue
                
                # Clear existing citations for this paper
                session.query(Citation).filter(
                    Citation.source_paper_id == paper.id
                ).delete()
                
                # Save new citations
                for ref in references:
                    if not ref.get('title'):
                        continue
                    
                    citation = Citation(
                        source_paper_id=paper.id,
                        title=ref['title'][:500],
                        author=ref.get('author', '')[:500],
                        year=ref.get('year'),
                        openalex_id=ref.get('openalex_id')  # Store OpenAlex ID
                    )
                    session.add(citation)
                    stats['citations'] += 1
                
                console.print(f"[green]  - Found {len(references)} references (cited by {oa_paper.get('cited_by_count', 0)})[/green]")
                
            except Exception as e:
                console.print(f"[red]  - Error: {e}[/red]")
                logger.error(f"Error fetching citations for paper {paper.id}: {e}")
                continue
        
        session.commit()
    
    console.print(f"\n[bold green]Done![/bold green]")
    console.print(f"  Papers found: {stats['found']}/{len(papers)}")
    console.print(f"  arXiv fallback: {stats['arxiv_fallback']}")
    console.print(f"  Citations fetched: {stats['citations']}")
    console.print(f"  Citation counts updated: {stats['updated']}")
    
    return stats