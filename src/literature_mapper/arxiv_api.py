"""
arxiv_api.py - arXiv API client for retrieving paper metadata.

Usage:
    client = ArxivClient()
    paper = client.find_paper(title="Attention Is All You Need", authors=["Vaswani"])
    arxiv_id = client.extract_arxiv_id("arXiv:1706.03762v5")
"""

import re
import time
import logging
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
from urllib.parse import quote

logger = logging.getLogger(__name__)


class ArxivClient:
    """
    Client for arXiv API.
    
    Docs: https://arxiv.org/help/api/
    
    No API key required. Rate limit: ~1 request per 3 seconds.
    """
    
    BASE_URL = "http://export.arxiv.org/api/query"
    
    def __init__(self):
        """Initialize client."""
        self.session = requests.Session()
        self.session.headers['User-Agent'] = 'LiteratureMapper/1.0'
        
        # Rate limiting
        self._last_request = 0
        self._min_interval = 3.0  # 3 seconds between requests (conservative)
    
    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request = time.time()
    
    def extract_arxiv_id(self, text: str) -> Optional[str]:
        """
        Extract arXiv ID from text.
        
        Patterns supported:
        - arXiv:1234.5678
        - arXiv:1234.5678v2
        - arXiv:hep-th/9901001
        
        Args:
            text: Text containing arXiv ID
            
        Returns:
            Normalized arXiv ID or None
        """
        if not text:
            return None
        
        # Modern format: arXiv:YYMM.NNNNN[vN]
        modern_pattern = r'arXiv:(\d{4}\.\d{4,5}(?:v\d+)?)'
        match = re.search(modern_pattern, text, re.IGNORECASE)
        
        if match:
            return match.group(1)
        
        # Legacy format: arXiv:archive/YYMMNNN
        legacy_pattern = r'arXiv:([a-z\-]+/\d{7})'
        match = re.search(legacy_pattern, text, re.IGNORECASE)
        
        if match:
            return match.group(1)
        
        return None
    
    def find_paper_by_arxiv_id(self, arxiv_id: str) -> Optional[Dict]:
        """
        Find paper by arXiv ID (exact match).
        
        Args:
            arxiv_id: arXiv identifier (e.g., "1706.03762" or "1706.03762v5")
            
        Returns:
            Paper dict or None if not found
        """
        self._rate_limit()
        
        # Normalize ID (remove version if present for search)
        clean_id = re.sub(r'v\d+$', '', arxiv_id)
        
        params = {
            'id_list': clean_id,
            'max_results': 1
        }
        
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            
            # Namespace handling
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            entries = root.findall('atom:entry', ns)
            
            if not entries:
                logger.debug(f"arXiv ID not found: {arxiv_id}")
                return None
            
            return self._parse_entry(entries[0], ns)
            
        except Exception as e:
            logger.error(f"arXiv API error for ID {arxiv_id}: {e}")
            return None
    
    def find_paper_by_title(
        self,
        title: str,
        authors: Optional[List[str]] = None,
        max_results: int = 5
    ) -> Optional[Dict]:
        """
        Find paper by title search (fuzzy match).
        
        Args:
            title: Paper title
            authors: Optional list of author names for disambiguation
            max_results: Maximum results to fetch
            
        Returns:
            Best matching paper dict or None
        """
        self._rate_limit()
        
        # Build search query
        # arXiv search syntax: ti:"exact title" or all:keywords
        # For better fuzzy matching, use all: with key words
        title_words = ' AND '.join([f'"{word}"' for word in title.split()[:5]])  # Limit to first 5 words
        
        search_query = f'ti:{title_words}'
        
        params = {
            'search_query': search_query,
            'max_results': max_results,
            'sortBy': 'relevance'
        }
        
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            entries = root.findall('atom:entry', ns)
            
            if not entries:
                logger.debug(f"No arXiv results for title: {title[:50]}")
                return None
            
            # Parse all entries and find best match
            candidates = [self._parse_entry(entry, ns) for entry in entries]
            
            # Simple title similarity matching
            best_match = self._find_best_match(candidates, title, authors)
            
            return best_match
            
        except Exception as e:
            logger.error(f"arXiv API error for title search: {e}")
            return None
    
    def _parse_entry(self, entry, ns: dict) -> Dict:
        """Parse arXiv API entry XML into our format."""
        
        # Extract ID
        id_elem = entry.find('atom:id', ns)
        arxiv_url = id_elem.text if id_elem is not None else ''
        arxiv_id = arxiv_url.split('/')[-1] if arxiv_url else ''
        
        # Extract title
        title_elem = entry.find('atom:title', ns)
        title = title_elem.text.strip() if title_elem is not None else ''
        
        # Extract authors
        authors = []
        for author_elem in entry.findall('atom:author', ns):
            name_elem = author_elem.find('atom:name', ns)
            if name_elem is not None:
                authors.append(name_elem.text.strip())
        
        # Extract published date (year)
        published_elem = entry.find('atom:published', ns)
        year = None
        if published_elem is not None:
            # Format: YYYY-MM-DDTHH:MM:SSZ
            date_str = published_elem.text
            year_match = re.match(r'(\d{4})', date_str)
            if year_match:
                year = int(year_match.group(1))
        
        # Extract abstract
        summary_elem = entry.find('atom:summary', ns)
        abstract = summary_elem.text.strip() if summary_elem is not None else ''
        
        # Extract DOI if present
        doi = None
        for link in entry.findall('atom:link', ns):
            if link.get('title') == 'doi':
                doi_url = link.get('href', '')
                doi = re.sub(r'^https?://doi\.org/', '', doi_url)
        
        return {
            'arxiv_id': arxiv_id,
            'title': title,
            'authors': authors,
            'author': ', '.join(authors),  # Convenience field
            'year': year,
            'abstract': abstract,
            'doi': doi,
        }
    
    def _find_best_match(
        self,
        candidates: List[Dict],
        title: str,
        authors: Optional[List[str]] = None
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
            cand_title = candidate.get('title', '').lower().strip()
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
            
            # Author match (if provided)
            if authors:
                cand_authors = candidate.get('authors', [])
                for auth in authors:
                    auth_lower = auth.lower()
                    for cand_auth in cand_authors:
                        if auth_lower in cand_auth.lower():
                            score += 15
                            break
            
            if score > best_score:
                best_score = score
                best_match = candidate
        
        # Require minimum score to accept match
        if best_score < 30:
            logger.debug(f"Best arXiv match score {best_score} below threshold")
            return None
        
        return best_match


def extract_arxiv_id_from_pdf_text(text: str) -> Optional[str]:
    """
    Convenience function to extract arXiv ID from PDF text.
    
    Args:
        text: PDF text (typically first page)
        
    Returns:
        arXiv ID or None
    """
    client = ArxivClient()
    return client.extract_arxiv_id(text)
