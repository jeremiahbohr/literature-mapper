"""
temporal.py - Temporal Analysis Engine

Populates and analyzes temporal statistics for concepts in the corpus.
Must be run after papers are processed to materialize the ConceptTemporalStats table
and compute trend metrics on Concept records.
"""

import logging
from pathlib import Path
from typing import Optional
from collections import defaultdict
from datetime import datetime

import numpy as np
from sqlalchemy import func

from .database import (
    get_db_session, 
    Paper, 
    Concept, 
    PaperConcept, 
    ConceptTemporalStats
)

logger = logging.getLogger(__name__)


def compute_temporal_stats(corpus_path: str | Path, verbose: bool = True) -> dict:
    """
    Compute and store temporal statistics for all concepts.
    
    This materializes the ConceptTemporalStats table and updates
    aggregate fields on Concept (first_year, last_year, peak_year, trend_slope).
    
    Run this after processing papers or fetching citations.
    
    Args:
        corpus_path: Path to corpus directory
        verbose: Print progress
        
    Returns:
        Dict with counts: {'concepts_updated': N, 'stats_rows': N}
    """
    corpus_path = Path(corpus_path)
    stats = {'concepts_updated': 0, 'stats_rows': 0}
    current_year = datetime.now().year
    
    with get_db_session(corpus_path) as session:
        # Clear existing temporal stats (full recompute)
        session.query(ConceptTemporalStats).delete()
        session.commit()
        
        concepts = session.query(Concept).all()
        
        if not concepts:
            if verbose:
                print("No concepts in corpus.")
            return stats
        
        if verbose:
            print(f"Computing temporal stats for {len(concepts)} concepts...")
        
        for concept in concepts:
            # Get all papers with this concept, grouped by year
            paper_data = (
                session.query(
                    Paper.year,
                    func.count(Paper.id).label('paper_count'),
                    func.sum(Paper.citation_count).label('citation_sum'),
                    func.avg(Paper.citations_per_year).label('avg_cpy')
                )
                .join(PaperConcept, Paper.id == PaperConcept.paper_id)
                .filter(PaperConcept.concept_id == concept.id)
                .filter(Paper.year.isnot(None))
                .group_by(Paper.year)
                .order_by(Paper.year)
                .all()
            )
            
            if not paper_data:
                continue
            
            years = []
            counts = []
            
            for row in paper_data:
                year = row.year
                paper_count = row.paper_count
                citation_sum = row.citation_sum
                avg_cpy = row.avg_cpy
                
                # Create temporal stats row
                ts = ConceptTemporalStats(
                    concept_id=concept.id,
                    year=year,
                    paper_count=paper_count,
                    citation_sum=citation_sum or 0,
                    citations_per_year=float(avg_cpy) if avg_cpy else None
                )
                session.add(ts)
                stats['stats_rows'] += 1
                
                years.append(year)
                counts.append(paper_count)
            
            # Update aggregate fields on Concept
            concept.first_year = min(years)
            concept.last_year = max(years)
            
            # Peak year = year with most papers
            max_count = max(counts)
            peak_idx = counts.index(max_count)
            concept.peak_year = years[peak_idx]
            
            # Trend slope: linear regression of paper_count over years
            # Positive = growing, negative = declining
            if len(years) >= 2:
                concept.trend_slope = _compute_trend_slope(years, counts)
            else:
                concept.trend_slope = 0.0
            
            stats['concepts_updated'] += 1
        
        session.commit()
    
    if verbose:
        print(f"Done. Updated {stats['concepts_updated']} concepts, "
              f"created {stats['stats_rows']} temporal stat rows.")
    
    return stats


def _compute_trend_slope(years: list[int], counts: list[int]) -> float:
    """
    Compute linear regression slope for concept usage over time.
    
    Returns papers-per-year change. Positive means growing adoption.
    """
    if len(years) < 2:
        return 0.0
    
    x = np.array(years, dtype=float)
    y = np.array(counts, dtype=float)
    
    # Normalize x to start at 0 for numerical stability
    x = x - x.min()
    
    # Simple linear regression: slope = cov(x,y) / var(x)
    n = len(x)
    x_mean = x.mean()
    y_mean = y.mean()
    
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    
    if denominator == 0:
        return 0.0
    
    slope = numerator / denominator
    return round(float(slope), 4)


def get_trending_concepts(
    corpus_path: str | Path,
    direction: str = "rising",
    min_papers: int = 3,
    limit: int = 20
) -> list[dict]:
    """
    Get concepts with strongest positive or negative trends.
    
    Args:
        corpus_path: Path to corpus directory
        direction: "rising" (positive slope) or "declining" (negative slope)
        min_papers: Minimum total papers for concept to be considered
        limit: Max results
        
    Returns:
        List of dicts with concept info and trend data
    """
    corpus_path = Path(corpus_path)
    
    with get_db_session(corpus_path) as session:
        query = (
            session.query(Concept)
            .join(PaperConcept, Concept.id == PaperConcept.concept_id)
            .group_by(Concept.id)
            .having(func.count(PaperConcept.paper_id) >= min_papers)
            .filter(Concept.trend_slope.isnot(None))
        )
        
        if direction == "rising":
            query = query.order_by(Concept.trend_slope.desc())
        else:
            query = query.order_by(Concept.trend_slope.asc())
        
        concepts = query.limit(limit).all()
        
        results = []
        for c in concepts:
            paper_count = (
                session.query(func.count(PaperConcept.paper_id))
                .filter(PaperConcept.concept_id == c.id)
                .scalar()
            )
            
            results.append({
                "concept": c.name,
                "trend_slope": c.trend_slope,
                "first_year": c.first_year,
                "last_year": c.last_year,
                "peak_year": c.peak_year,
                "total_papers": paper_count
            })
        
        return results


def get_concept_trajectory(corpus_path: str | Path, concept_name: str) -> list[dict]:
    """
    Get year-by-year trajectory for a specific concept.
    
    Args:
        corpus_path: Path to corpus directory
        concept_name: Concept to analyze (case-insensitive)
        
    Returns:
        List of dicts with year, paper_count, citation_sum, citations_per_year
    """
    corpus_path = Path(corpus_path)
    
    with get_db_session(corpus_path) as session:
        concept = (
            session.query(Concept)
            .filter(Concept.name.ilike(f"%{concept_name}%"))
            .first()
        )
        
        if not concept:
            return []
        
        stats = (
            session.query(ConceptTemporalStats)
            .filter(ConceptTemporalStats.concept_id == concept.id)
            .order_by(ConceptTemporalStats.year)
            .all()
        )
        
        return [
            {
                "year": s.year,
                "paper_count": s.paper_count,
                "citation_sum": s.citation_sum,
                "citations_per_year": s.citations_per_year
            }
            for s in stats
        ]


def detect_concept_eras(
    corpus_path: str | Path,
    gap_threshold: int = 3
) -> list[dict]:
    """
    Detect distinct eras/waves for concepts based on publication gaps.
    
    A "gap" is when a concept has no papers for gap_threshold+ years,
    then reappears. This can indicate paradigm shifts or revivals.
    
    Args:
        corpus_path: Path to corpus directory
        gap_threshold: Years of silence to constitute a gap
        
    Returns:
        List of concepts with multiple eras
    """
    corpus_path = Path(corpus_path)
    
    with get_db_session(corpus_path) as session:
        concepts = session.query(Concept).all()
        
        results = []
        
        for concept in concepts:
            stats = (
                session.query(ConceptTemporalStats)
                .filter(ConceptTemporalStats.concept_id == concept.id)
                .order_by(ConceptTemporalStats.year)
                .all()
            )
            
            if len(stats) < 2:
                continue
            
            years = [s.year for s in stats]
            
            # Find gaps
            eras = []
            era_start = years[0]
            
            for i in range(1, len(years)):
                gap = years[i] - years[i-1]
                if gap > gap_threshold:
                    eras.append((era_start, years[i-1]))
                    era_start = years[i]
            
            # Close final era
            eras.append((era_start, years[-1]))
            
            if len(eras) > 1:
                results.append({
                    "concept": concept.name,
                    "eras": eras,
                    "num_eras": len(eras),
                    "total_years_active": sum(e[1] - e[0] + 1 for e in eras)
                })
        
        # Sort by number of eras
        results.sort(key=lambda x: x["num_eras"], reverse=True)
        return results


__all__ = [
    'compute_temporal_stats',
    'get_trending_concepts', 
    'get_concept_trajectory',
    'detect_concept_eras'
]