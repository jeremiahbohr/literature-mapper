"""
retrieval.py - Enhanced Retrieval Engine for Literature Mapper
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from .database import get_db_session, Paper, KGNode, KGEdge
from .embeddings import cosine_similarity

logger = logging.getLogger(__name__)

# Current year for recency calculations
CURRENT_YEAR = datetime.now().year


@dataclass
class RetrievedNode:
    """Rich representation of a retrieved KG node with full context."""
    
    # Core identifiers
    node_id: int
    paper_id: int
    
    # Node content
    node_type: str
    label: str
    confidence: Optional[float] = None
    claim_type: Optional[str] = None
    
    # Scores
    semantic_score: float = 0.0
    influence_score: float = 0.0
    recency_score: float = 0.0
    final_score: float = 0.0
    
    # Paper context
    paper_title: str = ""
    paper_year: Optional[int] = None
    paper_journal: Optional[str] = None
    paper_methodology: Optional[str] = None
    paper_core_argument: Optional[str] = None
    paper_citations_per_year: Optional[float] = None
    authors: List[str] = field(default_factory=list)
    
    # Edge context (1-hop neighbors)
    connected_nodes: List[Dict[str, Any]] = field(default_factory=list)
    
    # Embedding for MMR
    vector: Optional[np.ndarray] = None
    
    def get_citation_key(self) -> str:
        """Format as 'Author et al., Year' citation."""
        if not self.authors:
            return f"Unknown, {self.paper_year or '?'}"
        
        first_author = self.authors[0].split()[-1]  # Last name
        if len(self.authors) > 1:
            return f"{first_author} et al., {self.paper_year}"
        return f"{first_author}, {self.paper_year}"


@dataclass 
class ConsensusGroup:
    """A group of similar claims from multiple papers."""
    
    canonical_label: str
    node_type: str
    supporting_nodes: List[RetrievedNode] = field(default_factory=list)
    avg_score: float = 0.0
    
    @property
    def paper_count(self) -> int:
        """Number of unique papers supporting this claim."""
        return len(set(n.paper_id for n in self.supporting_nodes))
    
    @property
    def years_range(self) -> Tuple[int, int]:
        """Range of years for supporting papers."""
        years = [n.paper_year for n in self.supporting_nodes if n.paper_year]
        if not years:
            return (0, 0)
        return (min(years), max(years))
    
    def get_citations(self) -> List[str]:
        """Get all citation keys for this consensus."""
        return [n.get_citation_key() for n in self.supporting_nodes]


class EnhancedRetriever:
    """
    Enhanced retrieval engine with rich context, edge traversal,
    consensus detection, MMR diversity, and blended scoring.
    """
    
    def __init__(
        self,
        corpus_path,
        embedding_generator,
        search_threshold: float = 0.4,
        # Scoring weights
        semantic_weight: float = 0.6,
        influence_weight: float = 0.2,
        recency_weight: float = 0.2,
        # MMR parameter
        mmr_lambda: float = 0.7,  # Balance relevance (1.0) vs diversity (0.0)
        # Consensus threshold
        consensus_similarity_threshold: float = 0.92,
    ):
        self.corpus_path = corpus_path
        self.embedding_generator = embedding_generator
        self.search_threshold = search_threshold
        
        # Scoring weights (should sum to ~1.0)
        self.semantic_weight = semantic_weight
        self.influence_weight = influence_weight
        self.recency_weight = recency_weight
        
        self.mmr_lambda = mmr_lambda
        self.consensus_similarity_threshold = consensus_similarity_threshold
    
    # Blended Scoring
    
    def _compute_influence_score(self, paper: Paper) -> float:
        """
        Compute normalized influence score from citations_per_year.
        
        Uses log scale to prevent highly-cited papers from dominating.
        Returns value in [0, 1] range.
        """
        cpy = paper.citations_per_year
        if cpy is None or cpy <= 0:
            return 0.3  # Default for papers without citation data
        
        # Log scale: log(1 + cpy) / log(1 + max_expected_cpy)
        # Assume ~100 citations/year is exceptional
        max_cpy = 100.0
        score = math.log(1 + cpy) / math.log(1 + max_cpy)
        return min(1.0, score)
    
    def _compute_recency_score(self, paper: Paper) -> float:
        """
        Compute recency score with boost for recent papers.
        
        - Papers from last 2 years: 1.0
        - Papers from 3-5 years ago: 0.8
        - Papers from 6-10 years ago: 0.6
        - Older papers: 0.4 (still valuable as foundational work)
        """
        if not paper.year:
            return 0.5  # Unknown year
        
        age = CURRENT_YEAR - paper.year
        
        if age <= 2:
            return 1.0
        elif age <= 5:
            return 0.8
        elif age <= 10:
            return 0.6
        else:
            return 0.4
    
    def _compute_final_score(
        self,
        semantic_score: float,
        influence_score: float,
        recency_score: float,
        confidence: Optional[float] = None,
    ) -> float:
        """
        Compute blended final score.
        
        If node has claim_confidence, factor it in as a multiplier.
        """
        base_score = (
            self.semantic_weight * semantic_score +
            self.influence_weight * influence_score +
            self.recency_weight * recency_score
        )
        
        # Confidence multiplier (default 1.0 if not present)
        confidence_mult = confidence if confidence is not None else 1.0
        
        return base_score * confidence_mult
    
    # 1-Hop Edge Traversal
    
    def _get_connected_nodes(self, session, node_id: int, limit: int = 5) -> List[Dict]:
        """
        Get 1-hop neighbors for a node via edges.
        
        Returns edges with relationship type and target node info.
        Prioritizes SUPPORTS, CONTRADICTS, EXTENDS relationships.
        """
        # Outgoing edges (this node -> other)
        outgoing = (
            session.query(KGEdge, KGNode)
            .join(KGNode, KGEdge.target_id == KGNode.id)
            .filter(KGEdge.source_id == node_id)
            .limit(limit)
            .all()
        )
        
        # Incoming edges (other -> this node)
        incoming = (
            session.query(KGEdge, KGNode)
            .join(KGNode, KGEdge.source_id == KGNode.id)
            .filter(KGEdge.target_id == node_id)
            .limit(limit)
            .all()
        )
        
        connected = []
        
        # Priority edge types
        priority_types = {'SUPPORTS', 'CONTRADICTS', 'EXTENDS', 'CHALLENGES', 'REFUTES'}
        
        for edge, target_node in outgoing:
            paper = session.get(Paper, target_node.source_paper_id)
            year = paper.year if paper else None
            
            connected.append({
                'direction': 'outgoing',
                'edge_type': edge.type,
                'node_type': target_node.type,
                'label': target_node.label,
                'year': year,
                'priority': edge.type in priority_types,
            })
        
        for edge, source_node in incoming:
            paper = session.get(Paper, source_node.source_paper_id)
            year = paper.year if paper else None
            
            connected.append({
                'direction': 'incoming',
                'edge_type': edge.type,
                'node_type': source_node.type,
                'label': source_node.label,
                'year': year,
                'priority': edge.type in priority_types,
            })
        
        # Sort by priority, then limit
        connected.sort(key=lambda x: (not x['priority'], x['label']))
        return connected[:limit]
    
    # MMR Diversity
    
    def _mmr_rerank(
        self,
        candidates: List[RetrievedNode],
        query_vector: np.ndarray,
        limit: int,
    ) -> List[RetrievedNode]:
        """
        Apply Maximal Marginal Relevance to balance relevance and diversity.
        
        MMR score = λ * relevance - (1-λ) * max_similarity_to_selected
        
        This prevents returning 5 variants of the same finding.
        """
        if len(candidates) <= limit:
            return candidates
        
        selected: List[RetrievedNode] = []
        remaining = list(candidates)
        
        while len(selected) < limit and remaining:
            best_score = float('-inf')
            best_idx = 0
            
            for i, candidate in enumerate(remaining):
                # Relevance: use final_score (already computed)
                relevance = candidate.final_score
                
                # Diversity: max similarity to any already-selected node
                max_sim = 0.0
                if selected and candidate.vector is not None:
                    for sel in selected:
                        if sel.vector is not None:
                            sim = cosine_similarity(candidate.vector, sel.vector)
                            max_sim = max(max_sim, sim)
                
                # MMR score
                mmr_score = self.mmr_lambda * relevance - (1 - self.mmr_lambda) * max_sim
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            selected.append(remaining.pop(best_idx))
        
        return selected
    
    # Consensus Grouping
    
    def _group_by_consensus(
        self,
        nodes: List[RetrievedNode],
    ) -> List[ConsensusGroup]:
        """
        Group similar claims together to identify consensus.
        
        Uses embedding similarity to cluster nodes with similar labels.
        Returns groups sorted by paper_count (more papers = stronger consensus).
        """
        if not nodes:
            return []
        
        # Group by approximate label similarity using embeddings
        groups: List[ConsensusGroup] = []
        assigned = set()
        
        for i, node in enumerate(nodes):
            if i in assigned:
                continue
            
            # Start new group
            group = ConsensusGroup(
                canonical_label=node.label,
                node_type=node.node_type,
                supporting_nodes=[node],
            )
            assigned.add(i)
            
            # Find similar nodes
            for j, other in enumerate(nodes):
                if j in assigned:
                    continue
                
                # Same type required
                if other.node_type != node.node_type:
                    continue
                
                # Check embedding similarity
                if node.vector is not None and other.vector is not None:
                    sim = cosine_similarity(node.vector, other.vector)
                    if sim >= self.consensus_similarity_threshold:
                        group.supporting_nodes.append(other)
                        assigned.add(j)
            
            # Compute average score
            if group.supporting_nodes:
                group.avg_score = sum(n.final_score for n in group.supporting_nodes) / len(group.supporting_nodes)
            
            groups.append(group)
        
        # Sort by paper count (consensus strength), then by score
        groups.sort(key=lambda g: (g.paper_count, g.avg_score), reverse=True)
        return groups
    
    # Rich Context Formatting
    
    def _build_retrieved_node(
        self,
        session,
        node: KGNode,
        semantic_score: float,
    ) -> RetrievedNode:
        """
        Build a fully-populated RetrievedNode with all context.
        """
        paper = session.get(Paper, node.source_paper_id)
        
        # Get authors
        authors = []
        if paper and paper.authors:
            authors = [a.name for a in paper.authors]
        
        # Compute scores
        influence_score = self._compute_influence_score(paper) if paper else 0.3
        recency_score = self._compute_recency_score(paper) if paper else 0.5
        final_score = self._compute_final_score(
            semantic_score,
            influence_score,
            recency_score,
            node.claim_confidence,
        )
        
        # Get connected nodes (1-hop)
        connected = self._get_connected_nodes(session, node.id)
        
        return RetrievedNode(
            node_id=node.id,
            paper_id=node.source_paper_id,
            node_type=node.type,
            label=node.label,
            confidence=node.claim_confidence,
            claim_type=node.claim_type,
            semantic_score=semantic_score,
            influence_score=influence_score,
            recency_score=recency_score,
            final_score=final_score,
            paper_title=paper.title if paper else "Unknown",
            paper_year=paper.year if paper else None,
            paper_journal=paper.journal if paper else None,
            paper_methodology=paper.methodology if paper else None,
            paper_core_argument=paper.core_argument if paper else None,
            paper_citations_per_year=paper.citations_per_year if paper else None,
            authors=authors,
            connected_nodes=connected,
            vector=node.vector,
        )
    
    # Main Retrieval Method
    
    def retrieve(
        self,
        query: str,
        limit: int = 15,
        node_types: Optional[List[str]] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        use_mmr: bool = True,
        group_consensus: bool = True,
    ) -> Dict[str, Any]:
        """
        Enhanced retrieval with all five improvements.
        
        Args:
            query: Search query
            limit: Max results to return
            node_types: Filter to specific node types (e.g., ['finding', 'limitation'])
            min_year: Filter papers from this year onwards
            max_year: Filter papers up to this year
            use_mmr: Apply MMR diversity reranking
            group_consensus: Group similar claims together
            
        Returns:
            Dict with:
                - 'nodes': List of RetrievedNode objects
                - 'consensus_groups': List of ConsensusGroup objects (if group_consensus=True)
                - 'query': Original query
                - 'total_candidates': Number before filtering/reranking
        """
        if not self.embedding_generator:
            return {'nodes': [], 'consensus_groups': [], 'query': query, 'total_candidates': 0}
        
        # Generate query embedding
        query_vector = self.embedding_generator.generate_query_embedding(query)
        if query_vector is None:
            return {'nodes': [], 'consensus_groups': [], 'query': query, 'total_candidates': 0}
        
        with get_db_session(self.corpus_path) as session:
            # Fetch all nodes with vectors
            nodes_query = session.query(KGNode).filter(KGNode.vector.isnot(None))
            
            # Filter by node type if specified
            if node_types:
                nodes_query = nodes_query.filter(KGNode.type.in_(node_types))
            
            all_nodes = nodes_query.all()
            
            # Score all nodes
            candidates: List[RetrievedNode] = []
            
            for node in all_nodes:
                sim = cosine_similarity(query_vector, node.vector)
                
                if sim < self.search_threshold:
                    continue
                
                # Build rich node
                retrieved = self._build_retrieved_node(session, node, sim)
                
                # Year filter
                if min_year and retrieved.paper_year and retrieved.paper_year < min_year:
                    continue
                if max_year and retrieved.paper_year and retrieved.paper_year > max_year:
                    continue
                
                candidates.append(retrieved)
            
            total_candidates = len(candidates)
            
            # Sort by final score
            candidates.sort(key=lambda x: x.final_score, reverse=True)
            
            # Apply MMR if requested
            if use_mmr and len(candidates) > limit:
                candidates = self._mmr_rerank(candidates, query_vector, limit * 2)
            
            # Truncate to limit (before consensus, we want more for grouping)
            candidates = candidates[:limit * 2]
            
            # Group by consensus if requested
            consensus_groups = []
            if group_consensus:
                consensus_groups = self._group_by_consensus(candidates)
            
            # Final truncation
            candidates = candidates[:limit]
            
            return {
                'nodes': candidates,
                'consensus_groups': consensus_groups,
                'query': query,
                'total_candidates': total_candidates,
            }
    
    # Context Formatting for Agents
    
    def format_context_for_agent(
        self,
        retrieval_result: Dict[str, Any],
        include_edges: bool = True,
        include_methodology: bool = True,
        include_core_argument: bool = True,
        include_consensus: bool = True,
    ) -> str:
        """
        Format retrieval results into rich context string for LLM agents.
        
        This is the key output that makes synthesis answers deeper.
        """
        nodes = retrieval_result.get('nodes', [])
        consensus_groups = retrieval_result.get('consensus_groups', [])
        
        if not nodes:
            return "No relevant evidence found in the corpus."
        
        lines = []
        
        # Section 1: Consensus findings (if any multi-paper agreements)
        if include_consensus and consensus_groups:
            strong_consensus = [g for g in consensus_groups if g.paper_count >= 2]
            if strong_consensus:
                lines.append("=== CONSENSUS FINDINGS ===")
                for group in strong_consensus[:5]:
                    years = group.years_range
                    citations = group.get_citations()
                    lines.append(
                        f"• [{group.node_type.upper()}] \"{group.canonical_label}\"\n"
                        f"  Supported by {group.paper_count} papers ({years[0]}-{years[1]}): "
                        f"{', '.join(citations[:3])}"
                        f"{'...' if len(citations) > 3 else ''}"
                    )
                lines.append("")
        
        # Section 2: Individual evidence items
        lines.append("=== EVIDENCE ===")
        
        for node in nodes:
            citation = node.get_citation_key()
            
            # Header line
            header = f"[{citation}]"
            if node.paper_journal:
                header += f" in {node.paper_journal}"
            if node.paper_citations_per_year:
                header += f" | {node.paper_citations_per_year:.1f} cites/yr"
            
            lines.append(header)
            
            # Methodology context
            if include_methodology and node.paper_methodology:
                method = node.paper_methodology[:100]
                if len(node.paper_methodology) > 100:
                    method += "..."
                lines.append(f"  Method: {method}")
            
            # The claim itself
            conf_str = f" (confidence: {node.confidence:.2f})" if node.confidence else ""
            claim_type = f"[{node.claim_type}] " if node.claim_type else ""
            lines.append(f"  {claim_type}{node.node_type.upper()}{conf_str}: \"{node.label}\"")
            
            # Connected nodes (edge context)
            if include_edges and node.connected_nodes:
                for conn in node.connected_nodes[:3]:
                    direction = "→" if conn['direction'] == 'outgoing' else "←"
                    year_str = f" ({conn['year']})" if conn['year'] else ""
                    lines.append(
                        f"    {direction} {conn['edge_type']}: \"{conn['label'][:60]}...\"{year_str}"
                    )
            
            # Core argument (paper's main thesis)
            if include_core_argument and node.paper_core_argument:
                arg = node.paper_core_argument[:150]
                if len(node.paper_core_argument) > 150:
                    arg += "..."
                lines.append(f"  Core Argument Summary: {arg}")
            
            lines.append("")
        
        return "\n".join(lines)


def format_node_for_legacy_api(node: RetrievedNode) -> Dict[str, Any]:
    """
    Convert RetrievedNode to the dict format expected by existing agents.
    
    This allows gradual migration - old code still works.
    """
    citation = node.get_citation_key()
    
    match_context = f"[{citation}: {node.label}] ({node.node_type})"
    if node.claim_type:
        match_context += f" [Type: {node.claim_type}]"
    if node.confidence is not None:
        match_context += f" [Conf: {node.confidence:.2f}]"
    
    return {
        "id": node.paper_id,
        "title": node.paper_title,
        "year": node.paper_year,
        "citations_per_year": node.paper_citations_per_year,
        "match_type": "semantic",
        "match_score": round(node.final_score, 3),
        "match_context": match_context,
        "methodology": node.paper_methodology,
        "core_argument": node.paper_core_argument,
        "connected_nodes": node.connected_nodes,
        "node_type": node.node_type,
        "confidence": node.confidence,
    }