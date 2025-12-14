"""
Thematic Agents for high-level reasoning over the Knowledge Graph.
"""

import logging
import json
import google.generativeai as genai
from typing import List, Dict, Any, Optional
from .ai_prompts import get_synthesis_prompt, get_hypothesis_validation_prompt
from .exceptions import APIError

logger = logging.getLogger(__name__)


class BaseAgent:
    """Shared logic for agents using Gemini."""
    
    def __init__(self, api_key: str, model_name: str):
        if not api_key:
            # Allow initialization without key, but methods will fail
            self.model = None
            return
            
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def _generate(self, prompt: str) -> str:
        """Generate content from LLM."""
        if not self.model:
            raise APIError("Agent not initialized with API key")
            
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error("Agent generation failed: %s", e)
            raise APIError(f"Agent generation failed: {e}")
    
    def _format_context_node_legacy(self, node: Dict) -> str:
        """
        Legacy format for backward compatibility.
        
        Used when context_nodes are in the old dict format.
        """
        year = node.get('year', '?')
        citations_per_year = node.get('citations_per_year')
        
        # Format influence indicator
        if citations_per_year is not None:
            influence = f"{citations_per_year:.1f}/yr"
        else:
            influence = "N/A"
        
        return (
            f"- [{node['match_context']}] "
            f"(Year: {year}, Influence: {influence}, Relevance: {node['match_score']:.2f})"
        )
    
    def _format_context_node_rich(self, node: Dict) -> str:
        """
        Rich format using enhanced retrieval data.
        
        Includes methodology, core argument, and connected nodes.
        """
        year = node.get('year', '?')
        citations_per_year = node.get('citations_per_year')
        
        # Build header
        influence = f"{citations_per_year:.1f}/yr" if citations_per_year else "N/A"
        
        lines = []
        
        # Main claim line
        confidence = node.get('confidence')
        conf_str = f" [conf: {confidence:.2f}]" if confidence else ""
        node_type = node.get('node_type', 'unknown').upper()
        
        lines.append(
            f"[{node.get('match_context', 'Unknown')}]"
            f" (Year: {year}, Influence: {influence}, Score: {node['match_score']:.2f})"
        )
        
        # Methodology (if present)
        if node.get('methodology'):
            method = node['methodology'][:80]
            if len(node.get('methodology', '')) > 80:
                method += "..."
            lines.append(f"    Method: {method}")
        
        # Core argument (if present)
        if node.get('core_argument'):
            arg = node['core_argument'][:120]
            if len(node.get('core_argument', '')) > 120:
                arg += "..."
            lines.append(f"    Core Argument Summary: {arg}")
        
        # Connected nodes (edge context)
        connected = node.get('connected_nodes', [])
        if connected:
            for conn in connected[:2]:  # Limit to 2 for brevity
                direction = "→" if conn.get('direction') == 'outgoing' else "←"
                edge_type = conn.get('edge_type', 'RELATED')
                label = conn.get('label', '')[:50]
                lines.append(f"    {direction} {edge_type}: \"{label}...\"")
        
        return "\n".join(lines)
    
    def _format_context_node(self, node: Dict) -> str:
        """
        Smart format selector - uses rich format if data available.
        """
        # Check if this is enhanced data (has methodology, core_argument, or connected_nodes)
        has_rich_data = any([
            node.get('methodology'),
            node.get('core_argument'),
            node.get('connected_nodes'),
        ])
        
        if has_rich_data:
            return self._format_context_node_rich(node)
        else:
            return self._format_context_node_legacy(node)
    
    def _format_consensus_section(self, consensus_groups: List[Dict]) -> str:
        """
        Format consensus groups into a summary section.
        
        Highlights claims supported by multiple papers.
        """
        if not consensus_groups:
            return ""
        
        # Filter to groups with 2+ papers
        strong = [g for g in consensus_groups if g.get('paper_count', 1) >= 2]
        if not strong:
            return ""
        
        lines = ["=== CONSENSUS (Multiple Papers Agree) ==="]
        
        for group in strong[:5]:
            papers = group.get('paper_count', 1)
            years = group.get('years_range', (0, 0))
            citations = group.get('citations', [])
            label = group.get('canonical_label', 'Unknown claim')
            node_type = group.get('node_type', 'finding').upper()
            
            cite_str = ", ".join(citations[:3])
            if len(citations) > 3:
                cite_str += "..."
            
            lines.append(
                f"• [{node_type}] \"{label}\"\n"
                f"  Supported by {papers} papers ({years[0]}-{years[1]}): {cite_str}"
            )
        
        lines.append("")
        return "\n".join(lines)


class ArgumentAgent(BaseAgent):
    """Synthesizes answers to research questions."""
    
    def synthesize(
        self, 
        query: str, 
        context_nodes: List[Dict],
        consensus_groups: Optional[List[Dict]] = None,
        pre_formatted_context: Optional[str] = None,
    ) -> str:
        """
        Synthesize an answer using the provided context nodes.
        
        Args:
            query: The research question
            context_nodes: List of node dicts (from search_corpus or EnhancedRetriever)
            consensus_groups: Optional list of consensus group dicts
            pre_formatted_context: If provided, use this instead of formatting nodes
            
        Returns:
            Synthesized text response
        """
        if not context_nodes and not pre_formatted_context:
            return "No relevant information found in the corpus to answer this question."
        
        # Use pre-formatted context if provided (from EnhancedRetriever)
        if pre_formatted_context:
            context_str = pre_formatted_context
        else:
            # Format context with temporal metadata
            parts = []
            
            # Add consensus section if available
            if consensus_groups:
                consensus_str = self._format_consensus_section(consensus_groups)
                if consensus_str:
                    parts.append(consensus_str)
            
            # Add individual evidence
            parts.append("=== INDIVIDUAL EVIDENCE ===")
            for node in context_nodes:
                parts.append(self._format_context_node(node))
            
            context_str = "\n".join(parts)
        
        prompt = get_synthesis_prompt(query, context_str)
        return self._generate(prompt)


class ValidationAgent(BaseAgent):
    """Critiques hypotheses against the evidence."""
    
    def validate_hypothesis(
        self, 
        hypothesis: str, 
        context_nodes: List[Dict],
        consensus_groups: Optional[List[Dict]] = None,
        pre_formatted_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Validate a hypothesis using the provided context nodes.
        
        Args:
            hypothesis: The user's claim
            context_nodes: List of node dicts
            consensus_groups: Optional list of consensus group dicts
            pre_formatted_context: If provided, use this instead of formatting nodes
            
        Returns:
            Dict with verdict, explanation, and citations
        """
        if not context_nodes and not pre_formatted_context:
            return {
                "verdict": "NOVEL",
                "explanation": "No direct evidence found in the current corpus to support or contradict this hypothesis.",
                "citations": []
            }
        
        # Use pre-formatted context if provided
        if pre_formatted_context:
            context_str = pre_formatted_context
        else:
            # Format context
            parts = []
            
            # Add consensus section if available
            if consensus_groups:
                consensus_str = self._format_consensus_section(consensus_groups)
                if consensus_str:
                    parts.append(consensus_str)
            
            # Add individual evidence
            parts.append("=== INDIVIDUAL EVIDENCE ===")
            for node in context_nodes:
                parts.append(self._format_context_node(node))
            
            context_str = "\n".join(parts)
        
        prompt = get_hypothesis_validation_prompt(hypothesis, context_str)
        response_text = self._generate(prompt)
        
        # Parse JSON response
        try:
            # Clean markdown code blocks
            cleaned_text = response_text.replace('```json', '').replace('```', '').strip()
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            logger.error("Failed to parse validation response: %s", response_text)
            return {
                "verdict": "ERROR",
                "explanation": "Failed to parse agent response.",
                "citations": []
            }