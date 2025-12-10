"""
Thematic Agents for high-level reasoning over the Knowledge Graph.
"""

import logging
import json
import google.generativeai as genai
from typing import List, Dict, Any, Optional
from .ai_prompts import get_synthesis_prompt, get_validation_prompt
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
    
    def _format_context_node(self, node: Dict) -> str:
        """
        Format a context node with temporal metadata for LLM reasoning.
        
        Includes year and influence (citations_per_year) so the agent can
        intelligently weigh recent work vs foundational classics.
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

class ArgumentAgent(BaseAgent):
    """Synthesizes answers to research questions."""
    
    def synthesize(self, query: str, context_nodes: List[Dict]) -> str:
        """
        Synthesize an answer using the provided context nodes.
        
        Args:
            query: The research question
            context_nodes: List of node dicts (from search_corpus)
            
        Returns:
            Synthesized text response
        """
        if not context_nodes:
            return "No relevant information found in the corpus to answer this question."
            
        # Format context with temporal metadata
        context_str = "\n".join([
            self._format_context_node(node)
            for node in context_nodes
        ])
        
        prompt = get_synthesis_prompt(query, context_str)
        return self._generate(prompt)

class ValidationAgent(BaseAgent):
    """Critiques hypotheses against the evidence."""
    
    def validate_hypothesis(self, hypothesis: str, context_nodes: List[Dict]) -> Dict[str, Any]:
        """
        Validate a hypothesis using the provided context nodes.
        
        Args:
            hypothesis: The user's claim
            context_nodes: List of node dicts
            
        Returns:
            Dict with verdict, explanation, and citations
        """
        if not context_nodes:
            return {
                "verdict": "NOVEL",
                "explanation": "No direct evidence found in the current corpus to support or contradict this hypothesis.",
                "citations": []
            }
            
        # Format context with temporal metadata
        context_str = "\n".join([
            self._format_context_node(node)
            for node in context_nodes
        ])
        
        prompt = get_validation_prompt(hypothesis, context_str)
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
