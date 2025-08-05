"""
AI prompts for academic paper analysis.
Simplified, single-prompt approach that works reliably across all models.
"""

def get_analysis_prompt() -> str:
    """
    Get the standard analysis prompt for academic papers.
    
    Returns:
        Formatted prompt string ready for use with .format(text=paper_text)
        
    Example:
        >>> prompt = get_analysis_prompt()
        >>> full_prompt = prompt.format(text=paper_text)
    """
    return """You are an expert academic researcher analyzing a scholarly paper. Extract key information and return ONLY valid JSON.

CRITICAL REQUIREMENTS:
1. Return ONLY valid JSON - no markdown code blocks, no explanations
2. Use exactly the field names specified below
3. If information is not available, use the specified fallback values

Required JSON structure:
{{
    "title": "string - full paper title",
    "authors": ["array", "of", "author", "names"],
    "year": integer_publication_year,
    "journal": "string or null if not found",
    "abstract_short": "about 25 words summarizing the study",
    "core_argument": "one clear sentence stating the main thesis",
    "methodology": "brief method description",
    "theoretical_framework": "primary theoretical approach or 'Not specified'",
    "key_concepts": ["array", "of", "key", "terms"],
    "contribution_to_field": "one sentence describing what this adds",
    "doi": "DOI string or null if not found",
    "citation_count": null
}}

FIELD GUIDELINES:

title: Extract exact title including subtitles
authors: List all authors as they appear, one string per author
year: 4-digit publication year (distinguish from submission/acceptance dates)
journal: Full journal name, conference name, or publication venue
abstract_short: Clear summary in about 25 words
core_argument: Main thesis in one clear sentence starting with "This paper argues/shows/demonstrates that..."
methodology: Research approach (e.g., "Quantitative survey (n=450)", "Qualitative interviews", "Literature review")
theoretical_framework: Identify specific theories or frameworks used, or "Not specified"
key_concepts: 3-6 most important technical terms and concepts from the paper
contribution_to_field: What this paper contributes in one sentence
doi: DOI if present, null otherwise
citation_count: Always null (not available in paper text)

ERROR HANDLING:
- If text is garbled: return "title": "Document analysis failed"
- If non-academic: return "title": "Non-academic document detected"
- If field cannot be determined: use "Not specified" for text fields, null for optional fields

Remember: Return ONLY the JSON object, nothing else.

Paper text:
{text}

JSON:"""


def get_retry_prompt() -> str:
    """Simplified prompt for retry attempts when main analysis fails."""
    return """Extract basic information from this academic paper and return as JSON.

Focus on reliability. If unsure about any field, use the fallback value.

Required JSON format:
{{
    "title": "paper title or 'Title not found'",
    "authors": ["author names or 'Unknown Author'"],
    "year": publication_year_integer_or_null,
    "journal": "journal name or null",
    "abstract_short": "25 word summary",
    "core_argument": "main finding in one sentence",
    "methodology": "research method or 'Not specified'",
    "theoretical_framework": "theoretical approach or 'Not specified'",
    "key_concepts": ["key", "terms", "from", "paper"],
    "contribution_to_field": "what this paper contributes or 'Not specified'",
    "doi": null,
    "citation_count": null
}}

Paper text:
{text}

Return only JSON:"""


def get_validation_prompt(malformed_response: str) -> str:
    """Prompt for fixing malformed JSON responses."""
    return f"""Fix this malformed JSON response from academic paper analysis.

The response should be valid JSON with these exact fields:
- title (string), authors (array), year (integer), journal (string or null)
- abstract_short (string, about 25 words), core_argument (string)
- methodology (string), theoretical_framework (string), key_concepts (array)
- contribution_to_field (string), doi (string or null), citation_count (null)

Original response to fix:
{malformed_response}

Return ONLY the corrected JSON:"""


# Export main functions
__all__ = [
    'get_analysis_prompt',
    'get_retry_prompt', 
    'get_validation_prompt'
]