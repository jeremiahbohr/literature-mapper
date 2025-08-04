"""
AI prompts for academic paper analysis with model-aware optimization.
"""

from typing import Optional

def get_analysis_prompt(model_type: str = "unknown") -> str:
    """
    Get analysis prompt optimized for specific model types.
    
    Args:
        model_type: Type of model ("flash", "pro", "ultra", or "unknown")
        
    Returns:
        Formatted prompt string ready for use
        
    Example:
        >>> prompt = get_analysis_prompt("flash")
        >>> full_prompt = prompt.format(text=paper_text)
    """
    # Base prompt structure with escaped braces
    base_instructions = """You are an expert academic researcher analyzing a scholarly paper. Extract key information and return ONLY valid JSON.

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
    "abstract_short": "exactly 25 words summarizing the study",
    "core_argument": "one clear sentence stating the main thesis",
    "methodology": "brief method description",
    "theoretical_framework": "primary theoretical approach or 'Not specified'",
    "key_concepts": ["array", "of", "key", "terms"],
    "contribution_to_field": "one sentence describing what this adds",
    "doi": "DOI string or null if not found",
    "citation_count": null
}}"""
    
    # Model-specific optimizations
    if model_type == "flash":
        specific_instructions = """

SPEED-OPTIMIZED: Focus on accuracy over completeness. Extract 3-5 key concepts maximum.
- abstract_short: Exactly 25 words capturing purpose, method, and key finding
- core_argument: One sentence starting with "This paper argues/shows/demonstrates that..."
- methodology: Brief method (e.g., "Survey", "Experiment", "Literature review")"""
        
    elif model_type == "pro":
        specific_instructions = """

BALANCED ANALYSIS: Provide thorough but efficient analysis. Extract 4-7 key concepts.
- abstract_short: Expert synthesis in exactly 25 words
- core_argument: Comprehensive thesis statement (30-50 words)
- methodology: Detailed approach with key parameters
- theoretical_framework: Identify specific theories or frameworks used"""
        
    elif model_type == "ultra":
        specific_instructions = """

COMPREHENSIVE ANALYSIS: Apply deep domain knowledge. Extract 5-8 key concepts.
- abstract_short: Sophisticated synthesis in exactly 25 words
- core_argument: Nuanced articulation of main thesis (40-60 words)
- methodology: Comprehensive description including design and analysis
- theoretical_framework: Precise theoretical positioning"""
        
    else:
        specific_instructions = """

STANDARD EXTRACTION: Extract 3-6 key concepts. Balance completeness with efficiency.
- abstract_short: Clear summary in exactly 25 words
- core_argument: Main thesis in one clear sentence
- methodology: Research method with basic details"""
    
    # Essential field guidelines
    field_guidelines = """

FIELD GUIDELINES:

title: Extract exact title including subtitles
authors: List all authors as they appear, one string per author
year: 4-digit publication year (distinguish from submission/acceptance dates)
journal: Full journal name, conference name, or publication venue
methodology: Research approach (e.g., "Quantitative survey (n=450)", "Qualitative interviews")
key_concepts: Most important technical terms and concepts from the paper

ERROR HANDLING:
- If text is garbled: return "title": "Document analysis failed"
- If non-academic: return "title": "Non-academic document detected"

Remember: Return ONLY the JSON object.

Paper text:
{text}

JSON:"""
    
    return base_instructions + specific_instructions + field_guidelines


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
- abstract_short (string, exactly 25 words), core_argument (string)
- methodology (string), theoretical_framework (string), key_concepts (array)
- contribution_to_field (string), doi (string or null), citation_count (null)

Original response to fix:
{malformed_response}

Return ONLY the corrected JSON:"""


def create_domain_prompt(domain: str, base_prompt: str) -> str:
    """Add domain-specific instructions to base prompt."""
    domain_enhancements = {
        "computer_science": "\nCS FOCUS: Pay attention to algorithms, datasets, performance metrics, programming languages.",
        "psychology": "\nPSYCH FOCUS: Emphasize psychological constructs, measurement scales, participant demographics.",
        "medicine": "\nMED FOCUS: Note clinical populations, interventions, outcome measures, study design.",
        "education": "\nEDU FOCUS: Identify educational settings, age groups, learning outcomes, pedagogical approaches.",
        "business": "\nBIZ FOCUS: Look for business metrics, organizational contexts, market analysis."
    }
    
    enhancement = domain_enhancements.get(domain, "")
    return base_prompt + enhancement


def detect_model_type(model_name: str) -> str:
    """Detect model type from model name for prompt optimization."""
    name_lower = model_name.lower()
    
    if "flash" in name_lower:
        return "flash"
    elif "pro" in name_lower:
        return "pro"
    elif "ultra" in name_lower:
        return "ultra"
    else:
        return "unknown"


def get_optimized_prompt(model_name: str, text: str, domain: Optional[str] = None) -> str:
    """
    Get fully optimized prompt for specific model and domain.
    
    Example:
        >>> prompt = get_optimized_prompt("gemini-2.5-flash", paper_text)
        >>> response = model.generate_content(prompt)
    """
    model_type = detect_model_type(model_name)
    base_prompt = get_analysis_prompt(model_type)
    
    if domain:
        base_prompt = create_domain_prompt(domain, base_prompt)
    
    return base_prompt.format(text=text)


# Export main functions
__all__ = [
    'get_analysis_prompt',
    'get_retry_prompt', 
    'get_validation_prompt',
    'create_domain_prompt',
    'detect_model_type',
    'get_optimized_prompt'
]