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


def get_kg_prompt(paper_title: str | None = None, text: str = "") -> str:
    """
    Get the prompt for Knowledge Graph extraction.
    
    Args:
        paper_title: Optional title to include in the prompt context
        text: The full text of the paper to analyze
        
    Returns:
        Formatted prompt string
    """
    title_line = f'The paper is titled "{paper_title}".' if paper_title else "The paper is described below."
    
    return f"""You are extracting a structured knowledge graph from an academic paper. {title_line}
Return ONLY valid JSON.

Instructions:
1. Extract the core argument, key findings, and methodology.
2. Identify the most important entities (concepts, authors, methods).
3. CRITICAL: Explicitly extract "challenges", "limitations", and "problem statements" as nodes, even if the paper is solution-oriented.
4. Define relationships between them.
5. CRITICAL: Limit your output to the top 30 most important nodes and 50 edges to ensure the JSON is complete and valid. Do not try to map everything.
6. Ensure the output is valid JSON.

JSON Structure:
```json
{{
    "nodes": [
        {{"id": "unique_id", "type": "concept|author|paper|method|finding", "label": "Name/Description"}}
    ],
    "edges": [
        {{"source": "source_id", "target": "target_id", "type": "RELATION_TYPE"}}
    ]
}}
```

Use only these node types:
- "paper": The paper itself (always required)
- "author": Authors of the paper
- "institution": Affiliations or organizations (e.g., "DeepMind", "MIT")
- "finding": Key results or claims (e.g., "Attention improves translation")
- "limitation": Weaknesses or gaps (e.g., "Quadratic complexity")
- "hypothesis": Proposed theory or question being tested
- "source": Publication venue. Use journal name, "book", "arxiv", "SSRN", or "Unknown".

For "metric" and "institution", if the specific name is not clear, use "Unknown" as the label.

Edges must have:
- "source": id of the source node
- "target": id of the target node
- "type": relationship type (e.g., "uses", "proposes", "evaluates_on", "authored_by")

Paper text:
{{text}}

JSON:"""

def get_synthesis_prompt(query: str, context_nodes: str) -> str:
    """
    Generate prompt for Argument Agent synthesis.
    """
    return f"""
    You are an expert research assistant. Your task is to answer the following research question based ONLY on the provided context from a literature knowledge graph.

    Research Question: "{query}"

    Context (Graph Nodes):
    {context_nodes}

    Instructions:
    1. Synthesize an answer that directly addresses the question.
    2. Group related findings or concepts together.
    3. Cite your sources using the format [Paper ID: Label].
    4. If the context does not contain enough information to answer fully, state what is known and what is missing.
    5. Do not hallucinate information not present in the context.

    Response Format:
    - A concise, well-structured paragraph or two.
    - Use bullet points if listing multiple distinct factors.
    """

def get_validation_prompt(hypothesis: str, context_nodes: str) -> str:
    """
    Generate prompt for Validation Agent critique.
    """
    return f"""You are a scientific validation agent. Your goal is to critique a user hypothesis based ONLY on the provided evidence from a literature corpus.

Hypothesis: "{hypothesis}"

Evidence from Corpus:
{context_nodes}

Instructions:
1. Analyze the evidence to determine if it supports, contradicts, or is neutral towards the hypothesis.
2. CRITICAL: You must consider related concepts and underlying causes. For example, if the hypothesis is about "hallucination" and the evidence discusses "context loss", "inconsistency", or "memory failure", these are relevant contradictions/support. Do not look for exact keyword matches only.
3. If the evidence explicitly supports the hypothesis, Verdict is SUPPORTED.
4. If the evidence explicitly contradicts the hypothesis (e.g., shows the problem is unsolved, or has different causes), Verdict is CONTRADICTED.
5. If the evidence is unrelated or insufficient, Verdict is NOVEL.
6. Cite specific papers (Author, Year) to back up your verdict.

Return valid JSON:
```json
{{
    "verdict": "SUPPORTED|CONTRADICTED|NOVEL",
    "explanation": "Detailed explanation citing specific evidence...",
    "citations": ["Author, Year: Claim", ...]
}}
```
"""


# Export main functions
__all__ = [
    'get_analysis_prompt',
    'get_kg_prompt',
    'get_synthesis_prompt',
    'get_validation_prompt'
]