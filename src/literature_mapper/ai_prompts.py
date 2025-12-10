"""
AI prompts for academic paper analysis.
"""


def get_analysis_prompt() -> str:
    """
    Get the standard analysis prompt for academic papers.
    
    Uses structured extraction with explicit field definitions and 
    fallback values. Designed for reliable JSON output.
    
    Returns:
        Formatted prompt string ready for use with .format(text=paper_text)
        
    Example:
        >>> prompt = get_analysis_prompt()
        >>> full_prompt = prompt.format(text=paper_text)
    """
    return """Analyze this academic paper and extract structured metadata.

OUTPUT FORMAT: Valid JSON only. No markdown, no commentary.

SCHEMA:
{{
    "title": "<exact paper title including subtitle>",
    "authors": ["<author 1>", "<author 2>"],
    "year": <4-digit integer>,
    "journal": "<venue name>" | null,
    "abstract_short": "<25-word summary of the study>",
    "core_argument": "<single sentence: This paper argues/shows/demonstrates that...>",
    "methodology": "<research approach, e.g., 'Survey (n=450)' or 'Case study'>",
    "theoretical_framework": "<named theory or framework>" | "Not specified",
    "key_concepts": ["<term1>", "<term2>", "<term3>"],
    "contribution_to_field": "<single sentence: what this adds to literature>",
    "doi": "<DOI string>" | null,
    "citation_count": null
}}

EXTRACTION RULES:
1. title: Copy verbatim from paper header. Include subtitles after colon.
2. authors: List each author as a separate string, preserving order.
3. year: Publication year only (not submission/revision dates).
4. journal: Conference name, journal name, or working paper series. Use null if unclear.
5. abstract_short: Compress abstract to ~25 words. Focus on: what was done, main finding.
6. core_argument: Begin with "This paper argues/shows/demonstrates that..." 
7. methodology: Be specific about method type and sample if quantitative.
8. theoretical_framework: Name specific frameworks (e.g., "Resource Dependency Theory"). 
   Use "Not specified" if paper is atheoretical.
9. key_concepts: Extract 3-6 domain-specific terms central to the argument.
10. contribution_to_field: State the novel contribution in one sentence.
11. doi: Extract if present in header/footer. Format: 10.XXXX/...
12. citation_count: Always null (not extractable from text).

UNCERTAINTY HANDLING:
- If text is garbled or unreadable: {{"title": "Document analysis failed", "authors": ["Unknown"], "year": null, ...}}
- If document is not academic: {{"title": "Non-academic document", "authors": ["Unknown"], "year": null, ...}}
- For ambiguous fields: prefer "Not specified" over guessing.

Paper text:
{text}

JSON:"""


def get_retry_prompt() -> str:
    """Simplified prompt for retry attempts when main analysis fails."""
    return """Extract basic paper metadata. Prioritize reliability over completeness.

OUTPUT: Valid JSON only.

{{
    "title": "<paper title>" | "Title not found",
    "authors": ["<name>"] | ["Unknown"],
    "year": <integer> | null,
    "journal": "<venue>" | null,
    "abstract_short": "<~25 word summary>",
    "core_argument": "<main finding in one sentence>",
    "methodology": "<method>" | "Not specified",
    "theoretical_framework": "Not specified",
    "key_concepts": ["<term1>", "<term2>"],
    "contribution_to_field": "Not specified",
    "doi": null,
    "citation_count": null
}}

If uncertain about a field, use the fallback value shown above.

Paper text:
{text}

JSON:"""


def get_json_repair_prompt(malformed_response: str) -> str:
    """
    Prompt for fixing malformed JSON responses from analysis.
    
    Note: This is distinct from get_hypothesis_validation_prompt which validates
    research hypotheses against evidence.
    """
    return f"""Repair this malformed JSON response. Output valid JSON only.

TARGET SCHEMA:
- title (string), authors (array of strings), year (integer or null)
- journal (string or null), abstract_short (string), core_argument (string)
- methodology (string), theoretical_framework (string), key_concepts (array)
- contribution_to_field (string), doi (string or null), citation_count (null)

MALFORMED INPUT:
{malformed_response}

CORRECTED JSON:"""


def get_kg_prompt(paper_title: str | None = None, text: str = "") -> str:
    """
    Get the prompt for Knowledge Graph extraction.
    
    Extracts a structured graph of concepts, findings, methods, and their
    relationships from an academic paper.
    
    Args:
        paper_title: Optional title to include in the prompt context
        text: The full text of the paper to analyze
        
    Returns:
        Formatted prompt string
    """
    title_context = f'Paper: "{paper_title}"' if paper_title else "Paper text follows."
    
    return f"""Extract a knowledge graph from this academic paper.

{title_context}

OUTPUT: Valid JSON only. No markdown code blocks.

SCHEMA:
{{
    "nodes": [
        {{
            "id": "<unique_string_id>",
            "type": "<node_type>",
            "label": "<descriptive label>",
            "confidence": <0.0-1.0>,
            "subtype": "<optional_subtype>"
        }}
    ],
    "edges": [
        {{
            "source": "<source_node_id>",
            "target": "<target_node_id>",
            "type": "<RELATIONSHIP_TYPE>"
        }}
    ]
}}

NODE TYPES (use exactly these):
- "paper": The paper itself (required, id="paper_main")
- "author": Paper authors
- "concept": Key theoretical concepts
- "method": Research methods or techniques
- "finding": Empirical results or conclusions (subtype: "finding", confidence: 0.8-1.0)
- "hypothesis": Proposed but untested claims (subtype: "hypothesis", confidence: 0.5-0.8)
- "limitation": Acknowledged weaknesses or gaps (subtype: "limitation")
- "institution": Organizations or affiliations
- "source": Publication venue (journal, conference, arxiv, etc.)

EDGE TYPES (use UPPERCASE):
- AUTHORED_BY, AFFILIATED_WITH, PUBLISHED_IN
- PROPOSES, USES, EVALUATES, CITES
- SUPPORTS, CONTRADICTS, EXTENDS
- HAS_LIMITATION, ADDRESSES_CHALLENGE

EXTRACTION GUIDELINES:
1. Create exactly one "paper" node with id="paper_main"
2. Extract 5-15 concept nodes for key theoretical terms
3. Create finding nodes for each major result (set confidence based on strength of evidence)
4. Explicitly extract limitations even if paper minimizes them
5. Connect all nodes to at least one other node
6. Limit output to 30 nodes and 50 edges maximum

CONFIDENCE SCORING:
- 1.0: Directly stated facts (author names, publication venue)
- 0.8-0.95: Well-supported findings with clear evidence
- 0.5-0.8: Proposed hypotheses or preliminary findings
- <0.5: Speculative claims

Paper text:
{text}

JSON:"""


def get_synthesis_prompt(query: str, context_nodes: str) -> str:
    """
    Generate prompt for Argument Agent synthesis.
    
    Synthesizes an answer to a research question using retrieved
    knowledge graph nodes as evidence.
    """
    return f"""Answer this research question using the provided evidence.

QUESTION: "{query}"

EVIDENCE (from literature corpus):
{context_nodes}

Each evidence item includes:
- Year: Publication date
- Influence: Citations per year (higher = more recognized)
- Relevance: Semantic match score to your question

RESPONSE REQUIREMENTS:
1. Address the question directly in 1-2 paragraphs
2. Cite sources using format: [Author et al., Year]
3. Group related findings thematically
4. Distinguish consensus findings (multiple sources) from single-source claims
5. Note temporal patterns: foundational work vs. recent developments
6. If evidence is insufficient, state what IS known and what gaps remain

CONFIDENCE CALIBRATION:
- High-influence older papers: weight for established consensus
- Recent papers: weight for current state-of-the-art
- Multiple agreeing sources: higher confidence
- Single source or conflicting evidence: note uncertainty

Respond with substantive analysis. If the evidence doesn't support an answer, 
say so rather than speculating beyond the provided context."""


def get_hypothesis_validation_prompt(hypothesis: str, context_nodes: str) -> str:
    """
    Generate prompt for Validation Agent critique.
    
    Evaluates a hypothesis against corpus evidence and returns
    a structured verdict.
    """
    return f"""Evaluate this hypothesis against the provided evidence.

HYPOTHESIS: "{hypothesis}"

EVIDENCE FROM CORPUS:
{context_nodes}

Each evidence item includes:
- Year: Publication date
- Influence: Citations per year
- Relevance: Semantic similarity to hypothesis

EVALUATION TASK:
Determine if the evidence SUPPORTS, CONTRADICTS, or is insufficient (NOVEL) for the hypothesis.

VERDICT CRITERIA:
- SUPPORTED: Multiple sources provide direct or strong indirect support
- CONTRADICTED: Evidence presents findings incompatible with the hypothesis
- NOVEL: Insufficient evidence to evaluate (hypothesis may be genuinely new)

ANALYSIS REQUIREMENTS:
1. Consider conceptual relationships, not just keyword matches
   (e.g., "context window limitations" relates to "hallucination causes")
2. Weight foundational papers more heavily for established claims
3. Weight recent papers for current state-of-the-art assessments
4. Cite specific evidence for your verdict

OUTPUT FORMAT (JSON only):
{{
    "verdict": "SUPPORTED" | "CONTRADICTED" | "NOVEL",
    "explanation": "<detailed reasoning with specific evidence citations>",
    "citations": [
        "Author, Year: <specific claim from that paper>",
        ...
    ],
    "confidence": <0.0-1.0 based on evidence strength>
}}

Evaluate systematically. A NOVEL verdict is valid when evidence is genuinely insufficient."""


def get_conceptual_ghost_prompt(papers_context: str) -> str:
    """
    Generate prompt for identifying Conceptual Ghosts (missing concepts).
    
    Analyzes a corpus to find important concepts that are conspicuously
    absent given the topics discussed.
    """
    return f"""Identify important concepts MISSING from this research corpus.

CORPUS SUMMARY:
{papers_context}

TASK: Find "Conceptual Ghosts" - ideas that SHOULD appear given the topics discussed 
but are absent. These represent potential blind spots or unexplored connections.

ANALYSIS APPROACH:
1. Identify the theoretical traditions represented
2. Note methodological approaches present
3. Find gaps: 
   - Missing counter-arguments to dominant claims
   - Absent theoretical bridges between subfields
   - Overlooked methodological alternatives
   - Unstated assumptions that deserve scrutiny

OUTPUT FORMAT (JSON array):
[
    {{
        "concept_name": "<name of missing concept>",
        "description": "<what this concept is and why it matters>",
        "reasoning": "<why is this absent? Reference specific papers by citation key>",
        "relevance_score": <0.0-1.0, higher = more critical gap>
    }}
]

QUALITY CRITERIA:
- Focus on substantive theoretical/methodological gaps, not missing keywords
- Reference specific papers when explaining why the absence matters
- Higher relevance_score for gaps that would significantly change conclusions
- Identify 3-7 meaningful gaps, not exhaustive lists"""


def get_genealogy_prompt(paper_title: str, paper_abstract: str, corpus_papers: str) -> str:
    """
    Generate prompt for identifying intellectual relationships to other papers.
    
    Traces how a paper relates to other works in the corpus through
    extension, challenge, or synthesis relationships.
    
    Args:
        paper_title: Title of the paper being analyzed
        paper_abstract: Abstract/core argument of the paper
        corpus_papers: Formatted list of other papers in the corpus
        
    Returns:
        Prompt string for genealogy extraction
    """
    return f"""Identify intellectual relationships between this paper and others in the corpus.

ANALYZED PAPER:
Title: "{paper_title}"
Core Argument: {paper_abstract}

OTHER PAPERS IN CORPUS:
{corpus_papers}

RELATIONSHIP TYPES:
- EXTENDS: Directly builds on another paper's theory or method
- CHALLENGES: Critiques, contradicts, or refutes findings
- SYNTHESIZES: Integrates ideas from multiple papers into new framework  
- BUILDS_ON: General foundational dependency (weaker than EXTENDS)

IDENTIFICATION CRITERIA:
1. Only identify relationships with clear intellectual connection
2. Provide textual evidence from the analyzed paper's argument
3. Assign confidence (0.0-1.0) based on how explicit the relationship is
4. A paper may have multiple relationships to different works
5. Return empty array if no clear relationships exist

OUTPUT FORMAT (JSON only):
{{
    "relationships": [
        {{
            "target_id": <integer paper ID>,
            "target_title": "<title of related paper>",
            "type": "EXTENDS" | "CHALLENGES" | "SYNTHESIZES" | "BUILDS_ON",
            "confidence": <0.0-1.0>,
            "evidence": "<quote or description showing the relationship>"
        }}
    ]
}}

Be conservative: only identify relationships you can justify with evidence."""


# Legacy alias for backward compatibility
def get_validation_prompt(hypothesis: str, context_nodes: str) -> str:
    """
    Alias for get_hypothesis_validation_prompt.
    
    Deprecated: Use get_hypothesis_validation_prompt directly.
    """
    return get_hypothesis_validation_prompt(hypothesis, context_nodes)


# Export main functions
__all__ = [
    'get_analysis_prompt',
    'get_retry_prompt',
    'get_json_repair_prompt',
    'get_kg_prompt',
    'get_synthesis_prompt',
    'get_hypothesis_validation_prompt',
    'get_validation_prompt',  # backward compatibility
    'get_conceptual_ghost_prompt',
    'get_genealogy_prompt',
]