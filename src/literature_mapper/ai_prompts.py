"""
AI prompts for academic paper analysis.

Includes:
- Paper metadata extraction (get_analysis_prompt, get_retry_prompt)
- Knowledge Graph extraction (get_kg_prompt, get_kg_nodes_prompt, get_kg_edges_prompt)
- Agent prompts (get_synthesis_prompt, get_hypothesis_validation_prompt)
- Ghost detection (get_conceptual_ghost_prompt)
- Genealogy extraction (get_genealogy_prompt)
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


# Knowledge Graph Extraction

def get_kg_prompt(paper_title: str | None = None, text: str = "") -> str:
    """
    Get the prompt for Knowledge Graph extraction (single-pass, legacy).
    
    For better results with large papers, use the two-pass approach:
    get_kg_nodes_prompt() followed by get_kg_edges_prompt().
    
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


def get_kg_nodes_prompt(paper_title: str | None = None, text: str = "") -> str:
    """
    Get the prompt for Knowledge Graph node extraction (pass 1 of 2).
    
    This is the first pass of the two-pass KG extraction approach.
    Extracts nodes only, which are then passed to get_kg_edges_prompt().
    
    Args:
        paper_title: Optional title to include in the prompt context
        text: The full text of the paper to analyze
        
    Returns:
        Formatted prompt string
    """
    title_context = f'Paper: "{paper_title}"' if paper_title else "Paper text follows."
    
    return f"""Extract knowledge graph NODES from this academic paper.

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
    ]
}}

NODE TYPES (use exactly these):
- "paper": The paper itself (REQUIRED, id="paper_main")
- "author": Paper authors (extract all authors)
- "concept": Key theoretical concepts (5-15 nodes)
- "method": Research methods or techniques used
- "finding": Empirical results or conclusions (subtype: "finding", confidence: 0.8-1.0)
- "hypothesis": Proposed but untested claims (subtype: "hypothesis", confidence: 0.5-0.8)
- "limitation": Acknowledged weaknesses or gaps (subtype: "limitation")
- "institution": Organizations or affiliations mentioned
- "source": Publication venue (journal, conference, arxiv, etc.)

EXTRACTION GUIDELINES:
1. REQUIRED: Create exactly one "paper" node with id="paper_main"
2. Extract ALL authors as separate nodes
3. Extract 5-15 concept nodes for key theoretical terms central to the paper
4. Create finding nodes for EACH major empirical result or conclusion
5. Create hypothesis nodes for proposed but untested claims
6. IMPORTANT: Extract limitations even if the paper minimizes them
7. Extract the publication venue as a "source" node
8. Use descriptive labels that capture the essence of each entity
9. Aim for 20-30 nodes total

CONFIDENCE SCORING:
- 1.0: Directly stated facts (author names, publication venue, explicit findings)
- 0.8-0.95: Well-supported findings with clear evidence
- 0.5-0.8: Proposed hypotheses or preliminary/tentative findings
- <0.5: Speculative claims or weak assertions

ID NAMING CONVENTION:
- paper_main (for the paper node)
- author_1, author_2, etc.
- concept_1, concept_2, etc.
- method_1, method_2, etc.
- finding_1, finding_2, etc.
- limitation_1, limitation_2, etc.

Paper text:
{text}

JSON:"""


def get_kg_edges_prompt(paper_title: str | None = None, nodes_json: str = "") -> str:
    """
    Get the prompt for Knowledge Graph edge extraction (pass 2 of 2).
    
    This is the second pass of the two-pass KG extraction approach.
    Takes the nodes from pass 1 and extracts relationships between them.
    
    Args:
        paper_title: Optional title to include in the prompt context
        nodes_json: JSON string of nodes extracted in pass 1
        
    Returns:
        Formatted prompt string
    """
    title_context = f'Paper: "{paper_title}"' if paper_title else ""
    
    return f"""Extract EDGES (relationships) between these knowledge graph nodes.

{title_context}

NODES (extracted from the paper):
{nodes_json}

OUTPUT: Valid JSON only. No markdown code blocks.

SCHEMA:
{{
    "edges": [
        {{
            "source": "<source_node_id>",
            "target": "<target_node_id>",
            "type": "<RELATIONSHIP_TYPE>"
        }}
    ]
}}

EDGE TYPES (use UPPERCASE, use exactly these):
- AUTHORED_BY: paper_main -> author nodes
- AFFILIATED_WITH: author -> institution
- PUBLISHED_IN: paper_main -> source (venue)
- PROPOSES: paper_main -> hypothesis or finding
- USES: paper_main -> method
- EVALUATES: method -> concept or finding
- SUPPORTS: finding -> finding, finding -> hypothesis, or concept -> concept
- CONTRADICTS: finding -> finding (when findings conflict)
- EXTENDS: concept -> concept, finding -> finding (when one builds on another)
- HAS_LIMITATION: paper_main -> limitation, or finding -> limitation
- ADDRESSES_CHALLENGE: paper_main -> concept (problems the paper addresses)
- RELATED_TO: concept -> concept (general conceptual relationship)

EXTRACTION GUIDELINES:
1. Connect paper_main to all author nodes with AUTHORED_BY
2. Connect paper_main to the source node with PUBLISHED_IN
3. Connect paper_main to all method nodes with USES
4. Connect paper_main to key finding nodes with PROPOSES
5. Connect related concepts with RELATED_TO or EXTENDS
6. Connect findings that support each other with SUPPORTS
7. Connect paper_main to limitation nodes with HAS_LIMITATION
8. Create CONTRADICTS edges for any conflicting findings
9. Every node should have at least one edge
10. Aim for 30-50 edges total

IMPORTANT:
- Only use node IDs that exist in the provided nodes list
- Ensure every node is connected to at least one other node
- Prioritize meaningful relationships over generic ones
- Use SUPPORTS and CONTRADICTS to show logical relationships between findings

JSON:"""


# Agent Prompts

def get_synthesis_prompt(query: str, context_nodes: str) -> str:
    """
    Generate prompt for Argument Agent synthesis.
    
    Enhanced to leverage richer context including:
    - Consensus findings (claims supported by multiple papers)
    - Methodology context for each claim
    - Edge relationships (what supports/contradicts what)
    - Paper core arguments for grounding
    """
    return f"""Answer this research question by synthesizing evidence from the literature corpus.

QUESTION: "{query}"

EVIDENCE FROM CORPUS:
{context_nodes}

The evidence above may include:
- CONSENSUS FINDINGS: Claims supported by multiple independent papers (strongest evidence)
- Individual claims with: methodology, confidence scores, and relationship edges
- Edge relationships showing what SUPPORTS, CONTRADICTS, or EXTENDS other claims
- Paper core arguments providing context for each finding

SYNTHESIS REQUIREMENTS:

1. LEAD WITH CONSENSUS: If multiple papers agree on a finding, state this first.
   Example: "There is strong agreement across 4 papers (Smith 2019; Jones 2020; Chen 2021; Lee 2022) that X..."

2. DISTINGUISH EVIDENCE QUALITY:
   - High confidence (0.8+) findings from well-established papers → state confidently
   - Lower confidence or single-source claims → hedge appropriately
   - Contradictions → acknowledge the debate explicitly

3. USE EDGE RELATIONSHIPS:
   - If finding A SUPPORTS finding B, synthesize them together
   - If finding A CONTRADICTS finding B, present both sides
   - If finding A EXTENDS finding B, show the progression

4. GROUND IN METHODOLOGY:
   - When citing quantitative findings, mention the method (e.g., "In a survey of 1,200 participants...")
   - Note methodological limitations where relevant

5. TEMPORAL AWARENESS:
   - Distinguish foundational work (older, highly cited) from recent developments
   - Note if consensus has shifted over time

6. CITE PRECISELY: Use [Author et al., Year] format for all claims.

7. ACKNOWLEDGE GAPS: If evidence is insufficient, say what IS known and what remains unclear.

8. NO DIRECT QUOTES: The text provided in "Core Argument Summary" is an AI-generated summary, not the original paper text. 
- DO NOT use quotation marks around this text.
- DO NOT attribute these exact words to the authors. 
- Instead, paraphrase: "Smith (2020) argues that..." or "The study suggests..."

Respond with substantive analysis (2-4 paragraphs). Do not simply list findings—synthesize them into a coherent answer."""


def get_hypothesis_validation_prompt(hypothesis: str, context_nodes: str) -> str:
    """
    Generate prompt for Validation Agent critique.
    
    Enhanced to leverage consensus detection and edge relationships
    for more accurate hypothesis evaluation.
    """
    return f"""Evaluate this hypothesis against the provided corpus evidence.

HYPOTHESIS: "{hypothesis}"

EVIDENCE FROM CORPUS:
{context_nodes}

The evidence may include:
- CONSENSUS FINDINGS: Claims supported by multiple papers (weight heavily)
- Individual findings with confidence scores and methodological context
- Edge relationships showing SUPPORTS/CONTRADICTS/EXTENDS between claims
- Paper core arguments for context

EVALUATION FRAMEWORK:

1. CONSENSUS CHECK: 
   - If consensus findings directly address the hypothesis, they are strong evidence
   - Multiple papers agreeing = high confidence in verdict

2. EDGE ANALYSIS:
   - Follow SUPPORTS edges: if hypothesis H is supported by finding F, and F is well-established, H gains support
   - Follow CONTRADICTS edges: if finding F contradicts hypothesis H, this is counter-evidence
   - Note chains: A supports B supports hypothesis → transitive support

3. METHODOLOGICAL WEIGHT:
   - Empirical findings (experiments, surveys) > theoretical claims
   - Larger samples > smaller samples
   - Recent replications > single studies

4. CONFIDENCE CALIBRATION:
   - SUPPORTED requires: 2+ papers with relevant findings, or 1 high-confidence (0.9+) finding directly on point
   - CONTRADICTED requires: clear counter-evidence, not merely absence of support
   - NOVEL: genuinely no relevant evidence (not just weak matches)

5. CONTRADICTION HANDLING:
   - If evidence is mixed (some supports, some contradicts), note the debate
   - Weight by paper influence and recency

OUTPUT FORMAT (JSON only):
{{
    "verdict": "SUPPORTED" | "CONTRADICTED" | "NOVEL" | "MIXED",
    "explanation": "<detailed reasoning citing specific evidence and edge relationships>",
    "supporting_evidence": [
        "Author, Year: <specific finding that supports hypothesis>"
    ],
    "contradicting_evidence": [
        "Author, Year: <specific finding that contradicts hypothesis>"
    ],
    "consensus_strength": "<strong/moderate/weak/none> - based on paper agreement",
    "confidence": <0.0-1.0 based on evidence quality and consensus>
}}

Note: Use MIXED verdict when substantial evidence exists on both sides. This is different from NOVEL (no evidence)."""


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





# Export main functions
__all__ = [
    'get_analysis_prompt',
    'get_retry_prompt',
    'get_json_repair_prompt',
    'get_kg_prompt',
    'get_kg_nodes_prompt',
    'get_kg_edges_prompt',
    'get_synthesis_prompt',
    'get_hypothesis_validation_prompt',
    'get_conceptual_ghost_prompt',
    'get_genealogy_prompt',
]