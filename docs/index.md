---
description: Literature Mapper — An AI-powered Python library for systematic analysis of academic literature. Transform PDFs into queryable knowledge graphs.
hide:
  - navigation
---

# Literature Mapper

**Transform a folder of PDFs into a structured, queryable Knowledge Graph.**

Literature Mapper is an AI-powered Python library that extracts typed claims (findings, methods, limitations) with confidence scores, enriches papers with OpenAlex citation data, and provides LLM agents that synthesize answers strictly from your corpus—no hallucinated citations, no external knowledge bleed.

<div class="grid cards" markdown>

- :material-graph-outline: **Knowledge Graph Extraction**
  
    Extract typed nodes and semantic edges (SUPPORTS, CONTRADICTS, EXTENDS) from your papers. The foundation for everything else.

- :material-magnify: **Semantic Search**
  
    Find evidence by meaning, not just keywords. Uses MMR reranking for diverse, high-quality retrieval.

- :material-chart-timeline-variant: **Temporal Analysis**
  
    Track concept trends, detect rising/declining topics, and identify revival patterns across publication years.

- :material-ghost-outline: **Gap Detection (Ghost Hunting)**
  
    Find papers and authors frequently cited by your corpus but missing from it.

- :material-brain: **Bounded Synthesis**
  
    RAG agents that answer questions and validate hypotheses using *only your corpus*, with full citation provenance.

- :material-family-tree: **Intellectual Genealogy**
  
    Trace the lineage of ideas. Identify seminal papers that spawned entire sub-fields and visualize the inheritance of concepts.

- :material-file-export-outline: **Graph Export**
  
    GEXF export for Gephi: co-authorship networks, concept maps, and paper similarity graphs.

</div>

---

## What is Literature Mapper?

Literature Mapper is designed for researchers working with curated collections of 10–500 papers. It systematically:

1. **Processes PDFs** — Extracts text, identifies metadata, and builds a knowledge graph
2. **Enriches with citations** — Fetches citation counts and references from OpenAlex
3. **Enables discovery** — Semantic search, ghost hunting, temporal trends
4. **Synthesizes knowledge** — AI agents answer questions grounded strictly in your corpus

---

## Installation

```bash
pip install literature-mapper
```

Set your API key:

```bash
export GEMINI_API_KEY="your_api_key_here"
```

---

## Quick Example

```python
from literature_mapper import LiteratureMapper

# Initialize with your PDF folder
mapper = LiteratureMapper("./my_research")

# Process all PDFs
mapper.process_new_papers(recursive=True)

# Synthesize an answer from your corpus
answer = mapper.synthesize_answer(
    "What are the main debates about network structure?"
)
print(answer)
```

---

## Frequently Asked Questions

??? question "What is a Knowledge Graph in Literature Mapper?"
    A Knowledge Graph (KG) is a structured representation of your academic literature. Each paper becomes a node, connected to concepts, findings, methods, and limitations it discusses. Edges represent relationships like SUPPORTS, CONTRADICTS, or EXTENDS.

??? question "How does Ghost Hunting work?"
    Ghost Hunting uses your corpus's citation data to find structural gaps. "Bibliographic ghosts" are papers frequently cited by your corpus but not included. "Author ghosts" are researchers cited often but not represented.

??? question "What models does Literature Mapper use?"
    Literature Mapper uses Google's Gemini models for text extraction and synthesis, and `text-embedding-004` for semantic embeddings. All processing happens via the Gemini API.

??? question "Can I use Literature Mapper without an API key?"
    No, a Gemini API key is required for processing PDFs and synthesis. However, once papers are processed, many analysis features (search, ghost hunting, export) work offline.

??? question "How much does it cost to process papers?"
    Approximately **$0.50 USD for 50 papers** via the Gemini API. Costs vary based on paper length.

??? question "Does Google train on my PDFs?"
    **No.** Literature Mapper uses the paid tier of the Gemini API (via your API key), which has a strict zero-data-retention policy. Your data is not used to train models.

??? question "Does it work with non-English papers?"
    **Yes.** Because Gemini is natively multilingual, Literature Mapper can process, index, and synthesize answers from papers in German, French, Chinese, Japanese, and many other languages without translation steps.

??? question "How do I see the graph?"
    You can export your Knowledge Graph to GEXF format using the `viz` command (`literature-mapper viz ./my_research`). This file can be opened in **Gephi**, a free open-source visualization tool, to explore connection clusters and intellectual genealogies.
