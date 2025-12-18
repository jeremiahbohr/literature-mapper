# Literature Mapper

An AI-powered Python library for systematic analysis of academic literature.

Literature Mapper transforms a folder of PDFs into a structured, queryable Knowledge Graph. It extracts typed claims (findings, methods, limitations) with confidence scores, enriches papers with OpenAlex citation data, and provides LLM agents that synthesize answers strictly from your corpus—no hallucinated citations, no external knowledge bleed.

Designed for researchers working with curated collections of 10–500 papers.

---

## Core Capabilities

| Capability | What it does |
|------------|--------------|
| **Knowledge Graph** | Extracts papers, concepts, findings, methods, and limitations as typed nodes with semantic edges (SUPPORTS, CONTRADICTS, EXTENDS) |
| **Temporal Analysis** | Tracks concept trends, detects rising/declining topics, identifies revival patterns across publication years |
| **Citation Enrichment** | Fetches citation counts and references from OpenAlex; computes influence metrics |
| **Gap Detection** | Finds "ghost" papers and authors—works frequently cited by your corpus but missing from it |
| **Bounded Synthesis** | RAG agents that answer questions and validate hypotheses using only your corpus, with full citation provenance |
| **Graph Export** | GEXF export for Gephi visualization (co-authorship networks, concept maps, paper similarity) |

---

## Installation

```bash
# Install from PyPI
pip install literature-mapper

# Or install the latest commit from GitHub
pip install git+https://github.com/jeremiahbohr/literature-mapper.git

# Configure your Google AI API key
export GEMINI_API_KEY="your_api_key_here"
```

## Quick Start (Jupyter / Python)

```python
from literature_mapper import LiteratureMapper

# 1: Initialize the mapper (creates corpus.db)
mapper = LiteratureMapper("./my_ai_research")

# 2: Process PDFs (Extracts Metadata + Knowledge Graph)
results = mapper.process_new_papers(recursive=True)
print(f"Processed: {results.processed}")

# 3: Fetch Citations (OpenAlex)
# Populates citation counts and references for processed papers
mapper.update_citations()

# 4: Synthesize Answers (Argument Agent)
answer = mapper.synthesize_answer("What are the limitations of current methods?")
print(answer)

# 5: Validate Hypotheses (Validation Agent)
critique = mapper.validate_hypothesis("Current methods have solved the problem of hallucination.")
print(critique['verdict'])  # e.g., "CONTRADICTED"
print(critique['explanation'])

# 6: Export Data
mapper.export_to_csv("corpus.csv")
```

---

## Command-Line Interface

Literature Mapper offers a powerful CLI for managing your research corpus.

### Core Workflow

1.  **Process PDFs**: Extract text and build the Knowledge Graph.
    ```bash
    literature-mapper process ./my_research --recursive
    ```

2.  **Fetch Citations**: Enrich your corpus with data from OpenAlex.
    ```bash
    literature-mapper citations ./my_research
    ```

3.  **Analyze Status**: View corpus statistics and health.
    ```bash
    literature-mapper status ./my_research
    ```

### Visualization

Export your corpus as a `.gexf` file for visualization in tools like [Gephi](https://gephi.org/).

```bash
# Default: Semantic Knowledge Graph
literature-mapper viz ./my_research --output graph.gexf
```

| Mode | Description | Best For |
|------|-------------|----------|
| `semantic` | **(Default)** The full Knowledge Graph (Concepts, Findings, Methods). | Understanding the logical structure of arguments. |
| `authors` | Co-authorship network (weighted by shared papers). | Identifying "Invisible Colleges" and key researchers. |
| `concepts` | Topic co-occurrence network. | Mapping the "Topic Landscape" of the field. |
| `river` | Same as `concepts`, but adds a `start` year attribute. | Creating dynamic networks (similar to ThemeRiver visualizations) in Gephi. |
| `similarity` | Paper similarity map based on shared concepts (Jaccard Index). | Finding thematically similar papers without direct citations. |

### Ghost Hunting

Identify missing links and gaps in your literature review.

```bash
literature-mapper ghosts ./my_research --mode <MODE>
```

| Mode | Description |
|------|-------------|
| `bibliographic` | **(Default)** Identifies papers frequently cited by your corpus but missing from it. Helps you find seminal works you missed. |
| `authors` | Identifies authors frequently cited by your corpus but not represented in it. Helps you find key voices in the field. |

### Temporal Analysis

Uncover trends and history in your field. Note: You must run `literature-mapper temporal` first to compute these stats.

```bash
# 1. Compute Temporal Stats (Run this first!)
literature-mapper temporal ./my_research

# 2. View Trending Concepts
literature-mapper trends ./my_research --direction rising
literature-mapper trends ./my_research --direction declining

# 3. Analyze a Specific Concept's Trajectory
literature-mapper trajectory "hallucination" ./my_research

# 4. Detect Concept Eras (Revivals)
literature-mapper eras ./my_research --gap 5
```

### Analysis Tools

```bash
# Synthesize an answer to a research question
literature-mapper synthesize ./my_research "What is the impact of X on Y?"

# Validate a hypothesis against the corpus
literature-mapper validate ./my_research "X causes Y."

# Identify Hubs (Most Cited Papers)
literature-mapper hubs ./my_research

# View Comprehensive Corpus Statistics
literature-mapper stats ./my_research
```

---

## Configuration via Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `GEMINI_API_KEY` | **Required.** Google AI key | None |
| `LITERATURE_MAPPER_MODEL` | Default model for CLI | `gemini-2.5-flash` |
| `LITERATURE_MAPPER_MAX_FILE_SIZE` | Max PDF size (bytes) | `52428800` (50 MB) |
| `LITERATURE_MAPPER_BATCH_SIZE` | PDFs processed per batch | `10` |
| `LITERATURE_MAPPER_LOG_LEVEL` | Log level (`DEBUG`, `INFO`, …) | `INFO` |
| `LITERATURE_MAPPER_VERBOSE` | Set to `true` for debug logs | `false` |

---

## Advanced Usage

### Embeddings & Retrieval
Literature Mapper uses Google's `models/text-embedding-004` to generate vector embeddings for every concept, finding, and paper title. The enhanced retrieval engine uses Maximal Marginal Relevance (MMR) to ensure you get distinct pieces of evidence rather than repetitive claims. It also detects *Consensus Groups*, identifying when multiple papers support the same finding, and presents them as a unified block of evidence.

### Temporal Logic
Temporal stats are computed using a linear regression on the number of papers mentioning a concept per year.
- **Trend Slope**: Positive values indicate a concept is "Rising" (appearing in more papers over time). Negative values indicate "Declining".
- **Eras**: The system detects "gaps" where a concept disappears for N years and then reappears. This is useful for finding forgotten methods that were later revived.

### OpenAlex Integration
The system uses OpenAlex to fetch high-quality citation data. It attempts to match papers by DOI first, then by title. This data is crucial for the `bibliographic` and `authors` ghost modes. No API key is required for OpenAlex, but the system is configured to be polite with rate limits.

---

## Requirements

* Python 3.10 or newer  
* Google AI API key ([create one here](https://aistudio.google.com/app/api-keys))  
* Internet connection (for Gemini API and OpenAlex)

---

## License

Released under the MIT License. See the `LICENSE` file for full text.