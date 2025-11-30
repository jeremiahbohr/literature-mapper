# Literature Mapper

An AI-powered Python library for systematic, scalable analysis of academic literature.

Literature Mapper turns a folder of PDF articles into a structured, queryable SQLite database. It combines local PDF processing with Gemini AI analysis and OpenAlex citation data to create a rich Knowledge Graph of your research field.

---

## Features

* **Knowledge Graph Extraction**: Automatically extracts concepts, authors, methods, and findings as connected nodes.
    * **Nodes**: Papers, Concepts, Findings, Methods, Authors, Institutions, Limitations.
    * **Edges**: `PAPER -> HAS_CONCEPT`, `PAPER -> HAS_METHOD`, `AUTHOR -> COAUTHORED_WITH`, `CONCEPT -> RELATED_TO`.
    * **Storage**: Normalized SQLite schema (`kg_nodes`, `kg_edges`), exportable to `.gexf` for graph tools.
* **OpenAlex Integration**: Automatically fetches citation counts and references for papers in your corpus, enabling robust bibliometric analysis.
* **Ghost Hunting**: Algorithms to identify missing pieces in your literature review:
    * **Bibliographic Ghosts**: Papers frequently cited by your corpus but missing from it.
    * **Missing Authors**: Influential authors cited by your corpus who aren't directly represented.
* **Thematic Agents**: Synthesize answers and validate hypotheses using the Knowledge Graph.
    * **Argument Agent**: Aggregates evidence to answer research questions.
    * **Validation Agent**: Critiques hypotheses against the literature.
* **Semantic Search**: Find relevant content by meaning using vector embeddings.
* **Gemini Models**: Works with any available Gemini model (default: `gemini-2.5-flash`).
* **Clean Database Schema**: SQLite with proper constraints and relational tables.
* **Simple CLI**: Process, query, and export directly from the terminal.

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
mapper.fetch_citations()

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

### Embeddings
Literature Mapper uses Google's `models/text-embedding-004` to generate vector embeddings for every concept, finding, and paper title in the Knowledge Graph. This enables the agents to find relevant information based on semantic meaning (e.g., matching "hallucination" with "context loss") rather than just keyword overlap.

### OpenAlex Integration
The system uses OpenAlex to fetch high-quality citation data. It attempts to match papers by DOI first, then by title. This data is crucial for the `bibliographic` and `authors` ghost modes. No API key is required for OpenAlex, but the system is configured to be polite with rate limits.

---

## Requirements

* Python 3.8 or newer  
* Google AI API key ([create one here](https://makersuite.google.com/app/apikey))  
* Internet connection (for Gemini API and OpenAlex)

---

## License

Released under the MIT License. See the `LICENSE` file for full text.