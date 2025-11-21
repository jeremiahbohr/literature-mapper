# Literature Mapper

An AI-powered Python library for systematic, scalable analysis of academic literature.

Literature Mapper turns a folder of PDF articles into a structured, queryable SQLite database. While primarily designed as a Python library for Jupyter and other interactive environments, it also offers a full-featured command-line interface (CLI) for quick tasks.

---

## Features

* **Knowledge Graph Extraction**: Automatically extracts concepts, authors, methods, and findings as connected nodes.
* **Thematic Agents**: Synthesize answers and validate hypotheses using the Knowledge Graph.
    * **Argument Agent**: Aggregates evidence to answer research questions.
    * **Validation Agent**: Critiques hypotheses against the literature.
* **Semantic Search**: Find relevant content by meaning using vector embeddings.
* **Gemini Models**: Works with any available Gemini model (default: `gemini-2.5-flash`).
* **Automated Metadata Extraction**: Titles, authors, methodologies, key concepts, contributions.
* **Duplicate Prevention**: Database constraints prevent processing the same paper twice.
* **Resilient Error Handling**: Gracefully skips corrupted PDFs and API hiccups.
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

# 3: Synthesize Answers (Argument Agent)
answer = mapper.synthesize_answer("What are the limitations of current methods?")
print(answer)

# 4: Validate Hypotheses (Validation Agent)
critique = mapper.validate_hypothesis("Current methods have solved the problem of hallucination.")
print(critique['verdict'])  # e.g., "CONTRADICTED"
print(critique['explanation'])

# 5: Export Data
mapper.export_to_csv("corpus.csv")
```

Need a different Gemini model? Just pass it in:

```python
mapper = LiteratureMapper("./my_ai_research", model_name="gemini-2.5-pro")
```

---

## Model Support

List available Gemini models:

```bash
literature-mapper models            # simple list
literature-mapper models --details  # table with guidance
```

**Model Recommendations:**
- **Flash**: Fast analysis, ideal for large batches
- **Pro**: Balanced analysis, best for most use cases  

Then process with any model:

```bash
literature-mapper process ./my_ai_research --model gemini-2.5-pro
```

---

## Data Management

```python
# Search for papers by methodology
survey_df = mapper.search_papers(column="methodology", query="survey")
print(survey_df[["id", "title", "methodology"]])

# Update paper metadata
ids = survey_df["id"].tolist()
mapper.update_papers(ids, {"methodology": "Systematic Review"})

# Get corpus statistics
stats = mapper.get_statistics()
print(f"Papers: {stats.total_papers}, Authors: {stats.total_authors}")
```

## Organizing PDFs

Literature Mapper can process PDFs from:
- **Main folder only** (default): `mapper.process_new_papers()`
- **All subfolders**: `mapper.process_new_papers(recursive=True)`

All papers are stored in the same database regardless of folder structure, so you can organize PDFs by topic, year, or any system you prefer while maintaining unified search and analysis.

---

## Command-Line Interface

```bash
# Process a folder of PDFs
literature-mapper process ./my_research

# Show corpus status and basic stats
literature-mapper status ./my_research

# Export to CSV
literature-mapper export ./my_research output.csv

# List recent papers
literature-mapper papers ./my_research --year 2024 --limit 10
```

Run `literature-mapper --help` for the full command tree.

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

Manual entries are also automatically embedded, ensuring they are fully accessible to the Argument and Validation agents.

### Manual Entry

```python
mapper.add_manual_entry(
    title="Seminal Survey of AI Ethics",
    authors=["Smith, J.", "Doe, A."],
    year=2025,
    methodology="Systematic Literature Review",
    theoretical_framework="Ethics Framework",
    contribution_to_field="Comprehensive review of AI ethics landscape",
    key_concepts=["AI ethics", "survey", "responsible AI"]
)
```

### Database Integrity

Literature Mapper prevents duplicate papers through database constraints:
- Papers with identical titles and years are automatically rejected
- PDFs are tracked by file path to prevent reprocessing
- Failed operations are rolled back to maintain consistency

---

## Requirements

* Python 3.8 or newer  
* Google AI API key ([create one here](https://makersuite.google.com/app/apikey))  
* A few MB of disk space for binaries plus additional space for your corpus database  

---

## License

Released under the MIT License. See the `LICENSE` file for full text.