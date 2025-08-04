# Literature Mapper

An AI-powered Python library for systematic, scalable analysis of academic literature.

Literature Mapper turns a folder of PDF articles into a structured, queryable SQLite database, enabling new forms of computational literature review. While primarily designed as a Python library for Jupyter and other interactive environments, it also offers a full-featured command-line interface (CLI) for quick tasks.

---

## Features

* **Gemini Models** – Works with any available Gemini model (default: `gemini-2.5-flash`)
* **Model-Aware Optimisation** – Automatically adjusts analysis depth based on model capabilities  
* **Automated Metadata Extraction** – Titles, authors, methodologies, key concepts, contributions  
* **Incremental Processing** – Only analyses new PDFs added since the last run  
* **Resilient Error Handling** – Gracefully skips corrupted PDFs, API hiccups, and edge cases with user-friendly messages
* **Flexible Database** – SQLite schema with relational tables for authors and concepts (allows duplicate paper titles)
* **Data Export** – One-line CSV export for R, Excel, or downstream ML pipelines  
* **Manual Entry** – Add papers that are not available as PDFs  
* **Simple CLI** – Process, query, and export directly from the terminal  

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

> **Tip:** Use a Python virtual environment  
> `python -m venv .venv && source .venv/bin/activate`  
> to keep dependencies isolated.

---

## Quick Start (Jupyter / Python)

```python
from literature_mapper import LiteratureMapper

# 1 – Initialise the mapper for your research folder
#     (creates ./my_ai_research/corpus.db on first run)
mapper = LiteratureMapper("./my_ai_research")

# 2 – Drop some PDF files into ./my_ai_research/

# 3 – Process any new papers
results = mapper.process_new_papers()
print(f"Processed: {results.processed}, Failed: {results.failed}, Skipped: {results.skipped}")
# Example output: "Processed: 12, Failed: 1, Skipped: 2"

# 4 – Load the analyses into a pandas DataFrame
df = mapper.get_all_analyses()
df.head()

# 5 – Optional: export the corpus to CSV
mapper.export_to_csv("ai_research_corpus.csv")
```

Need a different Gemini model? Just pass it in:

```python
mapper = LiteratureMapper("./my_ai_research", model_name="gemini-2.5-pro")
```

---

## Model Flexibility

List available Gemini models and their recommended use-cases:

```bash
literature-mapper models            # simple list
literature-mapper models --details  # table with guidance
```

**Model Recommendations:**
- **Flash**: Fast analysis, ideal for large batches
- **Pro**: Balanced analysis, best for most use cases  
- **Ultra**: Highest quality analysis, slower but most comprehensive

Then process with any model:

```bash
literature-mapper process ./my_ai_research --model gemini-2.5-pro
```

---

## Data Curation & Standardisation

```python
# Search for all papers that mention 'survey' as their methodology
survey_df = mapper.search_papers(column="methodology", query="survey")
print(survey_df[["id", "title", "methodology"]])

# Standardise the methodology field
ids = survey_df["id"].tolist()
mapper.update_papers(ids, {"methodology": "Survey"})
```

---

## Command-Line Interface Highlights

```bash
# Process a folder of PDFs
literature-mapper process ./my_research

# Show corpus status and basic stats
literature-mapper status ./my_research

# Export to CSV
literature-mapper export ./my_research output.csv

# List first 10 papers from 2024
literature-mapper papers ./my_research --year 2024 --limit 10
```

Run `literature-mapper --help` for the full command tree.

---

## Configuration via Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `GEMINI_API_KEY` | **Required.** Google AI key | – |
| `LITERATURE_MAPPER_MODEL` | Default model for CLI | `gemini-2.5-flash` |
| `LITERATURE_MAPPER_MAX_FILE_SIZE` | Max PDF size (bytes) | `52428800` (50 MB) |
| `LITERATURE_MAPPER_BATCH_SIZE` | PDFs processed per batch | `10` |
| `LITERATURE_MAPPER_LOG_LEVEL` | Log level (`DEBUG`, `INFO`, …) | `INFO` |
| `LITERATURE_MAPPER_VERBOSE` | Set to `true` for debug logs | `false` |

---

## Advanced Usage

### Robust Error Handling

Literature Mapper provides user-friendly error messages for common issues:

```python
from literature_mapper.exceptions import PDFProcessingError, APIError, ValidationError

try:
    results = mapper.process_new_papers()
except PDFProcessingError as e:
    print(f"PDF issue: {e.user_message}")  # e.g., "File 'paper.pdf' is password-protected"
except APIError as e:
    print(f"API issue: {e.user_message}")  # e.g., "Gemini API rate limit exceeded"
except ValidationError as e:
    print(f"Input error: {e.user_message}")  # e.g., "Invalid API key format"
```

### Corpus Statistics

```python
stats = mapper.get_statistics()
print(f"Total papers: {stats.total_papers}")
print(f"Unique authors: {stats.total_authors}")
print(f"Key concepts: {stats.total_concepts}")
```

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

---

## Testing

```bash
# Install development dependencies
pip install pytest

# Run the test suite
pytest tests/

# Run with coverage
pip install pytest-cov
pytest tests/ --cov=literature_mapper
```

---

## Requirements

* Python 3.8 or newer  
* Google AI API key ([create one here](https://makersuite.google.com/app/apikey))  
* A few MB of disk space for binaries plus additional space for your corpus database  

---

## Known Limitations

* **Duplicate papers**: Multiple papers with identical titles and years are allowed (common in academic literature with conference/journal versions)
* **PDF processing**: Requires readable text content (scanned documents without OCR may fail)
* **Processing speed**: Depends on chosen Gemini model and API rate limits
* **File size**: PDFs larger than 50MB are rejected by default (configurable)

---

## Design Philosophy

* **Simple** – Minimal setup, sensible defaults  
* **User-Centric** – Clear CLI and notebook ergonomics with helpful error messages
* **Secure** – Strict input validation and API-key handling  
* **Robust** – Comprehensive error handling and retry logic  
* **Future-Proof** – Model-agnostic architecture for the Gemini family  

---

## Contributing

Pull requests, feature ideas, and bug reports are welcome. Please open an issue first if you plan to work on a significant change.

For development:
```bash
git clone https://github.com/jeremiahbohr/literature-mapper.git
cd literature-mapper
pip install -e ".[dev]"
pytest tests/
```

---

## License

Released under the MIT License. See the `LICENSE` file for full text.