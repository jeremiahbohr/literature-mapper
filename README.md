# Literature Mapper

An AI-powered Python library for systematic, scalable analysis of academic literature.

Literature Mapper turns a folder of PDF articles into a structured, queryable SQLite database. While primarily designed as a Python library for Jupyter and other interactive environments, it also offers a full-featured command-line interface (CLI) for quick tasks.

---

## Features

* **Gemini Models** – Works with any available Gemini model (default: `gemini-2.5-flash`)
* **Automated Metadata Extraction** – Titles, authors, methodologies, key concepts, contributions  
* **Duplicate Prevention** – Database constraints prevent processing the same paper twice
* **Resilient Error Handling** – Gracefully skips corrupted PDFs, API hiccups, and edge cases with user-friendly messages
* **Clean Database Schema** – SQLite with proper constraints and relational tables for authors and concepts
* **Data Export** – One-line CSV export for downstream pipelines  
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

# 1 – Initialize the mapper for your research folder
#     (creates ./my_ai_research/corpus.db on first run)
mapper = LiteratureMapper("./my_ai_research")

# 2 – Drop some PDF files into ./my_ai_research/

# 3 – Process any new papers
results = mapper.process_new_papers(recursive=True)  # Include all subfolders
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

## Architecture

Literature Mapper follows clean architecture principles:

* **Single Responsibility** – Each module has one clear purpose
* **Dependency Injection** – Database sessions managed with context managers
* **Error Boundaries** – Comprehensive exception hierarchy with user-friendly messages
* **Input Validation** – Centralized validation with single source of truth
* **Resource Management** – Proper cleanup of database connections and API resources

---

## Known Limitations

* **PDF Processing** – Requires readable text content (scanned documents without OCR may fail)
* **Processing Speed** – Depends on chosen Gemini model and API rate limits
* **File Size** – PDFs larger than 50MB are rejected by default (configurable)
* **Duplicate Prevention** – Papers with identical titles and years are prevented by database constraints

---

## Design Philosophy

* **Reliable** – Predictable behavior with comprehensive error handling
* **Simple** – Minimal setup, sensible defaults, clear APIs
* **Secure** – Strict input validation and safe database operations
* **Maintainable** – Clean architecture with proper separation of concerns
* **User-Friendly** – Clear error messages and helpful CLI output

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