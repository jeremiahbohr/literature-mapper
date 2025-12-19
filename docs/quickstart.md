---
description: Get up and running with Literature Mapper in 5 minutes.
hide:
  - navigation
---

# Quick Start

Get Literature Mapper running in 5 minutes.

---

## Prerequisites

| Requirement | Details |
|:------------|:--------|
| Python | 3.10 or newer |
| API Key | `GEMINI_API_KEY` environment variable ([get one](https://aistudio.google.com/)) |
| Corpus | A folder containing PDF files |

---

## Installation

```bash
pip install literature-mapper
```

Configure your API key:

=== "Linux/macOS"
    ```bash
    export GEMINI_API_KEY="your_api_key_here"
    ```

=== "Windows (PowerShell)"
    ```powershell
    $env:GEMINI_API_KEY="your_api_key_here"
    ```

---

## Basic Workflow

### 1. Initialize the Mapper

```python
from literature_mapper import LiteratureMapper

# Point to your PDF folder (creates corpus.db automatically)
mapper = LiteratureMapper("./my_research")
```

### 2. Process Your PDFs

```python
# Process all PDFs (incremental—skips already-processed files)
result = mapper.process_new_papers(recursive=True)

print(f"Processed: {result.processed}")
print(f"Skipped: {result.skipped}")
```

!!! tip "Cost Estimate"
    Processing costs approximately **$0.50 USD for 50 papers** via the Gemini API.

### 3. Fetch Citation Data

```python
# Enrich with OpenAlex citation counts and references
mapper.update_citations()
```

### 4. Search Your Corpus

```python
# Semantic search (finds by meaning, not keywords)
results = mapper.search_corpus(
    query="influence of weak ties on information spread",
    semantic=True,
    limit=5
)

for r in results:
    print(f"[{r['match_score']:.2f}] {r['title']} ({r['year']})")
```

### 5. Synthesize Answers

```python
# Ask a research question—get a cited answer
answer = mapper.synthesize_answer(
    "What factors influence information spread in social networks?"
)
print(answer)
```

### 6. Validate Hypotheses

```python
# Test a claim against your corpus
result = mapper.validate_hypothesis(
    "Strong ties are more effective than weak ties for spreading information."
)

print(f"Verdict: {result['verdict']}")  # SUPPORTED, CONTRADICTED, MIXED, or NOVEL
print(result['explanation'])
```

---

## CLI Quick Reference

Literature Mapper also offers a complete command-line interface:

```bash
# Process PDFs
literature-mapper process ./my_research --recursive

# Fetch citations
literature-mapper citations ./my_research

# Synthesize an answer
literature-mapper synthesize ./my_research "What is the impact of X on Y?"

# Find missing papers (ghost hunting)
literature-mapper ghosts ./my_research --mode bibliographic

# Export for Gephi
literature-mapper viz ./my_research -o graph.gexf --mode semantic
```

---

## Next Steps

- **[Full Vignette](vignette/setup.md)** — Comprehensive walkthrough of all features
- **[CLI Reference](cli.md)** — Complete command documentation
- **[API Reference](api.md)** — Method signatures and parameters
