---
description: Complete API reference for Literature Mapper. Method signatures, parameters, and return types for LiteratureMapper, GhostHunter, and CorpusAnalyzer.
hide:
  - navigation
---

# API Reference

Complete method reference for the Literature Mapper Python API.

---

## LiteratureMapper

The main entry point for corpus management.

```python
from literature_mapper import LiteratureMapper

mapper = LiteratureMapper(corpus_path, model_name="gemini-3-flash-preview")
```

### Processing

| Method | Parameters | Returns | Description |
|:-------|:-----------|:--------|:------------|
| `process_new_papers()` | `recursive=False` | `ProcessingResult` | Process unprocessed PDFs |
| `update_citations()` | `email=None` | `None` | Fetch OpenAlex citation data |

### Retrieval

| Method | Parameters | Returns | Description |
|:-------|:-----------|:--------|:------------|
| `get_all_analyses()` | — | `DataFrame` | All papers as DataFrame |
| `get_paper_by_id()` | `paper_id: int` | `dict` | Single paper details |
| `search_papers()` | `column: str, query: str` | `DataFrame` | Keyword search |
| `search_corpus()` | `query, semantic, use_enhanced, min_year, max_year, node_types, limit` | `list[dict]` | Semantic search |
| `get_statistics()` | — | `CorpusStats` | Corpus counts |

### Normalization

| Method | Parameters | Returns | Description |
|:-------|:-----------|:--------|:------------|
| `normalize_authors()` | `mappings: dict` | `int` | Merge author aliases |
| `normalize_concepts()` | `mappings: dict` | `int` | Merge concept aliases |
| `get_canonical_author()` | `alias: str` | `str` | Resolve author alias |
| `get_canonical_concept()` | `alias: str` | `str` | Resolve concept alias |

### Temporal

| Method | Parameters | Returns | Description |
|:-------|:-----------|:--------|:------------|
| `get_concept_timeline()` | `concept=None, top_n=10` | `DataFrame` | Concept temporal data |
| `compute_temporal_stats()` | — | `None` | Compute trend statistics |
| `get_trending_concepts()` | `direction, limit` | `DataFrame` | Rising/declining concepts |
| `detect_concept_eras()` | `gap: int` | `DataFrame` | Revival detection |

### Genealogy

| Method | Parameters | Returns | Description |
|:-------|:-----------|:--------|:------------|
| `build_genealogy()` | `verbose=False` | `dict` | Infer paper relationships |
| `find_contradictions()` | `concept=None` | `DataFrame` | CHALLENGES edges |
| `get_argument_evolution()` | `concept: str` | `DataFrame` | Temporal relationship trace |

### Synthesis

| Method | Parameters | Returns | Description |
|:-------|:-----------|:--------|:------------|
| `synthesize_answer()` | `question, year_range=None, limit=10` | `str` | RAG answer synthesis |
| `validate_hypothesis()` | `hypothesis: str, limit=10` | `dict` | Hypothesis evaluation |

### Export

| Method | Parameters | Returns | Description |
|:-------|:-----------|:--------|:------------|
| `export_to_csv()` | `output_path: str` | `None` | Export to CSV |

---

## GhostHunter

Gap detection for missing papers and authors.

```python
from literature_mapper.ghosts import GhostHunter

hunter = GhostHunter(mapper)
```

| Method | Parameters | Returns | Description |
|:-------|:-----------|:--------|:------------|
| `find_bibliographic_ghosts()` | `threshold=3` | `DataFrame` | Missing cited papers |
| `find_missing_authors()` | `threshold=5` | `DataFrame` | Missing cited authors |

---

## CorpusAnalyzer

Corpus-level statistics and analytics.

```python
from literature_mapper.analysis import CorpusAnalyzer

analyzer = CorpusAnalyzer(corpus_path)
```

| Method | Parameters | Returns | Description |
|:-------|:-----------|:--------|:------------|
| `get_year_distribution()` | — | `DataFrame` | Papers per year |
| `get_top_authors()` | `limit=10` | `DataFrame` | Most prolific authors |
| `get_top_concepts()` | `limit=10` | `DataFrame` | Most frequent concepts |
| `find_hub_papers()` | `limit=10` | `DataFrame` | Most referenced papers |

---

## Visualization

```python
from literature_mapper.viz import export_to_gexf

export_to_gexf(corpus_path, output_path, mode="semantic", threshold=0.05)
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `corpus_path` | `str` | — | Path to corpus directory |
| `output_path` | `str` | — | Output `.gexf` file path |
| `mode` | `str` | `"semantic"` | Graph type: `semantic`, `authors`, `concepts`, `river`, `similarity` |
| `threshold` | `float` | `0.05` | Minimum edge weight (for `similarity` mode) |

---

## Data Classes

### ProcessingResult

```python
@dataclass
class ProcessingResult:
    processed: int    # Number of successfully processed PDFs
    skipped: int      # Already in database
    failed: int       # Processing errors
```

### CorpusStats

```python
@dataclass
class CorpusStats:
    total_papers: int
    total_authors: int
    total_concepts: int
```
