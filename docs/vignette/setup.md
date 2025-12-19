---
description: Set up Literature Mapper and process your first PDFs. Understand the database schema and corpus structure.
---

# Setup & Understanding Your Corpus

Initialize Literature Mapper, process PDFs, and explore the database structure.

---

## Prerequisites

| Requirement | Details |
|:------------|:--------|
| Python | 3.10 or newer |
| API Key | `GEMINI_API_KEY` environment variable ([get one](https://aistudio.google.com/)) |
| Corpus | A folder containing PDF files |
| Install | `pip install literature-mapper` |

!!! info "Cost Estimate"
    Processing costs approximately **$0.50 USD for 50 papers** via the Gemini API.

---

## Initialization

```python
import os
from pathlib import Path
from literature_mapper import LiteratureMapper

# Verify API key is set
if not os.getenv('GEMINI_API_KEY'):
    raise EnvironmentError("GEMINI_API_KEY not found")

# Define your corpus location
CORPUS_PATH = Path("./my_research")

# Initialize (creates corpus.db if needed)
mapper = LiteratureMapper(
    corpus_path=str(CORPUS_PATH),
    model_name="gemini-3-flash-preview"
)
```

---

## Processing PDFs

Processing is **incremental**—previously processed PDFs are skipped automatically.

```python
result = mapper.process_new_papers(recursive=True)

print(f"Processed: {result.processed}")
print(f"Skipped (already in DB): {result.skipped}")
print(f"Failed: {result.failed}")
```

??? warning "Common Issues"
    - **Scanned PDFs without OCR** → No extractable text
    - **Password-protected files** → Extraction fails
    - **Corrupted PDF structure** → Partial or no data

??? example "CLI Equivalent"
    ```bash
    literature-mapper process ./my_research --recursive
    ```

---

## Database Schema

Literature Mapper stores everything in SQLite (`corpus.db`). Key tables:

| Table | Description | Key Columns |
|:------|:------------|:------------|
| `papers` | Core metadata | `id`, `title`, `year`, `core_argument`, `methodology` |
| `authors` | Unique author names | `id`, `name`, `canonical_name` |
| `concepts` | Extracted key terms | `id`, `name`, `canonical_name` |
| `paper_authors` | Many-to-many link | `paper_id`, `author_id` |
| `paper_concepts` | Many-to-many link | `paper_id`, `concept_id` |
| `kg_nodes` | Knowledge graph nodes | `id`, `type`, `label`, `vector` (embedding) |
| `kg_edges` | Relationships | `source_id`, `target_id`, `relation` |
| `citations` | OpenAlex data | `paper_id`, `cited_doi`, `cited_title` |
| `intellectual_edges` | Genealogy relationships | `source_paper_id`, `target_paper_id`, `relation_type` |

---

## Corpus Statistics

```python
stats = mapper.get_statistics()

print(f"Papers:   {stats.total_papers}")
print(f"Authors:  {stats.total_authors}")
print(f"Concepts: {stats.total_concepts}")
```

### Viewing All Papers

```python
import pandas as pd

papers_df = mapper.get_all_analyses()
papers_df[['title', 'year', 'authors', 'journal']].head()
```

---

## Knowledge Graph Structure

The KG contains typed nodes with semantic edges:

| Node Type | Description |
|:----------|:------------|
| `paper` | The paper itself |
| `author` | Paper authors |
| `finding` | Key results or claims |
| `method` | Research methods |
| `concept` | Important terms |
| `limitation` | Acknowledged weaknesses |
| `hypothesis` | Proposed theories |

### Inspecting Node Distribution

```python
from literature_mapper.database import get_db_session, KGNode, KGEdge
from sqlalchemy import func

with get_db_session(CORPUS_PATH) as session:
    node_counts = (
        session.query(KGNode.type, func.count(KGNode.id))
        .group_by(KGNode.type)
        .order_by(func.count(KGNode.id).desc())
        .all()
    )
    edge_count = session.query(KGEdge).count()

print("Node Types:")
for node_type, count in node_counts:
    print(f"  {node_type:15s} {count:>5}")
print(f"\nTotal edges: {edge_count:,}")
```

---

## Temporal Distribution

Visualize when papers in your corpus were published:

```python
import matplotlib.pyplot as plt
from literature_mapper.analysis import CorpusAnalyzer

analyzer = CorpusAnalyzer(CORPUS_PATH)
year_dist = analyzer.get_year_distribution()

if not year_dist.empty:
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(year_dist['year'], year_dist['count'], color='steelblue')
    ax.set_xlabel('Year')
    ax.set_ylabel('Papers')
    ax.set_title('Publication Timeline')
    plt.show()
```
