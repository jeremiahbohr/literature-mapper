---
description: Export your corpus to CSV or GEXF for visualization in Gephi. Create co-authorship, concept, and knowledge graph visualizations.
---

# Export & Visualization

Export to CSV for analysis, GEXF for network visualization in Gephi.

---

## CSV Export

Export all paper metadata and analyses to a CSV file:

```python
mapper.export_to_csv("corpus_export.csv")
```

The CSV includes:

- Paper metadata (title, authors, year, journal)
- AI-extracted fields (core argument, methodology, key findings)
- Citation metrics (if fetched)
- Concept associations

---

## GEXF Export (Gephi)

Export your corpus as a network graph for visualization in [Gephi](https://gephi.org/):

```python
from literature_mapper.viz import export_to_gexf

export_to_gexf(
    corpus_path=str(CORPUS_PATH),
    output_path="network_semantic.gexf",
    threshold=0.05,
    mode='semantic'
)
```

### Export Modes

| Mode | Description | Best For |
|:-----|:------------|:---------|
| `semantic` | Full knowledge graph | Understanding argument structure |
| `authors` | Co-authorship network | Research communities |
| `concepts` | Topic co-occurrence | Topic landscape |
| `river` | Concepts with temporal data | Dynamic network visualization |
| `similarity` | Paper similarity (Jaccard) | Finding related papers |

---

## Mode Details

### Semantic (Default)

The full knowledge graph with papers, concepts, findings, methods, and limitations as nodes. Edges represent relationships like SUPPORTS, CONTRADICTS, EXTENDS.

```python
export_to_gexf(CORPUS_PATH, "semantic.gexf", mode="semantic")
```

### Co-Authorship

Network of authors weighted by shared publications:

```python
export_to_gexf(CORPUS_PATH, "authors.gexf", mode="authors")
```

!!! tip "Use Case"
    Identify "invisible colleges"—clusters of researchers who frequently collaborate.

### Concept Co-occurrence

Network of concepts that appear together in papers:

```python
export_to_gexf(CORPUS_PATH, "concepts.gexf", mode="concepts")
```

### River (Temporal Concepts)

Same as `concepts`, but adds a `start` year attribute for dynamic visualizations:

```python
export_to_gexf(CORPUS_PATH, "river.gexf", mode="river")
```

!!! tip "Use Case"
    Create ThemeRiver-style visualizations in Gephi showing concept evolution.

### Paper Similarity

Papers connected by shared concepts (Jaccard similarity):

```python
export_to_gexf(CORPUS_PATH, "similarity.gexf", mode="similarity", threshold=0.3)
```

The `threshold` parameter controls minimum similarity for an edge (0.0–1.0).

---

## CLI Usage

```bash
# Default: Semantic knowledge graph
literature-mapper viz ./my_research -o graph.gexf

# Co-authorship network
literature-mapper viz ./my_research -o authors.gexf --mode authors

# Concept co-occurrence
literature-mapper viz ./my_research -o concepts.gexf --mode concepts

# Paper similarity with threshold
literature-mapper viz ./my_research -o similar.gexf --mode similarity --threshold 0.3
```

---

## Gephi Workflow

1. **Open** the `.gexf` file in Gephi
2. **Run layout** (e.g., ForceAtlas2)
3. **Partition** by node type or community
4. **Size nodes** by degree or citation count
5. **Export** as PNG, SVG, or PDF
