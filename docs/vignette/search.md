---
description: Search your corpus with keyword and semantic search. Learn enhanced retrieval with MMR reranking and consensus detection.
---

# Search & Retrieval

Find evidence in your corpus using keyword matching or semantic search by meaning.

---

## Keyword Search

Case-insensitive substring match on specific columns:

```python
# Find papers using survey methodology
survey_papers = mapper.search_papers(column='methodology', query='survey')

print(f"Found {len(survey_papers)} papers using survey methodology:")
for _, row in survey_papers.head(3).iterrows():
    print(f"  • {row['title'][:60]}... ({row['year']})")
```

---

## Semantic Search

Semantic search uses embedding vectors to find content by **meaning**, not exact keywords.

```python
results = mapper.search_corpus(
    query="influence of social ties on information diffusion",
    semantic=True,
    limit=5
)

for r in results:
    print(f"[{r['match_score']:.3f}] {r['title']} ({r['year']})")
```

!!! tip "When to Use Semantic Search"
    Use semantic search when you're looking for conceptually related content that may not use your exact terminology. For example, searching for "network effects" might find papers discussing "social contagion" or "peer influence."

---

## Enhanced Retrieval

Enhanced mode adds two powerful features:

| Feature | Description |
|:--------|:------------|
| **MMR Reranking** | Maximal Marginal Relevance ensures diverse results, not just the most similar |
| **Consensus Grouping** | Identifies when multiple papers make the same claim |

```python
enhanced_results = mapper.search_corpus(
    query="social contagion vs homophily",
    semantic=True,
    use_enhanced=True,
    node_types=["finding", "limitation", "method", "hypothesis"],
    limit=5
)

for r in enhanced_results:
    print(f"[{r['match_score']:.2f}] {r['title']}...")
    print(f"  Type: {r['node_type']}")
    print(f"  Context: {r['match_context'][:100]}...\n")
```

### Filtering by Node Type

Control which knowledge graph node types appear in results:

| Node Type | What It Captures |
|:----------|:-----------------|
| `paper` | Paper titles and abstracts |
| `finding` | Key results and claims |
| `method` | Research methodologies |
| `limitation` | Acknowledged weaknesses |
| `hypothesis` | Theoretical propositions |

---

## Year-Range Filtering

Scope any search to specific time periods:

```python
# Only retrieve evidence from 2010-2020
recent_results = mapper.search_corpus(
    query="network centrality",
    semantic=True,
    min_year=2010,
    max_year=2020,
    limit=5
)

for r in recent_results:
    print(f"{r['year']}: {r['title']}")
```

This is especially useful for:

- Comparing historical vs. recent perspectives
- Focusing on foundational works only
- Analyzing how discourse changed over time
