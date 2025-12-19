---
description: Enrich your corpus with OpenAlex citation data. Identify influential hub papers and prolific authors.
---

# Citation & Influence Analysis

Enrich your corpus with citation data from OpenAlex and identify influential papers.

---

## OpenAlex Enrichment

Fetch citation counts and reference lists for each paper:

```python
# Fetch/update citation data
# Optionally provide email for faster rate limits
mapper.update_citations()  # email="you@example.com"
```

??? example "CLI Equivalent"
    ```bash
    literature-mapper citations ./my_research --email you@example.com
    ```

!!! info "About OpenAlex"
    OpenAlex is a free, open catalog of scholarly works. Literature Mapper matches your papers by DOI first, then by title. No API key required, but we respect rate limits.

---

## Citation Metrics

| Metric | Description | Use |
|:-------|:------------|:----|
| `citation_count` | Total citations | Overall influence |
| `citations_per_year` | Citations ÷ years since publication | Rising influence (controls for age) |

### Most Cited Papers

```python
papers_df = mapper.get_all_analyses()
papers_with_cites = papers_df[papers_df['citation_count'].notna()]

# Top by raw citations
print("Most Cited (raw):")
for _, row in papers_with_cites.nlargest(3, "citation_count").iterrows():
    print(f"  {row['citation_count']:,} citations")
    print(f"  {row['authors']} ({row['year']})")
    print(f"  {row['title'][:60]}...\n")

# Top by citations per year (normalized)
print("Most Cited (normalized):")
for _, row in papers_with_cites.nlargest(3, "citations_per_year").iterrows():
    print(f"  {row['citations_per_year']:.1f} citations/year")
    print(f"  {row['authors']} ({row['year']})")
    print(f"  {row['title'][:60]}...\n")
```

---

## Top Authors & Concepts

Identify the most prolific researchers and frequent themes:

```python
from literature_mapper.analysis import CorpusAnalyzer

analyzer = CorpusAnalyzer(CORPUS_PATH)

print("Prolific Authors:")
for _, row in analyzer.get_top_authors(limit=5).iterrows():
    print(f"  {row['paper_count']}× {row['author']}")

print("\nFrequent Concepts:")
for _, row in analyzer.get_top_concepts(limit=5).iterrows():
    print(f"  {row['paper_count']}× {row['concept']}")
```

---

## Finding Hub Papers

Hub papers are those most frequently cited by other papers *within your corpus*:

```python
hubs = analyzer.find_hub_papers(limit=5)

for _, row in hubs.iterrows():
    print(f"  {row['title'][:50]}...")
    print(f"    Referenced by {row['internal_citations']} corpus papers")
```

??? example "CLI Equivalent"
    ```bash
    literature-mapper hubs ./my_research --limit 10
    ```
