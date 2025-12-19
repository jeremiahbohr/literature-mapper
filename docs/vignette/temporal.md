---
description: Track concept emergence, detect rising and declining topics, and identify revival patterns in your corpus over time.
---

# Temporal Analysis

Track when concepts emerged, detect trends, and identify revivals in your corpus.

---

## Concept Timeline

See when key concepts first appeared and which papers introduced them *within your corpus*:

```python
timeline = mapper.get_concept_timeline(top_n=5)

print(f"{'Concept':<25} {'First':>6} {'Peak':>6} {'Papers':>7}")
print("-" * 50)
for _, row in timeline.iterrows():
    print(f"{row['concept'][:24]:<25} {row['first_year']:>6} {row['peak_year']:>6} {row['total_papers']:>7}")
```

### Tracing a Specific Concept

```python
# Timeline for a specific concept
specific = mapper.get_concept_timeline("homophily")

if not specific.empty:
    for _, row in specific.iterrows():
        print(f"{row['concept']}: First appeared {row['first_year']}")
        print(f"  Peak year: {row['peak_year']}")
        print(f"  Introduced by: {row['introduced_by']}")
```

---

## Trend Detection

After computing temporal statistics, identify rising and declining concepts:

```python
# First, compute temporal stats
mapper.compute_temporal_stats()

# Then view trends
rising = mapper.get_trending_concepts(direction="rising", limit=5)
declining = mapper.get_trending_concepts(direction="declining", limit=5)
```

### How Trends Are Computed

| Metric | Description |
|:-------|:------------|
| **Trend Slope** | Linear regression on papers/year mentioning the concept |
| **Positive slope** | Concept is "Rising" (appearing in more papers over time) |
| **Negative slope** | Concept is "Declining" (appearing in fewer papers over time) |

---

## Era Detection (Revivals)

Detect concepts that disappeared for a period and then reappeared—"revivals":

```python
eras = mapper.detect_concept_eras(gap=5)  # 5-year gap threshold

for _, row in eras.iterrows():
    print(f"{row['concept']}: {row['first_era']} → gap → {row['second_era']}")
```

!!! tip "Use Case"
    Era detection is useful for finding forgotten methods that were later revived, or concepts that experienced a renaissance.

---

## Year-Range Filtering in Search

Scope any search or synthesis to specific time periods:

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

---

## CLI Usage

```bash
# Compute temporal stats (run this first!)
literature-mapper temporal ./my_research

# View rising/declining concepts
literature-mapper trends ./my_research --direction rising
literature-mapper trends ./my_research --direction declining

# Analyze a specific concept's trajectory
literature-mapper trajectory "homophily" ./my_research

# Detect concept eras (revivals after gaps)
literature-mapper eras ./my_research --gap 5
```
