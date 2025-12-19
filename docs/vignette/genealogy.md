---
description: Build intellectual genealogy graphs showing how papers relate. Find contradictions, extensions, and argument evolution.
---

# Intellectual Genealogy

Build a graph of how arguments relate—extensions, challenges, and syntheses across papers.

---

## What is Intellectual Genealogy?

Intellectual genealogy maps how papers relate to each other beyond simple citations:

| Relation | Meaning |
|:---------|:--------|
| `EXTENDS` | Builds directly on prior work |
| `CHALLENGES` | Disputes or critiques findings |
| `BUILDS_ON` | Uses as theoretical foundation |
| `SYNTHESIZES` | Combines multiple perspectives |

---

## Building Genealogy

This makes LLM calls to infer relationships between papers. Results persist to the database.

```python
# Build once; results are stored in intellectual_edges table
result = mapper.build_genealogy(verbose=True)

print(f"Analyzed: {result['analyzed']} papers")
print(f"Relationships found: {result['relationships']}")
```

!!! warning "Time Estimate"
    Approximately **20-30 seconds per paper**. For a 50-paper corpus, expect ~25 minutes.

---

## Finding Contradictions

Identify intellectual debates—papers that challenge each other:

```python
import textwrap

debates = mapper.find_contradictions()

if not debates.empty:
    print("INTELLECTUAL DEBATES\n")
    
    for _, row in debates.head(3).iterrows():
        print(f"┌─ {row['paper_a'][:60]}")
        print(f"│     ↓ CHALLENGES")
        print(f"└─ {row['paper_b'][:60]}")
        print(f"\n   Confidence: {row['confidence']:.0%}")
        
        if row.get('evidence'):
            print(f"\n   Evidence:")
            for line in textwrap.wrap(row['evidence'], width=70):
                print(f"   {line}")
        print()
```

---

## Argument Evolution

Trace how a concept's treatment evolved over time:

```python
evolution = mapper.get_argument_evolution(concept="weak ties")

if not evolution.empty:
    for _, row in evolution.head(5).iterrows():
        print(f"{row['source_year']} {row['source_title'][:40]}...")
        print(f"    └─ {row['relation']} → {row['target_title'][:40]}... ({row['target_year']})\n")
```

### Example Output

```
1985 Economic Action and Social Structur...
    └─ BUILDS_ON → Social Resources and Strength of Ti... (1981)

2004 Structural Holes and Good Ideas...
    └─ EXTENDS → Social Resources and Strength of Ti... (1981)
```

---

## Filtering by Concept

Find debates specifically about a concept:

```python
# Find contradictions related to "homophily"
homophily_debates = mapper.find_contradictions(concept="homophily")
```

---

## Use Cases

| Goal | Method |
|:-----|:-------|
| Find debates in the field | `find_contradictions()` |
| Trace idea development | `get_argument_evolution(concept)` |
| Map theoretical lineages | Export genealogy edges to Gephi |
| Identify synthesis works | Filter for `SYNTHESIZES` relations |
