---
description: Find missing papers and authors in your literature review using Ghost Hunting. Identify structural gaps in your corpus.
---

# Gap Detection (Ghost Hunting)

Find frequently-cited papers and authors that are **missing** from your corpus—structural gaps in your literature review.

---

## What is Ghost Hunting?

Ghost hunting uses citation data to identify works that are *referenced* by your corpus but not *included* in it. This reveals:

- **Seminal works** you may have overlooked
- **Key researchers** whose voices are missing
- **Intellectual foundations** of your field

!!! warning "Prerequisite"
    Ghost hunting requires running `update_citations()` first to fetch OpenAlex data.

---

## Ghost Hunting Modes

| Mode | What It Finds |
|:-----|:--------------|
| `bibliographic` | Papers cited by your corpus but not included |
| `authors` | Researchers cited frequently but not represented |

---

## Bibliographic Ghosts

Find missing papers that are frequently cited by your corpus:

```python
from literature_mapper.ghosts import GhostHunter

hunter = GhostHunter(mapper)

# Find papers cited by at least 3 corpus papers
ghosts = hunter.find_bibliographic_ghosts(threshold=3)

if not ghosts.empty:
    print(f"Found {len(ghosts)} missing papers:\n")
    for _, row in ghosts.head(5).iterrows():
        print(f"  [{row['citation_count']:2d}×] {row['title'][:50]}...")
        print(f"       {row['author']} ({row['year']})")
else:
    print("No gaps found (run update_citations() first)")
```

### Interpreting Results

| Column | Meaning |
|:-------|:--------|
| `citation_count` | Number of corpus papers citing this work |
| `title` | Title of the missing paper |
| `author` | First author or author list |
| `year` | Publication year |

!!! tip "Actionable Gap Analysis"
    Papers with high citation counts (cited by many corpus papers) are strong candidates to add to your literature review.

---

## Missing Authors

Find researchers who are frequently cited but don't have papers in your corpus:

```python
# Find authors cited by at least 5 papers
missing = hunter.find_missing_authors(threshold=5)

if not missing.empty:
    print("Frequently cited authors not in corpus:\n")
    for _, row in missing.head(5).iterrows():
        print(f"  {row['author']} (cited by {row['cited_by_papers']} papers)")
```

---

## CLI Usage

```bash
# Find missing papers (threshold: cited by at least 3 corpus papers)
literature-mapper ghosts ./my_research --mode bibliographic --threshold 3

# Find missing authors (threshold: cited by at least 5 corpus papers)  
literature-mapper ghosts ./my_research --mode authors --threshold 5
```
