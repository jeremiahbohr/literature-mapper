---
description: Use AI synthesis agents to answer research questions and validate hypotheses with full citation provenance from your corpus.
---

# AI Synthesis Agents

Get corpus-grounded answers with citations; validate hypotheses against evidence.

---

## Overview

Literature Mapper includes two specialized RAG (Retrieval-Augmented Generation) agents:

| Agent | Purpose | Method |
|:------|:--------|:-------|
| **Argument Agent** | Answer research questions | `synthesize_answer()` |
| **Validation Agent** | Test claims against evidence | `validate_hypothesis()` |

!!! success "Key Feature: Bounded Synthesis"
    These agents synthesize answers using **only your corpus**—no hallucinated citations, no external knowledge bleed. Every claim is grounded in papers you've processed.

---

## Research Synthesis

Ask a research question and get a comprehensive, cited answer:

```python
import textwrap

question = "What factors influence information spread in social networks?"

answer = mapper.synthesize_answer(question, limit=15)
print(textwrap.fill(answer, width=100))
```

### How It Works

1. Retrieves relevant KG nodes using semantic search
2. Builds context from findings, methods, and limitations
3. Generates a synthesis with inline citations `[Author et al., Year]`
4. All citations reference papers **in your corpus**

---

## Hypothesis Validation

Test a claim against your corpus evidence:

```python
hypothesis = "Strong social ties are more effective than weak ties for spreading novel information."

result = mapper.validate_hypothesis(hypothesis)

print(f"Verdict: {result['verdict']}")
print(textwrap.fill(result['explanation'], width=100))
```

### Verdict Types

| Verdict | Meaning |
|:--------|:--------|
| `SUPPORTED` | Corpus evidence aligns with the hypothesis |
| `CONTRADICTED` | Corpus evidence opposes the hypothesis |
| `MIXED` | Both supporting and contradicting evidence exists |
| `NOVEL` | No relevant evidence found (may be original or out of scope) |

---

## Temporal Scoping

Compare how perspectives changed over time:

```python
question = "What are the main debates about network structure and information diffusion?"

# Early period
early = mapper.synthesize_answer(question, year_range=(1998, 2008), limit=10)
print("=== EARLY PERIOD (1998-2008) ===")
print(textwrap.fill(early, width=100))

# Recent period
recent = mapper.synthesize_answer(question, year_range=(2015, 2024), limit=10)
print("\n=== RECENT PERIOD (2015-2024) ===")
print(textwrap.fill(recent, width=100))
```

---

## Corpus Boundary Enforcement

The agent refuses to answer questions outside the corpus domain:

```python
# Test with an irrelevant query
off_topic = mapper.synthesize_answer("How does soil salinity affect crop yield?")
# Returns: "Based on the provided literature corpus, there is currently 
#           no relevant evidence available..."
```

```python
# Test an irrelevant hypothesis
result = mapper.validate_hypothesis(
    "Glacier retreat in the Himalayas accelerates groundwater depletion."
)
print(result['verdict'])  # "NOVEL"
```

!!! info "Why This Matters"
    Bounded synthesis prevents the common RAG failure mode where LLMs confidently cite papers that don't exist or make claims unsupported by your actual sources.

---

## CLI Usage

```bash
# Synthesize an answer
literature-mapper synthesize ./my_research "What is the impact of X on Y?"

# Validate a hypothesis
literature-mapper validate ./my_research "X causes Y."
```
