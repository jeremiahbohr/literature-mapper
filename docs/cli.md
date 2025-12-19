---
description: Complete command-line interface reference for Literature Mapper. Process PDFs, analyze corpora, and export data from the terminal.
hide:
  - navigation
---

# CLI Reference

Literature Mapper provides a complete command-line interface for managing your research corpus.

---

## Core Workflow

### Process PDFs

Extract text and build the Knowledge Graph:

```bash
literature-mapper process ./my_research --recursive
```

| Option | Description |
|:-------|:------------|
| `--recursive` | Process PDFs in subdirectories |

### Fetch Citations

Enrich your corpus with OpenAlex citation data:

```bash
literature-mapper citations ./my_research --email you@example.com
```

| Option | Description |
|:-------|:------------|
| `--email` | Your email (optional, for faster rate limits) |

### View Status

Check corpus health and statistics:

```bash
literature-mapper status ./my_research
```

---

## Search & Analysis

### Synthesize Answers

Ask a research question:

```bash
literature-mapper synthesize ./my_research "What is the impact of X on Y?"
```

### Validate Hypotheses

Test a claim against your corpus:

```bash
literature-mapper validate ./my_research "X causes Y."
```

### Find Hub Papers

Identify the most referenced papers:

```bash
literature-mapper hubs ./my_research --limit 10
```

### Corpus Statistics

View comprehensive stats:

```bash
literature-mapper stats ./my_research
```

---

## Temporal Analysis

!!! info "Run `temporal` first"
    Temporal commands require computing stats first with `literature-mapper temporal`.

```bash
# Compute temporal stats (run this first!)
literature-mapper temporal ./my_research

# View rising concepts
literature-mapper trends ./my_research --direction rising

# View declining concepts
literature-mapper trends ./my_research --direction declining

# Analyze a specific concept's trajectory
literature-mapper trajectory "homophily" ./my_research

# Detect concept eras (revivals after gaps)
literature-mapper eras ./my_research --gap 5
```

---

## Gap Detection (Ghost Hunting)

Find missing papers and authors:

```bash
# Find missing papers (bibliographic ghosts)
literature-mapper ghosts ./my_research --mode bibliographic --threshold 3

# Find missing authors
literature-mapper ghosts ./my_research --mode authors --threshold 5
```

| Option | Description |
|:-------|:------------|
| `--mode` | `bibliographic` (papers) or `authors` |
| `--threshold` | Minimum citations to be considered a "ghost" |

---

## Export & Visualization

### GEXF Export

```bash
# Semantic knowledge graph (default)
literature-mapper viz ./my_research -o graph.gexf

# Co-authorship network
literature-mapper viz ./my_research -o authors.gexf --mode authors

# Concept co-occurrence
literature-mapper viz ./my_research -o concepts.gexf --mode concepts

# Paper similarity
literature-mapper viz ./my_research -o similar.gexf --mode similarity --threshold 0.3
```

| Mode | Description |
|:-----|:------------|
| `semantic` | Full knowledge graph |
| `authors` | Co-authorship network |
| `concepts` | Topic co-occurrence |
| `river` | Concepts with temporal data |
| `similarity` | Paper similarity (Jaccard) |

### CSV Export

```bash
literature-mapper export ./my_research output.csv
```

---

## Environment Variables

| Variable | Purpose | Default |
|:---------|:--------|:--------|
| `GEMINI_API_KEY` | **Required.** Google AI key | None |
| `LITERATURE_MAPPER_MODEL` | Default model | `gemini-3-flash-preview` |
| `LITERATURE_MAPPER_MAX_FILE_SIZE` | Max PDF size (bytes) | `52428800` (50 MB) |
| `LITERATURE_MAPPER_BATCH_SIZE` | PDFs per batch | `10` |
| `LITERATURE_MAPPER_LOG_LEVEL` | Log level | `INFO` |
| `LITERATURE_MAPPER_VERBOSE` | Debug mode | `false` |
