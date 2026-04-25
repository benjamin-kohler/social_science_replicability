# LLM Replicability Evaluation Pipeline

This pipeline audits how accurately LLM-generated replication code reproduces
the results of published economics papers. It takes a set of papers, their
original Stata replication packages, and Python code produced by one or more
LLM agents, then systematically identifies where and why the agent code fails.

---

## Overview

```
examples/
  papers/{doi}/          Original replication packages (Stata code + data)
  results/{doi}/
    {model}_{doi}_{runner}/
      verification_report.json   Cell-level numerical comparison of agent vs. original
      workspace/
        *.py                     Agent-generated Python code
        methodology_summary.json Agent's structured description of the paper's methods

scripts/error_analysis/
  00_prep_setup.py       Assembles explainer_workspaces/ from examples/
  01_trace_failures.py   Step 01: traces output failures to code discrepancies
  02_detect_error_source.py  Step 02: runs consistency checks to assign root causes
  03_summarize_errors.py     Step 03: produces per-paper LaTeX divergence tables
  04_overview_stats.py       Step 04: produces summary plots across all papers/agents
  run_pipeline.sh        Runs all four steps for all papers and agents
  explainer_workspaces/  Working directories created by 00_prep_setup.py
  summaries/             Output: per-paper LaTeX tables (step 03)
  plots/                 Output: summary figures (step 04)
  taxonomy_tables.tex    LaTeX tables defining verdict categories and S-codes
  pipeline_figure.tex    TikZ figure illustrating the pipeline
```

---

## Pipeline steps

### Step 00 — Workspace preparation (`00_prep_setup.py`)

Assembles a clean working directory for each paper × agent combination by
copying relevant files from `examples/` into `explainer_workspaces/`:

```
explainer_workspaces/{doi}/{agent}/
  code/
    agent_code/          Agent's .py files
    original_code/       Full Stata replication package (code + data)
  error_source/
    paper_vs_original_code/   paper.pdf + Stata .do files
    paper_vs_summary/         paper.pdf + methodology_summary.json
    summary_vs_agent/         methodology_summary.json + agent .py files
```

```bash
python3 00_prep_setup.py \
  [--papers-dir  ../examples/papers]   \
  [--results-dir ../examples/results]  \
  [--output-dir  ./explainer_workspaces]
```

### Step 01 — Output-grounded failure tracing (`01_trace_failures.py`)

Starting from the cell-level numerical failures in `verification_report.json`,
an LLM agent traces each failed output back to the specific code discrepancy
that caused it. For each discrepancy the agent:

- checks whether the required data files are present in the replication package
- locates the relevant code in both the original Stata files and the agent Python code
- classifies the discrepancy by failure type (S1–S9) and severity
- identifies which other outputs the same discrepancy also explains (`also_explains`)

Outputs `divergences.json` in `{agent}/code/`.

```bash
python3 01_trace_failures.py \
  --code-dir            explainer_workspaces/{doi}/{agent}/code \
  --verification-report ../examples/results/{doi}/{run}/verification_report.json \
  --runner              claude-code|codex \
  [--rerun]
```

### Step 02 — Error source diagnosis (`02_detect_error_source.py`)

For each divergence from step 01, an LLM auditor runs three pairwise
consistency checks across the document chain:

| Check | Documents compared |
|---|---|
| Paper ↔ Code | Does the paper's stated methodology match what the Stata code does? |
| Paper ↔ Summary | Does the methodology summary faithfully represent the paper? |
| Summary ↔ Agent | Does the agent code follow the methodology summary? |

Each check yields one of four verdicts: **consistent** ✓, **contradicts** ✗,
**omission** ○, or **unclear** ?. The first non-consistent check identifies the
root cause:

| Root cause | Trigger |
|---|---|
| Data not in package | Required data absent (from step 01) |
| Paper underspecified | Paper ↔ Code inconsistent |
| Summary gap | Paper ↔ Summary inconsistent |
| Agent ignored instructions | Summary ↔ Agent inconsistent |
| Unexplained | All checks consistent |

Divergences already marked `data_available=missing` in step 01 skip all checks.
Code proofs (`original_proof`, `agent_proof`) from step 01 are embedded directly
in the prompts so agents only need to read `paper.pdf` or `methodology_summary.json`.

Outputs `divergences_enriched.json` in `{agent}/error_source/`.

```bash
python3 02_detect_error_source.py \
  --comparison explainer_workspaces/{doi}/{agent}/code/divergences.json \
  --workspace  explainer_workspaces/{doi}/{agent}/error_source \
  --output     explainer_workspaces/{doi}/{agent}/error_source/divergences_enriched.json \
  --runner     claude-code|codex \
  [--rerun]
```

### Step 03 — LaTeX table generation (`03_summarize_errors.py`)

Auto-discovers all `divergences_enriched.json` files and produces one LaTeX
`longtable` per paper listing every divergence with its failure type, severity,
and root cause.

```bash
python3 03_summarize_errors.py --output-dir summaries/ [--rerun]
```

### Step 04 — Summary plots (`04_overview_stats.py`)

Produces five figure types across all papers and agents:

- **Root-cause bar charts** — divergence counts by root cause, per agent
- **Pipeline cascade** — how divergences are attributable at each pipeline stage
- **Stage heatmap** — verdict patterns across papers
- **Failure-type bar chart** — S-code distribution
- **Divergence–output network** — bipartite graph linking each code discrepancy
  to the tables/figures it affects; nodes coloured by root cause when enriched
  data is available

```bash
python3 04_overview_stats.py [--rerun]
```

---

## Running the full pipeline

```bash
cd scripts/error_analysis

# Prepare workspaces (once)
python3 00_prep_setup.py

# Run steps 01–04 for all papers and agents
bash run_pipeline.sh

# Pass API keys explicitly if not set in environment
bash run_pipeline.sh \
  --anthropic-api-key sk-ant-... \
  --openai-api-key    sk-...

# Resume from a specific step (e.g. after step 01 is done)
bash run_pipeline.sh --from 2
```

Codex does not read `OPENAI_API_KEY` from the environment — run `codex login`
once interactively to cache credentials before using `--runner codex`.

---

## Failure type taxonomy (S-codes)

| Code | Category |
|---|---|
| S1 | Wrong model specification (fixed effects, clustering) |
| S2 | Wrong estimator or inference procedure |
| S3 | Data source substitution (proxy used instead of required source) |
| S4 | Wrong sample restriction |
| S5 | Wrong variable construction |
| S6 | Missing analysis component |
| S8 | Wrong data transformation or merge logic |
| S9 | Wrong sequencing / order of operations |
| S0 | Other |

Full definitions with examples are in `taxonomy_tables.tex`.

---

## Requirements

- Python 3.11+
- `matplotlib`, `numpy`, `pandas` (for step 04)
- `claude-code` CLI (for `--runner claude-code`)
- `codex` CLI with cached credentials (for `--runner codex`)
