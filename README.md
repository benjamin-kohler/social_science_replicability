# Read the Paper, Write the Code: Agentic Reproduction of Social-Science Results

Code accompanying the paper *"Read the Paper, Write the Code: Agentic
Reproduction of Social-Science Results"*.

**Authors**

- Benjamin Kohler, David Zollikofer, Johanna Einsiedler, Alexander Hoyle, Elliott Ash


[Read the paper](https://elliottash.com/wp-content/uploads/2026/04/Kohler-Zollikofer-Einsiedler-Hoyle-Ash-Read-Paper-Write-Code-Agentic-Reproduction-Social-Science-Results.pdf)






## Repository layout

```
src/                   Benchmark engine (one Python package)
  benchmark/             Runners: claude-code, codex, opencode, swe-agent
                         Judge, comparator, grader, explainer
  agents/extractor.py    Methodology extractor (paper PDF -> PaperSummary)
  models/                Pydantic schemas for inter-stage data
  utils/                 PDF parsing, sandboxed code execution, logging
  benchmark_cli.py       Entry point: python -m src.benchmark_cli

scripts/               Setup, batch runners, analysis
  audit_replication_data_v2.py   GPT audit of which files are raw data
  setup_i4rep_batch.py           Build the i4Replicate paper directories
  setup_postcutoff_batch.py      Same, for the post-cutoff papers
  run_i4rep_batch.sh             Run the benchmark on all i4R papers
  run_postcutoff_batch.sh        Same, post-cutoff
  analyze_i4rep_results.py       Generate every paper figure/table
  classify_table_categories.py   Table-category classifier used in analysis

  error_analysis/                Cross-paper error attribution pipeline
                                 (00_prep -> 01_trace -> 02_detect -> 03/04)
  validate_guardrails.py         Static guardrail / web-access audit
  classify_guardrail_breaches.py LLM-based guardrail + hardcoding classifier
  analyze_tool_usage.py          Tool-call composition across runs
  plot_guardrail_audits.py       Generate guardrail/hardcoding plots

config/                opencode.json (provider config for opencode CLI)
tests/                 Unit tests for parsers, schemas, executor
```

## Run the pipeline on your own

The benchmark also works on a single paper.

### 1. Lay out the paper directory

Create one directory per paper inside `data/mypapers/papers/<paper_id>/`:

```
data/mypapers/papers/my_paper/
├── paper.pdf                  # The published PDF
├── data/                      # Raw input data only — *not* pre-computed
│                              # tables, regression output, or final results.
│                              # Anything in here is what the agent gets to
│                              # see; everything else is hidden.
└── replication_package/       # (Optional) Original code + full package.
                               # Used only for the post-hoc explainer
                               # comparing agent code to the authors' code.
                               # The replicator never sees this.
```

If you start from a downloaded replication package (openICPSR, Dataverse,
Zenodo, or any zip / directory), run the GPT audit on it directly. The
audit accepts a full path to a single package — zip or directory — and
classifies every file as raw-data, intermediary, final, code, results,
or support, then identifies which files the replicator actually needs
as inputs:

```bash
python scripts/audit_replication_data_v2.py \
    --package /path/to/your_package.zip \
    --paper-id my_paper
# Writes data/audit_replication_data_v2.json (or wherever --output-dir points).
```

You can then consume that JSON when laying out
`data/mypapers/papers/my_paper/` — copy the files listed under
`replication_data_paths` into `data/`, and the rest into
`replication_package/`.

### 2. Configure API keys and install the CLI agents

```bash
cp .env.example .env
# Edit .env: OPENAI_API_KEY, ANTHROPIC_API_KEY (and optionally OPENROUTER_API_KEY).
```

The benchmark drives external CLI agents as subprocesses — install whichever
you want to compare:

| Agent | Install | Models |
|---|---|---|
| Claude Code | `npm i -g @anthropic-ai/claude-code` | `claude-opus-4-6`, `claude-sonnet-4-6`, … |
| OpenAI Codex | `npm i -g @openai/codex` | `gpt-5.4`, `gpt-5.3-codex`, … |
| opencode | <https://opencode.ai/install> | any provider in `config/opencode.json` |
| mini-SWE-agent | `pip install mini-swe-agent` | any chat model |

### 3. Write a one-paper config YAML

Save as `config/my_paper.yaml`:

```yaml
models:
  - provider: anthropic
    model_name: claude-opus-4-6
    api_key_env: ANTHROPIC_API_KEY
    approaches: [claude-code]

  - provider: openai
    model_name: gpt-5.4
    api_key_env: OPENAI_API_KEY
    approaches: [codex, opencode]

papers:
  - paper_id: my_paper
    pdf_path: data/mypapers/papers/my_paper/paper.pdf
    data_path: data/mypapers/papers/my_paper/data
    replication_package_path: data/mypapers/papers/my_paper/replication_package

approaches: [claude-code, codex, opencode]

judge:
  provider: openai
  model_name: gpt-5-mini
  use_vision: true
  comparator_cli_tool: claude-code
  comparator_model: claude-sonnet-4-6

extractor:
  model: gpt-5-mini
  use_vision: true

output_dir: data/mypapers/results
timeout_seconds: 7200
allow_web_access: false
item_types: [table]
```

### 4. Run the benchmark

```bash
python -m src.benchmark_cli --config config/my_paper.yaml
```

Each `(model × approach)` combination produces an isolated workspace.
Workspace contents:

```
data/mypapers/results/my_paper/<model>_my_paper_<approach>/
├── workspace/                 # What the agent saw — TASK.md, data, agent code
├── methodology_summary.json   # PaperSummary the extractor produced (shared)
└── verification_report.json   # Cell-level grades A–F + numerical comparisons
```

`verification_report.json` is the headline output — overall grade, item
grades, and a per-cell breakdown. The agent's generated Python code lives
in `workspace/*.py`.

### 5. Run the explainer pipeline

The **explainer pipeline** in `scripts/error_analysis/` runs four
cross-checks per failing cell — diffing agent code against the authors'
original Stata/R/Python code to attribute each error to one of:

- *Data missing* — the required input wasn't in `data/`
- *Original code error* — the authors' code itself can't reproduce the value
- *Paper-vs-code* — the paper text disagrees with the original code
- *Agent error* — the agent diverged where the original code is correct

Run all four stages on your results directory:

```bash
PROJECT_ROOT=$(pwd) \
PAPERS_DIR=data/mypapers/papers \
RESULTS_DIR=data/mypapers/results \
bash scripts/error_analysis/run_pipeline_all.sh
```

This drives, in order:

1. `00_prep_setup.py` — copies each paper × agent run into a clean
   `explainer_workspaces/<paper>/<agent>/` layout (agent code on one
   side, original replication package on the other).
2. `01_trace_failures.py` — for each failing cell, asks an LLM to
   identify the specific lines in agent code and original code that
   produce the divergent value.
3. `02_detect_error_source.py` — runs targeted consistency checks
   (paper vs. summary, summary vs. agent, paper vs. original code,
   data availability) to attribute a root cause.
4. `03_summarize_errors.py` — emits a per-paper LaTeX table of all
   divergences with their attributed root causes.
5. `04_overview_stats.py` — aggregates across all papers and agents
   into the cross-paper plots (`root_causes_*`, `divergence_types_*`)
   reported in the paper.

Outputs land in `scripts/error_analysis/explainer_workspaces/`,
`scripts/error_analysis/summaries/` (LaTeX), and
`scripts/error_analysis/plots/` (PDFs + PNGs). The pipeline is
incremental — re-running skips steps already completed for unchanged
inputs.
