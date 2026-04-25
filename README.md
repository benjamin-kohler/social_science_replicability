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
