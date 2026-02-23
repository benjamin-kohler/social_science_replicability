# Social Science Replicability System

An automated multi-agent system that replicates social science research papers, verifies the results, and explains any discrepancies — plus a benchmark framework for comparing LLM models on the replication task.

## Overview

This system uses a pipeline of four LLM-powered agents to automatically replicate the analysis in a social science paper:

1. **Extractor** — Parses the paper PDF and extracts methodology, regression specifications, table/figure structures, and data processing steps without revealing actual results.
2. **Replicator** — Generates and executes Python code to replicate the analysis using the provided data and the extracted methodology. Has no access to the paper or its results.
3. **Verifier** — Compares the replicated results against the original paper and assigns grades (A through F) for each table and figure.
4. **Explainer** — Analyzes discrepancies for any non-A grades, identifies likely causes, and attributes fault.

The **benchmark framework** compares multiple LLM models by running both a freestyle approach (opencode CLI, no hand-holding) and a structured approach (LangGraph pipeline) on the same papers, evaluated by a shared judge model.

## Installation

### Prerequisites

- Python 3.11 or higher
- An API key for OpenAI or Anthropic

### Setup

```bash
git clone <repository-url>
cd social_science_replicability

python -m venv venv
source venv/bin/activate

pip install -e ".[dev]"

cp .env.example .env
# Edit .env and add your OPENAI_API_KEY and/or ANTHROPIC_API_KEY
```

## Quick Start

### Single Paper Replication

```bash
# Basic usage (default model: gpt-5.3-codex)
python -m src.main --paper data/input/paper.pdf --data data/input/dataset.csv

# With Anthropic Claude
python -m src.main --paper paper.pdf --data dataset.csv \
    --model-provider anthropic --model-name claude-sonnet-4-20250514

# Extraction only (no replication)
python -m src.main --paper paper.pdf --extraction-only

# With replication package for the explanation step
python -m src.main --paper paper.pdf --data dataset.csv \
    --replication-package original_code/
```

### Model Benchmarking

```bash
# Edit config/benchmark_config.yaml with your models and papers, then:
python -m src.benchmark_cli -c config/benchmark_config.yaml

# Filter to specific approaches, models, or papers
python -m src.benchmark_cli -c config/benchmark_config.yaml \
    --approaches structured --models gpt-5.3-codex

# Custom output directory and timeout
python -m src.benchmark_cli -c config/benchmark_config.yaml \
    -o results/run1 --timeout 900
```

## CLI Reference

### Replication Pipeline

```
python -m src.main [OPTIONS]

Required:
  --paper PATH              Path to the PDF paper to replicate

Optional:
  --data PATH               Path to data files (required unless --extraction-only)
  --replication-package PATH
                            Path to original replication package (used by Explainer)
  --output-dir DIR          Output directory for reports (default: reports)
  --model-provider {openai,anthropic}
                            LLM provider (default: openai)
  --model-name NAME         Model name (default: gpt-5.3-codex)
  --config PATH             Path to YAML configuration file
  --extraction-only         Only extract methodology, skip replication
  --log-level {DEBUG,INFO,WARNING,ERROR}
```

### Benchmark

```
python -m src.benchmark_cli [OPTIONS]

  --config, -c PATH         Benchmark config YAML (default: config/benchmark_config.yaml)
  --approaches              Override approaches: freestyle, structured
  --papers                  Filter to specific paper IDs
  --models                  Filter to specific model names
  --output-dir, -o DIR      Override output directory
  --timeout SECONDS         Override timeout per run
```

## Architecture

### Replication Pipeline

```
┌──────────┐    PaperSummary    ┌────────────┐   ReplicationResults   ┌──────────┐
│ Extractor├───────────────────►│ Replicator ├───────────────────────►│ Verifier │
│ (Agent 1)│                    │ (Agent 2)  │                        │ (Agent 3)│
└────┬─────┘                    └─────┬──────┘                        └────┬─────┘
     │                                │                                    │
     │  Paper PDF                     │  Data + Summary                    │
     │                                │  (NO paper access)                 │
     │                                │                                    ▼
     │                                │                              ┌──────────┐
     └────────────────────────────────┴─────────────────────────────►│ Explainer│
                                                                     │ (Agent 4)│
                                                                     └──────────┘
```

The pipeline is orchestrated with **LangGraph** (`StateGraph` with conditional edges). Each agent uses LangChain chat models (OpenAI or Anthropic) via a common `BaseAgent` interface.

### Benchmark Architecture

```
BenchmarkRunner (iterates models × papers × approaches)
  │
  ├── _extract_summary(paper)      [judge model, cached per paper]
  │         │
  │         ▼
  │     PaperSummary (methodology only — no results, no code)
  │         │
  │         ├── OpencodeRunner (freestyle)  ──→ artifacts ──→ ArtifactParser
  │         │
  │         └── StructuredRunner (pipeline) ──→ run_from_summary()
  │                                                    │
  │                                              SharedEvaluator
  │                                              [judge model]
  │                                                    │
  │                                              ResultsAggregator
  │                                              ──→ summary.json + summary.csv
```

### Information Isolation

The replicator is deliberately **blind to results**. This is enforced at every level:

| What the replicator sees | What it does NOT see |
|---|---|
| Methodology summary (table specs, regression specs, data processing steps) | Original paper PDF |
| The dataset | Replication package / original code |
| | Actual results or numerical outcomes |

For benchmarking, the judge model extracts the methodology **once per paper**, and the same `PaperSummary` is given to all models and approaches — ensuring a level playing field.

### Key Design Decisions

- **Information isolation**: The Replicator never sees the original paper or results, preventing bias.
- **Shared judge**: A fixed judge model (Verifier + Explainer) grades all outputs for cross-model comparability.
- **Cached extraction**: Methodology is extracted once per paper, not once per run.
- **Sandboxed execution**: Generated code runs in isolated Jupyter kernels with timeout protection.
- **Structured outputs**: All inter-agent data uses Pydantic models for validation.
- **Graceful degradation**: If one step fails, the pipeline returns partial results rather than crashing.
- **Vision model support**: Figure comparison can use vision APIs with text-based fallback.

## Benchmark Configuration

```yaml
# config/benchmark_config.yaml
models:
  - provider: openai
    model_name: gpt-5.3-codex
    api_key_env: OPENAI_API_KEY
  - provider: anthropic
    model_name: claude-sonnet-4-20250514
    api_key_env: ANTHROPIC_API_KEY

papers:
  - paper_id: my_paper
    pdf_path: data/input/my_paper/paper.pdf
    data_path: data/input/my_paper/data/dataset.csv
    replication_package_path: data/input/my_paper/replication_package  # optional

approaches:
  - freestyle    # opencode CLI — raw model capability
  - structured   # LangGraph pipeline — guided workflow

judge:
  provider: openai
  model_name: gpt-5.3-codex

output_dir: data/benchmark_results
opencode_binary: opencode
timeout_seconds: 600
```

### Benchmark Output

```
data/benchmark_results/
├── summaries/
│   └── my_paper_summary.json          # Pre-extracted methodology (shared)
├── summary.json                        # Full results
├── summary.csv                         # One row per (model, paper, approach)
├── gpt-5.3-codex_my_paper_freestyle/
│   ├── result.json                     # Detailed run result
│   └── workspace/                      # TASK.md, data, generated code/tables/figures
└── gpt-5.3-codex_my_paper_structured/
    ├── result.json
    └── workspace/
```

## Grading Scale

| Grade | Meaning |
|-------|---------|
| **A** | Fully replicated. Results match within numerical precision (< 1% difference). |
| **B** | Same direction of effects with small discrepancies (1-5% difference). |
| **C** | Same direction of effects with large discrepancies (5-20% difference). |
| **D** | Results differ meaningfully in significance, direction, or magnitude. |
| **F** | Not comparable due to missing output or incompatible format. |

## Project Structure

```
social_science_replicability/
├── src/
│   ├── agents/
│   │   ├── base.py              # Base agent with LLM provider abstraction
│   │   ├── collector.py         # Step 0: Paper & data collection
│   │   ├── extractor.py         # Agent 1: Methodology extraction
│   │   ├── replicator.py        # Agent 2: Code generation and execution
│   │   ├── verifier.py          # Agent 3: Result comparison and grading
│   │   └── explainer.py         # Agent 4: Discrepancy analysis
│   ├── benchmark/
│   │   ├── config.py            # ModelSpec, PaperSpec, JudgeConfig, BenchmarkConfig
│   │   ├── runner.py            # BenchmarkRunner orchestrator
│   │   ├── opencode_runner.py   # Freestyle approach (opencode subprocess)
│   │   ├── structured_runner.py # Structured approach (LangGraph pipeline)
│   │   ├── evaluator.py         # SharedEvaluator (judge model grading)
│   │   ├── artifact_parser.py   # Parse freestyle output into ReplicationResults
│   │   └── results.py           # RunArtifacts, EvaluationResult, ResultsAggregator
│   ├── models/
│   │   ├── schemas.py           # All Pydantic data models
│   │   └── config.py            # Configuration management
│   ├── utils/
│   │   ├── pdf_parser.py        # PDF text and table extraction
│   │   ├── code_executor.py     # Jupyter kernel sandbox
│   │   ├── comparison.py        # Numerical comparison utilities
│   │   └── logging_utils.py     # Structured logging
│   ├── orchestrator.py          # LangGraph pipeline orchestration
│   ├── main.py                  # Replication CLI entry point
│   └── benchmark_cli.py         # Benchmark CLI entry point
├── config/
│   ├── default_config.yaml      # Default pipeline configuration
│   └── benchmark_config.yaml    # Example benchmark configuration
├── docs/
│   └── opencode_vs_langgraph.md # Freestyle vs. structured comparison
├── tests/                       # 206 tests
│   ├── conftest.py              # Shared fixtures
│   ├── test_agents.py           # Agent unit tests
│   ├── test_orchestrator.py     # Pipeline tests
│   ├── test_benchmark.py        # Benchmark framework tests (37 tests)
│   ├── test_schemas.py          # Schema validation tests
│   └── ...
├── data/
│   ├── input/                   # Input papers and datasets
│   └── benchmark_results/       # Benchmark output
├── pyproject.toml
├── requirements.txt
└── .env.example
```

## Running Tests

```bash
pytest tests/ -v                          # All 206 tests
pytest tests/test_benchmark.py -v         # Benchmark tests only (37)
pytest tests/ --cov=src --cov-report=term # With coverage
```

## Configuration

### Pipeline Configuration (`config/default_config.yaml`)

```yaml
langgraph:
  default_provider: "openai"
  default_model: "gpt-5.3-codex"
  temperature: 0.1
  max_tokens: 4000

execution:
  timeout_seconds: 300
  max_retries: 3
  sandbox_type: "jupyter"

verification:
  numerical_tolerance: 0.01      # 1% difference threshold
  use_vision_model: true

output:
  save_intermediate_results: true
  reports_dir: "reports"
  figures_format: "png"
```

### Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key (required for OpenAI models) |
| `ANTHROPIC_API_KEY` | Anthropic API key (required for Anthropic models) |

## License

TBD
