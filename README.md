# Social Science Replicability System

An automated multi-agent system that replicates social science research papers, verifies the results, and explains any discrepancies.

## Overview

This system uses a pipeline of four LLM-powered agents to automatically replicate the analysis in a social science paper:

1. **Extractor** - Parses the paper PDF and extracts methodology, regression specifications, table/figure structures, and data processing steps without revealing actual results.
2. **Replicator** - Generates and executes Python code to replicate the analysis using the provided data and the extracted methodology.
3. **Verifier** - Compares the replicated results against the original paper and assigns grades (A through F) for each table and figure.
4. **Explainer** - Analyzes discrepancies for any non-A grades, identifies likely causes, and attributes fault.

All runs produce a structured JSON report with the exact configuration and results.

## Installation

### Prerequisites

- Python 3.11 or higher
- An API key for OpenAI or Anthropic

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd social_science_replicability

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy the environment template and add your API key
cp .env.example src/.env
# Edit src/.env and add your OPENAI_API_KEY or ANTHROPIC_API_KEY
```

## Quick Start

```bash
# Basic usage with OpenAI (default)
python -m src.main --paper data/input/paper.pdf --data data/input/dataset.csv

# With Anthropic Claude
python -m src.main --paper paper.pdf --data dataset.csv \
    --model-provider anthropic --model-name claude-3-opus-20240229

# Extraction only (no replication, useful for testing)
python -m src.main --paper paper.pdf --extraction-only

# With original replication package for the explanation step
python -m src.main --paper paper.pdf --data dataset.csv \
    --replication-package original_code/
```

## CLI Reference

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
  --model-name NAME         Model name (default: gpt-4-turbo-preview)
  --config PATH             Path to YAML configuration file
  --extraction-only         Only extract methodology, skip replication
  --log-level {DEBUG,INFO,WARNING,ERROR}
                            Logging level (default: INFO)
```

## Configuration

Configuration is managed through a YAML file at `config/default_config.yaml` and environment variables.

### YAML Configuration

```yaml
open_agent:
  default_provider: "openai"       # or "anthropic"
  default_model: "gpt-4-turbo-preview"
  temperature: 0.1
  max_tokens: 4000

execution:
  timeout_seconds: 300             # Max time for code execution
  max_retries: 3
  sandbox_type: "jupyter"

extraction:
  focus_sections:
    - "Methods"
    - "Results"
    - "Data"
  extract_appendix: false

verification:
  numerical_tolerance: 0.01        # 1% difference threshold
  use_vision_model: true           # Use vision API for figure comparison

output:
  save_intermediate_results: true
  reports_dir: "reports"
  figures_format: "png"
```

### Environment Variables

Set these in `src/.env` or export them in your shell:

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key (required when using OpenAI provider) |
| `ANTHROPIC_API_KEY` | Anthropic API key (required when using Anthropic provider) |
| `LOG_LEVEL` | Logging level override (default: INFO) |

## Architecture

```
┌──────────┐    PaperSummary    ┌────────────┐   ReplicationResults   ┌──────────┐
│ Extractor├───────────────────►│ Replicator ├───────────────────────►│ Verifier │
│ (Agent 1)│                    │ (Agent 2)  │                        │ (Agent 3)│
└────┬─────┘                    └─────┬──────┘                        └────┬─────┘
     │                                │                                    │
     │  Paper PDF                     │  Data + Summary                    │  VerificationReport
     │                                │  (no paper access)                 │
     │                                │                                    ▼
     │                                │                              ┌──────────┐
     └────────────────────────────────┴─────────────────────────────►│ Explainer│
                                                                     │ (Agent 4)│
                                                                     └────┬─────┘
                                                                          │
                                                                          ▼
                                                                   ExplanationReport
```

The **Orchestrator** manages the sequential execution of all four agents, passing state between them and saving intermediate results. Each agent uses an LLM (OpenAI or Anthropic) via a common `BaseAgent` interface, making it straightforward to switch providers.

### Key Design Decisions

- **Information isolation**: The Replicator never sees the original paper or results, preventing bias.
- **Sandboxed execution**: Generated code runs in isolated Jupyter kernels with timeout protection.
- **Structured outputs**: All inter-agent data uses Pydantic models for validation.
- **Graceful degradation**: If one step fails, the pipeline returns partial results rather than crashing.
- **Vision model support**: Figure comparison can use vision APIs (OpenAI, Anthropic) for visual comparison, falling back to text-based assessment.

## Grading Scale

| Grade | Meaning |
|-------|---------|
| **A** | Fully replicated. Results match within numerical precision (< 1% difference). |
| **B** | Same direction of effects with small discrepancies (1-5% difference). |
| **C** | Same direction of effects with large discrepancies (5-20% difference). |
| **D** | Results differ meaningfully in significance, direction, or magnitude. |
| **F** | Not comparable due to missing output or incompatible format. |

## Example Output

After a run, the system produces a JSON report in the output directory:

```json
{
  "timestamp": "2025-01-15T14:30:00",
  "configuration": {
    "model_provider": "openai",
    "model_name": "gpt-4-turbo-preview",
    "paper_path": "data/input/paper.pdf",
    "data_path": "data/input/dataset.csv"
  },
  "status": "complete",
  "results": {
    "paper_summary": { "...extracted methodology..." },
    "verification_report": {
      "overall_grade": "B",
      "item_verifications": [
        { "item_id": "Table 1", "grade": "A", "comparison_notes": "..." },
        { "item_id": "Table 2", "grade": "B", "comparison_notes": "..." },
        { "item_id": "Figure 1", "grade": "B", "comparison_notes": "..." }
      ]
    },
    "explanation_report": {
      "analyses": [
        {
          "item_id": "Table 2",
          "grade": "B",
          "description_of_discrepancy": "...",
          "fault_attribution": "replicator",
          "is_identifiable": true
        }
      ]
    }
  },
  "errors": [],
  "warnings": []
}
```

Intermediate results (paper summary, replication code as a Jupyter notebook, verification report, explanation report) are also saved when `save_intermediate_results` is enabled.

## Project Structure

```
social_science_replicability/
├── src/
│   ├── agents/
│   │   ├── base.py           # Base agent with LLM provider abstraction
│   │   ├── extractor.py      # Agent 1: Methodology extraction
│   │   ├── replicator.py     # Agent 2: Code generation and execution
│   │   ├── verifier.py       # Agent 3: Result comparison and grading
│   │   └── explainer.py      # Agent 4: Discrepancy analysis
│   ├── models/
│   │   ├── schemas.py        # Pydantic data models for all agents
│   │   └── config.py         # Configuration management
│   ├── utils/
│   │   ├── pdf_parser.py     # PDF text and table extraction
│   │   ├── code_executor.py  # Jupyter kernel sandbox
│   │   ├── comparison.py     # Numerical comparison utilities
│   │   └── logging_utils.py  # Structured logging
│   ├── orchestrator.py       # Pipeline orchestration
│   └── main.py               # CLI entry point
├── config/
│   └── default_config.yaml   # Default configuration
├── tests/                    # Test suite
├── data/
│   ├── input/                # Input papers and datasets
│   └── output/               # Generated results
├── reports/                  # JSON reports from runs
├── requirements.txt
├── pyproject.toml
└── .env.example
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run a specific test file
pytest tests/test_comparison.py

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes and add tests
4. Run the test suite to verify (`pytest tests/`)
5. Commit your changes (`git commit -m "Add my feature"`)
6. Push to the branch (`git push origin feature/my-feature`)
7. Open a pull request

### Code Style

This project uses:
- **Black** for code formatting
- **Ruff** for linting
- **Type hints** throughout the codebase

## License

TBD

## Citation

If you use this system in your research, please cite:

```bibtex
@software{social_science_replicability,
  title={Social Science Replicability System},
  year={2025},
  description={An automated multi-agent system for replicating social science research papers}
}
```
