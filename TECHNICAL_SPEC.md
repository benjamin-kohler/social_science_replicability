# Technical Specification

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent orchestration | LangGraph (StateGraph with conditional edges) |
| LLM interface | LangChain (ChatOpenAI, ChatAnthropic) |
| LLM providers | OpenAI, Anthropic |
| Data validation | Pydantic v2 |
| PDF extraction | PyMuPDF, pdfplumber |
| Data analysis | pandas, numpy, statsmodels, scipy |
| Visualization | matplotlib, seaborn, plotly |
| Code execution | jupyter-client (kernel sandbox) |
| Configuration | YAML + python-dotenv |
| Testing | pytest (206 tests) |
| Python | 3.11+ |

Optional: `rpy2` for R code execution.

---

## Data Flow

### Replication Pipeline

```
Paper PDF ──► ExtractorAgent ──► PaperSummary (methodology, no results)
                                       │
Data files ────────────────────────────┤
                                       ▼
                               ReplicatorAgent ──► ReplicationResults
                                                   (code, tables, figures)
                                       │
Paper PDF ─────────────────────────────┤
                                       ▼
                               VerifierAgent ──► VerificationReport
                                                 (grades A-F per item)
                                       │
                                       ▼
                               ExplainerAgent ──► ExplanationReport
                                                  (discrepancy analyses)
```

### Benchmark Pipeline

```
Paper PDF ──► ExtractorAgent [judge model] ──► PaperSummary (cached per paper)
                                                      │
                                    ┌─────────────────┴─────────────────┐
                                    │                                   │
                              OpencodeRunner                    StructuredRunner
                              (subprocess)                    (run_from_summary)
                                    │                                   │
                                    ▼                                   ▼
                              RunArtifacts                        RunArtifacts
                              (workspace)                    (ReplicationResults)
                                    │                                   │
                                    └──────────┬────────────────────────┘
                                               │
                                        SharedEvaluator [judge model]
                                        (VerifierAgent + ExplainerAgent)
                                               │
                                               ▼
                                        EvaluationResult
                                        (grades, analyses)
```

---

## Core Schemas (`src/models/schemas.py`)

### Agent 1 Output: PaperSummary

Structured methodology description that **excludes all actual results**.

```python
PaperSummary
├── paper_id: str
├── title: Optional[str]
├── research_questions: list[str]
├── data_description: str
├── data_context: str
├── data_source: Optional[str]
├── sample_size: Optional[str]
├── time_period: Optional[str]
├── data_processing_steps: list[DataProcessingStep]
│   └── step_number, description, variables_involved
├── tables: list[TableSpec]
│   └── table_number, caption, column_names, row_names, regression_specs, notes
│       └── RegressionSpec: model_type, dependent_var, independent_vars, controls,
│                           fixed_effects, clustering, sample_restrictions
└── figures: list[PlotSpec]
    └── figure_number, caption, plot_type, x_axis, y_axis, grouping_vars, notes
```

### Agent 2 Output: ReplicationResults

```python
ReplicationResults
├── paper_id: str
├── code_files: list[GeneratedCode]        # language, code, dependencies, execution_order
├── tables: list[GeneratedTable]           # table_number, data (dict), execution_success
├── figures: list[GeneratedFigure]         # figure_number, file_path, execution_success
├── execution_log: str
├── errors: list[str]
└── warnings: list[str]
```

### Agent 3 Output: VerificationReport

```python
VerificationReport
├── paper_id: str
├── overall_grade: ReplicationGrade        # A, B, C, D, F
├── item_verifications: list[ItemVerification]
│   └── item_id, item_type, grade, comparison_notes, numerical_differences
├── summary: str
└── methodology_notes: Optional[str]
```

### Agent 4 Output: ExplanationReport

```python
ExplanationReport
├── paper_id: str
├── analyses: list[DiscrepancyAnalysis]
│   └── item_id, grade, description_of_discrepancy, likely_causes,
│       is_identifiable, fault_attribution, confidence, supporting_evidence
├── overall_assessment: str
├── recommendations: list[str]
└── replication_package_comparison: Optional[str]
```

### LangGraph State

```python
GraphState(TypedDict)
├── paper_pdf_path: str
├── data_path: str
├── output_dir: str
├── paper_id: str
├── replication_package_path: Optional[str]
├── paper_summary: Optional[PaperSummary]
├── replication_results: Optional[ReplicationResults]
├── verification_report: Optional[VerificationReport]
├── explanation_report: Optional[ExplanationReport]
├── errors: Annotated[list[str], operator.add]   # accumulates across nodes
├── warnings: Annotated[list[str], operator.add]
├── current_step: str
└── success: bool                                 # controls conditional edges
```

---

## Pipeline Orchestration (`src/orchestrator.py`)

### Graph Builders

| Function | Nodes | Use Case |
|----------|-------|----------|
| `build_replication_graph(config)` | extract → replicate → verify → explain | Full pipeline |
| `build_extraction_only_graph(config)` | extract | Methodology extraction only |
| `build_from_summary_graph(config)` | replicate → verify → explain | Skip extraction (pre-extracted summary) |
| `build_extract_replicate_graph(config)` | extract → replicate | Benchmarking (judge handles evaluation) |

All graphs use `should_continue(state)` for conditional routing: if `success=False`, the pipeline jumps to `END`.

### ReplicationOrchestrator Class

```python
class ReplicationOrchestrator:
    def run(paper_path, data_path, ...) -> ReplicationState          # Full 4-step pipeline
    def run_extraction_only(paper_path, ...) -> ReplicationState     # Extraction only
    def run_from_summary(paper_summary, data_path, ...) -> ReplicationState  # Skip extraction
```

---

## Benchmark Framework (`src/benchmark/`)

### Configuration (`config.py`)

```python
ModelSpec:      provider, model_name, api_key_env
PaperSpec:      paper_id, pdf_path, data_path, replication_package_path
JudgeConfig:    provider, model_name
BenchmarkConfig: models, papers, approaches, judge, output_dir, opencode_binary, timeout_seconds
```

### Information Isolation

The benchmark enforces strict separation between what the replicator can see:

**Allowed**: `PaperSummary` (methodology specs without results) + dataset files
**Blocked**: paper PDF, replication package, original code, actual results

This is enforced by:
1. **Pre-extraction**: The judge model extracts `PaperSummary` once per paper and caches it
2. **Freestyle runner**: Only copies data into workspace; writes methodology as text in `TASK.md`; no paper PDF
3. **Structured runner**: Uses `run_from_summary()` — the ReplicatorAgent only receives the summary

### Runners

**OpencodeRunner** (freestyle):
- Creates isolated workspace with only data + `TASK.md` (methodology summary as text)
- Runs `opencode run -m <model> -p TASK.md --permission allow` as subprocess
- Returns `RunArtifacts` (workspace dir, stdout, stderr, exit code, duration)

**StructuredRunner** (pipeline):
- Creates a `Config` targeting the specified model
- Calls `ReplicationOrchestrator.run_from_summary(paper_summary, data_path, ...)`
- Returns `RunArtifacts` with `ReplicationResults` directly

### Evaluation

**SharedEvaluator**:
- Uses a fixed judge model for all evaluations (cross-model comparability)
- For freestyle: `ArtifactParser` scans workspace for `.py`, `.csv`, `.png` files → `ReplicationResults`
- For structured: uses `ReplicationResults` directly from pipeline
- Runs `VerifierAgent` (judge model) → `VerificationReport`
- Runs `ExplainerAgent` (judge model) → `ExplanationReport` (if non-A grades exist)

### Results

```python
RunArtifacts:       workspace_dir, stdout, stderr, exit_code, duration, replication_results
EvaluationResult:   verification_report, explanation_report, overall_grade, item_grades
SingleRunResult:    model, paper, approach, artifacts, evaluation, duration
BenchmarkResults:   runs[], summary{}
```

`ResultsAggregator` saves per-run JSON and aggregated `summary.csv` + `summary.json`.

---

## Agent Implementation (`src/agents/`)

### BaseAgent

All agents inherit from `BaseAgent` which provides:
- `generate(prompt, system_prompt)` → text response via LangChain
- `generate_json(prompt, system_prompt)` → parsed JSON dict
- Lazy `chat_model` property (creates from config on first use)
- Supports dependency injection of chat model for testing

### Verifier — Vision Support

The VerifierAgent supports visual figure comparison:
1. Encodes replicated figure as base64
2. Sends to vision API (OpenAI or Anthropic) with paper's figure description
3. Falls back to text-only comparison if vision fails

Uses raw API clients (not LangChain) for multimodal requests.

### Replicator — Code Execution

1. Generates setup code (imports, data loading)
2. For each table: generates code via LLM → executes in Jupyter kernel → captures output
3. For each figure: generates code → executes → saves PNG
4. On Python failure: retries with R (via rpy2)
5. Saves all code as a Jupyter notebook

---

## Network Access

Both approaches have **unrestricted internet access**:
- LLM API calls (OpenAI/Anthropic) over HTTPS
- Jupyter kernel can `pip install` packages and make HTTP requests
- Opencode subprocess inherits full network access

No Docker/firewall isolation is currently implemented. For production benchmarking, consider:
- Docker containers per run (network-isolated)
- Firewall rules allowing only LLM API endpoints
- Restricting opencode's `--permission` flag

---

## Testing

206 tests total:

| Test File | Count | What it tests |
|-----------|-------|---------------|
| `test_agents.py` | 39 | All four agents + base agent |
| `test_benchmark.py` | 37 | Benchmark config, runners, evaluator, aggregator |
| `test_orchestrator.py` | 7 | Pipeline orchestration |
| `test_schemas.py` | 18 | Pydantic model validation |
| `test_comparison.py` | 18 | Numerical comparison utilities |
| `test_code_executor.py` | 14 | Jupyter kernel sandbox |
| `test_config.py` | 8 | Configuration loading |
| `test_collector.py` | 6 | Paper collection agent |
| `test_pdf_parser.py` | 9 | PDF extraction utilities |
| `test_logging_utils.py` | 8 | Logging utilities |

All tests use mocked LLM calls (no API keys needed).
