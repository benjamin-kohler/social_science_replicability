# Social Science Replicability System - Technical Specification

## Project Overview
Develop a multi-step, multi-agent system that automatically runs replications of social science papers, verifies results, and explains discrepancies.

---

## Tech Stack & Dependencies

### Core Framework
- **Agent Framework**: Open Agent (https://open-agent.io) - for flexible multi-agent orchestration with model-agnostic design
- **LLM Providers**: Supports OpenAI, Anthropic, and other providers through Open Agent's unified interface
- **Python Version**: 3.11+

### Key Libraries
```python
# Agent & LLM
open-agent>=0.1.0
openai>=1.0.0
anthropic>=0.20.0

# PDF Processing
pymupdf>=1.23.0  # PyMuPDF for text extraction
pdfplumber>=0.10.0  # Table extraction
pytesseract>=0.3.10  # OCR if needed

# Data Analysis
pandas>=2.1.0
numpy>=1.24.0
statsmodels>=0.14.0  # For regression analysis
scipy>=1.11.0

# Visualization
matplotlib>=3.8.0
seaborn>=0.13.0
plotly>=5.17.0

# Code Execution
jupyter-client>=8.0.0  # For notebook execution
nbformat>=5.9.0

# Utilities
pydantic>=2.5.0  # Data validation
python-dotenv>=1.0.0
```

### Optional (R Support)
- `rpy2>=3.5.0` for R code execution

---

## Project Structure

```
social_science_replicability/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── extractor.py          # Agent 1: Paper analysis
│   │   ├── replicator.py         # Agent 2: Code generation
│   │   ├── verifier.py           # Agent 3: Results comparison
│   │   └── explainer.py          # Agent 4: Discrepancy analysis
│   ├── models/
│   │   ├── __init__.py
│   │   ├── schemas.py            # Pydantic models for data structures
│   │   └── config.py             # Configuration management
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── pdf_parser.py         # PDF extraction utilities
│   │   ├── code_executor.py     # Safe code execution
│   │   ├── comparison.py        # Result comparison logic
│   │   └── logging_utils.py     # Structured logging
│   ├── orchestrator.py           # Open Agent workflow definition
│   └── main.py                   # Entry point
├── tests/
│   ├── test_extractor.py
│   ├── test_replicator.py
│   ├── test_verifier.py
│   └── test_integration.py
├── data/
│   ├── input/                    # Input papers and data
│   ├── output/                   # Generated results
│   └── examples/                 # Sample papers for testing
├── reports/                      # Generated JSON reports
├── notebooks/                    # Development notebooks
├── config/
│   └── default_config.yaml       # Default configuration
├── .env.example                  # Environment variables template
├── requirements.txt
├── pyproject.toml               # Poetry configuration
├── README.md
└── TECHNICAL_SPEC.md            # This file
```

---

## Data Structures & Interfaces

### 1. Agent 1 Output: PaperSummary
```python
from pydantic import BaseModel, Field
from typing import List, Optional

class DataProcessingStep(BaseModel):
    step_number: int
    description: str
    variables_involved: List[str]

class RegressionSpec(BaseModel):
    model_type: str  # "OLS", "Logit", "Fixed Effects", etc.
    dependent_var: str
    independent_vars: List[str]
    controls: List[str]
    fixed_effects: Optional[List[str]] = None
    clustering: Optional[str] = None
    sample_restrictions: Optional[str] = None

class TableSpec(BaseModel):
    table_number: str
    caption: str
    column_names: List[str]
    row_names: List[str]
    regression_specs: List[RegressionSpec]
    notes: Optional[str] = None

class PlotSpec(BaseModel):
    figure_number: str
    caption: str
    plot_type: str  # "scatter", "bar", "line", etc.
    x_axis: str
    y_axis: str
    grouping_vars: Optional[List[str]] = None
    notes: Optional[str] = None

class PaperSummary(BaseModel):
    paper_id: str
    research_questions: List[str]
    data_description: str
    data_context: str
    data_processing_steps: List[DataProcessingStep]
    tables: List[TableSpec]
    figures: List[PlotSpec]
    
    class Config:
        json_schema_extra = {
            "example": {
                "paper_id": "smith2023education",
                "research_questions": ["Does class size affect student achievement?"],
                # ... more examples
            }
        }
```

### 2. Agent 2 Output: ReplicationResults
```python
class GeneratedCode(BaseModel):
    language: str  # "python" or "r"
    code: str
    dependencies: List[str]
    execution_order: int

class GeneratedTable(BaseModel):
    table_number: str
    data: dict  # Could be list of dicts or nested structure
    format: str = "pandas_json"
    code_reference: str

class GeneratedFigure(BaseModel):
    figure_number: str
    file_path: str
    format: str = "png"
    code_reference: str

class ReplicationResults(BaseModel):
    paper_id: str
    code_files: List[GeneratedCode]
    tables: List[GeneratedTable]
    figures: List[GeneratedFigure]
    execution_log: str
    errors: List[str] = []
```

### 3. Agent 3 Output: VerificationReport
```python
from enum import Enum

class ReplicationGrade(str, Enum):
    A = "A"  # Fully replicated
    B = "B"  # Same direction, small discrepancies
    C = "C"  # Same direction, large discrepancies
    D = "D"  # Different results
    F = "F"  # Not comparable

class ItemVerification(BaseModel):
    item_id: str  # table_1, figure_2, etc.
    item_type: str  # "table" or "figure"
    grade: ReplicationGrade
    comparison_notes: str
    numerical_differences: Optional[dict] = None

class VerificationReport(BaseModel):
    paper_id: str
    overall_grade: ReplicationGrade
    item_verifications: List[ItemVerification]
    summary: str
```

### 4. Agent 4 Output: ExplanationReport
```python
class DiscrepancyAnalysis(BaseModel):
    item_id: str
    description_of_discrepancy: str
    likely_causes: List[str]
    is_identifiable: bool  # Can we determine the cause?
    fault_attribution: str  # "replicator", "original_paper", "unclear", "data_limitation"
    confidence: str  # "high", "medium", "low"

class ExplanationReport(BaseModel):
    paper_id: str
    analyses: List[DiscrepancyAnalysis]
    overall_assessment: str
    recommendations: List[str]
```

---

## System Architecture

### Open Agent Workflow

```python
# Pseudo-code structure in orchestrator.py
from open_agent import Agent, Workflow, State
from typing import TypedDict, Optional, List

class ReplicationState(State):
    """State object that flows through the agent workflow"""
    paper_pdf_path: str
    data_path: str
    replication_package_path: Optional[str]
    paper_summary: Optional[PaperSummary] = None
    replication_results: Optional[ReplicationResults] = None
    verification_report: Optional[VerificationReport] = None
    explanation_report: Optional[ExplanationReport] = None
    errors: List[str] = []

def create_workflow(model_provider: str = "openai", model_name: str = "gpt-4-turbo-preview"):
    """Create the Open Agent workflow for paper replication"""
    
    # Initialize agents with the specified model
    extractor = Agent(
        name="extractor",
        role="Extract methodology from papers",
        goal="Parse PDF and create structured PaperSummary without revealing results",
        model_provider=model_provider,
        model_name=model_name,
        function=extractor_agent
    )
    
    replicator = Agent(
        name="replicator",
        role="Generate replication code",
        goal="Create and execute code to replicate paper results",
        model_provider=model_provider,
        model_name=model_name,
        function=replicator_agent
    )
    
    verifier = Agent(
        name="verifier",
        role="Verify replication results",
        goal="Compare replicated results with original paper and assign grades",
        model_provider=model_provider,
        model_name=model_name,
        function=verifier_agent
    )
    
    explainer = Agent(
        name="explainer",
        role="Explain discrepancies",
        goal="Analyze and explain any differences in replication results",
        model_provider=model_provider,
        model_name=model_name,
        function=explainer_agent
    )
    
    # Create sequential workflow
    workflow = Workflow(
        name="Social Science Replication",
        agents=[extractor, replicator, verifier, explainer],
        state_class=ReplicationState
    )
    
    return workflow
```

---

## Agent Specifications

### Agent 1: Extractor (`agents/extractor.py`)

**Input**: 
- PDF file path
- Configuration (which sections to focus on, etc.)

**Process**:
1. Extract text from PDF using PyMuPDF
2. Identify sections (Methods, Results, Tables, Figures)
3. Parse table structures with pdfplumber
4. Use LLM to extract structured information per PaperSummary schema
5. Validate output with Pydantic

**Output**: `PaperSummary` object (JSON serializable)

**Key Implementation Notes**:
- MUST NOT extract actual numerical results from tables/figures
- Should use few-shot prompting with examples
- Open Agent handles retry logic automatically
- Should validate that no results are leaked

**Open Agent Implementation Pattern**:
```python
def extractor_agent(state: ReplicationState, agent: Agent) -> ReplicationState:
    """Agent function that processes state and returns updated state"""
    pdf_text = extract_text_from_pdf(state.paper_pdf_path)
    
    # Use agent.generate() for LLM calls
    prompt = create_extraction_prompt(pdf_text)
    response = agent.generate(prompt)
    
    # Parse and validate
    paper_summary = PaperSummary.parse_obj(response)
    state.paper_summary = paper_summary
    
    return state
```

### Agent 2: Replicator (`agents/replicator.py`)

**Input**:
- `PaperSummary` object
- Data file paths

**Process**:
1. Generate Python code (or R if specified) for each table/figure
2. Create modular code with clear functions
3. Execute code in isolated environment (using jupyter-client)
4. Capture outputs (tables as pandas DataFrames, figures as images)
5. Handle errors and retry with modifications

**Output**: `ReplicationResults` object

**Key Implementation Notes**:
- Generate code incrementally (data loading → cleaning → analysis)
- Use sandbox execution environment
- Implement timeout for long-running code
- Save intermediate results
- NO ACCESS to original paper or results

**Code Generation Prompt Template**:
```
Based on the following methodological description, generate Python code to:
1. Load the data from {data_path}
2. Apply the following data processing steps: {steps}
3. Generate {table/figure} with specifications: {specs}

Requirements:
- Use pandas, statsmodels, matplotlib/seaborn
- Include error handling
- Add comments explaining each step
- Output results in specified format
```

### Agent 3: Verifier (`agents/verifier.py`)

**Input**:
- Original paper PDF
- `ReplicationResults` object

**Process**:
1. Extract actual results from original paper (tables/figures)
2. Compare with replicated results
3. Calculate numerical differences for tables
4. Use vision LLM for figure comparison
5. Assign grades based on comparison criteria

**Output**: `VerificationReport` object

**Key Implementation Notes**:
- Use structured comparison metrics (e.g., % difference, correlation)
- For figures: use GPT-4 Vision or Claude Vision to compare images
- Document comparison methodology clearly
- Grade only on substance, not formatting

### Agent 4: Explainer (`agents/explainer.py`)

**Input**:
- Original paper PDF
- `PaperSummary`
- Generated code
- `ReplicationResults`
- `VerificationReport`
- Replication package (if available)

**Process**:
1. For each non-A grade item:
   - Compare generated code vs. original code (if available)
   - Analyze differences in assumptions/methods
   - Check for common issues (different stata/python implementations, etc.)
   - Determine if discrepancy is due to replicator error, paper ambiguity, or other
2. Generate structured analysis

**Output**: `ExplanationReport` object

**Key Implementation Notes**:
- Should be able to read R/Stata/Python code from replication packages
- Use code diffing techniques
- Consider common replication pitfalls (rounding, random seeds, software versions)

---

## Entry Point & CLI

### `main.py`

```python
import argparse
from pathlib import Path
from orchestrator import create_workflow
from models.schemas import ReplicationState
import json
from datetime import datetime

def run_replication(
    paper_path: str,
    data_path: str,
    replication_package_path: Optional[str] = None,
    output_dir: str = "reports",
    model_provider: str = "openai",
    model_name: str = "gpt-4-turbo-preview"
):
    """
    Run the full replication pipeline.
    
    Args:
        paper_path: Path to the PDF paper
        data_path: Path to the data files (directory or file)
        replication_package_path: Optional path to original replication package
        output_dir: Directory for output reports
        model_provider: "openai" or "anthropic"
        model_name: Specific model to use
    """
    # Initialize state
    initial_state = ReplicationState(
        paper_pdf_path=paper_path,
        data_path=data_path,
        replication_package_path=replication_package_path,
        errors=[]
    )
    
    # Create workflow
    workflow = create_workflow(
        model_provider=model_provider,
        model_name=model_name
    )
    
    # Run workflow
    final_state = workflow.invoke(initial_state)
    
    # Generate report
    report = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "model_provider": model_provider,
            "model_name": model_name,
            "paper_path": paper_path,
            "data_path": data_path
        },
        "results": {
            "paper_summary": final_state["paper_summary"].dict() if final_state["paper_summary"] else None,
            "verification_report": final_state["verification_report"].dict() if final_state["verification_report"] else None,
            "explanation_report": final_state["explanation_report"].dict() if final_state["explanation_report"] else None
        },
        "errors": final_state["errors"]
    }
    
    # Save report
    output_path = Path(output_dir) / f"replication_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Report saved to: {output_path}")
    return report

def main():
    parser = argparse.ArgumentParser(description="Social Science Replication System")
    parser.add_argument("--paper", required=True, help="Path to PDF paper")
    parser.add_argument("--data", required=True, help="Path to data files")
    parser.add_argument("--replication-package", help="Path to original replication package")
    parser.add_argument("--output-dir", default="reports", help="Output directory")
    parser.add_argument("--model-provider", default="openai", choices=["openai", "anthropic"])
    parser.add_argument("--model-name", default="gpt-4-turbo-preview")
    
    args = parser.parse_args()
    
    run_replication(
        paper_path=args.paper,
        data_path=args.data,
        replication_package_path=args.replication_package,
        output_dir=args.output_dir,
        model_provider=args.model_provider,
        model_name=args.model_name
    )

if __name__ == "__main__":
    main()
```

**Usage**:
```bash
# Basic usage
python -m src.main --paper data/input/paper.pdf --data data/input/dataset.csv

# With replication package
python -m src.main --paper paper.pdf --data dataset.csv --replication-package replication_files/

# Using Anthropic Claude
python -m src.main --paper paper.pdf --data dataset.csv --model-provider anthropic --model-name claude-3-opus-20240229
```

---

## Error Handling

### Error Categories
1. **PDF Parsing Errors**: Corrupted PDF, non-extractable text → Use OCR fallback
2. **LLM Errors**: API failures, rate limits → Implement exponential backoff
3. **Code Execution Errors**: Runtime errors, timeouts → Capture, log, attempt fixes
4. **Data Errors**: Missing files, format issues → Validate early, provide clear messages

### Implementation
- Use try-except blocks with specific exception types
- Log all errors to structured log file
- Store errors in state for final report
- Implement graceful degradation where possible

---

## Testing Strategy

### Unit Tests
- Test each agent independently with mock inputs
- Test data structure validation (Pydantic schemas)
- Test utility functions (PDF parsing, code execution)

### Integration Tests
- End-to-end test with sample paper and data
- Test error propagation through workflow
- Test different model providers

### Test Data
- Include 2-3 simple example papers with known results
- Papers should cover different methodologies (OLS, panel data, etc.)

---

## Configuration Management

### `config/default_config.yaml`
```yaml
open_agent:
  default_provider: "openai"  # or "anthropic", "ollama", etc.
  default_model: "gpt-4-turbo-preview"
  temperature: 0.1
  max_tokens: 4000
  
execution:
  timeout_seconds: 300
  max_retries: 3
  sandbox_type: "jupyter"
  
extraction:
  focus_sections: ["Methods", "Results", "Data"]
  extract_appendix: false
  
verification:
  numerical_tolerance: 0.01  # 1% difference threshold for grade B
  use_vision_model: true
  
output:
  save_intermediate_results: true
  reports_dir: "reports"
  figures_format: "png"
```

### Environment Variables (`.env.example`)
```bash
# API Keys
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here

# Optional: Open Agent Configuration
OPEN_AGENT_LOG_LEVEL=INFO
OPEN_AGENT_CACHE_DIR=.cache
```

---

## Development Workflow

### Phase 1: Setup & Infrastructure (Week 1)
- [ ] Set up project structure
- [ ] Configure dependencies
- [ ] Implement data schemas (Pydantic models)
- [ ] Create basic LangGraph workflow skeleton
- [ ] Set up testing framework

### Phase 2: Agent Development (Weeks 2-3)
- [ ] Implement Agent 1 (Extractor)
- [ ] Implement Agent 2 (Replicator)
- [ ] Implement Agent 3 (Verifier)
- [ ] Implement Agent 4 (Explainer)
- [ ] Unit tests for each agent

### Phase 3: Integration & Testing (Week 4)
- [ ] Connect agents in workflow
- [ ] End-to-end testing
- [ ] Error handling refinement
- [ ] Documentation

### Phase 4: Optimization (Week 5+)
- [ ] Improve prompts based on results
- [ ] Add caching for expensive operations
- [ ] Performance optimization
- [ ] Additional test cases

---

## README Requirements

The README.md should include:
1. Project overview and goals
2. Installation instructions (pip/poetry)
3. Quick start guide with example
4. CLI reference
5. Configuration options
6. Architecture diagram
7. Example outputs
8. Contributing guidelines
9. License information
10. Citation information

---

## Success Metrics

- **Accuracy**: % of papers where Grade A or B is achieved
- **Completeness**: % of tables/figures successfully generated
- **Speed**: Average time per paper replication
- **Robustness**: % of papers that complete without fatal errors

---

## Future Enhancements (Not for Initial Implementation)

- Web interface for easier usage
- Support for more statistical software (Stata, SAS)
- Automated data cleaning suggestions
- Interactive debugging mode
- Multi-paper batch processing
- Integration with academic databases (e.g., OSF, Dataverse)

---

## Original Requirements (for reference)

The original Claude.md contained these key requirements which are all addressed above:

### Goal
Develop a multi-step / multi-agent system that automatically runs a replication of a social science paper.

### Agent 1: Extractor
- Receives a paper in PDF as input
- Returns a structured document for Agent 2
- Includes: research questions, data descriptions, methods, data processing steps, regression specs, tables (structure only), plots (structure only)
- MUST NOT reveal actual outcomes or results

### Agent 2: Replicator
- Receives data and structured document from Step 1
- Outputs all tables and plots based on methodological summary
- Generates output by writing code (Python preferred, R allowed)
- MUST NOT access original paper or results

### Agent 3: Verifier
- Receives original paper PDF and replicated results
- Returns classification for each plot/table and overall classification
- Classifications: A (fully replicated), B (same direction, small discrepancies), C (same direction, large discrepancies), D (different results), F (not comparable)
- Based only on actual results, not formatting

### Agent 4: Explainer/Reasoning
- Receives all previous outputs plus original replication package
- For non-A grades, generates report with:
  - Description of discrepancies
  - Analysis of why they occurred
  - Binary classification of whether error is identifiable
  - Judgment on fault attribution (replicator vs. original paper)

### Methods
- Use open agent framework for easy model switching
- Entirely in Python
- Well documented README
- All runs generate JSON report with exact specifications
