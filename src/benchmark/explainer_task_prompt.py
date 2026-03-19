"""Workspace setup and task prompt for the agentic Explainer phase.

The Explainer receives EVERYTHING — paper PDF, replication package, replicator
code/outputs, methodology summary, logs, and judge results — and investigates
root causes of discrepancies for non-A items.
"""

import json
import shutil
from pathlib import Path
from typing import Optional

from ..models.schemas import PaperSummary, VerificationReport
from ..utils.logging_utils import get_logger
from .config import PaperSpec
from .results import EvaluationResult

logger = get_logger(__name__)


# CLAUDE.md for the explainer workspace — allows reading everything provided.
EXPLAINER_CLAUDE_MD = """\
# Explainer Agent

You are investigating why an AI replicator's outputs differ from a paper's results.
You have full access to all materials and may use any tools needed to diagnose issues.

Read TASK.md for your full instructions.

## Permissions
- You may read and write any files inside this workspace.
- You may execute Python and R code.
- You may search the internet for library documentation, statistical methods,
  and software behavior differences.
- You may install packages if needed.
- Your output MUST include `explainer_report.json` in the schema specified in TASK.md.
"""


EXPLAINER_TASK_TEMPLATE = """\
# Discrepancy Explanation Task

You are an expert research methods analyst. An AI replicator attempted to
reproduce the empirical results of a research paper. A deterministic judge
graded each item, and some received grades below A. Your task is to investigate
each discrepancy, diagnose its root cause, and categorize fault.

## Pipeline Context

The replication pipeline works as follows — understanding it helps identify
where errors may have originated:

1. **Extractor** (gpt-5-mini with vision): Reads the paper PDF and extracts a
   structured methodology summary (`methodology_summary.json`). This includes
   regression specifications, variable names, data processing steps, sample
   restrictions, and table/figure structure. It does NOT include any results.

2. **Results Extractor** (gpt-5-mini with vision): Reads the paper PDF again and
   extracts the original cell values from each table into a structured JSON
   (`original_table_values/`). These are then blinded (numeric values removed)
   to create JSON templates for the replicator.

3. **Replicator** (the AI agent being evaluated): Receives ONLY the methodology
   summary, blinded table templates, and raw data files. It does NOT receive the
   original paper PDF, original code, or any results. It writes Python scripts
   and fills in the table templates with computed values.

4. **Judge** (deterministic): Compares original extracted values against
   replicated values cell-by-cell. {judge_description}

Errors can originate at any stage:
- **Paper underspecified**: the paper itself does not describe the methodology in enough detail to replicate (ambiguous methods, missing variable definitions, unclear sample restrictions, unstated data transformations)
- **Extractor missed**: the paper clearly describes something but the extractor failed to include it in the methodology summary
- **Extractor misinterpreted**: the paper describes something but the extractor conveyed it incorrectly
- **Paper-code mismatch**: the paper describes one approach but the original authors' code implements something different
- **Results extractor errors**: original values were read incorrectly from the PDF
- **Results mismatched**: the deterministic cell matching between original and replicated tables failed — cells were compared against the wrong counterpart due to structural differences, label mismatches, or extra/missing rows
- **Replicator errors**: wrong specification, coding bugs, wrong variables despite having sufficient information
- **Data issues**: missing variables, proprietary data, wrong data files
- **Software differences**: Stata/R vs Python defaults (SEs, optimization, etc.)

## Materials Provided

| Path | Contents |
|------|----------|
| `paper.pdf` | The original research paper |
| `methodology_summary.json` | Methodology extracted by the extractor (what the replicator was given — no results) |
| `replicator_code/` | Python/R scripts written by the AI replicator |
| `replicator_outputs/` | JSON tables and PNG figures produced by the replicator |
| `replicator_log.txt` | The replicator's execution log |
| `replicator_trajectory.json` | Full agent conversation log (SWE-agent runs): messages, tool calls, outputs |
| `judge_results/verification_report.json` | Per-item grades and cell-by-cell comparisons |
{original_values_row}{original_code_row}

## Items Requiring Explanation

{items_to_explain}

## Your Task

### First: Read the original paper

Before analyzing any specific item, read `paper.pdf` carefully. This is critical
because you need to distinguish between:
- Information that IS in the paper but was missed by the extractor
- Information that is NOT in the paper (paper underspecified)
- Information in the paper that contradicts the original code

For each discrepancy, compare what the paper says against:
1. What the methodology summary told the replicator
2. What the original code actually implements (if available)
3. What the replicator did

{task_instructions}

## You MAY Execute Code and Search the Internet

You are allowed and encouraged to:
- Run Python code to test hypotheses (load outputs, re-run regressions, check data)
- Search the internet for library documentation, statistical method details,
  and software behavior differences (e.g., "statsmodels vs Stata robust SE")
- Install packages if needed

## Output — MANDATORY

You MUST write TWO files:

### 1. `explainer_report.json`

Write this file with EXACTLY this JSON structure:

```json
{{
    "paper_id": "{paper_id}",
    "analyses": [
        {{
            "item_id": "Table 1",
            "grade": "B",
            "verbal_explanation": "Multi-paragraph explanation of the root cause...",
            "code_comparison": {{
                "item_id": "Table 1",
                "replicator_approach": "Summary of what the replicator did...",
                "original_approach": "Summary of what the original code does...",
                "key_differences": ["difference 1", "difference 2"]
            }},
            "primary_fault": "extractor_missed",
            "additional_faults": ["software_differences"],
            "fault_explanation": "The paper clearly describes X in Section 3.2, but the methodology summary omitted it...",
            "confidence": "high",
            "supporting_evidence": ["file.py line 42: uses X instead of Y", "..."],
            "suggested_fix": "The replicator should have..."
        }}
    ],
    "overall_assessment": "Summary of patterns across all discrepancies...",
    "methodology_quality_notes": "Assessment of how well the extractor captured the methodology..."
}}
```

Notes on the schema:
- `code_comparison` may be `null` if no original replication package is available
- `primary_fault` must be one of:
  - `replicator`: coding bug or wrong specification despite sufficient information
  - `extractor_missed`: paper clearly describes it but extractor omitted it from summary
  - `extractor_misinterpreted`: paper describes it but extractor conveyed it incorrectly
  - `paper_underspecified`: paper does not provide enough methodological detail to replicate
  - `paper_code_mismatch`: paper describes one approach but original code implements another
  - `results_extractor`: original values read incorrectly from the PDF
  - `results_mismatched`: the deterministic cell matching between original and replicated JSONs failed (wrong cell alignment, label mismatch, missing cells due to structural differences)
  - `data_limitation`: required data missing, proprietary, or insufficient
  - `software_differences`: Stata/R vs Python implementation differences
  - `other`: issue does not fit any category above — explain in detail in fault_explanation
- `additional_faults` is a list of secondary contributing factors (same categories, may be empty)
- `confidence` must be one of: `high`, `medium`, `low`
- `supporting_evidence` should include specific file:line references
- Include one entry in `analyses` for EVERY item listed above

### 2. `explanation.md`

A human-readable markdown report summarizing your findings. Include:
- A summary table of items, grades, and fault categories
- Detailed per-item sections with your analysis
- An overall assessment section

## Approach

- Be thorough — trace through code line by line, compare variable names, check data
- Be specific — cite file names, line numbers, variable names, exact values
- When multiple faults contribute, list the primary one and additional ones
- Remember the replicator never saw the paper or original code — it only had
  the methodology summary and data
"""


def setup_explainer_workspace(
    paper: PaperSpec,
    paper_summary: PaperSummary,
    evaluation: EvaluationResult,
    replicator_workspace: Path,
    workspace_dir: Path,
) -> None:
    """Set up the explainer workspace with all materials.

    Args:
        paper: Paper specification (pdf_path, data_path, replication_package_path).
        paper_summary: Pre-extracted methodology summary.
        evaluation: Judge evaluation result (verification + explanation reports).
        replicator_workspace: Path to the replicator's workspace (contains code, outputs, logs).
        workspace_dir: Target directory for the explainer workspace.
    """
    workspace_dir.mkdir(parents=True, exist_ok=True)

    # --- Copy paper PDF ---
    pdf_src = Path(paper.pdf_path)
    if pdf_src.exists():
        shutil.copy2(pdf_src, workspace_dir / "paper.pdf")
    else:
        logger.warning(f"Paper PDF not found: {pdf_src}")

    # --- Save methodology summary ---
    (workspace_dir / "methodology_summary.json").write_text(
        json.dumps(paper_summary.model_dump(), indent=2, default=str)
    )

    # --- Copy replicator code files ---
    code_dir = workspace_dir / "replicator_code"
    code_dir.mkdir(exist_ok=True)
    repl_ws = Path(replicator_workspace)
    code_exts = {".py", ".r", ".R"}
    for ext in code_exts:
        for f in repl_ws.rglob(f"*{ext}"):
            # Skip CLAUDE.md-related, __pycache__, etc.
            rel = f.relative_to(repl_ws)
            if any(part.startswith(".") or part == "__pycache__" for part in rel.parts):
                continue
            dest = code_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)

    # --- Copy replicator outputs (CSVs, PNGs) ---
    output_dir = workspace_dir / "replicator_outputs"
    output_dir.mkdir(exist_ok=True)
    output_exts = {".csv", ".png", ".jpg", ".jpeg", ".json"}
    for ext in output_exts:
        for f in repl_ws.glob(f"*{ext}"):
            # Only top-level outputs, skip methodology_summary.json and run_log.json
            if f.name in ("methodology_summary.json", "run_log.json"):
                continue
            shutil.copy2(f, output_dir / f.name)

    # --- Copy replicator log ---
    log_src = repl_ws / "run_log.txt"
    if log_src.exists():
        shutil.copy2(log_src, workspace_dir / "replicator_log.txt")
    else:
        # Try JSONL log (Codex)
        for log_name in ("run_log.txt", "run_log.jsonl"):
            alt = repl_ws / log_name
            if alt.exists():
                shutil.copy2(alt, workspace_dir / f"replicator_log{alt.suffix}")
                break

    # --- Copy SWE-agent trajectory (most informative log for swe-agent runs) ---
    trajectory_src = repl_ws / "trajectory.json"
    if trajectory_src.exists():
        shutil.copy2(trajectory_src, workspace_dir / "replicator_trajectory.json")

    # --- Save judge results ---
    judge_dir = workspace_dir / "judge_results"
    judge_dir.mkdir(exist_ok=True)
    (judge_dir / "verification_report.json").write_text(
        json.dumps(evaluation.verification_report.model_dump(), indent=2, default=str)
    )
    if evaluation.explanation_report:
        (judge_dir / "explanation_report.json").write_text(
            json.dumps(evaluation.explanation_report.model_dump(), indent=2, default=str)
        )

    # --- Copy original replication package code (if available) ---
    has_original_code = False
    if paper.replication_package_path:
        pkg_src = Path(paper.replication_package_path)
        if pkg_src.exists():
            orig_dir = workspace_dir / "original_code"
            orig_dir.mkdir(exist_ok=True)
            pkg_code_exts = {".py", ".r", ".R", ".do", ".sas", ".m", ".ado"}
            copied = 0
            for ext in pkg_code_exts:
                for f in pkg_src.rglob(f"*{ext}"):
                    rel = f.relative_to(pkg_src)
                    if any(part.startswith(".") for part in rel.parts):
                        continue
                    dest = orig_dir / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dest)
                    copied += 1
            has_original_code = copied > 0
            logger.info(f"Copied {copied} code files from replication package")

    # --- Copy original extracted table values (for comparison) ---
    has_original_values = False
    run_dir = Path(replicator_workspace).parent
    results_dir = run_dir.parent
    results_json = results_dir / "summaries" / f"{paper_summary.paper_id}_results.json"
    if not results_json.exists():
        # Try flat layout (benchmark_cli)
        results_json = results_dir / "summaries" / f"{paper_summary.paper_id}_results.json"
    if results_json.exists():
        orig_vals_dir = workspace_dir / "original_table_values"
        orig_vals_dir.mkdir(exist_ok=True)
        shutil.copy2(results_json, orig_vals_dir / "results.json")
        has_original_values = True

    # --- Determine item types present ---
    table_items = [v for v in evaluation.verification_report.item_verifications
                   if v.item_type == "table" and v.grade.value != "A"
                   and not v.unverifiable and not v.judge_error]
    figure_items = [v for v in evaluation.verification_report.item_verifications
                    if v.item_type == "figure" and v.grade.value != "A"
                    and not v.unverifiable and not v.judge_error]
    has_tables = len(table_items) > 0
    has_figures = len(figure_items) > 0

    # --- Build items-to-explain list ---
    items_lines = []
    for v in table_items + figure_items:
        grade = v.grade.value
        notes_preview = v.comparison_notes[:200] if v.comparison_notes else ""
        # For tables with cell comparisons, summarize
        tc = v.table_comparison
        if tc and tc.cell_comparisons:
            from collections import Counter
            cell_grades = Counter(c.grade for c in tc.cell_comparisons)
            cell_summary = ", ".join(f"{g}:{n}" for g, n in sorted(cell_grades.items()))
            items_lines.append(
                f"- **{v.item_id}** (Grade {grade}, cells: {cell_summary})"
            )
        else:
            items_lines.append(
                f"- **{v.item_id}** (Grade {grade}): {notes_preview}"
            )
    items_text = "\n".join(items_lines) if items_lines else "No items to explain (all grade A)."

    # --- Build task instructions based on item types ---
    task_parts = []
    if has_tables:
        task_parts.append("""\
### For each table:

1. **Load the original values** (`original_table_values/results.json`): Find the table
   and examine the extracted cell values. These are what the judge compared against.

2. **Load the replicated values** (`replicator_outputs/table_N.json`): These are
   the values the replicator produced.

3. **Identify non-A cells**: Cross-reference with `judge_results/verification_report.json`
   to find which specific cells have large differences.

4. **Trace through the replicator's code** (`replicator_code/table_N.py`): What
   specification did it implement? What variables, sample restrictions, controls?

5. **Read the methodology summary**: Was the description the replicator received
   complete and accurate? Check variable definitions, sample restrictions, fixed
   effects, clustering.

6. **Compare with the original code** (`original_code/`, if available): Identify
   specific differences — variable names, filters, model specs, SE computation.

7. **Diagnose and categorize**: Why do specific cells differ? Cite file:line references.""")

    if has_figures:
        task_parts.append("""\
### For each figure:

1. **Compare visually**: Open the replicated figure (`replicator_outputs/figure_N.png`)
   and compare with the original in `paper.pdf`.

2. **Check the replicator's code** (`replicator_code/figure_N.py`): What data
   transformations were applied? What plot parameters were used?

3. **Compare with the original code** if available.

4. **Diagnose**: Why do the plotted data patterns differ?""")

    task_instructions = "\n\n".join(task_parts) if task_parts else "No items require explanation."

    # --- Judge description based on item types ---
    judge_parts = []
    if has_tables:
        judge_parts.append(
            "Tables are graded deterministically: A (<2% diff), B (2-20%), "
            "C (20-40%), D (40-60%), E (>60% or sign flip), F (missing)."
        )
    if has_figures:
        judge_parts.append(
            "Figures are graded by an LLM vision model comparing plotted data patterns."
        )
    judge_description = " ".join(judge_parts)

    # --- Build TASK.md ---
    original_values_row = (
        "| `original_table_values/` | Original cell values extracted from the paper PDF (what the judge compared against) |\n"
        if has_original_values
        else ""
    )
    original_code_row = (
        "| `original_code/` | Original authors' replication code (.py, .r, .R, .do, .sas, .m) |\n"
        if has_original_code
        else ""
    )

    task_prompt = EXPLAINER_TASK_TEMPLATE.format(
        paper_id=paper_summary.paper_id,
        items_to_explain=items_text,
        task_instructions=task_instructions,
        judge_description=judge_description,
        original_values_row=original_values_row,
        original_code_row=original_code_row,
    )
    (workspace_dir / "TASK.md").write_text(task_prompt)

    # --- Write workspace instructions as both CLAUDE.md and AGENTS.md ---
    (workspace_dir / "CLAUDE.md").write_text(EXPLAINER_CLAUDE_MD)
    (workspace_dir / "AGENTS.md").write_text(EXPLAINER_CLAUDE_MD)

    logger.info(
        f"Explainer workspace set up: {len(items_lines)} items to explain, "
        f"original_code={'yes' if has_original_code else 'no'}"
    )
