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
# Workspace Rules — READ CAREFULLY

You are running a discrepancy explanation task. Your job is to investigate
why an AI replicator's outputs differ from the original paper's results.

## File Access
- You may ONLY read and write files inside this directory.
- Do NOT read, list, or access any files outside this workspace.
- Do NOT navigate to parent directories (`..`) or absolute paths outside this folder.

## Internet Access
- You may search for Python library documentation (statsmodels, pandas, etc.).
- Do NOT search for this paper by title, authors, DOI, or any identifying information.
- Do NOT search for the paper's results or any external replication attempts.

## Task
- Read TASK.md for your full instructions.
- You have access to ALL materials: paper PDF, replication package, replicator code, etc.
- Your output MUST include `explainer_report.json` in the exact schema specified in TASK.md.
"""


EXPLAINER_TASK_TEMPLATE = """\
# Discrepancy Explanation Task

You are an expert research methods analyst. An AI replicator attempted to
reproduce the empirical results of a research paper, and a judge graded each
table/figure. Some items received grades below A, meaning discrepancies exist.

Your task is to investigate each discrepancy, diagnose its root cause, and
categorize who is at fault.

## Materials Provided

| Path | Contents |
|------|----------|
| `paper.pdf` | The original research paper |
| `methodology_summary.json` | Methodology extracted by an AI extractor (what the replicator was given) |
| `replicator_code/` | Python/R scripts written by the AI replicator |
| `replicator_outputs/` | CSV tables and PNG figures produced by the replicator |
| `replicator_log.txt` | The replicator's execution log |
| `judge_results/verification_report.json` | Per-item grades and comparison notes from the judge |
{explanation_report_row}{original_code_row}

## Items Requiring Explanation

The following items received grades below A and need root cause analysis:

{items_to_explain}

## Your Task

For EACH item listed above, follow these steps:

1. **Read the paper** (`paper.pdf`): Find the relevant table/figure and understand
   what the correct result should look like — coefficients, significance levels,
   sample sizes, trends, etc.

2. **Read the methodology summary** (`methodology_summary.json`): Check what the
   extractor told the replicator. Was the description complete? Were variable
   definitions clear? Were sample restrictions specified? Were control variables
   listed? Were fixed effects and clustering described?

3. **Read the replicator's code** (`replicator_code/`): Trace through the logic.
   What specification did the replicator actually implement? What variables did
   it use? What sample restrictions did it apply?

4. **Compare with the original code** (`original_code/`, if available): Identify
   specific differences — different variable names, different sample filters,
   different model specifications, different standard error computations, etc.

5. **Read the replicator's log** (`replicator_log.txt`): Look for errors,
   warnings, or decisions the replicator made that may explain the discrepancy.

6. **Read the judge's notes** (`judge_results/`): Understand what the judge
   flagged as different.

7. **Diagnose the root cause**: Why does the replicator's output differ from
   the paper? Be specific — cite file names, line numbers, variable names.

8. **Categorize fault** — pick ONE primary category:
   - `replicator`: The replicator made an error (wrong specification, coding bug,
     misunderstanding of the methodology summary, wrong variables, etc.)
   - `extractor`: The methodology summary was incomplete, misleading, or missing
     critical information that the replicator needed (e.g., control variables
     not listed, sample restrictions ambiguous, fixed effects not specified)
   - `original_authors`: The original paper or replication package has an issue
     (inconsistency between paper and code, undocumented data transformations, etc.)
   - `data_limitation`: The discrepancy is due to data issues (missing variables,
     different data version, insufficient observations, etc.)
   - `software_differences`: The discrepancy stems from differences between
     statistical software implementations (e.g., Stata vs Python, different
     optimization algorithms, different default standard errors)

## You MAY Execute Code

You are allowed (and encouraged) to run Python code to test hypotheses about
discrepancies. For example:
- Load the replicator's output CSV and compare specific values
- Re-run parts of the replicator's code with modifications
- Check data filtering differences
- Compute the percentage difference between expected and actual values

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
            "fault_category": "extractor",
            "fault_explanation": "The methodology summary did not specify...",
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
- `fault_category` must be one of: `replicator`, `extractor`, `original_authors`, `data_limitation`, `software_differences`
- `confidence` must be one of: `high`, `medium`, `low`
- `supporting_evidence` should include specific file:line references
- Include one entry in `analyses` for EVERY item listed above

### 2. `explanation.md`

A human-readable markdown report summarizing your findings. Include:
- A summary table of items, grades, and fault categories
- Detailed per-item sections with your analysis
- An overall assessment section

## Constraints

- Only access files inside this workspace
- Do NOT search the internet for this paper
- Be thorough but concise — focus on actionable root causes
- When in doubt between fault categories, explain the ambiguity in `fault_explanation`
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

    # --- Build items-to-explain list ---
    items_lines = []
    for v in evaluation.verification_report.item_verifications:
        if v.grade.value == "A" or v.unverifiable or v.judge_error:
            continue
        notes_preview = v.comparison_notes[:200] if v.comparison_notes else ""
        items_lines.append(
            f"- **{v.item_id}** (Grade {v.grade.value}): {notes_preview}"
        )
    items_text = "\n".join(items_lines) if items_lines else "No items to explain (all grade A)."

    # --- Build TASK.md ---
    explanation_report_row = (
        "| `judge_results/explanation_report.json` | Judge's discrepancy analysis |\n"
        if evaluation.explanation_report
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
        explanation_report_row=explanation_report_row,
        original_code_row=original_code_row,
    )
    (workspace_dir / "TASK.md").write_text(task_prompt)

    # --- Write CLAUDE.md ---
    (workspace_dir / "CLAUDE.md").write_text(EXPLAINER_CLAUDE_MD)

    logger.info(
        f"Explainer workspace set up: {len(items_lines)} items to explain, "
        f"original_code={'yes' if has_original_code else 'no'}"
    )
