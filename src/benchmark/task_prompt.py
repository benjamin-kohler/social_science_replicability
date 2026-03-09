"""Shared task prompt builder and workspace setup for CLI-based runners."""

import json
import shutil
from pathlib import Path

from ..models.schemas import PaperSummary
from .config import PaperSpec

# CLAUDE.md content written into each workspace — Claude Code reads this
# automatically as project instructions, enforcing isolation constraints.
WORKSPACE_CLAUDE_MD = """\
# Workspace Rules — READ CAREFULLY

You are running a benchmark replication task in an isolated workspace.

## File Access
- You may ONLY read and write files inside this directory.
- Do NOT read, list, or access any files outside this workspace.
- Do NOT navigate to parent directories (`..`) or absolute paths outside this folder.
- Do NOT use Glob, Grep, Read, or Bash to explore anything outside this workspace.

## Internet Access
- You may search for Python library documentation (statsmodels, pandas, matplotlib, scipy, numpy).
- Do NOT search for this paper by title, authors, DOI, or any identifying information.
- Do NOT search for this paper's results, replication code, replication packages, or related analyses.
- Do NOT search for any prior replication attempts of this paper.

## Task
- Read TASK.md for your full instructions.
- Work ONLY from the methodology summary and data provided in this workspace.
"""

TASK_TEMPLATE = """# Replication Task

You are given a methodological summary of a research paper and its associated
dataset. Your goal is to replicate the paper's empirical results using ONLY
the methodology description below and the data. You do NOT have access to the
original paper, its results, or any replication code.

## Data
The dataset is located at: `{data_filename}`

## Methodological Summary

**Paper**: {title} (ID: {paper_id})

**Research Questions**:
{research_questions}

**Data Description**: {data_description}

**Data Context**: {data_context}

{data_source}{sample_size}{time_period}

**Data Processing Steps**:
{processing_steps}

**Tables to Replicate**:
{table_specs}

**Figures to Replicate**:
{figure_specs}

## Constraints — MANDATORY

You are in an isolated workspace for fair benchmarking. These rules are strict
and non-negotiable:

1. **FILE ACCESS — workspace only.** You may ONLY read and write files inside
   this workspace directory. Do NOT access, read, list, or reference any files
   outside of it. Do NOT navigate to parent directories or any other location
   on disk. This workspace contains everything you need.

2. **NO searching for the paper.** Do NOT search the internet for this paper,
   its title, its authors, its published results, or any replication code or
   packages. Do NOT search for any prior attempts to replicate this paper.

3. **NO searching for results.** Do NOT look up expected coefficients, effect
   sizes, tables, or figures from this paper anywhere. Your replication must be
   derived entirely from the methodology summary below and the data provided.

4. **Allowed web use.** You MAY search for documentation on Python libraries
   (statsmodels, pandas, matplotlib, numpy, scipy, etc.) and general
   statistical methods (e.g. "how to run IV/2SLS in statsmodels"). Any other
   web searches are prohibited.

5. **Work independently.** Base your replication ONLY on the methodology
   description in this file and the dataset. Do NOT look for pre-existing
   solutions, related code, or reference implementations.

Violating any of these constraints invalidates the benchmark run.

## Instructions

EFFICIENCY IS CRITICAL — you have a limited number of turns. Do NOT waste turns
on excessive data exploration. The methodology summary above already describes
the variables, column names, and data structure in detail.

1. **Quick data check (1 turn max)**: Run ONE bash command to print actual
   column names from the data files:
   `python -c "import pandas as pd; [print(f, pd.read_stata(f'data/{{f}}').columns.tolist()[:10]) for f in __import__('os').listdir('data')]"`
   This confirms the real column names. Do NOT spend more turns exploring.

2. **Write `prepare_data.py`**: Load and clean the data following the processing
   steps described above. All table/figure scripts will import from this module.

3. **Write and execute ONE script at a time**: For each table/figure:
   a. Write the script (e.g., `table_1.py` → `table_1.csv`)
   b. Execute it with `python table_1.py`
   c. Fix any errors immediately
   d. Move on to the next item only after the output file is verified

   Naming: `table_2.1.py` → `table_2.1.csv`, `figure_3.1.py` → `figure_3.1.png`
   Use `statsmodels` for regressions (OLS, Logit, IV/2SLS). Do NOT implement OLS manually.

4. **CRITICAL**: You MUST actually execute every script and verify the output
   file exists. Do not stop after writing code.

5. **Save all outputs** in the current working directory.

Focus on substance and accuracy. Match the described methodology as closely as
possible, including sample restrictions, variable transformations, and
statistical specifications.
"""


# ---------------------------------------------------------------------------
# Paper-direct templates (replicator gets the raw PDF, not the summary)
# ---------------------------------------------------------------------------

WORKSPACE_CLAUDE_MD_PAPER_DIRECT = """\
# Workspace Rules — READ CAREFULLY

You are running a benchmark replication task in an isolated workspace.

## File Access
- You may ONLY read and write files inside this directory.
- Do NOT read, list, or access any files outside this workspace.
- Do NOT navigate to parent directories (`..`) or absolute paths outside this folder.
- Do NOT use Glob, Grep, Read, or Bash to explore anything outside this workspace.

## Internet Access
- You may search for Python library documentation (statsmodels, pandas, matplotlib, scipy, numpy).
- Do NOT search for this paper by title, authors, DOI, or any identifying information.
- Do NOT search for this paper's results, replication code, replication packages, or related analyses.
- Do NOT search for any prior replication attempts of this paper.

## Task
- Read TASK.md for your full instructions.
- Work from the paper PDF and data provided in this workspace.
"""

PAPER_DIRECT_TASK_TEMPLATE = """\
# Replication Task (Paper-Direct)

You are given a research paper (PDF) and its associated dataset.
Your goal is to replicate ALL empirical results (tables and figures)
using the paper and the data provided. You do NOT have access to any
replication code or replication package.

## Data
The dataset is located at: `{data_filename}`

## Paper
The paper PDF is located at: `{pdf_filename}`

If you cannot read the PDF directly, use `pymupdf` or `pdfplumber` to
extract text programmatically.

## Constraints — MANDATORY

You are in an isolated workspace for fair benchmarking. These rules are strict
and non-negotiable:

1. **FILE ACCESS — workspace only.** You may ONLY read and write files inside
   this workspace directory. Do NOT access, read, list, or reference any files
   outside of it. Do NOT navigate to parent directories or any other location
   on disk. This workspace contains everything you need.

2. **NO searching for the paper.** Do NOT search the internet for this paper,
   its title, its authors, its published results, or any replication code or
   packages. Do NOT search for any prior attempts to replicate this paper.

3. **NO searching for results.** Do NOT look up expected coefficients, effect
   sizes, tables, or figures from this paper anywhere. Your replication must be
   derived entirely from reading the paper PDF and using the data provided.

4. **Allowed web use.** You MAY search for documentation on Python libraries
   (statsmodels, pandas, matplotlib, numpy, scipy, etc.) and general
   statistical methods (e.g. "how to run IV/2SLS in statsmodels"). Any other
   web searches are prohibited.

5. **Work independently.** Base your replication ONLY on the paper and the
   dataset. Do NOT look for pre-existing solutions, related code, or
   reference implementations.

Violating any of these constraints invalidates the benchmark run.

## Instructions

EFFICIENCY IS CRITICAL — you have a limited number of turns.

1. **Read the paper**: Examine the PDF to identify all tables and figures
   with empirical results. Note the methodology, variable definitions,
   data processing steps, regression specifications, and sample restrictions.

2. **Quick data check (1 turn max)**: Run ONE bash command to check actual
   column names from the data files.

3. **Write `prepare_data.py`**: Load and clean the data following the processing
   steps described in the paper. All table/figure scripts will import from
   this module.

4. **Write and execute ONE script at a time**: For each table/figure:
   a. Write the script (e.g., `table_1.py` → `table_1.csv`)
   b. Execute it with `python table_1.py`
   c. Fix any errors immediately
   d. Move on to the next item only after the output file is verified

   Naming: `table_N.py` → `table_N.csv`, `figure_N.py` → `figure_N.png`
   Use `statsmodels` for regressions (OLS, Logit, IV/2SLS). Do NOT implement
   OLS manually.

5. **CRITICAL**: You MUST actually execute every script and verify the output
   file exists. Do not stop after writing code.

6. **Save all outputs** in the current working directory.

Focus on substance and accuracy. Match the paper's methodology as closely as
possible, including sample restrictions, variable transformations, and
statistical specifications.
"""


def setup_workspace_paper_direct(
    paper: PaperSpec,
    paper_summary: PaperSummary,
    workspace_dir: Path,
) -> str:
    """Set up an isolated workspace for paper-direct replication.

    Copies the data AND the paper PDF into the workspace (but no replication
    package). Writes a simplified TASK.md that instructs the replicator to
    work from the PDF. Also saves methodology_summary.json for the judge
    (the replicator is NOT told about this file).

    Args:
        paper: Paper specification (pdf_path, data_path).
        paper_summary: Pre-extracted methodology summary (saved for judge only).
        workspace_dir: Directory to set up.

    Returns:
        The data_filename used in the task prompt.
    """
    workspace_dir.mkdir(parents=True, exist_ok=True)

    # Copy data (same as standard setup)
    data_src = Path(paper.data_path)
    if data_src.is_dir():
        data_dest = workspace_dir / "data"
        if data_dest.exists():
            shutil.rmtree(data_dest)
        shutil.copytree(data_src, data_dest)
        data_filename = "data/"
    elif data_src.exists():
        shutil.copy2(data_src, workspace_dir / data_src.name)
        data_filename = data_src.name
    else:
        data_filename = data_src.name

    # Copy paper PDF into workspace
    pdf_src = Path(paper.pdf_path)
    pdf_filename = "paper.pdf"
    if pdf_src.exists():
        shutil.copy2(pdf_src, workspace_dir / pdf_filename)

    # Write paper-direct task prompt (NO methodology summary for the replicator)
    task_prompt = PAPER_DIRECT_TASK_TEMPLATE.format(
        data_filename=data_filename,
        pdf_filename=pdf_filename,
    )
    (workspace_dir / "TASK.md").write_text(task_prompt)

    # Save methodology summary as JSON for the JUDGE (not referenced in TASK.md)
    (workspace_dir / "methodology_summary.json").write_text(
        json.dumps(paper_summary.model_dump(), indent=2, default=str)
    )

    # Write CLAUDE.md with paper-direct rules
    (workspace_dir / "CLAUDE.md").write_text(WORKSPACE_CLAUDE_MD_PAPER_DIRECT)

    return data_filename


def build_task_prompt(summary: PaperSummary, data_filename: str) -> str:
    """Generate the task prompt from a methodology summary.

    Args:
        summary: Pre-extracted methodology summary (no results).
        data_filename: Filename or directory name for the data in the workspace.

    Returns:
        Formatted task prompt string.
    """
    # Format research questions
    rqs = "\n".join(f"- {q}" for q in summary.research_questions) or "- Not specified"

    # Format processing steps
    if summary.data_processing_steps:
        steps = "\n".join(
            f"{s.step_number}. {s.description}"
            + (f" (variables: {', '.join(s.variables_involved)})" if s.variables_involved else "")
            for s in summary.data_processing_steps
        )
    else:
        steps = "No specific steps listed."

    # Format table specs
    if summary.tables:
        table_parts = []
        for t in summary.tables:
            table_filename = t.table_number.replace(" ", "_").lower() + ".csv"
            script_name = t.table_number.replace(" ", "_").lower() + ".py"
            part = f"### {t.table_number}: {t.caption}\n"
            part += f"**Script**: `{script_name}` → **Output**: `{table_filename}`\n"
            if t.column_names:
                part += f"- Columns: {', '.join(t.column_names)}\n"
            if t.row_names:
                part += f"- Rows: {', '.join(t.row_names)}\n"
            if t.panel_structure:
                part += f"- Panel structure: {t.panel_structure}\n"
            for spec in t.regression_specs:
                part += f"- **Regression**: {spec.model_type}, DV={spec.dependent_var}"
                part += f", IVs=[{', '.join(spec.independent_vars)}]"
                if spec.controls:
                    part += f", Controls=[{', '.join(spec.controls)}]"
                if spec.fixed_effects:
                    part += f", FE=[{', '.join(spec.fixed_effects)}]"
                if spec.clustering:
                    part += f", Clustered by {spec.clustering}"
                if spec.sample_restrictions:
                    part += f", Sample: {spec.sample_restrictions}"
                part += "\n"
                if spec.equation_latex:
                    part += f"  - **Equation**: $${spec.equation_latex}$$\n"
                if spec.variable_definitions:
                    part += f"  - **Variable definitions**: {spec.variable_definitions}\n"
            if t.notes:
                part += f"- Notes: {t.notes}\n"
            if t.template_markdown:
                part += f"\n**Structural template** (your output MUST match this structure — replace XXX with computed values, leave empty cells empty):\n\n{t.template_markdown}\n"
            table_parts.append(part)
        table_specs = "\n".join(table_parts)
    else:
        table_specs = "No tables specified."

    # Format figure specs
    if summary.figures:
        fig_parts = []
        for f in summary.figures:
            fig_filename = f.figure_number.replace(" ", "_").lower() + ".png"
            script_name = f.figure_number.replace(" ", "_").lower() + ".py"
            part = f"### {f.figure_number}: {f.caption}\n"
            part += f"**Script**: `{script_name}` → **Output**: `{fig_filename}`\n"
            part += f"- Plot type: {f.plot_type}\n"
            part += f"- X-axis: {f.x_axis}, Y-axis: {f.y_axis}\n"
            if f.grouping_vars:
                part += f"- Grouping: {', '.join(f.grouping_vars)}\n"
            if f.subplot_structure:
                part += f"- Subplots: {f.subplot_structure}\n"
            for spec in f.regression_specs:
                part += f"- **Regression**: {spec.model_type}, DV={spec.dependent_var}"
                part += f", IVs=[{', '.join(spec.independent_vars)}]"
                if spec.controls:
                    part += f", Controls=[{', '.join(spec.controls)}]"
                if spec.fixed_effects:
                    part += f", FE=[{', '.join(spec.fixed_effects)}]"
                if spec.clustering:
                    part += f", Clustered by {spec.clustering}"
                if spec.sample_restrictions:
                    part += f", Sample: {spec.sample_restrictions}"
                part += "\n"
                if spec.equation_latex:
                    part += f"  - **Equation**: $${spec.equation_latex}$$\n"
                if spec.variable_definitions:
                    part += f"  - **Variable definitions**: {spec.variable_definitions}\n"
            if f.notes:
                part += f"- Notes: {f.notes}\n"
            if f.template_code:
                part += f"\n**Code skeleton** (use this as a starting point for the figure):\n\n```python\n{f.template_code}\n```\n"
            fig_parts.append(part)
        figure_specs = "\n".join(fig_parts)
    else:
        figure_specs = "No figures specified."

    # Optional fields
    data_source = f"**Data Source**: {summary.data_source}\n" if summary.data_source else ""
    sample_size = f"**Sample Size**: {summary.sample_size}\n" if summary.sample_size else ""
    time_period = f"**Time Period**: {summary.time_period}\n" if summary.time_period else ""

    return TASK_TEMPLATE.format(
        data_filename=data_filename,
        title=summary.title or summary.paper_id,
        paper_id=summary.paper_id,
        research_questions=rqs,
        data_description=summary.data_description,
        data_context=summary.data_context,
        data_source=data_source,
        sample_size=sample_size,
        time_period=time_period,
        processing_steps=steps,
        table_specs=table_specs,
        figure_specs=figure_specs,
    )


def setup_workspace(
    paper: PaperSpec,
    paper_summary: PaperSummary,
    workspace_dir: Path,
) -> str:
    """Set up an isolated workspace with data, TASK.md, and methodology JSON.

    Copies only the data into the workspace (no paper PDF, no replication package),
    writes the task prompt as TASK.md, and saves the methodology summary as JSON.

    Args:
        paper: Paper specification (used only for data_path).
        paper_summary: Pre-extracted methodology summary (no results).
        workspace_dir: Directory to set up.

    Returns:
        The data_filename used in the task prompt.
    """
    workspace_dir.mkdir(parents=True, exist_ok=True)

    # Copy ONLY the data into workspace (no paper PDF, no replication package)
    data_src = Path(paper.data_path)
    if data_src.is_dir():
        data_dest = workspace_dir / "data"
        if data_dest.exists():
            shutil.rmtree(data_dest)
        shutil.copytree(data_src, data_dest)
        data_filename = "data/"
    elif data_src.exists():
        shutil.copy2(data_src, workspace_dir / data_src.name)
        data_filename = data_src.name
    else:
        data_filename = data_src.name

    # Write task prompt with methodology summary
    task_prompt = build_task_prompt(paper_summary, data_filename)
    (workspace_dir / "TASK.md").write_text(task_prompt)

    # Save the summary as JSON for reference
    (workspace_dir / "methodology_summary.json").write_text(
        json.dumps(paper_summary.model_dump(), indent=2, default=str)
    )

    # Write CLAUDE.md — Claude Code reads this as project-level instructions
    (workspace_dir / "CLAUDE.md").write_text(WORKSPACE_CLAUDE_MD)

    return data_filename
