"""Shared task prompt builder and workspace setup for CLI-based runners."""

import json
import shutil
from pathlib import Path

from ..models.schemas import ExtractedTable, PaperSummary
from .config import PaperSpec

# Instructions written into each workspace as both CLAUDE.md (Claude Code reads
# this automatically) and AGENTS.md (other runners can read it explicitly).
WORKSPACE_INSTRUCTIONS = """\
Read TASK.md for your full instructions. Work only within this workspace.
Exception: the data/ directory may be a symlink — use it freely as your dataset.

RESTRICTIONS:
- Do NOT search the internet, fetch web pages, or look up the paper or its results online.
- Do NOT access any files outside this workspace and the data/ symlink.
- Do NOT hard-code results. All values must be computed from the data by closely
  following the methods described in TASK.md. If you cannot compute a value, leave it
  as null — do not guess or copy numbers from any source.

IMPORTANT: You must replicate ALL tables listed in TASK.md, not just the first one.
Check table_templates/ for the full list. Write and execute a separate script for each
table (table_1.py → table_1.json, table_2.py → table_2.json, etc.). Do NOT stop after
completing one table — continue until every template has a corresponding output JSON.

TIME MANAGEMENT: You have a strict time limit. Start writing and executing code immediately.
Do NOT spend excessive time exploring data files — write a quick Python script to inspect
data formats if needed, then move on to implementing the replication.
"""

TASK_TEMPLATE = """# Replication Task

You are given a methodological summary of a research paper and its associated
dataset. Your goal is to replicate the paper's empirical results using only
the methodology description below and the data. You do not have access to the
original paper, its results, or any replication code.

## Data
The dataset is located at: `{data_filename}`

## Methodological Summary

**Paper**: {title} (ID: {paper_id})

**Research Questions**:
{research_questions}

**Data Description**: {data_description}

**Data Context**: {data_context}

{data_source}{time_period}

**Data Processing Steps**:
{processing_steps}

{items_section}
## Constraints

You are in an isolated workspace for fair benchmarking.

1. **Workspace only.** Only read and write files inside this workspace directory.
   Do not access files outside of it or navigate to parent directories.
   **Exception:** the `data/` directory may be a symlink pointing outside the
   workspace — this is intentional and you should use it freely as your dataset.

2. **No searching for the paper.** Do not search the internet for this paper,
   its authors, its published results, or any replication code or packages.
   Do not search for prior replication attempts.

3. **No searching for results.** Do not look up expected coefficients, effect
   sizes, tables, or figures from this paper. Your replication must be derived
   entirely from the methodology summary and the data provided.

4. **Allowed web use.** You may search for Python library documentation
   (e.g. statsmodels, pandas, matplotlib) and general statistical methods.

5. **Work independently.** Base your replication only on the methodology
   description in this file and the dataset.

## Instructions

1. **Quick data check**: Inspect the data files to confirm column names and
   basic structure. The methodology summary above describes the variables in
   detail, so a brief check should suffice.

2. **Write `prepare_data.py`**: Load and clean the data following the processing
   steps described above. All scripts can import from this module.

3. **Write and execute one script at a time**: For each item:
   a. Write the script (see output filename specified for each item above)
   b. Execute it
   c. Fix any errors immediately
   d. Move on to the next item once the output file is verified
{table_instructions}{figure_instructions}
4. **R packages**: If the paper's methodology requires R-specific packages
   (e.g. for specialized estimators), you can call R from Python using `rpy2`.

5. Execute every script and verify the output file exists.

6. Save all outputs in the current working directory.

**Reasonable assumptions.** Where the methodology description is incomplete or
ambiguous, you are free to make reasonable assumptions based on common practice
in the field. Document your assumptions briefly in comments.

Focus on substance and accuracy. Match the described methodology as closely as
possible, including sample restrictions, variable transformations, and
statistical specifications.
"""


# ---------------------------------------------------------------------------
# Paper-direct templates (replicator gets the raw PDF, not the summary)
# ---------------------------------------------------------------------------

PAPER_DIRECT_TASK_TEMPLATE = """\
# Replication Task (Paper-Direct)

You are given a research paper (PDF) and its associated dataset.
Your goal is to replicate all empirical results (tables and figures)
using the paper and the data provided. You do not have access to any
replication code or replication package.

## Data
The dataset is located at: `{data_filename}`

## Paper
The paper PDF is located at: `{pdf_filename}`

If you cannot read the PDF directly, use `pymupdf` or `pdfplumber` to
extract text programmatically.

## Constraints

You are in an isolated workspace for fair benchmarking.

1. **Workspace only.** Only read and write files inside this workspace directory.
   Do not access files outside of it or navigate to parent directories.
   **Exception:** the `data/` directory may be a symlink pointing outside the
   workspace — this is intentional and you should use it freely as your dataset.

2. **No searching for replication code.** Do not search the internet for
   replication code, replication packages, or prior replication attempts
   for this paper.

3. **Allowed web use.** You may search for Python library documentation
   (e.g. statsmodels, pandas, matplotlib) and general statistical methods.

4. **Work independently.** Base your replication only on the paper and the
   dataset. Do not look for pre-existing solutions or reference implementations.

## Instructions

1. **Read the paper**: Examine the PDF to identify all tables and figures
   with empirical results. Note the methodology, variable definitions,
   data processing steps, regression specifications, and sample restrictions.

2. **Quick data check**: Inspect the data files to confirm column names and
   basic structure.

3. **Write `prepare_data.py`**: Load and clean the data following the processing
   steps described in the paper. All table/figure scripts can import from
   this module.

4. **Write and execute one script at a time**: For each table/figure:
   a. Write the script (e.g., `table_1.py` → `table_1.csv`)
   b. Execute it
   c. Fix any errors immediately
   d. Move on to the next item once the output file is verified

   Suggested naming: `table_N.py` → `table_N.csv`, `figure_N.py` → `figure_N.png`

5. Execute every script and verify the output file exists.

6. Save all outputs in the current working directory.

**Reasonable assumptions.** Where the paper's methodology is incomplete or
ambiguous, you are free to make reasonable assumptions based on common practice
in the field. Document your assumptions briefly in comments.

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

    # Symlink data into workspace (avoids duplicating large datasets)
    data_src = Path(paper.data_path).resolve()
    if data_src.is_dir():
        data_dest = workspace_dir / "data"
        if data_dest.is_symlink():
            data_dest.unlink()
        elif data_dest.exists():
            shutil.rmtree(data_dest)
        data_dest.symlink_to(data_src)
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

    # Write workspace instructions as both CLAUDE.md and AGENTS.md
    (workspace_dir / "CLAUDE.md").write_text(WORKSPACE_INSTRUCTIONS)
    (workspace_dir / "AGENTS.md").write_text(WORKSPACE_INSTRUCTIONS)

    # OpenCode project config: allow all permissions (symlinks, bash, etc.)
    _write_opencode_config(workspace_dir)

    return data_filename


def build_task_prompt(
    summary: PaperSummary,
    data_filename: str,
    item_types: list[str] | None = None,
) -> str:
    """Generate the task prompt from a methodology summary.

    Args:
        summary: Pre-extracted methodology summary (no results).
        data_filename: Filename or directory name for the data in the workspace.
        item_types: Item types to include ('table', 'figure', or both).
            Defaults to both.

    Returns:
        Formatted task prompt string.
    """
    if item_types is None:
        item_types = ["table", "figure"]
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

    # Build lookup for blinded extracted tables.
    # Key by exact table_id AND by normalized prefix (e.g. "Table 1") so that
    # "Table 1—Average Treatment Effects..." matches TableSpec "Table 1".
    blinded_lookup: dict[str, "ExtractedTable"] = {}
    for et in (summary.extracted_tables or []):
        blinded_lookup[et.table_id] = et
        # Also key by prefix up to first dash/colon/emdash after the number
        import re
        prefix_match = re.match(r"((?:Table|Figure)\s+\S+)", et.table_id)
        if prefix_match:
            blinded_lookup.setdefault(prefix_match.group(1), et)

    # Format table specs
    if summary.tables and "table" in item_types:
        table_parts = []
        for t in summary.tables:
            matched_et = blinded_lookup.get(t.table_number)

            if matched_et:
                # JSON template path: output as .json, show blinded structure
                table_filename = t.table_number.replace(" ", "_").lower() + ".json"
                script_name = t.table_number.replace(" ", "_").lower() + ".py"
                part = f"### {t.table_number}: {t.caption}\n"
                part += f"**Script**: `{script_name}` → **Output**: `{table_filename}`\n"
                part += (
                    "\n**JSON template** — A blinded table skeleton is provided in "
                    f"`table_templates/{t.table_number.replace(' ', '_').lower()}.json`. "
                    "For each cell, fill in `raw_text` (formatted value, e.g. "
                    "\"0.523***\", \"(0.102)\", \"Yes\") and `numeric_value` (float, "
                    "or null for non-numeric cells) based on your regression results. "
                    f"Write the completed table as `{table_filename}` using the same schema.\n"
                )
                # Show the blinded structure inline for reference
                blinded_json = json.dumps(matched_et.model_dump(), indent=2, default=str)
                part += f"\n<details><summary>Blinded table structure (click to expand)</summary>\n\n```json\n{blinded_json}\n```\n</details>\n"
            else:
                # No blinded template available — still use JSON output
                table_filename = t.table_number.replace(" ", "_").lower() + ".json"
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
            if t.data_processing_steps:
                part += "- **Table-specific processing steps**:\n"
                for s in t.data_processing_steps:
                    part += f"  {s.step_number}. {s.description}"
                    if s.variables_involved:
                        part += f" (variables: {', '.join(s.variables_involved)})"
                    part += "\n"
            if t.notes:
                part += f"- Notes: {t.notes}\n"
            table_parts.append(part)
        table_specs = "\n".join(table_parts)
    else:
        table_specs = "No tables specified."

    # Format figure specs
    if summary.figures and "figure" in item_types:
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
            if f.data_processing_steps:
                part += "- **Figure-specific processing steps**:\n"
                for s in f.data_processing_steps:
                    part += f"  {s.step_number}. {s.description}"
                    if s.variables_involved:
                        part += f" (variables: {', '.join(s.variables_involved)})"
                    part += "\n"
            if f.notes:
                part += f"- Notes: {f.notes}\n"
            if f.template_code:
                part += f"\n**Code skeleton** (use this as a starting point — you may adjust layout, colors, or labels to enhance readability):\n\n```python\n{f.template_code}\n```\n"
            fig_parts.append(part)
        figure_specs = "\n".join(fig_parts)
    else:
        figure_specs = "No figures specified."

    # Optional fields
    data_source = f"**Data Source**: {summary.data_source}\n" if summary.data_source else ""
    time_period = f"**Time Period**: {summary.time_period}\n" if summary.time_period else ""

    # Build items section conditionally
    items_parts = []
    if "table" in item_types and table_specs != "No tables specified.":
        items_parts.append(f"**Tables to Replicate**:\n{table_specs}")
    if "figure" in item_types and figure_specs != "No figures specified.":
        items_parts.append(f"**Figures to Replicate**:\n{figure_specs}")
    items_section = "\n".join(items_parts) + "\n" if items_parts else ""

    # Build conditional instruction blocks
    table_instructions = ""
    if "table" in item_types:
        table_instructions = (
            "\n   **Table output format**: Load the JSON template from `table_templates/`,\n"
            "   run your regressions, and fill in each cell's fields:\n"
            "   - `raw_text`: the formatted value as it would appear in the paper "
            '(e.g. "0.523***", "(0.102)", "Yes")\n'
            "   - `numeric_value`: the numeric value (float), or null if the cell is non-numeric\n"
            "   Write the completed table as `.json` using the same schema.\n"
        )
    figure_instructions = ""
    if "figure" in item_types:
        figure_instructions = "   Figures: write as `.png`.\n"

    return TASK_TEMPLATE.format(
        data_filename=data_filename,
        title=summary.title or summary.paper_id,
        paper_id=summary.paper_id,
        research_questions=rqs,
        data_description=summary.data_description,
        data_context=summary.data_context,
        data_source=data_source,
        time_period=time_period,
        processing_steps=steps,
        items_section=items_section,
        table_instructions=table_instructions,
        figure_instructions=figure_instructions,
    )


def _write_opencode_config(workspace_dir: Path) -> None:
    """Write opencode.json to auto-approve all tool calls except web access."""
    config = {
        "permission": {
            "*": "allow",
            "webfetch": "deny",
            "websearch": "deny",
        }
    }
    (workspace_dir / "opencode.json").write_text(json.dumps(config, indent=2))


def setup_workspace(
    paper: PaperSpec,
    paper_summary: PaperSummary,
    workspace_dir: Path,
    item_types: list[str] | None = None,
) -> str:
    """Set up an isolated workspace with data, TASK.md, and methodology JSON.

    Copies only the data into the workspace (no paper PDF, no replication package),
    writes the task prompt as TASK.md, and saves the methodology summary as JSON.

    Args:
        paper: Paper specification (used only for data_path).
        paper_summary: Pre-extracted methodology summary (no results).
        workspace_dir: Directory to set up.
        item_types: Item types to include ('table', 'figure', or both).

    Returns:
        The data_filename used in the task prompt.
    """
    if item_types is None:
        item_types = ["table", "figure"]
    workspace_dir.mkdir(parents=True, exist_ok=True)

    # Symlink data into workspace (avoids duplicating large datasets)
    data_src = Path(paper.data_path).resolve()
    if data_src.is_dir():
        data_dest = workspace_dir / "data"
        if data_dest.is_symlink():
            data_dest.unlink()
        elif data_dest.exists():
            shutil.rmtree(data_dest)
        data_dest.symlink_to(data_src)
        data_filename = "data/"
    elif data_src.exists():
        shutil.copy2(data_src, workspace_dir / data_src.name)
        data_filename = data_src.name
    else:
        data_filename = data_src.name

    # Write task prompt with methodology summary
    task_prompt = build_task_prompt(paper_summary, data_filename, item_types=item_types)
    (workspace_dir / "TASK.md").write_text(task_prompt)

    # Save the summary as JSON for reference
    (workspace_dir / "methodology_summary.json").write_text(
        json.dumps(paper_summary.model_dump(), indent=2, default=str)
    )

    # Write blinded JSON templates so replicator code can json.load() them.
    # Filenames use the normalized table_number from the TableSpec (e.g. "table_1.json"),
    # matched to extracted tables via the same prefix lookup used in build_task_prompt().
    if paper_summary.extracted_tables and "table" in item_types:
        import re
        blinded_lookup: dict[str, "ExtractedTable"] = {}
        for et in paper_summary.extracted_tables:
            blinded_lookup[et.table_id] = et
            prefix_match = re.match(r"((?:Table|Figure)\s+\S+)", et.table_id)
            if prefix_match:
                blinded_lookup.setdefault(prefix_match.group(1), et)

        templates_dir = workspace_dir / "table_templates"
        templates_dir.mkdir(parents=True, exist_ok=True)
        for t in paper_summary.tables:
            matched_et = blinded_lookup.get(t.table_number)
            if matched_et:
                fname = t.table_number.replace(" ", "_").lower() + ".json"
                (templates_dir / fname).write_text(
                    json.dumps(matched_et.model_dump(), indent=2, default=str)
                )

    # Write workspace instructions as both CLAUDE.md and AGENTS.md
    (workspace_dir / "CLAUDE.md").write_text(WORKSPACE_INSTRUCTIONS)
    (workspace_dir / "AGENTS.md").write_text(WORKSPACE_INSTRUCTIONS)

    # OpenCode project config: allow all permissions (symlinks, bash, etc.)
    _write_opencode_config(workspace_dir)

    return data_filename
