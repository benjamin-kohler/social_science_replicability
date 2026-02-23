"""Freestyle approach runner using opencode."""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from ..models.schemas import PaperSummary
from ..utils.logging_utils import get_logger
from .config import ModelSpec, PaperSpec
from .results import RunArtifacts

logger = get_logger(__name__)

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

## Instructions

1. **First, explore the data**: Use bash to run `head` and `python -c "import pandas as pd; df = pd.read_csv('...');  print(df.columns.tolist()); print(df.shape); print(df.dtypes)"` to understand the actual column names and data structure.

2. **Write Python code** to replicate each table and figure:
   - Use the ACTUAL column names from the data (step 1). Do NOT guess column names.
   - Load and clean the data following the data processing steps.
   - Implement the statistical models exactly as described.
   - Generate output tables as CSV files using the paper's numbering (e.g., `table_2.1.csv`, `table_3.1.csv`). See per-table instructions below.
   - Generate figures as PNG files using the paper's numbering (e.g., `figure_1.png`, `figure_2.1.png`). See per-figure instructions below.
   - Use `statsmodels` for regressions (OLS, Logit, IV/2SLS). Do NOT implement OLS manually.

3. **CRITICAL: Execute all code using bash** and fix any errors. Do not stop after writing the code — you MUST run it with `python <script>.py` and verify the output files exist.

4. **Save all outputs** in the current working directory.

You may search the internet for documentation on statistical methods or library
APIs, but do NOT search for or reference the original paper's results.

Focus on substance and accuracy. Match the described methodology as closely as
possible, including sample restrictions, variable transformations, and
statistical specifications.
"""


class OpencodeRunner:
    """Runs a freestyle replication using the opencode CLI.

    Creates an isolated workspace with only the methodology summary and data,
    then invokes opencode to let the model figure out the replication.
    The model does NOT receive the original paper PDF or replication package.
    """

    def __init__(self, opencode_binary: str = "opencode", timeout: int = 600):
        self.opencode_binary = opencode_binary
        self.timeout = timeout

    def run(
        self,
        model: ModelSpec,
        paper: PaperSpec,
        paper_summary: PaperSummary,
        workspace_dir: Path,
    ) -> RunArtifacts:
        """Run a freestyle replication from a methodology summary.

        Args:
            model: Model specification.
            paper: Paper specification (used only for data_path).
            paper_summary: Pre-extracted methodology summary (no results).
            workspace_dir: Isolated workspace directory for this run.

        Returns:
            RunArtifacts with workspace contents, stdout, stderr, exit code, duration.
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
        task_prompt = self._build_task_prompt(paper_summary, data_filename)
        (workspace_dir / "TASK.md").write_text(task_prompt)

        # Also save the summary as JSON for reference
        (workspace_dir / "methodology_summary.json").write_text(
            json.dumps(paper_summary.model_dump(), indent=2, default=str)
        )

        # Run opencode
        logger.info(
            f"Running opencode freestyle: model={model.model_name}, paper={paper.paper_id}"
        )
        start = time.time()

        try:
            # opencode CLI syntax: opencode run -m provider/model --dir workspace "message"
            model_id = f"{model.provider}/{model.model_name}"
            abs_workspace = str(Path(workspace_dir).resolve())
            result = subprocess.run(
                [
                    self.opencode_binary, "run",
                    "--print-logs",
                    "-m", model_id,
                    "--dir", abs_workspace,
                    "-f", "TASK.md",
                    "--",
                    "Read TASK.md. First explore the data files in data/ to learn the actual "
                    "column names. Then write Python scripts to replicate each table and figure. "
                    "You MUST execute the scripts with bash and fix any errors until they run "
                    "successfully. Use the exact output filenames specified in TASK.md for each item.",
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={**os.environ, "PATH": f"{Path.home() / '.opencode' / 'bin'}:{os.environ.get('PATH', '')}"},
            )
            stdout = result.stdout
            stderr = result.stderr
            exit_code = result.returncode
        except subprocess.TimeoutExpired:
            logger.warning(f"Opencode timed out after {self.timeout}s")
            stdout = ""
            stderr = f"Timed out after {self.timeout} seconds"
            exit_code = -1
        except FileNotFoundError:
            logger.error(f"opencode binary not found: {self.opencode_binary}")
            stdout = ""
            stderr = f"opencode binary not found: {self.opencode_binary}"
            exit_code = -2

        duration = time.time() - start

        logger.info(
            f"Opencode finished: exit_code={exit_code}, duration={duration:.1f}s"
        )

        return RunArtifacts(
            workspace_dir=str(workspace_dir),
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            duration_seconds=duration,
        )

    def _build_task_prompt(self, summary: PaperSummary, data_filename: str) -> str:
        """Generate the task prompt from a methodology summary."""
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
                part = f"### {t.table_number}: {t.caption}\n"
                part += f"**Output filename**: `{table_filename}`\n"
                if t.column_names:
                    part += f"- Columns: {', '.join(t.column_names)}\n"
                if t.row_names:
                    part += f"- Rows: {', '.join(t.row_names)}\n"
                if t.panel_structure:
                    part += f"- Panel structure: {t.panel_structure}\n"
                for spec in t.regression_specs:
                    part += f"- Regression: {spec.model_type}, DV={spec.dependent_var}"
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
                if t.notes:
                    part += f"- Notes: {t.notes}\n"
                if t.template_markdown:
                    part += f"\n**Structural template** (your output MUST match this structure — replace XXX with computed values):\n\n{t.template_markdown}\n"
                table_parts.append(part)
            table_specs = "\n".join(table_parts)
        else:
            table_specs = "No tables specified."

        # Format figure specs
        if summary.figures:
            fig_parts = []
            for f in summary.figures:
                fig_filename = f.figure_number.replace(" ", "_").lower() + ".png"
                part = f"### {f.figure_number}: {f.caption}\n"
                part += f"**Output filename**: `{fig_filename}`\n"
                part += f"- Plot type: {f.plot_type}\n"
                part += f"- X-axis: {f.x_axis}, Y-axis: {f.y_axis}\n"
                if f.grouping_vars:
                    part += f"- Grouping: {', '.join(f.grouping_vars)}\n"
                if f.subplot_structure:
                    part += f"- Subplots: {f.subplot_structure}\n"
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
