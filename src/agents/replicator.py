"""Agent 2: Replicator - Generates code to replicate paper results."""

import json
import re
from pathlib import Path
from typing import Optional

from langchain_core.language_models import BaseChatModel

from ..models.schemas import (
    PaperSummary,
    ReplicationResults,
    GeneratedCode,
    GeneratedTable,
    GeneratedFigure,
)
from ..models.config import Config
from ..utils.code_executor import CodeExecutor
from ..utils.logging_utils import get_logger, ExecutionLogger
from .base import BaseAgent

logger = get_logger(__name__)


REPLICATOR_SYSTEM_PROMPT = """You are an expert data scientist specializing in replicating social science research.

Your task is to generate Python code that replicates tables and figures from academic papers
based on methodological descriptions. You DO NOT have access to the original paper or its results.

Guidelines:
1. Generate clean, well-documented Python code
2. Use pandas, statsmodels, matplotlib, and seaborn
3. Follow the exact specifications provided
4. Handle missing data appropriately
5. Apply all specified data processing steps
6. Use robust standard errors where specified
7. Format output tables and figures professionally

IMPORTANT: You are working ONLY from methodological descriptions. Do not assume or
predict what the results should be - just implement the specified analysis correctly."""


CODE_GENERATION_PROMPT = """Generate Python code to replicate the analysis described below.

## Data
The data has been loaded in a previous cell as `df`. Do NOT reload it.
- Data path: {data_path}
{data_file_info}

## Actual Data Schema (from the loaded DataFrame)
{data_schema}

## Research Context:
{data_context}

## Data Description:
{data_description}

## Data Processing Steps:
{processing_steps}

## Analysis to Replicate:
{analysis_spec}

## Requirements:
1. `df` is already loaded — use it directly, do NOT reload
2. Apply all data processing steps in order
3. Implement the exact regression/analysis specification
4. Use ONLY column names that appear in the data schema above
5. For tables: print the results DataFrame with `print()` so the output is captured
6. For figures: use the ABSOLUTE path provided — do not construct your own path
7. Add comments explaining each step

Generate complete, directly executable Python code (NOT wrapped in a function).
Print all table results clearly so they can be captured.

```python
# Your code here
```"""


CODE_FIX_PROMPT = """The following Python code failed with an error. Fix the code so it runs successfully.

## Failed Code
```python
{failed_code}
```

## Error
```
{error}
```

## Data Schema (actual columns, dtypes, and sample values)
{data_schema}

## Original Task
{analysis_spec}

## Requirements:
1. `df` is already loaded — use it directly, do NOT reload
2. Use ONLY column names that appear in the data schema above
3. Fix the error while preserving the original analysis intent
4. For figures: use the ABSOLUTE path provided — do not construct your own path
5. Print all table results clearly so they can be captured

Generate the complete fixed code (not just the changed parts).

```python
# Your fixed code here
```"""


class ReplicatorAgent(BaseAgent):
    """Agent 2: Generates replication code and executes it.

    This agent receives methodological specifications and data,
    then generates and executes Python code to replicate the analysis.
    It does NOT have access to the original paper or results.
    """

    def __init__(self, config: Config, chat_model: Optional[BaseChatModel] = None):
        """Initialize the replicator agent.

        Args:
            config: Configuration object.
            chat_model: Optional LangChain chat model for DI/testing.
        """
        super().__init__(
            config=config,
            name="Replicator",
            role="data science replication specialist",
            goal="Generate and execute code to replicate paper analyses",
            chat_model=chat_model,
        )
        self.executor: Optional[CodeExecutor] = None
        self._data_schema_info: str = "Schema not yet captured."
        self._setup_code_str: str = ""

    def run(
        self,
        paper_summary: PaperSummary,
        data_path: str,
        output_dir: str = "data/output",
    ) -> ReplicationResults:
        """Generate and execute replication code.

        Args:
            paper_summary: Methodological summary from Agent 1.
            data_path: Path to the data file(s).
            output_dir: Directory for output files.

        Returns:
            ReplicationResults with generated tables and figures.
        """
        logger.info(f"Starting replication for: {paper_summary.paper_id}")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Initialize results
        results = ReplicationResults(
            paper_id=paper_summary.paper_id,
            code_files=[],
            tables=[],
            figures=[],
            execution_log="",
            errors=[],
        )

        # Initialize execution logger
        exec_logger = ExecutionLogger(paper_summary.paper_id, str(output_path))

        try:
            # Start code executor
            with CodeExecutor(
                timeout=self.config.execution.timeout_seconds,
                working_dir=str(output_path),
            ) as executor:
                self.executor = executor

                # Generate and execute setup code
                setup_code = self._generate_setup_code(data_path, paper_summary)
                results.code_files.append(setup_code)

                setup_result = executor.execute(setup_code.code)
                exec_logger.log_code_execution(
                    setup_code.code,
                    setup_result["success"],
                    setup_result["output"],
                    setup_result.get("error"),
                )

                if not setup_result["success"]:
                    results.errors.append(f"Setup failed: {setup_result['error']}")
                    logger.error(f"Setup failed: {setup_result['error']}")
                else:
                    # Capture data schema after successful setup
                    self._capture_data_schema()
                    self._setup_code_str = setup_code.code

                # Generate code for each table
                for i, table_spec in enumerate(paper_summary.tables):
                    table_result = self._replicate_table(
                        table_spec, paper_summary, data_path, output_path, exec_logger
                    )
                    if table_result:
                        results.tables.append(table_result["table"])
                        results.code_files.append(table_result["code"])
                        if table_result.get("error"):
                            results.errors.append(table_result["error"])

                # Generate code for each figure
                for i, figure_spec in enumerate(paper_summary.figures):
                    figure_result = self._replicate_figure(
                        figure_spec, paper_summary, data_path, output_path, exec_logger
                    )
                    if figure_result:
                        results.figures.append(figure_result["figure"])
                        results.code_files.append(figure_result["code"])
                        if figure_result.get("error"):
                            results.errors.append(figure_result["error"])

        except Exception as e:
            logger.error(f"Replication failed: {e}")
            results.errors.append(str(e))

        # Save execution log
        results.execution_log = exec_logger.get_full_log()
        exec_logger.save_log()

        logger.info(
            f"Replication complete: {len(results.tables)} tables, "
            f"{len(results.figures)} figures, {len(results.errors)} errors"
        )

        return results

    def _capture_data_schema(self) -> None:
        """Capture DataFrame schema (columns, dtypes, shape, sample rows) from the kernel."""
        parts = []

        shape_result = self.executor.execute("print(df.shape)")
        if shape_result["success"]:
            parts.append(f"Shape: {shape_result['output'].strip()}")

        dtypes_result = self.executor.execute("print(df.dtypes.to_string())")
        if dtypes_result["success"]:
            dtypes_text = dtypes_result["output"].strip()
            # Truncate to first 80 columns if very wide
            lines = dtypes_text.split("\n")
            if len(lines) > 80:
                dtypes_text = "\n".join(lines[:80]) + f"\n... ({len(lines) - 80} more columns)"
            parts.append(f"Columns and dtypes:\n{dtypes_text}")

        head_result = self.executor.execute("print(df.head(3).to_string())")
        if head_result["success"]:
            parts.append(f"Sample rows:\n{head_result['output'].strip()}")

        if parts:
            self._data_schema_info = "\n\n".join(parts)
            logger.info("Captured data schema from kernel")
        else:
            self._data_schema_info = "Could not capture data schema."
            logger.warning("Failed to capture data schema")

    @staticmethod
    def _strip_ansi(text: str) -> str:
        """Remove ANSI escape codes from text (e.g., Jupyter traceback colors)."""
        return re.sub(r"\x1b\[[0-9;]*m", "", text)

    def _save_script(
        self,
        code: str,
        label: str,
        output_path: Path,
        paper_id: str,
        success: bool,
    ) -> None:
        """Save a standalone .py script for a table or figure.

        Args:
            code: The analysis code that was executed.
            label: Label like 'table_1' or 'figure_2'.
            output_path: Directory to save into.
            paper_id: Paper identifier for the header comment.
            success: Whether the code executed successfully.
        """
        status = "SUCCESS" if success else "FAILED"
        header = (
            f"# Replication script: {label}\n"
            f"# Paper: {paper_id}\n"
            f"# Status: {status}\n"
            f"# Auto-generated by ReplicatorAgent\n\n"
        )

        # Build standalone script with setup boilerplate + analysis code
        script = header
        if self._setup_code_str:
            script += "# --- Setup and data loading ---\n"
            script += self._setup_code_str + "\n\n"
        script += "# --- Analysis ---\n"
        script += code + "\n"

        script_path = output_path / f"{label}.py"
        script_path.write_text(script)
        logger.info(f"Saved script: {script_path}")

    @staticmethod
    def _scan_data_files(data_path: str) -> str:
        """Scan data path and return a human-readable file listing."""
        import os

        path = Path(data_path)
        if not path.exists():
            return f"WARNING: Path does not exist: {data_path}"

        if path.is_file():
            size_mb = path.stat().st_size / 1e6
            return f"Single file: {path.name} ({size_mb:.1f} MB)"

        # It's a directory — walk it
        supported = ('.csv', '.dta', '.xlsx', '.xls', '.tsv', '.parquet', '.sas7bdat', '.sav')
        lines = []
        for root, dirs, files in os.walk(str(path)):
            rel_root = os.path.relpath(root, str(path))
            for f in sorted(files):
                ext = os.path.splitext(f)[1].lower()
                if ext in supported:
                    fp = os.path.join(root, f)
                    size_mb = os.path.getsize(fp) / 1e6
                    display_path = f if rel_root == '.' else os.path.join(rel_root, f)
                    lines.append(f"- {display_path} ({size_mb:.1f} MB, {ext})")
        if not lines:
            return f"Directory {data_path} contains no recognized data files."
        return f"Directory: {data_path}\n" + "\n".join(lines)

    def _generate_setup_code(
        self, data_path: str, summary: PaperSummary
    ) -> GeneratedCode:
        """Use the LLM to generate setup and data loading code."""
        file_listing = self._scan_data_files(data_path)

        setup_prompt = f"""Generate Python setup code for a replication analysis.

## Available Data
{file_listing}

## Research Context
**Paper**: {summary.title or summary.paper_id}
**Data description**: {summary.data_description}
**Data context**: {summary.data_context}
{f"**Data source**: {summary.data_source}" if summary.data_source else ""}

## Requirements
1. Import libraries: pandas, numpy, statsmodels (api + formula), matplotlib, seaborn, warnings
2. Suppress warnings and set pandas display options (max_columns=None, width=None)
3. Load the data file(s) into a DataFrame called `df`. Choose the right reader for the format (.dta → read_stata, .csv → read_csv, etc.)
4. If the analysis needs multiple files merged, do so and store the result in `df`
5. Print df.shape and df.columns so the next cells know what's available
6. The primary variable MUST be named `df` — all subsequent analysis cells depend on this

Generate directly executable Python code (no function wrapping).

```python
# Your setup code here
```"""

        response = self.generate(setup_prompt, system_prompt=REPLICATOR_SYSTEM_PROMPT)
        code, _ = self._extract_code(response)

        return GeneratedCode(
            language="python",
            code=code,
            dependencies=["pandas", "numpy", "statsmodels", "matplotlib", "seaborn"],
            execution_order=0,
            description="Setup and data loading",
        )

    def _replicate_table(
        self,
        table_spec,
        summary: PaperSummary,
        data_path: str,
        output_path: Path,
        exec_logger: ExecutionLogger,
    ) -> Optional[dict]:
        """Generate and execute code for a table with error-driven retries."""
        logger.info(f"Replicating {table_spec.table_number}")

        # Format processing steps
        processing_steps = "\n".join(
            f"{s.step_number}. {s.description}"
            for s in summary.data_processing_steps
        )

        # Format regression specs
        reg_specs = []
        for spec in table_spec.regression_specs:
            reg_specs.append(f"""
- Model type: {spec.model_type}
- Dependent variable: {spec.dependent_var}
- Independent variables: {', '.join(spec.independent_vars)}
- Controls: {', '.join(spec.controls) if spec.controls else 'None'}
- Fixed effects: {', '.join(spec.fixed_effects) if spec.fixed_effects else 'None'}
- Clustering: {spec.clustering or 'None'}
""")

        analysis_spec = f"""
Table: {table_spec.table_number}
Caption: {table_spec.caption}
Columns: {', '.join(table_spec.column_names)}
Rows: {', '.join(table_spec.row_names)}
Regression Specifications:
{''.join(reg_specs)}
Notes: {table_spec.notes or 'None'}
"""
        if table_spec.template_markdown:
            analysis_spec += f"""
Structural Template:
Your output MUST match this structure exactly. Replace XXX with computed values.
Use (XXX) for standard errors. Use --- for structurally empty cells.

{table_spec.template_markdown}
"""

        # Generate code using LLM
        data_file_info = self._scan_data_files(data_path)
        prompt = CODE_GENERATION_PROMPT.format(
            data_path=data_path,
            data_file_info=data_file_info,
            data_context=summary.data_context,
            data_description=summary.data_description,
            processing_steps=processing_steps or "No specific steps listed",
            analysis_spec=analysis_spec,
            data_schema=self._data_schema_info,
        )

        max_retries = self.config.execution.max_retries

        try:
            # Attempt 0: generate from original prompt
            response = self.generate(prompt, system_prompt=REPLICATOR_SYSTEM_PROMPT)
            code, language = self._extract_code(response)

            if language == "r":
                result = self._execute_r_code(code)
            else:
                result = self.executor.execute(code)

            exec_logger.log_code_execution(
                code, result["success"], result["output"], result.get("error")
            )

            # Retry loop: attempts 1..max_retries with error feedback
            attempt = 1
            while not result["success"] and language == "python" and attempt <= max_retries:
                logger.info(
                    f"Retry {attempt}/{max_retries} for {table_spec.table_number}"
                )
                error_text = self._strip_ansi(result.get("error", "Unknown error"))
                fix_prompt = CODE_FIX_PROMPT.format(
                    failed_code=code,
                    error=error_text,
                    data_schema=self._data_schema_info,
                    analysis_spec=analysis_spec,
                )
                fix_response = self.generate(
                    fix_prompt, system_prompt=REPLICATOR_SYSTEM_PROMPT
                )
                code, language = self._extract_code(fix_response)
                result = self.executor.execute(code)
                exec_logger.log_code_execution(
                    code, result["success"], result["output"], result.get("error")
                )
                attempt += 1

            # Last resort: try R if all Python attempts failed
            if not result["success"] and language == "python":
                logger.info(f"All Python retries failed for {table_spec.table_number}, trying R...")
                r_prompt = prompt.replace(
                    "Generate complete, directly executable Python code (NOT wrapped in a function).",
                    "Generate complete, executable R code.",
                ).replace("```python", "```r")
                r_response = self.generate(r_prompt, system_prompt=REPLICATOR_SYSTEM_PROMPT)
                r_code, _ = self._extract_code(r_response)
                r_result = self._execute_r_code(r_code)
                if r_result["success"]:
                    code, language, result = r_code, "r", r_result
                    exec_logger.log_code_execution(
                        code, result["success"], result["output"], result.get("error")
                    )

            # Save standalone script (use paper numbering for filenames)
            table_label = table_spec.table_number.replace(" ", "_").lower()
            self._save_script(
                code, table_label, output_path,
                summary.paper_id, result["success"],
            )

            # Save table output as CSV if execution succeeded
            if result["success"] and result["output"]:
                csv_path = output_path / f"{table_label}.csv"
                save_csv_code = (
                    f"try:\n"
                    f"    import pandas as _pd\n"
                    f"    # Try to save the last DataFrame result as CSV\n"
                    f"    _last_df = [v for v in dir() if isinstance(eval(v), _pd.DataFrame) and not v.startswith('_')]\n"
                    f"    if _last_df:\n"
                    f"        eval(_last_df[-1]).to_csv(r'{csv_path}', index=True)\n"
                    f"except Exception:\n"
                    f"    pass\n"
                )
                self.executor.execute(save_csv_code)

            # Create GeneratedCode object
            gen_code = GeneratedCode(
                language=language,
                code=code,
                dependencies=[],
                execution_order=len(summary.tables),
                description=f"Code for {table_spec.table_number}",
            )

            # Create GeneratedTable object
            gen_table = GeneratedTable(
                table_number=table_spec.table_number,
                data={"output": result["output"]},
                code_reference=table_spec.table_number,
                execution_success=result["success"],
                error_message=result.get("error"),
            )

            return {
                "table": gen_table,
                "code": gen_code,
                "error": result.get("error"),
            }

        except Exception as e:
            logger.error(f"Failed to replicate {table_spec.table_number}: {e}")
            return None

    def _replicate_figure(
        self,
        figure_spec,
        summary: PaperSummary,
        data_path: str,
        output_path: Path,
        exec_logger: ExecutionLogger,
    ) -> Optional[dict]:
        """Generate and execute code for a figure with error-driven retries."""
        logger.info(f"Replicating {figure_spec.figure_number}")

        # Format processing steps
        processing_steps = "\n".join(
            f"{s.step_number}. {s.description}"
            for s in summary.data_processing_steps
        )

        # Compute absolute save path for the figure
        fig_filename = f"{figure_spec.figure_number.replace(' ', '_').lower()}.png"
        fig_abs_path = (output_path / fig_filename).resolve()

        analysis_spec = f"""
Figure: {figure_spec.figure_number}
Caption: {figure_spec.caption}
Plot type: {figure_spec.plot_type}
X-axis: {figure_spec.x_axis}
Y-axis: {figure_spec.y_axis}
Grouping variables: {', '.join(figure_spec.grouping_vars) if figure_spec.grouping_vars else 'None'}
Notes: {figure_spec.notes or 'None'}

Save the figure using this EXACT absolute path: {fig_abs_path}
Use this EXACT absolute path for plt.savefig() — do not construct your own path.
"""
        if figure_spec.template_code:
            analysis_spec += f"""
Code skeleton (use as a starting point, fill in the data):

```python
{figure_spec.template_code}
```
"""

        data_file_info = self._scan_data_files(data_path)
        prompt = CODE_GENERATION_PROMPT.format(
            data_path=data_path,
            data_file_info=data_file_info,
            data_context=summary.data_context,
            data_description=summary.data_description,
            processing_steps=processing_steps or "No specific steps listed",
            analysis_spec=analysis_spec,
            data_schema=self._data_schema_info,
        )

        max_retries = self.config.execution.max_retries

        try:
            # Attempt 0: generate from original prompt
            response = self.generate(prompt, system_prompt=REPLICATOR_SYSTEM_PROMPT)
            code, language = self._extract_code(response)

            if language == "r":
                result = self._execute_r_code(code)
            else:
                result = self.executor.execute(code)

            exec_logger.log_code_execution(
                code, result["success"], result["output"], result.get("error")
            )

            # Retry loop: attempts 1..max_retries with error feedback
            attempt = 1
            while not result["success"] and language == "python" and attempt <= max_retries:
                logger.info(
                    f"Retry {attempt}/{max_retries} for {figure_spec.figure_number}"
                )
                error_text = self._strip_ansi(result.get("error", "Unknown error"))
                fix_prompt = CODE_FIX_PROMPT.format(
                    failed_code=code,
                    error=error_text,
                    data_schema=self._data_schema_info,
                    analysis_spec=analysis_spec,
                )
                fix_response = self.generate(
                    fix_prompt, system_prompt=REPLICATOR_SYSTEM_PROMPT
                )
                code, language = self._extract_code(fix_response)
                result = self.executor.execute(code)
                exec_logger.log_code_execution(
                    code, result["success"], result["output"], result.get("error")
                )
                attempt += 1

            # Last resort: try R if all Python attempts failed
            if not result["success"] and language == "python":
                logger.info(f"All Python retries failed for {figure_spec.figure_number}, trying R...")
                r_prompt = prompt.replace(
                    "Generate complete, directly executable Python code (NOT wrapped in a function).",
                    "Generate complete, executable R code.",
                ).replace("```python", "```r")
                r_response = self.generate(r_prompt, system_prompt=REPLICATOR_SYSTEM_PROMPT)
                r_code, _ = self._extract_code(r_response)
                r_result = self._execute_r_code(r_code)
                if r_result["success"]:
                    code, language, result = r_code, "r", r_result
                    exec_logger.log_code_execution(
                        code, result["success"], result["output"], result.get("error")
                    )

            # Save standalone script
            fig_label = figure_spec.figure_number.replace(" ", "_").lower()
            self._save_script(
                code, fig_label, output_path,
                summary.paper_id, result["success"],
            )

            gen_code = GeneratedCode(
                language=language,
                code=code,
                dependencies=["matplotlib", "seaborn"] if language == "python" else ["ggplot2"],
                execution_order=len(summary.tables) + len(summary.figures),
                description=f"Code for {figure_spec.figure_number}",
            )

            gen_figure = GeneratedFigure(
                figure_number=figure_spec.figure_number,
                file_path=str(fig_abs_path),
                code_reference=figure_spec.figure_number,
                execution_success=result["success"],
                error_message=result.get("error"),
            )

            return {
                "figure": gen_figure,
                "code": gen_code,
                "error": result.get("error"),
            }

        except Exception as e:
            logger.error(f"Failed to replicate {figure_spec.figure_number}: {e}")
            return None

    def _extract_code(self, response: str) -> tuple[str, str]:
        """Extract code from LLM response and detect language.

        Returns:
            Tuple of (code, language) where language is 'python' or 'r'.
        """
        # Try Python code block
        code_match = re.search(r"```python\s*(.*?)\s*```", response, re.DOTALL)
        if code_match:
            return self._clean_code(code_match.group(1)), "python"

        # Try R code block
        code_match = re.search(r"```r\s*(.*?)\s*```", response, re.DOTALL | re.IGNORECASE)
        if code_match:
            return self._clean_code(code_match.group(1)), "r"

        # Try generic code block
        code_match = re.search(r"```\s*(.*?)\s*```", response, re.DOTALL)
        if code_match:
            code = self._clean_code(code_match.group(1))
            # Heuristic: check if it looks like R
            if re.search(r"\blibrary\s*\(", code) or re.search(r"<-\s*", code):
                return code, "r"
            return code, "python"

        # Return full response if no code blocks found
        return self._clean_code(response), "python"

    @staticmethod
    def _clean_code(code: str) -> str:
        """Clean extracted code by removing backtick contamination and calling functions."""
        # Remove any residual markdown backtick lines
        lines = code.split("\n")
        cleaned = [line for line in lines if not re.match(r"^\s*```", line)]
        code = "\n".join(cleaned)

        # If code defines run_analysis() but never calls it, append a call
        if re.search(r"def\s+run_analysis\s*\(", code):
            # Check if run_analysis is actually called (not just defined)
            if not re.search(r"(?<!def\s)run_analysis\s*\(", code):
                code += '\n\n# Execute the analysis\nrun_analysis(DATA_PATH, ".")\n'

        return code.strip()

    def _execute_r_code(self, code: str) -> dict:
        """Execute R code using rpy2.

        Args:
            code: R code string to execute.

        Returns:
            Dict with 'success', 'output', and optional 'error' keys.
        """
        try:
            import rpy2.robjects as robjects
            result = robjects.r(code)
            return {"success": True, "output": str(result)}
        except ImportError:
            return {
                "success": False,
                "output": "",
                "error": "rpy2 not installed. Install with: pip install rpy2",
            }
        except Exception as e:
            return {"success": False, "output": "", "error": str(e)}
