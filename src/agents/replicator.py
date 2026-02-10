"""Agent 2: Replicator - Generates code to replicate paper results."""

import json
from pathlib import Path
from typing import Optional

from ..models.schemas import (
    PaperSummary,
    ReplicationResults,
    GeneratedCode,
    GeneratedTable,
    GeneratedFigure,
)
from ..models.config import Config
from ..utils.code_executor import CodeExecutor, create_notebook_from_code
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

## Data Information:
- Data path: {data_path}
- Data format: {data_format}

## Research Context:
{data_context}

## Data Description:
{data_description}

## Data Processing Steps:
{processing_steps}

## Analysis to Replicate:
{analysis_spec}

## Requirements:
1. Load the data from the specified path
2. Apply all data processing steps in order
3. Implement the exact regression/analysis specification
4. Format results as a pandas DataFrame for tables
5. Save figures to the output directory
6. Include error handling and logging
7. Add comments explaining each step

Generate complete, executable Python code.
Wrap the code in a function called `run_analysis()` that returns the results.
The function should take `data_path` and `output_dir` as parameters.

```python
# Your code here
```"""


class ReplicatorAgent(BaseAgent):
    """Agent 2: Generates replication code and executes it.

    This agent receives methodological specifications and data,
    then generates and executes Python code to replicate the analysis.
    It does NOT have access to the original paper or results.
    """

    def __init__(self, config: Config):
        """Initialize the replicator agent.

        Args:
            config: Configuration object.
        """
        super().__init__(
            config=config,
            name="Replicator",
            role="data science replication specialist",
            goal="Generate and execute code to replicate paper analyses",
        )
        self.executor: Optional[CodeExecutor] = None

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

        # Save notebook
        self._save_as_notebook(results, output_path)

        logger.info(
            f"Replication complete: {len(results.tables)} tables, "
            f"{len(results.figures)} figures, {len(results.errors)} errors"
        )

        return results

    def _generate_setup_code(
        self, data_path: str, summary: PaperSummary
    ) -> GeneratedCode:
        """Generate setup/import code."""
        setup_code = f'''"""
Replication Setup for: {summary.paper_id}
Research Question: {summary.research_questions[0] if summary.research_questions else 'N/A'}
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Load data
DATA_PATH = "{data_path}"
try:
    if DATA_PATH.endswith('.csv'):
        df = pd.read_csv(DATA_PATH)
    elif DATA_PATH.endswith('.dta'):
        df = pd.read_stata(DATA_PATH)
    elif DATA_PATH.endswith('.xlsx') or DATA_PATH.endswith('.xls'):
        df = pd.read_excel(DATA_PATH)
    else:
        df = pd.read_csv(DATA_PATH)  # Default to CSV
    print(f"Data loaded: {{df.shape[0]}} rows, {{df.shape[1]}} columns")
    print(f"Columns: {{list(df.columns)}}")
except Exception as e:
    print(f"Error loading data: {{e}}")
    df = None
'''
        return GeneratedCode(
            language="python",
            code=setup_code,
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
        """Generate and execute code for a table."""
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

        # Generate code using LLM
        prompt = CODE_GENERATION_PROMPT.format(
            data_path=data_path,
            data_format=Path(data_path).suffix,
            data_context=summary.data_context,
            data_description=summary.data_description,
            processing_steps=processing_steps or "No specific steps listed",
            analysis_spec=analysis_spec,
        )

        try:
            response = self.generate(prompt, system_prompt=REPLICATOR_SYSTEM_PROMPT)

            # Extract code from response
            code = self._extract_code(response)

            # Execute code
            result = self.executor.execute(code)

            exec_logger.log_code_execution(
                code, result["success"], result["output"], result.get("error")
            )

            # Create GeneratedCode object
            gen_code = GeneratedCode(
                language="python",
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
        """Generate and execute code for a figure."""
        logger.info(f"Replicating {figure_spec.figure_number}")

        # Format processing steps
        processing_steps = "\n".join(
            f"{s.step_number}. {s.description}"
            for s in summary.data_processing_steps
        )

        analysis_spec = f"""
Figure: {figure_spec.figure_number}
Caption: {figure_spec.caption}
Plot type: {figure_spec.plot_type}
X-axis: {figure_spec.x_axis}
Y-axis: {figure_spec.y_axis}
Grouping variables: {', '.join(figure_spec.grouping_vars) if figure_spec.grouping_vars else 'None'}
Notes: {figure_spec.notes or 'None'}

Save the figure to: {output_path}/{figure_spec.figure_number.replace(' ', '_').lower()}.png
"""

        prompt = CODE_GENERATION_PROMPT.format(
            data_path=data_path,
            data_format=Path(data_path).suffix,
            data_context=summary.data_context,
            data_description=summary.data_description,
            processing_steps=processing_steps or "No specific steps listed",
            analysis_spec=analysis_spec,
        )

        try:
            response = self.generate(prompt, system_prompt=REPLICATOR_SYSTEM_PROMPT)
            code = self._extract_code(response)

            result = self.executor.execute(code)

            exec_logger.log_code_execution(
                code, result["success"], result["output"], result.get("error")
            )

            fig_filename = f"{figure_spec.figure_number.replace(' ', '_').lower()}.png"
            fig_path = output_path / fig_filename

            gen_code = GeneratedCode(
                language="python",
                code=code,
                dependencies=["matplotlib", "seaborn"],
                execution_order=len(summary.tables) + len(summary.figures),
                description=f"Code for {figure_spec.figure_number}",
            )

            gen_figure = GeneratedFigure(
                figure_number=figure_spec.figure_number,
                file_path=str(fig_path),
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

    def _extract_code(self, response: str) -> str:
        """Extract Python code from LLM response."""
        import re

        # Try to find code block
        code_match = re.search(r"```python\s*(.*?)\s*```", response, re.DOTALL)
        if code_match:
            return code_match.group(1)

        # Try generic code block
        code_match = re.search(r"```\s*(.*?)\s*```", response, re.DOTALL)
        if code_match:
            return code_match.group(1)

        # Return full response if no code blocks found
        return response

    def _save_as_notebook(self, results: ReplicationResults, output_path: Path) -> str:
        """Save all code as a Jupyter notebook."""
        code_blocks = [c.code for c in results.code_files]
        descriptions = [c.description for c in results.code_files]

        notebook_path = output_path / f"replication_{results.paper_id}.ipynb"
        return create_notebook_from_code(code_blocks, str(notebook_path), descriptions)
