"""Pydantic schemas for the replication system.

This module defines all data structures used for communication between agents,
including the LangGraph GraphState for workflow orchestration.
"""

import operator
from enum import Enum
from typing import Annotated, Any, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# =============================================================================
# Agent 1 Output: PaperSummary
# =============================================================================


class DataProcessingStep(BaseModel):
    """A single data processing step described in the paper."""

    step_number: int = Field(..., description="Order of this step in the processing pipeline")
    description: str = Field(..., description="Detailed description of what this step does")
    variables_involved: list[str] = Field(
        default_factory=list, description="Variables affected by this step"
    )


class RegressionSpec(BaseModel):
    """Specification for a regression model."""

    model_type: str = Field(
        ..., description="Type of regression: OLS, Logit, Probit, Fixed Effects, etc."
    )
    dependent_var: str = Field(..., description="The dependent/outcome variable")
    independent_vars: list[str] = Field(
        default_factory=list, description="Independent/explanatory variables"
    )
    controls: list[str] = Field(default_factory=list, description="Control variables")
    fixed_effects: Optional[list[str]] = Field(
        default=None, description="Fixed effects (e.g., year, entity)"
    )
    clustering: Optional[str] = Field(
        default=None, description="Clustering level for standard errors"
    )
    sample_restrictions: Optional[str] = Field(
        default=None, description="Any sample restrictions applied"
    )
    equation_latex: Optional[str] = Field(
        default=None,
        description="LaTeX formula for the regression equation, e.g. 'Y_i = \\alpha + \\beta X_i + \\gamma Z_i + \\varepsilon_i'",
    )
    variable_definitions: Optional[str] = Field(
        default=None,
        description="Verbal definitions of each variable in the equation, e.g. 'Y_i: acceptance of carbon tax (binary); X_i: believes does not lose (binary); Z_i: vector of controls'",
    )
    omitted_categories: Optional[dict[str, str]] = Field(
        default=None,
        description="Mapping of categorical variable names to their omitted/reference category, e.g. {'Yellow Vests': 'opposes', 'labor_status': 'Unemployed/Inactive'}",
    )
    additional_notes: Optional[str] = Field(
        default=None, description="Additional specifications or notes"
    )


class TableSpec(BaseModel):
    """Specification for a table in the paper (without actual results)."""

    table_number: str = Field(..., description="Table identifier (e.g., 'Table 1', 'Table A1')")
    caption: str = Field(..., description="Table caption/title")
    column_names: list[str] = Field(default_factory=list, description="Column headers")
    row_names: list[str] = Field(default_factory=list, description="Row labels")
    regression_specs: list[RegressionSpec] = Field(
        default_factory=list, description="Regression specifications for each column"
    )
    data_processing_steps: list[DataProcessingStep] = Field(
        default_factory=list,
        description="Data processing steps specific to this table (e.g., additional filtering, variable construction, subsample definitions). General steps shared across all tables go in PaperSummary.data_processing_steps.",
    )
    notes: Optional[str] = Field(default=None, description="Table notes (excluding results)")
    data_source: Optional[str] = Field(
        default=None,
        description="Data source for this specific table if different from the main dataset (e.g., 'EL 2013 housing survey, N=27,137')",
    )
    panel_structure: Optional[str] = Field(
        default=None, description="Panel structure if applicable (e.g., Panel A, Panel B)"
    )
    template_markdown: Optional[str] = Field(
        default=None,
        description="Markdown table template with XXX for values and --- for empty cells",
    )


class PlotSpec(BaseModel):
    """Specification for a figure/plot in the paper (without actual results)."""

    figure_number: str = Field(
        ..., description="Figure identifier (e.g., 'Figure 1', 'Figure A1')"
    )
    caption: str = Field(..., description="Figure caption/title")
    plot_type: str = Field(
        ..., description="Type of plot: scatter, bar, line, histogram, etc."
    )
    x_axis: Optional[str] = Field(default=None, description="X-axis variable or label")
    y_axis: Optional[str] = Field(default=None, description="Y-axis variable or label")
    grouping_vars: Optional[list[str]] = Field(
        default=None, description="Variables used for grouping/coloring"
    )
    regression_specs: list[RegressionSpec] = Field(
        default_factory=list,
        description="Regression specifications underlying this figure (e.g., for coefficient plots, RDD plots, binned scatters with fit lines)",
    )
    data_processing_steps: list[DataProcessingStep] = Field(
        default_factory=list,
        description="Data processing steps specific to this figure (e.g., aggregation, rolling averages, additional filtering). General steps shared across all figures go in PaperSummary.data_processing_steps.",
    )
    notes: Optional[str] = Field(default=None, description="Figure notes")
    data_source: Optional[str] = Field(
        default=None,
        description="Data source for this specific figure if different from the main dataset",
    )
    subplot_structure: Optional[str] = Field(
        default=None, description="Subplot arrangement if applicable"
    )
    template_code: Optional[str] = Field(
        default=None,
        description="Matplotlib code skeleton with axes/labels/legend but no data",
    )


class PaperSummary(BaseModel):
    """Complete methodological summary of a paper (Agent 1 output).

    This summary contains all information needed to replicate the paper's
    analysis WITHOUT revealing any actual results.
    """

    paper_id: str = Field(..., description="Unique identifier for the paper")
    title: Optional[str] = Field(default=None, description="Paper title")
    research_questions: list[str] = Field(
        default_factory=list, description="Main research questions addressed"
    )
    data_description: str = Field(
        ..., description="Description of the dataset(s) used"
    )
    data_context: str = Field(
        ..., description="Context and background relevant for the analysis"
    )
    data_source: Optional[str] = Field(
        default=None, description="Source of the data"
    )
    sample_size: Optional[str] = Field(
        default=None, description="Sample size information"
    )
    time_period: Optional[str] = Field(
        default=None, description="Time period covered by the data"
    )
    data_processing_steps: list[DataProcessingStep] = Field(
        default_factory=list, description="All data processing, filtering, and cleaning steps"
    )
    tables: list[TableSpec] = Field(
        default_factory=list, description="Specifications for all tables in main analysis"
    )
    figures: list[PlotSpec] = Field(
        default_factory=list, description="Specifications for all figures in main analysis"
    )
    extracted_tables: list["ExtractedTable"] = Field(
        default_factory=list,
        description="Blinded table structures from results extraction (no numeric values). "
                    "When present, replaces markdown templates for the replicator."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "paper_id": "smith2023education",
                "title": "The Effect of Class Size on Student Achievement",
                "research_questions": [
                    "Does reducing class size improve student test scores?"
                ],
                "data_description": "Administrative data from Tennessee STAR experiment",
                "data_context": "Randomized experiment in Tennessee elementary schools",
            }
        }
    }


# =============================================================================
# Agent 2 Output: ReplicationResults
# =============================================================================


class GeneratedCode(BaseModel):
    """Code generated by the replicator agent."""

    language: str = Field(..., description="Programming language: 'python' or 'r'")
    code: str = Field(..., description="The actual code")
    dependencies: list[str] = Field(
        default_factory=list, description="Required libraries/packages"
    )
    execution_order: int = Field(..., description="Order in which this code should run")
    description: Optional[str] = Field(
        default=None, description="What this code block does"
    )


class GeneratedTable(BaseModel):
    """A table generated by the replicator."""

    table_number: str = Field(..., description="Corresponding table number from PaperSummary")
    data: dict[str, Any] = Field(..., description="Table data as dictionary")
    format: str = Field(default="pandas_json", description="Data format")
    code_reference: str = Field(
        ..., description="Reference to the code that generated this table"
    )
    execution_success: bool = Field(default=True, description="Whether code executed successfully")
    error_message: Optional[str] = Field(default=None, description="Error if execution failed")
    replicated_extracted_table: Optional["ExtractedTable"] = Field(
        default=None,
        description="Structured table output when replicator uses JSON template"
    )


class GeneratedFigure(BaseModel):
    """A figure generated by the replicator."""

    figure_number: str = Field(..., description="Corresponding figure number from PaperSummary")
    file_path: str = Field(..., description="Path to the saved figure file")
    format: str = Field(default="png", description="Image format")
    code_reference: str = Field(
        ..., description="Reference to the code that generated this figure"
    )
    execution_success: bool = Field(default=True, description="Whether code executed successfully")
    error_message: Optional[str] = Field(default=None, description="Error if execution failed")


class ReplicationResults(BaseModel):
    """Complete results from the replicator agent (Agent 2 output)."""

    paper_id: str = Field(..., description="Paper identifier matching PaperSummary")
    code_files: list[GeneratedCode] = Field(
        default_factory=list, description="All generated code"
    )
    tables: list[GeneratedTable] = Field(
        default_factory=list, description="Generated tables"
    )
    figures: list[GeneratedFigure] = Field(
        default_factory=list, description="Generated figures"
    )
    execution_log: str = Field(default="", description="Full execution log")
    errors: list[str] = Field(default_factory=list, description="Any errors encountered")
    warnings: list[str] = Field(default_factory=list, description="Any warnings")


# =============================================================================
# Results Extraction: PaperResults (original table values from the paper)
# =============================================================================


class CellValue(BaseModel):
    """A single cell extracted from an original paper table."""

    row_label: str = Field(..., description="Row label as it appears in the paper")
    column_label: str = Field(..., description="Column label as it appears in the paper")
    raw_text: str = Field(..., description="Exact text as it appears in the paper cell")
    numeric_value: Optional[float] = Field(
        default=None, description="Parsed numeric value (None if non-numeric)"
    )
    is_standard_error: bool = Field(
        default=False, description="True if this is a standard error in parentheses"
    )
    significance_stars: int = Field(
        default=0, description="Number of significance stars (0, 1, 2, or 3)"
    )
    significance_level: Optional[float] = Field(
        default=None,
        description="Inferred significance level: 0.1, 0.05, 0.01, or None",
    )
    is_string: bool = Field(
        default=False, description="True if cell is non-numeric text (Yes/No/checkmark)"
    )
    row_type: str = Field(
        default="coefficient",
        description="One of: coefficient, se, statistic, string, panel_header",
    )


class ExtractedTable(BaseModel):
    """Complete extracted values for one table from the original paper."""

    table_id: str = Field(..., description="Table identifier matching PaperSummary (e.g., 'Table 1')")
    column_labels: list[str] = Field(default_factory=list, description="Ordered column headers")
    row_labels: list[str] = Field(default_factory=list, description="Ordered row labels")
    cells: list[CellValue] = Field(default_factory=list, description="All extracted cell values")
    significance_convention: Optional[str] = Field(
        default=None, description="E.g., '*** p<0.01, ** p<0.05, * p<0.1'"
    )
    notes: Optional[str] = Field(default=None, description="Any extraction notes or warnings")

    def to_csv(self, use_raw_text: bool = False) -> str:
        """Convert to a CSV string for inspection.

        Groups cells into rows by consecutive (row_label, is_standard_error)
        so that duplicate row_labels (common for SE rows labeled "( )") are
        preserved in order rather than deduplicated.

        Args:
            use_raw_text: If True, use raw_text. If False, use numeric_value
                          with stars appended for coefficients and parens for SEs.
        """
        import io
        import csv

        col_labels = self.column_labels or []

        # Group cells into rows by consecutive (row_label, is_se).
        # This preserves the ordering from the JSON and handles repeated
        # row_labels (e.g. two "( )" SE rows for different coefficients).
        CsvRow = tuple[str, bool, dict[str, "CellValue"]]  # (label, is_se, col→cell)
        rows: list[CsvRow] = []
        for cell in self.cells:
            key = (cell.row_label, cell.is_standard_error)
            # Append to current row if same group, otherwise start new row
            if rows and (rows[-1][0], rows[-1][1]) == key:
                rows[-1][2][cell.column_label] = cell
            else:
                rows.append((cell.row_label, cell.is_standard_error, {cell.column_label: cell}))

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([""] + col_labels)

        for row_label, is_se, col_cells in rows:
            # Panel headers
            sample_cell = next(iter(col_cells.values()), None)
            if sample_cell and sample_cell.row_type == "panel_header":
                writer.writerow([row_label] + [""] * len(col_labels))
                continue

            label = "(SE)" if is_se else row_label
            csv_row = [label]
            for cl in col_labels:
                cell = col_cells.get(cl)
                if cell is None:
                    csv_row.append("")
                elif use_raw_text:
                    csv_row.append(cell.raw_text)
                elif cell.is_string:
                    csv_row.append(cell.raw_text)
                elif cell.numeric_value is not None:
                    val = f"{cell.numeric_value}"
                    if not is_se and cell.significance_stars:
                        val += "*" * cell.significance_stars
                    csv_row.append(val)
                else:
                    csv_row.append(cell.raw_text)
            writer.writerow(csv_row)

        return buf.getvalue()

    def to_blinded(self) -> "ExtractedTable":
        """Return a copy with numeric values removed (blinded for the replicator).

        Keeps structural information: row/column labels, cell positions, row types.
        String cells (is_string=True) are kept as-is since they're structural (Yes/No).
        Numeric cells get numeric_value=None, raw_text="", significance cleared.
        """
        blinded_cells = []
        for cell in self.cells:
            if cell.is_string:
                blinded_cells.append(cell.model_copy())
            else:
                blinded_cells.append(cell.model_copy(update={
                    "numeric_value": None,
                    "raw_text": "",
                    "significance_stars": 0,
                    "significance_level": None,
                }))
        return ExtractedTable(
            table_id=self.table_id,
            column_labels=list(self.column_labels),
            row_labels=list(self.row_labels),
            cells=blinded_cells,
            significance_convention=self.significance_convention,
            notes=self.notes,
        )


class PaperResults(BaseModel):
    """All extracted numeric results from the original paper.

    Produced once per paper by the ResultsExtractor. Cached alongside PaperSummary.
    """

    paper_id: str = Field(..., description="Paper identifier")
    tables: list[ExtractedTable] = Field(default_factory=list, description="Extracted table values")
    extraction_model: str = Field(..., description="Model used for extraction")
    extraction_timestamp: Optional[str] = Field(default=None)

    def get_table(self, table_id: str) -> Optional["ExtractedTable"]:
        """Look up an extracted table by ID.

        Tries exact match first, then prefix match (e.g. "Table 1" matches
        "Table 1—Average Treatment Effects...").
        """
        for t in self.tables:
            if t.table_id == table_id:
                return t
        # Prefix match: table_id might have caption appended
        for t in self.tables:
            if t.table_id.startswith(table_id) and (
                len(t.table_id) == len(table_id)
                or t.table_id[len(table_id)] in "—–-:,"
            ):
                return t
        return None

    def export_csvs(self, output_dir: str, use_raw_text: bool = False) -> list[str]:
        """Export all extracted tables as CSV files for inspection.

        Args:
            output_dir: Directory to write CSV files to.
            use_raw_text: If True, use raw_text from the paper. If False,
                          use parsed numeric_value with stars/parens.

        Returns:
            List of written file paths.
        """
        from pathlib import Path
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = []
        for table in self.tables:
            fname = table.table_id.replace(" ", "_").lower() + ".csv"
            path = out / fname
            path.write_text(table.to_csv(use_raw_text=use_raw_text))
            paths.append(str(path))
        return paths


# =============================================================================
# Table Comparison: Cell-level comparison results
# =============================================================================


class CellComparison(BaseModel):
    """Comparison result for a single cell."""

    row_label: str = Field(..., description="Row label")
    column_label: str = Field(..., description="Column label")
    original_value: Optional[float] = Field(default=None, description="Original numeric value")
    replicated_value: Optional[float] = Field(default=None, description="Replicated numeric value")
    absolute_difference: Optional[float] = Field(default=None, description="|replicated - original|")
    percent_difference: Optional[float] = Field(default=None, description="Percentage difference")
    sign_match: Optional[bool] = Field(default=None, description="Whether signs match")
    grade: str = Field(default="F", description="Per-cell grade A-F")
    note: str = Field(default="", description="Explanation for this cell's grade")


class TableComparison(BaseModel):
    """Complete cell-by-cell comparison for one table."""

    table_id: str = Field(..., description="Table identifier")
    cell_comparisons: list[CellComparison] = Field(
        default_factory=list, description="All cell comparisons"
    )
    overall_grade: str = Field(default="F", description="Overall table grade A-F")
    summary: str = Field(default="", description="Comparison summary")
    alignment_notes: str = Field(
        default="", description="Notes on how rows/columns were aligned"
    )
    scale_factor: Optional[float] = Field(
        default=None,
        description="Global scale factor applied to replicated values (e.g. 100.0 if "
        "replicated was in proportions and original in %). None if no rescaling.",
    )
    scale_note: str = Field(
        default="", description="Explanation of any global rescaling applied"
    )
    row_scale_factors: dict[str, float] = Field(
        default_factory=dict,
        description="Per-row scale factors applied after global rescaling. "
        "Keys are row labels, values are the scale factor for that row.",
    )
    row_scale_notes: dict[str, str] = Field(
        default_factory=dict,
        description="Explanation for each per-row scale factor.",
    )
    comparison_code: str = Field(
        default="", description="Python code used to compute the comparison"
    )


# =============================================================================
# Agent 3 Output: VerificationReport
# =============================================================================


class ReplicationGrade(str, Enum):
    """Grading scale for replication quality."""

    A = "A"  # Fully replicated (<1% difference)
    B = "B"  # Same direction, small discrepancies (1-5%)
    C = "C"  # Same direction, moderate discrepancies (5-20%)
    D = "D"  # Same direction, large discrepancies (20-50%)
    E = "E"  # Different sign, significance, or >50% difference
    F = "F"  # Not comparable (missing output or incompatible format)


class ItemVerification(BaseModel):
    """Verification result for a single table or figure."""

    item_id: str = Field(..., description="Item identifier (e.g., 'table_1', 'figure_2')")
    item_type: str = Field(..., description="'table' or 'figure'")
    grade: ReplicationGrade = Field(..., description="Assigned grade")
    comparison_notes: str = Field(..., description="Detailed comparison notes")
    numerical_differences: Optional[dict[str, Any]] = Field(
        default=None, description="Quantified differences for tables"
    )
    key_findings_match: Optional[bool] = Field(
        default=None, description="Whether key findings/conclusions match"
    )
    table_comparison: Optional[TableComparison] = Field(
        default=None, description="Cell-by-cell comparison detail (tables only)"
    )
    judge_error: bool = Field(
        default=False,
        description="True when grade F was assigned due to a judge/LLM error, not a replication failure",
    )
    unverifiable: bool = Field(
        default=False,
        description="True when no output was produced to verify (execution failure, missing file). "
                    "Excluded from overall grade calculation.",
    )


class VerificationReport(BaseModel):
    """Complete verification report (Agent 3 output)."""

    paper_id: str = Field(..., description="Paper identifier")
    overall_grade: ReplicationGrade = Field(..., description="Overall replication grade")
    item_verifications: list[ItemVerification] = Field(
        default_factory=list, description="Per-item verification results"
    )
    summary: str = Field(..., description="Executive summary of verification")
    methodology_notes: Optional[str] = Field(
        default=None, description="Notes on comparison methodology used"
    )


# =============================================================================
# Agent 4 Output: ExplanationReport
# =============================================================================


class DiscrepancyAnalysis(BaseModel):
    """Analysis of a discrepancy for a non-A graded item."""

    item_id: str = Field(..., description="Item identifier")
    grade: ReplicationGrade = Field(..., description="Grade received")
    description_of_discrepancy: str = Field(
        ..., description="Detailed description of what differs"
    )
    likely_causes: list[str] = Field(
        default_factory=list, description="Possible reasons for the discrepancy"
    )
    is_identifiable: bool = Field(
        ..., description="Whether the cause can be definitively identified"
    )
    fault_attribution: str = Field(
        ..., description="'replicator', 'original_paper', 'unclear', or 'data_limitation'"
    )
    confidence: str = Field(..., description="Confidence level: 'high', 'medium', or 'low'")
    supporting_evidence: Optional[str] = Field(
        default=None, description="Evidence supporting the analysis"
    )


class ExplanationReport(BaseModel):
    """Complete explanation report for discrepancies (Agent 4 output)."""

    paper_id: str = Field(..., description="Paper identifier")
    analyses: list[DiscrepancyAnalysis] = Field(
        default_factory=list, description="Analysis for each non-A item"
    )
    overall_assessment: str = Field(
        ..., description="Overall assessment of the replication effort"
    )
    recommendations: list[str] = Field(
        default_factory=list, description="Recommendations for improvement"
    )
    replication_package_comparison: Optional[str] = Field(
        default=None, description="Comparison with original replication package if available"
    )


# =============================================================================
# Agentic Explainer Output: AgenticExplanationReport
# =============================================================================


class CodeComparison(BaseModel):
    """Side-by-side comparison of replicator vs. original code for one item."""

    item_id: str = Field(..., description="Item identifier (e.g., 'Table 1')")
    replicator_approach: str = Field(
        ..., description="Summary of what the replicator's code does"
    )
    original_approach: str = Field(
        ..., description="Summary of what the original replication package code does"
    )
    key_differences: list[str] = Field(
        default_factory=list, description="Specific code-level differences identified"
    )


class AgenticDiscrepancyAnalysis(BaseModel):
    """Deep analysis of a discrepancy, produced by the agentic Explainer."""

    item_id: str = Field(..., description="Item identifier (e.g., 'Table 1')")
    grade: ReplicationGrade = Field(..., description="Grade from the judge")
    verbal_explanation: str = Field(
        ..., description="Multi-paragraph root cause analysis"
    )
    code_comparison: Optional[CodeComparison] = Field(
        default=None,
        description="Detailed code comparison (if original replication package available)",
    )
    fault_category: str = Field(
        ...,
        description="One of: replicator, extractor, original_authors, data_limitation, software_differences",
    )
    fault_explanation: str = Field(
        ..., description="Why this fault category was chosen"
    )
    confidence: str = Field(..., description="Confidence level: high, medium, or low")
    supporting_evidence: list[str] = Field(
        default_factory=list,
        description="Specific file references, line numbers, variable names, etc.",
    )
    suggested_fix: Optional[str] = Field(
        default=None,
        description="What the replicator could have done differently to get a better grade",
    )


class AgenticExplanationReport(BaseModel):
    """Deep explanation report produced by the agentic Explainer phase."""

    paper_id: str = Field(..., description="Paper identifier")
    analyses: list[AgenticDiscrepancyAnalysis] = Field(
        default_factory=list, description="Per-item deep analysis for non-A items"
    )
    overall_assessment: str = Field(
        ..., description="Overall synthesis of discrepancy patterns"
    )
    methodology_quality_notes: str = Field(
        ..., description="Assessment of the methodology extraction quality"
    )
    fault_summary: dict[str, int] = Field(
        default_factory=dict,
        description="Count of items per fault category, e.g. {'replicator': 3, 'extractor': 2}",
    )
    runner_model: str = Field(..., description="Model that ran the explanation")
    runner_type: str = Field(
        ..., description="CLI runner type: 'claude-code' or 'codex'"
    )
    duration_seconds: float = Field(default=0.0, description="Wall-clock duration")
    usage: Optional[dict] = Field(default=None, description="Token usage summary")


# =============================================================================
# Workflow State
# =============================================================================


class ReplicationState(BaseModel):
    """State object that flows through the agent workflow."""

    # Input paths
    paper_pdf_path: str = Field(..., description="Path to the paper PDF")
    data_path: str = Field(..., description="Path to the data files")
    replication_package_path: Optional[str] = Field(
        default=None, description="Path to original replication package"
    )

    # Agent outputs
    paper_summary: Optional[PaperSummary] = Field(
        default=None, description="Output from Agent 1"
    )
    replication_results: Optional[ReplicationResults] = Field(
        default=None, description="Output from Agent 2"
    )
    verification_report: Optional[VerificationReport] = Field(
        default=None, description="Output from Agent 3"
    )
    explanation_report: Optional[ExplanationReport] = Field(
        default=None, description="Output from Agent 4"
    )

    # Metadata
    errors: list[str] = Field(default_factory=list, description="Accumulated errors")
    warnings: list[str] = Field(default_factory=list, description="Accumulated warnings")
    current_step: Optional[str] = Field(default=None, description="Current workflow step")


# =============================================================================
# Collector (Step 0) - Paper Entry
# =============================================================================


class PaperEntry(BaseModel):
    """Metadata for a paper to be processed by the Collector agent."""

    paper_id: str = Field(..., description="Unique identifier for the paper")
    pdf_path: str = Field(..., description="Path to the paper PDF file")
    data_paths: list[str] = Field(
        default_factory=list, description="Paths to associated data files"
    )
    replication_package_path: Optional[str] = Field(
        default=None, description="Path to original replication package"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata (authors, year, etc.)"
    )


# =============================================================================
# LangGraph State
# =============================================================================


class GraphState(TypedDict, total=False):
    """State for the LangGraph replication workflow.

    Uses Annotated types with operator.add for accumulation of errors/warnings
    across graph nodes.
    """

    # Input paths
    paper_pdf_path: str
    data_path: str
    output_dir: str
    paper_id: str
    replication_package_path: Optional[str]

    # Agent outputs (set by individual nodes)
    paper_summary: Optional[PaperSummary]
    replication_results: Optional[ReplicationResults]
    verification_report: Optional[VerificationReport]
    explanation_report: Optional[ExplanationReport]

    # Accumulating metadata
    errors: Annotated[list[str], operator.add]
    warnings: Annotated[list[str], operator.add]
    current_step: str

    # Flow control
    success: bool
