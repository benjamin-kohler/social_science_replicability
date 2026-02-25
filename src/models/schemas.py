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
# Agent 3 Output: VerificationReport
# =============================================================================


class ReplicationGrade(str, Enum):
    """Grading scale for replication quality."""

    A = "A"  # Fully replicated the results
    B = "B"  # Same direction, small discrepancies (<5% difference)
    C = "C"  # Same direction, large discrepancies (>5% difference)
    D = "D"  # Different results, opposite direction or non-significant
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
