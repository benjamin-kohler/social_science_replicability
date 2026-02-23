"""Shared fixtures for the test suite."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.models.config import Config, LangGraphConfig, ExecutionConfig, VerificationConfig
from src.models.schemas import (
    DataProcessingStep,
    DiscrepancyAnalysis,
    ExplanationReport,
    GeneratedCode,
    GeneratedFigure,
    GeneratedTable,
    ItemVerification,
    PaperSummary,
    PlotSpec,
    RegressionSpec,
    ReplicationGrade,
    ReplicationResults,
    ReplicationState,
    TableSpec,
    VerificationReport,
)


@pytest.fixture
def config():
    """Create a test configuration."""
    return Config(
        langgraph=LangGraphConfig(
            default_provider="openai",
            default_model="gpt-5.3-codex",
            temperature=0.1,
            max_tokens=4000,
        ),
        execution=ExecutionConfig(timeout_seconds=60),
        verification=VerificationConfig(numerical_tolerance=0.01, use_vision_model=False),
        openai_api_key="test-key",
        anthropic_api_key="test-key",
    )


@pytest.fixture
def paper_summary():
    """Create a sample PaperSummary."""
    return PaperSummary(
        paper_id="test_paper_2024",
        title="The Effect of X on Y",
        research_questions=["Does X affect Y?"],
        data_description="Panel data from country Z, 2000-2020",
        data_context="Natural experiment in country Z",
        data_source="Administrative records",
        sample_size="10,000 observations",
        time_period="2000-2020",
        data_processing_steps=[
            DataProcessingStep(
                step_number=1,
                description="Drop observations with missing values in outcome variable",
                variables_involved=["outcome_var"],
            ),
            DataProcessingStep(
                step_number=2,
                description="Winsorize continuous variables at 1st and 99th percentiles",
                variables_involved=["income", "spending"],
            ),
        ],
        tables=[
            TableSpec(
                table_number="Table 1",
                caption="Summary Statistics",
                column_names=["Mean", "Std. Dev.", "Min", "Max", "N"],
                row_names=["outcome_var", "treatment", "income", "age"],
                regression_specs=[],
            ),
            TableSpec(
                table_number="Table 2",
                caption="Main Regression Results",
                column_names=["(1)", "(2)", "(3)"],
                row_names=["treatment", "income", "age", "Observations", "R-squared"],
                regression_specs=[
                    RegressionSpec(
                        model_type="OLS",
                        dependent_var="outcome_var",
                        independent_vars=["treatment"],
                        controls=["income", "age"],
                        fixed_effects=["year"],
                        clustering="region",
                    )
                ],
            ),
        ],
        figures=[
            PlotSpec(
                figure_number="Figure 1",
                caption="Treatment Effect Over Time",
                plot_type="line",
                x_axis="Year",
                y_axis="Outcome",
                grouping_vars=["treatment_group"],
            ),
        ],
    )


@pytest.fixture
def replication_results():
    """Create sample ReplicationResults."""
    return ReplicationResults(
        paper_id="test_paper_2024",
        code_files=[
            GeneratedCode(
                language="python",
                code="import pandas as pd\ndf = pd.read_csv('data.csv')",
                dependencies=["pandas"],
                execution_order=0,
                description="Setup and data loading",
            ),
            GeneratedCode(
                language="python",
                code="result = df.describe()",
                dependencies=["pandas"],
                execution_order=1,
                description="Code for Table 1",
            ),
        ],
        tables=[
            GeneratedTable(
                table_number="Table 1",
                data={"output": "summary stats output"},
                code_reference="Table 1",
                execution_success=True,
            ),
            GeneratedTable(
                table_number="Table 2",
                data={"output": "regression output"},
                code_reference="Table 2",
                execution_success=True,
            ),
        ],
        figures=[
            GeneratedFigure(
                figure_number="Figure 1",
                file_path="/tmp/figure_1.png",
                code_reference="Figure 1",
                execution_success=True,
            ),
        ],
    )


@pytest.fixture
def verification_report():
    """Create a sample VerificationReport."""
    return VerificationReport(
        paper_id="test_paper_2024",
        overall_grade=ReplicationGrade.B,
        item_verifications=[
            ItemVerification(
                item_id="Table 1",
                item_type="table",
                grade=ReplicationGrade.A,
                comparison_notes="Summary statistics match exactly.",
                key_findings_match=True,
            ),
            ItemVerification(
                item_id="Table 2",
                item_type="table",
                grade=ReplicationGrade.B,
                comparison_notes="Coefficients within 3% of original.",
                numerical_differences={"max_difference_percent": 3.2},
                key_findings_match=True,
            ),
            ItemVerification(
                item_id="Figure 1",
                item_type="figure",
                grade=ReplicationGrade.C,
                comparison_notes="Trends match but magnitudes differ.",
                key_findings_match=False,
            ),
        ],
        summary="Overall grade: B. 3 items verified.",
    )


@pytest.fixture
def tmp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_pdf(tmp_dir):
    """Create a minimal PDF file for testing.

    This creates a valid one-page PDF with some text content. Uses
    raw PDF syntax to avoid additional dependencies.
    """
    pdf_path = tmp_dir / "test_paper.pdf"
    # Minimal valid PDF with text content
    content = """%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT /F1 12 Tf 100 700 Td (Abstract) Tj ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000266 00000 n
0000000360 00000 n
trailer
<< /Size 6 /Root 1 0 R >>
startxref
441
%%EOF"""
    pdf_path.write_text(content)
    return str(pdf_path)
