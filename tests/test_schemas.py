"""Tests for Pydantic data models."""

import pytest
from pydantic import ValidationError

from src.models.schemas import (
    DataProcessingStep,
    DiscrepancyAnalysis,
    ExplanationReport,
    GeneratedCode,
    GeneratedFigure,
    GeneratedTable,
    GraphState,
    ItemVerification,
    PaperEntry,
    PaperSummary,
    PlotSpec,
    RegressionSpec,
    ReplicationGrade,
    ReplicationResults,
    ReplicationState,
    TableSpec,
    VerificationReport,
)


class TestDataProcessingStep:
    def test_create(self):
        step = DataProcessingStep(
            step_number=1,
            description="Drop missing values",
            variables_involved=["x", "y"],
        )
        assert step.step_number == 1
        assert step.description == "Drop missing values"
        assert step.variables_involved == ["x", "y"]

    def test_defaults(self):
        step = DataProcessingStep(step_number=1, description="test")
        assert step.variables_involved == []

    def test_missing_required(self):
        with pytest.raises(ValidationError):
            DataProcessingStep(step_number=1)


class TestRegressionSpec:
    def test_create_full(self):
        spec = RegressionSpec(
            model_type="OLS",
            dependent_var="y",
            independent_vars=["x1", "x2"],
            controls=["age"],
            fixed_effects=["year"],
            clustering="state",
            sample_restrictions="age > 18",
        )
        assert spec.model_type == "OLS"
        assert spec.clustering == "state"

    def test_defaults(self):
        spec = RegressionSpec(model_type="OLS", dependent_var="y")
        assert spec.independent_vars == []
        assert spec.controls == []
        assert spec.fixed_effects is None
        assert spec.clustering is None


class TestTableSpec:
    def test_create(self):
        table = TableSpec(
            table_number="Table 1",
            caption="Main results",
            column_names=["(1)", "(2)"],
            row_names=["treatment", "control"],
        )
        assert table.table_number == "Table 1"

    def test_with_regression_specs(self):
        table = TableSpec(
            table_number="Table 2",
            caption="Regressions",
            regression_specs=[
                RegressionSpec(model_type="OLS", dependent_var="y"),
            ],
        )
        assert len(table.regression_specs) == 1


class TestPlotSpec:
    def test_create(self):
        plot = PlotSpec(
            figure_number="Figure 1",
            caption="Trend plot",
            plot_type="line",
            x_axis="Year",
            y_axis="Value",
        )
        assert plot.plot_type == "line"

    def test_with_grouping(self):
        plot = PlotSpec(
            figure_number="Figure 2",
            caption="Group comparison",
            plot_type="bar",
            x_axis="Category",
            y_axis="Count",
            grouping_vars=["gender", "age_group"],
        )
        assert len(plot.grouping_vars) == 2


class TestPaperSummary:
    def test_create_minimal(self):
        summary = PaperSummary(
            paper_id="test2024",
            data_description="Some data",
            data_context="Some context",
        )
        assert summary.paper_id == "test2024"
        assert summary.tables == []
        assert summary.figures == []

    def test_serialization(self, paper_summary):
        d = paper_summary.model_dump()
        assert d["paper_id"] == "test_paper_2024"
        assert len(d["tables"]) == 2
        assert len(d["figures"]) == 1

        # Round-trip
        restored = PaperSummary(**d)
        assert restored.paper_id == paper_summary.paper_id


class TestReplicationGrade:
    def test_values(self):
        assert ReplicationGrade.A.value == "A"
        assert ReplicationGrade.F.value == "F"

    def test_from_string(self):
        grade = ReplicationGrade("B")
        assert grade == ReplicationGrade.B

    def test_invalid(self):
        with pytest.raises(ValueError):
            ReplicationGrade("X")


class TestGeneratedTable:
    def test_create(self):
        table = GeneratedTable(
            table_number="Table 1",
            data={"col1": [1, 2], "col2": [3, 4]},
            code_reference="table_1",
        )
        assert table.execution_success is True

    def test_failed_execution(self):
        table = GeneratedTable(
            table_number="Table 1",
            data={},
            code_reference="table_1",
            execution_success=False,
            error_message="Import error",
        )
        assert not table.execution_success


class TestReplicationResults:
    def test_create(self, replication_results):
        assert replication_results.paper_id == "test_paper_2024"
        assert len(replication_results.tables) == 2
        assert len(replication_results.figures) == 1


class TestVerificationReport:
    def test_create(self, verification_report):
        assert verification_report.overall_grade == ReplicationGrade.B
        assert len(verification_report.item_verifications) == 3

    def test_serialization(self, verification_report):
        d = verification_report.model_dump()
        assert d["overall_grade"] == "B"


class TestDiscrepancyAnalysis:
    def test_create(self):
        analysis = DiscrepancyAnalysis(
            item_id="Table 2",
            grade=ReplicationGrade.C,
            description_of_discrepancy="Coefficients differ by 15%",
            likely_causes=["Different standard error computation"],
            is_identifiable=True,
            fault_attribution="replicator",
            confidence="high",
        )
        assert analysis.is_identifiable
        assert analysis.fault_attribution == "replicator"


class TestReplicationState:
    def test_initial_state(self):
        state = ReplicationState(
            paper_pdf_path="paper.pdf",
            data_path="data.csv",
        )
        assert state.paper_summary is None
        assert state.replication_results is None
        assert state.errors == []

    def test_with_results(self, paper_summary, verification_report):
        state = ReplicationState(
            paper_pdf_path="paper.pdf",
            data_path="data.csv",
            paper_summary=paper_summary,
            verification_report=verification_report,
            current_step="verification",
        )
        assert state.paper_summary is not None
        assert state.current_step == "verification"


class TestPaperEntry:
    def test_create_minimal(self):
        entry = PaperEntry(
            paper_id="smith2024",
            pdf_path="/path/to/paper.pdf",
        )
        assert entry.paper_id == "smith2024"
        assert entry.data_paths == []
        assert entry.replication_package_path is None
        assert entry.metadata == {}

    def test_create_full(self):
        entry = PaperEntry(
            paper_id="smith2024",
            pdf_path="/path/to/paper.pdf",
            data_paths=["/path/to/data.csv", "/path/to/data2.dta"],
            replication_package_path="/path/to/package/",
            metadata={"authors": ["Smith", "Jones"], "year": 2024},
        )
        assert len(entry.data_paths) == 2
        assert entry.metadata["year"] == 2024

    def test_missing_required(self):
        with pytest.raises(ValidationError):
            PaperEntry(paper_id="test")  # Missing pdf_path


class TestGraphState:
    def test_is_typed_dict(self):
        """GraphState should be a TypedDict, not a Pydantic model."""
        state: GraphState = {
            "paper_pdf_path": "paper.pdf",
            "data_path": "data.csv",
            "output_dir": "output/",
            "paper_id": "test",
            "errors": [],
            "warnings": [],
            "current_step": "starting",
            "success": True,
        }
        assert state["paper_pdf_path"] == "paper.pdf"
        assert state["errors"] == []

    def test_errors_accumulation_annotation(self):
        """Verify errors field has the operator.add annotation for LangGraph."""
        import operator
        from typing import get_type_hints, Annotated

        hints = get_type_hints(GraphState, include_extras=True)
        errors_hint = hints["errors"]
        # Check it's Annotated with operator.add
        assert hasattr(errors_hint, "__metadata__")
        assert operator.add in errors_hint.__metadata__
