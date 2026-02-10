"""Tests for the agent classes.

These tests mock LLM calls to test agent logic without requiring API keys.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from src.agents.base import BaseAgent
from src.agents.extractor import ExtractorAgent
from src.agents.replicator import ReplicatorAgent
from src.agents.verifier import VerifierAgent
from src.agents.explainer import ExplainerAgent
from src.models.schemas import (
    ReplicationGrade,
    ItemVerification,
    VerificationReport,
)


# ── BaseAgent Tests ──────────────────────────────────────────────────────


class ConcreteAgent(BaseAgent):
    """Concrete implementation for testing the abstract base class."""

    def run(self, **kwargs):
        return "ran"


class TestBaseAgent:
    def test_init(self, config):
        agent = ConcreteAgent(
            config=config, name="Test", role="tester", goal="test things"
        )
        assert agent.name == "Test"
        assert agent.provider == "openai"

    def test_run(self, config):
        agent = ConcreteAgent(config=config, name="Test", role="tester", goal="test")
        assert agent.run() == "ran"

    def test_generate_openai(self, config):
        agent = ConcreteAgent(config=config, name="Test", role="tester", goal="test")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello from LLM"

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        agent._client = mock_client

        result = agent.generate("test prompt")
        assert result == "Hello from LLM"

    def test_generate_anthropic(self, config):
        config.open_agent.default_provider = "anthropic"
        agent = ConcreteAgent(config=config, name="Test", role="tester", goal="test")

        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "Hello from Claude"

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        agent._client = mock_client

        result = agent.generate("test prompt")
        assert result == "Hello from Claude"

    def test_generate_unsupported_provider(self, config):
        config.open_agent.default_provider = "unsupported"
        agent = ConcreteAgent(config=config, name="Test", role="tester", goal="test")
        with pytest.raises(ValueError, match="Unsupported provider"):
            agent.generate("test")

    def test_generate_json(self, config):
        agent = ConcreteAgent(config=config, name="Test", role="tester", goal="test")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"key": "value"}'

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        agent._client = mock_client

        result = agent.generate_json("give me json")
        assert result == {"key": "value"}

    def test_generate_json_extracts_from_text(self, config):
        agent = ConcreteAgent(config=config, name="Test", role="tester", goal="test")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = 'Here is the JSON:\n{"key": "value"}\nDone.'

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        agent._client = mock_client

        result = agent.generate_json("give me json")
        assert result == {"key": "value"}

    def test_generate_json_invalid(self, config):
        agent = ConcreteAgent(config=config, name="Test", role="tester", goal="test")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "not json at all"

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        agent._client = mock_client

        with pytest.raises(ValueError, match="No JSON found"):
            agent.generate_json("give me json")


# ── ExtractorAgent Tests ─────────────────────────────────────────────────


class TestExtractorAgent:
    def test_init(self, config):
        agent = ExtractorAgent(config)
        assert agent.name == "Extractor"

    @patch("src.agents.extractor.extract_text_from_pdf")
    @patch("src.agents.extractor.extract_table_captions")
    @patch("src.agents.extractor.extract_figure_captions")
    def test_run(self, mock_fig_caps, mock_tbl_caps, mock_extract, config):
        mock_extract.return_value = "Methods\nWe use OLS regression..."
        mock_tbl_caps.return_value = [
            {"table_number": "Table 1", "caption": "Summary Stats"}
        ]
        mock_fig_caps.return_value = []

        agent = ExtractorAgent(config)

        # Mock the LLM response
        llm_response = {
            "paper_id": "test",
            "title": "Test Paper",
            "research_questions": ["Does X affect Y?"],
            "data_description": "Panel data",
            "data_context": "Natural experiment",
            "data_processing_steps": [],
            "tables": [
                {
                    "table_number": "Table 1",
                    "caption": "Summary Stats",
                    "column_names": ["Mean"],
                    "row_names": ["X"],
                    "regression_specs": [],
                }
            ],
            "figures": [],
        }

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(llm_response)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        agent._client = mock_client

        result = agent.run(paper_path="test.pdf", paper_id="test")
        assert result.paper_id == "test"
        assert len(result.tables) == 1

    def test_validate_no_results_warns_on_pvalues(self, config):
        agent = ExtractorAgent(config)
        from src.models.schemas import PaperSummary

        # This should log a warning but not raise
        summary = PaperSummary(
            paper_id="test",
            data_description="Data with p < 0.05 values",
            data_context="Context",
        )
        agent._validate_no_results(summary)  # Should not raise


# ── VerifierAgent Tests ──────────────────────────────────────────────────


class TestVerifierAgent:
    def test_init(self, config):
        agent = VerifierAgent(config)
        assert agent.name == "Verifier"
        assert agent.tolerance == 0.01

    def test_calculate_overall_grade_all_a(self, config):
        agent = VerifierAgent(config)
        grades = [ReplicationGrade.A, ReplicationGrade.A, ReplicationGrade.A]
        assert agent._calculate_overall_grade(grades) == ReplicationGrade.A

    def test_calculate_overall_grade_mixed(self, config):
        agent = VerifierAgent(config)
        grades = [ReplicationGrade.A, ReplicationGrade.B, ReplicationGrade.C]
        result = agent._calculate_overall_grade(grades)
        assert result == ReplicationGrade.B  # avg = 3.0

    def test_calculate_overall_grade_low(self, config):
        agent = VerifierAgent(config)
        grades = [ReplicationGrade.D, ReplicationGrade.F, ReplicationGrade.F]
        result = agent._calculate_overall_grade(grades)
        assert result in (ReplicationGrade.D, ReplicationGrade.F)

    def test_calculate_overall_grade_empty(self, config):
        agent = VerifierAgent(config)
        assert agent._calculate_overall_grade([]) == ReplicationGrade.F

    def test_generate_summary(self, config):
        agent = VerifierAgent(config)
        verifications = [
            ItemVerification(
                item_id="Table 1", item_type="table",
                grade=ReplicationGrade.A, comparison_notes="Perfect match",
            ),
            ItemVerification(
                item_id="Table 2", item_type="table",
                grade=ReplicationGrade.D, comparison_notes="Major differences found",
            ),
        ]
        summary = agent._generate_summary(verifications, ReplicationGrade.B)
        assert "Overall replication grade: B" in summary
        assert "Total items verified: 2" in summary
        assert "Table 2" in summary

    def test_extract_table_section(self, config):
        agent = VerifierAgent(config)
        text = "Some text\nTable 1 shows the results\nMore detail about Table 1\nOther stuff"
        section = agent._extract_table_section(text, "Table 1")
        assert "Table 1" in section

    def test_extract_table_section_not_found(self, config):
        agent = VerifierAgent(config)
        section = agent._extract_table_section("no tables here", "Table 99")
        assert "not found" in section.lower() or section == ""

    def test_verify_table_failed_execution(self, config, replication_results):
        agent = VerifierAgent(config)
        from src.models.schemas import GeneratedTable

        failed_table = GeneratedTable(
            table_number="Table 1",
            data={},
            code_reference="Table 1",
            execution_success=False,
            error_message="ImportError",
        )
        result = agent._verify_table(failed_table, "paper text", [])
        assert result.grade == ReplicationGrade.F

    def test_verify_figure_failed_execution(self, config):
        agent = VerifierAgent(config)
        from src.models.schemas import GeneratedFigure

        failed_fig = GeneratedFigure(
            figure_number="Figure 1",
            file_path="/tmp/nonexistent.png",
            code_reference="Figure 1",
            execution_success=False,
            error_message="Error",
        )
        result = agent._verify_figure(failed_fig, "paper text", "paper.pdf")
        assert result.grade == ReplicationGrade.F

    def test_verify_figure_file_not_found(self, config):
        agent = VerifierAgent(config)
        from src.models.schemas import GeneratedFigure

        fig = GeneratedFigure(
            figure_number="Figure 1",
            file_path="/tmp/definitely_not_here.png",
            code_reference="Figure 1",
            execution_success=True,
        )
        result = agent._verify_figure(fig, "paper text", "paper.pdf")
        assert result.grade == ReplicationGrade.F
        assert "not found" in result.comparison_notes.lower()


# ── ExplainerAgent Tests ─────────────────────────────────────────────────


class TestExplainerAgent:
    def test_init(self, config):
        agent = ExplainerAgent(config)
        assert agent.name == "Explainer"

    def test_load_replication_package_none(self, config):
        agent = ExplainerAgent(config)
        assert agent._load_replication_package(None) is None

    def test_load_replication_package_missing(self, config):
        agent = ExplainerAgent(config)
        assert agent._load_replication_package("/nonexistent/path") is None

    def test_load_replication_package(self, config, tmp_path):
        agent = ExplainerAgent(config)
        # Create some files
        (tmp_path / "analysis.py").write_text("import pandas as pd\n")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "helper.R").write_text("library(tidyverse)\n")

        package = agent._load_replication_package(str(tmp_path))
        assert package is not None
        assert len(package["files"]) == 2

    def test_get_method_summary_found(self, config, paper_summary):
        agent = ExplainerAgent(config)
        result = agent._get_method_summary(paper_summary, "Table 1")
        assert "Summary Statistics" in result

    def test_get_method_summary_not_found(self, config, paper_summary):
        agent = ExplainerAgent(config)
        result = agent._get_method_summary(paper_summary, "Table 99")
        assert "not found" in result.lower()

    def test_get_replication_code_found(self, config, replication_results):
        agent = ExplainerAgent(config)
        code = agent._get_replication_code(replication_results, "Table 1")
        assert "describe" in code

    def test_get_replication_code_not_found(self, config, replication_results):
        agent = ExplainerAgent(config)
        code = agent._get_replication_code(replication_results, "Table 99")
        assert "not found" in code.lower()

    def test_generate_overall_assessment_no_analyses(self, config):
        agent = ExplainerAgent(config)
        report = VerificationReport(
            paper_id="test",
            overall_grade=ReplicationGrade.A,
            item_verifications=[],
            summary="All good",
        )
        result = agent._generate_overall_assessment([], report)
        assert "fully successful" in result.lower()

    def test_generate_recommendations_software(self, config):
        agent = ExplainerAgent(config)
        from src.models.schemas import DiscrepancyAnalysis

        analyses = [
            DiscrepancyAnalysis(
                item_id="Table 1",
                grade=ReplicationGrade.C,
                description_of_discrepancy="Different results",
                likely_causes=["Different software implementation"],
                is_identifiable=True,
                fault_attribution="replicator",
                confidence="medium",
            )
        ]
        recs = agent._generate_recommendations(analyses)
        assert any("software" in r.lower() for r in recs)

    def test_generate_recommendations_default(self, config):
        agent = ExplainerAgent(config)
        from src.models.schemas import DiscrepancyAnalysis

        analyses = [
            DiscrepancyAnalysis(
                item_id="Table 1",
                grade=ReplicationGrade.D,
                description_of_discrepancy="Unknown",
                likely_causes=["Unknown cause"],
                is_identifiable=False,
                fault_attribution="unclear",
                confidence="low",
            )
        ]
        recs = agent._generate_recommendations(analyses)
        assert len(recs) > 0


# ── ReplicatorAgent Tests ────────────────────────────────────────────────


class TestReplicatorAgent:
    def test_init(self, config):
        agent = ReplicatorAgent(config)
        assert agent.name == "Replicator"

    def test_extract_code_python_block(self, config):
        agent = ReplicatorAgent(config)
        response = "Here is the code:\n```python\nimport pandas as pd\ndf = pd.read_csv('data.csv')\n```\nDone."
        code = agent._extract_code(response)
        assert "import pandas" in code
        assert "```" not in code

    def test_extract_code_generic_block(self, config):
        agent = ReplicatorAgent(config)
        response = "```\nprint('hello')\n```"
        code = agent._extract_code(response)
        assert code == "print('hello')"

    def test_extract_code_no_block(self, config):
        agent = ReplicatorAgent(config)
        response = "x = 1 + 2"
        code = agent._extract_code(response)
        assert code == "x = 1 + 2"

    def test_generate_setup_code(self, config, paper_summary):
        agent = ReplicatorAgent(config)
        code = agent._generate_setup_code("data/test.csv", paper_summary)
        assert code.language == "python"
        assert "data/test.csv" in code.code
        assert "pandas" in code.code
        assert code.execution_order == 0
