"""Tests for the orchestrator pipeline.

The orchestrator now uses LangGraph under the hood but preserves the same
public API (run, run_extraction_only, run_from_summary).
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.models.schemas import (
    ExplanationReport,
    PaperSummary,
    ReplicationGrade,
    ReplicationResults,
    ReplicationState,
    VerificationReport,
)
from src.orchestrator import ReplicationOrchestrator


class TestReplicationOrchestrator:
    def test_init(self, config):
        orch = ReplicationOrchestrator(config=config)
        assert orch.config is config

    def test_init_overrides(self, config):
        orch = ReplicationOrchestrator(
            config=config, model_provider="anthropic", model_name="claude-3"
        )
        assert orch.config.langgraph.default_provider == "anthropic"
        assert orch.config.langgraph.default_model == "claude-3"

    @patch("src.orchestrator.ExtractorAgent")
    def test_run_extraction_only(self, mock_ext, config, paper_summary):
        mock_ext_instance = MagicMock()
        mock_ext_instance.run.return_value = paper_summary
        mock_ext.return_value = mock_ext_instance

        orch = ReplicationOrchestrator(config=config)
        state = orch.run_extraction_only(paper_path="test.pdf", paper_id="test")

        assert state.paper_summary is not None
        assert state.paper_summary.paper_id == "test_paper_2024"
        assert state.current_step == "extraction_complete"
        mock_ext_instance.run.assert_called_once()

    @patch("src.orchestrator.ExtractorAgent")
    def test_run_extraction_failure(self, mock_ext, config):
        mock_ext_instance = MagicMock()
        mock_ext_instance.run.side_effect = Exception("PDF parse error")
        mock_ext.return_value = mock_ext_instance

        orch = ReplicationOrchestrator(config=config)
        state = orch.run_extraction_only(paper_path="bad.pdf")

        assert state.paper_summary is None
        assert len(state.errors) == 1
        assert "PDF parse error" in state.errors[0]

    @patch("src.orchestrator.ExtractorAgent")
    @patch("src.orchestrator.ReplicatorAgent")
    @patch("src.orchestrator.VerifierAgent")
    @patch("src.orchestrator.ExplainerAgent")
    def test_run_full_pipeline(
        self,
        mock_exp,
        mock_ver,
        mock_rep,
        mock_ext,
        config,
        paper_summary,
        replication_results,
        verification_report,
        tmp_dir,
    ):
        # Disable intermediate saves to simplify test
        config.output.save_intermediate_results = False

        mock_ext.return_value.run.return_value = paper_summary
        mock_rep.return_value.run.return_value = replication_results
        mock_ver.return_value.run.return_value = verification_report
        mock_exp.return_value.run.return_value = ExplanationReport(
            paper_id="test_paper_2024",
            analyses=[],
            overall_assessment="Good replication",
            recommendations=[],
        )

        orch = ReplicationOrchestrator(config=config)
        state = orch.run(
            paper_path="test.pdf",
            data_path="test.csv",
            output_dir=str(tmp_dir),
        )

        assert state.current_step == "complete"
        assert state.paper_summary is not None
        assert state.replication_results is not None
        assert state.verification_report is not None
        assert state.explanation_report is not None
        assert len(state.errors) == 0

    @patch("src.orchestrator.ExtractorAgent")
    @patch("src.orchestrator.ReplicatorAgent")
    @patch("src.orchestrator.VerifierAgent")
    @patch("src.orchestrator.ExplainerAgent")
    def test_run_replication_failure_returns_partial(
        self,
        mock_exp,
        mock_ver,
        mock_rep,
        mock_ext,
        config,
        paper_summary,
        tmp_dir,
    ):
        config.output.save_intermediate_results = False

        mock_ext.return_value.run.return_value = paper_summary
        mock_rep.return_value.run.side_effect = Exception("Code execution failed")

        orch = ReplicationOrchestrator(config=config)
        state = orch.run(
            paper_path="test.pdf",
            data_path="test.csv",
            output_dir=str(tmp_dir),
        )

        # Should have extraction but not further
        assert state.paper_summary is not None
        assert state.replication_results is None
        assert state.current_step == "replication"
        assert len(state.errors) == 1

    @patch("src.orchestrator.ExtractorAgent")
    @patch("src.orchestrator.ReplicatorAgent")
    @patch("src.orchestrator.VerifierAgent")
    @patch("src.orchestrator.ExplainerAgent")
    def test_save_intermediate(
        self, mock_exp, mock_ver, mock_rep, mock_ext, config, paper_summary, tmp_dir
    ):
        config.output.save_intermediate_results = True

        mock_ext.return_value.run.return_value = paper_summary
        mock_rep.return_value.run.side_effect = Exception("fail")

        orch = ReplicationOrchestrator(config=config)
        orch.run(
            paper_path="test.pdf",
            data_path="test.csv",
            output_dir=str(tmp_dir),
        )

        # Check that intermediate result was saved
        summary_path = tmp_dir / "paper_summary.json"
        assert summary_path.exists()
        with open(summary_path) as f:
            data = json.load(f)
        assert data["paper_id"] == "test_paper_2024"

    @patch("src.orchestrator.ExtractorAgent")
    @patch("src.orchestrator.ReplicatorAgent")
    @patch("src.orchestrator.VerifierAgent")
    @patch("src.orchestrator.ExplainerAgent")
    def test_run_from_summary(
        self,
        mock_exp,
        mock_ver,
        mock_rep,
        mock_ext,
        config,
        paper_summary,
        replication_results,
        verification_report,
        tmp_dir,
    ):
        config.output.save_intermediate_results = False

        mock_rep.return_value.run.return_value = replication_results
        mock_ver.return_value.run.return_value = verification_report
        mock_exp.return_value.run.return_value = ExplanationReport(
            paper_id="test_paper_2024",
            analyses=[],
            overall_assessment="Good",
            recommendations=[],
        )

        orch = ReplicationOrchestrator(config=config)
        state = orch.run_from_summary(
            paper_summary=paper_summary,
            data_path="test.csv",
            paper_path="test.pdf",
            output_dir=str(tmp_dir),
        )

        assert state.current_step == "complete"
        # Extractor should NOT have been called
        mock_ext.return_value.run.assert_not_called()
