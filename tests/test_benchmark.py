"""Tests for the benchmark framework."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from src.benchmark.config import BenchmarkConfig, JudgeConfig, ModelSpec, PaperSpec
from src.benchmark.results import (
    BenchmarkResults,
    EvaluationResult,
    ResultsAggregator,
    RunArtifacts,
    SingleRunResult,
)
from src.benchmark.artifact_parser import ArtifactParser, _infer_item_number
from src.benchmark.opencode_runner import OpencodeRunner
from src.benchmark.structured_runner import StructuredRunner
from src.benchmark.evaluator import SharedEvaluator
from src.benchmark.runner import BenchmarkRunner, run_benchmark
from src.models.schemas import (
    DataProcessingStep,
    GeneratedCode,
    GeneratedTable,
    GeneratedFigure,
    ItemVerification,
    PaperSummary,
    PlotSpec,
    RegressionSpec,
    ReplicationGrade,
    ReplicationResults,
    TableSpec,
    VerificationReport,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def model_spec():
    return ModelSpec(provider="openai", model_name="gpt-4o", api_key_env="OPENAI_API_KEY")


@pytest.fixture
def paper_spec(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_text("fake pdf")
    data = tmp_path / "data.csv"
    data.write_text("a,b\n1,2\n3,4")
    return PaperSpec(
        paper_id="test_paper",
        pdf_path=str(pdf),
        data_path=str(data),
    )


@pytest.fixture
def paper_summary():
    """Methodology summary without results — what the replicator sees."""
    return PaperSummary(
        paper_id="test_paper",
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
                description="Drop missing values in outcome variable",
                variables_involved=["outcome_var"],
            ),
        ],
        tables=[
            TableSpec(
                table_number="Table 1",
                caption="Summary Statistics",
                column_names=["Mean", "Std. Dev.", "N"],
                row_names=["outcome_var", "treatment"],
                regression_specs=[],
            ),
            TableSpec(
                table_number="Table 2",
                caption="Main Regression Results",
                column_names=["(1)", "(2)"],
                row_names=["treatment", "Observations", "R-squared"],
                regression_specs=[
                    RegressionSpec(
                        model_type="OLS",
                        dependent_var="outcome_var",
                        independent_vars=["treatment"],
                        controls=["income"],
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
def benchmark_config(paper_spec, model_spec):
    return BenchmarkConfig(
        models=[model_spec],
        papers=[paper_spec],
        approaches=["freestyle", "structured"],
        judge=JudgeConfig(provider="openai", model_name="gpt-4o"),
        output_dir="/tmp/benchmark_test_output",
        timeout_seconds=60,
    )


@pytest.fixture
def sample_replication_results():
    return ReplicationResults(
        paper_id="test_paper",
        code_files=[
            GeneratedCode(
                language="python", code="print('hello')",
                dependencies=[], execution_order=0,
            )
        ],
        tables=[
            GeneratedTable(
                table_number="Table 1",
                data={"col": [1, 2]},
                code_reference="test.py",
                execution_success=True,
            )
        ],
        figures=[],
    )


@pytest.fixture
def sample_verification_report():
    return VerificationReport(
        paper_id="test_paper",
        overall_grade=ReplicationGrade.B,
        item_verifications=[
            ItemVerification(
                item_id="Table 1",
                item_type="table",
                grade=ReplicationGrade.B,
                comparison_notes="Close match.",
                key_findings_match=True,
            )
        ],
        summary="Overall grade: B.",
    )


@pytest.fixture
def sample_evaluation_result(sample_verification_report):
    return EvaluationResult(
        verification_report=sample_verification_report,
        explanation_report=None,
        overall_grade="B",
        item_grades={"Table 1": "B"},
    )


@pytest.fixture
def sample_run_artifacts(tmp_path, sample_replication_results):
    return RunArtifacts(
        workspace_dir=str(tmp_path),
        stdout="ok",
        stderr="",
        exit_code=0,
        duration_seconds=10.5,
        replication_results=sample_replication_results,
    )


# =============================================================================
# TestBenchmarkConfig
# =============================================================================


class TestBenchmarkConfig:
    def test_load_basic(self, model_spec, paper_spec):
        config = BenchmarkConfig(
            models=[model_spec],
            papers=[paper_spec],
        )
        assert config.approaches == ["freestyle", "structured"]
        assert config.timeout_seconds == 600
        assert config.judge.provider == "openai"

    def test_requires_models_field(self):
        """models is a required field."""
        with pytest.raises(Exception):
            BenchmarkConfig(papers=[])  # models not provided at all

    def test_model_spec_fields(self):
        m = ModelSpec(provider="anthropic", model_name="claude-3-opus", api_key_env="ANTHROPIC_API_KEY")
        assert m.provider == "anthropic"
        assert m.model_name == "claude-3-opus"

    def test_paper_spec_optional_package(self):
        p = PaperSpec(paper_id="x", pdf_path="/a.pdf", data_path="/b.csv")
        assert p.replication_package_path is None

    def test_judge_defaults(self):
        j = JudgeConfig()
        assert j.provider == "openai"
        assert j.model_name == "gpt-4o"

    def test_from_dict(self, paper_spec):
        data = {
            "models": [{"provider": "openai", "model_name": "gpt-4o", "api_key_env": "OPENAI_API_KEY"}],
            "papers": [paper_spec.model_dump()],
            "approaches": ["structured"],
            "timeout_seconds": 120,
        }
        config = BenchmarkConfig(**data)
        assert len(config.models) == 1
        assert config.approaches == ["structured"]
        assert config.timeout_seconds == 120


# =============================================================================
# TestArtifactParser
# =============================================================================


class TestArtifactParser:
    def test_parse_empty_workspace(self, tmp_path):
        results = ArtifactParser.parse(tmp_path, "paper1")
        assert results.paper_id == "paper1"
        assert results.code_files == []
        assert results.tables == []
        assert results.figures == []

    def test_parse_nonexistent_workspace(self, tmp_path):
        results = ArtifactParser.parse(tmp_path / "nonexistent", "paper1")
        assert results.paper_id == "paper1"
        assert results.code_files == []

    def test_parse_code_files(self, tmp_path):
        (tmp_path / "analysis.py").write_text("import pandas as pd\ndf = pd.read_csv('data.csv')")
        (tmp_path / "helpers.r").write_text("library(ggplot2)")

        results = ArtifactParser.parse(tmp_path, "paper1")
        assert len(results.code_files) == 2
        langs = {c.language for c in results.code_files}
        assert langs == {"python", "r"}

    def test_parse_csv_table(self, tmp_path):
        (tmp_path / "table_1.csv").write_text("x,y\n1,2\n3,4")

        results = ArtifactParser.parse(tmp_path, "paper1")
        assert len(results.tables) == 1
        assert results.tables[0].table_number == "Table 1"

    def test_parse_json_table(self, tmp_path):
        (tmp_path / "table_2.json").write_text('{"col": [1, 2]}')

        results = ArtifactParser.parse(tmp_path, "paper1")
        assert len(results.tables) == 1
        assert results.tables[0].table_number == "Table 2"

    def test_parse_figure(self, tmp_path):
        (tmp_path / "figure_1.png").write_bytes(b"\x89PNG")

        results = ArtifactParser.parse(tmp_path, "paper1")
        assert len(results.figures) == 1
        assert results.figures[0].figure_number == "Figure 1"
        assert results.figures[0].format == "png"

    def test_parse_mixed_workspace(self, tmp_path):
        (tmp_path / "code.py").write_text("x = 1")
        (tmp_path / "table_1.csv").write_text("a,b\n1,2")
        (tmp_path / "fig_2.png").write_bytes(b"\x89PNG")

        results = ArtifactParser.parse(tmp_path, "paper1")
        assert len(results.code_files) == 1
        assert len(results.tables) == 1
        assert len(results.figures) == 1

    def test_infer_item_number_table(self):
        assert _infer_item_number("table_1", "Table") == "Table 1"
        assert _infer_item_number("table1", "Table") == "Table 1"
        assert _infer_item_number("table-3", "Table") == "Table 3"

    def test_infer_item_number_figure(self):
        assert _infer_item_number("figure_1", "Figure") == "Figure 1"
        assert _infer_item_number("fig2", "Figure") == "Figure 2"

    def test_infer_item_number_fallback(self):
        result = _infer_item_number("summary_stats", "Table")
        assert result == "Table (summary_stats)"


# =============================================================================
# TestOpencodeRunner
# =============================================================================


class TestOpencodeRunner:
    def test_run_success(self, tmp_path, model_spec, paper_spec, paper_summary):
        runner = OpencodeRunner(opencode_binary="opencode", timeout=30)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="Replication complete",
                stderr="",
                returncode=0,
            )
            workspace = tmp_path / "workspace"
            artifacts = runner.run(model_spec, paper_spec, paper_summary, workspace)

        assert artifacts.exit_code == 0
        assert "Replication complete" in artifacts.stdout
        assert artifacts.duration_seconds > 0
        mock_run.assert_called_once()

    def test_run_timeout(self, tmp_path, model_spec, paper_spec, paper_summary):
        import subprocess

        runner = OpencodeRunner(opencode_binary="opencode", timeout=1)

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("opencode", 1)):
            workspace = tmp_path / "workspace"
            artifacts = runner.run(model_spec, paper_spec, paper_summary, workspace)

        assert artifacts.exit_code == -1
        assert "Timed out" in artifacts.stderr

    def test_run_binary_not_found(self, tmp_path, model_spec, paper_spec, paper_summary):
        runner = OpencodeRunner(opencode_binary="/nonexistent/binary", timeout=30)

        with patch("subprocess.run", side_effect=FileNotFoundError):
            workspace = tmp_path / "workspace"
            artifacts = runner.run(model_spec, paper_spec, paper_summary, workspace)

        assert artifacts.exit_code == -2
        assert "not found" in artifacts.stderr

    def test_task_prompt_contains_methodology(self, tmp_path, model_spec, paper_spec, paper_summary):
        runner = OpencodeRunner(opencode_binary="opencode", timeout=30)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="", stderr="", returncode=0)
            workspace = tmp_path / "workspace"
            runner.run(model_spec, paper_spec, paper_summary, workspace)

        assert (workspace / "TASK.md").exists()
        content = (workspace / "TASK.md").read_text()
        assert "Replication Task" in content
        # Should contain methodology details, not paper PDF reference
        assert "Methodological Summary" in content
        assert "The Effect of X on Y" in content
        assert "OLS" in content
        assert "outcome_var" in content
        # Should NOT reference the paper PDF
        assert "paper.pdf" not in content

    def test_does_not_copy_paper_pdf(self, tmp_path, model_spec, paper_spec, paper_summary):
        """Paper PDF must NOT be in the workspace — replicator is blind to results."""
        runner = OpencodeRunner(opencode_binary="opencode", timeout=30)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="", stderr="", returncode=0)
            workspace = tmp_path / "workspace"
            runner.run(model_spec, paper_spec, paper_summary, workspace)

        # Data should be there
        assert (workspace / "data.csv").exists()
        # Paper PDF should NOT be there
        assert not (workspace / "paper.pdf").exists()

    def test_saves_methodology_json(self, tmp_path, model_spec, paper_spec, paper_summary):
        runner = OpencodeRunner(opencode_binary="opencode", timeout=30)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="", stderr="", returncode=0)
            workspace = tmp_path / "workspace"
            runner.run(model_spec, paper_spec, paper_summary, workspace)

        assert (workspace / "methodology_summary.json").exists()
        data = json.loads((workspace / "methodology_summary.json").read_text())
        assert data["paper_id"] == "test_paper"


# =============================================================================
# TestStructuredRunner
# =============================================================================


class TestStructuredRunner:
    def test_run_success(self, tmp_path, model_spec, paper_spec, paper_summary,
                         sample_replication_results):
        runner = StructuredRunner(timeout=60)

        mock_state = MagicMock()
        mock_state.replication_results = sample_replication_results
        mock_state.errors = []
        mock_state.current_step = "complete"

        with patch("src.benchmark.structured_runner.ReplicationOrchestrator") as MockOrch:
            MockOrch.return_value.run_from_summary.return_value = mock_state
            workspace = tmp_path / "workspace"
            artifacts = runner.run(model_spec, paper_spec, paper_summary, workspace)

        assert artifacts.exit_code == 0
        assert artifacts.replication_results is not None
        assert artifacts.replication_results.paper_id == "test_paper"
        # Verify run_from_summary was called (not run)
        MockOrch.return_value.run_from_summary.assert_called_once()
        MockOrch.return_value.run.assert_not_called()

    def test_run_uses_paper_summary(self, tmp_path, model_spec, paper_spec, paper_summary,
                                     sample_replication_results):
        """Verify the pre-extracted summary is passed to the orchestrator."""
        runner = StructuredRunner(timeout=60)

        mock_state = MagicMock()
        mock_state.replication_results = sample_replication_results
        mock_state.errors = []
        mock_state.current_step = "complete"

        with patch("src.benchmark.structured_runner.ReplicationOrchestrator") as MockOrch:
            MockOrch.return_value.run_from_summary.return_value = mock_state
            workspace = tmp_path / "workspace"
            runner.run(model_spec, paper_spec, paper_summary, workspace)

        call_kwargs = MockOrch.return_value.run_from_summary.call_args
        assert call_kwargs.kwargs["paper_summary"].paper_id == "test_paper"

    def test_run_failure(self, tmp_path, model_spec, paper_spec, paper_summary):
        runner = StructuredRunner(timeout=60)

        with patch("src.benchmark.structured_runner.ReplicationOrchestrator") as MockOrch:
            MockOrch.return_value.run_from_summary.side_effect = RuntimeError("Pipeline crashed")
            workspace = tmp_path / "workspace"
            artifacts = runner.run(model_spec, paper_spec, paper_summary, workspace)

        assert artifacts.exit_code == 1
        assert "Pipeline crashed" in artifacts.stderr
        assert artifacts.replication_results is None


# =============================================================================
# TestSharedEvaluator
# =============================================================================


class TestSharedEvaluator:
    def test_evaluate_with_existing_results(
        self, paper_spec, sample_run_artifacts, sample_verification_report
    ):
        evaluator = SharedEvaluator(JudgeConfig())

        with patch("src.benchmark.evaluator.VerifierAgent") as MockVerifier:
            MockVerifier.return_value.run.return_value = sample_verification_report

            result = evaluator.evaluate(paper_spec, sample_run_artifacts)

        assert result.overall_grade == "B"
        assert "Table 1" in result.item_grades
        assert result.item_grades["Table 1"] == "B"

    def test_evaluate_parses_freestyle_artifacts(
        self, tmp_path, paper_spec, sample_verification_report
    ):
        # Create a freestyle workspace with files
        (tmp_path / "code.py").write_text("x = 1")
        (tmp_path / "table_1.csv").write_text("a,b\n1,2")

        artifacts = RunArtifacts(
            workspace_dir=str(tmp_path),
            stdout="", stderr="", exit_code=0, duration_seconds=5.0,
            replication_results=None,  # freestyle: no pre-parsed results
        )

        evaluator = SharedEvaluator(JudgeConfig())

        with patch("src.benchmark.evaluator.VerifierAgent") as MockVerifier:
            MockVerifier.return_value.run.return_value = sample_verification_report
            result = evaluator.evaluate(paper_spec, artifacts)

        assert result.overall_grade == "B"
        # Verify ArtifactParser was used (VerifierAgent was called with parsed results)
        call_args = MockVerifier.return_value.run.call_args
        repl_results = call_args.kwargs.get("replication_results") or call_args[1].get("replication_results")
        if repl_results is None:
            repl_results = call_args[0][1]  # positional arg
        assert repl_results.paper_id == "test_paper"


# =============================================================================
# TestResultsAggregator
# =============================================================================


class TestResultsAggregator:
    def test_save_run(
        self, tmp_path, model_spec, paper_spec,
        sample_run_artifacts, sample_evaluation_result,
    ):
        run = SingleRunResult(
            model=model_spec,
            paper=paper_spec,
            approach="freestyle",
            artifacts=sample_run_artifacts,
            evaluation=sample_evaluation_result,
            duration_seconds=15.0,
        )
        ResultsAggregator.save_run(run, tmp_path)

        expected_dir = tmp_path / "gpt-4o_test_paper_freestyle"
        assert expected_dir.exists()
        result_json = json.loads((expected_dir / "result.json").read_text())
        assert result_json["approach"] == "freestyle"

    def test_save_summary(
        self, tmp_path, model_spec, paper_spec,
        sample_run_artifacts, sample_evaluation_result,
    ):
        run = SingleRunResult(
            model=model_spec,
            paper=paper_spec,
            approach="structured",
            artifacts=sample_run_artifacts,
            evaluation=sample_evaluation_result,
            duration_seconds=20.0,
        )
        results = BenchmarkResults(runs=[run])
        ResultsAggregator.save_summary(results, tmp_path)

        assert (tmp_path / "summary.json").exists()
        assert (tmp_path / "summary.csv").exists()

        csv_content = (tmp_path / "summary.csv").read_text()
        assert "gpt-4o" in csv_content
        assert "structured" in csv_content


# =============================================================================
# TestBenchmarkRunner
# =============================================================================


class TestBenchmarkRunner:
    def test_run_iterates_all_combinations(
        self, tmp_path, benchmark_config, paper_summary,
        sample_verification_report, sample_replication_results,
    ):
        benchmark_config.output_dir = str(tmp_path)
        runner = BenchmarkRunner(benchmark_config)

        # Pre-populate the summary cache to skip extraction
        paper_id = benchmark_config.papers[0].paper_id
        runner._summary_cache[paper_id] = paper_summary

        mock_freestyle_artifacts = RunArtifacts(
            workspace_dir=str(tmp_path / "f"), stdout="ok", stderr="",
            exit_code=0, duration_seconds=5.0,
        )
        mock_structured_artifacts = RunArtifacts(
            workspace_dir=str(tmp_path / "s"), stdout="ok", stderr="",
            exit_code=0, duration_seconds=5.0,
            replication_results=sample_replication_results,
        )

        with patch.object(runner.opencode_runner, "run", return_value=mock_freestyle_artifacts), \
             patch.object(runner.structured_runner, "run", return_value=mock_structured_artifacts), \
             patch.object(runner.evaluator, "evaluate") as mock_eval:

            mock_eval.return_value = EvaluationResult(
                verification_report=sample_verification_report,
                explanation_report=None,
                overall_grade="B",
                item_grades={"Table 1": "B"},
            )

            results = runner.run()

        # 1 model × 1 paper × 2 approaches = 2 runs
        assert len(results.runs) == 2
        approaches = {r.approach for r in results.runs}
        assert approaches == {"freestyle", "structured"}

    def test_run_single_freestyle(
        self, tmp_path, benchmark_config, model_spec, paper_spec,
        paper_summary, sample_verification_report,
    ):
        benchmark_config.output_dir = str(tmp_path)
        runner = BenchmarkRunner(benchmark_config)
        runner._summary_cache[paper_spec.paper_id] = paper_summary

        mock_artifacts = RunArtifacts(
            workspace_dir=str(tmp_path), stdout="ok", stderr="",
            exit_code=0, duration_seconds=5.0,
        )

        with patch.object(runner.opencode_runner, "run", return_value=mock_artifacts), \
             patch.object(runner.evaluator, "evaluate") as mock_eval:

            mock_eval.return_value = EvaluationResult(
                verification_report=sample_verification_report,
                explanation_report=None,
                overall_grade="B",
                item_grades={"Table 1": "B"},
            )

            result = runner.run_single(model_spec, paper_spec, "freestyle")

        assert result.approach == "freestyle"
        assert result.evaluation.overall_grade == "B"

    def test_run_single_passes_summary_to_runners(
        self, tmp_path, benchmark_config, model_spec, paper_spec,
        paper_summary, sample_verification_report,
    ):
        """Both runners must receive the pre-extracted summary, not the paper."""
        benchmark_config.output_dir = str(tmp_path)
        runner = BenchmarkRunner(benchmark_config)
        runner._summary_cache[paper_spec.paper_id] = paper_summary

        mock_artifacts = RunArtifacts(
            workspace_dir=str(tmp_path), stdout="ok", stderr="",
            exit_code=0, duration_seconds=5.0,
        )

        with patch.object(runner.opencode_runner, "run", return_value=mock_artifacts) as mock_oc, \
             patch.object(runner.evaluator, "evaluate") as mock_eval:

            mock_eval.return_value = EvaluationResult(
                verification_report=sample_verification_report,
                explanation_report=None,
                overall_grade="B",
                item_grades={"Table 1": "B"},
            )

            runner.run_single(model_spec, paper_spec, "freestyle")

        # Verify paper_summary was passed as third positional arg
        call_args = mock_oc.call_args
        assert call_args[0][2] == paper_summary  # (model, paper, paper_summary, workspace)

    def test_run_single_unknown_approach(
        self, tmp_path, benchmark_config, model_spec, paper_spec, paper_summary,
    ):
        benchmark_config.output_dir = str(tmp_path)
        runner = BenchmarkRunner(benchmark_config)
        runner._summary_cache[paper_spec.paper_id] = paper_summary

        with pytest.raises(ValueError, match="Unknown approach"):
            runner.run_single(model_spec, paper_spec, "unknown")

    def test_extract_summary_caches(self, tmp_path, benchmark_config, paper_spec, paper_summary):
        """Extraction runs once per paper then returns cached result."""
        benchmark_config.output_dir = str(tmp_path)
        runner = BenchmarkRunner(benchmark_config)

        with patch("src.benchmark.runner.ExtractorAgent") as MockExtractor:
            MockExtractor.return_value.run.return_value = paper_summary

            s1 = runner._extract_summary(paper_spec)
            s2 = runner._extract_summary(paper_spec)

        assert s1 is s2
        # ExtractorAgent.run should only be called once
        MockExtractor.return_value.run.assert_called_once()

    def test_run_benchmark_convenience(self, tmp_path):
        config = BenchmarkConfig(
            models=[ModelSpec(provider="openai", model_name="gpt-4o", api_key_env="OPENAI_API_KEY")],
            papers=[PaperSpec(paper_id="p1", pdf_path="/fake.pdf", data_path="/fake.csv")],
            approaches=["freestyle"],
            output_dir=str(tmp_path),
        )

        with patch("src.benchmark.runner.BenchmarkRunner") as MockRunner:
            MockRunner.return_value.run.return_value = BenchmarkResults()
            result = run_benchmark(config=config)

        assert isinstance(result, BenchmarkResults)
        MockRunner.assert_called_once_with(config)

    def test_run_benchmark_from_yaml(self, tmp_path):
        config_data = {
            "models": [{"provider": "openai", "model_name": "gpt-4o", "api_key_env": "OPENAI_API_KEY"}],
            "papers": [{"paper_id": "p1", "pdf_path": "/fake.pdf", "data_path": "/fake.csv"}],
        }
        config_file = tmp_path / "config.yaml"
        import yaml
        config_file.write_text(yaml.dump(config_data))

        with patch("src.benchmark.runner.BenchmarkRunner") as MockRunner:
            MockRunner.return_value.run.return_value = BenchmarkResults()
            result = run_benchmark(config_path=str(config_file))

        assert isinstance(result, BenchmarkResults)

    def test_run_benchmark_requires_config(self):
        with pytest.raises(ValueError, match="Either config_path or config"):
            run_benchmark()
