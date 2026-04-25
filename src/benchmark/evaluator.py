"""Shared judge evaluator for benchmark runs."""

import json
import os
from pathlib import Path

from ..models.schemas import PaperResults, PaperSummary
from ..utils.logging_utils import get_logger
from .artifact_parser import ArtifactParser
from .comparator import ComparatorAgent
from .config import JudgeConfig, PaperSpec
from .judge import Judge
from .results import EvaluationResult, RunArtifacts

logger = get_logger(__name__)


class SharedEvaluator:
    """Evaluates run artifacts using a fixed judge model.

    Uses the Judge class (plain SDK) with a consistent model so that
    grading is comparable across different benchmark runs.
    """

    def __init__(
        self,
        judge_config: JudgeConfig,
        api_keys: dict[str, str] | None = None,
        item_types: list[str] | None = None,
        deterministic_only: bool = False,
    ):
        self.judge_config = judge_config
        self._api_keys = api_keys or {}
        self._item_types = item_types or ["table", "figure"]
        self._deterministic_only = deterministic_only
        self._comparator = self._build_comparator()
        self._judge = self._build_judge()

    def _build_comparator(self) -> ComparatorAgent:
        """Build a ComparatorAgent using CLI tool config."""
        return ComparatorAgent(
            cli_tool=getattr(self.judge_config, "comparator_cli_tool", "claude-code"),
            model=getattr(self.judge_config, "comparator_model", "claude-sonnet-4-6"),
            timeout=getattr(self.judge_config, "comparator_timeout", 300),
        )

    def _build_judge(self) -> Judge:
        """Build a Judge from the config."""
        provider = self.judge_config.provider.lower()
        if provider == "openai":
            env_var = "OPENAI_API_KEY"
        elif provider == "anthropic":
            env_var = "ANTHROPIC_API_KEY"
        else:
            env_var = ""
        api_key = self._api_keys.get(env_var) or os.environ.get(env_var, "")
        return Judge(
            provider=provider,
            model=self.judge_config.model_name,
            api_key=api_key,
            use_vision=self.judge_config.use_vision,
            comparator=self._comparator,
            item_types=self._item_types,
            deterministic_only=self._deterministic_only,
        )

    def evaluate(
        self,
        paper: PaperSpec,
        artifacts: RunArtifacts,
        paper_summary: PaperSummary | None = None,
        paper_results: PaperResults | None = None,
    ) -> EvaluationResult:
        """Evaluate artifacts from a benchmark run.

        Args:
            paper: Paper specification with paths.
            artifacts: Run artifacts (workspace_dir must exist).
            paper_summary: Methodology summary. If None, loaded from workspace.
            paper_results: Extracted original table values. If None, loaded from
                summaries dir or tables fall back to LLM-only judging.

        Returns:
            EvaluationResult with grades and analyses.
        """
        # Parse artifacts into ReplicationResults
        replication_results = artifacts.replication_results
        if replication_results is None:
            logger.info("Parsing workspace artifacts into ReplicationResults")
            replication_results = ArtifactParser.parse(
                Path(artifacts.workspace_dir), paper.paper_id
            )

        # Load paper_summary from workspace if not provided
        if paper_summary is None:
            paper_summary = self._load_summary_from_workspace(
                artifacts.workspace_dir, paper.paper_id
            )

        # Try to load paper_results from summaries dir if not provided
        if paper_results is None:
            paper_results = self._load_results_from_summaries(
                artifacts.workspace_dir, paper.paper_id
            )

        # Single judge call produces both reports
        logger.info(f"Running judge for {paper.paper_id}")
        verification_report, explanation_report = self._judge.run(
            paper_path=paper.pdf_path,
            paper_summary=paper_summary,
            replication_results=replication_results,
            paper_results=paper_results,
            workspace_dir=artifacts.workspace_dir,
        )

        # Save reports and usage
        self._save_reports(
            artifacts.workspace_dir, verification_report, explanation_report,
        )
        self._save_judge_usage(artifacts.workspace_dir, self._judge.usage_summary)

        item_grades = {
            v.item_id: v.grade.value
            for v in verification_report.item_verifications
        }

        return EvaluationResult(
            verification_report=verification_report,
            explanation_report=explanation_report,
            overall_grade=verification_report.overall_grade.value,
            item_grades=item_grades,
        )

    @staticmethod
    def _load_summary_from_workspace(
        workspace_dir: str, paper_id: str,
    ) -> PaperSummary:
        """Load PaperSummary from the workspace's methodology_summary.json."""
        summary_path = Path(workspace_dir) / "methodology_summary.json"
        if summary_path.exists():
            data = json.loads(summary_path.read_text())
            return PaperSummary(**data)
        logger.warning(f"No methodology_summary.json in {workspace_dir}, using minimal summary")
        return PaperSummary(
            paper_id=paper_id, title="Unknown",
            data_description="Unknown", data_context="Unknown",
        )

    @staticmethod
    def _load_results_from_summaries(
        workspace_dir: str, paper_id: str,
    ) -> PaperResults | None:
        """Try to load PaperResults from the summaries directory."""
        # Walk up from workspace to find summaries dir
        run_dir = Path(workspace_dir).parent  # model_paper_approach dir
        output_dir = run_dir.parent  # benchmark output dir
        results_path = output_dir / "summaries" / f"{paper_id}_results.json"
        if results_path.exists():
            try:
                data = json.loads(results_path.read_text())
                return PaperResults(**data)
            except Exception as e:
                logger.warning(f"Could not load paper results from {results_path}: {e}")
        return None

    @staticmethod
    def _save_judge_usage(workspace_dir: str, usage: dict) -> None:
        """Save judge token usage to the run directory."""
        run_dir = Path(workspace_dir).parent
        try:
            usage_path = run_dir / "judge_usage.json"
            usage_path.write_text(json.dumps(usage, indent=2))
            logger.info(f"Saved judge usage: {usage_path}")
        except Exception as e:
            logger.warning(f"Could not save judge usage: {e}")

    @staticmethod
    def _save_reports(workspace_dir: str, verification_report, explanation_report) -> None:
        """Save reports to the run directory (parent of workspace)."""
        run_dir = Path(workspace_dir).parent
        try:
            report_path = run_dir / "verification_report.json"
            report_path.write_text(
                json.dumps(verification_report.model_dump(), indent=2, default=str)
            )
            logger.info(f"Saved verification report: {report_path}")

            if explanation_report is not None:
                expl_path = run_dir / "explanation_report.json"
                expl_path.write_text(
                    json.dumps(explanation_report.model_dump(), indent=2, default=str)
                )
                logger.info(f"Saved explanation report: {expl_path}")
        except Exception as e:
            logger.warning(f"Could not save reports to disk: {e}")
