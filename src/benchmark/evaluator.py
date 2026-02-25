"""Shared judge evaluator for benchmark runs."""

import json
import os
from pathlib import Path

from ..models.schemas import PaperSummary
from ..utils.logging_utils import get_logger
from .artifact_parser import ArtifactParser
from .config import JudgeConfig, PaperSpec
from .judge import Judge
from .results import EvaluationResult, RunArtifacts

logger = get_logger(__name__)


class SharedEvaluator:
    """Evaluates run artifacts using a fixed judge model.

    Uses the Judge class (plain SDK) with a consistent model so that
    grading is comparable across different benchmark runs.
    """

    def __init__(self, judge_config: JudgeConfig):
        self.judge_config = judge_config
        self._judge = self._build_judge()

    def _build_judge(self) -> Judge:
        """Build a Judge from the config."""
        provider = self.judge_config.provider.lower()
        if provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY", "")
        elif provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        else:
            api_key = ""
        return Judge(
            provider=provider,
            model=self.judge_config.model_name,
            api_key=api_key,
        )

    def evaluate(
        self,
        paper: PaperSpec,
        artifacts: RunArtifacts,
        paper_summary: PaperSummary | None = None,
    ) -> EvaluationResult:
        """Evaluate artifacts from a benchmark run.

        Args:
            paper: Paper specification with paths.
            artifacts: Run artifacts (workspace_dir must exist).
            paper_summary: Methodology summary. If None, loaded from workspace.

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

        # Single judge call produces both reports
        logger.info(f"Running judge for {paper.paper_id}")
        verification_report, explanation_report = self._judge.run(
            paper_path=paper.pdf_path,
            paper_summary=paper_summary,
            replication_results=replication_results,
            replication_package_path=paper.replication_package_path,
        )

        # Save reports
        self._save_reports(
            artifacts.workspace_dir, verification_report, explanation_report,
        )

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
