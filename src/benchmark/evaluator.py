"""Shared judge evaluator using Verifier + Explainer agents."""

import json
import os
from pathlib import Path

from ..agents.verifier import VerifierAgent
from ..agents.explainer import ExplainerAgent
from ..models.config import Config, LangGraphConfig
from ..models.schemas import PaperSummary, ReplicationResults
from ..utils.logging_utils import get_logger
from .artifact_parser import ArtifactParser
from .config import JudgeConfig, PaperSpec
from .results import EvaluationResult, RunArtifacts

logger = get_logger(__name__)


class SharedEvaluator:
    """Evaluates run artifacts using a fixed judge model.

    Uses the existing VerifierAgent and ExplainerAgent with a consistent
    judge model so that grading is comparable across different benchmark runs.
    """

    def __init__(self, judge_config: JudgeConfig):
        """Initialize the evaluator with a judge model config.

        Args:
            judge_config: Specifies the provider and model to use for grading.
        """
        self.judge_config = judge_config
        self._config = self._build_judge_config()

    def _build_judge_config(self) -> Config:
        """Build a Config locked to the judge model."""
        # Determine which API key env var to use
        provider = self.judge_config.provider.lower()
        config = Config(
            langgraph=LangGraphConfig(
                default_provider=provider,
                default_model=self.judge_config.model_name,
            ),
        )
        if provider == "openai":
            config.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        elif provider == "anthropic":
            config.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        return config

    def evaluate(
        self,
        paper: PaperSpec,
        artifacts: RunArtifacts,
        paper_summary: PaperSummary | None = None,
    ) -> EvaluationResult:
        """Evaluate artifacts from a benchmark run.

        Args:
            paper: Paper specification with paths.
            artifacts: Run artifacts (may or may not contain parsed ReplicationResults).
            paper_summary: Optional PaperSummary for the Explainer agent.

        Returns:
            EvaluationResult with grades and analyses.
        """
        # Get or parse ReplicationResults
        replication_results = artifacts.replication_results
        if replication_results is None:
            logger.info("Parsing freestyle artifacts into ReplicationResults")
            replication_results = ArtifactParser.parse(
                Path(artifacts.workspace_dir), paper.paper_id
            )

        # Run Verifier with judge model
        logger.info(f"Running judge verification for {paper.paper_id}")
        verifier = VerifierAgent(self._config)
        verification_report = verifier.run(
            paper_path=paper.pdf_path,
            replication_results=replication_results,
        )

        # Run Explainer if there are non-A grades and we have a paper summary
        explanation_report = None
        non_a_items = [
            v for v in verification_report.item_verifications
            if v.grade.value != "A"
        ]
        if non_a_items and paper_summary is not None:
            logger.info(f"Running judge explanation for {len(non_a_items)} discrepancies")
            explainer = ExplainerAgent(self._config)
            explanation_report = explainer.run(
                paper_path=paper.pdf_path,
                paper_summary=paper_summary,
                replication_results=replication_results,
                verification_report=verification_report,
                replication_package_path=paper.replication_package_path,
            )

        # Save reports to disk alongside workspace
        self._save_reports(
            artifacts.workspace_dir, verification_report, explanation_report
        )

        # Build item grades map
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
    def _save_reports(
        workspace_dir: str,
        verification_report,
        explanation_report,
    ) -> None:
        """Save verification and explanation reports to the run directory."""
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
