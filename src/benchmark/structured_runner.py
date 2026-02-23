"""Structured approach runner using the LangGraph pipeline."""

import os
import time
from pathlib import Path

from ..models.config import Config, LangGraphConfig
from ..models.schemas import PaperSummary
from ..orchestrator import ReplicationOrchestrator
from ..utils.logging_utils import get_logger
from .config import ModelSpec, PaperSpec
from .results import RunArtifacts

logger = get_logger(__name__)


class StructuredRunner:
    """Runs a structured replication through the LangGraph pipeline.

    Uses run_from_summary() so the replicator only sees the pre-extracted
    methodology summary and data — never the original paper or replication
    package. Verification and explanation are handled separately by the
    benchmark's SharedEvaluator with the judge model.
    """

    def __init__(self, timeout: int = 600):
        self.timeout = timeout

    def run(
        self,
        model: ModelSpec,
        paper: PaperSpec,
        paper_summary: PaperSummary,
        workspace_dir: Path,
    ) -> RunArtifacts:
        """Run the structured pipeline from a pre-extracted summary.

        Args:
            model: Model specification.
            paper: Paper specification (used for data_path).
            paper_summary: Pre-extracted methodology summary (no results).
            workspace_dir: Output directory for this run.

        Returns:
            RunArtifacts with ReplicationResults from the pipeline.
        """
        workspace_dir.mkdir(parents=True, exist_ok=True)

        # Build a Config targeting the specified model
        api_key = os.environ.get(model.api_key_env, "")
        config = Config(
            langgraph=LangGraphConfig(
                default_provider=model.provider,
                default_model=model.model_name,
            ),
        )
        # Set the appropriate API key
        if model.provider.lower() == "openai":
            config.openai_api_key = api_key
        elif model.provider.lower() == "anthropic":
            config.anthropic_api_key = api_key

        logger.info(
            f"Running structured pipeline: model={model.model_name}, paper={paper.paper_id}"
        )
        start = time.time()

        try:
            orchestrator = ReplicationOrchestrator(config=config)
            # Use run_from_summary: replicator only sees methodology summary + data.
            # No paper PDF, no replication package passed to the replication step.
            # Resolve to absolute paths so they work from the workspace directory.
            abs_data_path = str(Path(paper.data_path).resolve())
            abs_paper_path = str(Path(paper.pdf_path).resolve())
            state = orchestrator.run_from_summary(
                paper_summary=paper_summary,
                data_path=abs_data_path,
                paper_path=abs_paper_path,
                output_dir=str(workspace_dir),
            )

            duration = time.time() - start
            errors = state.errors if state.errors else []

            return RunArtifacts(
                workspace_dir=str(workspace_dir),
                stdout=f"Pipeline completed. Step: {state.current_step}",
                stderr="\n".join(errors),
                exit_code=0 if not errors else 1,
                duration_seconds=duration,
                replication_results=state.replication_results,
            )

        except Exception as e:
            duration = time.time() - start
            logger.error(f"Structured pipeline failed: {e}")
            return RunArtifacts(
                workspace_dir=str(workspace_dir),
                stdout="",
                stderr=str(e),
                exit_code=1,
                duration_seconds=duration,
            )
