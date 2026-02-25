"""Structured approach runner using the LangGraph pipeline."""

import io
import logging
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

    def __init__(self, timeout: int = 600, allow_web_access: bool = False):
        self.timeout = timeout
        self.allow_web_access = allow_web_access

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

        web_status = "ALLOWED" if self.allow_web_access else "BLOCKED"
        logger.info(
            f"Running structured pipeline: model={model.model_name}, "
            f"paper={paper.paper_id}, web_access={web_status}"
        )
        start = time.time()

        # Capture all log output during the pipeline run
        log_buffer = io.StringIO()
        log_handler = logging.StreamHandler(log_buffer)
        log_handler.setLevel(logging.DEBUG)
        log_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
        # Attach to root logger to capture all modules (agents, orchestrator, executor)
        root_logger = logging.getLogger()
        root_logger.addHandler(log_handler)

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

            stdout = f"Pipeline completed. Step: {state.current_step}"
            stderr = "\n".join(errors)
            exit_code = 0 if not errors else 1

        except Exception as e:
            duration = time.time() - start
            logger.error(f"Structured pipeline failed: {e}")
            stdout = ""
            stderr = str(e)
            exit_code = 1
            state = None

        finally:
            root_logger.removeHandler(log_handler)
            log_handler.close()

        # Save full pipeline log to workspace
        captured_log = log_buffer.getvalue()
        log_path = workspace_dir / "run_log.txt"
        log_path.write_text(
            f"=== STRUCTURED PIPELINE RUN LOG ===\n"
            f"Model: {model.provider}/{model.model_name}\n"
            f"Paper: {paper.paper_id}\n"
            f"Web access: {web_status}\n"
            f"Exit code: {exit_code}\n"
            f"Duration: {duration:.1f}s\n\n"
            f"=== PIPELINE LOG ===\n{captured_log}\n\n"
            f"=== ERRORS ===\n{stderr}\n"
        )

        return RunArtifacts(
            workspace_dir=str(workspace_dir),
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            duration_seconds=duration,
            replication_results=state.replication_results if state else None,
        )
