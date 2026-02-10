"""Orchestrator for the multi-agent replication workflow.

This module coordinates the four agents in the replication pipeline:
1. Extractor: Extracts methodology from papers
2. Replicator: Generates and executes replication code
3. Verifier: Compares results with original
4. Explainer: Analyzes discrepancies
"""

from pathlib import Path
from typing import Optional

from .agents import ExtractorAgent, ReplicatorAgent, VerifierAgent, ExplainerAgent
from .models.schemas import ReplicationState
from .models.config import Config, load_config
from .utils.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


class ReplicationOrchestrator:
    """Orchestrates the multi-agent replication workflow.

    This class manages the sequential execution of all four agents
    and handles state passing between them.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        model_provider: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        """Initialize the orchestrator.

        Args:
            config: Configuration object. If None, loads default config.
            model_provider: Override config's model provider.
            model_name: Override config's model name.
        """
        self.config = config or load_config()

        # Override model settings if provided
        if model_provider:
            self.config.open_agent.default_provider = model_provider
        if model_name:
            self.config.open_agent.default_model = model_name

        # Initialize agents
        self.extractor = ExtractorAgent(self.config)
        self.replicator = ReplicatorAgent(self.config)
        self.verifier = VerifierAgent(self.config)
        self.explainer = ExplainerAgent(self.config)

        logger.info(
            f"Orchestrator initialized with {self.config.open_agent.default_provider}/"
            f"{self.config.open_agent.default_model}"
        )

    def run(
        self,
        paper_path: str,
        data_path: str,
        replication_package_path: Optional[str] = None,
        output_dir: str = "data/output",
        paper_id: Optional[str] = None,
    ) -> ReplicationState:
        """Run the full replication pipeline.

        Args:
            paper_path: Path to the paper PDF.
            data_path: Path to the data file(s).
            replication_package_path: Optional path to original replication package.
            output_dir: Directory for output files.
            paper_id: Optional identifier for the paper.

        Returns:
            ReplicationState with all results.
        """
        logger.info(f"Starting replication pipeline for: {paper_path}")

        # Initialize state
        state = ReplicationState(
            paper_pdf_path=paper_path,
            data_path=data_path,
            replication_package_path=replication_package_path,
        )

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            # Step 1: Extract methodology
            state.current_step = "extraction"
            logger.info("Step 1: Extracting methodology...")

            state.paper_summary = self.extractor.run(
                paper_path=paper_path,
                paper_id=paper_id,
            )

            logger.info(
                f"Extraction complete: {len(state.paper_summary.tables)} tables, "
                f"{len(state.paper_summary.figures)} figures"
            )

            # Save intermediate result
            if self.config.output.save_intermediate_results:
                self._save_intermediate(state.paper_summary, output_path / "paper_summary.json")

        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            state.errors.append(f"Extraction failed: {e}")
            return state

        try:
            # Step 2: Replicate
            state.current_step = "replication"
            logger.info("Step 2: Generating replication...")

            state.replication_results = self.replicator.run(
                paper_summary=state.paper_summary,
                data_path=data_path,
                output_dir=str(output_path),
            )

            logger.info(
                f"Replication complete: {len(state.replication_results.tables)} tables, "
                f"{len(state.replication_results.figures)} figures"
            )

            # Save intermediate result
            if self.config.output.save_intermediate_results:
                self._save_intermediate(
                    state.replication_results, output_path / "replication_results.json"
                )

        except Exception as e:
            logger.error(f"Replication failed: {e}")
            state.errors.append(f"Replication failed: {e}")
            return state

        try:
            # Step 3: Verify
            state.current_step = "verification"
            logger.info("Step 3: Verifying results...")

            state.verification_report = self.verifier.run(
                paper_path=paper_path,
                replication_results=state.replication_results,
            )

            logger.info(f"Verification complete. Overall grade: {state.verification_report.overall_grade.value}")

            # Save intermediate result
            if self.config.output.save_intermediate_results:
                self._save_intermediate(
                    state.verification_report, output_path / "verification_report.json"
                )

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            state.errors.append(f"Verification failed: {e}")
            return state

        try:
            # Step 4: Explain discrepancies
            state.current_step = "explanation"
            logger.info("Step 4: Analyzing discrepancies...")

            state.explanation_report = self.explainer.run(
                paper_path=paper_path,
                paper_summary=state.paper_summary,
                replication_results=state.replication_results,
                verification_report=state.verification_report,
                replication_package_path=replication_package_path,
            )

            logger.info(f"Explanation complete: {len(state.explanation_report.analyses)} discrepancies analyzed")

            # Save intermediate result
            if self.config.output.save_intermediate_results:
                self._save_intermediate(
                    state.explanation_report, output_path / "explanation_report.json"
                )

        except Exception as e:
            logger.error(f"Explanation failed: {e}")
            state.errors.append(f"Explanation failed: {e}")
            return state

        state.current_step = "complete"
        logger.info("Replication pipeline complete!")

        return state

    def run_extraction_only(
        self,
        paper_path: str,
        paper_id: Optional[str] = None,
    ) -> ReplicationState:
        """Run only the extraction step (Agent 1).

        Useful for testing or when you only need the methodology summary.

        Args:
            paper_path: Path to the paper PDF.
            paper_id: Optional identifier for the paper.

        Returns:
            ReplicationState with paper_summary populated.
        """
        state = ReplicationState(
            paper_pdf_path=paper_path,
            data_path="",  # Not used in extraction only
        )

        try:
            state.paper_summary = self.extractor.run(
                paper_path=paper_path,
                paper_id=paper_id,
            )
            state.current_step = "extraction_complete"
        except Exception as e:
            state.errors.append(f"Extraction failed: {e}")

        return state

    def run_from_summary(
        self,
        paper_summary,
        data_path: str,
        paper_path: str,
        replication_package_path: Optional[str] = None,
        output_dir: str = "data/output",
    ) -> ReplicationState:
        """Run the pipeline starting from an existing paper summary.

        Useful when you've already extracted the methodology and want
        to re-run the replication with different data or settings.

        Args:
            paper_summary: Existing PaperSummary object.
            data_path: Path to the data file(s).
            paper_path: Path to the original paper (for verification).
            replication_package_path: Optional path to replication package.
            output_dir: Directory for output files.

        Returns:
            ReplicationState with all results.
        """
        state = ReplicationState(
            paper_pdf_path=paper_path,
            data_path=data_path,
            replication_package_path=replication_package_path,
            paper_summary=paper_summary,
        )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Continue from step 2
        try:
            state.current_step = "replication"
            state.replication_results = self.replicator.run(
                paper_summary=paper_summary,
                data_path=data_path,
                output_dir=str(output_path),
            )
        except Exception as e:
            state.errors.append(f"Replication failed: {e}")
            return state

        try:
            state.current_step = "verification"
            state.verification_report = self.verifier.run(
                paper_path=paper_path,
                replication_results=state.replication_results,
            )
        except Exception as e:
            state.errors.append(f"Verification failed: {e}")
            return state

        try:
            state.current_step = "explanation"
            state.explanation_report = self.explainer.run(
                paper_path=paper_path,
                paper_summary=paper_summary,
                replication_results=state.replication_results,
                verification_report=state.verification_report,
                replication_package_path=replication_package_path,
            )
        except Exception as e:
            state.errors.append(f"Explanation failed: {e}")
            return state

        state.current_step = "complete"
        return state

    def _save_intermediate(self, obj, path: Path) -> None:
        """Save an intermediate result to JSON."""
        import json

        with open(path, "w") as f:
            json.dump(obj.model_dump(), f, indent=2, default=str)
        logger.debug(f"Saved intermediate result to: {path}")
