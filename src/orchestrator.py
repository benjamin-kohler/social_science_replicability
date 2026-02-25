"""Orchestrator for the multi-agent replication workflow.

This module coordinates the agents in the replication pipeline using LangGraph:
0. Collector: Organizes input papers and data (optional)
1. Extractor: Extracts methodology from papers
2. Replicator: Generates and executes replication code
3. Verifier: Compares results with original
4. Explainer: Analyzes discrepancies
"""

import json
from pathlib import Path
from typing import Optional

from langgraph.graph import StateGraph, END

from .agents import ExtractorAgent, ReplicatorAgent
from .benchmark.judge import Judge
from .models.schemas import GraphState, ReplicationState
from .models.config import Config, load_config
from .utils.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


# =============================================================================
# Graph Node Functions
# =============================================================================


def create_extraction_node(config: Config):
    """Create a graph node that runs the Extractor agent."""

    def extraction_node(state: GraphState) -> dict:
        logger.info("Step 1: Extracting methodology...")
        try:
            agent = ExtractorAgent(config)
            paper_summary = agent.run(
                paper_path=state["paper_pdf_path"],
                paper_id=state.get("paper_id"),
            )
            logger.info(
                f"Extraction complete: {len(paper_summary.tables)} tables, "
                f"{len(paper_summary.figures)} figures"
            )

            # Save intermediate result
            output_dir = state.get("output_dir")
            if output_dir and config.output.save_intermediate_results:
                _save_intermediate(paper_summary, Path(output_dir) / "paper_summary.json")

            return {
                "paper_summary": paper_summary,
                "current_step": "extraction",
                "success": True,
            }
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return {
                "errors": [f"Extraction failed: {e}"],
                "current_step": "extraction",
                "success": False,
            }

    return extraction_node


def create_replication_node(config: Config):
    """Create a graph node that runs the Replicator agent."""

    def replication_node(state: GraphState) -> dict:
        logger.info("Step 2: Generating replication...")
        try:
            agent = ReplicatorAgent(config)
            replication_results = agent.run(
                paper_summary=state["paper_summary"],
                data_path=state["data_path"],
                output_dir=state.get("output_dir", "data/output"),
            )
            logger.info(
                f"Replication complete: {len(replication_results.tables)} tables, "
                f"{len(replication_results.figures)} figures"
            )

            output_dir = state.get("output_dir")
            if output_dir and config.output.save_intermediate_results:
                _save_intermediate(
                    replication_results, Path(output_dir) / "replication_results.json"
                )

            return {
                "replication_results": replication_results,
                "current_step": "replication",
                "success": True,
            }
        except Exception as e:
            logger.error(f"Replication failed: {e}")
            return {
                "errors": [f"Replication failed: {e}"],
                "current_step": "replication",
                "success": False,
            }

    return replication_node


def create_judge_node(config: Config):
    """Create a graph node that runs the unified Judge."""

    def judge_node(state: GraphState) -> dict:
        logger.info("Step 3: Judging results...")
        try:
            import os
            provider = config.langgraph.default_provider.lower()
            if provider == "openai":
                api_key = os.environ.get("OPENAI_API_KEY", "")
            elif provider == "anthropic":
                api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            else:
                api_key = ""

            judge = Judge(
                provider=provider,
                model=config.langgraph.default_model,
                api_key=api_key,
            )
            verification_report, explanation_report = judge.run(
                paper_path=state["paper_pdf_path"],
                paper_summary=state["paper_summary"],
                replication_results=state["replication_results"],
                replication_package_path=state.get("replication_package_path"),
            )
            logger.info(
                f"Judging complete. Overall grade: "
                f"{verification_report.overall_grade.value}"
            )

            output_dir = state.get("output_dir")
            if output_dir and config.output.save_intermediate_results:
                _save_intermediate(
                    verification_report, Path(output_dir) / "verification_report.json"
                )
                if explanation_report:
                    _save_intermediate(
                        explanation_report, Path(output_dir) / "explanation_report.json"
                    )

            result = {
                "verification_report": verification_report,
                "current_step": "complete",
                "success": True,
            }
            if explanation_report:
                result["explanation_report"] = explanation_report
            return result
        except Exception as e:
            logger.error(f"Judging failed: {e}")
            return {
                "errors": [f"Judging failed: {e}"],
                "current_step": "judge",
                "success": False,
            }

    return judge_node


# =============================================================================
# Conditional Edge Functions
# =============================================================================


def should_continue(state: GraphState) -> str:
    """Route to next node or END based on success flag."""
    if state.get("success", False):
        return "continue"
    return "stop"


# =============================================================================
# Graph Builders
# =============================================================================


def build_replication_graph(config: Config) -> StateGraph:
    """Build the full 4-step replication graph.

    Returns:
        A compiled LangGraph StateGraph.
    """
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("extract", create_extraction_node(config))
    graph.add_node("replicate", create_replication_node(config))
    graph.add_node("judge", create_judge_node(config))

    # Set entry point
    graph.set_entry_point("extract")

    # Add conditional edges: stop pipeline on failure
    graph.add_conditional_edges("extract", should_continue, {"continue": "replicate", "stop": END})
    graph.add_conditional_edges("replicate", should_continue, {"continue": "judge", "stop": END})
    graph.add_edge("judge", END)

    return graph.compile()


def build_extraction_only_graph(config: Config) -> StateGraph:
    """Build a graph that only runs extraction.

    Returns:
        A compiled LangGraph StateGraph.
    """
    graph = StateGraph(GraphState)
    graph.add_node("extract", create_extraction_node(config))
    graph.set_entry_point("extract")
    graph.add_edge("extract", END)
    return graph.compile()


def build_from_summary_graph(config: Config) -> StateGraph:
    """Build a graph that starts from an existing summary (skips extraction).

    Returns:
        A compiled LangGraph StateGraph.
    """
    graph = StateGraph(GraphState)
    graph.add_node("replicate", create_replication_node(config))
    graph.add_node("judge", create_judge_node(config))

    graph.set_entry_point("replicate")
    graph.add_conditional_edges("replicate", should_continue, {"continue": "judge", "stop": END})
    graph.add_edge("judge", END)

    return graph.compile()


def build_extract_replicate_graph(config: Config) -> StateGraph:
    """Build a graph that runs extraction + replication only (no verification/explanation).

    Useful for benchmarking, where a separate judge model handles evaluation.

    Returns:
        A compiled LangGraph StateGraph.
    """
    graph = StateGraph(GraphState)
    graph.add_node("extract", create_extraction_node(config))
    graph.add_node("replicate", create_replication_node(config))

    graph.set_entry_point("extract")
    graph.add_conditional_edges("extract", should_continue, {"continue": "replicate", "stop": END})
    graph.add_edge("replicate", END)

    return graph.compile()


# =============================================================================
# Helper Functions
# =============================================================================


def _save_intermediate(obj, path: Path) -> None:
    """Save an intermediate result to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj.model_dump(), f, indent=2, default=str)
    logger.debug(f"Saved intermediate result to: {path}")


def _graph_state_to_replication_state(
    graph_result: dict,
    paper_pdf_path: str,
    data_path: str,
    replication_package_path: Optional[str] = None,
) -> ReplicationState:
    """Convert LangGraph output dict to a ReplicationState for backward compatibility."""
    return ReplicationState(
        paper_pdf_path=paper_pdf_path,
        data_path=data_path,
        replication_package_path=replication_package_path,
        paper_summary=graph_result.get("paper_summary"),
        replication_results=graph_result.get("replication_results"),
        verification_report=graph_result.get("verification_report"),
        explanation_report=graph_result.get("explanation_report"),
        errors=graph_result.get("errors", []),
        warnings=graph_result.get("warnings", []),
        current_step=graph_result.get("current_step"),
    )


# =============================================================================
# Backward-Compatible Orchestrator
# =============================================================================


class ReplicationOrchestrator:
    """Orchestrates the multi-agent replication workflow.

    This class provides the same API as the original orchestrator
    but uses LangGraph under the hood.
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
            self.config.langgraph.default_provider = model_provider
        if model_name:
            self.config.langgraph.default_model = model_name

        logger.info(
            f"Orchestrator initialized with {self.config.langgraph.default_provider}/"
            f"{self.config.langgraph.default_model}"
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

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Build and run the graph
        compiled_graph = build_replication_graph(self.config)

        initial_state: GraphState = {
            "paper_pdf_path": paper_path,
            "data_path": data_path,
            "output_dir": str(output_path),
            "paper_id": paper_id or Path(paper_path).stem,
            "replication_package_path": replication_package_path,
            "errors": [],
            "warnings": [],
            "current_step": "starting",
            "success": True,
        }

        result = compiled_graph.invoke(initial_state)

        logger.info("Replication pipeline complete!")
        return _graph_state_to_replication_state(
            result, paper_path, data_path, replication_package_path
        )

    def run_extraction_only(
        self,
        paper_path: str,
        paper_id: Optional[str] = None,
    ) -> ReplicationState:
        """Run only the extraction step (Agent 1).

        Args:
            paper_path: Path to the paper PDF.
            paper_id: Optional identifier for the paper.

        Returns:
            ReplicationState with paper_summary populated.
        """
        compiled_graph = build_extraction_only_graph(self.config)

        initial_state: GraphState = {
            "paper_pdf_path": paper_path,
            "data_path": "",
            "paper_id": paper_id or Path(paper_path).stem,
            "errors": [],
            "warnings": [],
            "current_step": "starting",
            "success": True,
        }

        result = compiled_graph.invoke(initial_state)

        # Map current_step for backward compatibility
        if result.get("paper_summary") is not None:
            result["current_step"] = "extraction_complete"

        return _graph_state_to_replication_state(result, paper_path, "")

    def run_from_summary(
        self,
        paper_summary,
        data_path: str,
        paper_path: str,
        replication_package_path: Optional[str] = None,
        output_dir: str = "data/output",
    ) -> ReplicationState:
        """Run the pipeline starting from an existing paper summary.

        Args:
            paper_summary: Existing PaperSummary object.
            data_path: Path to the data file(s).
            paper_path: Path to the original paper (for verification).
            replication_package_path: Optional path to replication package.
            output_dir: Directory for output files.

        Returns:
            ReplicationState with all results.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        compiled_graph = build_from_summary_graph(self.config)

        initial_state: GraphState = {
            "paper_pdf_path": paper_path,
            "data_path": data_path,
            "output_dir": str(output_path),
            "paper_id": paper_summary.paper_id,
            "replication_package_path": replication_package_path,
            "paper_summary": paper_summary,
            "errors": [],
            "warnings": [],
            "current_step": "starting",
            "success": True,
        }

        result = compiled_graph.invoke(initial_state)

        return _graph_state_to_replication_state(
            result, paper_path, data_path, replication_package_path
        )
