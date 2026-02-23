"""Main benchmark orchestrator."""

import os
import time
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

load_dotenv()

from ..agents.extractor import ExtractorAgent
from ..models.config import Config, LangGraphConfig
from ..models.schemas import PaperSummary
from ..utils.logging_utils import get_logger
from .config import BenchmarkConfig, ModelSpec, PaperSpec
from .evaluator import SharedEvaluator
from .opencode_runner import OpencodeRunner
from .structured_runner import StructuredRunner
from .results import (
    BenchmarkResults,
    ResultsAggregator,
    SingleRunResult,
)

logger = get_logger(__name__)


class BenchmarkRunner:
    """Orchestrates benchmark runs across models, papers, and approaches.

    Extracts a methodology summary once per paper using the judge model,
    then feeds the same summary to all (model × approach) combinations.
    This ensures:
    - The replicator only sees methodology + data (no paper, no results, no original code)
    - All models work from identical input for fair comparison
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.evaluator = SharedEvaluator(config.judge)
        self.opencode_runner = OpencodeRunner(
            opencode_binary=config.opencode_binary,
            timeout=config.timeout_seconds,
        )
        self.structured_runner = StructuredRunner(timeout=config.timeout_seconds)
        self.output_dir = Path(config.output_dir)
        self._summary_cache: dict[str, PaperSummary] = {}

    def _extract_summary(self, paper: PaperSpec) -> PaperSummary:
        """Extract methodology summary for a paper using the judge model.

        Cached per paper_id so extraction only runs once regardless of how
        many (model × approach) combinations use it.

        Args:
            paper: Paper specification with PDF path.

        Returns:
            PaperSummary containing methodology without results.
        """
        if paper.paper_id in self._summary_cache:
            logger.info(f"Using cached summary for {paper.paper_id}")
            return self._summary_cache[paper.paper_id]

        logger.info(f"Extracting methodology summary for {paper.paper_id} using judge model")

        # Build config for the judge model
        judge = self.config.judge
        config = Config(
            langgraph=LangGraphConfig(
                default_provider=judge.provider,
                default_model=judge.model_name,
            ),
        )
        if judge.provider.lower() == "openai":
            config.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        elif judge.provider.lower() == "anthropic":
            config.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")

        extractor = ExtractorAgent(config)
        summary = extractor.run(
            paper_path=paper.pdf_path,
            paper_id=paper.paper_id,
        )

        self._summary_cache[paper.paper_id] = summary

        # Save the summary for inspection
        import json
        summary_dir = self.output_dir / "summaries"
        summary_dir.mkdir(parents=True, exist_ok=True)
        with open(summary_dir / f"{paper.paper_id}_summary.json", "w") as f:
            json.dump(summary.model_dump(), f, indent=2, default=str)

        logger.info(
            f"Extracted summary for {paper.paper_id}: "
            f"{len(summary.tables)} tables, {len(summary.figures)} figures"
        )
        return summary

    def run(self) -> BenchmarkResults:
        """Run all benchmark combinations.

        Returns:
            BenchmarkResults with all individual run results and summary.
        """
        results = BenchmarkResults()
        total = len(self.config.models) * len(self.config.papers) * len(self.config.approaches)
        logger.info(f"Starting benchmark: {total} runs ({len(self.config.models)} models × "
                     f"{len(self.config.papers)} papers × {len(self.config.approaches)} approaches)")

        # Pre-extract summaries for all papers
        for paper in self.config.papers:
            try:
                self._extract_summary(paper)
            except Exception as e:
                logger.error(f"Failed to extract summary for {paper.paper_id}: {e}")

        run_idx = 0
        for model in self.config.models:
            # Filter approaches for this model
            model_approaches = (
                [a for a in self.config.approaches if a in model.approaches]
                if model.approaches
                else self.config.approaches
            )
            for paper in self.config.papers:
                if paper.paper_id not in self._summary_cache:
                    logger.error(f"Skipping {paper.paper_id}: no summary available")
                    continue

                for approach in model_approaches:
                    run_idx += 1
                    logger.info(
                        f"[{run_idx}/{total}] {model.model_name} × "
                        f"{paper.paper_id} × {approach}"
                    )
                    try:
                        run_result = self.run_single(model, paper, approach)
                        results.runs.append(run_result)
                        ResultsAggregator.save_run(run_result, self.output_dir)
                    except Exception as e:
                        logger.error(
                            f"Run failed: {model.model_name}/{paper.paper_id}/{approach}: {e}"
                        )

        # Save aggregated summary
        ResultsAggregator.save_summary(results, self.output_dir)
        logger.info(f"Benchmark complete. {len(results.runs)}/{total} runs succeeded.")
        return results

    def run_single(
        self,
        model: ModelSpec,
        paper: PaperSpec,
        approach: str,
    ) -> SingleRunResult:
        """Run a single benchmark combination.

        Args:
            model: Model specification.
            paper: Paper specification.
            approach: 'freestyle' or 'structured'.

        Returns:
            SingleRunResult with artifacts and evaluation.
        """
        start = time.time()
        workspace = self.output_dir / f"{model.model_name}_{paper.paper_id}_{approach}" / "workspace"
        paper_summary = self._extract_summary(paper)

        if approach == "freestyle":
            artifacts = self.opencode_runner.run(model, paper, paper_summary, workspace)
        elif approach == "structured":
            artifacts = self.structured_runner.run(model, paper, paper_summary, workspace)
        else:
            raise ValueError(f"Unknown approach: {approach}")

        # Evaluate with shared judge, passing summary for the Explainer
        evaluation = self.evaluator.evaluate(paper, artifacts, paper_summary=paper_summary)

        total_duration = time.time() - start

        return SingleRunResult(
            model=model,
            paper=paper,
            approach=approach,
            artifacts=artifacts,
            evaluation=evaluation,
            duration_seconds=total_duration,
        )


def run_benchmark(
    config_path: Optional[str] = None,
    config: Optional[BenchmarkConfig] = None,
) -> BenchmarkResults:
    """Convenience function to run a benchmark.

    Args:
        config_path: Path to a YAML config file.
        config: Direct BenchmarkConfig object (takes precedence over config_path).

    Returns:
        BenchmarkResults.
    """
    if config is None:
        if config_path is None:
            raise ValueError("Either config_path or config must be provided")
        with open(config_path) as f:
            data = yaml.safe_load(f)
        config = BenchmarkConfig(**data)

    runner = BenchmarkRunner(config)
    return runner.run()
