"""Main benchmark orchestrator.

Three-phase execution:
  Phase 1 — Replication: run all (model × paper × approach) combos, persist
            artifacts immediately.  Existing workspaces are reused (resumption).
  Phase 2 — Evaluation:  judge all completed runs with the shared evaluator.
            Runs already evaluated (verification_report.json exists) are skipped.
  Phase 3 — Explanation: (optional) run agentic explainer on non-A items to
            investigate root causes of discrepancies.
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

load_dotenv()

from ..agents.extractor import ExtractorAgent
from ..models.config import Config, LangGraphConfig
from ..models.schemas import AgenticExplanationReport, PaperResults, PaperSummary, ReplicationGrade
from ..utils.logging_utils import get_logger
from .results_extractor import ResultsExtractor
from .base_runner import BaseReplicationRunner
from .explainer_runner import ExplainerRunner
from .config import (
    BenchmarkConfig,
    ModelSpec,
    PaperSpec,
    parse_approach,
    PAPER_DIRECT_INCOMPATIBLE,
)
from .evaluator import SharedEvaluator
from .claude_code_runner import ClaudeCodeRunner
from .codex_runner import CodexRunner
from .opencode_runner import OpencodeRunner
from .structured_runner import StructuredRunner
from .swe_agent_runner import SweAgentRunner
from .results import (
    BenchmarkResults,
    EvaluationResult,
    ResultsAggregator,
    RunArtifacts,
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

    Execution is split into two phases so that a judge crash never loses
    replication work, and interrupted benchmarks can be resumed.
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)

        # --- Centralised API keys -------------------------------------------
        self.api_keys: dict[str, str] = {}
        for model in config.models:
            self.api_keys[model.api_key_env] = os.environ.get(model.api_key_env, "")
        for env_var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            if env_var not in self.api_keys:
                self.api_keys[env_var] = os.environ.get(env_var, "")

        self.evaluator = SharedEvaluator(config.judge, api_keys=self.api_keys)

        # --- Runners ---------------------------------------------------------
        self._runners: dict[str, BaseReplicationRunner] = {
            "freestyle": OpencodeRunner(
                opencode_binary=config.opencode_binary,
                timeout=config.timeout_seconds,
                allow_web_access=config.allow_web_access,
            ),
            "structured": StructuredRunner(
                timeout=config.timeout_seconds,
                allow_web_access=config.allow_web_access,
            ),
            "claude-code": ClaudeCodeRunner(
                claude_binary=config.claude_code_binary,
                timeout=config.timeout_seconds,
                allow_web_access=config.allow_web_access,
            ),
            "codex": CodexRunner(
                codex_binary=config.codex_binary,
                timeout=config.timeout_seconds,
                allow_web_access=config.allow_web_access,
            ),
            "swe-agent": SweAgentRunner(
                timeout=config.timeout_seconds,
                allow_web_access=config.allow_web_access,
            ),
            "opencode": OpencodeRunner(
                opencode_binary=config.opencode_binary,
                timeout=config.timeout_seconds,
                allow_web_access=config.allow_web_access,
            ),
        }

        self._summary_cache: dict[str, PaperSummary] = {}
        self._results_cache: dict[str, PaperResults] = {}

        # --- Explainer (Phase 3, optional) -----------------------------------
        self._explainer: Optional[ExplainerRunner] = None
        if config.run_explainer:
            self._explainer = ExplainerRunner(
                runner_type=config.explainer_runner_type,
                claude_binary=config.claude_code_binary,
                codex_binary=config.codex_binary,
                timeout=config.explainer_timeout_seconds,
                max_turns=config.explainer_max_turns,
            )

    # -- Convenience aliases for backward compat with tests that access
    #    runner.opencode_runner, runner.evaluator, etc. directly.
    @property
    def opencode_runner(self) -> OpencodeRunner:
        return self._runners["freestyle"]  # type: ignore[return-value]

    @property
    def structured_runner(self) -> StructuredRunner:
        return self._runners["structured"]  # type: ignore[return-value]

    @property
    def claude_code_runner(self) -> ClaudeCodeRunner:
        return self._runners["claude-code"]  # type: ignore[return-value]

    @property
    def codex_runner(self) -> CodexRunner:
        return self._runners["codex"]  # type: ignore[return-value]

    @property
    def swe_agent_runner(self) -> SweAgentRunner:
        return self._runners["swe-agent"]  # type: ignore[return-value]

    # -------------------------------------------------------------------------
    # Summary extraction
    # -------------------------------------------------------------------------

    def _extract_summary(self, paper: PaperSpec) -> PaperSummary:
        """Extract methodology summary for a paper using the judge model.

        Cached per paper_id so extraction only runs once regardless of how
        many (model × approach) combinations use it.
        """
        if paper.paper_id in self._summary_cache:
            logger.info(f"Using cached summary for {paper.paper_id}")
            return self._summary_cache[paper.paper_id]

        # Check on-disk cache
        summary_path = self.output_dir / "summaries" / f"{paper.paper_id}_summary.json"
        if summary_path.exists():
            try:
                data = json.loads(summary_path.read_text())
                summary = PaperSummary(**data)
                self._summary_cache[paper.paper_id] = summary
                logger.info(f"Loaded cached summary for {paper.paper_id} from disk")
                return summary
            except Exception as e:
                logger.warning(f"Could not load cached summary: {e}")

        logger.info(f"Extracting methodology summary for {paper.paper_id} using judge model")

        judge = self.config.judge
        config = Config(
            langgraph=LangGraphConfig(
                default_provider=judge.provider,
                default_model=judge.model_name,
            ),
        )
        if judge.provider.lower() == "openai":
            config.openai_api_key = self.api_keys.get("OPENAI_API_KEY", "")
        elif judge.provider.lower() == "anthropic":
            config.anthropic_api_key = self.api_keys.get("ANTHROPIC_API_KEY", "")

        ext = self.config.extractor
        extractor = ExtractorAgent(
            config,
            model=ext.model,
            max_tokens=ext.max_tokens,
            use_vision=ext.use_vision,
            vision_dpi=ext.vision_dpi,
        )
        summary, usage = extractor.run(
            paper_path=paper.pdf_path,
            paper_id=paper.paper_id,
        )

        self._summary_cache[paper.paper_id] = summary

        # Persist for inspection
        summary_dir = self.output_dir / "summaries"
        summary_dir.mkdir(parents=True, exist_ok=True)
        with open(summary_dir / f"{paper.paper_id}_summary.json", "w") as f:
            json.dump(summary.model_dump(), f, indent=2, default=str)
        with open(summary_dir / f"{paper.paper_id}_usage.json", "w") as f:
            json.dump(usage.summary_dict(), f, indent=2)

        logger.info(
            f"Extracted summary for {paper.paper_id}: "
            f"{len(summary.tables)} tables, {len(summary.figures)} figures"
        )
        return summary

    def _extract_results(self, paper: PaperSpec, paper_summary: PaperSummary) -> PaperResults:
        """Extract original table values from the paper.

        Cached per paper_id so extraction only runs once.
        """
        if paper.paper_id in self._results_cache:
            logger.info(f"Using cached results for {paper.paper_id}")
            return self._results_cache[paper.paper_id]

        # Check on-disk cache first
        results_path = self.output_dir / "summaries" / f"{paper.paper_id}_results.json"
        if results_path.exists():
            try:
                data = json.loads(results_path.read_text())
                paper_results = PaperResults(**data)
                self._results_cache[paper.paper_id] = paper_results
                logger.info(f"Loaded cached results for {paper.paper_id} from disk")
                return paper_results
            except Exception as e:
                logger.warning(f"Could not load cached results: {e}")

        logger.info(f"Extracting original results for {paper.paper_id}")

        ext_config = self.config.extractor
        api_key = self.api_keys.get("OPENAI_API_KEY", "")

        extractor = ResultsExtractor(
            provider="openai",
            model=ext_config.model,
            api_key=api_key,
            use_vision=ext_config.use_vision,
            vision_dpi=ext_config.vision_dpi,
        )

        paper_results = extractor.run(
            paper_path=paper.pdf_path,
            paper_summary=paper_summary,
        )

        self._results_cache[paper.paper_id] = paper_results

        # Persist
        summary_dir = self.output_dir / "summaries"
        summary_dir.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(paper_results.model_dump(), f, indent=2, default=str)
        with open(summary_dir / f"{paper.paper_id}_results_usage.json", "w") as f:
            json.dump(extractor.usage_summary, f, indent=2)

        logger.info(
            f"Extracted results for {paper.paper_id}: "
            f"{len(paper_results.tables)} tables"
        )
        return paper_results

    # -------------------------------------------------------------------------
    # Resumption helpers
    # -------------------------------------------------------------------------

    def _run_dir(self, model: ModelSpec, paper: PaperSpec, approach: str) -> Path:
        return self.output_dir / f"{model.model_name}_{paper.paper_id}_{approach}"

    @staticmethod
    def _has_existing_artifacts(workspace: Path) -> bool:
        """Check whether a previous replication produced usable output."""
        if not workspace.is_dir():
            return False
        # Require at least one artifact file (table csv, figure png, or run log)
        for pattern in ("*.csv", "*.png", "run_log.txt", "run_log.json", "run_log.jsonl"):
            if list(workspace.glob(pattern)):
                return True
        return False

    @staticmethod
    def _load_existing_artifacts(workspace: Path) -> RunArtifacts:
        """Reconstruct RunArtifacts from a previously completed workspace."""
        run_dir = workspace.parent
        usage = None
        usage_path = run_dir / "usage.json"
        if usage_path.exists():
            try:
                usage = json.loads(usage_path.read_text())
            except Exception:
                pass
        return RunArtifacts(
            workspace_dir=str(workspace),
            duration_seconds=0.0,
            usage=usage,
        )

    @staticmethod
    def _has_existing_evaluation(run_dir: Path) -> bool:
        return (run_dir / "verification_report.json").exists()

    @staticmethod
    def _load_existing_evaluation(run_dir: Path) -> EvaluationResult:
        """Load an EvaluationResult from previously saved judge reports."""
        from ..models.schemas import (
            ExplanationReport,
            VerificationReport,
        )

        vr_path = run_dir / "verification_report.json"
        vr = VerificationReport(**json.loads(vr_path.read_text()))

        er = None
        er_path = run_dir / "explanation_report.json"
        if er_path.exists():
            er = ExplanationReport(**json.loads(er_path.read_text()))

        item_grades = {v.item_id: v.grade.value for v in vr.item_verifications}

        return EvaluationResult(
            verification_report=vr,
            explanation_report=er,
            overall_grade=vr.overall_grade.value,
            item_grades=item_grades,
        )

    @staticmethod
    def _has_existing_explainer(run_dir: Path) -> bool:
        return (run_dir / "explainer_report.json").exists()

    @staticmethod
    def _load_existing_explainer(run_dir: Path) -> AgenticExplanationReport:
        """Load an AgenticExplanationReport from a previously saved JSON."""
        path = run_dir / "explainer_report.json"
        return AgenticExplanationReport(**json.loads(path.read_text()))

    @staticmethod
    def _save_explainer_report(run_dir: Path, report: AgenticExplanationReport) -> None:
        """Persist the explainer report to the run directory."""
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "explainer_report.json", "w") as f:
            json.dump(report.model_dump(), f, indent=2, default=str)

    # -------------------------------------------------------------------------
    # Replication dispatch
    # -------------------------------------------------------------------------

    def _run_replication(
        self,
        model: ModelSpec,
        paper: PaperSpec,
        approach: str,
    ) -> RunArtifacts:
        """Dispatch a single replication to the appropriate runner.

        If a workspace with artifacts already exists (from a previous run),
        reuse it instead of re-running.
        """
        base_approach, is_paper_direct = parse_approach(approach)
        workspace = self._run_dir(model, paper, approach) / "workspace"

        # Resumption: skip if workspace already has output
        if self._has_existing_artifacts(workspace):
            logger.info(f"Resuming: reusing existing artifacts in {workspace}")
            return self._load_existing_artifacts(workspace)

        if is_paper_direct and base_approach in PAPER_DIRECT_INCOMPATIBLE:
            raise ValueError(
                f"Approach '{approach}' is not supported: "
                f"'{base_approach}' requires structured PaperSummary input "
                f"and cannot work from a raw PDF."
            )

        runner = self._runners.get(base_approach)
        if runner is None:
            raise ValueError(f"Unknown approach: {base_approach}")

        paper_summary = self._extract_summary(paper)
        return runner.run(
            model, paper, paper_summary, workspace,
            paper_direct=is_paper_direct,
        )

    # -------------------------------------------------------------------------
    # Two-phase benchmark
    # -------------------------------------------------------------------------

    def run(self) -> BenchmarkResults:
        """Run all benchmark combinations in two phases.

        Phase 1 — Replication: run all (model × paper × approach) combos.
        Phase 2 — Evaluation:  judge all completed runs with the shared evaluator.

        Returns:
            BenchmarkResults with all individual run results and summary.
        """
        results = BenchmarkResults()

        # Build the list of (model, paper, approach) combos
        combos: list[tuple[ModelSpec, PaperSpec, str]] = []
        for model in self.config.models:
            model_approaches = (
                [a for a in self.config.approaches if a in model.approaches]
                if model.approaches
                else self.config.approaches
            )
            for paper in self.config.papers:
                for approach in model_approaches:
                    combos.append((model, paper, approach))

        total = len(combos)
        logger.info(
            f"Starting benchmark: {total} runs "
            f"({len(self.config.models)} models × "
            f"{len(self.config.papers)} papers × "
            f"{len(self.config.approaches)} approaches)"
        )

        # Pre-extract summaries and original results for all papers.
        # When results extraction succeeds, inject blinded table skeletons into
        # the summary so the replicator gets pre-aligned JSON templates.
        # If the results extractor returns fewer tables than the summary specifies,
        # retry up to MAX_EXTRACTION_RETRIES times (clearing cache to force re-run).
        MAX_EXTRACTION_RETRIES = 3
        for paper in self.config.papers:
            try:
                summary = self._extract_summary(paper)
                n_summary_tables = len(summary.tables)

                try:
                    paper_results = None
                    for attempt in range(1, MAX_EXTRACTION_RETRIES + 1):
                        paper_results = self._extract_results(paper, summary)
                        n_extracted_with_cells = sum(
                            1 for t in paper_results.tables if t.cells
                        )

                        if n_extracted_with_cells >= n_summary_tables:
                            break

                        if attempt < MAX_EXTRACTION_RETRIES:
                            logger.warning(
                                f"Results extraction for {paper.paper_id}: "
                                f"{n_extracted_with_cells}/{n_summary_tables} tables "
                                f"have cells (attempt {attempt}/{MAX_EXTRACTION_RETRIES}). "
                                f"Retrying..."
                            )
                            # Clear caches to force re-extraction
                            self._results_cache.pop(paper.paper_id, None)
                            results_path = self.output_dir / "summaries" / f"{paper.paper_id}_results.json"
                            if results_path.exists():
                                results_path.unlink()
                        else:
                            logger.warning(
                                f"Results extraction for {paper.paper_id}: "
                                f"{n_extracted_with_cells}/{n_summary_tables} tables "
                                f"have cells after {MAX_EXTRACTION_RETRIES} attempts. "
                                f"Proceeding with partial extraction."
                            )

                    blinded = [t.to_blinded() for t in paper_results.tables if t.cells]
                    summary.extracted_tables = blinded
                    self._summary_cache[paper.paper_id] = summary
                    # Re-persist enriched summary
                    summary_path = self.output_dir / "summaries" / f"{paper.paper_id}_summary.json"
                    summary_path.write_text(
                        json.dumps(summary.model_dump(), indent=2, default=str)
                    )
                    logger.info(
                        f"Injected {len(blinded)}/{n_summary_tables} blinded tables "
                        f"into summary for {paper.paper_id}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to extract results for {paper.paper_id}: {e}")
                    logger.warning("Falling back to markdown templates for this paper")
            except Exception as e:
                logger.error(f"Failed to extract summary for {paper.paper_id}: {e}")

        # --- Phase 1: Replication -------------------------------------------
        logger.info("=== Phase 1: Replication ===")
        pending: list[SingleRunResult] = []

        for idx, (model, paper, approach) in enumerate(combos, 1):
            if paper.paper_id not in self._summary_cache:
                logger.error(f"Skipping {paper.paper_id}: no summary available")
                continue

            logger.info(f"[{idx}/{total}] Replicate: {model.model_name} × {paper.paper_id} × {approach}")
            try:
                artifacts = self._run_replication(model, paper, approach)
                run_result = SingleRunResult(
                    model=model,
                    paper=paper,
                    approach=approach,
                    artifacts=artifacts,
                    duration_seconds=artifacts.duration_seconds,
                )
                # Persist artifacts immediately (evaluation=None → PENDING)
                ResultsAggregator.save_run(run_result, self.output_dir)
                pending.append(run_result)
            except Exception as e:
                logger.error(f"Replication failed: {model.model_name}/{paper.paper_id}/{approach}: {e}")

        logger.info(f"Phase 1 complete: {len(pending)}/{total} replications succeeded.")

        # --- Phase 2: Evaluation --------------------------------------------
        logger.info("=== Phase 2: Evaluation ===")

        for idx, run_result in enumerate(pending, 1):
            run_dir = self._run_dir(run_result.model, run_result.paper, run_result.approach)
            label = f"{run_result.model.model_name}/{run_result.paper.paper_id}/{run_result.approach}"

            # Resumption: skip if already evaluated
            if self._has_existing_evaluation(run_dir):
                logger.info(f"[{idx}/{len(pending)}] Resuming: judge already evaluated {label}")
                try:
                    run_result.evaluation = self._load_existing_evaluation(run_dir)
                except Exception as e:
                    logger.warning(f"Could not load existing evaluation for {label}: {e}")
            else:
                logger.info(f"[{idx}/{len(pending)}] Evaluate: {label}")
                try:
                    paper_summary = self._summary_cache.get(run_result.paper.paper_id)
                    paper_results = self._results_cache.get(run_result.paper.paper_id)
                    evaluation = self.evaluator.evaluate(
                        run_result.paper, run_result.artifacts,
                        paper_summary=paper_summary,
                        paper_results=paper_results,
                    )
                    run_result.evaluation = evaluation
                except Exception as e:
                    logger.error(f"Evaluation failed for {label}: {e}")

            # Re-persist with evaluation included
            ResultsAggregator.save_run(run_result, self.output_dir)
            results.runs.append(run_result)

        # --- Phase 3: Explanation (optional) -----------------------------------
        if self._explainer and self.config.run_explainer:
            logger.info("=== Phase 3: Explanation ===")
            explainer_model = self.config.explainer_model or self.config.models[0]

            for idx, run_result in enumerate(results.runs, 1):
                label = f"{run_result.model.model_name}/{run_result.paper.paper_id}/{run_result.approach}"
                if run_result.evaluation is None:
                    continue

                # Skip if all verifiable items are grade A
                non_a = [
                    v for v in run_result.evaluation.verification_report.item_verifications
                    if v.grade != ReplicationGrade.A and not v.unverifiable and not v.judge_error
                ]
                if not non_a:
                    logger.info(f"[{idx}/{len(results.runs)}] Skip explainer for {label}: all items grade A")
                    continue

                run_dir = self._run_dir(run_result.model, run_result.paper, run_result.approach)

                # Resumption: skip if already explained
                if self._has_existing_explainer(run_dir):
                    logger.info(f"[{idx}/{len(results.runs)}] Resuming: explainer already ran for {label}")
                    try:
                        run_result.agentic_explanation = self._load_existing_explainer(run_dir)
                    except Exception as e:
                        logger.warning(f"Could not load existing explainer report for {label}: {e}")
                    continue

                logger.info(f"[{idx}/{len(results.runs)}] Explain: {label} ({len(non_a)} non-A items)")
                try:
                    replicator_workspace = Path(run_result.artifacts.workspace_dir)
                    explainer_workspace = run_dir / "explainer_workspace"
                    paper_summary = self._summary_cache.get(run_result.paper.paper_id)

                    report = self._explainer.run(
                        model=explainer_model,
                        paper=run_result.paper,
                        paper_summary=paper_summary,
                        replicator_workspace=replicator_workspace,
                        evaluation=run_result.evaluation,
                        workspace_dir=explainer_workspace,
                    )
                    run_result.agentic_explanation = report

                    # Persist explainer report
                    self._save_explainer_report(run_dir, report)
                except Exception as e:
                    logger.error(f"Explainer failed for {label}: {e}")

                # Re-persist with explanation included
                ResultsAggregator.save_run(run_result, self.output_dir)

        # Save aggregated summary
        ResultsAggregator.save_summary(results, self.output_dir)
        logger.info(f"Benchmark complete. {len(results.runs)}/{total} runs finished.")
        return results

    # -------------------------------------------------------------------------
    # Single-run convenience (backward compat)
    # -------------------------------------------------------------------------

    def run_single(
        self,
        model: ModelSpec,
        paper: PaperSpec,
        approach: str,
    ) -> SingleRunResult:
        """Run a single benchmark combination (replicate + evaluate).

        Kept for backward compatibility and ad-hoc usage.
        """
        start = time.time()
        artifacts = self._run_replication(model, paper, approach)
        paper_summary = self._extract_summary(paper)
        try:
            paper_results = self._extract_results(paper, paper_summary)
        except Exception as e:
            logger.warning(f"Could not extract results for {paper.paper_id}: {e}")
            paper_results = None
        evaluation = self.evaluator.evaluate(
            paper, artifacts,
            paper_summary=paper_summary,
            paper_results=paper_results,
        )

        return SingleRunResult(
            model=model,
            paper=paper,
            approach=approach,
            artifacts=artifacts,
            evaluation=evaluation,
            duration_seconds=time.time() - start,
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
