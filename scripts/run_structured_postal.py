"""Re-run only the structured approach for postal_systems using the cached summary."""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.benchmark.config import BenchmarkConfig, JudgeConfig, ModelSpec, PaperSpec
from src.benchmark.evaluator import SharedEvaluator
from src.benchmark.results import ResultsAggregator, SingleRunResult, BenchmarkResults
from src.benchmark.structured_runner import StructuredRunner
from src.models.schemas import PaperSummary
from src.utils.logging_utils import setup_logging, get_logger

setup_logging(level="INFO")
logger = get_logger(__name__)

# ── Config ──────────────────────────────────────────────────────────────
model = ModelSpec(
    provider="openai",
    model_name="gpt-5.2-codex",
    api_key_env="OPENAI_API_KEY",
    approaches=["structured"],
)
paper = PaperSpec(
    paper_id="postal_systems",
    pdf_path="data/input/postal_systems/paper.pdf",
    data_path="data/input/postal_systems/data",
    replication_package_path="data/input/postal_systems/replication_package",
)
judge_config = JudgeConfig(provider="openai", model_name="gpt-5-mini")
output_dir = Path("data/benchmark_results/postal_systems")

# ── Load cached summary ─────────────────────────────────────────────────
summary_path = output_dir / "summaries" / "postal_systems_summary.json"
logger.info(f"Loading cached summary from {summary_path}")
with open(summary_path) as f:
    paper_summary = PaperSummary(**json.load(f))

logger.info(
    f"Summary: {len(paper_summary.tables)} tables, "
    f"{len(paper_summary.figures)} figures"
)

# ── Run structured approach ─────────────────────────────────────────────
runner = StructuredRunner(timeout=3600, allow_web_access=False)
workspace = output_dir / f"{model.model_name}_{paper.paper_id}_structured" / "workspace"

logger.info("Running structured approach...")
start = time.time()
artifacts = runner.run(model, paper, paper_summary, workspace)
run_duration = time.time() - start

logger.info(f"Structured run complete in {run_duration:.1f}s (exit_code={artifacts.exit_code})")

# ── Evaluate with judge ─────────────────────────────────────────────────
logger.info("Evaluating with judge model...")
evaluator = SharedEvaluator(judge_config)
evaluation = evaluator.evaluate(paper, artifacts, paper_summary=paper_summary)

total_duration = time.time() - start

result = SingleRunResult(
    model=model,
    paper=paper,
    approach="structured",
    artifacts=artifacts,
    evaluation=evaluation,
    duration_seconds=total_duration,
)

# ── Save results ────────────────────────────────────────────────────────
ResultsAggregator.save_run(result, output_dir)
results = BenchmarkResults(runs=[result])
ResultsAggregator.save_summary(results, output_dir)

logger.info(f"Done. Results saved to {output_dir}")
if evaluation and evaluation.verification_report:
    logger.info(f"Overall grade: {evaluation.verification_report.overall_grade.value}")
