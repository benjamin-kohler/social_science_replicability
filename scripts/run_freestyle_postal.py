"""Re-run the freestyle approach for postal_systems with gpt-5.2-codex.

Uses the cached PaperSummary (no re-extraction), runs opencode CLI,
then evaluates with the shared judge (gpt-5-mini).
"""

import json
import os
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.benchmark.config import JudgeConfig, ModelSpec, PaperSpec
from src.benchmark.evaluator import SharedEvaluator
from src.benchmark.opencode_runner import OpencodeRunner
from src.benchmark.results import ResultsAggregator, SingleRunResult
from src.models.schemas import PaperSummary

# ── Paths ─────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parent.parent
summary_path = project_root / "data/benchmark_results/postal_systems/summaries/postal_systems_summary.json"
output_dir = project_root / "data/benchmark_results/postal_systems"

# ── Load cached summary ──────────────────────────────────────
with open(summary_path) as f:
    paper_summary = PaperSummary(**json.load(f))
print(f"Loaded summary: {paper_summary.paper_id} — {len(paper_summary.tables)} tables, {len(paper_summary.figures)} figures")

# ── Configure ────────────────────────────────────────────────
model = ModelSpec(
    provider="openai",
    model_name="gpt-5.2-codex",
    api_key_env="OPENAI_API_KEY",
    approaches=["freestyle"],
)
paper = PaperSpec(
    paper_id="postal_systems",
    pdf_path=str(project_root / "data/input/postal_systems/paper.pdf"),
    data_path=str(project_root / "data/input/postal_systems/data"),
    replication_package_path=str(project_root / "data/input/postal_systems/replication_package"),
)
judge_config = JudgeConfig(provider="openai", model_name="gpt-5-mini")

# ── Run freestyle approach ───────────────────────────────────
runner = OpencodeRunner(
    opencode_binary=os.path.expanduser("~/.opencode/bin/opencode"),
    timeout=3600,
    allow_web_access=False,
)
workspace = output_dir / f"{model.model_name}_{paper.paper_id}_freestyle" / "workspace"

print(f"\nRunning freestyle for {paper.paper_id} with {model.model_name}...")
print(f"Workspace: {workspace}")
print(f"Timeout: {runner.timeout}s")
print()

artifacts = runner.run(model, paper, paper_summary, workspace)

print(f"\n{'='*60}")
print(f"Opencode finished: exit_code={artifacts.exit_code}, duration={artifacts.duration_seconds:.1f}s")

# List workspace output files
ws = Path(artifacts.workspace_dir)
output_files = [f.name for f in ws.iterdir() if f.suffix in (".csv", ".png", ".py")]
print(f"Output files: {sorted(output_files)}")

# ── Evaluate with judge ──────────────────────────────────────
print(f"\nRunning judge ({judge_config.model_name})...")
evaluator = SharedEvaluator(judge_config)
evaluation = evaluator.evaluate(paper, artifacts, paper_summary=paper_summary)

print(f"\n{'='*60}")
print(f"Overall grade: {evaluation.overall_grade}")
print(f"Item grades:")
for item_id, grade in sorted(evaluation.item_grades.items()):
    v = next(v for v in evaluation.verification_report.item_verifications if v.item_id == item_id)
    flags = []
    if v.unverifiable:
        flags.append("unverifiable")
    if v.judge_error:
        flags.append("judge_error")
    flag_str = f" [{', '.join(flags)}]" if flags else ""
    print(f"  {item_id}: {grade}{flag_str}")

# ── Save results ─────────────────────────────────────────────
result = SingleRunResult(
    model=model,
    paper=paper,
    approach="freestyle",
    artifacts=artifacts,
    evaluation=evaluation,
    duration_seconds=artifacts.duration_seconds,
)
ResultsAggregator.save_run(result, output_dir)
print(f"\nResults saved to: {output_dir / f'{model.model_name}_{paper.paper_id}_freestyle'}")
