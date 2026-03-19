#!/usr/bin/env python3
"""Run the explainer on all completed benchmark runs that have non-A items.

Usage:
  python scripts/run_explainer.py --results-dir data/i4replicate/results
  python scripts/run_explainer.py --results-dir data/i4replicate/results --runner codex
  python scripts/run_explainer.py --results-dir data/i4replicate/results --papers 10.2139_ssrn.3838127
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.benchmark.config import ModelSpec, PaperSpec
from src.benchmark.explainer_runner import ExplainerRunner
from src.benchmark.results import EvaluationResult
from src.models.schemas import PaperSummary, VerificationReport
from src.utils.logging_utils import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run explainer on completed benchmark runs")
    parser.add_argument("--results-dir", required=True, help="Results directory")
    parser.add_argument("--papers-dir", default=None, help="Papers directory (default: inferred)")
    parser.add_argument("--runner", choices=["claude-code", "codex"], default="codex")
    parser.add_argument("--model", default=None, help="Explainer model (default: auto)")
    parser.add_argument("--papers", nargs="*", default=None, help="Filter to specific paper IDs")
    parser.add_argument("--approaches", nargs="*", default=None, help="Filter to specific approaches")
    parser.add_argument("--timeout", type=int, default=1800, help="Timeout per explainer run")
    parser.add_argument("--force", action="store_true", help="Re-run even if explainer report exists")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if args.papers_dir:
        papers_dir = Path(args.papers_dir)
    else:
        # Infer from results_dir (e.g. data/i4replicate/results -> data/i4replicate/papers)
        papers_dir = results_dir.parent / "papers"

    # Default model
    if args.model:
        model_name = args.model
    elif args.runner == "codex":
        model_name = "gpt-5.3-codex"
    else:
        model_name = "claude-opus-4-6"

    explainer_model = ModelSpec(
        provider="openai" if args.runner == "codex" else "anthropic",
        model_name=model_name,
        api_key_env="OPENAI_API_KEY" if args.runner == "codex" else "ANTHROPIC_API_KEY",
    )

    runner = ExplainerRunner(
        runner_type=args.runner,
        timeout=args.timeout,
    )

    # Find all run directories
    run_dirs = []
    for p in sorted(results_dir.rglob("verification_report.json")):
        run_dir = p.parent
        # Skip explainer workspaces and summaries
        if "explainer" in str(run_dir) or run_dir.name == "summaries":
            continue
        run_dirs.append(run_dir)

    total = 0
    success = 0
    skipped = 0
    errors = 0

    for run_dir in run_dirs:
        # Extract paper_id and approach from dir name
        dir_name = run_dir.name
        parts = dir_name.rsplit("_", 1)
        if len(parts) != 2:
            continue
        approach = parts[1]
        # Paper ID is between first model part and approach
        # e.g. "gpt-5.3-codex_10.2139_ssrn.3838127_codex" -> paper = "10.2139_ssrn.3838127"
        # Find paper ID by checking which paper dir exists
        paper_id = None
        for papers_sub in papers_dir.iterdir():
            if papers_sub.name in dir_name:
                paper_id = papers_sub.name
                break
        if not paper_id:
            # Try extracting from between model and approach
            # Remove approach suffix and try to find paper
            prefix = dir_name[:-(len(approach) + 1)]
            # Remove model prefix (everything up to first paper-like pattern)
            for pd in papers_dir.iterdir():
                if pd.name in prefix:
                    paper_id = pd.name
                    break

        if not paper_id:
            logger.warning(f"Could not determine paper_id for {dir_name}")
            continue

        if args.papers and paper_id not in args.papers:
            continue
        if args.approaches and approach not in args.approaches:
            continue

        # Check if explainer already ran
        explainer_ws = run_dir / "explainer_workspace"
        explainer_report = run_dir / "explainer_report.json"
        if explainer_report.exists() and not args.force:
            skipped += 1
            continue

        # Load verification report
        vr_path = run_dir / "verification_report.json"
        vr = VerificationReport(**json.loads(vr_path.read_text()))

        # Check if there are non-A items
        non_a = [v for v in vr.item_verifications
                 if v.grade.value != "A" and not v.unverifiable and not v.judge_error]
        if not non_a:
            skipped += 1
            continue

        # Load summary
        summaries_dir = run_dir.parent / "summaries"
        if not summaries_dir.exists():
            # Flat layout
            summaries_dir = results_dir / "summaries"
        summary_path = summaries_dir / f"{paper_id}_summary.json"
        if not summary_path.exists():
            logger.warning(f"No summary for {paper_id}")
            continue
        summary = PaperSummary(**json.loads(summary_path.read_text()))

        # Build paper spec
        paper_dir = papers_dir / paper_id
        paper = PaperSpec(
            paper_id=paper_id,
            pdf_path=str(paper_dir / "paper.pdf"),
            data_path=str(paper_dir / "data"),
            replication_package_path=str(paper_dir / "replication_package"),
        )

        # Build evaluation result
        evaluation = EvaluationResult(
            verification_report=vr,
            explanation_report=None,
            overall_grade=vr.overall_grade.value,
            item_grades={v.item_id: v.grade.value for v in vr.item_verifications},
        )

        total += 1
        logger.info(f"[{total}] Explaining {paper_id}/{approach} ({len(non_a)} non-A items)")

        try:
            report = runner.run(
                model=explainer_model,
                paper=paper,
                paper_summary=summary,
                replicator_workspace=run_dir / "workspace",
                evaluation=evaluation,
                workspace_dir=explainer_ws,
            )

            # Save report to run dir
            report_path = run_dir / "explainer_report.json"
            report_path.write_text(
                json.dumps(report.model_dump(), indent=2, default=str)
            )
            logger.info(f"  Saved explainer_report.json")
            success += 1

        except Exception as e:
            logger.error(f"  FAILED: {e}")
            errors += 1

    print(f"\n{'=' * 60}")
    print(f"EXPLAINER COMPLETE")
    print(f"  Total:   {total}")
    print(f"  Success: {success}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors:  {errors}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
