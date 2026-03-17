#!/usr/bin/env python3
"""Re-evaluate all existing benchmark results using the current grader/judge.

Scans results directories for completed workspaces and re-runs the evaluation
pipeline (artifact parsing + judge) without re-running replication. Overwrites
verification_report.json and explanation_report.json.

Usage:
  python scripts/reeval_results.py --collection i4rep
  python scripts/reeval_results.py --collection postcutoff
  python scripts/reeval_results.py --collection i4rep --papers 10.1093_ej_ueab096
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.benchmark.evaluator import SharedEvaluator
from src.benchmark.config import JudgeConfig, PaperSpec
from src.benchmark.results import RunArtifacts
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Re-evaluate benchmark results")
    parser.add_argument("--collection", choices=["i4rep", "postcutoff", "all"], default="all")
    parser.add_argument("--papers", nargs="*", default=None)
    parser.add_argument("--judge-model", default="gpt-5-mini")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        logger.error("OPENAI_API_KEY not set")
        return

    judge_config = JudgeConfig(
        provider="openai",
        model_name=args.judge_model,
        use_vision=True,
    )
    evaluator = SharedEvaluator(
        judge_config=judge_config,
        api_keys={"OPENAI_API_KEY": api_key},
    )

    # Find all result directories
    collections = []
    if args.collection in ("i4rep", "all"):
        collections.append(("i4rep", project_root / "data" / "i4replicate"))
    if args.collection in ("postcutoff", "all"):
        collections.append(("postcutoff", project_root / "data" / "postcutoff"))

    total = 0
    success = 0
    errors = 0

    for col_name, col_dir in collections:
        results_dir = col_dir / "results"
        papers_dir = col_dir / "papers"
        if not results_dir.exists():
            continue

        for paper_dir in sorted(results_dir.iterdir()):
            if not paper_dir.is_dir() or paper_dir.name == "summaries":
                continue
            paper_id = paper_dir.name

            if args.papers and paper_id not in args.papers:
                continue

            # Find PDF
            pdf_path = papers_dir / paper_id / "paper.pdf"
            if not pdf_path.exists():
                logger.warning(f"No PDF for {paper_id}, skipping")
                continue

            paper = PaperSpec(
                paper_id=paper_id,
                pdf_path=str(pdf_path),
                data_path=str(papers_dir / paper_id / "data"),
            )

            # Find all run directories (model_paper_approach)
            for run_dir in sorted(paper_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                workspace = run_dir / "workspace"
                if not workspace.exists():
                    continue

                # Determine approach from dir name
                approach = run_dir.name.split("_")[-1]
                total += 1

                logger.info(f"[{total}] Re-evaluating {paper_id}/{approach}")

                # Remove old reports
                for f in ["verification_report.json", "explanation_report.json"]:
                    (run_dir / f).unlink(missing_ok=True)

                try:
                    artifacts = RunArtifacts(workspace_dir=str(workspace))
                    result = evaluator.evaluate(paper, artifacts)
                    logger.info(
                        f"  {paper_id}/{approach}: {result.overall_grade} "
                        f"({', '.join(f'{k}={v}' for k, v in result.item_grades.items())})"
                    )
                    success += 1
                except Exception as e:
                    logger.error(f"  {paper_id}/{approach}: FAILED — {e}")
                    errors += 1

    print(f"\n{'=' * 60}")
    print(f"RE-EVALUATION COMPLETE")
    print(f"  Total runs: {total}")
    print(f"  Success:    {success}")
    print(f"  Errors:     {errors}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
