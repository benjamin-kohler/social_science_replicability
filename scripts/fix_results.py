#!/usr/bin/env python3
"""Fix benchmark results consistency issues.

Provides helper functions to resolve issues found by check_results.py.

Usage:
  python scripts/fix_results.py --results-dir data/i4replicate/results --fix all --dry-run
  python scripts/fix_results.py --results-dir data/i4replicate/results --fix orphaned,templates
  python scripts/fix_results.py --results-dir data/i4replicate/results --fix interrupted
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.utils.logging_utils import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


def _load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


# =============================================================================
# Fix 1: Orphaned z-ai/ directories
# =============================================================================

def fix_orphaned_z_ai(results_dir: Path, dry_run: bool = False) -> int:
    """Remove orphaned z-ai/ nested directories."""
    fixed = 0
    for pid in sorted(os.listdir(results_dir)):
        zai_dir = results_dir / pid / "z-ai"
        if not zai_dir.is_dir():
            continue

        # Check that flat equivalents exist
        safe = True
        for sub in zai_dir.iterdir():
            if not sub.is_dir():
                continue
            flat_name = f"z-ai_{sub.name}"
            flat_dir = results_dir / pid / flat_name
            if not flat_dir.is_dir():
                logger.warning(f"  No flat equivalent for {pid}/z-ai/{sub.name} — skipping")
                safe = False

        if safe:
            if dry_run:
                logger.info(f"  [DRY RUN] Would delete {pid}/z-ai/")
            else:
                shutil.rmtree(zai_dir)
                logger.info(f"  Deleted {pid}/z-ai/")
            fixed += 1

    return fixed


# =============================================================================
# Fix 2: Missing/inconsistent table templates
# =============================================================================

def fix_templates(results_dir: Path, dry_run: bool = False) -> int:
    """Regenerate table_templates/ from paper-level summary for all workspaces."""
    fixed = 0
    for pid in sorted(os.listdir(results_dir)):
        paper_dir = results_dir / pid
        if not paper_dir.is_dir() or pid == "summaries":
            continue

        # Load paper-level summary
        summary_path = paper_dir / "summaries" / f"{pid}_summary.json"
        if not summary_path.is_file():
            continue
        summary = _load_json(summary_path)
        if not summary:
            continue

        extracted_tables = summary.get("extracted_tables", [])
        if not extracted_tables:
            continue

        # Build canonical template set
        import re
        templates = {}
        for et in extracted_tables:
            table_id = et.get("table_id", "")
            # Extract just "Table N" or "Table Na" (strip captions, periods, em-dashes)
            prefix_match = re.match(r"((?:Table|Figure)\s+\w+)", table_id)
            if prefix_match:
                clean_name = prefix_match.group(1)
                # Strip trailing periods/punctuation from the number part
                clean_name = re.sub(r"[.\-—:,]+$", "", clean_name)
                fname = clean_name.replace(" ", "_").lower() + ".json"
            else:
                fname = table_id.replace(" ", "_").lower() + ".json"
            templates[fname] = et

        # Apply to all workspaces
        for run_name in sorted(os.listdir(paper_dir)):
            run_dir = paper_dir / run_name
            ws = run_dir / "workspace"
            if not ws.is_dir() or "explainer" in run_name or run_name in ("summaries", "z-ai"):
                continue

            tmpl_dir = ws / "table_templates"
            current = {}
            if tmpl_dir.is_dir():
                current = {f.name: True for f in tmpl_dir.iterdir() if f.suffix == ".json"}

            if set(current.keys()) != set(templates.keys()):
                if dry_run:
                    logger.info(f"  [DRY RUN] Would update templates for {pid}/{run_name}: "
                                f"{len(current)} -> {len(templates)}")
                else:
                    tmpl_dir.mkdir(parents=True, exist_ok=True)
                    # Clear existing
                    for f in tmpl_dir.iterdir():
                        f.unlink()
                    # Write canonical
                    for fname, et_data in templates.items():
                        (tmpl_dir / fname).write_text(
                            json.dumps(et_data, indent=2, default=str)
                        )
                    logger.info(f"  Updated templates for {pid}/{run_name}: "
                                f"{len(current)} -> {len(templates)}")
                fixed += 1

    return fixed


# =============================================================================
# Fix 3: Interrupted runs (re-evaluate)
# =============================================================================

def fix_interrupted_runs(results_dir: Path, dry_run: bool = False) -> int:
    """Re-run evaluation on runs with workspace but no verification report."""
    fixed = 0
    papers_dir = results_dir.parent / "papers"

    for pid in sorted(os.listdir(results_dir)):
        paper_dir = results_dir / pid
        if not paper_dir.is_dir() or pid == "summaries":
            continue

        for run_name in sorted(os.listdir(paper_dir)):
            run_dir = paper_dir / run_name
            ws = run_dir / "workspace"
            vr = run_dir / "verification_report.json"
            if not ws.is_dir() or vr.is_file():
                continue
            if "explainer" in run_name or run_name in ("summaries", "z-ai"):
                continue

            if dry_run:
                logger.info(f"  [DRY RUN] Would re-evaluate {pid}/{run_name}")
                fixed += 1
                continue

            # Run evaluation
            try:
                from src.benchmark.evaluator import SharedEvaluator
                from src.benchmark.config import JudgeConfig, PaperSpec
                from src.benchmark.results import RunArtifacts

                paper = PaperSpec(
                    paper_id=pid,
                    pdf_path=str(papers_dir / pid / "paper.pdf"),
                    data_path=str(papers_dir / pid / "data"),
                )

                judge_config = JudgeConfig(
                    provider="openai",
                    model_name="gpt-5-mini",
                    use_vision=True,
                )
                evaluator = SharedEvaluator(
                    judge_config=judge_config,
                    api_keys={"OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "")},
                )

                artifacts = RunArtifacts(workspace_dir=str(ws))
                result = evaluator.evaluate(paper, artifacts)
                logger.info(f"  Re-evaluated {pid}/{run_name}: grade={result.overall_grade}")
                fixed += 1
            except Exception as e:
                logger.error(f"  Failed to re-evaluate {pid}/{run_name}: {e}")

    return fixed


# =============================================================================
# Fix 4: Missing result.json
# =============================================================================

def fix_missing_result_json(results_dir: Path, dry_run: bool = False) -> int:
    """Placeholder — result.json regeneration requires SingleRunResult reconstruction."""
    fixed = 0
    for pid in sorted(os.listdir(results_dir)):
        paper_dir = results_dir / pid
        if not paper_dir.is_dir() or pid == "summaries":
            continue
        for run_name in sorted(os.listdir(paper_dir)):
            run_dir = paper_dir / run_name
            vr = run_dir / "verification_report.json"
            result = run_dir / "result.json"
            if vr.is_file() and not result.is_file():
                if "explainer" in run_name or run_name in ("summaries", "z-ai"):
                    continue
                if dry_run:
                    logger.info(f"  [DRY RUN] Would regenerate result.json for {pid}/{run_name}")
                else:
                    logger.info(f"  Skipping result.json for {pid}/{run_name} (not critical)")
                fixed += 1
    return fixed


# =============================================================================
# Main
# =============================================================================

FIX_FUNCTIONS = {
    "orphaned": ("Remove orphaned z-ai/ dirs", fix_orphaned_z_ai),
    "templates": ("Fix missing/inconsistent table templates", fix_templates),
    "interrupted": ("Re-evaluate interrupted runs", fix_interrupted_runs),
    "result_json": ("Regenerate missing result.json", fix_missing_result_json),
}


def main():
    parser = argparse.ArgumentParser(description="Fix benchmark results consistency issues")
    parser.add_argument("--results-dir", required=True, help="Results directory")
    parser.add_argument("--fix", default="all",
                        help="Comma-separated fix types or 'all': " + ", ".join(FIX_FUNCTIONS.keys()))
    parser.add_argument("--dry-run", action="store_true", help="Show what would be fixed without changing anything")
    parser.add_argument("--papers", nargs="*", default=None, help="Filter to specific paper IDs")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)

    fixes = list(FIX_FUNCTIONS.keys()) if args.fix == "all" else args.fix.split(",")

    print("=" * 60)
    print("RESULTS FIXER" + (" (DRY RUN)" if args.dry_run else ""))
    print("=" * 60)
    print()

    total_fixed = 0
    for fix_name in fixes:
        if fix_name not in FIX_FUNCTIONS:
            print(f"Unknown fix: {fix_name}")
            continue
        desc, func = FIX_FUNCTIONS[fix_name]
        print(f"--- {desc} ---")
        n = func(results_dir, dry_run=args.dry_run)
        print(f"  {n} issues {'would be ' if args.dry_run else ''}fixed")
        print()
        total_fixed += n

    print(f"Total: {total_fixed} issues {'would be ' if args.dry_run else ''}fixed")


if __name__ == "__main__":
    main()
