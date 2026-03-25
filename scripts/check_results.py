#!/usr/bin/env python3
"""Diagnostic checker for benchmark results consistency.

Scans all results directories and reports issues: interrupted runs,
missing files, inconsistent templates, orphaned directories, etc.

Usage:
  python scripts/check_results.py --results-dir data/i4replicate/results
  python scripts/check_results.py --results-dir data/i4replicate/results --csv report.csv
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def check_results(results_dir: Path) -> list[dict]:
    """Run all consistency checks. Returns list of issue dicts."""
    issues = []
    papers_dir = results_dir.parent / "papers"

    # Collect all run directories
    all_runs = []  # (paper_id, run_name, run_dir)
    paper_ids = set()

    for pid in sorted(os.listdir(results_dir)):
        paper_dir = results_dir / pid
        if not paper_dir.is_dir() or pid == "summaries":
            continue
        paper_ids.add(pid)

        for run_name in sorted(os.listdir(paper_dir)):
            run_dir = paper_dir / run_name
            if not run_dir.is_dir():
                continue
            if "explainer" in run_name or run_name in ("summaries", "z-ai"):
                continue
            # Skip nested z-ai dirs (handled separately)
            if run_name == "z-ai":
                continue
            all_runs.append((pid, run_name, run_dir))

    # Parse approach/model from run_name
    def parse_run(run_name, pid):
        approach = run_name.rsplit("_", 1)[-1] if "_" in run_name else "?"
        model = run_name.split("_" + pid)[0] if pid in run_name else "?"
        return model, approach

    # =========================================================================
    # Check 1: Interrupted runs (workspace but no VR)
    # =========================================================================
    for pid, run_name, run_dir in all_runs:
        ws = run_dir / "workspace"
        vr = run_dir / "verification_report.json"
        if ws.is_dir() and not vr.is_file():
            n_py = sum(1 for f in ws.iterdir() if f.suffix == ".py")
            n_tbl = sum(1 for f in ws.iterdir() if f.name.startswith("table_") and f.suffix == ".json")
            issues.append({
                "paper_id": pid, "run_name": run_name,
                "issue": "interrupted_run",
                "details": f"workspace exists (py={n_py}, tables={n_tbl}) but no verification_report.json",
            })

    # =========================================================================
    # Check 2: Missing result.json
    # =========================================================================
    for pid, run_name, run_dir in all_runs:
        vr = run_dir / "verification_report.json"
        result = run_dir / "result.json"
        if vr.is_file() and not result.is_file():
            issues.append({
                "paper_id": pid, "run_name": run_name,
                "issue": "missing_result_json",
                "details": "has verification_report.json but no result.json",
            })

    # =========================================================================
    # Check 3: Missing usage files
    # =========================================================================
    for pid, run_name, run_dir in all_runs:
        vr = run_dir / "verification_report.json"
        if not vr.is_file():
            continue
        for fname in ["usage.json", "judge_usage.json"]:
            if not (run_dir / fname).is_file():
                issues.append({
                    "paper_id": pid, "run_name": run_name,
                    "issue": "missing_usage",
                    "details": f"missing {fname}",
                })

    # =========================================================================
    # Check 4: Empty workspaces (0 py files)
    # =========================================================================
    for pid, run_name, run_dir in all_runs:
        ws = run_dir / "workspace"
        if not ws.is_dir():
            continue
        n_py = sum(1 for f in ws.iterdir() if f.suffix == ".py")
        if n_py == 0:
            # Check if this is a figure-only paper
            summary_path = results_dir / pid / "summaries" / f"{pid}_summary.json"
            n_tables = 0
            if summary_path.is_file():
                s = _load_json(summary_path)
                if s:
                    n_tables = len(s.get("tables", []))
            if n_tables > 0:  # not figure-only
                issues.append({
                    "paper_id": pid, "run_name": run_name,
                    "issue": "empty_workspace",
                    "details": f"0 .py files (paper has {n_tables} tables)",
                })

    # =========================================================================
    # Check 5: Code but no tables
    # =========================================================================
    for pid, run_name, run_dir in all_runs:
        ws = run_dir / "workspace"
        if not ws.is_dir():
            continue
        n_py = sum(1 for f in ws.iterdir() if f.suffix == ".py")
        n_tbl = sum(1 for f in ws.iterdir() if f.name.startswith("table_") and f.suffix == ".json")
        if n_py > 0 and n_tbl == 0:
            # Check if figure-only
            summary_path = results_dir / pid / "summaries" / f"{pid}_summary.json"
            n_spec_tables = 0
            if summary_path.is_file():
                s = _load_json(summary_path)
                if s:
                    n_spec_tables = len(s.get("tables", []))
            if n_spec_tables > 0:
                issues.append({
                    "paper_id": pid, "run_name": run_name,
                    "issue": "code_no_tables",
                    "details": f"{n_py} .py files but 0 table JSONs (paper has {n_spec_tables} tables)",
                })

    # =========================================================================
    # Check 6: Orphaned z-ai/ dirs
    # =========================================================================
    for pid in paper_ids:
        zai_dir = results_dir / pid / "z-ai"
        if zai_dir.is_dir():
            contents = list(zai_dir.iterdir())
            issues.append({
                "paper_id": pid, "run_name": "z-ai/",
                "issue": "orphaned_z_ai",
                "details": f"nested z-ai/ directory with {len(contents)} entries",
            })

    # =========================================================================
    # Check 7: Missing table templates
    # =========================================================================
    for pid, run_name, run_dir in all_runs:
        ws = run_dir / "workspace"
        if not ws.is_dir():
            continue
        tmpl_dir = ws / "table_templates"
        n_templates = 0
        if tmpl_dir.is_dir():
            n_templates = sum(1 for f in tmpl_dir.iterdir() if f.suffix == ".json")

        # Compare with paper-level summary
        summary_path = results_dir / pid / "summaries" / f"{pid}_summary.json"
        n_expected = 0
        if summary_path.is_file():
            s = _load_json(summary_path)
            if s:
                n_expected = len(s.get("extracted_tables", []))

        if n_expected > 0 and n_templates == 0:
            issues.append({
                "paper_id": pid, "run_name": run_name,
                "issue": "missing_templates",
                "details": f"0 templates but paper summary has {n_expected} extracted_tables",
            })

    # =========================================================================
    # Check 8: Inconsistent template counts across approaches
    # =========================================================================
    templates_by_paper = defaultdict(dict)  # pid -> {run_name: count}
    for pid, run_name, run_dir in all_runs:
        ws = run_dir / "workspace"
        if not ws.is_dir():
            continue
        tmpl_dir = ws / "table_templates"
        n = 0
        if tmpl_dir.is_dir():
            n = sum(1 for f in tmpl_dir.iterdir() if f.suffix == ".json")
        templates_by_paper[pid][run_name] = n

    for pid, counts in templates_by_paper.items():
        nonzero = {k: v for k, v in counts.items() if v > 0}
        if len(set(nonzero.values())) > 1:
            detail_parts = [f"{parse_run(k, pid)[1]}({parse_run(k, pid)[0]})={v}"
                            for k, v in sorted(nonzero.items())]
            issues.append({
                "paper_id": pid, "run_name": "(multiple)",
                "issue": "inconsistent_templates",
                "details": f"different template counts: {', '.join(detail_parts)}",
            })

    # =========================================================================
    # Check 9: Table output shortfall
    # =========================================================================
    for pid, run_name, run_dir in all_runs:
        ws = run_dir / "workspace"
        if not ws.is_dir():
            continue
        tmpl_dir = ws / "table_templates"
        n_templates = 0
        if tmpl_dir.is_dir():
            n_templates = sum(1 for f in tmpl_dir.iterdir() if f.suffix == ".json")
        if n_templates == 0:
            continue  # no templates to compare against

        n_produced = sum(1 for f in ws.iterdir()
                         if f.name.startswith("table_") and f.suffix == ".json")
        if n_produced < n_templates:
            model, approach = parse_run(run_name, pid)
            issues.append({
                "paper_id": pid, "run_name": run_name,
                "issue": "table_shortfall",
                "details": f"{n_produced}/{n_templates} tables produced ({n_templates - n_produced} missing)",
            })

    # =========================================================================
    # Check 10: Missing summaries
    # =========================================================================
    for pid in paper_ids:
        summaries_dir = results_dir / pid / "summaries"
        summary_path = summaries_dir / f"{pid}_summary.json"
        results_path = summaries_dir / f"{pid}_results.json"
        if not summary_path.is_file():
            issues.append({
                "paper_id": pid, "run_name": "summaries/",
                "issue": "missing_summary",
                "details": f"no {pid}_summary.json",
            })
        if not results_path.is_file():
            issues.append({
                "paper_id": pid, "run_name": "summaries/",
                "issue": "missing_results_json",
                "details": f"no {pid}_results.json (extractor results)",
            })

    # =========================================================================
    # Check 11: Inconsistent paper coverage
    # =========================================================================
    approaches_per_paper = defaultdict(set)
    for pid, run_name, run_dir in all_runs:
        vr = run_dir / "verification_report.json"
        if vr.is_file():
            model, approach = parse_run(run_name, pid)
            approaches_per_paper[pid].add(f"{model}/{approach}")

    all_approaches = set()
    for approaches in approaches_per_paper.values():
        all_approaches |= approaches

    for pid in paper_ids:
        present = approaches_per_paper.get(pid, set())
        missing = all_approaches - present
        if missing and present:  # only flag if paper has some but not all
            issues.append({
                "paper_id": pid, "run_name": "(coverage)",
                "issue": "incomplete_coverage",
                "details": f"missing approaches: {', '.join(sorted(missing))}",
            })

    # =========================================================================
    # Check 12: Explainer coverage
    # =========================================================================
    for pid, run_name, run_dir in all_runs:
        vr_path = run_dir / "verification_report.json"
        explainer_path = run_dir / "explainer_report.json"
        if not vr_path.is_file():
            continue
        vr = _load_json(vr_path)
        if not vr:
            continue
        grade = vr.get("overall_grade", "F")
        if grade == "A":
            continue  # no explanation needed
        items = vr.get("item_verifications", [])
        non_a = [v for v in items if v.get("grade") != "A"
                 and not v.get("unverifiable") and not v.get("judge_error")]
        if non_a and not explainer_path.is_file():
            issues.append({
                "paper_id": pid, "run_name": run_name,
                "issue": "missing_explainer",
                "details": f"grade={grade}, {len(non_a)} non-A items but no explainer_report.json",
            })

    # =========================================================================
    # Check 13: Non-numerical tables
    # =========================================================================
    non_numerical_tables = []
    non_numerical_paper_set = set()
    for pid in paper_ids:
        rp = results_dir / pid / "summaries" / f"{pid}_results.json"
        if not rp.is_file():
            continue
        r = _load_json(rp)
        if not r:
            continue
        all_non_num = True
        has_any_cells = False
        for t in r.get("tables", []):
            cells = t.get("cells", [])
            if not cells:
                continue
            has_any_cells = True
            n_numeric = sum(1 for c in cells if c.get("numeric_value") is not None and not c.get("is_string"))
            if n_numeric == 0:
                non_numerical_tables.append((pid, t["table_id"][:50], len(cells)))
            else:
                all_non_num = False
        if all_non_num and has_any_cells:
            non_numerical_paper_set.add(pid)

    for pid, tid, n_cells in non_numerical_tables:
        issues.append({
            "paper_id": pid, "run_name": "(extractor)",
            "issue": "non_numerical_table",
            "details": f"{tid}: {n_cells} cells, all string/no numeric values",
        })
    for pid in non_numerical_paper_set:
        issues.append({
            "paper_id": pid, "run_name": "(paper)",
            "issue": "all_tables_non_numerical",
            "details": "all tables in this paper are non-numerical (text-only)",
        })

    # =========================================================================
    # Check 14: Replicator loaded methodology summary (information leak check)
    # =========================================================================
    import re as _re
    for pid, run_name, run_dir in all_runs:
        ws = run_dir / "workspace"
        if not ws.is_dir():
            continue
        flagged_files = []
        for py_file in sorted(ws.glob("*.py")):
            try:
                code = py_file.read_text(errors="replace")
            except Exception:
                continue
            # Check if the script references methodology_summary.json in any
            # loading context: direct open(), assigned to a variable then opened,
            # or read via Path. Exclude lines that only write *_summary.json.
            has_ref = _re.search(r"""methodology_summary\.json""", code)
            if not has_ref:
                continue
            # Exclude if the only reference is in a write context
            lines_with_ref = [
                line for line in code.splitlines()
                if "methodology_summary" in line
            ]
            is_write_only = all(
                _re.search(r"""open\s*\(.*["']w["']""", line) or
                _re.search(r"""write_text|\.write\(""", line)
                for line in lines_with_ref
            )
            if not is_write_only:
                flagged_files.append(py_file.name)
        if flagged_files:
            issues.append({
                "paper_id": pid, "run_name": run_name,
                "issue": "reads_methodology_summary",
                "details": f"{', '.join(flagged_files)} references methodology_summary.json (potential results leak)",
            })

    # =========================================================================
    # Check 15: Missing per-table Python scripts
    # =========================================================================
    for pid, run_name, run_dir in all_runs:
        ws = run_dir / "workspace"
        if not ws.is_dir():
            continue
        # Find table JSONs produced (excluding templates)
        table_jsons = sorted(
            f.stem for f in ws.iterdir()
            if f.name.startswith("table_") and f.suffix == ".json"
        )
        if not table_jsons:
            continue
        # Find Python scripts that look like per-table scripts
        py_files = {f.stem for f in ws.iterdir() if f.suffix == ".py"}
        # Count tables that have no matching .py file
        tables_without_script = [t for t in table_jsons if t not in py_files]
        if tables_without_script:
            # Check if there's a single "do-everything" script instead
            n_py = len(py_files)
            issues.append({
                "paper_id": pid, "run_name": run_name,
                "issue": "tables_without_script",
                "details": (
                    f"{len(tables_without_script)}/{len(table_jsons)} table JSONs "
                    f"have no matching .py file ({', '.join(tables_without_script[:5])})"
                    f"{' ...' if len(tables_without_script) > 5 else ''}"
                    f"; {n_py} total .py files in workspace"
                ),
            })

    # =========================================================================
    # Check 16: Grading method distribution (pre-aligned vs LLM-only)
    # =========================================================================
    grading_stats = {"pre_aligned": 0, "llm_only": 0, "no_items": 0}
    grading_by_approach = defaultdict(lambda: {"pre_aligned": 0, "llm_only": 0})
    llm_only_papers = defaultdict(int)  # paper_id -> count of LLM-only tables
    pre_aligned_papers = defaultdict(int)  # paper_id -> count of pre-aligned tables
    llm_only_details = []  # (pid, approach, model, item_id, grade)

    for pid, run_name, run_dir in all_runs:
        vr_path = run_dir / "verification_report.json"
        if not vr_path.is_file():
            continue
        vr = _load_json(vr_path)
        if not vr:
            continue
        items = vr.get("item_verifications", [])
        if not items:
            grading_stats["no_items"] += 1
            continue

        model, approach = parse_run(run_name, pid)
        key = f"{model}/{approach}"

        for v in items:
            # Skip figures and non-numerical tables from grading method stats
            item_type = v.get("item_type", "")
            if item_type == "figure":
                grading_stats.setdefault("skipped_figures", 0)
                grading_stats["skipped_figures"] += 1
                continue
            notes = v.get("comparison_notes", "")
            if "non-numerical" in notes.lower():
                grading_stats.setdefault("skipped_non_numerical", 0)
                grading_stats["skipped_non_numerical"] += 1
                continue

            tc = v.get("table_comparison")
            if tc and tc.get("cell_comparisons") and len(tc["cell_comparisons"]) > 0:
                grading_stats["pre_aligned"] += 1
                grading_by_approach[key]["pre_aligned"] += 1
                pre_aligned_papers[pid] += 1
            else:
                grading_stats["llm_only"] += 1
                grading_by_approach[key]["llm_only"] += 1
                llm_only_papers[pid] += 1
                llm_only_details.append((pid, approach, model, v.get("item_id", "?"), v.get("grade", "?")))

    # Report papers where ALL tables used LLM-only (zero pre-aligned)
    for pid, count in sorted(llm_only_papers.items(), key=lambda x: -x[1]):
        if pre_aligned_papers.get(pid, 0) == 0 and count >= 3:
            issues.append({
                "paper_id": pid, "run_name": "(grading)",
                "issue": "all_llm_only",
                "details": f"{count} table evaluations used LLM-only, 0 pre-aligned (possible table ID mismatch)",
            })

    return issues, grading_stats, grading_by_approach, llm_only_details


def print_summary(issues: list[dict], grading_stats=None, grading_by_approach=None,
                   llm_only_details=None):
    """Print a summary of issues to console."""
    by_type = Counter(i["issue"] for i in issues)
    print("=" * 60)
    print("RESULTS CONSISTENCY CHECK")
    print("=" * 60)
    print()
    print("Issues by type:")
    for issue_type, count in by_type.most_common():
        print(f"  {issue_type:<30s} {count:>5d}")
    print(f"  {'TOTAL':<30s} {len(issues):>5d}")

    # Per-approach breakdown for key issues
    print()
    print("Key issues by approach:")
    approach_issues = defaultdict(Counter)
    for i in issues:
        if i["issue"] in ("interrupted_run", "empty_workspace", "code_no_tables", "table_shortfall"):
            # Extract approach from run_name
            parts = i["run_name"].rsplit("_", 1)
            approach = parts[-1] if len(parts) > 1 else i["run_name"]
            approach_issues[approach][i["issue"]] += 1

    for approach in sorted(approach_issues):
        counts = approach_issues[approach]
        parts = [f"{k}={v}" for k, v in counts.most_common()]
        print(f"  {approach:<15s} {', '.join(parts)}")

    # Grading method distribution (numerical tables only — figures and non-numerical excluded)
    if grading_stats:
        total_graded = grading_stats["pre_aligned"] + grading_stats["llm_only"]
        skipped_fig = grading_stats.get("skipped_figures", 0)
        skipped_nn = grading_stats.get("skipped_non_numerical", 0)
        print()
        print("Grading method distribution (numerical tables only):")
        if total_graded:
            print(f"  Pre-aligned (deterministic): {grading_stats['pre_aligned']:>5d} ({grading_stats['pre_aligned']/total_graded*100:.0f}%)")
            print(f"  LLM-only (fallback):         {grading_stats['llm_only']:>5d} ({grading_stats['llm_only']/total_graded*100:.0f}%)")
        print(f"  Skipped figures:             {skipped_fig:>5d}")
        print(f"  Skipped non-numerical:       {skipped_nn:>5d}")
        print(f"  Runs with no items:          {grading_stats['no_items']:>5d}")

    if grading_by_approach:
        print()
        print("Grading method by approach (numerical tables only):")
        print(f"  {'Approach':<35s} {'Pre-aligned':>12s} {'LLM-only':>10s} {'% Pre-aligned':>14s}")
        print(f"  {'-'*71}")
        for key in sorted(grading_by_approach, key=lambda k: -grading_by_approach[k]["pre_aligned"]):
            s = grading_by_approach[key]
            total = s["pre_aligned"] + s["llm_only"]
            pct = s["pre_aligned"] / total * 100 if total else 0
            print(f"  {key:<35s} {s['pre_aligned']:>12d} {s['llm_only']:>10d} {pct:>13.0f}%")

    # LLM-only table details
    if llm_only_details:
        print()
        print(f"LLM-only numerical tables ({len(llm_only_details)} items):")
        print(f"  {'paper_id':<40s} {'approach':<14s} {'model':<26s} {'item_id':<20s} {'grade'}")
        print(f"  {'-'*110}")
        for pid, approach, model, item_id, grade in sorted(llm_only_details):
            print(f"  {pid:<40s} {approach:<14s} {model:<26s} {item_id:<20s} {grade}")


def write_csv(issues: list[dict], path: str):
    """Write issues to CSV."""
    if not issues:
        print("No issues found.")
        return
    fieldnames = ["paper_id", "run_name", "issue", "details"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in sorted(issues, key=lambda x: (x["issue"], x["paper_id"])):
            writer.writerow(i)
    print(f"\nWrote {len(issues)} issues to {path}")


def main():
    parser = argparse.ArgumentParser(description="Check benchmark results consistency")
    parser.add_argument("--results-dir", required=True, help="Results directory")
    parser.add_argument("--csv", default=None, help="Write detailed report to CSV")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)

    issues, grading_stats, grading_by_approach, llm_only_details = check_results(results_dir)
    print_summary(issues, grading_stats, grading_by_approach, llm_only_details)

    if args.csv:
        write_csv(issues, args.csv)
    else:
        # Default CSV path
        csv_path = results_dir.parent / "check_results_report.csv"
        write_csv(issues, str(csv_path))


if __name__ == "__main__":
    main()
