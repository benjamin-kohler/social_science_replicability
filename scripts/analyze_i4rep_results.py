#!/usr/bin/env python3
"""Analyze i4rep benchmark results and generate publication-quality plots.

Usage:
    python scripts/analyze_i4rep_results.py
    python scripts/analyze_i4rep_results.py --results-dir path/to/results --output-dir plots/
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ============================================================================
# Constants
# ============================================================================

GRADE_ORDER = ["A", "B", "C", "D", "E", "F"]
GRADE_TO_NUM = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1, "F": 0}
NUM_TO_GRADE = {v: k for k, v in GRADE_TO_NUM.items()}

APPROACH_ORDER_RAW = ["claude-code", "codex", "swe-agent", "opencode"]

# Approach × model combo keys (used as the primary grouping throughout).
APPROACH_MODEL_ORDER = [
    "claude-code/claude-opus-4-6",
    "codex/gpt-5.3-codex",
    "codex/gpt-5.4",
    "swe-agent/gpt-5.4",
    "swe-agent/z-ai_glm-5",
    "opencode/gpt-5.4",
    "opencode/z-ai_glm-5",
]

APPROACH_MODEL_LABELS = {
    "claude-code/claude-opus-4-6": "Claude Code\nOpus 4.6",
    "codex/gpt-5.4": "Codex CLI\nGPT-5.4",
    "codex/gpt-5.3-codex": "Codex CLI\nGPT-5.3",
    "swe-agent/gpt-5.4": "SWE-Agent\nGPT-5.4",
    "swe-agent/gpt-5.2-codex": "SWE-Agent\nGPT-5.2",
    "swe-agent/z-ai_glm-5": "SWE-Agent\nGLM-5",
    "opencode/gpt-5.4": "OpenCode\nGPT-5.4",
    "opencode/gpt-5.2-codex": "OpenCode\nGPT-5.2",
    "opencode/z-ai_glm-5": "OpenCode\nGLM-5",
}

APPROACH_MODEL_COLORS = {
    "claude-code/claude-opus-4-6": "#E07B39",
    "codex/gpt-5.4": "#10A37F",
    "codex/gpt-5.3-codex": "#0D8A6A",
    "swe-agent/gpt-5.4": "#6C5CE7",
    "swe-agent/gpt-5.2-codex": "#8E7CF7",
    "swe-agent/z-ai_glm-5": "#A29BFE",
    "opencode/gpt-5.4": "#0984E3",
    "opencode/gpt-5.2-codex": "#3AA0F0",
    "opencode/z-ai_glm-5": "#74B9FF",
}

# Aliases used by plot functions (set to approach_model variants)
APPROACH_ORDER = APPROACH_MODEL_ORDER
APPROACH_LABELS = APPROACH_MODEL_LABELS
APPROACH_COLORS = APPROACH_MODEL_COLORS

GRADE_COLORS = {
    "A": "#27ae60",
    "B": "#2ecc71",
    "C": "#f1c40f",
    "D": "#e67e22",
    "E": "#e74c3c",
    "F": "#7f8c8d",
}

TEXTLAB_BASE = Path("/data/individual/benjamin/social_science_replicability/data/i4replicate")
LOCAL_BASE = Path("data/i4replicate")

# Journal mapping from DOI prefixes
JOURNAL_MAP = {
    "10.1017_s00030554": "APSR",
    "10.1086_71": "JOP",
    "10.1093_ej_": "EJ",
    "10.1093_restud_": "REStud",
    "10.1093_qje_": "QJE",
    "10.1111_ajps": "AJPS",
    "10.1257_aer": "AER",
    "10.1257_aeri": "AER:I",
    "10.1257_app": "AEJ:AP",
    "10.1257_pol": "AEJ:Pol",
    "10.2139_ssrn": "SSRN",
    "10.1163_": "Other",
}

# Journal → Discipline mapping
JOURNAL_DISCIPLINE = {
    "AER": "Economics", "AER:I": "Economics",
    "AEJ:AP": "Economics", "AEJ:Pol": "Economics", "AEJ:Mac": "Economics",
    "QJE": "Economics", "JPE": "Economics", "EJ": "Economics",
    "REStud": "Economics", "Econometrica": "Economics",
    "AJPS": "Political Science", "APSR": "Political Science",
    "JOP": "Political Science",
    "SSRN": "Other", "Other": "Other",
}

# Fallback titles for papers missing metadata.json
FALLBACK_TITLES = {
    "10.1093_ej_ueab096": "Hobo Economicus: The Causes and Effects of Homelessness",
}


# ============================================================================
# Style
# ============================================================================

def setup_style():
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
    })


def apply_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=14)


def save_figure(fig, output_dir: Path, name: str, subdir: str = ""):
    target = output_dir / subdir if subdir else output_dir
    target.mkdir(parents=True, exist_ok=True)
    fig.savefig(target / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(target / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    prefix = f"{subdir}/" if subdir else ""
    print(f"  Saved {prefix}{name}")


# ============================================================================
# Helpers
# ============================================================================

def _load_json(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _parse_item_number(item_id: str) -> int | None:
    m = re.search(r"(\d+)", item_id)
    return int(m.group(1)) if m else None


def _parse_item_type(item_id: str, item_type: str | None) -> str:
    if item_type and item_type.lower() in ("table", "figure"):
        return item_type.lower()
    low = item_id.lower()
    if "table" in low:
        return "table"
    if "fig" in low:
        return "figure"
    return "other"


def _infer_journal(paper_slug: str) -> str:
    for prefix, journal in JOURNAL_MAP.items():
        if paper_slug.startswith(prefix):
            return journal
    return "Other"


def _count_code_in_workspace(workspace_dir: Path) -> tuple[int, int]:
    n_files = 0
    total_chars = 0
    if not workspace_dir.exists():
        return 0, 0
    for ext in ("*.py", "*.R", "*.r", "*.do", "*.m"):
        for f in workspace_dir.rglob(ext):
            n_files += 1
            try:
                total_chars += f.stat().st_size
            except OSError:
                pass
    return n_files, total_chars


CODE_EXT_TO_LANG = {
    ".do": "Stata", ".ado": "Stata", ".dct": "Stata",
    ".R": "R", ".r": "R", ".Rmd": "R", ".rmd": "R", ".Rnw": "R",
    ".py": "Python",
    ".m": "Matlab",
    ".jl": "Julia",
    ".sas": "SAS",
    ".sps": "SPSS",
}


def _classify_original_language(paper_dir: Path) -> tuple[str, list[str]]:
    """Return (primary_language, sorted list of all languages) for a replication package."""
    lang_counts: dict[str, int] = {}
    if not paper_dir.exists():
        return "Unknown", []
    for f in paper_dir.rglob("*"):
        if not f.is_file():
            continue
        lang = CODE_EXT_TO_LANG.get(f.suffix)
        if lang:
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
    if not lang_counts:
        return "Unknown", []
    primary = max(lang_counts, key=lang_counts.get)
    all_langs = sorted(lang_counts.keys(), key=lambda l: -lang_counts[l])
    return primary, all_langs


def _dataset_stats(data_dir: Path) -> tuple[int, int]:
    n_files = 0
    total_bytes = 0
    if not data_dir.exists():
        return 0, 0
    for f in data_dir.rglob("*"):
        if f.is_file():
            n_files += 1
            try:
                total_bytes += f.stat().st_size
            except OSError:
                pass
    return n_files, total_bytes


def _query_opencode_db_for_workspace(workspace_dir: str) -> dict | None:
    import sqlite3
    db_path = Path.home() / ".local" / "share" / "opencode" / "opencode.db"
    if not db_path.exists():
        return None
    try:
        db = sqlite3.connect(str(db_path))
        cur = db.cursor()
        cur.execute("SELECT id FROM session WHERE directory = ?", (workspace_dir,))
        row = cur.fetchone()
        if not row:
            db.close()
            return None
        session_id = row[0]
        cur.execute(
            "SELECT data FROM message WHERE session_id = ? "
            "AND json_extract(data, '$.role') = 'assistant'",
            (session_id,),
        )
        total_input = total_output = 0
        total_cost = 0.0
        for (data_str,) in cur.fetchall():
            d = json.loads(data_str)
            tokens = d.get("tokens", {})
            total_input += tokens.get("input", 0)
            total_output += tokens.get("output", 0)
            total_cost += d.get("cost", 0)
        db.close()
        if total_input == 0 and total_output == 0:
            return None
        return {
            "prompt_tokens": total_input,
            "completion_tokens": total_output,
            "total_tokens": total_input + total_output,
            "total_cost_usd": total_cost,
        }
    except Exception:
        return None


def _parse_approach_from_dirname(dirname: str, paper_slug: str) -> tuple[str, str] | None:
    idx = dirname.find(f"_{paper_slug}_")
    if idx == -1:
        return None
    model = dirname[:idx]
    approach = dirname[idx + len(f"_{paper_slug}_"):]
    if approach not in APPROACH_ORDER_RAW:
        return None
    return model, approach


# ============================================================================
# Data Loading
# ============================================================================

def _load_extracted_table_row_types(run_dir: Path) -> dict[tuple[str, str, str], str]:
    lookup = {}
    for ms_path in [
        run_dir / "explainer_workspace" / "methodology_summary.json",
        run_dir / "workspace" / "methodology_summary.json",
    ]:
        if not ms_path.exists():
            continue
        ms = _load_json(ms_path)
        if not ms:
            continue
        for table in ms.get("extracted_tables", []):
            table_id = table.get("table_id", "")
            for cell in table.get("cells", []):
                key = (table_id, cell.get("row_label", ""), cell.get("column_label", ""))
                lookup[key] = cell.get("row_type", "")
        break
    return lookup


def _load_replicator_se_values(run_dir: Path) -> dict[tuple[str, str, str], float | None]:
    """Build lookup (table_id, coeff_row_label, column_label) -> replicated SE value."""
    se_lookup = {}
    for base in [run_dir / "explainer_workspace" / "replicator_outputs",
                 run_dir / "workspace"]:
        if not base.exists():
            continue
        for table_json in sorted(base.glob("table_*.json")):
            data = _load_json(table_json)
            if not data or "cells" not in data:
                continue
            table_id = data.get("table_id", "")
            cells = data["cells"]
            if cells and isinstance(cells[0], list):
                cells = [c for row in cells for c in row if isinstance(c, dict)]
            by_col: dict[str, list[dict]] = {}
            for c in cells:
                if not isinstance(c, dict):
                    continue
                col = c.get("column_label", "")
                by_col.setdefault(col, []).append(c)
            for col, col_cells in by_col.items():
                for i, c in enumerate(col_cells):
                    if c.get("row_type") == "coefficient" and i + 1 < len(col_cells):
                        next_c = col_cells[i + 1]
                        if next_c.get("row_type") == "se" or next_c.get("is_standard_error"):
                            se_val = next_c.get("numeric_value")
                            coeff_row_label = c.get("row_label", "")
                            se_lookup[(table_id, coeff_row_label, col)] = se_val
        break
    return se_lookup


def _load_original_se_values(run_dir: Path) -> dict[tuple[str, str, str], float | None]:
    """Build lookup (table_id, coeff_row_label, column_label) -> original SE value."""
    se_lookup = {}
    for ms_path in [
        run_dir / "explainer_workspace" / "methodology_summary.json",
        run_dir / "workspace" / "methodology_summary.json",
    ]:
        if not ms_path.exists():
            continue
        ms = _load_json(ms_path)
        if not ms:
            continue
        for table in ms.get("extracted_tables", []):
            table_id = table.get("table_id", "")
            cells = table.get("cells", [])
            if cells and isinstance(cells[0], list):
                cells = [c for row in cells for c in row if isinstance(c, dict)]
            by_col: dict[str, list[dict]] = {}
            for c in cells:
                if not isinstance(c, dict):
                    continue
                col = c.get("column_label", "")
                by_col.setdefault(col, []).append(c)
            for col, col_cells in by_col.items():
                for i, c in enumerate(col_cells):
                    if c.get("row_type") == "coefficient" and i + 1 < len(col_cells):
                        next_c = col_cells[i + 1]
                        if next_c.get("row_type") == "se" or next_c.get("is_standard_error"):
                            se_val = next_c.get("numeric_value")
                            coeff_row_label = c.get("row_label", "")
                            se_lookup[(table_id, coeff_row_label, col)] = se_val
        break
    return se_lookup


def _load_original_significance(results_dir: Path, paper_slug: str) -> dict[tuple[str, str, str], int]:
    lookup = {}
    summaries_dir = results_dir / paper_slug / "summaries"
    results_json = summaries_dir / f"{paper_slug}_results.json"
    if not results_json.exists():
        return lookup
    data = _load_json(results_json)
    if not data:
        return lookup
    for table in data.get("tables", []):
        table_id = table.get("table_id", "")
        for cell in table.get("cells", []):
            if cell.get("row_type") != "coefficient":
                continue
            key = (table_id, cell.get("row_label", ""), cell.get("column_label", ""))
            stars = cell.get("significance_stars", 0)
            lookup[key] = int(stars) if stars is not None else 0
    return lookup


def _load_replicator_significance(run_dir: Path) -> dict[tuple[str, str, str], int]:
    lookup = {}
    for base in [run_dir / "explainer_workspace" / "replicator_outputs",
                 run_dir / "workspace"]:
        if not base.exists():
            continue
        for table_json in sorted(base.glob("table_*.json")):
            data = _load_json(table_json)
            if not data or "cells" not in data:
                continue
            table_id = data.get("table_id", "")
            cells = data["cells"]
            if cells and isinstance(cells[0], list):
                cells = [c for row in cells for c in row if isinstance(c, dict)]
            for c in cells:
                if not isinstance(c, dict):
                    continue
                if c.get("row_type") != "coefficient":
                    continue
                key = (table_id, c.get("row_label", ""), c.get("column_label", ""))
                stars = c.get("significance_stars", 0)
                lookup[key] = int(stars) if stars is not None else 0
        break
    return lookup


def load_results(results_dir: Path, papers_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all benchmark results into run-level, item-level, and cell-level DataFrames."""
    run_rows = []
    item_rows = []
    cell_rows = []

    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        sys.exit(1)

    paper_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()])
    print(f"Loading results from {len(paper_dirs)} paper directories...")

    for paper_dir in paper_dirs:
        paper_slug = paper_dir.name

        for run_dir in sorted(paper_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            if run_dir.name in ("judge_results", "summaries"):
                continue

            parsed = _parse_approach_from_dirname(run_dir.name, paper_slug)
            if parsed is None:
                continue
            model, approach = parsed

            # Load verification report (required)
            vr = _load_json(run_dir / "verification_report.json")
            if vr is None:
                continue

            # Load optional files
            er = _load_json(run_dir / "explanation_report.json")
            # Prefer explainer_workspace copy (has primary_fault); fall back to top-level
            explainer = _load_json(run_dir / "explainer_workspace" / "explainer_report.json")
            if explainer is None:
                explainer = _load_json(run_dir / "explainer_report.json")
            result = _load_json(run_dir / "result.json")
            usage = _load_json(run_dir / "usage.json")
            workspace = run_dir / "workspace"
            meth_summary = _load_json(workspace / "methodology_summary.json")

            # Load row_type, SE, and significance lookups for cell enrichment
            row_type_lookup = _load_extracted_table_row_types(run_dir)
            replicator_se_lookup = _load_replicator_se_values(run_dir)
            original_se_lookup = _load_original_se_values(run_dir)
            orig_sig_lookup = _load_original_significance(results_dir, paper_slug)
            repl_sig_lookup = _load_replicator_significance(run_dir)

            # Build explanation lookup (from explanation_report)
            explanation_map = {}
            if er and "analyses" in er:
                for analysis in er["analyses"]:
                    explanation_map[analysis.get("item_id", "")] = analysis

            # Build explainer lookup (from explainer_report — has primary_fault)
            explainer_map = {}
            if explainer and "analyses" in explainer:
                for analysis in explainer["analyses"]:
                    explainer_map[analysis.get("item_id", "")] = analysis

            # Parse items
            items = vr.get("item_verifications", [])
            grade_counts = Counter()
            n_tables = 0
            n_figures = 0
            n_judge_errors = 0
            n_unverifiable = 0

            for item in items:
                item_id = item.get("item_id", "")
                item_type = _parse_item_type(item_id, item.get("item_type"))
                item_number = _parse_item_number(item_id)
                grade = item.get("grade", "F")
                judge_error = item.get("judge_error", False)
                unverifiable = item.get("unverifiable", False)
                comparison_notes = item.get("comparison_notes", "")
                non_numerical = "non-numerical" in comparison_notes.lower()

                grade_counts[grade] += 1
                if item_type == "table":
                    n_tables += 1
                elif item_type == "figure":
                    n_figures += 1
                if judge_error:
                    n_judge_errors += 1
                if unverifiable:
                    n_unverifiable += 1

                # Table comparison stats
                tc = item.get("table_comparison")
                n_cells = 0
                mean_pct = np.nan
                if tc and "cell_comparisons" in tc:
                    cells = tc["cell_comparisons"]
                    n_cells = len(cells)
                    pct_diffs = [c.get("percent_difference") for c in cells
                                 if c.get("percent_difference") is not None]
                    if pct_diffs:
                        mean_pct = np.mean(pct_diffs)

                    # Extract cell-level rows
                    for cell in cells:
                        ov = cell.get("original_value")
                        rv = cell.get("replicated_value")
                        is_numeric = (ov is not None) or (cell.get("percent_difference") is not None)
                        if not is_numeric and rv is not None:
                            try:
                                float(rv)
                                is_numeric = True
                            except (TypeError, ValueError):
                                pass

                        # Enrich with row_type from extracted tables
                        cell_row_label = cell.get("row_label", "")
                        cell_col_label = cell.get("column_label", "")
                        rt_key = (item_id, cell_row_label, cell_col_label)
                        row_type = row_type_lookup.get(rt_key, "")

                        # SE values and significance for coefficient cells
                        orig_se = None
                        repl_se = None
                        sig_stars_orig = None
                        sig_stars_repl = None
                        if row_type == "coefficient":
                            se_key = (item_id, cell_row_label, cell_col_label)
                            orig_se = original_se_lookup.get(se_key)
                            repl_se = replicator_se_lookup.get(se_key)
                            sig_stars_orig = orig_sig_lookup.get(se_key)
                            sig_stars_repl = repl_sig_lookup.get(se_key)

                        cell_rows.append({
                            "paper_slug": paper_slug,
                            "approach": approach,
                            "model": model,
                            "item_id": item_id,
                            "item_grade": grade,
                            "row_label": cell_row_label,
                            "column_label": cell_col_label,
                            "original_value": ov,
                            "replicated_value": rv,
                            "percent_difference": cell.get("percent_difference"),
                            "absolute_difference": cell.get("absolute_difference"),
                            "sign_match": cell.get("sign_match"),
                            "cell_grade": cell.get("grade", ""),
                            "is_numeric": is_numeric,
                            "row_type": row_type,
                            "original_se": orig_se,
                            "replicated_se": repl_se,
                            "significance_stars_orig": sig_stars_orig,
                            "significance_stars_repl": sig_stars_repl,
                        })

                # Explanation data (from both reports)
                expl = explanation_map.get(item_id, {})
                expl2 = explainer_map.get(item_id, {})

                item_rows.append({
                    "paper_slug": paper_slug,
                    "approach": approach,
                    "model": model,
                    "item_id": item_id,
                    "item_type": item_type,
                    "item_number": item_number,
                    "grade": grade,
                    "grade_num": GRADE_TO_NUM.get(grade, np.nan),
                    "judge_error": judge_error,
                    "unverifiable": unverifiable,
                    "non_numerical": non_numerical,
                    "n_cells_compared": n_cells,
                    "mean_pct_diff": mean_pct,
                    "primary_fault": expl2.get("primary_fault", ""),
                    "additional_faults": expl2.get("additional_faults", []),
                    "fault_explanation": expl2.get("fault_explanation", ""),
                    "confidence": expl2.get("confidence", expl.get("confidence")),
                    "likely_causes": expl.get("likely_causes", []),
                    "description_of_discrepancy": expl.get("description_of_discrepancy", ""),
                })

            # Run-level fields
            overall_grade = vr.get("overall_grade", "F")

            # Get duration from run_log.txt (authoritative source)
            duration = None
            for log_path in [workspace / "run_log.txt",
                             run_dir / "explainer_workspace" / "run_log.txt"]:
                if log_path.exists():
                    m = re.search(r"^Duration:\s*([\d.]+)s", log_path.read_text(), re.MULTILINE)
                    if m:
                        duration = float(m.group(1))
                        break
            # Fall back to result.json if no run_log.txt
            if duration is None and result:
                duration = result.get("duration_seconds") or None

            n_code_files, total_code_chars = _count_code_in_workspace(workspace)

            meth_len = 0
            n_tables_summary = 0
            n_figures_summary = 0
            n_table_templates = 0
            if meth_summary:
                meth_len = len(json.dumps(meth_summary))
                n_tables_summary = len(meth_summary.get("tables", []))
                n_figures_summary = len(meth_summary.get("figures", []))
                n_table_templates = len(meth_summary.get("extracted_tables", []))

            if n_table_templates == 0:
                ew_ms = _load_json(run_dir / "explainer_workspace" / "methodology_summary.json")
                if ew_ms:
                    n_table_templates = len(ew_ms.get("extracted_tables", []))

            # Count table_*.json files produced by replicator
            n_table_jsons = 0
            for base in [run_dir / "explainer_workspace" / "replicator_outputs",
                         run_dir / "workspace"]:
                if base.exists():
                    n_table_jsons = len(list(base.glob("table_*.json")))
                    break

            # Count .py files produced
            n_py_files = 0
            for base in [run_dir / "explainer_workspace",
                         run_dir / "workspace"]:
                if base.exists():
                    n_py_files = len(list(base.rglob("*.py")))
                    break

            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            total_cost_usd = 0.0
            if usage:
                prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
                completion_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
                total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens) or 0
                total_cost_usd = usage.get("total_cost_usd", usage.get("cost_usd", 0)) or 0

            # Backfill tokens from SWE-Agent trajectory files
            if total_tokens == 0 and approach == "swe-agent":
                traj_path = workspace / "trajectory.json"
                if traj_path.exists():
                    traj = _load_json(traj_path)
                    if traj:
                        ms = (traj.get("info") or {}).get("model_stats", {})
                        prompt_tokens = ms.get("tokens_sent", 0)
                        completion_tokens = ms.get("tokens_received", 0)
                        total_tokens = prompt_tokens + completion_tokens
                        total_cost_usd = ms.get("instance_cost", 0)

            # Backfill tokens from OpenCode SQLite DB
            if total_tokens == 0 and approach == "opencode":
                oc_usage = _query_opencode_db_for_workspace(str(workspace))
                if oc_usage:
                    prompt_tokens = oc_usage["prompt_tokens"]
                    completion_tokens = oc_usage["completion_tokens"]
                    total_tokens = oc_usage["total_tokens"]
                    total_cost_usd = oc_usage.get("total_cost_usd", 0)

            # Paper-level data stats
            n_datasets = 0
            total_data_bytes = 0
            paper_title = FALLBACK_TITLES.get(paper_slug, paper_slug)
            journal = _infer_journal(paper_slug)
            original_language = "Unknown"
            original_languages_all = []
            if papers_dir:
                paper_data_dir = papers_dir / paper_slug / "data"
                n_datasets, total_data_bytes = _dataset_stats(paper_data_dir)
                meta = _load_json(papers_dir / paper_slug / "metadata.json")
                if meta and "title" in meta:
                    paper_title = meta["title"]
                original_language, original_languages_all = _classify_original_language(papers_dir / paper_slug)

            run_rows.append({
                "paper_slug": paper_slug,
                "paper_title": paper_title,
                "journal": journal,
                "original_language": original_language,
                "original_languages_all": "+".join(original_languages_all) if original_languages_all else "Unknown",
                "approach": approach,
                "model": model,
                "provider": "anthropic" if "claude" in model else "openai",
                "overall_grade": overall_grade,
                "overall_grade_num": GRADE_TO_NUM.get(overall_grade, np.nan),
                "n_items": len(items),
                "n_tables": n_tables,
                "n_figures": n_figures,
                **{f"n_grade_{g}": grade_counts.get(g, 0) for g in GRADE_ORDER},
                "duration_seconds": duration,
                "methodology_summary_len": meth_len,
                "n_tables_in_summary": n_tables_summary,
                "n_figures_in_summary": n_figures_summary,
                "n_table_templates": n_table_templates,
                "n_table_jsons": n_table_jsons,
                "n_py_files": n_py_files,
                "n_code_files": n_code_files,
                "total_code_chars": total_code_chars,
                "n_datasets": n_datasets,
                "total_data_size_bytes": total_data_bytes,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "total_cost_usd": total_cost_usd,
                "n_judge_errors": n_judge_errors,
                "n_unverifiable": n_unverifiable,
            })

    df_runs = pd.DataFrame(run_rows)
    df_items = pd.DataFrame(item_rows)
    df_cells = pd.DataFrame(cell_rows)

    # Add discipline column to runs
    if not df_runs.empty:
        df_runs["discipline"] = df_runs["journal"].map(
            lambda j: JOURNAL_DISCIPLINE.get(j, "Other")
        )

    # Create composite approach/model key for all DataFrames
    for df in [df_runs, df_items, df_cells]:
        if not df.empty:
            df["approach_raw"] = df["approach"]
            df["approach"] = df["approach"].astype(str) + "/" + df["model"].astype(str)

    # Build ordered category list from data (keeps only combos that exist)
    combos_present = []
    if not df_runs.empty:
        for combo in APPROACH_MODEL_ORDER:
            if combo in df_runs["approach"].values:
                combos_present.append(combo)
        for combo in df_runs["approach"].unique():
            if combo not in combos_present:
                combos_present.append(combo)

    # Enforce categorical ordering
    if not df_runs.empty:
        df_runs["approach"] = pd.Categorical(df_runs["approach"], categories=combos_present, ordered=True)
        df_runs["overall_grade"] = pd.Categorical(df_runs["overall_grade"], categories=GRADE_ORDER, ordered=True)
    if not df_items.empty:
        df_items["approach"] = pd.Categorical(df_items["approach"], categories=combos_present, ordered=True)
        df_items["grade"] = pd.Categorical(df_items["grade"], categories=GRADE_ORDER, ordered=True)
    if not df_cells.empty:
        df_cells["approach"] = pd.Categorical(df_cells["approach"], categories=combos_present, ordered=True)
        df_cells["cell_grade"] = pd.Categorical(df_cells["cell_grade"], categories=GRADE_ORDER, ordered=True)

    print(f"Loaded {len(df_runs)} runs, {len(df_items)} items, {len(df_cells)} cells")
    return df_runs, df_items, df_cells


# ============================================================================
# Plot functions (helper to get approach list from data)
# ============================================================================

def _approaches_in(df, col="approach"):
    """Return approaches present in data, ordered by APPROACH_ORDER."""
    present = df[col].unique() if not df.empty else []
    return ([a for a in APPROACH_ORDER if a in present]
            + [a for a in present if a not in APPROACH_ORDER])


# ============================================================================
# Section: Setup & Descriptives
# ============================================================================

def plot_extractor_row_type_distribution(df_cells: pd.DataFrame, output_dir: Path, subdir: str = ""):
    """Distribution of cell row_type values (single panel, approach-independent counts)."""
    df = df_cells[df_cells["row_type"].notna() & (df_cells["row_type"] != "")].copy()
    if df.empty:
        print("  Skipping extractor_row_type_distribution: no row_type data")
        return
    df_dedup = df.drop_duplicates(subset=["paper_slug", "item_id", "row_label", "column_label"])
    type_counts = df_dedup["row_type"].value_counts()
    type_order = type_counts.index.tolist()

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(type_order))
    ax.bar(x, type_counts.values, color="#3498db", edgecolor="white", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(type_order, rotation=45, ha="right", fontsize=11)
    ax.set_ylabel("Number of cells", fontsize=16, fontweight="bold")
    for i, v in enumerate(type_counts.values):
        ax.text(i, v + max(type_counts.values) * 0.01, str(v), ha="center", fontsize=11, fontweight="bold")
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig, output_dir, "extractor_row_type_distribution", subdir)


def plot_first_fail_distribution(df_items: pd.DataFrame, output_dir: Path, subdir: str = ""):
    """Bar chart: number of papers that completely failed (all items F) by approach."""
    paper_approach = df_items.groupby(["paper_slug", "approach"], observed=True).agg(
        n_items=("grade", "size"),
        n_f=("grade", lambda x: (x == "F").sum()),
    ).reset_index()
    paper_approach["all_f"] = paper_approach["n_items"] == paper_approach["n_f"]
    approaches = _approaches_in(paper_approach)

    fig, ax = plt.subplots(figsize=(10, 6))
    xlabels, counts_all_f, counts_total, colors = [], [], [], []
    for a in approaches:
        sub = paper_approach[paper_approach["approach"] == a]
        if sub.empty:
            continue
        xlabels.append(APPROACH_LABELS.get(a, a).replace("\n", " "))
        counts_all_f.append(int(sub["all_f"].sum()))
        counts_total.append(len(sub))
        colors.append(APPROACH_COLORS.get(a, "#95a5a6"))

    x = np.arange(len(xlabels))
    width = 0.35
    ax.bar(x - width / 2, counts_total, width, label="Total papers", color=colors, alpha=0.3, edgecolor="white")
    ax.bar(x + width / 2, counts_all_f, width, label="Completely failed (all F)", color=colors, alpha=0.8, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=10, rotation=25, ha="right")
    ax.set_ylabel("Number of papers", fontsize=16, fontweight="bold")
    for i, (nf, nt) in enumerate(zip(counts_all_f, counts_total)):
        pct = nf / nt * 100 if nt > 0 else 0
        ax.text(i + width / 2, nf + 0.3, f"{nf} ({pct:.0f}%)", ha="center", fontsize=10, fontweight="bold")
    ax.legend(fontsize=12)
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig, output_dir, "first_fail_distribution", subdir)


def plot_extractor_cells(df_cells: pd.DataFrame, output_dir: Path, subdir: str = ""):
    """Extractor output: numeric cells per table distribution + replicator fill rate."""
    if df_cells.empty:
        return
    df_num = df_cells[df_cells["is_numeric"]].copy()
    if df_num.empty:
        return

    table_stats = df_num.groupby(["paper_slug", "approach", "item_id"], observed=True).apply(
        lambda g: pd.Series({
            "n_extractor": g["original_value"].notna().sum(),
            "n_replicator": g["replicated_value"].notna().sum(),
        })
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    extractor_vals = table_stats["n_extractor"].values
    extractor_vals = extractor_vals[extractor_vals > 0]
    ax.hist(extractor_vals, bins=30, color="#3498db", edgecolor="white", alpha=0.8)
    ax.axvline(np.median(extractor_vals), color="black", linestyle="--", linewidth=2,
               label=f"Median: {np.median(extractor_vals):.0f}")
    ax.set_xlabel("Numeric cells per table (from extractor)", fontsize=16, fontweight="bold")
    ax.set_ylabel("Number of tables", fontsize=16, fontweight="bold")
    ax.legend(fontsize=14)
    apply_style(ax)

    ax2 = axes[1]
    approaches = _approaches_in(table_stats)
    x = np.arange(len(approaches))
    width = 0.35
    ext_totals = [table_stats.loc[table_stats["approach"] == a, "n_extractor"].sum() for a in approaches]
    rep_totals = [table_stats.loc[table_stats["approach"] == a, "n_replicator"].sum() for a in approaches]
    ax2.bar(x - width / 2, ext_totals, width, label="Extractor (original)", color="#3498db", alpha=0.8, edgecolor="white")
    ax2.bar(x + width / 2, rep_totals, width, label="Replicator (filled)", color="#e67e22", alpha=0.8, edgecolor="white")
    for i, (e, r) in enumerate(zip(ext_totals, rep_totals)):
        pct = r / e * 100 if e > 0 else 0
        ax2.annotate(f"{pct:.0f}%", xy=(x[i] + width / 2, r), xytext=(0, 5),
                     textcoords="offset points", ha="center", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([APPROACH_LABELS.get(a, a).replace("\n", " ") for a in approaches], fontsize=9, rotation=25, ha="right")
    ax2.set_ylabel("Total cells", fontsize=16, fontweight="bold")
    ax2.legend(fontsize=12)
    apply_style(ax2)

    plt.tight_layout()
    save_figure(fig, output_dir, "extractor_cells", subdir)


def generate_summary_table(df_runs: pd.DataFrame, df_items: pd.DataFrame, output_dir: Path, subdir: str = ""):
    rows = []
    for approach in _approaches_in(df_runs):
        sub = df_runs[df_runs["approach"] == approach]
        items = df_items[df_items["approach"] == approach]
        if sub.empty:
            continue
        n = len(sub)
        mean_grade = sub["overall_grade_num"].mean()
        pct_ab = ((sub["overall_grade"].isin(["A", "B"])).sum() / n * 100) if n else 0
        pct_f = ((sub["overall_grade"] == "F").sum() / n * 100) if n else 0
        mean_dur = sub["duration_seconds"].mean()
        n_items_total = len(items)
        item_pct_ab = ((items["grade"].isin(["A", "B"])).sum() / n_items_total * 100) if n_items_total else 0

        rows.append({
            "Approach": APPROACH_LABELS.get(approach, approach).replace("\n", " "),
            "Runs": n,
            "Mean Grade": f"{mean_grade:.2f}",
            "% A-B (runs)": f"{pct_ab:.1f}",
            "% F (runs)": f"{pct_f:.1f}",
            "Items": n_items_total,
            "% A-B (items)": f"{item_pct_ab:.1f}",
            "Mean Duration (min)": f"{mean_dur / 60:.1f}" if pd.notna(mean_dur) else "—",
        })

    summary = pd.DataFrame(rows)
    target = output_dir / subdir if subdir else output_dir
    target.mkdir(parents=True, exist_ok=True)
    summary.to_csv(target / "summary_table.csv", index=False)
    latex = summary.to_latex(index=False, escape=True, column_format="l" + "r" * (len(summary.columns) - 1))
    (target / "summary_table.tex").write_text(latex)
    print("  Saved summary_table")


def generate_overview_csv(df_runs: pd.DataFrame, df_items: pd.DataFrame, output_dir: Path, subdir: str = ""):
    rows = []
    for approach_model, grp in df_runs.groupby("approach", observed=True):
        approach_raw = grp["approach_raw"].iloc[0]
        model_val = grp["model"].iloc[0]
        n_papers = grp["paper_slug"].nunique()
        n_papers_with_templates = (grp["n_table_templates"] > 0).sum()
        n_table_templates_total = grp["n_table_templates"].sum()
        n_runs_with_py = (grp["n_py_files"] > 0).sum()
        n_table_jsons_total = grp["n_table_jsons"].sum()
        items = df_items[df_items["approach"] == approach_model]
        table_items = items[items["item_type"] == "table"]
        n_tables_total = len(table_items)
        n_tables_f = (table_items["grade"] == "F").sum()
        n_tables_non_f = n_tables_total - n_tables_f
        rows.append({
            "approach": approach_raw, "model": model_val,
            "n_papers": int(n_papers),
            "n_papers_with_table_templates": int(n_papers_with_templates),
            "n_table_templates_total": int(n_table_templates_total),
            "n_runs_with_py_files": int(n_runs_with_py),
            "n_table_jsons_produced": int(n_table_jsons_total),
            "n_tables_in_verification": int(n_tables_total),
            "n_tables_non_f": int(n_tables_non_f),
            "n_tables_f": int(n_tables_f),
        })
    overview = pd.DataFrame(rows)
    target = output_dir / subdir if subdir else output_dir
    target.mkdir(parents=True, exist_ok=True)
    overview.to_csv(target / "overview_by_approach_model.csv", index=False)
    print("  Saved overview_by_approach_model.csv")
    print(overview.to_string(index=False))


# ============================================================================
# Section: Paper Level
# ============================================================================

def plot_overall_grade_distribution(df_runs: pd.DataFrame, output_dir: Path, subdir: str = ""):
    """Grouped bar chart of overall grades by approach (excludes F)."""
    df = df_runs[df_runs["overall_grade"] != "F"]
    grades_shown = [g for g in GRADE_ORDER if g != "F"]
    ct = pd.crosstab(df["approach"], df["overall_grade"], normalize="index") * 100
    ct = ct.reindex(columns=grades_shown, fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    present = _approaches_in(df)
    x = np.arange(len(present))
    width = 0.15
    for i, grade in enumerate(grades_shown):
        vals = [ct.loc[a, grade] if a in ct.index else 0 for a in present]
        ax.bar(x + i * width, vals, width, label=grade, color=GRADE_COLORS[grade], edgecolor="white")

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([APPROACH_LABELS.get(a, a).replace("\n", " ") for a in present], fontsize=10, rotation=25, ha="right")
    ax.set_ylabel("Share of runs (%)", fontsize=18, fontweight="bold")
    ax.legend(fontsize=14, ncol=6, loc="upper right")
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig, output_dir, "overall_grades", subdir)


def plot_agreement_matrix(df_items: pd.DataFrame, output_dir: Path, subdir: str = ""):
    deduped = df_items.drop_duplicates(subset=["paper_slug", "item_id", "approach"])
    pivot_data = deduped[["paper_slug", "item_id", "approach", "grade_num"]].copy()
    pivot_data["approach"] = pivot_data["approach"].astype(str)
    pivot = pivot_data.pivot_table(index=["paper_slug", "item_id"], columns="approach", values="grade_num", aggfunc="first")
    approaches = _approaches_in(df_items)
    approaches = [a for a in approaches if a in pivot.columns]
    if len(approaches) < 2:
        return

    n = len(approaches)
    agreement = np.zeros((n, n))
    within_one = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            a_vals = pivot[approaches[i]].values
            b_vals = pivot[approaches[j]].values
            mask = ~(np.isnan(a_vals) | np.isnan(b_vals))
            if mask.sum() == 0:
                continue
            agreement[i, j] = np.mean(a_vals[mask] == b_vals[mask]) * 100
            within_one[i, j] = np.mean(np.abs(a_vals[mask] - b_vals[mask]) <= 1) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    labels = [APPROACH_LABELS.get(a, a).replace("\n", " ") for a in approaches]
    for ax, data, subtitle in zip(axes, [agreement, within_one], ["Exact Match (%)", "Within 1 Grade (%)"]):
        sns.heatmap(data, annot=True, fmt=".0f", xticklabels=labels, yticklabels=labels,
                    cmap="YlGn", vmin=0, vmax=100, ax=ax, square=True, cbar_kws={"shrink": 0.8})
        ax.set_xlabel(subtitle, fontsize=16, fontweight="bold")
        ax.tick_params(labelsize=10)
    plt.tight_layout()
    save_figure(fig, output_dir, "agreement_matrix", subdir)


def plot_paper_difficulty(df_runs: pd.DataFrame, output_dir: Path, subdir: str = "", exclude_f: bool = False):
    df = df_runs if not exclude_f else df_runs[df_runs["overall_grade"] != "F"]
    agg = df.groupby(["paper_slug", "paper_title"])["overall_grade_num"].agg(["mean", "min", "max"])
    agg = agg.sort_values("mean", ascending=True)
    if len(agg) > 40:
        agg = pd.concat([agg.head(20), agg.tail(20)])
    labels = []
    for slug, title in agg.index:
        journal = _infer_journal(slug)
        display = title[:45] if title != slug else slug
        labels.append(f"{journal} — {display}" if journal != "Other" else display)

    fig, ax = plt.subplots(figsize=(12, max(8, len(agg) * 0.35)))
    y_pos = range(len(agg))
    colors = [GRADE_COLORS.get(NUM_TO_GRADE.get(round(v), "F"), "#95a5a6") for v in agg["mean"].values]
    ax.barh(y_pos, agg["mean"].values, color=colors, edgecolor="white", alpha=0.8)
    xerr_low = agg["mean"].values - agg["min"].values
    xerr_high = agg["max"].values - agg["mean"].values
    ax.errorbar(agg["mean"].values, y_pos, xerr=[xerr_low, xerr_high],
                fmt="none", ecolor="black", elinewidth=1.2, capsize=3, capthick=1.0)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xticks(range(6))
    ax.set_xticklabels(GRADE_ORDER[::-1])
    ax.set_xlabel("Mean Grade (across approaches)", fontsize=18, fontweight="bold")
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig, output_dir, "paper_difficulty", subdir)


def plot_scatter_vs_grade(df: pd.DataFrame, x_col: str, x_label: str, output_dir: Path,
                          name: str, log_x: bool = False, grade_col: str = "overall_grade_num",
                          subdir: str = "", exclude_f: bool = False):
    df_plot = df.dropna(subset=[x_col, grade_col]).copy()
    if exclude_f:
        grade_str_col = "overall_grade" if grade_col == "overall_grade_num" else "grade"
        if grade_str_col in df_plot.columns:
            df_plot = df_plot[df_plot[grade_str_col] != "F"]
    if df_plot.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    for approach in _approaches_in(df_plot):
        sub = df_plot[df_plot["approach"] == approach]
        if sub.empty:
            continue
        ax.scatter(sub[x_col], sub[grade_col] + np.random.uniform(-0.15, 0.15, len(sub)),
                   color=APPROACH_COLORS.get(approach, "#95a5a6"), label=APPROACH_LABELS.get(approach, approach).replace("\n", " "),
                   alpha=0.6, s=60, edgecolor="white", linewidth=0.5)

    if log_x and df_plot[x_col].gt(0).any():
        ax.set_xscale("log")
    ax.set_xlabel(x_label, fontsize=18, fontweight="bold")
    ax.set_ylabel("Overall Grade", fontsize=18, fontweight="bold")
    grades_shown = [g for g in GRADE_ORDER if g != "F"] if exclude_f else GRADE_ORDER
    ax.set_yticks([GRADE_TO_NUM[g] for g in grades_shown])
    ax.set_yticklabels(grades_shown[::-1])
    ax.legend(fontsize=12)
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig, output_dir, name, subdir)


def plot_grade_by_discipline(df_runs: pd.DataFrame, output_dir: Path, subdir: str = ""):
    df = df_runs[df_runs["overall_grade"] != "F"]
    grades_shown = [g for g in GRADE_ORDER if g != "F"]
    ct = pd.crosstab(df["discipline"], df["overall_grade"], normalize="index") * 100
    ct = ct.reindex(columns=grades_shown, fill_value=0)
    ct["_mean"] = sum(ct[g] * GRADE_TO_NUM[g] for g in grades_shown if g in ct.columns) / 100
    ct = ct.sort_values("_mean", ascending=False).drop(columns="_mean")

    fig, ax = plt.subplots(figsize=(10, 6))
    ct.plot(kind="bar", stacked=True, ax=ax, color=[GRADE_COLORS[g] for g in ct.columns], edgecolor="white", width=0.7)
    ax.set_xlabel("Discipline", fontsize=18, fontweight="bold")
    ax.set_ylabel("Share (%)", fontsize=18, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(fontsize=12, ncol=6, loc="upper right")
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig, output_dir, "grade_by_discipline", subdir)


def plot_grade_by_language(df_runs: pd.DataFrame, output_dir: Path, subdir: str = ""):
    df = df_runs[(df_runs["overall_grade"] != "F") & (df_runs["original_language"] != "Unknown")]
    grades_shown = [g for g in GRADE_ORDER if g != "F"]
    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Left: primary language
    ax = axes[0]
    ct = pd.crosstab(df["original_language"], df["overall_grade"], normalize="index") * 100
    ct = ct.reindex(columns=grades_shown, fill_value=0)
    ct["_mean"] = sum(ct[g] * GRADE_TO_NUM[g] for g in grades_shown if g in ct.columns) / 100
    ct = ct.sort_values("_mean", ascending=False).drop(columns="_mean")
    lang_counts = df.drop_duplicates("paper_slug").groupby("original_language").size()
    ct.index = [f"{lang} (n={lang_counts.get(lang, 0)})" for lang in ct.index]
    ct.plot(kind="bar", stacked=True, ax=ax, color=[GRADE_COLORS[g] for g in ct.columns], edgecolor="white", width=0.7)
    ax.set_xlabel("Primary Language", fontsize=16, fontweight="bold")
    ax.set_ylabel("Share (%)", fontsize=18, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(fontsize=11, ncol=5, loc="upper right")
    apply_style(ax)

    # Right: full language combo
    ax2 = axes[1]
    ct2 = pd.crosstab(df["original_languages_all"], df["overall_grade"], normalize="index") * 100
    ct2 = ct2.reindex(columns=grades_shown, fill_value=0)
    ct2["_mean"] = sum(ct2[g] * GRADE_TO_NUM[g] for g in grades_shown if g in ct2.columns) / 100
    ct2 = ct2.sort_values("_mean", ascending=False).drop(columns="_mean")
    combo_counts = df.drop_duplicates("paper_slug").groupby("original_languages_all").size()
    ct2.index = [f"{combo} (n={combo_counts.get(combo, 0)})" for combo in ct2.index]
    ct2.plot(kind="bar", stacked=True, ax=ax2, color=[GRADE_COLORS[g] for g in ct2.columns], edgecolor="white", width=0.7)
    ax2.set_xlabel("Language Combination", fontsize=16, fontweight="bold")
    ax2.set_ylabel("Share (%)", fontsize=18, fontweight="bold")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, ha="right")
    ax2.legend(fontsize=11, ncol=5, loc="upper right")
    apply_style(ax2)

    plt.tight_layout()
    save_figure(fig, output_dir, "grade_by_language", subdir)


def plot_duration_vs_grade(df_runs: pd.DataFrame, output_dir: Path, subdir: str = ""):
    """Left: duration distribution per approach. Right: duration vs grade. Both excl. F."""
    df = df_runs[df_runs["duration_seconds"].notna() & df_runs["overall_grade_num"].notna() & (df_runs["overall_grade"] != "F")].copy()
    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    approaches = _approaches_in(df)
    data, labels, colors = [], [], []
    for a in approaches:
        vals = df.loc[df["approach"] == a, "duration_seconds"].dropna() / 60
        if not vals.empty:
            data.append(vals.values)
            labels.append(APPROACH_LABELS.get(a, a).replace("\n", " "))
            colors.append(APPROACH_COLORS.get(a, "#95a5a6"))
    if data:
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6, showfliers=False)
        plt.setp(ax.get_xticklabels(), fontsize=10, rotation=25, ha="right")
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(2)
        for i, vals in enumerate(data):
            med = np.median(vals)
            ax.annotate(f"{med:.0f}m", xy=(i + 1, med), xytext=(i + 1.3, med),
                        fontsize=11, color="black", fontweight="bold", va="center")
    ax.set_ylabel("Duration (minutes)", fontsize=18, fontweight="bold")
    apply_style(ax)

    ax2 = axes[1]
    for approach in _approaches_in(df):
        sub = df[df["approach"] == approach]
        if sub.empty:
            continue
        ax2.scatter(sub["duration_seconds"] / 60, sub["overall_grade_num"] + np.random.uniform(-0.15, 0.15, len(sub)),
                   color=APPROACH_COLORS.get(approach, "#95a5a6"), label=APPROACH_LABELS.get(approach, approach).replace("\n", " "),
                   alpha=0.6, s=60, edgecolor="white", linewidth=0.5)
    ax2.set_xlabel("Duration (minutes)", fontsize=18, fontweight="bold")
    ax2.set_ylabel("Overall Grade", fontsize=18, fontweight="bold")
    grades_shown = [g for g in GRADE_ORDER if g != "F"]
    ax2.set_yticks([GRADE_TO_NUM[g] for g in grades_shown])
    ax2.set_yticklabels(grades_shown[::-1])
    ax2.legend(fontsize=12)
    apply_style(ax2)

    plt.tight_layout()
    save_figure(fig, output_dir, "duration_vs_grade", subdir)


# ============================================================================
# Section: Item Level — Tables
# ============================================================================

def plot_item_grade_by_type(df_items: pd.DataFrame, output_dir: Path, item_type: str | None, name: str,
                            subdir: str = "", exclude_f: bool = True):
    df = df_items if item_type is None else df_items[df_items["item_type"] == item_type]
    if exclude_f:
        df = df[df["grade"] != "F"]
    grades_shown = [g for g in GRADE_ORDER if g != "F"] if exclude_f else GRADE_ORDER
    if df.empty:
        return

    ct = pd.crosstab(df["approach"], df["grade"], normalize="index") * 100
    ct = ct.reindex(columns=grades_shown, fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    present = _approaches_in(df)
    ct.loc[[a for a in present if a in ct.index]].plot(kind="bar", ax=ax, color=[GRADE_COLORS[g] for g in ct.columns],
                                edgecolor="white", width=0.8)
    ax.set_xticklabels([APPROACH_LABELS.get(a, a).replace("\n", " ") for a in present if a in ct.index], fontsize=10, rotation=25, ha="right")
    ax.set_ylabel("Share of items (%)", fontsize=18, fontweight="bold")
    ax.legend(fontsize=14, ncol=6, loc="upper right")
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig, output_dir, name, subdir)


def plot_item_number_vs_grade(df_items: pd.DataFrame, output_dir: Path, subdir: str = ""):
    df = df_items[df_items["item_number"].notna() & (df_items["item_type"] == "table") & (df_items["grade"] != "F")].copy()
    df["item_number"] = df["item_number"].astype(int)
    df = df[df["item_number"] <= 8]
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for approach in _approaches_in(df):
        asub = df[df["approach"] == approach]
        if asub.empty:
            continue
        grouped = asub.groupby("item_number")["grade_num"]
        means = grouped.mean()
        sems = grouped.sem()
        ax.errorbar(means.index, means.values, yerr=1.96 * sems.fillna(0).values,
                    label=APPROACH_LABELS.get(approach, approach).replace("\n", " "),
                    color=APPROACH_COLORS.get(approach, "#95a5a6"),
                    marker="o", capsize=3, linewidth=2)

    ax.set_xlabel("Table Number", fontsize=18, fontweight="bold")
    ax.set_ylabel("Mean Grade", fontsize=18, fontweight="bold")
    ax.set_yticks(range(6))
    ax.set_yticklabels(GRADE_ORDER[::-1])
    ax.legend(fontsize=12, loc="lower left")
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig, output_dir, "item_number_vs_grade", subdir)


# ============================================================================
# Section: Cell Level
# ============================================================================

def plot_coefficient_se_scaled(df_cells: pd.DataFrame, output_dir: Path, subdir: str = ""):
    df = df_cells[
        (df_cells["row_type"] == "coefficient") &
        df_cells["original_value"].notna() &
        df_cells["replicated_value"].notna() &
        df_cells["is_numeric"] &
        (df_cells["item_grade"] != "F")
    ].copy()
    if df.empty:
        return

    df["abs_diff"] = (df["original_value"].astype(float) - df["replicated_value"].astype(float)).abs()

    # Use original SE if available, fall back to replicated SE
    df["se"] = df["original_se"]
    mask_no_orig = df["se"].isna()
    df.loc[mask_no_orig, "se"] = df.loc[mask_no_orig, "replicated_se"]

    # Filter to rows with valid SE > 0
    df = df[df["se"].notna() & (df["se"].astype(float) > 0)].copy()
    if df.empty:
        print("  Skipping coefficient_se_scaled: no coefficients with SE data")
        return

    df["diff_over_se"] = df["abs_diff"] / df["se"].astype(float)
    df["diff_over_se_capped"] = df["diff_over_se"].clip(upper=10)

    approaches = _approaches_in(df)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: boxplot by approach (no outlier dots)
    ax = axes[0]
    data, labels, colors = [], [], []
    for a in approaches:
        vals = df.loc[df["approach"] == a, "diff_over_se_capped"].values
        if len(vals) > 0:
            data.append(vals)
            labels.append(APPROACH_LABELS.get(a, a).replace("\n", " "))
            colors.append(APPROACH_COLORS.get(a, "#95a5a6"))

    if data:
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6, showfliers=False)
        plt.setp(ax.get_xticklabels(), fontsize=10, rotation=25, ha="right")
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(2)
        for i, vals in enumerate(data):
            med = np.median(vals)
            ax.annotate(f"{med:.2f}", xy=(i + 1, med), xytext=(i + 1.3, med),
                        fontsize=11, color="black", fontweight="bold", va="center")
    ax.set_ylabel("|Coeff. difference| / SE (capped at 10)", fontsize=14, fontweight="bold")
    ax.axhline(y=1.96, color="red", linestyle="--", alpha=0.5, label="1.96 (95% CI)")
    ax.legend(fontsize=12)
    apply_style(ax)

    # Right: cumulative distribution per approach
    ax2 = axes[1]
    for a in approaches:
        vals = df.loc[df["approach"] == a, "diff_over_se"].sort_values().values
        if len(vals) == 0:
            continue
        cdf_y = np.arange(1, len(vals) + 1) / len(vals) * 100
        ax2.plot(vals, cdf_y, label=APPROACH_LABELS.get(a, a).replace("\n", " "),
                 color=APPROACH_COLORS.get(a, "#95a5a6"), linewidth=2)
    ax2.set_xlim(0, 10)
    ax2.axvline(x=1.96, color="red", linestyle="--", alpha=0.5, label="1.96")
    ax2.set_xlabel("|Coeff. difference| / SE", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Cumulative share of coefficients (%)", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=9)
    apply_style(ax2)

    plt.tight_layout()
    save_figure(fig, output_dir, "coefficient_se_scaled", subdir)


def plot_same_significance(df_cells: pd.DataFrame, output_dir: Path, subdir: str = ""):
    df = df_cells[
        (df_cells["row_type"] == "coefficient") &
        df_cells["significance_stars_orig"].notna() &
        df_cells["significance_stars_repl"].notna() &
        (df_cells["item_grade"] != "F")
    ].copy()
    if df.empty:
        return

    df["sig_match"] = df["significance_stars_orig"] == df["significance_stars_repl"]
    approaches = _approaches_in(df)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    match_rates, xlabels, colors = [], [], []
    for a in approaches:
        sub = df[df["approach"] == a]
        if sub.empty:
            continue
        match_rates.append(sub["sig_match"].mean() * 100)
        xlabels.append(APPROACH_LABELS.get(a, a).replace("\n", " "))
        colors.append(APPROACH_COLORS.get(a, "#95a5a6"))

    x = np.arange(len(xlabels))
    ax.bar(x, match_rates, color=colors, edgecolor="white", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=10, rotation=25, ha="right")
    ax.set_ylabel("Coefficients with same significance (%)", fontsize=14, fontweight="bold")
    for i, rate in enumerate(match_rates):
        ax.text(i, rate + 1, f"{rate:.0f}%", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 105)
    apply_style(ax)

    ax2 = axes[1]
    df["sig_diff"] = df["significance_stars_repl"].astype(int) - df["significance_stars_orig"].astype(int)
    bins = [-4, -2, -1, 0, 1, 2, 4]
    bin_labels = ["<-1", "-1", "0", "+1", "+2", ">+2"]
    df["sig_diff_bin"] = pd.cut(df["sig_diff"], bins=bins, labels=bin_labels)
    ct = pd.crosstab(df["sig_diff_bin"], df["approach"].astype(str), normalize="columns") * 100
    ct = ct.reindex(columns=[a for a in approaches if a in ct.columns])
    x2 = np.arange(len(bin_labels))
    width = 0.8 / max(len(approaches), 1)
    for i, a in enumerate(approaches):
        if a not in ct.columns:
            continue
        ax2.bar(x2 + i * width, ct[a].values, width,
                label=APPROACH_LABELS.get(a, a).replace("\n", " "),
                color=APPROACH_COLORS.get(a, "#95a5a6"), alpha=0.8, edgecolor="white")
    ax2.set_xticks(x2 + width * (len(approaches) - 1) / 2)
    ax2.set_xticklabels(bin_labels, fontsize=11)
    ax2.set_xlabel("Stars difference (replicated - original)", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Share of coefficients (%)", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=9)
    apply_style(ax2)

    plt.tight_layout()
    save_figure(fig, output_dir, "same_significance", subdir)


def plot_same_sign(df_cells: pd.DataFrame, output_dir: Path, subdir: str = ""):
    df = df_cells[
        (df_cells["row_type"] == "coefficient") &
        df_cells["sign_match"].notna() &
        (df_cells["item_grade"] != "F")
    ].copy()
    if df.empty:
        return

    approaches = _approaches_in(df)
    fig, ax = plt.subplots(figsize=(10, 6))
    match_rates, xlabels, colors, n_cells_list = [], [], [], []
    for a in approaches:
        sub = df[df["approach"] == a]
        if sub.empty:
            continue
        match_rates.append(sub["sign_match"].mean() * 100)
        xlabels.append(APPROACH_LABELS.get(a, a).replace("\n", " "))
        colors.append(APPROACH_COLORS.get(a, "#95a5a6"))
        n_cells_list.append(len(sub))

    x = np.arange(len(xlabels))
    ax.bar(x, match_rates, color=colors, edgecolor="white", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=10, rotation=25, ha="right")
    ax.set_ylabel("Coefficients with same sign (%)", fontsize=16, fontweight="bold")
    for i, (rate, n) in enumerate(zip(match_rates, n_cells_list)):
        ax.text(i, rate + 0.5, f"{rate:.1f}%\n(n={n})", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 105)
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig, output_dir, "same_sign", subdir)


def plot_statistic_pct_difference(df_cells: pd.DataFrame, output_dir: Path,
                                   row_type_filter: str, name: str,
                                   ylabel: str = "", subdir: str = ""):
    df = df_cells[
        (df_cells["row_type"] == row_type_filter) &
        df_cells["percent_difference"].notna() &
        df_cells["is_numeric"] &
        (df_cells["item_grade"] != "F")
    ].copy()
    if df.empty:
        return

    df["pct_capped"] = df["percent_difference"].clip(upper=200)
    approaches = _approaches_in(df)

    fig, ax = plt.subplots(figsize=(10, 6))
    data, labels, colors = [], [], []
    for a in approaches:
        vals = df.loc[df["approach"] == a, "pct_capped"].values
        if len(vals) > 0:
            data.append(vals)
            labels.append(APPROACH_LABELS.get(a, a).replace("\n", " "))
            colors.append(APPROACH_COLORS.get(a, "#95a5a6"))

    if not data:
        plt.close(fig)
        return

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6, showfliers=False)
    plt.setp(ax.get_xticklabels(), fontsize=10, rotation=25, ha="right")
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2)
    for i, vals in enumerate(data):
        med = np.median(vals)
        ax.annotate(f"{med:.1f}%", xy=(i + 1, med), xytext=(i + 1.3, med),
                    fontsize=11, color="black", fontweight="bold", va="center")

    ax.set_ylabel(ylabel or f"% difference ({row_type_filter})", fontsize=16, fontweight="bold")
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig, output_dir, name, subdir)


# ============================================================================
# Section: Error Analysis
# ============================================================================

def plot_fault_attribution(df_items: pd.DataFrame, output_dir: Path, subdir: str = ""):
    """Stacked bar of primary_fault from explainer_report, by approach."""
    df = df_items[
        df_items["primary_fault"].notna() &
        (df_items["primary_fault"] != "") &
        (df_items["grade"] != "F")
    ].copy()
    if df.empty:
        return

    cat_order = [
        "replicator", "extractor_misinterpreted", "extractor_missed",
        "data_limitation", "paper_underspecified", "paper_code_mismatch",
        "results_extractor", "software_differences", "results_mismatched", "other",
    ]
    cat_labels = {
        "replicator": "Replicator", "extractor_misinterpreted": "Extractor misinterpreted",
        "extractor_missed": "Extractor missed", "data_limitation": "Data limitation",
        "paper_underspecified": "Paper underspecified", "paper_code_mismatch": "Paper–code mismatch",
        "results_extractor": "Results extractor", "software_differences": "Software differences",
        "results_mismatched": "Results mismatched", "other": "Other",
    }
    cat_colors = {
        "replicator": "#e74c3c", "extractor_misinterpreted": "#e67e22",
        "extractor_missed": "#f39c12", "data_limitation": "#3498db",
        "paper_underspecified": "#9b59b6", "paper_code_mismatch": "#8e44ad",
        "results_extractor": "#1abc9c", "software_differences": "#2ecc71",
        "results_mismatched": "#34495e", "other": "#95a5a6",
    }

    extra_cats = [c for c in df["primary_fault"].unique() if c not in cat_order]
    cat_order = cat_order + extra_cats

    ct = pd.crosstab(df["approach"], df["primary_fault"], normalize="index") * 100
    ct = ct.reindex(columns=[c for c in cat_order if c in ct.columns], fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    present = [a for a in _approaches_in(df) if a in ct.index]
    ct_present = ct.loc[present].copy()
    ct_present.columns = [cat_labels.get(c, c) for c in ct_present.columns]
    display_colors = [cat_colors.get(c, "#95a5a6") for c in ct.columns]

    ct_present.plot(kind="bar", stacked=True, ax=ax, color=display_colors, edgecolor="white", width=0.7)
    ax.set_xticklabels([APPROACH_LABELS.get(a, a).replace("\n", " ") for a in present], fontsize=10, rotation=25, ha="right")
    ax.set_ylabel("Share (%)", fontsize=18, fontweight="bold")
    ax.legend(fontsize=10, title="Fault Category", loc="upper right")
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig, output_dir, "fault_attribution", subdir)


def generate_fault_by_grade_table(df_items: pd.DataFrame, output_dir: Path, subdir: str = ""):
    df = df_items[
        df_items["primary_fault"].notna() &
        (df_items["primary_fault"] != "") &
        (df_items["grade"] != "F")
    ].copy()
    if df.empty:
        return

    ct = pd.crosstab(df["grade"], df["primary_fault"])
    grades_shown = [g for g in GRADE_ORDER if g != "F" and g in ct.index]
    cat_order = [
        "replicator", "extractor_misinterpreted", "extractor_missed",
        "data_limitation", "paper_underspecified", "paper_code_mismatch",
        "results_extractor", "software_differences", "results_mismatched", "other",
    ]
    extra = [c for c in ct.columns if c not in cat_order]
    cat_order = [c for c in cat_order if c in ct.columns] + extra
    ct = ct.reindex(index=grades_shown, columns=cat_order, fill_value=0)

    ct["Total"] = ct.sum(axis=1)
    ct_pct = ct.div(ct["Total"], axis=0).drop(columns="Total") * 100

    combined = pd.DataFrame(index=ct.index)
    for col in cat_order:
        combined[col] = ct[col].astype(str) + " (" + ct_pct[col].round(0).astype(int).astype(str) + "%)"
    combined["Total"] = ct["Total"].astype(int)

    target = output_dir / subdir if subdir else output_dir
    target.mkdir(parents=True, exist_ok=True)
    combined.to_csv(target / "fault_by_grade.csv")
    latex = combined.to_latex(escape=True, column_format="l" + "r" * len(combined.columns))
    (target / "fault_by_grade.tex").write_text(latex)
    print(f"  Saved fault_by_grade")
    print(combined.to_string())


def plot_within_table_error_agreement(df_items: pd.DataFrame, output_dir: Path, subdir: str = ""):
    df = df_items[
        (df_items["item_type"] == "table") &
        df_items["primary_fault"].notna() &
        (df_items["primary_fault"] != "") &
        (df_items["grade"] != "F")
    ].copy()
    if df.empty:
        return

    df["approach"] = df["approach"].astype(str)
    pivot = df.pivot_table(index=["paper_slug", "item_id"], columns="approach", values="primary_fault", aggfunc="first")
    approaches = _approaches_in(df)
    approaches = [a for a in approaches if a in pivot.columns]
    if len(approaches) < 2:
        return

    n = len(approaches)
    agreement = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            a_vals = pivot[approaches[i]] if approaches[i] in pivot.columns else pd.Series(dtype=str)
            b_vals = pivot[approaches[j]] if approaches[j] in pivot.columns else pd.Series(dtype=str)
            mask = a_vals.notna() & b_vals.notna()
            if mask.sum() == 0:
                continue
            agreement[i, j] = (a_vals[mask] == b_vals[mask]).mean() * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    labels = [APPROACH_LABELS.get(a, a).replace("\n", " ") for a in approaches]
    sns.heatmap(agreement, annot=True, fmt=".0f", xticklabels=labels, yticklabels=labels,
                cmap="YlGn", vmin=0, vmax=100, ax=ax, square=True, cbar_kws={"shrink": 0.8})
    ax.set_xlabel("Error Attribution Agreement (%)", fontsize=14, fontweight="bold")
    ax.tick_params(labelsize=10)
    plt.tight_layout()
    save_figure(fig, output_dir, "within_table_error_agreement", subdir)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze i4rep benchmark results")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--papers-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="analysis_output")
    args = parser.parse_args()

    if args.results_dir is None:
        if TEXTLAB_BASE.exists():
            args.results_dir = str(TEXTLAB_BASE / "results")
            if args.papers_dir is None:
                args.papers_dir = str(TEXTLAB_BASE / "papers")
        elif LOCAL_BASE.exists():
            args.results_dir = str(LOCAL_BASE / "results")
            if args.papers_dir is None:
                args.papers_dir = str(LOCAL_BASE / "papers")
        else:
            print("ERROR: Cannot find results directory. Use --results-dir.")
            sys.exit(1)

    results_dir = Path(args.results_dir)
    papers_dir = Path(args.papers_dir) if args.papers_dir else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results: {results_dir}")
    print(f"Papers:  {papers_dir}")
    print(f"Output:  {output_dir}")
    print()

    setup_style()
    df_runs, df_items, df_cells = load_results(results_dir, papers_dir)

    if df_runs.empty:
        print("No results found. Exiting.")
        return

    # Filter out non-numerical tables from item-level analysis
    n_non_numerical = df_items["non_numerical"].sum() if "non_numerical" in df_items.columns else 0
    if n_non_numerical > 0:
        df_items = df_items[~df_items["non_numerical"]].copy()
        combos = [a for a in APPROACH_ORDER if a in df_items["approach"].unique()]
        df_items["approach"] = pd.Categorical(df_items["approach"], categories=combos, ordered=True)
        df_items["grade"] = pd.Categorical(df_items["grade"], categories=GRADE_ORDER, ordered=True)
        print(f"Excluded {n_non_numerical} non-numerical table items from analysis")

    # Print quick summary
    print(f"\n{'='*60}")
    print(f"Runs:  {len(df_runs)} ({df_runs['paper_slug'].nunique()} papers, "
          f"{df_runs['approach'].nunique()} approaches)")
    print(f"Items: {len(df_items)}")
    for a in df_runs["approach"].cat.categories:
        sub = df_runs[df_runs["approach"] == a]
        if sub.empty:
            continue
        label = APPROACH_LABELS.get(a, a).replace("\n", " ")
        print(f"  {label:25s}: {len(sub):3d} runs, "
              f"mean={sub['overall_grade_num'].mean():.2f}, "
              f"A-B={sub['overall_grade'].isin(['A','B']).mean()*100:.0f}%, "
              f"F={sub['overall_grade'].eq('F').mean()*100:.0f}%")
    print(f"{'='*60}\n")

    # Save DataFrames
    df_runs.to_csv(output_dir / "df_runs.csv", index=False)
    df_items.to_csv(output_dir / "df_items.csv", index=False)
    df_cells.to_csv(output_dir / "df_cells.csv", index=False, escapechar="\\")
    print(f"  Saved df_runs.csv, df_items.csv, df_cells.csv ({len(df_cells)} cells)")

    # ── Setup & Descriptives ──────────────────────────────────────
    SD = "setup_descriptives"
    print(f"\n{SD}")
    plot_extractor_row_type_distribution(df_cells, output_dir, subdir=SD)
    plot_first_fail_distribution(df_items, output_dir, subdir=SD)
    plot_extractor_cells(df_cells, output_dir, subdir=SD)
    generate_summary_table(df_runs, df_items, output_dir, subdir=SD)
    generate_overview_csv(df_runs, df_items, output_dir, subdir=SD)

    # ── Paper Level ───────────────────────────────────────────────
    df_items_tables = df_items[df_items["item_type"] == "table"]
    PL = "paper_level"
    print(f"\n{PL}")
    plot_agreement_matrix(df_items_tables, output_dir, subdir=PL)
    plot_overall_grade_distribution(df_runs, output_dir, subdir=PL)
    plot_paper_difficulty(df_runs, output_dir, subdir=PL, exclude_f=True)
    plot_scatter_vs_grade(df_runs, "total_data_size_bytes", "Total Data Size (bytes)",
                          output_dir, "data_size_vs_grade", log_x=True, subdir=PL, exclude_f=True)
    plot_grade_by_discipline(df_runs, output_dir, subdir=PL)
    plot_grade_by_language(df_runs, output_dir, subdir=PL)
    plot_duration_vs_grade(df_runs, output_dir, subdir=PL)

    # ── Item Level — Tables ───────────────────────────────────────
    IT = "item_tables"
    print(f"\n{IT}")
    plot_item_grade_by_type(df_items, output_dir, "table", "table_grade_distribution", subdir=IT)
    plot_item_number_vs_grade(df_items, output_dir, subdir=IT)
    plot_scatter_vs_grade(df_runs, "methodology_summary_len", "Methodology Summary Length (chars)",
                          output_dir, "methodology_length_vs_grade", subdir=IT, exclude_f=True)
    plot_scatter_vs_grade(df_runs, "total_code_chars", "Total Code Size (chars)",
                          output_dir, "code_length_vs_grade", log_x=True, subdir=IT, exclude_f=True)

    # ── Item Level — Figures ──────────────────────────────────────
    IF_ = "item_figures"
    print(f"\n{IF_}")
    plot_item_grade_by_type(df_items, output_dir, "figure", "figure_grade_distribution", subdir=IF_)

    # ── Cell Level ────────────────────────────────────────────────
    CL = "cell_level"
    print(f"\n{CL}")
    plot_coefficient_se_scaled(df_cells, output_dir, subdir=CL)
    plot_same_significance(df_cells, output_dir, subdir=CL)
    plot_same_sign(df_cells, output_dir, subdir=CL)
    plot_statistic_pct_difference(df_cells, output_dir, "statistic_n_obs",
                                   "n_obs_pct_difference",
                                   ylabel="% difference (N observations)", subdir=CL)
    plot_statistic_pct_difference(df_cells, output_dir, "statistic_r2",
                                   "r2_pct_difference",
                                   ylabel="% difference (R²)", subdir=CL)

    # ── Error Analysis ────────────────────────────────────────────
    EA = "error_analysis"
    print(f"\n{EA}")
    plot_fault_attribution(df_items, output_dir, subdir=EA)
    generate_fault_by_grade_table(df_items, output_dir, subdir=EA)
    plot_within_table_error_agreement(df_items, output_dir, subdir=EA)

    print(f"\nDone! All outputs in {output_dir}/")


if __name__ == "__main__":
    main()
