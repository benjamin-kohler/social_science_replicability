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

APPROACH_ORDER = ["claude-code", "codex", "swe-agent", "opencode"]
APPROACH_LABELS = {
    "claude-code": "Claude Code",
    "codex": "Codex CLI",
    "swe-agent": "SWE-Agent",
    "opencode": "OpenCode",
}

APPROACH_COLORS = {
    "claude-code": "#E07B39",
    "codex": "#10A37F",
    "swe-agent": "#6C5CE7",
    "opencode": "#0984E3",
}

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


def save_figure(fig, output_dir: Path, name: str):
    fig.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


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
    """Extract numeric part from 'Table 3', 'Figure A.2', etc."""
    m = re.search(r"(\d+)", item_id)
    return int(m.group(1)) if m else None


def _parse_item_type(item_id: str, item_type: str | None) -> str:
    """Normalize item type."""
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
    """Return (n_code_files, total_chars)."""
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


def _dataset_stats(data_dir: Path) -> tuple[int, int]:
    """Return (n_files, total_bytes) for a paper's data directory."""
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
    """Query OpenCode's SQLite DB for token usage matching a workspace path."""
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
    """Parse model and approach from run directory name.

    Dir format: {model}_{paper_slug}_{approach}
    Returns (model, approach) or None.
    """
    # Find the paper_slug in the dirname and split around it
    idx = dirname.find(f"_{paper_slug}_")
    if idx == -1:
        return None
    model = dirname[:idx]
    approach = dirname[idx + len(f"_{paper_slug}_"):]
    if approach not in APPROACH_ORDER:
        return None
    return model, approach


# ============================================================================
# Data Loading
# ============================================================================

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
            result = _load_json(run_dir / "result.json")
            usage = _load_json(run_dir / "usage.json")
            workspace = run_dir / "workspace"
            meth_summary = _load_json(workspace / "methodology_summary.json")

            # Build explanation lookup
            explanation_map = {}
            if er and "analyses" in er:
                for analysis in er["analyses"]:
                    explanation_map[analysis.get("item_id", "")] = analysis

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
                        note = cell.get("note", "")
                        is_numeric = (ov is not None) or (cell.get("percent_difference") is not None)
                        if not is_numeric and rv is not None:
                            # replicated_value exists but no original — still numeric
                            try:
                                float(rv)
                                is_numeric = True
                            except (TypeError, ValueError):
                                pass
                        cell_rows.append({
                            "paper_slug": paper_slug,
                            "approach": approach,
                            "model": model,
                            "item_id": item_id,
                            "item_grade": grade,
                            "row_label": cell.get("row_label", ""),
                            "column_label": cell.get("column_label", ""),
                            "original_value": ov,
                            "replicated_value": rv,
                            "percent_difference": cell.get("percent_difference"),
                            "absolute_difference": cell.get("absolute_difference"),
                            "sign_match": cell.get("sign_match"),
                            "cell_grade": cell.get("grade", ""),
                            "is_numeric": is_numeric,
                        })

                # Explanation data
                expl = explanation_map.get(item_id, {})

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
                    "n_cells_compared": n_cells,
                    "mean_pct_diff": mean_pct,
                    "fault_attribution": expl.get("fault_attribution"),
                    "confidence": expl.get("confidence"),
                    "likely_causes": expl.get("likely_causes", []),
                    "description_of_discrepancy": expl.get("description_of_discrepancy", ""),
                })

            # Run-level fields
            overall_grade = vr.get("overall_grade", "F")
            duration = result.get("duration_seconds") if result else None
            n_code_files, total_code_chars = _count_code_in_workspace(workspace)

            meth_len = 0
            n_tables_summary = 0
            n_figures_summary = 0
            if meth_summary:
                meth_len = len(json.dumps(meth_summary))
                n_tables_summary = len(meth_summary.get("tables", []))
                n_figures_summary = len(meth_summary.get("figures", []))

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
            if papers_dir:
                paper_data_dir = papers_dir / paper_slug / "data"
                n_datasets, total_data_bytes = _dataset_stats(paper_data_dir)
                meta = _load_json(papers_dir / paper_slug / "metadata.json")
                if meta and "title" in meta:
                    paper_title = meta["title"]

            run_rows.append({
                "paper_slug": paper_slug,
                "paper_title": paper_title,
                "journal": journal,
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

    # Enforce categorical ordering
    if not df_runs.empty:
        df_runs["approach"] = pd.Categorical(df_runs["approach"], categories=APPROACH_ORDER, ordered=True)
        df_runs["overall_grade"] = pd.Categorical(df_runs["overall_grade"], categories=GRADE_ORDER, ordered=True)
    if not df_items.empty:
        df_items["approach"] = pd.Categorical(df_items["approach"], categories=APPROACH_ORDER, ordered=True)
        df_items["grade"] = pd.Categorical(df_items["grade"], categories=GRADE_ORDER, ordered=True)
    if not df_cells.empty:
        df_cells["approach"] = pd.Categorical(df_cells["approach"], categories=APPROACH_ORDER, ordered=True)
        df_cells["cell_grade"] = pd.Categorical(df_cells["cell_grade"], categories=GRADE_ORDER, ordered=True)

    print(f"Loaded {len(df_runs)} runs, {len(df_items)} items, {len(df_cells)} cells")
    return df_runs, df_items, df_cells


# ============================================================================
# Section 1: Performance Distribution
# ============================================================================

def plot_overall_grade_distribution(df_runs: pd.DataFrame, output_dir: Path):
    """Grouped bar chart of overall grades by approach."""
    ct = pd.crosstab(df_runs["approach"], df_runs["overall_grade"], normalize="index") * 100
    ct = ct.reindex(columns=GRADE_ORDER, fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(APPROACH_ORDER))
    width = 0.13
    for i, grade in enumerate(GRADE_ORDER):
        if grade in ct.columns:
            vals = [ct.loc[a, grade] if a in ct.index else 0 for a in APPROACH_ORDER]
        else:
            vals = [0] * len(APPROACH_ORDER)
        ax.bar(x + i * width, vals, width, label=grade, color=GRADE_COLORS[grade], edgecolor="white")

    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels([APPROACH_LABELS[a] for a in APPROACH_ORDER])
    ax.set_ylabel("Share of runs (%)", fontsize=18, fontweight="bold")
    ax.legend(fontsize=14, ncol=6, loc="upper right")
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig, output_dir, "overall_grades")


def plot_item_grade_by_type(df_items: pd.DataFrame, output_dir: Path, item_type: str | None, name: str):
    """Grade distribution for items, optionally filtered by type."""
    df = df_items if item_type is None else df_items[df_items["item_type"] == item_type]
    if df.empty:
        print(f"  Skipping {name}: no data")
        return

    ct = pd.crosstab(df["approach"], df["grade"], normalize="index") * 100
    ct = ct.reindex(columns=GRADE_ORDER, fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    ct.loc[APPROACH_ORDER].plot(kind="bar", ax=ax, color=[GRADE_COLORS[g] for g in ct.columns],
                                edgecolor="white", width=0.8)
    ax.set_xticklabels([APPROACH_LABELS.get(a, a) for a in APPROACH_ORDER], rotation=0)
    ax.set_ylabel("Share of items (%)", fontsize=18, fontweight="bold")
    ax.legend(fontsize=14, ncol=6, loc="upper right")
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig, output_dir, name)


def plot_f_grade_breakdown(df_items: pd.DataFrame, df_runs: pd.DataFrame, output_dir: Path):
    """Breakdown of F grades: unverifiable, judge_error, no code, other."""
    f_items = df_items[df_items["grade"] == "F"].copy()
    if f_items.empty:
        print("  Skipping f_grade_breakdown: no F grades")
        return

    # Build no-code lookup from df_runs
    no_code_runs = set(
        df_runs.loc[df_runs["n_code_files"] == 0, ["paper_slug", "approach"]]
        .apply(tuple, axis=1)
    )

    def categorize(row):
        if row["judge_error"]:
            return "Judge Error"
        if row["unverifiable"]:
            return "Unverifiable"
        if (row["paper_slug"], row["approach"]) in no_code_runs:
            return "No Code"
        return "Other F"

    f_items["f_category"] = f_items.apply(categorize, axis=1)

    ct = pd.crosstab(f_items["approach"], f_items["f_category"])
    cat_order = ["No Code", "Unverifiable", "Judge Error", "Other F"]
    ct = ct.reindex(columns=[c for c in cat_order if c in ct.columns], fill_value=0)
    cat_colors = {"No Code": "#e74c3c", "Unverifiable": "#e67e22", "Judge Error": "#9b59b6", "Other F": "#7f8c8d"}

    fig, ax = plt.subplots(figsize=(10, 6))
    ct.loc[[a for a in APPROACH_ORDER if a in ct.index]].plot(
        kind="bar", stacked=True, ax=ax,
        color=[cat_colors.get(c, "#95a5a6") for c in ct.columns],
        edgecolor="white", width=0.7
    )
    ax.set_xticklabels([APPROACH_LABELS.get(a, a) for a in APPROACH_ORDER if a in ct.index], rotation=0)
    ax.set_ylabel("Number of F-graded items", fontsize=18, fontweight="bold")
    ax.legend(fontsize=14)
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig, output_dir, "f_grade_breakdown")


def plot_f_grade_table_vs_figure(df_items: pd.DataFrame, df_runs: pd.DataFrame, output_dir: Path):
    """F-grade breakdown comparing tables vs figures, by approach."""
    f_items = df_items[df_items["grade"] == "F"].copy()
    if f_items.empty:
        print("  Skipping f_grade_table_vs_figure: no F grades")
        return

    # Only tables and figures
    f_items = f_items[f_items["item_type"].isin(["table", "figure"])]

    no_code_runs = set(
        df_runs.loc[df_runs["n_code_files"] == 0, ["paper_slug", "approach"]]
        .apply(tuple, axis=1)
    )

    def categorize(row):
        if row["judge_error"]:
            return "Judge Error"
        if row["unverifiable"]:
            return "Unverifiable"
        if (row["paper_slug"], row["approach"]) in no_code_runs:
            return "No Code"
        return "Other F"

    f_items["f_category"] = f_items.apply(categorize, axis=1)
    cat_order = ["No Code", "Unverifiable", "Judge Error", "Other F"]
    cat_colors = {"No Code": "#e74c3c", "Unverifiable": "#e67e22", "Judge Error": "#9b59b6", "Other F": "#7f8c8d"}

    approaches = [a for a in APPROACH_ORDER if a in f_items["approach"].values]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, itype in zip(axes, ["table", "figure"]):
        sub = f_items[f_items["item_type"] == itype]
        if sub.empty:
            ax.set_visible(False)
            continue

        ct = pd.crosstab(sub["approach"], sub["f_category"])
        ct = ct.reindex(columns=[c for c in cat_order if c in ct.columns], fill_value=0)
        ct = ct.reindex([a for a in approaches if a in ct.index])

        ct.plot(kind="bar", stacked=True, ax=ax,
                color=[cat_colors.get(c, "#95a5a6") for c in ct.columns],
                edgecolor="white", width=0.7, legend=False)
        ax.set_xticklabels([APPROACH_LABELS.get(a, a) for a in ct.index], rotation=0, fontsize=12)
        ax.set_xlabel(f"{itype.capitalize()}s", fontsize=18, fontweight="bold")
        apply_style(ax)

    axes[0].set_ylabel("Number of F-graded items", fontsize=18, fontweight="bold")
    # Single legend
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=14, loc="upper right", bbox_to_anchor=(0.98, 0.98))
    plt.tight_layout(rect=[0, 0, 0.98, 1])
    save_figure(fig, output_dir, "f_grade_table_vs_figure")


def plot_cell_count_distribution(df_cells: pd.DataFrame, output_dir: Path):
    """Distribution of the number of numeric cells compared per table, by approach."""
    if df_cells.empty:
        print("  Skipping cell_count_distribution: no cell data")
        return

    # Count numeric cells per (paper, approach, item)
    df_num = df_cells[df_cells["is_numeric"]]
    cell_counts = df_num.groupby(["paper_slug", "approach", "item_id"]).size().reset_index(name="n_cells")
    approaches = [a for a in APPROACH_ORDER if a in cell_counts["approach"].values]

    fig, ax = plt.subplots(figsize=(10, 6))
    data = []
    labels = []
    colors = []
    for a in approaches:
        vals = cell_counts.loc[cell_counts["approach"] == a, "n_cells"].values
        if len(vals) > 0:
            data.append(vals)
            labels.append(APPROACH_LABELS.get(a, a))
            colors.append(APPROACH_COLORS[a])

    if not data:
        return

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2)

    for i, (vals, c) in enumerate(zip(data, colors)):
        jitter = np.random.normal(0, 0.04, size=len(vals))
        ax.scatter(np.full(len(vals), i + 1) + jitter, vals,
                   alpha=0.3, s=15, color=c, zorder=3)

    ax.set_ylabel("Number of cells per table", fontsize=18, fontweight="bold")
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig, output_dir, "cell_count_distribution")


def plot_cell_grade_distribution(df_cells: pd.DataFrame, output_dir: Path):
    """Grade distribution at the individual cell level (numeric cells only), by approach."""
    if df_cells.empty:
        print("  Skipping cell_grade_distribution: no cell data")
        return

    df = df_cells[df_cells["cell_grade"].isin(GRADE_ORDER) & df_cells["is_numeric"]].copy()
    if df.empty:
        return

    approaches = [a for a in APPROACH_ORDER if a in df["approach"].values]
    ct = pd.crosstab(df["approach"], df["cell_grade"], normalize="index") * 100
    ct = ct.reindex(columns=GRADE_ORDER, fill_value=0)
    ct = ct.reindex([a for a in approaches if a in ct.index])

    fig, ax = plt.subplots(figsize=(10, 6))
    ct.plot(kind="bar", stacked=True, ax=ax,
            color=[GRADE_COLORS[g] for g in ct.columns], edgecolor="white", width=0.7)
    ax.set_xticklabels([APPROACH_LABELS.get(a, a) for a in ct.index], rotation=0, fontsize=14)
    ax.set_ylabel("Share of cells (%)", fontsize=18, fontweight="bold")
    ax.set_xlabel("")
    ax.legend(fontsize=14, ncol=6, loc="upper right")
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig, output_dir, "cell_grade_distribution")


def plot_cell_pct_difference(df_cells: pd.DataFrame, output_dir: Path):
    """Distribution of percent difference at the cell level, by approach."""
    df = df_cells[df_cells["percent_difference"].notna()].copy()
    if df.empty:
        print("  Skipping cell_pct_difference: no percent difference data")
        return

    # Cap at 200% for readability
    df["pct_capped"] = df["percent_difference"].clip(upper=200)
    approaches = [a for a in APPROACH_ORDER if a in df["approach"].values]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: boxplot by approach
    ax = axes[0]
    data = []
    labels = []
    colors = []
    for a in approaches:
        vals = df.loc[df["approach"] == a, "pct_capped"].values
        if len(vals) > 0:
            data.append(vals)
            labels.append(APPROACH_LABELS.get(a, a))
            colors.append(APPROACH_COLORS[a])

    if data:
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(2)
    ax.set_ylabel("Percent difference (capped at 200%)", fontsize=16, fontweight="bold")
    apply_style(ax)

    # Right: histogram of pct_diff buckets
    ax2 = axes[1]
    bins = [0, 1, 5, 10, 25, 50, 100, 200, float("inf")]
    bin_labels = ["<1%", "1-5%", "5-10%", "10-25%", "25-50%", "50-100%", "100-200%", ">200%"]
    df["pct_bin"] = pd.cut(df["percent_difference"], bins=bins, labels=bin_labels, right=True)

    ct = pd.crosstab(df["pct_bin"], df["approach"], normalize="columns") * 100
    ct = ct.reindex(columns=[a for a in approaches if a in ct.columns])

    x = np.arange(len(bin_labels))
    width = 0.8 / len(approaches)
    for i, a in enumerate(approaches):
        if a not in ct.columns:
            continue
        ax2.bar(x + i * width, ct[a].values, width, label=APPROACH_LABELS.get(a, a),
                color=APPROACH_COLORS[a], alpha=0.8, edgecolor="white")

    ax2.set_xticks(x + width * (len(approaches) - 1) / 2)
    ax2.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=11)
    ax2.set_ylabel("Share of cells (%)", fontsize=16, fontweight="bold")
    ax2.legend(fontsize=11)
    apply_style(ax2)

    plt.tight_layout()
    save_figure(fig, output_dir, "cell_pct_difference")


def plot_extractor_cells(df_cells: pd.DataFrame, output_dir: Path):
    """Extractor output: numeric cells per table distribution + replicator fill rate."""
    if df_cells.empty:
        print("  Skipping extractor_cells: no cell data")
        return

    # Filter to numeric cells only
    df_num = df_cells[df_cells["is_numeric"]].copy()
    if df_num.empty:
        print("  Skipping extractor_cells: no numeric cells")
        return

    # Compute per-table stats: extractor cells (original_value not null) and replicator cells
    table_stats = df_num.groupby(["paper_slug", "approach", "item_id"]).apply(
        lambda g: pd.Series({
            "n_extractor": g["original_value"].notna().sum(),
            "n_replicator": g["replicated_value"].notna().sum(),
        })
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2, 1]})

    # Left: extractor numeric cells per table distribution (histogram)
    ax = axes[0]
    extractor_vals = table_stats["n_extractor"].values
    extractor_vals = extractor_vals[extractor_vals > 0]
    ax.hist(extractor_vals, bins=30, color="#3498db", edgecolor="white", alpha=0.8)
    ax.axvline(np.median(extractor_vals), color="black", linestyle="--", linewidth=2,
               label=f"Median: {np.median(extractor_vals):.0f}")
    ax.set_xlabel("Numeric cells per table (from extractor)", fontsize=18, fontweight="bold")
    ax.set_ylabel("Number of tables", fontsize=18, fontweight="bold")
    ax.legend(fontsize=14)
    apply_style(ax)

    # Right: total extractor cells vs replicator-filled cells, by approach
    ax2 = axes[1]
    approaches = [a for a in APPROACH_ORDER if a in table_stats["approach"].values]
    x = np.arange(len(approaches))
    width = 0.35

    ext_totals = [table_stats.loc[table_stats["approach"] == a, "n_extractor"].sum() for a in approaches]
    rep_totals = [table_stats.loc[table_stats["approach"] == a, "n_replicator"].sum() for a in approaches]

    bars1 = ax2.bar(x - width / 2, ext_totals, width, label="Extractor (original)",
                    color="#3498db", alpha=0.8, edgecolor="white")
    bars2 = ax2.bar(x + width / 2, rep_totals, width, label="Replicator (filled)",
                    color="#e67e22", alpha=0.8, edgecolor="white")

    # Add fill rate labels
    for i, (e, r) in enumerate(zip(ext_totals, rep_totals)):
        pct = r / e * 100 if e > 0 else 0
        ax2.annotate(f"{pct:.0f}%", xy=(x[i] + width / 2, r), xytext=(0, 5),
                     textcoords="offset points", ha="center", fontsize=11, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels([APPROACH_LABELS.get(a, a) for a in approaches], fontsize=11, rotation=15, ha="right")
    ax2.set_ylabel("Total cells", fontsize=18, fontweight="bold")
    ax2.legend(fontsize=12)
    apply_style(ax2)

    plt.tight_layout()
    save_figure(fig, output_dir, "extractor_cells")


# ============================================================================
# Section 2: Determinants of Performance
# ============================================================================

def plot_item_number_vs_grade(df_items: pd.DataFrame, output_dir: Path):
    """Mean grade by item number, faceted by table/figure."""
    df = df_items[df_items["item_number"].notna() & df_items["item_type"].isin(["table", "figure"])].copy()
    df["item_number"] = df["item_number"].astype(int)
    # Cap at item 10 for readability
    df = df[df["item_number"] <= 10]

    if df.empty:
        print("  Skipping item_number_vs_grade: no data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, itype in zip(axes, ["table", "figure"]):
        sub = df[df["item_type"] == itype]
        if sub.empty:
            ax.set_visible(False)
            continue

        for approach in APPROACH_ORDER:
            asub = sub[sub["approach"] == approach]
            if asub.empty:
                continue
            grouped = asub.groupby("item_number")["grade_num"]
            means = grouped.mean()
            sems = grouped.sem()
            ax.errorbar(means.index, means.values, yerr=1.96 * sems.fillna(0).values,
                        label=APPROACH_LABELS[approach], color=APPROACH_COLORS[approach],
                        marker="o", capsize=3, linewidth=2)

        ax.set_xlabel(f"{itype.title()} Number", fontsize=18, fontweight="bold")
        ax.set_yticks(range(6))
        ax.set_yticklabels(GRADE_ORDER[::-1])
        apply_style(ax)

    axes[0].set_ylabel("Mean Grade", fontsize=18, fontweight="bold")
    axes[1].legend(fontsize=14, loc="lower left")
    plt.tight_layout()
    save_figure(fig, output_dir, "item_number_vs_grade")


def plot_scatter_vs_grade(df: pd.DataFrame, x_col: str, x_label: str, output_dir: Path,
                          name: str, log_x: bool = False, grade_col: str = "overall_grade_num"):
    """Generic scatter plot of x_col vs grade, colored by approach."""
    df_plot = df.dropna(subset=[x_col, grade_col])
    if df_plot.empty:
        print(f"  Skipping {name}: no data")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    for approach in APPROACH_ORDER:
        sub = df_plot[df_plot["approach"] == approach]
        if sub.empty:
            continue
        ax.scatter(sub[x_col], sub[grade_col] + np.random.uniform(-0.15, 0.15, len(sub)),
                   color=APPROACH_COLORS[approach], label=APPROACH_LABELS[approach],
                   alpha=0.6, s=60, edgecolor="white", linewidth=0.5)

    if log_x and df_plot[x_col].gt(0).any():
        ax.set_xscale("log")
    ax.set_xlabel(x_label, fontsize=18, fontweight="bold")
    ax.set_ylabel("Overall Grade", fontsize=18, fontweight="bold")
    ax.set_yticks(range(6))
    ax.set_yticklabels(GRADE_ORDER[::-1])
    ax.legend(fontsize=14)
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig, output_dir, name)


def plot_duration_vs_grade(df_runs: pd.DataFrame, output_dir: Path):
    """Scatter + box: duration vs overall grade, by approach."""
    df = df_runs.dropna(subset=["duration_seconds", "overall_grade_num"])
    if df.empty:
        print("  Skipping duration_vs_grade: no data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [2, 1]})

    # Scatter
    ax = axes[0]
    for approach in APPROACH_ORDER:
        sub = df[df["approach"] == approach]
        if sub.empty:
            continue
        ax.scatter(sub["duration_seconds"] / 60, sub["overall_grade_num"] +
                   np.random.uniform(-0.15, 0.15, len(sub)),
                   color=APPROACH_COLORS[approach], label=APPROACH_LABELS[approach],
                   alpha=0.6, s=60, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Duration (minutes)", fontsize=18, fontweight="bold")
    ax.set_ylabel("Overall Grade", fontsize=18, fontweight="bold")
    ax.set_yticks(range(6))
    ax.set_yticklabels(GRADE_ORDER[::-1])
    ax.legend(fontsize=14)
    apply_style(ax)

    # Box by grade
    ax2 = axes[1]
    grade_data = [df[df["overall_grade"] == g]["duration_seconds"].dropna() / 60 for g in GRADE_ORDER]
    bp = ax2.boxplot(grade_data, tick_labels=GRADE_ORDER, patch_artist=True, widths=0.6)
    for patch, grade in zip(bp["boxes"], GRADE_ORDER):
        patch.set_facecolor(GRADE_COLORS[grade])
        patch.set_alpha(0.7)
    ax2.set_xlabel("Grade", fontsize=18, fontweight="bold")
    ax2.set_ylabel("Duration (minutes)", fontsize=18, fontweight="bold")
    apply_style(ax2)

    plt.tight_layout()
    save_figure(fig, output_dir, "duration_vs_grade")


def plot_cost_distribution(df_runs: pd.DataFrame, output_dir: Path):
    """Distribution of cost per run by approach (approaches with cost data only)."""
    df = df_runs[df_runs["total_cost_usd"] > 0].copy()
    if df.empty:
        print("  Skipping cost_distribution: no cost data")
        return

    approaches = [a for a in APPROACH_ORDER if a in df["approach"].values]
    fig, ax = plt.subplots(figsize=(10, 6))

    data = []
    labels = []
    colors = []
    for a in approaches:
        vals = df.loc[df["approach"] == a, "total_cost_usd"].dropna()
        if not vals.empty:
            data.append(vals.values)
            labels.append(APPROACH_LABELS.get(a, a))
            colors.append(APPROACH_COLORS[a])

    if not data:
        print("  Skipping cost_distribution: no data after filtering")
        return

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2)

    for i, (vals, c) in enumerate(zip(data, colors)):
        jitter = np.random.normal(0, 0.04, size=len(vals))
        ax.scatter(np.full(len(vals), i + 1) + jitter, vals,
                   alpha=0.3, s=20, color=c, zorder=3)

    # Add median labels
    for i, vals in enumerate(data):
        med = np.median(vals)
        ax.annotate(f"${med:.2f}", xy=(i + 1, med), xytext=(i + 1.3, med),
                    fontsize=11, color="black", fontweight="bold", va="center")

    ax.set_ylabel("Cost per run (USD)", fontsize=18, fontweight="bold")
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig, output_dir, "cost_distribution")


def plot_token_usage(df_runs: pd.DataFrame, output_dir: Path):
    """Distribution of input and output tokens (approaches with token data only)."""
    df = df_runs[df_runs["total_tokens"] > 0].copy()
    if df.empty:
        print("  Skipping token_usage: no token data")
        return

    approaches = [a for a in APPROACH_ORDER if a in df["approach"].values]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, col, label in zip(axes, ["prompt_tokens", "completion_tokens"],
                               ["Input Tokens", "Output Tokens"]):
        data = []
        labels = []
        colors = []
        for a in approaches:
            vals = df.loc[df["approach"] == a, col].dropna()
            if not vals.empty:
                data.append(vals.values / 1000)  # in thousands
                labels.append(APPROACH_LABELS.get(a, a))
                colors.append(APPROACH_COLORS[a])

        if not data:
            ax.set_visible(False)
            continue

        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(2)

        for i, (vals, c) in enumerate(zip(data, colors)):
            jitter = np.random.normal(0, 0.04, size=len(vals))
            ax.scatter(np.full(len(vals), i + 1) + jitter, vals,
                       alpha=0.3, s=15, color=c, zorder=3)

        ax.set_ylabel(f"{label} (thousands)", fontsize=18, fontweight="bold")
        ax.tick_params(labelsize=12)
        apply_style(ax)

    plt.tight_layout()
    save_figure(fig, output_dir, "token_usage")


def plot_duration_distribution(df_runs: pd.DataFrame, output_dir: Path):
    """Distribution of run duration by approach (non-skipped runs only)."""
    df = df_runs[df_runs["duration_seconds"].notna() & (df_runs["duration_seconds"] > 0)].copy()
    if df.empty:
        print("  Skipping duration_distribution: no duration data")
        return

    approaches = [a for a in APPROACH_ORDER if a in df["approach"].values]
    fig, ax = plt.subplots(figsize=(10, 6))

    data = []
    labels = []
    colors = []
    for a in approaches:
        vals = df.loc[df["approach"] == a, "duration_seconds"].dropna() / 60  # minutes
        if not vals.empty:
            data.append(vals.values)
            labels.append(APPROACH_LABELS.get(a, a))
            colors.append(APPROACH_COLORS[a])

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2)

    # Overlay individual points
    for i, (vals, c) in enumerate(zip(data, colors)):
        jitter = np.random.normal(0, 0.04, size=len(vals))
        ax.scatter(np.full(len(vals), i + 1) + jitter, vals,
                   alpha=0.3, s=20, color=c, zorder=3)

    # Add median labels
    for i, vals in enumerate(data):
        med = np.median(vals)
        ax.annotate(f"{med:.0f}m", xy=(i + 1, med), xytext=(i + 1.3, med),
                    fontsize=11, color="black", fontweight="bold", va="center")

    ax.set_ylabel("Duration (minutes)", fontsize=18, fontweight="bold")
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig, output_dir, "duration_distribution")


def plot_n_datasets_vs_grade(df_runs: pd.DataFrame, output_dir: Path):
    """Box plot of grade by binned dataset count."""
    df = df_runs[df_runs["n_datasets"] > 0].copy()
    if df.empty:
        print("  Skipping n_datasets_vs_grade: no data")
        return

    bins = [0, 3, 10, 30, 100, float("inf")]
    labels = ["1-3", "4-10", "11-30", "31-100", "100+"]
    df["dataset_bin"] = pd.cut(df["n_datasets"], bins=bins, labels=labels, right=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    order = [l for l in labels if l in df["dataset_bin"].cat.categories]
    sns.boxplot(data=df, x="dataset_bin", y="overall_grade_num", order=order,
                hue="approach", hue_order=APPROACH_ORDER,
                palette=APPROACH_COLORS, ax=ax, fliersize=3)
    ax.set_xlabel("Number of Data Files", fontsize=18, fontweight="bold")
    ax.set_ylabel("Overall Grade", fontsize=18, fontweight="bold")
    ax.set_yticks(range(6))
    ax.set_yticklabels(GRADE_ORDER[::-1])
    ax.legend(fontsize=12, title="Approach")
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig, output_dir, "n_datasets_vs_grade")


# ============================================================================
# Section 3: Explainer Analysis
# ============================================================================

def plot_fault_attribution(df_items: pd.DataFrame, output_dir: Path):
    """Stacked bar of fault attribution by approach.

    NOTE: The current explainer pipeline sets all attributions to 'unclear'.
    This plot will become useful once the explainer is improved to produce
    real attributions. For now, we derive a proxy from the discrepancy
    descriptions by keyword matching.
    """
    df = df_items[df_items["description_of_discrepancy"].str.len() > 0].copy()
    if df.empty:
        print("  Skipping fault_attribution: no discrepancy descriptions")
        return

    # Derive proxy attribution from discrepancy text
    def classify_discrepancy(text):
        text = text.lower()
        if any(k in text for k in ["cli exit_code", "did not produce", "no output", "crash", "timeout", "failed to run"]):
            return "execution_failure"
        if any(k in text for k in ["scale mismatch", "scale factor", "percentage vs decimal", "units"]):
            return "scale_mismatch"
        if any(k in text for k in ["missing row", "missing col", "missing panel", "not found in replicated"]):
            return "missing_output"
        if any(k in text for k in ["sign mismatch", "opposite sign", "different sign"]):
            return "sign_difference"
        if any(k in text for k in ["within 5%", "within 2%", "small difference", "close match"]):
            return "minor_difference"
        return "other_discrepancy"

    df["proxy_attribution"] = df["description_of_discrepancy"].apply(classify_discrepancy)

    ct = pd.crosstab(df["approach"], df["proxy_attribution"], normalize="index") * 100
    attr_order = ["execution_failure", "missing_output", "sign_difference", "scale_mismatch",
                   "minor_difference", "other_discrepancy"]
    ct = ct.reindex(columns=[c for c in attr_order if c in ct.columns], fill_value=0)
    attr_colors = {
        "execution_failure": "#e74c3c",
        "missing_output": "#e67e22",
        "sign_difference": "#9b59b6",
        "scale_mismatch": "#f39c12",
        "minor_difference": "#2ecc71",
        "other_discrepancy": "#95a5a6",
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    ct.loc[[a for a in APPROACH_ORDER if a in ct.index]].plot(
        kind="bar", stacked=True, ax=ax,
        color=[attr_colors.get(c, "#bdc3c7") for c in ct.columns],
        edgecolor="white", width=0.7
    )
    ax.set_xticklabels([APPROACH_LABELS.get(a, a) for a in APPROACH_ORDER if a in ct.index], rotation=0)
    ax.set_ylabel("Share (%)", fontsize=18, fontweight="bold")
    ax.legend(fontsize=12, title="Discrepancy Type", loc="upper right")
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig, output_dir, "fault_attribution")


def plot_likely_causes_frequency(df_items: pd.DataFrame, output_dir: Path):
    """Extract key terms from discrepancy descriptions and show frequency."""
    df = df_items[df_items["description_of_discrepancy"].str.len() > 10].copy()
    if df.empty:
        print("  Skipping likely_causes_replicator: no discrepancy data")
        return

    # Define meaningful term categories to search for in descriptions
    term_categories = {
        "scale/unit mismatch": r"scale[ _](?:factor|mismatch)|percentage|decimal|units?\b",
        "sign mismatch": r"sign[ _]mismatch|opposite sign|different sign|sign_match.*false",
        "missing rows/columns": r"missing (?:row|col|panel|variable)",
        "CLI/execution failure": r"cli exit_code|did not produce|crash|failed to (?:run|execute)",
        "row/column alignment": r"row(?:s)? (?:matched|aligned)|column(?:s)? (?:matched|aligned)",
        "standard error mismatch": r"(?:standard error|se |s\.e\.).*(?:mismatch|differ)",
        "panel structure issue": r"panel [a-z]|two.panel|multi.panel",
        "R-squared difference": r"r.squared|r²|r-squared",
        "sample size difference": r"(?:sample size|observations|n =).*differ",
        "coefficient difference": r"coefficient.*differ|different (?:coefficient|estimate)",
        "large % difference": r"(?:\d{2,3})% difference|>(?:20|50)%",
        "small % difference": r"within (?:2|5)%|<(?:5|10)%",
        "duplicate/identical values": r"identical values|duplicate|same values",
        "data processing difference": r"(?:data processing|cleaning|filtering).*differ",
        "regression specification": r"(?:specification|regression|model).*differ|different (?:model|spec)",
        "figure comparison": r"figure|plot|graph|visual",
        "timeout/incomplete": r"timeout|timed out|incomplete",
    }

    all_texts = " ".join(df["description_of_discrepancy"].str.lower())
    counts = {}
    for label, pattern in term_categories.items():
        n = len(re.findall(pattern, all_texts))
        if n > 0:
            counts[label] = n

    if not counts:
        print("  Skipping likely_causes_replicator: no term matches")
        return

    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    labels, values = zip(*sorted_counts)

    fig, ax = plt.subplots(figsize=(12, max(6, len(labels) * 0.4)))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color="#e74c3c", alpha=0.8, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("Frequency in discrepancy descriptions", fontsize=18, fontweight="bold")
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig, output_dir, "likely_causes_replicator")


def plot_no_code_analysis(df_runs: pd.DataFrame, output_dir: Path):
    """Analyze why some runs produced no code."""
    no_code = df_runs[df_runs["n_code_files"] == 0].copy()
    if no_code.empty:
        print("  Skipping no_code_analysis: all runs produced code")
        return

    # Categorize by duration
    def categorize_failure(row):
        dur = row.get("duration_seconds") or 0
        if dur == 0:
            return "Runner crash (0s)"
        elif dur < 30:
            return "Quick failure (<30s)"
        elif dur < 300:
            return "Early exit (<5min)"
        else:
            return "Ran but no output"

    no_code["failure_type"] = no_code.apply(categorize_failure, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: count by approach
    ax = axes[0]
    ct = pd.crosstab(no_code["approach"], no_code["failure_type"])
    type_order = ["Runner crash (0s)", "Quick failure (<30s)", "Early exit (<5min)", "Ran but no output"]
    ct = ct.reindex(columns=[c for c in type_order if c in ct.columns], fill_value=0)
    type_colors = {"Runner crash (0s)": "#2c3e50", "Quick failure (<30s)": "#e74c3c",
                   "Early exit (<5min)": "#e67e22", "Ran but no output": "#f39c12"}
    ct.loc[[a for a in APPROACH_ORDER if a in ct.index]].plot(
        kind="bar", stacked=True, ax=ax,
        color=[type_colors.get(c, "#95a5a6") for c in ct.columns],
        edgecolor="white", width=0.7
    )
    ax.set_xticklabels([APPROACH_LABELS.get(a, a) for a in APPROACH_ORDER if a in ct.index], rotation=0)
    ax.set_ylabel("Number of no-code runs", fontsize=18, fontweight="bold")
    ax.legend(fontsize=11, title="Failure Type")
    apply_style(ax)

    # Right: total runs vs no-code runs by approach
    ax2 = axes[1]
    total_by_approach = df_runs.groupby("approach").size()
    nocode_by_approach = no_code.groupby("approach").size()
    x = np.arange(len(APPROACH_ORDER))
    width = 0.35
    totals = [total_by_approach.get(a, 0) for a in APPROACH_ORDER]
    nocodes = [nocode_by_approach.get(a, 0) for a in APPROACH_ORDER]
    ax2.bar(x - width/2, totals, width, label="Total runs", color="#3498db", alpha=0.7, edgecolor="white")
    ax2.bar(x + width/2, nocodes, width, label="No code produced", color="#e74c3c", alpha=0.7, edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels([APPROACH_LABELS[a] for a in APPROACH_ORDER], rotation=0)
    ax2.set_ylabel("Count", fontsize=18, fontweight="bold")
    ax2.legend(fontsize=14)
    # Add % labels
    for i, (t, nc) in enumerate(zip(totals, nocodes)):
        if t > 0:
            ax2.text(i + width/2, nc + 0.3, f"{nc/t*100:.0f}%", ha="center", fontsize=12, fontweight="bold")
    apply_style(ax2)

    plt.tight_layout()
    save_figure(fig, output_dir, "no_code_analysis")


# ============================================================================
# Section 4: Additional Analyses
# ============================================================================

def plot_agreement_matrix(df_items: pd.DataFrame, output_dir: Path):
    """Heatmap: pairwise grade agreement rate between approaches."""
    # Build pivot with plain float values (not categorical)
    deduped = df_items.drop_duplicates(subset=["paper_slug", "item_id", "approach"])
    pivot_data = deduped[["paper_slug", "item_id", "approach", "grade_num"]].copy()
    pivot_data["approach"] = pivot_data["approach"].astype(str)
    pivot = pivot_data.pivot_table(index=["paper_slug", "item_id"], columns="approach",
                                   values="grade_num", aggfunc="first")
    approaches = [a for a in APPROACH_ORDER if a in pivot.columns]
    if len(approaches) < 2:
        print("  Skipping agreement_matrix: need >= 2 approaches")
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
            a_clean = a_vals[mask]
            b_clean = b_vals[mask]
            agreement[i, j] = np.mean(a_clean == b_clean) * 100
            within_one[i, j] = np.mean(np.abs(a_clean - b_clean) <= 1) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    labels = [APPROACH_LABELS[a] for a in approaches]

    for ax, data, subtitle in zip(axes, [agreement, within_one], ["Exact Match (%)", "Within 1 Grade (%)"]):
        sns.heatmap(data, annot=True, fmt=".0f", xticklabels=labels, yticklabels=labels,
                    cmap="YlGn", vmin=0, vmax=100, ax=ax, square=True,
                    cbar_kws={"shrink": 0.8})
        ax.set_xlabel(subtitle, fontsize=16, fontweight="bold")
        ax.tick_params(labelsize=12)

    plt.tight_layout()
    save_figure(fig, output_dir, "agreement_matrix")


def plot_best_of_k(df_runs: pd.DataFrame, output_dir: Path):
    """Best grade per paper for k=1..4 approaches."""
    approaches = [a for a in APPROACH_ORDER if a in df_runs["approach"].values]
    if not approaches:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(GRADE_ORDER))
    width = 0.18

    # Individual approaches
    for i, approach in enumerate(approaches):
        sub = df_runs[df_runs["approach"] == approach]
        dist = sub["overall_grade"].value_counts(normalize=True).reindex(GRADE_ORDER, fill_value=0) * 100
        ax.bar(x + i * width, dist.values, width, label=APPROACH_LABELS[approach],
               color=APPROACH_COLORS[approach], alpha=0.5, edgecolor="white")

    # Best-of-all
    best = df_runs.groupby("paper_slug")["overall_grade_num"].max().map(
        lambda v: NUM_TO_GRADE.get(int(v), "F") if pd.notna(v) else "F"
    )
    best_dist = best.value_counts(normalize=True).reindex(GRADE_ORDER, fill_value=0) * 100
    ax.bar(x + len(approaches) * width, best_dist.values, width, label="Best of All",
           color="#2c3e50", edgecolor="white")

    ax.set_xticks(x + width * len(approaches) / 2)
    ax.set_xticklabels(GRADE_ORDER)
    ax.set_xlabel("Grade", fontsize=18, fontweight="bold")
    ax.set_ylabel("Share of papers (%)", fontsize=18, fontweight="bold")
    ax.legend(fontsize=12)
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig, output_dir, "best_of_k")


def plot_paper_difficulty(df_runs: pd.DataFrame, output_dir: Path):
    """Horizontal bar: papers ranked by mean grade across approaches, with grade range."""
    # Compute mean, min, max grade per paper
    agg = df_runs.groupby(["paper_slug", "paper_title"])["overall_grade_num"].agg(["mean", "min", "max"])
    agg = agg.sort_values("mean", ascending=True)

    if len(agg) > 40:
        hardest = agg.head(20)
        easiest = agg.tail(20)
        agg = pd.concat([hardest, easiest])

    # Build labels: "JOURNAL — Title" (truncated)
    labels = []
    for slug, title in agg.index:
        journal = _infer_journal(slug)
        display = title[:45] if title != slug else slug
        if journal != "Other":
            labels.append(f"{journal} — {display}")
        else:
            labels.append(display)

    fig, ax = plt.subplots(figsize=(12, max(8, len(agg) * 0.35)))
    y_pos = range(len(agg))
    colors = [GRADE_COLORS.get(NUM_TO_GRADE.get(round(v), "F"), "#95a5a6") for v in agg["mean"].values]

    # Draw mean bars
    ax.barh(y_pos, agg["mean"].values, color=colors, edgecolor="white", alpha=0.8)
    # Draw grade range (min-max) as horizontal error bars
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
    save_figure(fig, output_dir, "paper_difficulty")


def plot_pairwise_dominance(df_runs: pd.DataFrame, output_dir: Path):
    """Heatmap: approach A beats B on X% of papers."""
    pivot_data = df_runs[["paper_slug", "approach", "overall_grade_num"]].copy()
    pivot_data["approach"] = pivot_data["approach"].astype(str)
    pivot = pivot_data.pivot_table(index="paper_slug", columns="approach",
                                   values="overall_grade_num", aggfunc="first")
    approaches = [a for a in APPROACH_ORDER if a in pivot.columns]
    if len(approaches) < 2:
        return

    n = len(approaches)
    wins = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                wins[i, j] = np.nan
                continue
            a_vals = pivot[approaches[i]].values
            b_vals = pivot[approaches[j]].values
            mask = ~(np.isnan(a_vals) | np.isnan(b_vals))
            if mask.sum() == 0:
                continue
            wins[i, j] = np.mean(a_vals[mask] > b_vals[mask]) * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    labels = [APPROACH_LABELS[a] for a in approaches]
    mask = np.eye(n, dtype=bool)
    sns.heatmap(wins, annot=True, fmt=".0f", xticklabels=labels, yticklabels=labels,
                cmap="RdYlGn", vmin=0, vmax=100, ax=ax, mask=mask, square=True,
                cbar_kws={"label": "Win Rate (%)", "shrink": 0.8})
    ax.set_xlabel("Column approach", fontsize=16, fontweight="bold")
    ax.set_ylabel("Row approach beats →", fontsize=16, fontweight="bold")
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    save_figure(fig, output_dir, "pairwise_dominance")


def plot_grade_by_journal(df_runs: pd.DataFrame, output_dir: Path):
    """Grade distribution grouped by journal."""
    ct = pd.crosstab(df_runs["journal"], df_runs["overall_grade"], normalize="index") * 100
    ct = ct.reindex(columns=GRADE_ORDER, fill_value=0)
    # Sort by mean grade
    ct["_mean"] = sum(ct[g] * GRADE_TO_NUM[g] for g in GRADE_ORDER if g in ct.columns) / 100
    ct = ct.sort_values("_mean", ascending=False)
    ct = ct.drop(columns="_mean")

    fig, ax = plt.subplots(figsize=(12, 6))
    ct.plot(kind="bar", stacked=True, ax=ax,
            color=[GRADE_COLORS[g] for g in ct.columns], edgecolor="white", width=0.7)
    ax.set_xlabel("Journal", fontsize=18, fontweight="bold")
    ax.set_ylabel("Share (%)", fontsize=18, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(fontsize=12, ncol=6, loc="upper right")
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig, output_dir, "grade_by_journal")


def plot_cumulative_success(df_items: pd.DataFrame, output_dir: Path):
    """Fraction of items graded A or B as items accumulate in order."""
    df = df_items[df_items["item_number"].notna()].copy()
    df["item_number"] = df["item_number"].astype(int)
    df = df.sort_values(["paper_slug", "approach", "item_number"])

    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for approach in APPROACH_ORDER:
        sub = df[df["approach"] == approach].copy()
        if sub.empty:
            continue
        sub["is_good"] = sub["grade_num"] >= 4  # A or B
        # Group by item number position, compute cumulative fraction
        by_num = sub.groupby("item_number")["is_good"].mean() * 100
        ax.plot(by_num.index, by_num.values, marker="o",
                color=APPROACH_COLORS[approach], label=APPROACH_LABELS[approach],
                linewidth=2, markersize=6)

    ax.set_xlabel("Item Number", fontsize=18, fontweight="bold")
    ax.set_ylabel("% Items Graded A or B", fontsize=18, fontweight="bold")
    ax.legend(fontsize=14)
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig, output_dir, "cumulative_success")


# ============================================================================
# Summary Table
# ============================================================================

def generate_summary_table(df_runs: pd.DataFrame, df_items: pd.DataFrame, output_dir: Path):
    """Generate summary table as CSV and LaTeX."""
    rows = []
    for approach in APPROACH_ORDER:
        sub = df_runs[df_runs["approach"] == approach]
        items = df_items[df_items["approach"] == approach]
        if sub.empty:
            continue

        n = len(sub)
        mean_grade = sub["overall_grade_num"].mean()
        pct_ab = ((sub["overall_grade"].isin(["A", "B"])).sum() / n * 100) if n else 0
        pct_f = ((sub["overall_grade"] == "F").sum() / n * 100) if n else 0
        mean_dur = sub["duration_seconds"].mean()
        mean_tokens = sub["total_tokens"].mean()
        n_items_total = len(items)
        item_pct_ab = ((items["grade"].isin(["A", "B"])).sum() / n_items_total * 100) if n_items_total else 0
        item_pct_f = ((items["grade"] == "F").sum() / n_items_total * 100) if n_items_total else 0

        rows.append({
            "Approach": APPROACH_LABELS[approach],
            "Runs": n,
            "Mean Grade": f"{mean_grade:.2f}",
            "% A-B (runs)": f"{pct_ab:.1f}",
            "% F (runs)": f"{pct_f:.1f}",
            "Items": n_items_total,
            "% A-B (items)": f"{item_pct_ab:.1f}",
            "% F (items)": f"{item_pct_f:.1f}",
            "Mean Duration (min)": f"{mean_dur / 60:.1f}" if pd.notna(mean_dur) else "—",
            "Mean Tokens": f"{mean_tokens:,.0f}" if pd.notna(mean_tokens) else "—",
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "summary_table.csv", index=False)

    # LaTeX
    latex = summary.to_latex(index=False, escape=True, column_format="l" + "r" * (len(summary.columns) - 1))
    (output_dir / "summary_table.tex").write_text(latex)
    print("  Saved summary_table")


# ============================================================================
# Plot Index
# ============================================================================

PLOT_INDEX = [
    # (filename, description, data_source)
    # data_source: "runs" = run-level (overall grade), "items:both" = table+figure items,
    #              "items:table" = table items only, "items:figure" = figure items only
    # Section 1
    ("overall_grades", "Overall grade distribution by approach", "runs"),
    ("table_grades", "Grade distribution for table items", "items:table"),
    ("figure_grades", "Grade distribution for figure items", "items:figure"),
    ("item_grades", "Grade distribution for all items (tables + figures)", "items:both"),
    ("f_grade_breakdown", "F-grade subcategories by approach (all items)", "items:both"),
    ("f_grade_table_vs_figure", "F-grade subcategories: tables vs figures side-by-side", "items:both"),
    # Section 1b: Cell-level
    ("cell_count_distribution", "Number of cells compared per table, by approach", "cells"),
    ("cell_grade_distribution", "Grade distribution at individual cell level", "cells"),
    ("cell_pct_difference", "Percent difference distribution at cell level", "cells"),
    ("extractor_cells", "Extractor cells per table + replicator fill rate", "cells"),
    # Section 2
    ("item_number_vs_grade", "Mean grade by item number, faceted by table/figure", "items:both"),
    ("methodology_length_vs_grade", "Methodology summary length vs overall grade", "runs"),
    ("code_length_vs_grade", "Total code size vs overall grade", "runs"),
    ("n_datasets_vs_grade", "Number of datasets vs overall grade", "runs"),
    ("data_size_vs_grade", "Total data size vs overall grade", "runs"),
    ("duration_vs_grade", "Run duration vs overall grade", "runs"),
    ("cost_distribution", "Cost per run by approach (approaches with cost data)", "runs"),
    ("token_usage", "Input and output token distribution (approaches with token data)", "runs"),
    ("duration_distribution", "Distribution of run duration by approach", "runs"),
    # Section 3
    ("fault_attribution", "Proxy fault attribution from discrepancy text", "items:both"),
    ("likely_causes_replicator", "Frequency of discrepancy term categories", "items:both"),
    ("no_code_analysis", "Runs that produced no code, by approach and failure type", "runs"),
    # Section 4
    ("agreement_matrix", "Pairwise grade agreement between approaches", "items:both"),
    ("best_of_k", "Best grade per paper for 1..k approaches", "runs"),
    ("paper_difficulty", "Papers ranked by mean grade with range", "runs"),
    ("tokens_vs_grade", "Total tokens vs overall grade", "runs"),
    ("pairwise_dominance", "Approach A beats B on X% of papers", "runs"),
    ("grade_by_journal", "Grade distribution grouped by journal", "runs"),
    ("cumulative_success", "Fraction of A/B items as items accumulate", "items:both"),
]


def _generate_plot_index(output_dir: Path):
    """Write a markdown index of all plots with data source info."""
    lines = [
        "# i4rep Benchmark Analysis — Plot Index\n",
        "## Data sources\n",
        "- **runs**: One observation per (paper, approach) — uses overall run-level grade",
        "- **items:both**: One observation per (paper, approach, item) — uses tables AND figures",
        "- **items:table**: Table items only",
        "- **items:figure**: Figure items only\n",
    ]

    current_section = None
    sections = {
        0: "Section 1: Performance Distribution",
        6: "Section 1b: Cell-Level Analysis",
        10: "Section 2: Determinants of Performance",
        19: "Section 3: Explainer Analysis",
        22: "Section 4: Additional Analyses",
    }

    for i, (fname, desc, source) in enumerate(PLOT_INDEX):
        if i in sections:
            current_section = sections[i]
            lines.append(f"\n## {current_section}\n")
            lines.append("| Plot | Description | Data |")
            lines.append("|------|-------------|------|")
        lines.append(f"| [{fname}]({fname}.png) | {desc} | `{source}` |")

    lines.append("\n## Summary outputs\n")
    lines.append("| File | Description |")
    lines.append("|------|-------------|")
    lines.append("| [df_runs.csv](df_runs.csv) | Run-level DataFrame |")
    lines.append("| [df_items.csv](df_items.csv) | Item-level DataFrame |")
    lines.append("| [summary_table.csv](summary_table.csv) | Summary statistics by approach |")
    lines.append("| [summary_table.tex](summary_table.tex) | LaTeX version of summary table |")
    lines.append("")

    (output_dir / "README.md").write_text("\n".join(lines))
    print("  Saved README.md")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze i4rep benchmark results")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--papers-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="analysis_output")
    args = parser.parse_args()

    # Auto-detect paths
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

    # Print quick summary
    print(f"\n{'='*60}")
    print(f"Runs:  {len(df_runs)} ({df_runs['paper_slug'].nunique()} papers, "
          f"{df_runs['approach'].nunique()} approaches)")
    print(f"Items: {len(df_items)}")
    for a in APPROACH_ORDER:
        sub = df_runs[df_runs["approach"] == a]
        if sub.empty:
            continue
        print(f"  {APPROACH_LABELS[a]:12s}: {len(sub):3d} runs, "
              f"mean={sub['overall_grade_num'].mean():.2f}, "
              f"A-B={sub['overall_grade'].isin(['A','B']).mean()*100:.0f}%, "
              f"F={sub['overall_grade'].eq('F').mean()*100:.0f}%")
    print(f"{'='*60}\n")

    # Save DataFrames
    df_runs.to_csv(output_dir / "df_runs.csv", index=False)
    df_items.to_csv(output_dir / "df_items.csv", index=False)
    df_cells.to_csv(output_dir / "df_cells.csv", index=False)
    print(f"  Saved df_runs.csv, df_items.csv, df_cells.csv ({len(df_cells)} cells)")

    # Section 1: Performance Distribution
    print("\nSection 1: Performance Distribution")
    plot_overall_grade_distribution(df_runs, output_dir)
    plot_item_grade_by_type(df_items, output_dir, "table", "table_grades")
    plot_item_grade_by_type(df_items, output_dir, "figure", "figure_grades")
    plot_item_grade_by_type(df_items, output_dir, None, "item_grades")
    plot_f_grade_breakdown(df_items, df_runs, output_dir)
    plot_f_grade_table_vs_figure(df_items, df_runs, output_dir)

    # Section 1b: Cell-level analysis
    print("\nSection 1b: Cell-Level Analysis")
    plot_cell_count_distribution(df_cells, output_dir)
    plot_cell_grade_distribution(df_cells, output_dir)
    plot_cell_pct_difference(df_cells, output_dir)
    plot_extractor_cells(df_cells, output_dir)

    # Section 2: Determinants
    print("\nSection 2: Determinants of Performance")
    plot_item_number_vs_grade(df_items, output_dir)
    plot_scatter_vs_grade(df_runs, "methodology_summary_len", "Methodology Summary Length (chars)",
                          output_dir, "methodology_length_vs_grade")
    plot_scatter_vs_grade(df_runs, "total_code_chars", "Total Code Size (chars)",
                          output_dir, "code_length_vs_grade", log_x=True)
    plot_n_datasets_vs_grade(df_runs, output_dir)
    plot_scatter_vs_grade(df_runs, "total_data_size_bytes", "Total Data Size (bytes)",
                          output_dir, "data_size_vs_grade", log_x=True)
    plot_duration_vs_grade(df_runs, output_dir)
    plot_cost_distribution(df_runs, output_dir)
    plot_token_usage(df_runs, output_dir)
    plot_duration_distribution(df_runs, output_dir)

    # Section 3: Explainer
    print("\nSection 3: Explainer Analysis")
    plot_fault_attribution(df_items, output_dir)
    plot_likely_causes_frequency(df_items, output_dir)
    plot_no_code_analysis(df_runs, output_dir)

    # Section 4: Additional
    print("\nSection 4: Additional Analyses")
    plot_agreement_matrix(df_items, output_dir)
    plot_best_of_k(df_runs, output_dir)
    plot_paper_difficulty(df_runs, output_dir)
    plot_scatter_vs_grade(df_runs, "total_tokens", "Total Tokens",
                          output_dir, "tokens_vs_grade", log_x=True)
    plot_pairwise_dominance(df_runs, output_dir)
    plot_grade_by_journal(df_runs, output_dir)
    plot_cumulative_success(df_items, output_dir)

    # Summary
    print("\nSummary Table")
    generate_summary_table(df_runs, df_items, output_dir)

    # Plot index markdown
    print("\nGenerating plot index")
    _generate_plot_index(output_dir)

    print(f"\nDone! All outputs in {output_dir}/")


if __name__ == "__main__":
    main()
