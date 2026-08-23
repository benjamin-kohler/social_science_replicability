#!/usr/bin/env python3
"""Compare pre-cutoff (i4rep) vs post-cutoff table grade distributions.

Inherits styles from analyze_i4rep_results.py.

Usage:
    python scripts/analyze_pre_post_cutoff.py
    python scripts/analyze_pre_post_cutoff.py --output-dir analysis_output/pre_post_cutoff
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Import shared constants and styles
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.analyze_i4rep_results import (
    GRADE_ORDER, GRADE_TO_NUM, GRADE_COLORS,
    APPROACH_MODEL_LABELS, APPROACH_MODEL_COLORS,
    setup_style, apply_style, place_legend, save_figure,
    _load_json, _parse_item_type, _parse_item_number,
)

TEXTLAB_BASE = Path("/data/individual/benjamin/social_science_replicability/data")
LOCAL_BASE = Path("data")

APPROACH_ORDER_RAW = ["claude-code", "codex", "swe-agent", "opencode"]


def load_collection(results_dir: Path, label: str) -> pd.DataFrame:
    """Load item-level grades from a results directory."""
    rows = []
    if not results_dir.exists():
        return pd.DataFrame()

    for paper_dir in sorted(results_dir.iterdir()):
        if not paper_dir.is_dir() or paper_dir.name.startswith("_") or paper_dir.name.startswith("batch"):
            continue
        paper_slug = paper_dir.name

        for run_dir in sorted(paper_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            vr = _load_json(run_dir / "verification_report.json")
            if vr is None:
                continue

            # Parse approach
            dirname = run_dir.name
            idx = dirname.find(f"_{paper_slug}_")
            if idx < 0:
                continue
            model = dirname[:idx]
            approach = dirname[idx + len(f"_{paper_slug}_"):]
            if approach not in APPROACH_ORDER_RAW:
                continue

            approach_key = f"{approach}/{model}"

            for item in vr.get("item_verifications", []):
                item_id = item.get("item_id", "")
                item_type = _parse_item_type(item_id, item.get("item_type"))
                if item_type != "table":
                    continue
                grade = item.get("grade", "F")
                # Skip non-numerical
                notes = item.get("comparison_notes", "")
                if "non-numerical" in notes.lower():
                    continue

                rows.append({
                    "collection": label,
                    "paper_slug": paper_slug,
                    "approach": approach_key,
                    "item_id": item_id,
                    "grade": grade,
                    "grade_num": GRADE_TO_NUM.get(grade, np.nan),
                })

    return pd.DataFrame(rows)


def load_cells(results_dir: Path, label: str) -> pd.DataFrame:
    """Load cell-level percent differences from a results directory."""
    rows = []
    if not results_dir.exists():
        return pd.DataFrame()

    for paper_dir in sorted(results_dir.iterdir()):
        if not paper_dir.is_dir() or paper_dir.name.startswith("_") or paper_dir.name.startswith("batch"):
            continue
        paper_slug = paper_dir.name

        for run_dir in sorted(paper_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            vr = _load_json(run_dir / "verification_report.json")
            if vr is None:
                continue

            dirname = run_dir.name
            idx = dirname.find(f"_{paper_slug}_")
            if idx < 0:
                continue
            model = dirname[:idx]
            approach = dirname[idx + len(f"_{paper_slug}_"):]
            if approach not in APPROACH_ORDER_RAW:
                continue

            approach_key = f"{approach}/{model}"

            for item in vr.get("item_verifications", []):
                item_id = item.get("item_id", "")
                item_type = _parse_item_type(item_id, item.get("item_type"))
                if item_type != "table":
                    continue
                grade = item.get("grade", "F")
                notes = item.get("comparison_notes", "")
                if "non-numerical" in notes.lower():
                    continue
                if grade == "F":
                    continue

                tc = item.get("table_comparison", {})
                for cell in (tc.get("cell_comparisons", []) if tc else []):
                    pct = cell.get("percent_difference")
                    if pct is None:
                        continue
                    rows.append({
                        "collection": label,
                        "paper_slug": paper_slug,
                        "approach": approach_key,
                        "item_id": item_id,
                        "percent_difference": pct,
                    })

    return pd.DataFrame(rows)


def plot_pct_diff_comparison(df_cells: pd.DataFrame, output_dir: Path):
    """Boxplot of |% difference| pre vs post cutoff, by approach."""
    if df_cells.empty:
        print("  No cell data for pct diff comparison")
        return

    approaches_pre = set(df_cells[df_cells["collection"] == "Pre-cutoff"]["approach"].unique())
    approaches_post = set(df_cells[df_cells["collection"] == "Post-cutoff"]["approach"].unique())
    common = sorted(approaches_pre & approaches_post)
    if not common:
        print("  No common approaches for pct diff comparison")
        return

    df = df_cells[df_cells["approach"].isin(common)].copy()
    df["pct_abs"] = df["percent_difference"].abs().clip(upper=200)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for ax, collection in zip(axes, ["Pre-cutoff", "Post-cutoff"]):
        sub = df[df["collection"] == collection]
        data, labels, colors = [], [], []
        for a in common:
            vals = sub.loc[sub["approach"] == a, "pct_abs"].dropna().values
            if len(vals) > 0:
                data.append(vals)
                labels.append(APPROACH_MODEL_LABELS.get(a, a).replace("\n", " "))
                colors.append(APPROACH_MODEL_COLORS.get(a, "#95a5a6"))

        if not data:
            ax.set_title(collection, fontsize=16, fontweight="bold")
            continue

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

        n_papers = sub[sub["approach"].isin(common)]["paper_slug"].nunique()
        n_cells = len(sub[sub["approach"].isin(common)])
        ax.text(0.02, 0.95, f"n={n_papers} papers, {n_cells} cells",
                transform=ax.transAxes, fontsize=10, va="top")
        ax.set_title(collection, fontsize=16, fontweight="bold")
        ax.set_xlabel("")
        apply_style(ax)

    axes[0].set_ylabel("|% Difference| (Capped at 200%)", fontsize=16, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, output_dir, "pct_diff_pre_post")


def plot_pct_diff_cdf_comparison(df_cells: pd.DataFrame, output_dir: Path):
    """CDF of |% difference| pre vs post cutoff, per approach."""
    if df_cells.empty:
        return

    approaches_pre = set(df_cells[df_cells["collection"] == "Pre-cutoff"]["approach"].unique())
    approaches_post = set(df_cells[df_cells["collection"] == "Post-cutoff"]["approach"].unique())
    common = sorted(approaches_pre & approaches_post)
    if not common:
        return

    df = df_cells[df_cells["approach"].isin(common)].copy()
    df["pct_abs"] = df["percent_difference"].abs()

    fig, ax = plt.subplots(figsize=(10, 6))

    for a in common:
        color = APPROACH_MODEL_COLORS.get(a, "#95a5a6")
        label_base = APPROACH_MODEL_LABELS.get(a, a).replace("\n", " ")

        for collection, linestyle in [("Pre-cutoff", "-"), ("Post-cutoff", "--")]:
            vals = df[(df["collection"] == collection) & (df["approach"] == a)]["pct_abs"].sort_values().values
            if len(vals) == 0:
                continue
            cdf_y = np.arange(1, len(vals) + 1) / len(vals) * 100
            suffix = " (post)" if collection == "Post-cutoff" else ""
            ax.plot(vals, cdf_y, label=f"{label_base}{suffix}",
                    color=color, linestyle=linestyle, linewidth=2)

    ax.set_xlim(0, 100)
    ax.set_xlabel("|% Difference|", fontsize=16, fontweight="bold")
    ax.set_ylabel("Cumulative Share of Cells (%)", fontsize=16, fontweight="bold")
    ax.axvline(x=2, color="gray", linestyle=":", alpha=0.5, label="2% (Grade A)")
    ax.axvline(x=20, color="gray", linestyle=":", alpha=0.3, label="20% (Grade B)")
    place_legend(fig, ax, fontsize=11, ncol=3)
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig, output_dir, "pct_diff_cdf_pre_post")


def plot_grade_comparison(df: pd.DataFrame, output_dir: Path):
    """Side-by-side grade distribution: pre vs post cutoff, by approach."""
    if df.empty:
        print("  No data for grade comparison")
        return

    # Only approaches present in both collections
    approaches_pre = set(df[df["collection"] == "Pre-cutoff"]["approach"].unique())
    approaches_post = set(df[df["collection"] == "Post-cutoff"]["approach"].unique())
    common = sorted(approaches_pre & approaches_post)

    if not common:
        print("  No common approaches between pre and post cutoff")
        return

    grades_shown = [g for g in GRADE_ORDER if g != "F"]
    df_nof = df[df["grade"] != "F"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for ax, collection in zip(axes, ["Pre-cutoff", "Post-cutoff"]):
        sub = df_nof[df_nof["collection"] == collection]
        if sub.empty:
            ax.set_title(collection, fontsize=16, fontweight="bold")
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        ct = pd.crosstab(sub["approach"], sub["grade"], normalize="index") * 100
        ct = ct.reindex(columns=grades_shown, fill_value=0)
        present = [a for a in common if a in ct.index]
        if not present:
            continue

        x = np.arange(len(present))
        width = 0.15
        for i, grade in enumerate(grades_shown):
            vals = [ct.loc[a, grade] if a in ct.index else 0 for a in present]
            ax.bar(x + i * width, vals, width, label=grade, color=GRADE_COLORS[grade], edgecolor="white")

        ax.set_xticks(x + width * 2)
        ax.set_xticklabels([APPROACH_MODEL_LABELS.get(a, a).replace("\n", " ") for a in present],
                           fontsize=10, rotation=25, ha="right")
        ax.set_xlabel("")
        ax.set_title(collection, fontsize=16, fontweight="bold")
        n_papers = sub[sub["approach"].isin(present)]["paper_slug"].nunique()
        n_tables = len(sub[sub["approach"].isin(present)])
        ax.text(0.02, 0.95, f"n={n_papers} papers, {n_tables} tables",
                transform=ax.transAxes, fontsize=10, va="top")
        apply_style(ax)

    axes[0].set_ylabel("Share of Tables (%)", fontsize=18, fontweight="bold")
    place_legend(fig, axes[0], fontsize=14, ncol=5)
    plt.tight_layout()
    save_figure(fig, output_dir, "grade_comparison_pre_post")


def plot_mean_grade_comparison(df: pd.DataFrame, output_dir: Path):
    """Mean grade by approach: pre vs post cutoff, grouped bars."""
    if df.empty:
        return

    approaches_pre = set(df[df["collection"] == "Pre-cutoff"]["approach"].unique())
    approaches_post = set(df[df["collection"] == "Post-cutoff"]["approach"].unique())
    common = sorted(approaches_pre & approaches_post)
    if not common:
        return

    df_nof = df[df["grade"] != "F"]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(common))
    width = 0.35

    for i, (collection, color, offset) in enumerate([
        ("Pre-cutoff", "#3498db", -width/2),
        ("Post-cutoff", "#e74c3c", width/2),
    ]):
        means = []
        sds = []
        for a in common:
            vals = df_nof[(df_nof["collection"] == collection) & (df_nof["approach"] == a)]["grade_num"]
            means.append(vals.mean() if len(vals) > 0 else 0)
            sds.append(vals.std() if len(vals) > 1 else 0)

        yerr_low = [min(m, s) for m, s in zip(means, sds)]
        ax.bar(x + offset, means, width, label=collection, color=color, alpha=0.8, edgecolor="white")
        ax.errorbar(x + offset, means, yerr=[yerr_low, sds], fmt="none",
                    ecolor="black", capsize=4, capthick=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels([APPROACH_MODEL_LABELS.get(a, a).replace("\n", " ") for a in common],
                       fontsize=10, rotation=25, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("Mean Grade (excl. F)", fontsize=18, fontweight="bold")
    ax.set_yticks(range(6))
    ax.set_yticklabels(GRADE_ORDER[::-1])
    place_legend(fig, ax, fontsize=14)
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig, output_dir, "mean_grade_pre_post")


def _bootstrap_ci(vals, n_boot=10000, ci=0.95):
    """Compute bootstrap confidence interval for the mean."""
    vals = np.array(vals)
    n = len(vals)
    if n < 2:
        return vals.mean(), vals.mean(), vals.mean()
    rng = np.random.default_rng(42)
    boot_means = [rng.choice(vals, size=n, replace=True).mean() for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    lo = np.quantile(boot_means, alpha)
    hi = np.quantile(boot_means, 1 - alpha)
    return vals.mean(), lo, hi


def _permutation_test(vals_a, vals_b, n_perm=10000):
    """Two-sided permutation test for difference in means. Returns p-value."""
    vals_a, vals_b = np.array(vals_a), np.array(vals_b)
    observed = abs(vals_a.mean() - vals_b.mean())
    combined = np.concatenate([vals_a, vals_b])
    n_a = len(vals_a)
    rng = np.random.default_rng(42)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(combined)
        diff = abs(perm[:n_a].mean() - perm[n_a:].mean())
        if diff >= observed:
            count += 1
    return count / n_perm


def plot_bootstrap_comparison(df: pd.DataFrame, output_dir: Path):
    """Mean grade with bootstrap 95% CIs: pre vs post cutoff, per approach."""
    if df.empty:
        return

    approaches_pre = set(df[df["collection"] == "Pre-cutoff"]["approach"].unique())
    approaches_post = set(df[df["collection"] == "Post-cutoff"]["approach"].unique())
    common = sorted(approaches_pre & approaches_post)
    if not common:
        return

    df_nof = df[df["grade"] != "F"]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(common))
    width = 0.3
    collections = [("Pre-cutoff", "#3498db", -width/2), ("Post-cutoff", "#e74c3c", width/2)]

    for collection, color, offset in collections:
        means, ci_lo, ci_hi, ns = [], [], [], []
        for a in common:
            vals = df_nof[(df_nof["collection"] == collection) & (df_nof["approach"] == a)]["grade_num"].values
            if len(vals) > 0:
                m, lo, hi = _bootstrap_ci(vals)
                means.append(m)
                ci_lo.append(m - lo)
                ci_hi.append(hi - m)
                ns.append(len(vals))
            else:
                means.append(np.nan)
                ci_lo.append(0)
                ci_hi.append(0)
                ns.append(0)

        ax.bar(x + offset, means, width, label=collection, color=color, alpha=0.8, edgecolor="white")
        ax.errorbar(x + offset, means, yerr=[ci_lo, ci_hi], fmt="none",
                    ecolor="black", capsize=5, capthick=1.5, linewidth=1.5)

        # Add n labels
        for i, n in enumerate(ns):
            ax.text(x[i] + offset, 0.15, f"n={n}", ha="center", fontsize=9, color="white", fontweight="bold")

    # Permutation test p-values
    for i, a in enumerate(common):
        pre_vals = df_nof[(df_nof["collection"] == "Pre-cutoff") & (df_nof["approach"] == a)]["grade_num"].values
        post_vals = df_nof[(df_nof["collection"] == "Post-cutoff") & (df_nof["approach"] == a)]["grade_num"].values
        if len(pre_vals) > 0 and len(post_vals) > 0:
            p = _permutation_test(pre_vals, post_vals)
            # Draw bracket
            y_max = max(
                _bootstrap_ci(pre_vals)[2],
                _bootstrap_ci(post_vals)[2],
            ) + 0.15
            ax.plot([x[i] - width/2, x[i] - width/2, x[i] + width/2, x[i] + width/2],
                    [y_max, y_max + 0.1, y_max + 0.1, y_max], color="black", linewidth=1)
            sig = "n.s." if p > 0.05 else (f"p={p:.3f}" if p > 0.001 else "p<0.001")
            ax.text(x[i], y_max + 0.15, sig, ha="center", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels([APPROACH_MODEL_LABELS.get(a, a).replace("\n", " ") for a in common],
                       fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel("Mean Table Grade (excl. F)", fontsize=18, fontweight="bold")
    ax.set_yticks(range(6))
    ax.set_yticklabels(GRADE_ORDER[::-1])
    place_legend(fig, ax, fontsize=14)
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig, output_dir, "bootstrap_grade_comparison")


def main():
    parser = argparse.ArgumentParser(description="Compare pre vs post cutoff results")
    parser.add_argument("--precutoff-results", type=str, default=None)
    parser.add_argument("--postcutoff-results", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="analysis_output/pre_post_cutoff")
    args = parser.parse_args()

    # Auto-detect paths
    if args.precutoff_results is None:
        for base in [TEXTLAB_BASE, LOCAL_BASE]:
            p = base / "precutoff" / "results"
            if p.exists():
                args.precutoff_results = str(p)
                break
    if args.postcutoff_results is None:
        for base in [TEXTLAB_BASE, LOCAL_BASE]:
            p = base / "postcutoff" / "results"
            if p.exists():
                args.postcutoff_results = str(p)
                break

    if not args.precutoff_results or not args.postcutoff_results:
        sys.exit("Cannot find results directories. Use --precutoff-results and --postcutoff-results.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Pre-cutoff:  {args.precutoff_results}")
    print(f"Post-cutoff: {args.postcutoff_results}")
    print(f"Output:      {output_dir}")

    setup_style()

    df_pre = load_collection(Path(args.precutoff_results), "Pre-cutoff")
    df_post = load_collection(Path(args.postcutoff_results), "Post-cutoff")

    print(f"\nPre-cutoff:  {len(df_pre)} table items from {df_pre['paper_slug'].nunique() if not df_pre.empty else 0} papers")
    print(f"Post-cutoff: {len(df_post)} table items from {df_post['paper_slug'].nunique() if not df_post.empty else 0} papers")

    df = pd.concat([df_pre, df_post], ignore_index=True)
    if df.empty:
        print("No data found!")
        return

    # Print summary
    for col in ["Pre-cutoff", "Post-cutoff"]:
        sub = df[df["collection"] == col]
        if sub.empty:
            continue
        print(f"\n{col}:")
        for a in sorted(sub["approach"].unique()):
            asub = sub[sub["approach"] == a]
            nof = asub[asub["grade"] != "F"]
            print(f"  {a:35s}: {len(asub):3d} tables, mean={nof['grade_num'].mean():.2f}, "
                  f"A-B={asub['grade'].isin(['A','B']).mean()*100:.0f}%, "
                  f"F={asub['grade'].eq('F').mean()*100:.0f}%")

    plot_grade_comparison(df, output_dir)
    plot_mean_grade_comparison(df, output_dir)
    plot_bootstrap_comparison(df, output_dir)

    # Cell-level percent difference plots
    df_cells_pre = load_cells(Path(args.precutoff_results), "Pre-cutoff")
    df_cells_post = load_cells(Path(args.postcutoff_results), "Post-cutoff")
    df_cells = pd.concat([df_cells_pre, df_cells_post], ignore_index=True)
    print(f"\nCells: {len(df_cells_pre)} pre-cutoff, {len(df_cells_post)} post-cutoff")

    if not df_cells.empty:
        plot_pct_diff_comparison(df_cells, output_dir)
        plot_pct_diff_cdf_comparison(df_cells, output_dir)

    # Save data
    df.to_csv(output_dir / "pre_post_cutoff.csv", index=False)
    print(f"\nSaved pre_post_cutoff.csv")

    print(f"\nDone! Outputs in {output_dir}/")


if __name__ == "__main__":
    main()
