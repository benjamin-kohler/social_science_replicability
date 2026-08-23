"""Render a single-column-width version of the root-cause stacked bar chart.

Generates `root_causes_coarse_absolute_narrow.{pdf,png}` next to the wide version,
sized to fit a ~3.1 in column in a two-column publication (EMNLP/ACL).

Usage:
    python scripts/plot_root_causes_narrow.py \
        --csv analysis_output_regrade_na/df_divergences.csv \
        --out analysis_output_regrade_na/discrepancy_analysis
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --- Mappings copied from scripts/analyze_i4rep_results.py ------------------

APPROACH_ORDER = [
    "claude-code/claude-opus-4-6",
    "codex/gpt-5.3-codex",
    "codex/gpt-5.4",
    "swe-agent/gpt-5.4",
    "swe-agent/z-ai_glm-5",
    "opencode/gpt-5.4",
    "opencode/z-ai_glm-5",
]

APPROACH_LABELS = {
    "claude-code/claude-opus-4-6": "Claude Code Opus 4.6",
    "codex/gpt-5.4": "Codex CLI GPT-5.4",
    "codex/gpt-5.3-codex": "Codex CLI GPT-5.3",
    "swe-agent/gpt-5.4": "SWE-Agent GPT-5.4",
    "swe-agent/z-ai_glm-5": "SWE-Agent GLM-5",
    "opencode/gpt-5.4": "OpenCode GPT-5.4",
    "opencode/z-ai_glm-5": "OpenCode GLM-5",
}

APPROACH_COLORS = {
    "claude-code/claude-opus-4-6": "#E07B39",
    "codex/gpt-5.4": "#10A37F",
    "codex/gpt-5.3-codex": "#0D8A6A",
    "swe-agent/gpt-5.4": "#6C5CE7",
    "swe-agent/z-ai_glm-5": "#A29BFE",
    "opencode/gpt-5.4": "#0984E3",
    "opencode/z-ai_glm-5": "#74B9FF",
}

_ROOT_CAUSE_COARSE_MAP = {
    "Agent contradicted summary": "Extraction vs Agent",
    "Agent missed summary info": "Extraction vs Agent",
    "Summary gap (contradicts)": "Paper vs Extraction",
    "Summary gap (omission)": "Paper vs Extraction",
    "Paper underspecified": "Paper vs Code",
    "Paper-code mismatch": "Paper vs Code",
    "Data not in package": "Missing data",
    "Insufficient specification": "Other",
    "Unexplained": "Other",
}

_ROOT_CAUSE_COARSE_RENAMED = {
    "Extraction vs Agent": "Agent error",
    "Paper vs Extraction": "Extractor error",
    "Paper vs Code": "Original error",
    "Missing data": "Data missing",
    "Other": "Other",
}


# --- Narrow-figure plot ----------------------------------------------------

def plot_narrow(df_div: pd.DataFrame, output_dir: Path) -> None:
    if df_div.empty:
        print("No divergence rows to plot.")
        return

    df = df_div[~df_div["parse_failed"]].copy()
    df["root_cause_coarse"] = df["root_cause"].map(_ROOT_CAUSE_COARSE_MAP).fillna("Other")
    df["root_cause_renamed"] = df["root_cause_coarse"].map(_ROOT_CAUSE_COARSE_RENAMED)

    cause_order = ["Data missing", "Original error", "Extractor error",
                   "Agent error", "Other"]
    cause_colors = {
        "Data missing": "#3498db",
        "Original error": "#9b59b6",
        "Extractor error": "#e67e22",
        "Agent error": "#e74c3c",
        "Other": "#95a5a6",
    }

    ct = pd.crosstab(df["approach"], df["root_cause_renamed"])
    present_causes = [c for c in cause_order if c in ct.columns]
    ct = ct.reindex(columns=present_causes, fill_value=0)

    approaches = [a for a in APPROACH_ORDER if a in df["approach"].values]
    extra = [a for a in df["approach"].unique() if a not in approaches]
    approaches = approaches + extra
    ct_plot = ct.loc[[a for a in approaches if a in ct.index]]
    ct_plot = ct_plot.loc[ct_plot.sum(axis=1).sort_values().index]

    n_rows = len(ct_plot)
    # 3.3 in wide = single ACL column. Tighter row spacing for shorter figure.
    fig, ax = plt.subplots(figsize=(3.3, max(2.0, n_rows * 0.24 + 0.9)))

    y = np.arange(n_rows)
    lefts = np.zeros(n_rows)
    for cause in present_causes:
        vals = ct_plot[cause].values.astype(float)
        if vals.sum() == 0:
            continue
        ax.barh(y, vals, left=lefts, color=cause_colors.get(cause, "#bdc3c7"),
                label=cause, height=0.85, edgecolor="white", linewidth=0.4)
        lefts += vals

    # Total counts at the end of each bar
    max_total = lefts.max() if len(lefts) else 0
    for i, total in enumerate(lefts):
        ax.text(total + max_total * 0.015, i, str(int(total)),
                va="center", fontsize=6.5, fontweight="bold")

    ax.set_yticks(y)
    labels = [APPROACH_LABELS.get(a, a) for a in ct_plot.index]
    colors = [APPROACH_COLORS.get(a, "#95a5a6") for a in ct_plot.index]
    ax.set_yticklabels(labels, fontsize=6.5)
    for tick_label, color in zip(ax.get_yticklabels(), colors):
        tick_label.set_color(color)
        tick_label.set_fontweight("bold")

    ax.set_xlabel("Number of divergences", fontsize=7.5, fontweight="bold")
    ax.tick_params(axis="x", labelsize=6.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False)
    ax.grid(visible=False, which="both")

    # Slight headroom for the count labels
    ax.set_xlim(0, max_total * 1.10 + 1)

    # Legend below the figure, three per row
    handles, lbls = ax.get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=3,
               fontsize=6.5, frameon=False,
               bbox_to_anchor=(0.5, -0.08),
               handletextpad=0.4, columnspacing=0.9)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.22)

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "root_causes_coarse_absolute_narrow.pdf"
    png_path = output_dir / "root_causes_coarse_absolute_narrow.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="analysis_output_regrade_na/df_divergences.csv",
        help="Path to df_divergences.csv produced by analyze_i4rep_results.py",
    )
    parser.add_argument(
        "--out",
        default="analysis_output_regrade_na/discrepancy_analysis",
        help="Output directory for the narrow figure",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    plot_narrow(df, Path(args.out))


if __name__ == "__main__":
    main()
