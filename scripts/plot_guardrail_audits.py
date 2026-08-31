#!/usr/bin/env python3
"""Plot guardrail and hardcoding audit results.

Originally an Jupyter notebook (audit_stats_v2.ipynb); converted to a script.
Reads the latest guardrails_audit_*.csv and hardcoding_audit_*.csv produced
by classify_guardrail_breaches.py and writes 4 PDF/PNG plots into ./plots/.
"""

# # Audit Stats v2
# 
# Analysis notebook for the split guardrail (access breaches) and hardcoding audits.
# 
# - **Section 1**: Guardrails — forbidden access / provenance violations
# - **Section 2**: Hardcoding — hardcoded result values in .py scripts

import argparse
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--audit-dir", type=Path, default=Path.cwd(),
    help="Directory containing guardrails_audit_*.csv and hardcoding_audit_*.csv.",
)
parser.add_argument(
    "--results-dir", type=Path,
    default=Path(os.environ.get(
        "I4REPLICATE_RESULTS_DIR",
        str(PROJECT_ROOT / "data" / "i4replicate" / "results"),
    )),
    help="Benchmark results root used to attach verification grades.",
)
parser.add_argument(
    "--output-dir", type=Path, default=None,
    help="Figure output directory (default: AUDIT_DIR/plots).",
)
args = parser.parse_args()

AUDIT_DIR = args.audit_dir.expanduser().resolve()
RESULTS_ROOT = args.results_dir.expanduser().resolve()
PLOTS_DIR = (
    args.output_dir.expanduser().resolve()
    if args.output_dir is not None
    else AUDIT_DIR / "plots"
)

sns.set_theme(style="whitegrid", font_scale=1.4)
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "sans-serif",
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_APPROACH_LABELS = {
    ("claude-opus-4-6", "claude-code"): "Claude Code\nOpus 4.6",
    ("gpt-5.3-codex", "codex"):        "Codex CLI\nGPT-5.3 Codex",
    ("gpt-5.4", "codex"):              "Codex CLI\nGPT-5.4",
    ("gpt-5.4", "swe-agent"):          "SWE-Agent\nGPT-5.4",
    ("gpt-5.4", "opencode"):           "OpenCode\nGPT-5.4",
    ("z-ai_glm-5", "opencode"):        "OpenCode\nGLM-5",
    ("z-ai_glm-5", "swe-agent"):       "SWE-Agent\nGLM-5",
}

def load_latest_csv(pattern: str) -> tuple[Path, pd.DataFrame]:
    """Load the most recent CSV matching the pattern."""
    csv_paths = sorted(AUDIT_DIR.glob(pattern))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files matching {pattern}")
    path = csv_paths[-1]
    print(f"Loading: {path.name}")
    return path, pd.read_csv(path)


def apply_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=14)


def place_legend(fig, ax, ncol=None, fontsize=12, **kwargs):
    """Place legend below the plot area."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    leg = ax.get_legend()
    if leg:
        leg.remove()
    if ncol is None:
        ncol = min(len(handles), 4)
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.08), ncol=ncol,
               fontsize=fontsize, frameon=True, fancybox=True, **kwargs)


def save_figure(fig, name: str):
    fig.savefig(PLOTS_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(PLOTS_DIR / f"{name}.pdf", bbox_inches="tight")
    print(f"  Saved {name}")

# ---
# # Section 1: Guardrails (Access Breaches)

guardrail_csv_path, guardrail_df = load_latest_csv("guardrails_audit_*.csv")
guardrail_run_df = guardrail_df.groupby("run_name", as_index=False).first()
print(f"Rows: {len(guardrail_df)}, Runs: {len(guardrail_run_df)}, Papers: {guardrail_run_df['paper_id'].nunique()}")
guardrail_run_df[["run_name", "paper_id", "model", "approach", "overall_assessment"]].head()

# ### LLM Assessment Overview

guardrail_run_df["overall_assessment"].value_counts(dropna=False).rename_axis("overall_assessment").to_frame("runs")

(
    guardrail_run_df.groupby(["approach", "overall_assessment"]).size()
    .unstack(fill_value=0)
    .sort_index()
)

# ### Replication Grades for Guardrail-Flagged Runs

import json

# (RESULTS_ROOT defined at top of file)
GRADE_TO_SCORE = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1, "F": 0}
SCORE_TO_GRADE = {v: k for k, v in GRADE_TO_SCORE.items()}

def extract_verification_grade(run_dir: Path):
    path = run_dir / "verification_report.json"
    if not path.exists():
        return np.nan
    try:
        data = json.loads(path.read_text())
    except Exception:
        return np.nan
    grade = data.get("overall_grade") or data.get("grade")
    return grade if grade in GRADE_TO_SCORE else np.nan

guardrail_grade_df = guardrail_run_df.copy()
guardrail_grade_df["overall_grade"] = guardrail_grade_df.apply(
    lambda r: extract_verification_grade(RESULTS_ROOT / r["paper_id"] / r["run_name"]),
    axis=1,
)
guardrail_grade_df["grade_score"] = guardrail_grade_df["overall_grade"].map(GRADE_TO_SCORE)

flagged_mask = guardrail_grade_df["overall_assessment"].isin(["likely_violation", "severe_violation"])
clean_mask = guardrail_grade_df["overall_assessment"].eq("clean")
other_mask = ~flagged_mask

def summarize_subset(df, label):
    return pd.DataFrame({
        "runs": [len(df)],
        "runs_with_grade": [int(df["grade_score"].notna().sum())],
        "avg_grade_score": [df["grade_score"].mean()],
        "median_grade_score": [df["grade_score"].median()],
    }, index=[label])

guardrail_grade_summary = pd.concat([
    summarize_subset(guardrail_grade_df, "all_runs"),
    summarize_subset(guardrail_grade_df.loc[clean_mask], "clean_runs"),
    summarize_subset(guardrail_grade_df.loc[other_mask], "all_other_runs"),
    summarize_subset(guardrail_grade_df.loc[flagged_mask], "all_flagged_runs"),
    guardrail_grade_df.loc[flagged_mask]
    .groupby("overall_assessment")
    .agg(
        runs=("run_name", "size"),
        runs_with_grade=("grade_score", lambda s: s.notna().sum()),
        avg_grade_score=("grade_score", "mean"),
        median_grade_score=("grade_score", "median"),
    )
])

guardrail_grade_summary["avg_grade_score"] = guardrail_grade_summary["avg_grade_score"].round(2)
guardrail_grade_summary["median_grade_score"] = guardrail_grade_summary["median_grade_score"].round(2)
guardrail_grade_summary["avg_grade_approx"] = guardrail_grade_summary["avg_grade_score"].round().map(SCORE_TO_GRADE)
guardrail_grade_summary["median_grade_approx"] = guardrail_grade_summary["median_grade_score"].round().map(SCORE_TO_GRADE)

guardrail_grade_summary

# ### Access Breach Types

guardrail_breach_df = guardrail_df[guardrail_df["type"].fillna("") != ""].copy()
guardrail_breach_run_type = (
    guardrail_breach_df
    .drop_duplicates(["run_name", "model", "approach", "type"])
    .copy()
)
print(f"Breach rows: {len(guardrail_breach_df)}")
print(f"Run-type pairs: {len(guardrail_breach_run_type)}")
print(f"Distinct violating runs: {guardrail_breach_run_type['run_name'].nunique()}")

guardrail_breach_run_type["type"].value_counts().rename_axis("type").to_frame("runs")

(
    guardrail_breach_run_type.groupby(["approach", "type"]).size()
    .unstack(fill_value=0)
    .sort_index()
)

# ### Regex vs LLM Signal Comparison

# Compare regex deterministic findings vs LLM assessment
regex_cols = ["regex_any_forbidden", "regex_any_forbidden_replication_package",
              "regex_any_forbidden_paper_pdf", "regex_any_web"]
available_regex_cols = [c for c in regex_cols if c in guardrail_run_df.columns]

if available_regex_cols:
    print("Regex signal summary:")
    for col in available_regex_cols:
        n_true = guardrail_run_df[col].sum()
        print(f"  {col}: {int(n_true)} / {len(guardrail_run_df)} runs")

    # Cross-tab: regex_any_forbidden vs LLM overall_assessment
    print("\nRegex forbidden vs LLM assessment:")
    if "regex_any_forbidden" in guardrail_run_df.columns:
        display(pd.crosstab(
            guardrail_run_df["regex_any_forbidden"],
            guardrail_run_df["overall_assessment"],
            margins=True,
        ))
else:
    print("No regex columns found in CSV (run with v2 script to get them)")

# Runs where regex found forbidden paths
if "regex_forbidden_paths" in guardrail_run_df.columns:
    flagged = guardrail_run_df[guardrail_run_df["regex_forbidden_paths"].fillna("") != ""].copy()
    if len(flagged):
        print(f"Runs with regex-detected forbidden paths: {len(flagged)}")
        display(flagged[["run_name", "overall_assessment", "regex_forbidden_paths", "regex_web_urls"]].head(20))
    else:
        print("No runs with regex-detected forbidden paths.")

# ### Guardrail Plots

GUARDRAIL_ASSESSMENT_LABELS = {
    "clean": "Clean",
    "insufficient_evidence": "Insufficient Evidence",
    "suspicious": "Suspicious",
    "likely_violation": "Likely Violation",
    "severe_violation": "Severe Violation",
}

GUARDRAIL_ASSESSMENT_COLORS = {
    "Clean":                  "#27ae60",
    "Insufficient Evidence":  "#f1c40f",
    "Suspicious":             "#e67e22",
    "Likely Violation":       "#e74c3c",
    "Severe Violation":       "#8e44ad",
}

assessment_order = ["clean", "insufficient_evidence", "suspicious", "likely_violation", "severe_violation"]
assessment_by_ma = (
    guardrail_run_df.groupby(["model", "approach", "overall_assessment"]).size()
    .unstack(fill_value=0)
    .reindex(columns=assessment_order, fill_value=0)
)
assessment_by_ma = assessment_by_ma.loc[:, assessment_by_ma.sum() > 0]

ordered_keys = [k for k in MODEL_APPROACH_LABELS if k in assessment_by_ma.index]
assessment_by_ma = assessment_by_ma.loc[ordered_keys]
assessment_by_ma.columns = [GUARDRAIL_ASSESSMENT_LABELS[c] for c in assessment_by_ma.columns]
assessment_by_ma.index = [MODEL_APPROACH_LABELS[k] for k in assessment_by_ma.index]

fig, ax = plt.subplots(figsize=(12, 7))
colors = [GUARDRAIL_ASSESSMENT_COLORS[c] for c in assessment_by_ma.columns]
assessment_by_ma.plot(kind="bar", stacked=True, color=colors, ax=ax, width=0.7)

# Label segments >= 3 centered inside; collect small segments per bar for annotation above
MIN_HEIGHT_FOR_CENTER = 3
n_bars = len(assessment_by_ma)
small_segments = {i: [] for i in range(n_bars)}  # bar_idx -> [(count, color_name)]

for container_idx, container in enumerate(ax.containers):
    col_name = assessment_by_ma.columns[container_idx]
    for bar_idx, bar in enumerate(container):
        h = bar.get_height()
        if h == 0:
            continue
        if h >= MIN_HEIGHT_FOR_CENTER:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + h / 2,
                    f"{int(h)}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color="white")
        else:
            small_segments[bar_idx].append((int(h), col_name))

# Place combined annotation above each bar for small segments
for bar_idx, segs in small_segments.items():
    if not segs:
        continue
    # Build a single string like "1 IE, 2 LV"
    abbrevs = {"Insufficient Evidence": "IE", "Suspicious": "S",
               "Likely Violation": "LV", "Severe Violation": "SV"}
    parts = [f"{count} {abbrevs.get(name, name)}" for count, name in segs]
    label = ", ".join(parts)
    # Get bar x-center and total height
    first_container = ax.containers[0]
    bar = first_container[bar_idx]
    x_center = bar.get_x() + bar.get_width() / 2
    total_h = assessment_by_ma.iloc[bar_idx].sum()
    ax.text(x_center, total_h + 0.5, label, ha="center", va="bottom",
            fontsize=8.5, fontweight="bold", color="#333333")

ax.set_title("Guardrail Assessments by Model & Approach", fontsize=18, fontweight="bold", pad=12)
ax.set_xlabel("")
ax.set_ylabel("Runs", fontsize=18, fontweight="bold")
apply_style(ax)
ax.grid(True, alpha=0.3, axis="y")
ax.set_ylim(0, assessment_by_ma.sum(axis=1).max() * 1.1)
plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

place_legend(fig, ax, ncol=len(assessment_by_ma.columns), fontsize=11)

fig.tight_layout()
save_figure(fig, "guardrail_assessments_by_model_approach")
plt.show()

GUARDRAIL_BREACH_TYPE_COLORS = {
    "external_result_lookup":  "#0984E3",
    "forbidden_paper_access":  "#DADE0B",
    "forbidden_code_access":   "#e74c3c",
    "web_search":              "#00b894",
    "other":                   "#95a5a6",
}

def _pretty_label(s: str) -> str:
    """Convert snake_case to Title Case (single line)."""
    return " ".join(w.capitalize() for w in s.split("_"))

if len(guardrail_breach_run_type):
    type_by_ma = (
        guardrail_breach_run_type.groupby(["model", "approach", "type"]).size()
        .unstack(fill_value=0)
        .sort_index()
    )
    ordered_keys = [k for k in MODEL_APPROACH_LABELS if k in type_by_ma.index]
    type_by_ma = type_by_ma.loc[ordered_keys]
    type_by_ma.index = [MODEL_APPROACH_LABELS[k] for k in type_by_ma.index]
    col_order = type_by_ma.sum().sort_values(ascending=False).index
    type_by_ma = type_by_ma[col_order]

    colors = [GUARDRAIL_BREACH_TYPE_COLORS.get(c, "#95a5a6") for c in type_by_ma.columns]
    type_by_ma.columns = [_pretty_label(c) for c in type_by_ma.columns]

    fig, ax = plt.subplots(figsize=(10, 5))
    type_by_ma.plot(kind="barh", stacked=True, color=colors, ax=ax, width=0.7)

    for container in ax.containers:
        labels = [f"{int(v.get_width())}" if v.get_width() >= 1 else "" for v in container]
        ax.bar_label(container, labels=labels, label_type="center",
                     fontsize=11, fontweight="bold", color="white")

    ax.set_title("Access Breach Types by Model & Approach", fontsize=18, fontweight="bold", pad=12)
    ax.set_ylabel("")
    ax.set_xlabel("Runs", fontsize=18, fontweight="bold")
    apply_style(ax)
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()

    place_legend(fig, ax, fontsize=11)

    fig.tight_layout()
    save_figure(fig, "guardrail_breach_types_by_model_approach")
    plt.show()
else:
    print("No access breaches found.")

# ---
# # Section 2: Hardcoding (Hardcoded Result Values)

hardcoding_csv_path, hardcoding_df = load_latest_csv("hardcoding_audit_*.csv")
hardcoding_run_df = hardcoding_df.groupby("run_name", as_index=False).first()
print(f"Rows: {len(hardcoding_df)}, Runs: {len(hardcoding_run_df)}, Papers: {hardcoding_run_df['paper_id'].nunique()}")
hardcoding_run_df[["run_name", "paper_id", "model", "approach", "overall_assessment"]].head()

# ### Hardcoding Assessment Overview

hardcoding_run_df["overall_assessment"].value_counts(dropna=False).rename_axis("overall_assessment").to_frame("runs")

(
    hardcoding_run_df.groupby(["approach", "overall_assessment"]).size()
    .unstack(fill_value=0)
    .sort_index()
)

# ### Replication Grades for Hardcoding-Flagged Runs

import json

# (RESULTS_ROOT defined at top of file)
GRADE_TO_SCORE = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1, "F": 0}
SCORE_TO_GRADE = {v: k for k, v in GRADE_TO_SCORE.items()}

def extract_verification_grade(run_dir: Path):
    path = run_dir / "verification_report.json"
    if not path.exists():
        return np.nan
    try:
        data = json.loads(path.read_text())
    except Exception:
        return np.nan
    grade = data.get("overall_grade") or data.get("grade")
    if grade in GRADE_TO_SCORE:
        return grade
    return np.nan

hardcoding_grade_df = hardcoding_run_df.copy()
hardcoding_grade_df["overall_grade"] = hardcoding_grade_df.apply(
    lambda r: extract_verification_grade(RESULTS_ROOT / r["paper_id"] / r["run_name"]),
    axis=1,
)
hardcoding_grade_df["grade_score"] = hardcoding_grade_df["overall_grade"].map(GRADE_TO_SCORE)

flagged_mask = hardcoding_grade_df["overall_assessment"].isin([
    "suspicious", "likely_hardcoding", "severe_hardcoding"
])
clean_mask = hardcoding_grade_df["overall_assessment"].eq("clean")
insufficient_mask = hardcoding_grade_df["overall_assessment"].eq("insufficient_evidence")
clean_or_insufficient_mask = clean_mask | insufficient_mask
non_clean_mask = ~clean_mask  # flagged + insufficient_evidence

def summarize_subset(df, label):
    return pd.DataFrame({
        "runs": [len(df)],
        "runs_with_grade": [int(df["grade_score"].notna().sum())],
        "avg_grade_score": [df["grade_score"].mean()],
        "median_grade_score": [df["grade_score"].median()],
    }, index=[label])

hardcoding_grade_summary = pd.concat([
    summarize_subset(hardcoding_grade_df, "all_runs"),
    summarize_subset(hardcoding_grade_df.loc[clean_mask], "clean_runs"),
    summarize_subset(hardcoding_grade_df.loc[insufficient_mask], "insufficient_evidence_runs"),
    summarize_subset(hardcoding_grade_df.loc[clean_or_insufficient_mask], "clean_or_insufficient_runs"),
    summarize_subset(hardcoding_grade_df.loc[flagged_mask], "all_flagged_runs"),
    summarize_subset(hardcoding_grade_df.loc[non_clean_mask], "all_non_clean_runs"),
    hardcoding_grade_df.loc[flagged_mask]
    .groupby("overall_assessment")
    .agg(
        runs=("run_name", "size"),
        runs_with_grade=("grade_score", lambda s: s.notna().sum()),
        avg_grade_score=("grade_score", "mean"),
        median_grade_score=("grade_score", "median"),
    )
])

hardcoding_grade_summary["avg_grade_score"] = hardcoding_grade_summary["avg_grade_score"].round(2)
hardcoding_grade_summary["median_grade_score"] = hardcoding_grade_summary["median_grade_score"].round(2)
hardcoding_grade_summary["avg_grade_approx"] = hardcoding_grade_summary["avg_grade_score"].round().map(SCORE_TO_GRADE)
hardcoding_grade_summary["median_grade_approx"] = hardcoding_grade_summary["median_grade_score"].round().map(SCORE_TO_GRADE)

hardcoding_grade_summary

# ### Hardcoding Instance Types

hardcoding_instance_df = hardcoding_df[hardcoding_df["type"].fillna("") != ""].copy()
hardcoding_run_type = (
    hardcoding_instance_df
    .drop_duplicates(["run_name", "model", "approach", "type"])
    .copy()
)
print(f"Instance rows: {len(hardcoding_instance_df)}")
print(f"Run-type pairs: {len(hardcoding_run_type)}")
print(f"Distinct flagged runs: {hardcoding_run_type['run_name'].nunique()}")

hardcoding_run_type["type"].value_counts().rename_axis("type").to_frame("runs")

(
    hardcoding_run_type.groupby(["approach", "type"]).size()
    .unstack(fill_value=0)
    .sort_index()
)

# ### Hardcoding Plots

HARDCODING_ASSESSMENT_LABELS = {
    "clean": "Clean",
    "insufficient_evidence": "Insufficient Evidence",
    "suspicious": "Suspicious",
    "likely_hardcoding": "Likely Hardcoding",
    "severe_hardcoding": "Severe Hardcoding",
}

HARDCODING_ASSESSMENT_COLORS = {
    "Clean":                  "#27ae60",
    "Insufficient Evidence":  "#f1c40f",
    "Suspicious":             "#e67e22",
    "Likely Hardcoding":      "#e74c3c",
    "Severe Hardcoding":      "#8e44ad",
}

hc_assessment_order = ["clean", "insufficient_evidence", "suspicious", "likely_hardcoding", "severe_hardcoding"]
hc_assessment_by_ma = (
    hardcoding_run_df.groupby(["model", "approach", "overall_assessment"]).size()
    .unstack(fill_value=0)
    .reindex(columns=hc_assessment_order, fill_value=0)
)
hc_assessment_by_ma = hc_assessment_by_ma.loc[:, hc_assessment_by_ma.sum() > 0]

ordered_keys = [k for k in MODEL_APPROACH_LABELS if k in hc_assessment_by_ma.index]
hc_assessment_by_ma = hc_assessment_by_ma.loc[ordered_keys]
hc_assessment_by_ma.columns = [HARDCODING_ASSESSMENT_LABELS[c] for c in hc_assessment_by_ma.columns]
hc_assessment_by_ma.index = [MODEL_APPROACH_LABELS[k] for k in hc_assessment_by_ma.index]

fig, ax = plt.subplots(figsize=(12, 7))
colors = [HARDCODING_ASSESSMENT_COLORS[c] for c in hc_assessment_by_ma.columns]
hc_assessment_by_ma.plot(kind="bar", stacked=True, color=colors, ax=ax, width=0.7)

# Label segments >= 3 centered inside; collect small segments per bar for annotation above
MIN_HEIGHT_FOR_CENTER = 3
n_bars = len(hc_assessment_by_ma)
small_segments = {i: [] for i in range(n_bars)}

for container_idx, container in enumerate(ax.containers):
    col_name = hc_assessment_by_ma.columns[container_idx]
    for bar_idx, bar in enumerate(container):
        h = bar.get_height()
        if h == 0:
            continue
        if h >= MIN_HEIGHT_FOR_CENTER:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + h / 2,
                    f"{int(h)}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color="white")
        else:
            small_segments[bar_idx].append((int(h), col_name))

# Place combined annotation above each bar for small segments
for bar_idx, segs in small_segments.items():
    if not segs:
        continue
    abbrevs = {"Insufficient Evidence": "IE", "Suspicious": "S",
               "Likely Hardcoding": "LH", "Severe Hardcoding": "SH"}
    parts = [f"{count} {abbrevs.get(name, name)}" for count, name in segs]
    label = ", ".join(parts)
    first_container = ax.containers[0]
    bar = first_container[bar_idx]
    x_center = bar.get_x() + bar.get_width() / 2
    total_h = hc_assessment_by_ma.iloc[bar_idx].sum()
    ax.text(x_center, total_h + 0.5, label, ha="center", va="bottom",
            fontsize=8.5, fontweight="bold", color="#333333")

ax.set_title("Hardcoding Assessments by Model & Approach", fontsize=18, fontweight="bold", pad=12)
ax.set_xlabel("")
ax.set_ylabel("Runs", fontsize=18, fontweight="bold")
apply_style(ax)
ax.grid(True, alpha=0.3, axis="y")
ax.set_ylim(0, hc_assessment_by_ma.sum(axis=1).max() * 1.1)
plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

place_legend(fig, ax, ncol=len(hc_assessment_by_ma.columns), fontsize=11)

fig.tight_layout()
save_figure(fig, "hardcoding_assessments_by_model_approach")
plt.show()

HARDCODING_TYPE_COLORS = {
    "hardcoded_value":            "#e74c3c",
    "manual_value_fill":          "#e74c3c",
    "placeholder_or_estimate_fill": "#0984E3",
}

if len(hardcoding_run_type):
    hc_type_by_ma = (
        hardcoding_run_type.groupby(["model", "approach", "type"]).size()
        .unstack(fill_value=0)
        .sort_index()
    )
    ordered_keys = [k for k in MODEL_APPROACH_LABELS if k in hc_type_by_ma.index]
    hc_type_by_ma = hc_type_by_ma.loc[ordered_keys]
    hc_type_by_ma.index = [MODEL_APPROACH_LABELS[k] for k in hc_type_by_ma.index]
    col_order = hc_type_by_ma.sum().sort_values(ascending=False).index
    hc_type_by_ma = hc_type_by_ma[col_order]

    colors = [HARDCODING_TYPE_COLORS.get(c, "#95a5a6") for c in hc_type_by_ma.columns]
    hc_type_by_ma.columns = [_pretty_label(c) for c in hc_type_by_ma.columns]

    fig, ax = plt.subplots(figsize=(10, 5))
    hc_type_by_ma.plot(kind="barh", stacked=True, color=colors, ax=ax, width=0.7)

    for container in ax.containers:
        labels = [f"{int(v.get_width())}" if v.get_width() >= 1 else "" for v in container]
        ax.bar_label(container, labels=labels, label_type="center",
                     fontsize=11, fontweight="bold", color="white")

    ax.set_title("Hardcoding Instance Types by Model & Approach", fontsize=18, fontweight="bold", pad=12)
    ax.set_ylabel("")
    ax.set_xlabel("Runs", fontsize=18, fontweight="bold")
    apply_style(ax)
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()

    place_legend(fig, ax, fontsize=11)

    fig.tight_layout()
    save_figure(fig, "hardcoding_types_by_model_approach")
    plt.show()
else:
    print("No hardcoding instances found.")

