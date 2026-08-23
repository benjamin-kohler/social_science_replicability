"""Extractor-variant figures used by the paper.

This is the analysis-only portion of the private extractor-variant worktree.
It does not include the experimental extractor implementation.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import analyze_i4rep_results as _core

# Reuse the corrected production loader, grade rules, styles, and constants,
# including private helpers used by the original analysis implementation.
globals().update({
    name: value for name, value in vars(_core).items()
    if not name.startswith("__")
})

# ============================================================================
# Extractor-variant analysis (markdown methods documents vs structured summary)
# ============================================================================
# Compares a variant results root (same papers, same judge, different methods
# extraction) against the main run, using the stability runs as the re-run
# noise yardstick. Sources are encoded by BOTH color and linestyle/marker so
# identity is never color-alone: control = neutral gray dashed / open marker,
# variant = the approach's own color solid / filled marker, stability = thin
# light-gray lines / tick marks.

CONTROL_COLOR = "#555555"
STABILITY_COLOR = "#b8b8b8"
DELTA_WORSE_COLOR = GRADE_COLORS["E"]    # red
DELTA_BETTER_COLOR = GRADE_COLORS["A"]   # green
DELTA_SAME_COLOR = "#95a5a6"


def _paper_short_label(slug: str) -> str:
    """Short human-readable paper label: journal + slug tail."""
    tail = slug.split("_")[-1][-8:]
    return f"{_infer_journal(slug)} {tail}"


def _variant_diff_over_se(df_cells: pd.DataFrame) -> pd.DataFrame:
    """|coeff difference| / SE per cell (same filter as plot_coefficient_se_cdf)."""
    df = df_cells[
        (df_cells["row_type"] == "coefficient") &
        df_cells["original_value"].notna() &
        df_cells["replicated_value"].notna() &
        df_cells["is_numeric"] &
        (df_cells["item_grade"] != "NA")
    ].copy()
    if df.empty:
        return df
    df["abs_diff"] = (df["original_value"].astype(float) - df["replicated_value"].astype(float)).abs()
    df["se"] = df["original_se"]
    mask = df["se"].isna()
    df.loc[mask, "se"] = df.loc[mask, "replicated_se"]
    df = df[df["se"].notna() & (df["se"].astype(float) > 0)].copy()
    if not df.empty:
        df["diff_over_se"] = df["abs_diff"] / df["se"].astype(float)
    return df


# Label used for the markdown-extraction line/marker in every variant figure.
VARIANT_LINE_LABEL = "Method Extractor Variant"


def _variant_legend(fig, handles, ncol=None, fontsize=11, legend_h=1.1):
    """Finalize layout and place the legend in a reserved band below the plots.

    Grows the figure by `legend_h` inches (so the plot region keeps its size,
    not compressed) and runs tight_layout with that band reserved, so the axes'
    x-labels stay inside the plot region and the framed legend sits cleanly
    below them — no overlap regardless of figure aspect. Style matches
    place_legend (lower-center, framed, fancybox).
    """
    if ncol is None:
        ncol = min(len(handles), 3)
    H = fig.get_figheight()
    new_H = H + legend_h
    fig.set_figheight(new_H)
    band = legend_h / new_H
    fig.tight_layout(rect=(0, band, 1, 1))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, band * 0.04),
               ncol=ncol, fontsize=fontsize, frameon=True, fancybox=True)


def plot_variant_coefficient_se_cdf(
    cells_by_source: dict[str, pd.DataFrame],
    output_dir: Path, subdir: str = "",
):
    """Viz 1: CDF of |coeff diff|/SE — control vs variant, stability runs as noise band.

    cells_by_source: {"control": df, "variant": df, "stability_1": df, ...}
    One panel per approach present in the variant.
    """
    variant_cells = cells_by_source.get("variant")
    if variant_cells is None or variant_cells.empty:
        return
    per_source = {k: _variant_diff_over_se(v) for k, v in cells_by_source.items()
                  if v is not None and not v.empty}
    approaches = [a for a in APPROACH_ORDER
                  if a in per_source["variant"]["approach"].astype(str).unique()]
    if not approaches:
        return

    x_max = 10
    n = len(approaches)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.9 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax_idx, approach in enumerate(approaches):
        ax = axes[ax_idx]
        color = APPROACH_COLORS.get(approach, "#95a5a6")
        # Stability runs first (thin, light, underneath)
        for src, df in per_source.items():
            if not src.startswith("stability"):
                continue
            vals = df.loc[df["approach"].astype(str) == approach, "diff_over_se"].sort_values().values
            if len(vals) == 0:
                continue
            ax.plot(vals, np.arange(1, len(vals) + 1) / len(vals) * 100,
                    color=STABILITY_COLOR, linewidth=1.1, zorder=1)
        # Control (gray dashed) and variant (approach color solid)
        for src, lcolor, lstyle, lwidth, z in (
            ("control", CONTROL_COLOR, "--", 2.0, 2),
            ("variant", color, "-", 2.4, 3),
        ):
            df = per_source.get(src)
            if df is None:
                continue
            vals = df.loc[df["approach"].astype(str) == approach, "diff_over_se"].sort_values().values
            if len(vals) == 0:
                continue
            ax.plot(vals, np.arange(1, len(vals) + 1) / len(vals) * 100,
                    color=lcolor, linestyle=lstyle, linewidth=lwidth, zorder=z)
        ax.axvline(x=1.96, color="red", linestyle=":", alpha=0.4, linewidth=1)
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, 105)
        ax.set_title(APPROACH_LABELS.get(approach, approach).replace("\n", " "),
                     fontsize=12, fontweight="bold")
        # House style: x-label on every panel, y-label on the left column only.
        ax.set_xlabel("|Coeff. difference| / SE", fontsize=13, fontweight="bold")
        if ax_idx % ncols == 0:
            ax.set_ylabel("Cumulative share of coefficients (%)",
                          fontsize=12, fontweight="bold")
        apply_style(ax)
    for ax_idx in range(len(approaches), len(axes)):
        axes[ax_idx].set_visible(False)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color=CONTROL_COLOR, linestyle="--", linewidth=2,
               label="Control (structured extraction)"),
        Line2D([], [], color="#333333", linestyle="-", linewidth=2.4,
               label=VARIANT_LINE_LABEL),
        Line2D([], [], color=STABILITY_COLOR, linestyle="-", linewidth=1.1,
               label="Stability re-runs (structured)"),
    ]
    _variant_legend(fig, handles, ncol=3)
    save_figure(fig, output_dir, "variant_coefficient_se_cdf", subdir)


def plot_variant_grade_dumbbell(
    runs_by_source: dict[str, pd.DataFrame],
    output_dir: Path, subdir: str = "",
):
    """Viz 4: paper-level dumbbell — control grade → variant grade per paper × approach.

    Stability-run grades are drawn as light tick marks (the re-run noise band).
    """
    df_v = runs_by_source.get("variant")
    df_c = runs_by_source.get("control")
    if df_v is None or df_v.empty or df_c is None or df_c.empty:
        return
    approaches = [a for a in APPROACH_ORDER
                  if a in df_v["approach"].astype(str).unique()]
    papers = sorted(df_v["paper_slug"].unique())
    if not approaches or not papers:
        return

    def _grade_lookup(df):
        out = {}
        for _, r in df.iterrows():
            g = GRADE_TO_NUM.get(str(r.get("overall_grade")))
            if g is not None:
                out[(r["paper_slug"], str(r["approach"]))] = g
        return out

    lookups = {src: _grade_lookup(df) for src, df in runs_by_source.items()
               if df is not None and not df.empty}

    ncols = min(3, len(approaches))
    nrows = int(np.ceil(len(approaches) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.9 * ncols, (0.42 * len(papers) + 1.4) * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    y_pos = {p: i for i, p in enumerate(papers)}
    # x-label goes on the bottom-most visible panel of each column (house style).
    last_row_start = (nrows - 1) * ncols

    for ax_idx, approach in enumerate(approaches):
        ax = axes[ax_idx]
        color = APPROACH_COLORS.get(approach, "#95a5a6")
        for p in papers:
            y = y_pos[p]
            c = lookups["control"].get((p, approach))
            v = lookups["variant"].get((p, approach))
            # Stability ticks (noise band) underneath
            for src, lk in lookups.items():
                if not src.startswith("stability"):
                    continue
                s = lk.get((p, approach))
                if s is not None:
                    ax.plot(s, y, marker="|", markersize=13, markeredgewidth=2.2,
                            color=STABILITY_COLOR, zorder=1)
            if c is None and v is None:
                continue
            if c is not None and v is not None and c != v:
                lcolor = DELTA_WORSE_COLOR if v < c else DELTA_BETTER_COLOR
                ax.plot([c, v], [y, y], color=lcolor, linewidth=2.4,
                        solid_capstyle="round", zorder=2, alpha=0.85)
            if c is not None:
                ax.plot(c, y, marker="o", markersize=9, markerfacecolor="white",
                        markeredgecolor=CONTROL_COLOR, markeredgewidth=1.8, zorder=3)
            if v is not None:
                ax.plot(v, y, marker="o", markersize=9, color=color,
                        markeredgecolor="white", markeredgewidth=1.2, zorder=4)
        ax.set_title(APPROACH_LABELS.get(approach, approach).replace("\n", " "),
                     fontsize=12, fontweight="bold")
        ax.set_xlim(-0.5, 5.5)
        ax.set_xticks(range(6))
        ax.set_xticklabels([NUM_TO_GRADE[i] for i in range(6)])
        ax.set_yticks(range(len(papers)))
        ax.set_yticklabels([_paper_short_label(p) for p in papers], fontsize=9)
        ax.invert_yaxis()
        if ax_idx >= last_row_start:
            ax.set_xlabel("Overall paper grade", fontsize=13, fontweight="bold")
        apply_style(ax)
    for ax_idx in range(len(approaches), len(axes)):
        axes[ax_idx].set_visible(False)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="o", markersize=9, markerfacecolor="white",
               markeredgecolor=CONTROL_COLOR, markeredgewidth=1.8, linestyle="none",
               label="Control (structured)"),
        Line2D([], [], marker="o", markersize=9, color="#333333",
               linestyle="none", label=VARIANT_LINE_LABEL),
        Line2D([], [], marker="|", markersize=12, markeredgewidth=2.2,
               color=STABILITY_COLOR, linestyle="none", label="Stability re-runs"),
        Line2D([], [], color=DELTA_WORSE_COLOR, linewidth=2.4, label="Worse under variant"),
        Line2D([], [], color=DELTA_BETTER_COLOR, linewidth=2.4, label="Better under variant"),
    ]
    _variant_legend(fig, handles, ncol=3, legend_h=1.5)
    save_figure(fig, output_dir, "variant_grade_dumbbell", subdir)


def plot_variant_grade_transitions(
    items_by_source: dict[str, pd.DataFrame],
    output_dir: Path, subdir: str = "",
):
    """Viz 5: table-level grade transition heatmaps.

    Left: control → variant transitions. Right: stability run-to-run transitions
    (the noise reference), restricted to the approaches that have stability runs.
    Row-normalized shares (%), annotated with counts, single-hue ramp.
    """
    grades = [g for g in GRADE_ORDER if g != "NA"]

    def _item_grades(df):
        sub = df[df["item_type"] == "table"]
        out = {}
        for _, r in sub.iterrows():
            g = str(r["grade"])
            if g in grades:
                out[(r["paper_slug"], str(r["approach"]), r["item_id"])] = g
        return out

    lookups = {src: _item_grades(df) for src, df in items_by_source.items()
               if df is not None and not df.empty}
    if "control" not in lookups or "variant" not in lookups:
        return

    def _transition_matrix(pairs):
        mat = pd.DataFrame(0, index=grades, columns=grades, dtype=int)
        for a, b in pairs:
            mat.loc[a, b] += 1
        return mat

    # Control → variant pairs
    cv_pairs = [(g, lookups["variant"][k]) for k, g in lookups["control"].items()
                if k in lookups["variant"]]
    mat_cv = _transition_matrix(cv_pairs)

    # Stability reference: control → each stability run, on approaches with stability data
    stab_sources = [s for s in lookups if s.startswith("stability")]
    stab_approaches = set()
    for s in stab_sources:
        stab_approaches |= {k[1] for k in lookups[s]}
    ss_pairs = []
    for s in stab_sources:
        for k, g in lookups["control"].items():
            if k[1] in stab_approaches and k in lookups[s]:
                ss_pairs.append((g, lookups[s][k]))
    mat_ss = _transition_matrix(ss_pairs)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    for ax, mat, title in (
        (axes[0], mat_cv, f"Control → Variant (n={mat_cv.values.sum()})"),
        (axes[1], mat_ss, f"Control → Stability re-runs (n={mat_ss.values.sum()})"),
    ):
        row_sums = mat.sum(axis=1).replace(0, np.nan)
        pct = mat.div(row_sums, axis=0) * 100
        sns.heatmap(pct, ax=ax, cmap="Blues", vmin=0, vmax=100,
                    annot=mat.values, fmt="d", annot_kws={"fontsize": 10},
                    cbar=(ax is axes[1]),
                    cbar_kws={"label": "Share of row (%)"} if ax is axes[1] else None,
                    linewidths=1, linecolor="white", square=True)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Grade after", fontsize=12, fontweight="bold")
        ax.set_ylabel("Grade before (control)" if ax is axes[0] else "", fontsize=12,
                      fontweight="bold")
        ax.tick_params(labelsize=11)
    plt.tight_layout()
    save_figure(fig, output_dir, "variant_grade_transitions", subdir)


def plot_variant_grade_delta_bars(
    runs_by_source: dict[str, pd.DataFrame],
    output_dir: Path, subdir: str = "",
):
    """Viz 6: paper-level grade change (variant − control) in grade steps,
    with whiskers spanning what stability re-runs alone produce."""
    df_v = runs_by_source.get("variant")
    df_c = runs_by_source.get("control")
    if df_v is None or df_v.empty or df_c is None or df_c.empty:
        return

    def _grade_lookup(df):
        out = {}
        for _, r in df.iterrows():
            g = GRADE_TO_NUM.get(str(r.get("overall_grade")))
            if g is not None:
                out[(r["paper_slug"], str(r["approach"]))] = g
        return out

    lookups = {src: _grade_lookup(df) for src, df in runs_by_source.items()
               if df is not None and not df.empty}
    approaches = [a for a in APPROACH_ORDER
                  if a in df_v["approach"].astype(str).unique()]
    papers = sorted(df_v["paper_slug"].unique())
    if not approaches or not papers:
        return

    ncols = min(3, len(approaches))
    nrows = int(np.ceil(len(approaches) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.9 * ncols, 3.6 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    x = np.arange(len(papers))

    for ax_idx, approach in enumerate(approaches):
        ax = axes[ax_idx]
        deltas, colors = [], []
        stab_lo, stab_hi = [], []
        for p in papers:
            c = lookups["control"].get((p, approach))
            v = lookups["variant"].get((p, approach))
            d = (v - c) if (c is not None and v is not None) else np.nan
            deltas.append(d)
            colors.append(DELTA_SAME_COLOR if (pd.isna(d) or d == 0)
                          else (DELTA_WORSE_COLOR if d < 0 else DELTA_BETTER_COLOR))
            svals = [lk.get((p, approach)) for s, lk in lookups.items()
                     if s.startswith("stability")]
            svals = [s for s in svals if s is not None]
            if svals and c is not None:
                stab_lo.append(min(svals) - c)
                stab_hi.append(max(svals) - c)
            else:
                stab_lo.append(np.nan)
                stab_hi.append(np.nan)
        ax.bar(x, deltas, color=colors, edgecolor="white", width=0.65, zorder=2)
        # Stability whiskers: what re-runs alone do to the same control grade
        for xi, (lo, hi) in enumerate(zip(stab_lo, stab_hi)):
            if not (pd.isna(lo) or pd.isna(hi)):
                ax.plot([xi, xi], [lo, hi], color="#333333", linewidth=1.4,
                        zorder=3, alpha=0.8)
                ax.plot([xi - 0.14, xi + 0.14], [lo, lo], color="#333333",
                        linewidth=1.2, zorder=3, alpha=0.8)
                ax.plot([xi - 0.14, xi + 0.14], [hi, hi], color="#333333",
                        linewidth=1.2, zorder=3, alpha=0.8)
        ax.axhline(0, color="#888888", linewidth=1)
        ax.set_title(APPROACH_LABELS.get(approach, approach).replace("\n", " "),
                     fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([_paper_short_label(p) for p in papers],
                           fontsize=8, rotation=45, ha="right")
        if ax_idx % ncols == 0:
            ax.set_ylabel("Grade change vs control (steps)",
                          fontsize=12, fontweight="bold")
        apply_style(ax)
    for ax_idx in range(len(approaches), len(axes)):
        axes[ax_idx].set_visible(False)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [
        Patch(color=DELTA_WORSE_COLOR, label="Worse under variant"),
        Patch(color=DELTA_BETTER_COLOR, label="Better under variant"),
        Patch(color=DELTA_SAME_COLOR, label="Unchanged"),
        Line2D([], [], color="#333333", linewidth=1.4,
               label="Range from stability re-runs (structured)"),
    ]
    _variant_legend(fig, handles, ncol=4)
    save_figure(fig, output_dir, "variant_grade_delta_bars", subdir)


def _run_extractor_variant_analysis(
    df_runs_main: pd.DataFrame,
    df_items_main: pd.DataFrame,
    df_cells_main: pd.DataFrame,
    output_dir: Path,
    subdir: str,
    variant_dir: Path,
    stability_dirs: list[Path] | None = None,
    papers_dir: Path | None = None,
    regrade_na: bool = True,
    approaches: list[str] | None = None,
):
    """Compare an extractor-variant results root against the main run.

    Loads the variant (and stability runs) with the same load_results() +
    regrade pipeline as the main run, restricts everything to the variant's
    paper sample, and produces:
      - variant_coefficient_se_cdf: cell-level CDF, control vs variant vs stability
      - variant_grade_dumbbell: paper-level paired grades
      - variant_grade_transitions: table-level transition matrices vs noise reference
      - variant_grade_delta_bars: paper-level grade deltas with stability whiskers
      - variant_vs_control_papers.csv / variant_vs_control_items.csv

    If `approaches` is given (list of approach keys like "claude-code/claude-opus-4-6"),
    every source is restricted to those arms. Passing the arms that also have
    stability runs (claude-code + codex) makes the noise reference genuinely
    apples-to-apples with the variant/control transition matrix.
    """
    target = output_dir / subdir
    target.mkdir(parents=True, exist_ok=True)
    approaches_set = set(approaches) if approaches else None

    def _load(root: Path):
        dr, di, dc = load_results(root, papers_dir)
        if dr.empty:
            return None
        if regrade_na:
            dr, di, dc = regrade_with_na(dr, di, dc)
        return dr, di, dc

    loaded_v = _load(variant_dir)
    if loaded_v is None:
        print(f"  No variant results found at {variant_dir}")
        return
    df_runs_v, df_items_v, df_cells_v = loaded_v
    variant_papers = set(df_runs_v["paper_slug"].unique())

    def _restrict(df):
        out = df[df["paper_slug"].isin(variant_papers)].copy()
        if approaches_set is not None:
            out = out[out["approach"].astype(str).isin(approaches_set)].copy()
        return out

    # Also filter the variant frames themselves when arms are specified.
    if approaches_set is not None:
        df_runs_v = df_runs_v[df_runs_v["approach"].astype(str).isin(approaches_set)].copy()
        df_items_v = df_items_v[df_items_v["approach"].astype(str).isin(approaches_set)].copy()
        df_cells_v = df_cells_v[df_cells_v["approach"].astype(str).isin(approaches_set)].copy()
        print(f"  Restricting to arms: {sorted(approaches_set)}")
    print(f"  Variant sample: {len(variant_papers)} papers, {len(df_runs_v)} runs")

    runs_by_source = {"control": _restrict(df_runs_main), "variant": df_runs_v}
    items_by_source = {"control": _restrict(df_items_main), "variant": df_items_v}
    cells_by_source = {"control": _restrict(df_cells_main), "variant": df_cells_v}

    for i, sdir in enumerate(stability_dirs or [], start=1):
        if not Path(sdir).exists():
            continue
        loaded_s = _load(Path(sdir))
        if loaded_s is None:
            continue
        dr, di, dc = loaded_s
        runs_by_source[f"stability_{i}"] = _restrict(dr)
        items_by_source[f"stability_{i}"] = _restrict(di)
        cells_by_source[f"stability_{i}"] = _restrict(dc)
    n_stab = sum(1 for k in runs_by_source if k.startswith("stability"))
    print(f"  Noise reference: {n_stab} stability runs")

    # Paired CSVs for downstream use
    recs = []
    for src, df in runs_by_source.items():
        for _, r in df.iterrows():
            recs.append({"source": src, "paper_slug": r["paper_slug"],
                         "approach": str(r["approach"]),
                         "overall_grade": str(r.get("overall_grade"))})
    pd.DataFrame(recs).to_csv(target / "variant_vs_control_papers.csv", index=False)
    recs = []
    for src, df in items_by_source.items():
        sub = df[df["item_type"] == "table"]
        for _, r in sub.iterrows():
            recs.append({"source": src, "paper_slug": r["paper_slug"],
                         "approach": str(r["approach"]), "item_id": r["item_id"],
                         "grade": str(r["grade"])})
    pd.DataFrame(recs).to_csv(target / "variant_vs_control_items.csv", index=False)
    print(f"  Saved {subdir}/variant_vs_control_papers.csv, variant_vs_control_items.csv")

    plot_variant_coefficient_se_cdf(cells_by_source, output_dir, subdir)
    plot_variant_grade_dumbbell(runs_by_source, output_dir, subdir)
    plot_variant_grade_transitions(items_by_source, output_dir, subdir)
    plot_variant_grade_delta_bars(runs_by_source, output_dir, subdir)
