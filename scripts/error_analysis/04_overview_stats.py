"""Step 04: produce overview plots from enriched divergence results.

Plot types:
  1. Root-cause stacked bar charts per paper (one figure per agent).
  2. Pipeline cascade: each divergence traced from entry to attribution stage.
  3. Stage × failure-type heatmap.
  4. S-code (error type) distribution grouped bar chart.
  5. Divergence → output network (one figure per paper/agent).

data_available per divergence comes from step 01 (01_trace_failures.py).
Consistency-check verdicts come from step 02 (02_detect_error_source.py).

Usage
-----
    python 04_overview_stats.py
    python 04_overview_stats.py --workspace-dir explainer_workspaces/ --output-dir plots/
    python 04_overview_stats.py --rerun
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# Root-cause taxonomy (display order + colours)
# ---------------------------------------------------------------------------

ROOT_CAUSES = [
    "Agent ignored instructions",
    "Data not in package",
    "Summary gap",
    "Paper underspecified",
    "Unexplained",
]
RC_COLORS = {
    "Agent ignored instructions": "#e41a1c",
    "Data not in package":        "#ff7f00",
    "Summary gap":                "#377eb8",
    "Paper underspecified":       "#4daf4a",
    "Unexplained":                "#999999",
}

# Pipeline stages (x positions for cascade plot)
STAGES = [
    (1, "Data\navailable"),
    (2, "Paper\n≠ Code"),
    (3, "Paper\n≠ Summary"),
    (4, "Summary\n≠ Agent"),
]

# One colour per paper (up to 5)
PAPER_PALETTE = [
    "#e41a1c", "#377eb8", "#4daf4a", "#ff7f00", "#984ea3",
    "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62",
    "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494",
    "#b3b3b3", "#1b9e77", "#d95f02", "#7570b3", "#e7298a",
]

SEV_LABELS  = ["critical", "medium", "minor"]
SEV_COLORS  = {"critical": "#e41a1c", "medium": "#ff7f00", "minor": "#4daf4a"}

# Failure type (S-code) taxonomy order and colours
FTYPE_ORDER = ["S1", "S2", "S3", "S4", "S5", "S6", "S8", "S9", "S0"]
FTYPE_LABELS = {
    "S1": "S1 Wrong model spec.",
    "S2": "S2 Wrong estimator",
    "S3": "S3 Data substitution",
    "S4": "S4 Wrong sample",
    "S5": "S5 Wrong variable",
    "S6": "S6 Missing component",
    "S8": "S8 Wrong merge/transform",
    "S9": "S9 Wrong sequencing",
    "S0": "S0 Other",
}
FTYPE_COLORS = [
    "#e41a1c", "#ff7f00", "#4daf4a", "#377eb8",
    "#984ea3", "#a65628", "#f781bf", "#999999", "#dede00",
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _short_paper_id(paper_id: str) -> str:
    parts = paper_id.split("_", 1)
    return parts[-1] if len(parts) > 1 else paper_id


_TRIGGERS = {"contradicts", "omission"}


def _derive_source(d: dict) -> str:
    p_code  = d.get("paper_vs_original_code", "unclear")
    p_sum   = d.get("paper_vs_summary",       "unclear")
    s_agent = d.get("summary_vs_agent",       "unclear")
    data    = d.get("data_available",          None)

    if data    == "missing":       return "Data not in package"
    if p_code  in _TRIGGERS:      return "Paper underspecified"
    if p_sum   in _TRIGGERS:      return "Summary gap"
    if s_agent in _TRIGGERS:      return "Agent ignored instructions"
    return "Unexplained"


def _attribution_stage(d: dict) -> float:
    """Return the x-axis stage position where this divergence is attributed."""
    p_code  = d.get("paper_vs_original_code", "unclear")
    p_sum   = d.get("paper_vs_summary",       "unclear")
    s_agent = d.get("summary_vs_agent",       "unclear")
    data    = d.get("data_available",          None)

    if data    == "missing":      return 1   # data availability (first stage)
    if p_code  in _TRIGGERS:      return 2   # code ↔ paper
    if p_sum   in _TRIGGERS:      return 3   # paper ↔ summary
    if s_agent in _TRIGGERS:      return 4   # summary ↔ agent
    return 4.5                               # unexplained


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_workspace(ws_root: Path) -> dict[str, dict[str, list[dict]]]:
    """Returns {paper_id: {agent: [divergence_dicts]}}."""
    data: dict[str, dict[str, list[dict]]] = {}
    for paper_dir in sorted(ws_root.iterdir()):
        if not paper_dir.is_dir():
            continue
        paper_id = paper_dir.name
        for agent_dir in sorted(paper_dir.iterdir()):
            if not agent_dir.is_dir():
                continue
            agent = agent_dir.name
            for fname in ("error_source/divergences_enriched.json", "code/divergences.json"):
                p = agent_dir / fname
                if p.exists():
                    try:
                        raw  = json.loads(p.read_text(encoding="utf-8"))
                        divs = raw.get("divergences", raw.get("discrepancies", []))
                        data.setdefault(paper_id, {})[agent] = divs
                    except json.JSONDecodeError:
                        print(f"  WARNING: malformed JSON skipped: {p}")
                    break
    return data


# ---------------------------------------------------------------------------
# Plot 1 — Root-cause stacked bar charts (one figure per agent)
# ---------------------------------------------------------------------------

def _plot_root_cause_bars(
    all_data: dict,
    agents: list[str],
    papers: list[str],
    out_dir: Path,
    rerun: bool,
) -> None:
    for agent in agents:
        out_path = out_dir / f"root_causes_{agent}.pdf"
        if out_path.exists() and not rerun:
            print(f"SKIP: {out_path} (use --rerun)"); continue

        divs_per_paper = [all_data[p].get(agent, []) for p in papers]

        x       = np.arange(len(papers))
        bottoms = np.zeros(len(papers))

        fig, ax = plt.subplots(figsize=(max(5, len(papers) * 1.8), 4))

        for rc in ROOT_CAUSES:
            counts = np.array([
                sum(1 for d in divs if _derive_source(d) == rc)
                for divs in divs_per_paper
            ], dtype=float)
            if counts.sum() == 0:
                continue
            ax.bar(x, counts, bottom=bottoms, color=RC_COLORS[rc], label=rc, width=0.55)
            bottoms += counts

        ax.set_xticks(x)
        ax.set_xticklabels([_short_paper_id(p) for p in papers], rotation=15, ha="right")
        ax.set_ylabel("Number of divergences")
        ax.set_title(f"Root causes — {agent}", fontweight="bold")
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        handles = [
            mpatches.Patch(color=RC_COLORS[rc], label=rc)
            for rc in ROOT_CAUSES
            if any(_derive_source(d) == rc for divs in divs_per_paper for d in divs)
        ]
        ax.legend(handles=handles, fontsize=7, loc="upper right", framealpha=0.8)

        fig.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"-> Saved {out_path}")


# ---------------------------------------------------------------------------
# Plot 1b — Aggregated root-cause bar chart (averaged across all agents)
# ---------------------------------------------------------------------------

def _plot_root_cause_aggregate(
    all_data: dict,
    agents: list[str],
    papers: list[str],
    out_dir: Path,
    rerun: bool,
) -> None:
    out_path = out_dir / "root_causes_aggregate.pdf"
    if out_path.exists() and not rerun:
        print(f"SKIP: {out_path} (use --rerun)"); return

    # Count root causes across ALL agents
    rc_counts = {rc: 0 for rc in ROOT_CAUSES}
    total = 0
    for paper in papers:
        for agent in agents:
            for d in all_data.get(paper, {}).get(agent, []):
                src = _derive_source(d)
                rc_counts[src] = rc_counts.get(src, 0) + 1
                total += 1

    if total == 0:
        return

    # Horizontal bar chart sorted by count
    rcs = [(rc, rc_counts[rc]) for rc in ROOT_CAUSES if rc_counts[rc] > 0]
    rcs.sort(key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=(7, max(3, len(rcs) * 0.6 + 1)))
    y = np.arange(len(rcs))
    for i, (rc, count) in enumerate(rcs):
        pct = count / total * 100
        ax.barh(i, pct, color=RC_COLORS.get(rc, "#999999"), height=0.6)
        ax.text(pct + 1, i, f"{pct:.0f}% ({count})", va="center", fontsize=10,
                fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([rc for rc, _ in rcs], fontsize=11)
    ax.set_xlabel("Share of divergences (%)", fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(c / total * 100 for _, c in rcs) + 15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"-> Saved {out_path}")


# ---------------------------------------------------------------------------
# Plot 2 — KM-style pipeline survival plot
# ---------------------------------------------------------------------------

def _km_curve(stages: list[float], x_max: float = 4.0) -> tuple[list, list]:
    """
    Build (x, y) arrays for an aggregate Kaplan-Meier step curve.
    y = proportion of divergences NOT YET attributed at each pipeline stage.
    """
    n = len(stages)
    if n == 0:
        return [0, x_max], [1.0, 1.0]

    xs = [0.0]
    ys = [1.0]
    for sx in sorted({1.0, 2.0, 3.0, 4.0}):
        surviving = sum(1 for s in stages if s > sx) / n
        xs += [sx, sx]
        ys += [ys[-1], surviving]
    xs.append(x_max)
    ys.append(ys[-1])
    return xs, ys


def _plot_cascade(
    all_data: dict,
    agents: list[str],
    papers: list[str],
    out_dir: Path,
    rerun: bool,
) -> None:
    out_path = out_dir / "pipeline_cascade.pdf"
    if out_path.exists() and not rerun:
        print(f"SKIP: {out_path} (use --rerun)"); return

    paper_colors = {p: c for p, c in zip(papers, PAPER_PALETTE)}
    n_agents     = len(agents)
    x_max        = 4.2

    fig, axes = plt.subplots(1, n_agents, figsize=(5.5 * n_agents, 4), sharey=True)
    if n_agents == 1:
        axes = [axes]

    stage_labels = ["Data\navail.", "Paper\n↔ Code", "Paper\n↔ Sum.", "Sum.\n↔ Agent"]

    for ax, agent in zip(axes, agents):
        for paper in papers:
            divs   = all_data.get(paper, {}).get(agent, [])
            if not divs:
                continue
            color  = paper_colors[paper]
            stages = [min(_attribution_stage(d), 4.0) for d in divs]

            # ── Individual semi-transparent step lines (one per divergence) ──
            for s in stages:
                # Step: stay at 1.0 until attribution stage, drop to 0.0
                ax.plot(
                    [0, s, s, x_max], [1, 1, 0, 0],
                    color=color, alpha=0.18, linewidth=1.4,
                )

            # ── Bold aggregate KM curve ──────────────────────────────────────
            xs, ys = _km_curve(stages, x_max)
            ax.plot(xs, ys, color=color, linewidth=2.5, alpha=0.9,
                    label=_short_paper_id(paper))

        # Stage reference lines and x-axis labels
        for sx, slabel in STAGES:
            ax.axvline(sx, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)

        ax.set_xlim(-0.1, x_max)
        ax.set_ylim(-0.04, 1.08)
        ax.set_xticks([sx for sx, _ in STAGES])
        ax.set_xticklabels(stage_labels, fontsize=8)
        ax.set_title(agent, fontweight="bold", fontsize=11)
        ax.set_ylabel("Proportion unattributed", fontsize=8)
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
        ax.grid(axis="y", color="gray", alpha=0.2, linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Legend
    handles = [mpatches.Patch(color=paper_colors[p], label=_short_paper_id(p))
               for p in papers]
    fig.legend(handles=handles, fontsize=8, title="Paper",
               loc="lower center", ncol=len(papers), bbox_to_anchor=(0.5, -0.06))

    fig.suptitle("Pipeline survival: proportion of divergences unattributed at each check",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"-> Saved {out_path}")


# ---------------------------------------------------------------------------
# Plot 3 — Stage × failure-type heatmap (aggregated across papers)
# ---------------------------------------------------------------------------

_STAGE_FIELDS = [
    ("data_available",         "Data"),
    ("paper_vs_original_code", "P↔C"),
    ("paper_vs_summary",       "P↔S"),
    ("summary_vs_agent",       "S↔A"),
]

_FTYPE_ORDER  = ["missing", "contradicts", "omission", "unclear"]
_FTYPE_COLORS = {
    "missing":     "#984ea3",
    "contradicts": "#e41a1c",
    "omission":    "#ff7f00",
    "unclear":     "#999999",
}


def _plot_stage_heatmap(
    all_data: dict,
    agents: list[str],
    papers: list[str],
    out_dir: Path,
    rerun: bool,
) -> None:
    """
    One panel per agent.  Each panel is a heatmap: rows = papers, cols = stages.
    Each cell is a stacked mini-bar showing counts of contradicts / omission / unclear
    at that stage for that paper (only divergences where this is the *first* trigger).
    """
    out_path = out_dir / "stage_heatmap.pdf"
    if out_path.exists() and not rerun:
        print(f"SKIP: {out_path} (use --rerun)"); return

    def _first_trigger(d: dict) -> tuple[str, str] | None:
        """Return (stage_label, verdict) at the first non-consistent check, or None."""
        if d.get("data_available") == "missing":
            return "Data", "missing"
        for field, label in _STAGE_FIELDS[1:]:   # skip data_available — handled above
            v = d.get(field, "unclear")
            if v in _TRIGGERS or v == "unclear":
                return label, v
        return None  # all consistent

    n_agents = len(agents)
    stage_labels = [lbl for _, lbl in _STAGE_FIELDS]
    n_stages = len(stage_labels)
    n_papers = len(papers)

    fig, axes = plt.subplots(
        1, n_agents,
        figsize=(3.5 * n_agents + 1.5, max(2.5, n_papers * 0.55 + 1.2)),
        squeeze=False,
    )

    for ax, agent in zip(axes[0], agents):
        # counts[paper_idx][stage_idx][ftype] = int
        counts = [
            [{ft: 0 for ft in _FTYPE_ORDER} for _ in stage_labels]
            for _ in papers
        ]
        for pi, paper in enumerate(papers):
            for d in all_data.get(paper, {}).get(agent, []):
                result = _first_trigger(d)
                if result is None:
                    continue
                stage_lbl, verdict = result
                si = stage_labels.index(stage_lbl)
                if verdict in counts[pi][si]:
                    counts[pi][si][verdict] += 1

        # Draw each cell as a stacked horizontal bar
        y_positions = np.arange(n_papers)
        cell_height = 0.65

        # Compute per-cell max for normalisation (so bars fill the cell width)
        cell_max = max(
            (sum(counts[pi][si].values()) for pi in range(n_papers) for si in range(n_stages)),
            default=1,
        ) or 1

        for si, stage_lbl in enumerate(stage_labels):
            x_center = si
            for pi in range(n_papers):
                total = sum(counts[pi][si].values())
                if total == 0:
                    # Empty cell — light grey background
                    ax.barh(
                        y_positions[pi] + si * 0,  # dummy; we use scatter of rects
                        0, left=x_center - 0.4, height=cell_height,
                        color="#f0f0f0", align="center",
                    )
                    continue
                left = x_center - 0.4
                scale = 0.8 / cell_max  # normalise so max-count cell fills 80% of gap
                for ft in _FTYPE_ORDER:
                    w = counts[pi][si][ft] * scale
                    if w > 0:
                        ax.barh(
                            y_positions[pi], w, left=left,
                            height=cell_height, color=_FTYPE_COLORS[ft],
                            align="center",
                        )
                        left += w
                # Annotate total count
                ax.text(
                    x_center + 0.42, y_positions[pi], str(total),
                    va="center", ha="left", fontsize=7, color="black",
                )

        # Grid lines between stages
        for si in range(n_stages + 1):
            ax.axvline(si - 0.5, color="white", linewidth=1.5)

        ax.set_xlim(-0.5, n_stages - 0.5 + 0.6)
        ax.set_xticks(range(n_stages))
        ax.set_xticklabels(stage_labels, fontsize=8)
        ax.set_yticks(y_positions)
        ax.set_yticklabels([_short_paper_id(p) for p in papers], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(agent, fontweight="bold", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", length=0)

    # Legend
    handles = [
        mpatches.Patch(color=_FTYPE_COLORS[ft], label=ft)
        for ft in _FTYPE_ORDER
    ]
    fig.legend(handles=handles, fontsize=8, title="Failure type",
               loc="lower center", ncol=len(_FTYPE_ORDER), bbox_to_anchor=(0.5, -0.06))

    fig.suptitle(
        "Failures by pipeline stage and type (first trigger per divergence)",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"-> Saved {out_path}")


# ---------------------------------------------------------------------------
# Plot 4 — Error-type (S-code) distribution
# ---------------------------------------------------------------------------

def _plot_error_types(
    all_data: dict,
    agents: list[str],
    papers: list[str],
    out_dir: Path,
    rerun: bool,
) -> None:
    """
    Grouped + stacked bar chart.
    x-axis = S-codes  |  bars grouped by paper  |  stacked by agent (claude / codex).
    """
    out_path = out_dir / "error_types.pdf"
    if out_path.exists() and not rerun:
        print(f"SKIP: {out_path} (use --rerun)"); return

    # Collect all S-codes that actually appear
    all_codes = set()
    for paper_agents in all_data.values():
        for divs in paper_agents.values():
            for d in divs:
                code = d.get("divergence_type", "")
                if code:
                    all_codes.add(code)
    codes = [c for c in FTYPE_ORDER if c in all_codes]
    if not codes:
        print("  WARNING: no divergence_type codes found — skipping error_types plot")
        return

    color_map = {c: FTYPE_COLORS[i % len(FTYPE_COLORS)] for i, c in enumerate(codes)}

    # One subplot per agent
    n_agents = len(agents)
    fig, axes = plt.subplots(1, n_agents, figsize=(max(6, len(codes) * 1.0 * n_agents), 4),
                             sharey=True, squeeze=False)

    paper_hatches = ["", "///", "...", "xxx", "+++"]
    paper_hatch = {p: paper_hatches[i % len(paper_hatches)] for i, p in enumerate(papers)}

    for ax, agent in zip(axes[0], agents):
        x = np.arange(len(codes))
        n_papers = len(papers)
        bar_w = 0.8 / n_papers
        bottoms = np.zeros(len(codes))

        # For stacked-by-paper within each S-code
        for pi, paper in enumerate(papers):
            divs = all_data.get(paper, {}).get(agent, [])
            counts = np.array([
                sum(1 for d in divs if d.get("divergence_type", "") == c)
                for c in codes
            ], dtype=float)
            offset = (pi - n_papers / 2 + 0.5) * bar_w
            bars = ax.bar(
                x + offset, counts,
                width=bar_w * 0.92,
                color=[color_map[c] for c in codes],
                hatch=paper_hatch[paper],
                edgecolor="white",
                linewidth=0.5,
                label=_short_paper_id(paper),
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [FTYPE_LABELS.get(c, c) for c in codes],
            rotation=30, ha="right", fontsize=7,
        )
        ax.set_ylabel("Number of divergences")
        ax.set_title(agent, fontweight="bold")
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Legend: papers distinguished by hatch
    paper_handles = [
        mpatches.Patch(facecolor="gray", hatch=paper_hatch[p], edgecolor="white",
                       label=_short_paper_id(p))
        for p in papers
    ]
    fig.legend(handles=paper_handles, fontsize=7, title="Paper",
               loc="lower center", ncol=len(papers), bbox_to_anchor=(0.5, -0.06))

    fig.suptitle("Error type distribution (S-codes) by agent and paper",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"-> Saved {out_path}")


# ---------------------------------------------------------------------------
# Plot 5 — Divergence → Output network  (one figure per paper/agent)
# ---------------------------------------------------------------------------

def _draw_divergence_network(
    divs: list[dict],
    paper_id: str,
    agent: str,
    out_path: Path,
    out_dir: Path,
    enriched: bool = False,
) -> None:
    """Bipartite network: divergence nodes (left) → output nodes (right).

    Edge width ∝ number of directly-affected cells.
    Dashed edges for `also_explains` entries (partial attribution, no cell count).
    When *enriched* is True (data from divergences_enriched.json), nodes are
    coloured by root cause and the S-code is shown as a small badge.
    """
    # ── Build edge sets ──────────────────────────────────────────────────
    direct_edges: dict[tuple[int, str], int] = {}   # (div_id, item_id) -> cell count
    partial_edges: set[tuple[int, str]]      = set()
    all_outputs: list[str]                   = []

    for d in divs:
        cell_counts: dict[str, int] = {}
        for cell in (d.get("affected_cells") or []):
            iid = cell["item_id"]
            cell_counts[iid] = cell_counts.get(iid, 0) + 1

        # Always connect divergence to its primary output (even if affected_cells is empty)
        primary = d.get("output", "")
        if primary and primary not in cell_counts:
            cell_counts[primary] = 0

        for iid, cnt in cell_counts.items():
            direct_edges[(d["id"], iid)] = cnt
            if iid not in all_outputs:
                all_outputs.append(iid)

        for entry in (d.get("also_explains") or []):
            iid = entry if isinstance(entry, str) else entry.get("item_id", "")
            if iid and (d["id"], iid) not in direct_edges:
                partial_edges.add((d["id"], iid))
                if iid not in all_outputs:
                    all_outputs.append(iid)

    all_outputs.sort(key=lambda x: (0 if "Table" in x else 1, x))

    n_div  = len(divs)
    n_out  = len(all_outputs)
    n_rows = max(n_div, n_out)

    # ── Layout ───────────────────────────────────────────────────────────
    fig_h = max(6.0, n_rows * 0.95 + 2.5)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    ax.set_xlim(-0.2, 10.2)
    ax.set_ylim(-0.6, n_rows + 0.5)
    ax.axis("off")

    div_ys: dict[int, float]  = {
        d["id"]: float(y)
        for d, y in zip(divs, np.linspace(n_rows - 0.5, 0.5, n_div))
    }
    out_ys: dict[str, float] = {
        iid: float(y)
        for iid, y in zip(all_outputs, np.linspace(n_rows - 0.5, 0.5, n_out))
    }

    X_L, X_R = 2.0, 8.0   # node centre x
    W_L, W_R = 3.6, 1.8   # node full width
    H        = 0.72        # node full height
    stype_color = {c: FTYPE_COLORS[i] for i, c in enumerate(FTYPE_ORDER)}
    max_cells   = max((v for v in direct_edges.values() if v > 0), default=1)

    def _node_color(d: dict) -> str:
        if enriched:
            return RC_COLORS.get(_derive_source(d), "#999999")
        return stype_color.get(d.get("divergence_type", "S0"), "#999999")

    # ── Edges (drawn first, behind nodes) ────────────────────────────────
    for (did, iid), cnt in direct_edges.items():
        x0, y0 = X_L + W_L / 2, div_ys[did]
        x1, y1 = X_R - W_R / 2, out_ys[iid]
        lw = 0.5 + 3.5 * (cnt / max_cells) if cnt else 0.5
        ax.plot([x0, x1], [y0, y1],
                color="#777777", lw=lw, alpha=0.55, zorder=1,
                solid_capstyle="round")
        label = str(cnt) if cnt else "—"
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.04, label,
                ha="center", va="bottom", fontsize=5.5, color="#444444",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.5),
                zorder=3)

    for (did, iid) in partial_edges:
        x0, y0 = X_L + W_L / 2, div_ys[did]
        x1, y1 = X_R - W_R / 2, out_ys[iid]
        ax.plot([x0, x1], [y0, y1],
                color="#aaaaaa", lw=0.7, linestyle=(0, (4, 3)),
                alpha=0.5, zorder=1)

    # ── Divergence nodes (left column) ───────────────────────────────────
    for d in divs:
        y   = div_ys[d["id"]]
        col = _node_color(d)
        ax.add_patch(mpatches.FancyBboxPatch(
            (X_L - W_L / 2, y - H / 2), W_L, H,
            boxstyle="round,pad=0.04",
            facecolor=col, alpha=0.18, edgecolor=col, lw=1.2, zorder=2,
        ))
        dtype = d.get("divergence_type", "?")
        desc  = d.get("description", "")
        desc_trunc = desc[:55] + ("…" if len(desc) > 55 else "")
        if enriched:
            rc_short = _derive_source(d)
            header = f"D{d['id']}  [{dtype}]  {rc_short}"
        else:
            header = f"D{d['id']}  [{dtype}]"
        ax.text(X_L - W_L / 2 + 0.12, y + 0.10,
                header,
                ha="left", va="bottom", fontsize=6.5, fontweight="bold",
                color="#111111", zorder=3)
        ax.text(X_L - W_L / 2 + 0.12, y - 0.08,
                desc_trunc,
                ha="left", va="top", fontsize=4.8,
                color="#444444", zorder=3)

    # ── Output nodes (right column) ───────────────────────────────────────
    for iid in all_outputs:
        y        = out_ys[iid]
        n_direct = sum(cnt for (did, item), cnt in direct_edges.items() if item == iid)
        n_part   = sum(1 for (did, item) in partial_edges if item == iid)
        ax.add_patch(mpatches.FancyBboxPatch(
            (X_R - W_R / 2, y - H / 2), W_R, H,
            boxstyle="round,pad=0.04",
            facecolor="#aec6e8", alpha=0.55, edgecolor="#4472c4", lw=1.2, zorder=2,
        ))
        sub = f"{n_direct} cells" + (f"  +{n_part}▸" if n_part else "")
        ax.text(X_R, y + 0.08, iid,
                ha="center", va="bottom", fontsize=7, fontweight="bold",
                color="#1a3d6e", zorder=3)
        ax.text(X_R, y - 0.08, sub,
                ha="center", va="top", fontsize=5.5, color="#555555", zorder=3)

    # ── Column headers ────────────────────────────────────────────────────
    ax.text(X_L, n_rows + 0.15, "Code Divergences",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.text(X_R, n_rows + 0.15, "Outputs Affected",
            ha="center", va="bottom", fontsize=9, fontweight="bold")

    # ── Legend ────────────────────────────────────────────────────────────
    if enriched:
        used_rc = [rc for rc in ROOT_CAUSES if any(_derive_source(d) == rc for d in divs)]
        color_patches = [
            mpatches.Patch(facecolor=RC_COLORS[rc], alpha=0.4,
                           edgecolor=RC_COLORS[rc], label=rc)
            for rc in used_rc
        ]
    else:
        used = sorted(
            {d.get("divergence_type", "S0") for d in divs},
            key=lambda c: FTYPE_ORDER.index(c) if c in FTYPE_ORDER else 99,
        )
        color_patches = [
            mpatches.Patch(facecolor=stype_color.get(c, "#999999"), alpha=0.4,
                           edgecolor=stype_color.get(c, "#999999"),
                           label=FTYPE_LABELS.get(c, c))
            for c in used
        ]
    leg = color_patches + [
        plt.Line2D([0], [0], color="#777777", lw=2.5, alpha=0.7,
                   label="Direct  (width ∝ cell count)"),
        plt.Line2D([0], [0], color="#aaaaaa", lw=0.8, linestyle=(0, (4, 3)),
                   label="Partial  (also_explains)"),
    ]
    ax.legend(handles=leg, fontsize=6, loc="lower center",
              bbox_to_anchor=(0.5, -0.01), ncol=min(5, len(leg)), framealpha=0.85)

    fig.suptitle(
        f"Divergence–Output Network  ·  {_short_paper_id(paper_id)} / {agent}",
        fontsize=10, fontweight="bold", y=1.0,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"-> Saved {out_path}")


def _plot_divergence_networks(ws_root: Path, out_dir: Path, rerun: bool) -> None:
    """Generate one network figure per paper/agent.

    Prefers ``divergences_enriched.json`` (step 02 output, includes root-cause
    verdicts) and falls back to ``divergences.json`` (step 01 output).
    """
    for paper_dir in sorted(ws_root.iterdir()):
        if not paper_dir.is_dir():
            continue
        for agent_dir in sorted(paper_dir.iterdir()):
            if not agent_dir.is_dir():
                continue
            # Prefer enriched output (has root-cause verdicts); fall back to raw
            enriched_path = agent_dir / "error_source" / "divergences_enriched.json"
            raw_path      = agent_dir / "code" / "divergences.json"
            if enriched_path.exists():
                div_path, enriched = enriched_path, True
            elif raw_path.exists():
                div_path, enriched = raw_path, False
            else:
                continue
            out_path = out_dir / f"div_network_{paper_dir.name}_{agent_dir.name}.pdf"
            if out_path.exists() and not rerun:
                print(f"SKIP: {out_path} (use --rerun)")
                continue
            try:
                data = json.loads(div_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                print(f"WARNING: malformed JSON skipped: {div_path}")
                continue
            divs = data.get("divergences", data.get("discrepancies", []))
            if not divs:
                continue
            _draw_divergence_network(
                divs, paper_dir.name, agent_dir.name, out_path, out_dir,
                enriched=enriched,
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Plot root-cause bar charts and pipeline cascade."
    )
    parser.add_argument("--workspace-dir", default=str(here / "explainer_workspaces"))
    parser.add_argument("--output-dir",    default=str(here / "plots"))
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def main() -> None:
    args    = parse_args()
    ws_root = Path(args.workspace_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()

    if not ws_root.is_dir():
        import sys; sys.exit(f"ERROR: {ws_root} does not exist")

    all_data = _load_workspace(ws_root)
    if not all_data:
        import sys; sys.exit(f"ERROR: no divergence files found under {ws_root}")

    agents = sorted({a for p in all_data.values() for a in p})
    papers = sorted(all_data.keys())

    total = sum(len(d) for p in all_data.values() for d in p.values())
    print(f"Papers: {len(papers)}  Agents: {agents}  Divergences: {total}\n")

    _plot_root_cause_bars(all_data, agents, papers, out_dir, args.rerun)
    _plot_root_cause_aggregate(all_data, agents, papers, out_dir, args.rerun)
    _plot_cascade(all_data, agents, papers, out_dir, args.rerun)
    _plot_stage_heatmap(all_data, agents, papers, out_dir, args.rerun)
    _plot_error_types(all_data, agents, papers, out_dir, args.rerun)
    _plot_divergence_networks(ws_root, out_dir, args.rerun)
    print("\nDone.")


if __name__ == "__main__":
    main()
