#!/usr/bin/env python3
"""Analyze i4rep benchmark results and generate publication-quality plots.

Usage:
    python scripts/analyze_i4rep_results.py
    python scripts/analyze_i4rep_results.py --results-dir path/to/results --output-dir plots/
"""

import argparse
import json
import os
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

# OLD: GRADE_ORDER = ["A", "B", "C", "D", "E", "F"]
GRADE_ORDER = ["A", "B", "C", "D", "E", "F", "NA"]
# OLD: GRADE_TO_NUM = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1, "F": 0}
GRADE_TO_NUM = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1, "F": 0, "NA": None}
NUM_TO_GRADE = {v: k for k, v in GRADE_TO_NUM.items() if v is not None}

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
    "codex/gpt-5.3-codex": "Codex CLI\nGPT-5.3 Codex",
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
    "NA": "#bdc3c7",  # light gray — not assessable
}

DEFAULT_BASE = Path(os.environ.get("I4REPLICATE_BASE", "data/i4replicate"))

# Journal mapping from DOI prefixes
JOURNAL_MAP = {
    # Paper-specific overrides — checked first via startswith match. The
    # generic prefixes below would otherwise capture them.
    "10.1163_2210-7975_hrd-9985-20180068": "REStud",   # Alesina/Miano/Stantcheva, REStud 2023
    "10.2139_ssrn.3838127":                 "AJPS",     # "Entertaining Beliefs...", AJPS 2022
    # Generic DOI-prefix matches
    "10.1017_s00030554": "APSR",
    "10.1086_71": "JOP",
    "10.1093_ej_": "EJ",
    "10.1093_restud_": "REStud",
    "10.1093_qje_": "QJE",
    "10.1111_ajps": "AJPS",
    "10.1257_aer": "AER",
    "10.1257_aeri": "AER:I",
    "10.1257_app": "AEJ:AP",
    "10.1257_mac": "AEJ:Mac",
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


def apply_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=14)


def place_legend(fig, ax, ncol=None, fontsize=12, title=None, **kwargs):
    """Place legend below the plot area."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    # Remove any existing legend on the axes
    leg = ax.get_legend()
    if leg:
        leg.remove()
    if ncol is None:
        ncol = min(len(handles), 4)
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.08), ncol=ncol,
               fontsize=fontsize, title=title,
               frameon=True, fancybox=True, **kwargs)


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


def count_decimals(x) -> int | None:
    """Number of decimal digits in the shortest round-trippable form of ``x``.

    Returns None for None/NaN/inf, 0 for integers. Handles scientific notation
    (e.g. ``1e-05`` → 5 decimals). Trailing zeros in the float's repr are
    stripped since repr already gives the shortest form, so ``0.50`` → 1.
    """
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if np.isnan(f) or np.isinf(f):
        return None
    s = repr(f)
    if "e" in s or "E" in s:
        mantissa, _, exp_str = s.lower().partition("e")
        exp = int(exp_str)
        if "." in mantissa:
            mant_frac = mantissa.split(".", 1)[1].rstrip("0")
        else:
            mant_frac = ""
        return max(0, len(mant_frac) - exp)
    if "." in s:
        frac = s.split(".", 1)[1].rstrip("0")
        return len(frac)
    return 0


def apply_adaptive_rounding(df_cells: pd.DataFrame) -> pd.DataFrame:
    """Add columns that round the replicated value to the original's precision.

    For each numeric cell, counts decimal digits of ``original_value`` and uses
    :func:`numpy.round` to round both original and replicated to that many
    digits, then recomputes percent / absolute differences. Plots suffixed
    ``_rounded`` are produced against these new columns.
    """
    if df_cells.empty:
        return df_cells

    orig = pd.to_numeric(df_cells["original_value"], errors="coerce").to_numpy()
    repl = pd.to_numeric(df_cells["replicated_value"], errors="coerce").to_numpy()

    n_dec_series = df_cells["original_value"].map(count_decimals)
    nd_missing = n_dec_series.isna().to_numpy()
    nd_int = n_dec_series.fillna(0).astype(int).to_numpy()
    scale = np.power(10.0, nd_int)

    # Vectorized round-to-n-decimals via scale/unscale — equivalent to
    # np.round(x, d) applied row-wise, but avoids a Python loop over 80k+ rows.
    orig_round = np.round(orig * scale) / scale
    repl_round = np.round(repl * scale) / scale
    orig_round[nd_missing] = np.nan
    repl_round[nd_missing] = np.nan

    abs_diff = np.abs(repl_round - orig_round)
    with np.errstate(divide="ignore", invalid="ignore"):
        # Unsigned percent diff, matching the convention used elsewhere
        # (df_cells["percent_difference"] is stored unsigned in verification reports).
        pct_diff = np.where(
            orig_round != 0,
            100.0 * abs_diff / np.abs(orig_round),
            np.nan,
        )
    # If both rounded to the same value (incl. both zero), pct_diff is 0.
    same = (orig_round == repl_round) & ~np.isnan(orig_round) & ~np.isnan(repl_round)
    pct_diff = np.where(same, 0.0, pct_diff)

    df_cells = df_cells.copy()
    df_cells["n_decimals"] = n_dec_series
    df_cells["original_value_rounded"] = orig_round
    df_cells["replicated_value_rounded"] = repl_round
    df_cells["absolute_difference_rounded"] = abs_diff
    df_cells["percent_difference_rounded"] = pct_diff
    return df_cells


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


# Match Python imports: "import foo" / "import foo, bar" / "from foo import ...".
# Captures only the module expression; we split and take the top-level token below.
_PY_IMPORT_RE = re.compile(
    r"^\s*(?:"
    r"from\s+(?P<from_mod>[\w.]+)\s+import"
    r"|"
    r"import\s+(?P<imp_mods>[\w.]+(?:\s*,\s*[\w.]+)*)"
    r")",
    re.MULTILINE,
)


def _analyze_agent_code(workspace_dir: Path) -> tuple[int, list[str]]:
    """Return ``(total_loc, sorted_unique_top_level_imports)`` for the agent's
    Python output in ``workspace_dir``.

    LOC is a naive line count across every ``.py`` file under the workspace.
    Imports are extracted via ``ast.parse`` (so ``import numpy as np, scipy``
    yields both ``numpy`` and ``scipy``). Falls back to a regex if a file
    fails to parse (agents sometimes produce syntactically broken code).
    Both paths reduce to top-level package names and skip relative imports.
    """
    import ast as _ast
    if not workspace_dir.exists():
        return 0, []
    total_loc = 0
    imports: set[str] = set()
    for py in workspace_dir.rglob("*.py"):
        try:
            text = py.read_text(errors="ignore")
        except OSError:
            continue
        if text:
            total_loc += text.count("\n") + (0 if text.endswith("\n") else 1)
        try:
            tree = _ast.parse(text)
        except (SyntaxError, ValueError):
            for m in _PY_IMPORT_RE.finditer(text):
                mods = m.group("from_mod") or m.group("imp_mods") or ""
                for mod in mods.split(","):
                    mod = mod.strip()
                    if not mod or mod.startswith("."):
                        continue
                    imports.add(mod.split(".")[0])
            continue
        for node in _ast.walk(tree):
            if isinstance(node, _ast.Import):
                for alias in node.names:
                    if alias.name:
                        imports.add(alias.name.split(".")[0])
            elif isinstance(node, _ast.ImportFrom):
                if (node.level or 0) > 0:
                    continue
                if node.module:
                    imports.add(node.module.split(".")[0])
    return total_loc, sorted(imports)


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
    """Parse OpenCode run_log.jsonl for total token usage.

    Each ``step_finish`` event in the JSONL log contains a ``tokens`` dict
    with ``input`` (uncached), ``cache.read``, ``output``, and ``reasoning``
    fields, plus a ``cost`` float.  We sum ``input + cache.read`` to get the
    full context sent per API call — comparable to how Codex reports
    ``input_tokens`` (which includes cached tokens).

    Falls back to the OpenCode SQLite DB if the JSONL log is not found.
    """
    ws = Path(workspace_dir)
    jsonl = ws / "run_log.jsonl"
    if jsonl.exists():
        try:
            total_input = 0     # uncached input
            total_cache = 0     # cache-read tokens
            total_output = 0
            total_reasoning = 0
            total_cost = 0.0
            with jsonl.open() as f:
                for line in f:
                    d = json.loads(line)
                    if d.get("type") != "step_finish":
                        continue
                    part = d.get("part") or {}
                    tokens = part.get("tokens") or {}
                    if not tokens:
                        continue
                    total_input += tokens.get("input", 0)
                    cache = tokens.get("cache") or {}
                    total_cache += cache.get("read", 0)
                    total_output += tokens.get("output", 0)
                    total_reasoning += tokens.get("reasoning", 0)
                    total_cost += part.get("cost", 0) or 0
            prompt_tokens = total_input + total_cache  # full context, like Codex
            if prompt_tokens == 0 and total_output == 0:
                return None
            return {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": total_output,
                "total_tokens": prompt_tokens + total_output,
                "total_cost_usd": total_cost,
            }
        except Exception:
            pass  # fall through to DB

    # Fallback: query OpenCode SQLite DB (older runs without JSONL)
    import sqlite3
    db_path = Path.home() / ".local" / "share" / "opencode" / "opencode.db"
    if not db_path.exists():
        return None
    try:
        db = sqlite3.connect(str(db_path))
        cur = db.cursor()
        cur.execute(
            "SELECT id, time_created FROM session WHERE directory = ? ORDER BY time_created",
            (workspace_dir,),
        )
        sessions = cur.fetchall()
        if not sessions:
            db.close()
            return None
        setup_time_ms = 0
        for marker in ["opencode.json", "TASK.md", "AGENTS.md"]:
            marker_path = ws / marker
            if marker_path.exists():
                setup_time_ms = int(marker_path.stat().st_mtime * 1000)
                break
        cutoff = setup_time_ms - 5 * 60 * 1000 if setup_time_ms > 0 else 0
        main_sessions = [(sid, ts) for sid, ts in sessions if ts >= cutoff]
        main_sid = None
        if main_sessions:
            ph = ",".join(["?"] * len(main_sessions))
            sids_after = [s[0] for s in main_sessions]
            cur.execute(
                "SELECT id, time_created FROM session WHERE id IN (" + ph + ") "
                "AND parent_id IS NULL ORDER BY time_created DESC LIMIT 1",
                sids_after,
            )
            row = cur.fetchone()
            if row:
                main_sid = row[0]
        if main_sid:
            cur.execute(
                "SELECT id FROM session WHERE directory = ? AND "
                "(id = ? OR parent_id = ?)",
                (workspace_dir, main_sid, main_sid),
            )
            recent_sids = [r[0] for r in cur.fetchall()]
        else:
            recent_sids = [sid for sid, ts in sessions if ts >= cutoff]
        total_input = total_output = 0
        total_cost = 0.0
        for sid in recent_sids:
            cur.execute(
                "SELECT data FROM message WHERE session_id = ?",
                (sid,),
            )
            for (data_str,) in cur.fetchall():
                d_msg = json.loads(data_str)
                tokens = d_msg.get("tokens") or {}
                total_input += tokens.get("input", 0)
                cache = tokens.get("cache") or {}
                total_input += cache.get("read", 0)
                total_output += tokens.get("output", 0)
                total_cost += d_msg.get("cost", 0) or 0
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
    paper_slug = run_dir.parent.name
    for ms_path in [
        # run_dir / "explainer_workspace" / "methodology_summary.json",
        run_dir / "workspace" / "methodology_summary.json",
        # Extractor-variant workspaces contain a reduced Markdown-methods
        # summary without extracted_tables. Fall back to the shared full
        # summary so their judged cells receive the same row-type enrichment.
        run_dir.parent / "summaries" / f"{paper_slug}_summary.json",
    ]:
        if not ms_path.exists():
            continue
        ms = _load_json(ms_path)
        if not ms or not ms.get("extracted_tables"):
            continue
        for table in ms.get("extracted_tables", []):
            table_id = table.get("table_id", "")
            for cell in table.get("cells", []):
                key = (table_id, cell.get("row_label", ""), cell.get("column_label", ""))
                lookup[key] = cell.get("row_type", "")
        break
    return lookup


def _coefficient_metadata_by_position(
    table_id: str, cells: list,
) -> dict[tuple[str, str, str, int], dict[str, float | int | None]]:
    """Index coefficient SEs/stars without collapsing repeated row labels.

    Tables with panels often repeat the same coefficient label in the same
    column. A three-part ``(table, row_label, column_label)`` key therefore
    overwrites earlier panels. The fourth key component is the coefficient's
    zero-based occurrence among cells with the same labels, in table order.

    Prefer the extractor's explicit ``row_index``/``col_index`` and ``refers_to``
    links when finding a coefficient's SE. Older artifacts without those fields
    fall back to the immediately following cell in the same column.
    """
    if cells and isinstance(cells[0], list):
        cells = [c for row in cells for c in row if isinstance(c, dict)]
    else:
        cells = [c for c in cells if isinstance(c, dict)]

    se_by_parent_position = {}
    for cell in cells:
        if cell.get("row_type") != "se" and not cell.get("is_standard_error"):
            continue
        parent_row = cell.get("refers_to")
        col_index = cell.get("col_index")
        if parent_row is not None and col_index is not None:
            se_by_parent_position[(parent_row, col_index)] = cell.get("numeric_value")

    by_col: dict[str, list[dict]] = {}
    for cell in cells:
        by_col.setdefault(cell.get("column_label", ""), []).append(cell)

    occurrence_counts: Counter = Counter()
    lookup = {}
    for col, col_cells in by_col.items():
        for i, cell in enumerate(col_cells):
            if cell.get("row_type") != "coefficient":
                continue

            base_key = (table_id, cell.get("row_label", ""), col)
            occurrence = occurrence_counts[base_key]
            occurrence_counts[base_key] += 1

            se_val = None
            row_index = cell.get("row_index")
            col_index = cell.get("col_index")
            if row_index is not None and col_index is not None:
                se_val = se_by_parent_position.get((row_index, col_index))
            if se_val is None and i + 1 < len(col_cells):
                next_cell = col_cells[i + 1]
                if (next_cell.get("row_type") == "se"
                        or next_cell.get("is_standard_error")):
                    se_val = next_cell.get("numeric_value")

            stars = cell.get("significance_stars", 0)
            lookup[base_key + (occurrence,)] = {
                "se": se_val,
                "stars": int(stars) if stars is not None else 0,
            }
    return lookup


def _load_replicator_se_values(run_dir: Path) -> dict[tuple[str, str, str, int], float | None]:
    """Build occurrence-aware lookup for replicated coefficient SE values."""
    se_lookup = {}
    for base in [
        # run_dir / "explainer_workspace" / "replicator_outputs",
                 run_dir / "workspace"]:
        if not base.exists():
            continue
        for table_json in sorted(base.glob("table_*.json")):
            data = _load_json(table_json)
            if not data or "cells" not in data:
                continue
            table_id = data.get("table_id", "")
            metadata = _coefficient_metadata_by_position(table_id, data["cells"])
            se_lookup.update({key: value["se"] for key, value in metadata.items()})
        break
    return se_lookup


def _load_original_se_values(results_dir: Path, paper_slug: str) -> dict[tuple[str, str, str, int], float | None]:
    """Build occurrence-aware lookup for original coefficient SE values.

    Reads from the unblinded {paper}_results.json (extractor output with actual
    values), not the blinded workspace summary.
    """
    se_lookup: dict[tuple[str, str, str, int], float | None] = {}
    results_path = results_dir / paper_slug / "summaries" / f"{paper_slug}_results.json"
    if not results_path.exists():
        return se_lookup
    data = _load_json(results_path)
    if not data:
        return se_lookup
    for table in data.get("tables", []):
        table_id = table.get("table_id", "")
        metadata = _coefficient_metadata_by_position(table_id, table.get("cells", []))
        se_lookup.update({key: value["se"] for key, value in metadata.items()})
    return se_lookup


def _load_original_significance(results_dir: Path, paper_slug: str) -> dict[tuple[str, str, str, int], int]:
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
        metadata = _coefficient_metadata_by_position(table_id, table.get("cells", []))
        lookup.update({key: value["stars"] for key, value in metadata.items()})
    return lookup


def _load_replicator_significance(run_dir: Path) -> dict[tuple[str, str, str, int], int]:
    lookup = {}
    for base in [
        # run_dir / "explainer_workspace" / "replicator_outputs",
                 run_dir / "workspace"]:
        if not base.exists():
            continue
        for table_json in sorted(base.glob("table_*.json")):
            data = _load_json(table_json)
            if not data or "cells" not in data:
                continue
            table_id = data.get("table_id", "")
            metadata = _coefficient_metadata_by_position(table_id, data["cells"])
            lookup.update({key: value["stars"] for key, value in metadata.items()})
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
            original_se_lookup = _load_original_se_values(results_dir, paper_slug)
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
                    coefficient_occurrences: Counter = Counter()
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
                            se_key_base = (item_id, cell_row_label, cell_col_label)
                            occurrence = coefficient_occurrences[se_key_base]
                            coefficient_occurrences[se_key_base] += 1
                            se_key = se_key_base + (occurrence,)
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
                    # grade_num: NA maps to None→NaN, F maps to 0 (included in averages)
                    "grade_num": GRADE_TO_NUM.get(grade, np.nan) if GRADE_TO_NUM.get(grade) is not None else np.nan,
                    "f_reason": (
                        "not_produced" if "not produced" in comparison_notes.lower()
                        else "no_pre_aligned" if "no pre-aligned" in comparison_notes
                        else "pre_aligned_all_f" if grade == "F" and "pre-aligned" in comparison_notes.lower()
                        else "non_numerical" if "non-numerical" in comparison_notes.lower()
                        else "other_f" if grade == "F"
                        else ""
                    ),
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
                            #  run_dir / "explainer_workspace" / "run_log.txt"
                             ]:
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

            # Count templates from table_templates/ dir (ground truth for replicator)
            tmpl_dir = workspace / "table_templates"
            if tmpl_dir.is_dir():
                n_table_templates = sum(1 for f in tmpl_dir.iterdir() if f.suffix == ".json")
            # Fallback to extracted_tables count in methodology summary
            if n_table_templates == 0 and meth_summary:
                n_table_templates = len(meth_summary.get("extracted_tables", []))

            # Count table_*.json files produced by replicator (workspace first)
            n_table_jsons = 0
            n_extra_table_jsons = 0
            for base in [run_dir / "workspace",
                        #  run_dir / "explainer_workspace" / "replicator_outputs"
                         ]:
                if base.exists():
                    produced = {f.stem for f in base.glob("table_*.json")}
                    n_table_jsons = len(produced)
                    # Count produced JSONs whose stem is NOT in table_templates/
                    tmpl_stems = set()
                    tt = base / "table_templates"
                    if tt.is_dir():
                        tmpl_stems = {f.stem for f in tt.iterdir() if f.suffix == ".json"}
                    n_extra_table_jsons = len(produced - tmpl_stems)
                    break

            # Count .py files produced
            n_py_files = 0
            for base in [run_dir / "workspace",
                        #  run_dir / "explainer_workspace"
                         ]:
                if base.exists():
                    n_py_files = len(list(base.rglob("*.py")))
                    break

            # Agent-produced LOC + top-level Python imports (naive line count;
            # regex-based import extraction — robust to partially invalid .py).
            agent_loc, agent_libs = _analyze_agent_code(workspace)

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
            if approach == "swe-agent":
                traj_path = workspace / "trajectory.json"
                if traj_path.exists():
                    traj = _load_json(traj_path)
                    if traj:
                        # Sum per-message usage from extra.response.usage
                        traj_input = traj_output = 0
                        traj_cost = 0.0
                        for msg in traj.get("messages", []):
                            if not isinstance(msg, dict):
                                continue
                            extra = msg.get("extra", {})
                            resp = extra.get("response", {})
                            if isinstance(resp, dict):
                                usage_obj = resp.get("usage", {})
                                if isinstance(usage_obj, dict):
                                    traj_input += usage_obj.get("prompt_tokens", 0)
                                    traj_output += usage_obj.get("completion_tokens", 0)
                            traj_cost += extra.get("cost", 0) or 0
                        if traj_input + traj_output > total_tokens:
                            prompt_tokens = traj_input
                            completion_tokens = traj_output
                            total_tokens = traj_input + traj_output
                        if traj_cost > total_cost_usd:
                            total_cost_usd = traj_cost
                        # Fallback to model_stats if per-message parsing got nothing
                        if total_cost_usd == 0:
                            ms = (traj.get("info") or {}).get("model_stats", {})
                            total_cost_usd = ms.get("instance_cost", 0)

            # Always use OpenCode DB for token/cost (usage.json only captures one session)
            if approach == "opencode":
                oc_usage = _query_opencode_db_for_workspace(str(workspace))
                if oc_usage and oc_usage["total_tokens"] > total_tokens:
                    prompt_tokens = oc_usage["prompt_tokens"]
                    completion_tokens = oc_usage["completion_tokens"]
                    total_tokens = oc_usage["total_tokens"]
                    total_cost_usd = oc_usage.get("total_cost_usd", 0)

            # A release bundle cannot ship the user's OpenCode database.  The
            # release builder freezes the already-audited per-run telemetry in
            # a small JSON sidecar so analysis remains byte-for-byte portable.
            release_usage = _load_json(run_dir / "release_usage.json")
            if release_usage:
                prompt_tokens = release_usage.get("prompt_tokens", prompt_tokens)
                completion_tokens = release_usage.get("completion_tokens", completion_tokens)
                total_tokens = release_usage.get("total_tokens", total_tokens)
                total_cost_usd = release_usage.get("total_cost_usd", total_cost_usd)
                duration = release_usage.get("duration_seconds", duration)

            # Paper-level data stats
            n_datasets = 0
            total_data_bytes = 0
            paper_title = FALLBACK_TITLES.get(paper_slug, paper_slug)
            journal = _infer_journal(paper_slug)
            original_language = "Unknown"
            original_languages_all = []
            if papers_dir:
                release_meta = _load_json(papers_dir / paper_slug / "release_metadata.json")
                if release_meta:
                    n_datasets = int(release_meta.get("n_datasets", 0))
                    total_data_bytes = int(release_meta.get("total_data_size_bytes", 0))
                    paper_title = release_meta.get("paper_title", paper_title)
                    original_language = release_meta.get("original_language", "Unknown")
                    original_languages_all = release_meta.get("original_languages_all", [])
                else:
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
                # overall_grade_num: NA maps to None→NaN, F maps to 0
                "overall_grade_num": GRADE_TO_NUM.get(overall_grade, np.nan) if GRADE_TO_NUM.get(overall_grade) is not None else np.nan,
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
                "n_extra_table_jsons": n_extra_table_jsons,
                "n_py_files": n_py_files,
                "agent_loc": agent_loc,
                "agent_libraries": agent_libs,
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
    df_cells = apply_adaptive_rounding(df_cells)

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


def validate_sample_manifest(df_runs: pd.DataFrame, manifest_path: Path) -> None:
    """Fail if the loaded production run set differs from a pinned manifest."""
    manifest = _load_json(manifest_path)
    if not manifest:
        raise ValueError(f"Could not read sample manifest: {manifest_path}")

    papers = set(manifest.get("paper_ids", []))
    combinations = set(manifest.get("approach_model_combinations", []))
    if not papers or not combinations:
        raise ValueError("Sample manifest must list paper_ids and approach_model_combinations")

    per_paper = manifest.get("paper_specific_included_approach_model_combinations", {})
    unknown_papers = set(per_paper) - papers
    invalid_combinations = {
        (paper, combo)
        for paper, paper_combinations in per_paper.items()
        for combo in paper_combinations
        if combo not in combinations
    }
    if unknown_papers or invalid_combinations:
        raise ValueError(
            "Manifest has invalid paper-specific included runs: "
            f"unknown_papers={sorted(unknown_papers)}, "
            f"invalid_combinations={sorted(invalid_combinations)}"
        )
    expected = {
        (paper, combo)
        for paper in papers
        for combo in per_paper.get(paper, combinations)
    }
    expected_count = manifest.get("included_run_count")
    if expected_count is not None and int(expected_count) != len(expected):
        raise ValueError(
            f"Manifest included_run_count={expected_count} but included run set "
            f"contains {len(expected)}"
        )

    duplicates = df_runs.duplicated(["paper_slug", "approach"], keep=False)
    if duplicates.any():
        duplicate_rows = sorted(set(
            zip(df_runs.loc[duplicates, "paper_slug"].astype(str),
                df_runs.loc[duplicates, "approach"].astype(str))
        ))
        raise ValueError(f"Duplicate production runs found: {duplicate_rows}")

    actual = set(zip(
        df_runs["paper_slug"].astype(str),
        df_runs["approach"].astype(str),
    ))
    unexpected = sorted(actual - expected)
    absent = sorted(expected - actual)
    if unexpected or absent:
        details = []
        if unexpected:
            details.append(f"unexpected={unexpected}")
        if absent:
            details.append(f"missing={absent}")
        raise ValueError("Loaded run set does not match sample manifest: " + "; ".join(details))

    print(
        f"  Sample manifest validated: {len(papers)} papers, "
        f"{len(combinations)} combinations, {len(actual)} runs"
    )


# ============================================================================
# Plot functions (helper to get approach list from data)
# ============================================================================

def _approaches_in(df, col="approach"):
    """Return approaches present in data, ordered by APPROACH_ORDER."""
    present = df[col].unique() if not df.empty else []
    return ([a for a in APPROACH_ORDER if a in present]
            + [a for a in present if a not in APPROACH_ORDER])


# F-grade handling modes used by grade-distribution plots.
F_MODES = ["all_f", "no_f", "at_least_one_non_f"]
F_MODE_SUFFIX = {
    "all_f": "_all_f",
    "no_f": "_no_f",
    "at_least_one_non_f": "_at_least_one_non_f",
}


def _filter_f_mode(df: pd.DataFrame, f_mode: str, level: str,
                   grade_col: str | None = None) -> pd.DataFrame:
    """Filter rows by F-grade handling mode.

    Modes:
      - "all_f": no filter (include F rows).
      - "no_f": drop rows with grade F.
      - "at_least_one_non_f": for each item (at the specified level), keep all rows
        iff at least one row for that item has a grade other than F or NA.
        Drop every row belonging to an item where every approach produced F/NA.

    Args:
        df: DataFrame with approach/grade columns.
        f_mode: one of F_MODES.
        level: "paper" | "item" | "cell" — groups rows into "items" accordingly.
        grade_col: grade column name; defaults by level.
    """
    if df.empty or f_mode == "all_f":
        return df
    if grade_col is None:
        grade_col = {
            "paper": "overall_grade",
            "item": "grade",
            "cell": "cell_grade",
        }.get(level, "grade")
    if grade_col not in df.columns:
        return df
    grade_str = df[grade_col].astype("object")
    if f_mode == "no_f":
        return df[grade_str != "F"].copy()
    if f_mode == "at_least_one_non_f":
        if level == "paper":
            keys = ["paper_slug"]
        elif level == "item":
            keys = ["paper_slug", "item_id"]
        elif level == "cell":
            keys = ["paper_slug", "item_id", "row_label", "column_label", "row_type"]
        else:
            raise ValueError(f"Unknown level: {level}")
        keys = [k for k in keys if k in df.columns]
        if not keys:
            return df
        non_f = (~grade_str.isin(["F", "NA"])).astype(int)
        any_non_f = non_f.groupby([df[k] for k in keys], dropna=False).transform("max")
        return df[any_non_f.astype(bool)].copy()
    raise ValueError(f"Unknown f_mode: {f_mode}")


# ============================================================================
# Section: Setup & Descriptives
# ============================================================================

def plot_extractor_row_type_distribution(df_cells: pd.DataFrame, output_dir: Path,
                                         results_dir: Path | None = None, subdir: str = ""):
    """Distribution of cell row_type values from original paper extractions.

    Uses _results.json (original extractor output) for accurate counts rather
    than df_cells which only contains cells that were compared by the grader.
    Falls back to df_cells deduplication if results_dir is not available.
    """
    from collections import Counter

    type_counts_dict: Counter = Counter()

    # Try to count from _results.json for accurate original cell counts
    # Only count cells with non-null numeric_value (actual extracted values)
    if results_dir and results_dir.is_dir():
        for pid in sorted(os.listdir(results_dir)):
            if pid.startswith("_") or ".bak" in pid:
                continue
            rp = results_dir / pid / "summaries" / f"{pid}_results.json"
            if not rp.is_file():
                continue
            data = _load_json(rp)
            if not data:
                continue
            for t in data.get("tables", []):
                for c in t.get("cells", []):
                    rt = c.get("row_type", "")
                    if rt and c.get("numeric_value") is not None:
                        type_counts_dict[rt] += 1
    else:
        # Fallback: deduplicate df_cells by paper+item+labels
        df = df_cells[df_cells["row_type"].notna() & (df_cells["row_type"] != "")].copy()
        if df.empty:
            print("  Skipping extractor_row_type_distribution: no row_type data")
            return
        df_dedup = df.drop_duplicates(subset=["paper_slug", "item_id", "row_label", "column_label"])
        type_counts_dict = Counter(df_dedup["row_type"].value_counts().to_dict())

    if not type_counts_dict:
        print("  Skipping extractor_row_type_distribution: no data")
        return

    type_order = [k for k, _ in type_counts_dict.most_common()]
    type_values = [type_counts_dict[k] for k in type_order]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(type_order))
    ax.bar(x, type_values, color="#3498db", edgecolor="white", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(type_order, rotation=45, ha="right", fontsize=11)
    ax.set_ylabel("Number of cells (original paper)", fontsize=16, fontweight="bold")
    for i, v in enumerate(type_values):
        ax.text(i, v + max(type_values) * 0.01, str(v), ha="center", fontsize=11, fontweight="bold")
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

    fig, ax = plt.subplots(figsize=(7, 5))
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
    place_legend(fig, ax, fontsize=12)
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig, output_dir, "first_fail_distribution", subdir)


def plot_agent_loc_distribution(df_runs: pd.DataFrame, output_dir: Path, subdir: str = ""):
    """Boxplot of agent-produced LOC per approach (Python files in workspace).

    Drops runs with agent_loc == 0 (workspace empty / not produced). Median
    annotated next to each box.
    """
    if df_runs.empty or "agent_loc" not in df_runs.columns:
        return
    df = df_runs[df_runs["agent_loc"] > 0].copy()
    if df.empty:
        print("  Skipping agent_loc_distribution: no runs with agent_loc > 0")
        return

    approaches = _approaches_in(df)
    data, labels, colors = [], [], []
    for a in approaches:
        vals = df.loc[df["approach"] == a, "agent_loc"].values
        if len(vals) >= 3:
            data.append(vals)
            labels.append(APPROACH_LABELS.get(a, a).replace("\n", " "))
            colors.append(APPROACH_COLORS.get(a, "#95a5a6"))
    if not data:
        return

    # Widen the figure so each approach has enough horizontal room for its
    # median label on the right side of its box.
    fig, ax = plt.subplots(figsize=(max(11, len(labels) * 1.8), 5.4))
    # Narrow boxes open up horizontal breathing room for the labels.
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                    widths=0.45, showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.65)
    for m in bp["medians"]:
        m.set_color("black"); m.set_linewidth(2)
    # Median annotation: place label clearly to the right of the box at the
    # median line's y-coordinate, with a pixel-based offset so spacing is
    # invariant to figure size / axes limits.
    for i, vs in enumerate(data):
        med = np.median(vs)
        ax.annotate(
            f"{int(med):,}",
            xy=(i + 1, med), xytext=(32, 0),
            textcoords="offset points",
            ha="left", va="center",
            fontsize=10, fontweight="bold",
        )
    plt.setp(ax.get_xticklabels(), fontsize=10, rotation=25, ha="right")
    ax.set_ylabel("Lines of code produced by agent", fontsize=12, fontweight="bold")
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig, output_dir, "agent_loc_distribution", subdir)


def plot_agent_libraries(df_runs: pd.DataFrame, output_dir: Path, subdir: str = "",
                          top_n: int = 15):
    """Heatmap of top-N library usage across approaches.

    Cell value: share of that approach's runs that imported the library (%).
    Libraries selected are the ones with the highest *average* share across
    approaches. Also writes ``agent_libraries.csv`` with the full counts.
    """
    if df_runs.empty or "agent_libraries" not in df_runs.columns:
        return
    import ast as _ast

    def _parse_libs(v):
        if isinstance(v, list):
            return v
        if isinstance(v, str) and v.strip().startswith("["):
            try:
                return _ast.literal_eval(v)
            except (ValueError, SyntaxError):
                return []
        return []

    df = df_runs.copy()
    df["_libs"] = df["agent_libraries"].map(_parse_libs)
    df = df[df["_libs"].map(len) > 0]
    if df.empty:
        print("  Skipping agent_libraries: no libraries detected")
        return

    approaches = _approaches_in(df)
    per_approach: dict[str, dict[str, float]] = {}
    per_approach_counts: dict[str, dict[str, int]] = {}
    for a in approaches:
        sub = df[df["approach"] == a]
        n_runs = len(sub)
        if n_runs == 0:
            continue
        lib_counts: dict[str, int] = {}
        for libs in sub["_libs"]:
            for lib in libs:
                lib_counts[lib] = lib_counts.get(lib, 0) + 1
        per_approach_counts[a] = lib_counts
        per_approach[a] = {lib: n / n_runs * 100 for lib, n in lib_counts.items()}

    # Rank by mean share across approaches
    avg_share: dict[str, list[float]] = {}
    for a, libs in per_approach.items():
        for lib, share in libs.items():
            avg_share.setdefault(lib, []).append(share)
    avg_mean = {lib: sum(s) / len(per_approach) for lib, s in avg_share.items()}
    top_libs = sorted(avg_mean, key=avg_mean.get, reverse=True)[:top_n]

    approaches_ordered = [a for a in approaches if a in per_approach]
    matrix = np.zeros((len(approaches_ordered), len(top_libs)))
    for i, a in enumerate(approaches_ordered):
        for j, lib in enumerate(top_libs):
            matrix[i, j] = per_approach[a].get(lib, 0)

    fig, ax = plt.subplots(
        figsize=(max(9, len(top_libs) * 0.75), max(4, len(approaches_ordered) * 0.5))
    )
    sns.heatmap(
        matrix, annot=True, fmt=".0f", cmap="Blues", vmin=0, vmax=100,
        xticklabels=top_libs,
        yticklabels=[APPROACH_LABELS.get(a, a).replace("\n", " ") for a in approaches_ordered],
        cbar_kws={"label": "% of runs using library"},
        ax=ax, square=False, linewidths=0.3, linecolor="white",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=10)
    ax.set_xlabel(f"Top {top_n} libraries (by mean share across approaches)",
                  fontsize=11, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, output_dir, "agent_libraries_heatmap", subdir)

    # Also write a flat CSV with per-(approach, lib) counts and shares, for
    # downstream inspection beyond the top-N heatmap.
    target = output_dir / subdir if subdir else output_dir
    target.mkdir(parents=True, exist_ok=True)
    csv_rows = []
    for a in approaches_ordered:
        n_runs = int((df["approach"] == a).sum())
        for lib, n in per_approach_counts[a].items():
            csv_rows.append({
                "approach": a, "library": lib,
                "n_runs_using": n, "n_runs_total": n_runs,
                "share_pct": round(100 * n / n_runs, 1) if n_runs else 0.0,
            })
    if csv_rows:
        pd.DataFrame(csv_rows).sort_values(
            ["approach", "n_runs_using"], ascending=[True, False]
        ).to_csv(target / "agent_libraries.csv", index=False)
        print(f"  Saved {subdir + '/' if subdir else ''}agent_libraries.csv "
              f"({len(csv_rows)} approach × library rows)")


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
    place_legend(fig, ax, fontsize=14)
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
    place_legend(fig, ax2, fontsize=12)
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
        pct_na = ((sub["overall_grade"] == "NA").sum() / n * 100) if n else 0
        mean_dur = sub["duration_seconds"].mean()
        n_items_total = len(items)
        item_pct_ab = ((items["grade"].isin(["A", "B"])).sum() / n_items_total * 100) if n_items_total else 0
        n_items_na = (items["grade"] == "NA").sum() if n_items_total else 0

        rows.append({
            "Approach": APPROACH_LABELS.get(approach, approach).replace("\n", " "),
            "Runs": n,
            "Mean Grade": f"{mean_grade:.2f}",
            "% A-B (runs)": f"{pct_ab:.1f}",
            "% F (runs)": f"{pct_f:.1f}",
            "% NA (runs)": f"{pct_na:.1f}",
            "Items": n_items_total,
            "% A-B (items)": f"{item_pct_ab:.1f}",
            "N NA (items)": int(n_items_na),
            "Mean Duration (min)": f"{mean_dur / 60:.1f}" if pd.notna(mean_dur) else "—",
        })

    summary = pd.DataFrame(rows)
    target = output_dir / subdir if subdir else output_dir
    target.mkdir(parents=True, exist_ok=True)
    summary.to_csv(target / "summary_table.csv", index=False)
    latex = summary.to_latex(index=False, escape=True, column_format="l" + "r" * (len(summary.columns) - 1))
    (target / "summary_table.tex").write_text(latex)
    print("  Saved summary_table")


def generate_summary_stats(df_cells: pd.DataFrame, df_items: pd.DataFrame,
                            output_dir: Path, subdir: str = "",
                            df_runs: pd.DataFrame | None = None):
    """Summary statistics on originals vs reproductions, computed from df_cells.

    Produces:
      - summary_stats.csv            — one row per source (Originals + approaches)
      - summary_stats_transposed.csv — pivot: stats as rows, sources as columns
      - summary_stats.tex            — LaTeX version of the transposed table

    Originals and reproductions both come from df_cells. For originals, cells are
    deduplicated across approaches using a positional rank within each
    (approach, paper, item, row_label, column_label, row_type) group — needed
    because the extractor marks SE rows with row_label=NaN, so multiple distinct
    SE cells in the same column collapse under naive label-based dedup.

    Note: this approach only covers tables where at least one approach produced
    output. Tables where all approaches failed are invisible. For a ground-truth
    count, load directly from {paper}_results.json files instead.
    """
    target = output_dir / subdir if subdir else output_dir
    target.mkdir(parents=True, exist_ok=True)

    if df_cells.empty:
        print("  Skipping summary_stats: empty df_cells")
        return

    # Row-type constants — match extractor schema
    COEF, SE, PV, CI, TS = "coefficient", "se", "p_value", "ci", "t_stat"
    R2, NOBS, FSTAT = "statistic_r2", "statistic_n_obs", "statistic_f"
    OTHER = "statistic_other"

    def _count_row(cells: pd.DataFrame, value_col: str, source: str,
                    n_papers: int, n_tables: int,
                    n_tables_non_f: int | None = None,
                    n_papers_non_f: int | None = None) -> dict:
        val_num = pd.to_numeric(cells[value_col], errors="coerce")
        rt = cells["row_type"].fillna("")
        present = val_num.notna()
        is_coef = (rt == COEF)
        coef_vals = val_num.where(is_coef)
        return {
            "Source": source,
            "N papers": int(n_papers),
            "N papers non-F": int(n_papers_non_f) if n_papers_non_f is not None else int(n_papers),
            "N tables": int(n_tables),
            "N tables non-F": int(n_tables_non_f) if n_tables_non_f is not None else int(n_tables),
            "N cells (total)": int(len(cells)),
            "N cells (present)": int(present.sum()),
            "N cells (missing)": int((~present).sum()),
            "N coefficient": int(is_coef.sum()),
            "N SE": int((rt == SE).sum()),
            "N p-value": int((rt == PV).sum()),
            "N t-stat": int((rt == TS).sum()),
            "N CI": int((rt == CI).sum()),
            "N R-squared": int((rt == R2).sum()),
            "N N_obs": int((rt == NOBS).sum()),
            "N F-stat": int((rt == FSTAT).sum()),
            "N other numeric": int((rt == OTHER).sum()),
            "N pos. coef.": int((coef_vals > 0).sum()),
            "N neg. coef.": int((coef_vals < 0).sum()),
            "N zero coef.": int((coef_vals == 0).sum()),
            "N NA cells": int((cells["cell_grade"] == "NA").sum()) if "cell_grade" in cells.columns else 0,
        }

    rows = []

    # Originals: positional-rank dedup across approaches. Cells appear in table
    # order within cell_comparisons, so cumcount within each group gives a stable
    # rank that disambiguates cells with identical labels (e.g. SE rows with
    # row_label=NaN in the same column).
    orig_src = df_cells.copy()
    key_cols_base = ["paper_slug", "item_id", "row_label", "column_label", "row_type"]
    for c in key_cols_base:
        orig_src[c] = orig_src[c].fillna("__NA__")
    orig_src["cell_rank"] = orig_src.groupby(
        ["approach", "model"] + key_cols_base, observed=True
    ).cumcount()
    orig = orig_src.drop_duplicates(subset=key_cols_base + ["cell_rank"], keep="first")
    n_papers_orig = orig["paper_slug"].nunique()
    n_tables_orig = orig.groupby(["paper_slug", "item_id"]).ngroups
    rows.append(_count_row(orig, "original_value", "Originals (paper)",
                            n_papers_orig, n_tables_orig))

    # Reproductions: per approach × model
    for (approach, model), grp in df_cells.groupby(["approach", "model"], observed=True):
        label = APPROACH_LABELS.get(approach, approach).replace("\n", " ")
        source = f"{label} ({model})"
        n_papers_r = grp["paper_slug"].nunique()
        n_tables_r = grp.groupby(["paper_slug", "item_id"]).ngroups
        # Count non-F tables from df_items (excluding NA too, since NA = unassessable)
        n_non_f = 0
        if not df_items.empty:
            item_sub = df_items[
                (df_items["approach"] == approach) &
                (df_items["item_type"] == "table") &
                (~df_items["grade"].isin(["F", "NA"]))
            ]
            n_non_f = len(item_sub)
        # Count non-F papers: paper's overall_grade is not F (and not NA)
        n_papers_non_f = 0
        if df_runs is not None and not df_runs.empty:
            run_sub = df_runs[
                (df_runs["approach"] == approach) &
                (~df_runs["overall_grade"].isin(["F", "NA"]))
            ]
            n_papers_non_f = run_sub["paper_slug"].nunique()
        elif not df_items.empty:
            # Fallback: compute overall grade from table grades
            GRADE_VAL = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1, "F": 0}
            item_sub = df_items[
                (df_items["approach"] == approach) &
                (df_items["item_type"] == "table") &
                (df_items["grade"] != "NA")
            ].copy()
            if not item_sub.empty:
                item_sub["gn"] = item_sub["grade"].map(GRADE_VAL)
                paper_avg = item_sub.groupby("paper_slug")["gn"].mean()
                n_papers_non_f = (paper_avg >= 0.5).sum()
        rows.append(_count_row(grp, "replicated_value", source, n_papers_r, n_tables_r, n_non_f, n_papers_non_f))

    summary = pd.DataFrame(rows)
    summary.to_csv(target / "summary_stats.csv", index=False)

    summary_t = summary.set_index("Source").T
    summary_t.index.name = "Statistic"
    summary_t.to_csv(target / "summary_stats_transposed.csv")

    # ── LaTeX table: selected rows with relative availability (%) ─────────
    # For reproduction approaches, show "count (pct%)" where pct = count / originals × 100.
    latex_rows_spec = [
        ("N papers", "Papers"),
        ("N papers non-F", "Papers (non-F)"),
        ("N tables", "Tables"),
        ("N tables non-F", "Tables (non-F)"),
        ("N cells (present)", "Cells (present)"),
        ("N coefficient", "Coefficients"),
        ("N pos. coef.", "Positive coefficients"),
        ("N SE", "Standard errors"),
        ("N p-value", "p-values"),
        ("N t-stat", "t-statistics"),
        ("N CI", "Confidence intervals"),
        ("N R-squared", "R-squared"),
        ("N N_obs", "N observations"),
        ("N F-stat", "F-statistics"),
        ("N other numeric", "Other numeric"),
        ("N NA cells", "NA cells (not assessable)"),
    ]
    orig_row = summary.iloc[0]  # Originals row

    # Build column headers: short approach labels
    sources = list(summary["Source"])
    short_labels = []
    for s in sources:
        if "Originals" in s:
            short_labels.append("Originals")
        else:
            # Extract approach name before the parenthesized model
            parts = s.split(" (")
            short_labels.append(parts[0] if parts else s)

    # Build LaTeX manually for precise control
    n_cols = len(sources)
    col_fmt = "l" + "r" * n_cols
    lines = []
    lines.append(r"\begin{tabular}{" + col_fmt + "}")
    lines.append(r"\toprule")
    header = " & ".join([""] + [f"\\textbf{{{l}}}" for l in short_labels]) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for stat_key, stat_label in latex_rows_spec:
        cells = []
        orig_val = int(orig_row.get(stat_key, 0))
        for i, row_data in summary.iterrows():
            val = int(row_data.get(stat_key, 0))
            if stat_key == "N pos. coef.":
                # Show share of own coefficients that are positive
                own_coef = int(row_data.get("N coefficient", 0))
                pct = (val / own_coef * 100) if own_coef > 0 else 0
                cells.append(f"{pct:.0f}\\%")
            elif i == 0:
                cells.append(f"{val:,}")
            else:
                pct = (val / orig_val * 100) if orig_val > 0 else 0
                cells.append(f"{val:,} ({pct:.0f}\\%)")
        line = f"{stat_label} & " + " & ".join(cells) + r" \\"
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    latex = "\n".join(lines)
    (target / "summary_stats.tex").write_text(latex)
    print(f"  Saved {subdir}/summary_stats (CSV + transposed CSV + LaTeX)")


def generate_summary_stats_panels(df_cells: pd.DataFrame, df_items: pd.DataFrame,
                                    output_dir: Path, subdir: str = "",
                                    df_runs: pd.DataFrame | None = None):
    """Two-panel variant of ``summary_stats``.

    Panel A — per-paper summary (Originals only): for every stat type, report
    the mean / SD / min / max across the 48 papers. E.g. "Coefficients per
    paper: mean=108, SD=83, min=8, max=332". No per-approach columns.

    Panel B — agent systems: per-approach counts as % of the originals, plus an
    "Avg" column (mean across approaches). Adds a ``Completion rate`` row at
    the bottom that shows, per approach, the mean across papers of
    ``(cells produced / cells expected)``. Positive-coefficient row is
    dropped from this variant.
    """
    target = output_dir / subdir if subdir else output_dir
    target.mkdir(parents=True, exist_ok=True)
    if df_cells.empty:
        print("  Skipping summary_stats_panels: empty df_cells")
        return

    COEF, SE, PV, CI, TS = "coefficient", "se", "p_value", "ci", "t_stat"
    R2, NOBS, FSTAT, OTHER = "statistic_r2", "statistic_n_obs", "statistic_f", "statistic_other"

    # Row-type categories to report (label, matching row_type or special key).
    ROW_SPEC = [
        ("Tables", "__tables__"),
        ("Cells (present)", "__cells_present__"),
        ("Coefficients", COEF),
        ("Standard errors", SE),
        ("p-values", PV),
        ("t-statistics", TS),
        ("Confidence intervals", CI),
        ("R-squared", R2),
        ("N observations", NOBS),
        ("F-statistics", FSTAT),
        ("Other numeric", OTHER),
    ]

    # Originals: same positional-rank dedup as the existing summary_stats.
    key_cols_base = ["paper_slug", "item_id", "row_label", "column_label", "row_type"]
    orig_src = df_cells.copy()
    for c in key_cols_base:
        orig_src[c] = orig_src[c].fillna("__NA__")
    orig_src["_rank"] = orig_src.groupby(
        ["approach", "model"] + key_cols_base, observed=True
    ).cumcount()
    orig = orig_src.drop_duplicates(subset=key_cols_base + ["_rank"], keep="first")

    # ── Panel A: per-paper mean / SD / min / max (Originals only) ─────────
    def _per_paper_count(df: pd.DataFrame, row_type_key: str) -> pd.Series:
        """Return a Series indexed by paper_slug giving the count for this key."""
        if row_type_key == "__tables__":
            return df.groupby("paper_slug", observed=True)["item_id"].nunique()
        if row_type_key == "__cells_present__":
            present = df[pd.to_numeric(df["original_value"], errors="coerce").notna()]
            return present.groupby("paper_slug", observed=True).size()
        sub = df[df["row_type"].fillna("") == row_type_key]
        return sub.groupby("paper_slug", observed=True).size()

    all_papers = orig["paper_slug"].unique()
    panel_a_rows = []
    for label, key in ROW_SPEC:
        s = _per_paper_count(orig, key).reindex(all_papers, fill_value=0)
        panel_a_rows.append({
            "Statistic": label,
            "Total": int(s.sum()),
            "Mean": round(float(s.mean()), 1),
            "SD":   round(float(s.std()), 1),
            "Min":  int(s.min()),
            "Max":  int(s.max()),
        })
    panel_a = pd.DataFrame(panel_a_rows)

    # ── Panel B: per-approach counts (share-of-originals %) + Avg col ─────
    # Column per approach × model combination.
    present_approaches = set(df_cells["approach"].astype(str).unique())
    approaches = [a for a in APPROACH_MODEL_ORDER if a in present_approaches]
    # Fall back to whatever is present if the canonical order doesn't match.
    if not approaches:
        approaches = sorted(present_approaches)

    # Short header label per approach (strip model parenthetical).
    def _short(a: str) -> str:
        return APPROACH_LABELS.get(a, a).replace("\n", " ")

    # Global totals per row (Originals)
    orig_totals = {}
    for label, key in ROW_SPEC:
        orig_totals[label] = int(_per_paper_count(orig, key).sum())

    # Agent totals per (approach, row)
    agent_totals: dict[str, dict[str, int]] = {}
    for a in approaches:
        sub = df_cells[df_cells["approach"] == a]
        vals = {}
        for label, key in ROW_SPEC:
            if key == "__tables__":
                vals[label] = sub.groupby("paper_slug", observed=True)["item_id"].nunique().sum()
            elif key == "__cells_present__":
                present = sub[pd.to_numeric(sub["replicated_value"], errors="coerce").notna()]
                vals[label] = int(len(present))
            else:
                vals[label] = int((sub["row_type"].fillna("") == key).sum())
        agent_totals[a] = vals

    # Papers attempted per approach (from df_runs if present, else cells)
    if df_runs is not None and not df_runs.empty:
        n_papers_by_approach = df_runs.groupby("approach", observed=True)["paper_slug"].nunique().to_dict()
    else:
        n_papers_by_approach = {
            a: int(df_cells.loc[df_cells["approach"] == a, "paper_slug"].nunique())
            for a in approaches
        }
    n_papers_orig = int(orig["paper_slug"].nunique())

    # Completion rate per approach: mean across papers of
    # (n_present_cells_agent / n_present_cells_original)  (equal per-paper weight).
    orig_per_paper = _per_paper_count(orig, "__cells_present__")
    completion_pct: dict[str, float] = {}
    for a in approaches:
        sub = df_cells[df_cells["approach"] == a]
        present = sub[pd.to_numeric(sub["replicated_value"], errors="coerce").notna()]
        agent_per_paper = present.groupby("paper_slug", observed=True).size()
        ratios = []
        for p in all_papers:
            o = int(orig_per_paper.get(p, 0) or 0)
            if o == 0:
                continue
            g = int(agent_per_paper.get(p, 0) or 0)
            ratios.append(min(100.0, g / o * 100.0))
        completion_pct[a] = round(sum(ratios) / len(ratios), 1) if ratios else 0.0

    # "Non-F papers" here = papers where *at least one* table has a non-F,
    # non-NA grade under the `all_f` mode. Complement: papers where every
    # table was F ("complete failures"). Uses df_items rather than the
    # aggregated paper grade so we measure at-the-table-level coverage.
    n_papers_non_f: dict[str, int] = {}
    if "grade_all_f" in df_items.columns:
        tbl = df_items[df_items["item_type"] == "table"]
        for a in approaches:
            sub = tbl[tbl["approach"] == a]
            # A paper is "non-F" if at least one of its tables has grade not in {F, NA}.
            g = sub.groupby("paper_slug", observed=True)["grade_all_f"].apply(
                lambda s: any(x not in ("F", "NA") for x in s)
            )
            n_papers_non_f[a] = int(g.sum())
    else:
        for a in approaches:
            n_papers_non_f[a] = int(n_papers_by_approach.get(a, 0))

    # Build Panel B
    panel_b_rows = []

    def _paper_count_row(label: str, counts_by_approach: dict[str, int]) -> dict:
        row = {"Statistic": label}
        pcts = []
        for a in approaches:
            n = int(counts_by_approach.get(a, 0))
            pct = (n / n_papers_orig * 100.0) if n_papers_orig else 0.0
            pcts.append(pct)
            row[_short(a)] = f"{n} ({pct:.0f}\\%)" if n_papers_orig else f"{n}"
        mean_n = sum(counts_by_approach.get(a, 0) for a in approaches) / len(approaches)
        mean_pct = (mean_n / n_papers_orig * 100.0) if n_papers_orig else 0.0
        row["Avg"] = f"{mean_n:.1f} ({mean_pct:.0f}\\%)"
        return row

    panel_b_rows.append(_paper_count_row("Papers", n_papers_by_approach))
    panel_b_rows.append(_paper_count_row("Papers (non-F)", n_papers_non_f))
    # Per-stat rows
    for label, _ in ROW_SPEC:
        row = {"Statistic": label}
        vals = []
        for a in approaches:
            v = agent_totals[a][label]
            orig_v = orig_totals[label]
            pct = (v / orig_v * 100.0) if orig_v else 0.0
            vals.append(pct)
            row[_short(a)] = f"{v:,} ({pct:.0f}\\%)"
        row["Avg"] = f"{sum(vals)/len(vals):.0f}\\%" if vals else ""
        panel_b_rows.append(row)
    # Completion rate row
    row = {"Statistic": "Completion rate"}
    comp_vals = []
    for a in approaches:
        p = completion_pct[a]
        comp_vals.append(p)
        row[_short(a)] = f"{p:.0f}\\%"
    row["Avg"] = f"{sum(comp_vals)/len(comp_vals):.0f}\\%" if comp_vals else ""
    panel_b_rows.append(row)

    panel_b = pd.DataFrame(panel_b_rows)

    # CSVs (drop LaTeX escapes for CSV)
    def _strip(v):
        return str(v).replace("\\%", "%").replace("\\_", "_")
    panel_a.to_csv(target / "summary_stats_panel_a.csv", index=False)
    panel_b.map(_strip).to_csv(target / "summary_stats_panel_b.csv", index=False)

    # LaTeX — two panels in one table.
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    # Panel A
    lines.append(r"\textbf{Panel A: Originals (per paper, $N=48$)}\\[3pt]")
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(r" & Total & Mean & SD & Min & Max \\")
    lines.append(r"\midrule")
    for _, r in panel_a.iterrows():
        lines.append(f"{r['Statistic']} & {int(r['Total']):,} & {r['Mean']} & {r['SD']} & {r['Min']} & {r['Max']} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"")
    lines.append(r"\vspace{1.5em}")
    lines.append(r"")
    # Panel B
    lines.append(r"\textbf{Panel B: Agent systems (share of originals, \%)}\\[3pt]")
    n_cols = len(approaches) + 1  # + Avg
    col_fmt = "l" + "r" * n_cols
    lines.append(r"\begin{tabular}{" + col_fmt + "}")
    lines.append(r"\toprule")
    header = " & " + " & ".join([f"\\textbf{{{_short(a)}}}" for a in approaches] + [r"\textbf{Avg}"]) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")
    for _, r in panel_b.iterrows():
        if r["Statistic"] == "Completion rate":
            lines.append(r"\midrule")
        cells = [str(r[_short(a)]) for a in approaches] + [str(r["Avg"])]
        lines.append(f"{r['Statistic']} & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{\textbf{Descriptive overview — two panels.} Panel A: per-paper summary statistics on the original outputs "
                 r"($N=48$ papers). Panel B: reproductions, cell counts given as absolute totals with the percentage of originals in parentheses; "
                 r"\textit{Avg} is the mean of the shares across the 7 approach--model combinations. "
                 r"\textit{Completion rate} = mean across papers of (cells produced by the agent / cells present in the original), equal weight per paper.}")
    lines.append(r"\label{tab:summary_stats_panels}")
    lines.append(r"\end{table}")
    (target / "summary_stats_panels.tex").write_text("\n".join(lines))
    print(f"  Saved {subdir + '/' if subdir else ''}summary_stats_panels.tex "
          f"(+ panel_a.csv, panel_b.csv)")


def generate_journal_discipline_table(df_runs: pd.DataFrame, output_dir: Path, subdir: str = ""):
    """One row per journal: journal, discipline, number of papers; sorted desc by count."""
    target = output_dir / subdir if subdir else output_dir
    target.mkdir(parents=True, exist_ok=True)
    if df_runs.empty:
        return
    # Deduplicate to (paper_slug, journal) pairs
    pj = df_runs.drop_duplicates("paper_slug")[["paper_slug", "journal"]].copy()
    pj["discipline"] = pj["journal"].map(lambda j: JOURNAL_DISCIPLINE.get(j, "Other"))
    counts = pj.groupby(["journal", "discipline"], observed=True).size().reset_index(name="n_papers")
    counts = counts.sort_values(["n_papers", "journal"], ascending=[False, True])
    counts.to_csv(target / "journal_discipline.csv", index=False)

    total = int(counts["n_papers"].sum())
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Papers by journal and discipline (final sample, $N=" + str(total) + r"$).}",
        r"\label{tab:journal_discipline}",
        r"\begin{tabular}{llr}",
        r"\toprule",
        r"Journal & Discipline & N papers \\",
        r"\midrule",
    ]
    for _, r in counts.iterrows():
        lines.append(f"{r['journal']} & {r['discipline']} & {int(r['n_papers'])} \\\\")
    lines.append(r"\midrule")
    lines.append(f"Total &  & {total} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    (target / "journal_discipline.tex").write_text("\n".join(lines))
    print(f"  Saved {subdir + '/' if subdir else ''}journal_discipline.tex (+ .csv)")


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
        n_tables_na = (table_items["grade"] == "NA").sum()
        n_tables_non_f = n_tables_total - n_tables_f - n_tables_na
        n_papers_non_f = (grp["overall_grade"] != "F").sum()
        n_papers_non_na = (~grp["overall_grade"].isin(["F", "NA"])).sum()
        n_extra_total = grp["n_extra_table_jsons"].sum()
        # F reason breakdown
        f_reasons = table_items[table_items["grade"] == "F"]["f_reason"].value_counts()
        rows.append({
            "approach": approach_raw, "model": model_val,
            "n_papers": int(n_papers),
            "n_papers_non_f": int(n_papers_non_f),
            "n_papers_non_f_na": int(n_papers_non_na),
            "n_papers_with_table_templates": int(n_papers_with_templates),
            "n_table_templates_total": int(n_table_templates_total),
            "n_runs_with_py_files": int(n_runs_with_py),
            "n_table_jsons_produced": int(n_table_jsons_total),
            "n_extra_table_jsons": int(n_extra_total),
            "n_tables_in_verification": int(n_tables_total),
            "n_tables_non_f": int(n_tables_non_f),
            "n_tables_f": int(n_tables_f),
            "n_tables_na": int(n_tables_na),
            "n_f_not_produced": int(f_reasons.get("not_produced", 0)),
            "n_f_pre_aligned_all_f": int(f_reasons.get("pre_aligned_all_f", 0)),
            "n_f_no_pre_aligned": int(f_reasons.get("no_pre_aligned", 0)),
            "n_f_other": int(f_reasons.get("other_f", 0)),
        })
    overview = pd.DataFrame(rows)
    target = output_dir / subdir if subdir else output_dir
    target.mkdir(parents=True, exist_ok=True)
    overview.to_csv(target / "overview_by_approach_model.csv", index=False)
    print("  Saved overview_by_approach_model.csv")
    print(overview.to_string(index=False))


def generate_missingness_reports(df_runs: pd.DataFrame, df_items: pd.DataFrame, output_dir: Path, subdir: str = ""):
    """Generate CSV reports on missingness: F papers, missing approaches, table gaps."""
    target = output_dir / subdir if subdir else output_dir
    target.mkdir(parents=True, exist_ok=True)

    all_approaches = sorted(df_runs["approach"].unique())
    all_papers = sorted(df_runs["paper_slug"].unique())

    # --- 1. For each approach, which papers are completely missing (overall F) ---
    rows_f = []
    for approach in all_approaches:
        sub = df_runs[df_runs["approach"] == approach]
        n_total = len(sub)
        f_papers = sorted(sub[sub["overall_grade"] == "F"]["paper_slug"].tolist())
        missing_papers = sorted(set(all_papers) - set(sub["paper_slug"].unique()))
        rows_f.append({
            "approach": approach,
            "n_papers_total": n_total,
            "n_papers_F": len(f_papers),
            "n_papers_missing": len(missing_papers),
            "n_papers_F_or_missing": len(f_papers) + len(missing_papers),
            "F_papers": "; ".join(f_papers),
            "missing_papers": "; ".join(missing_papers),
        })
    df_f = pd.DataFrame(rows_f)
    df_f.to_csv(target / "missingness_f_papers_by_approach.csv", index=False)

    # --- 2. For each paper, which approaches are missing entirely ---
    rows_missing = []
    for paper in all_papers:
        paper_approaches = set(df_runs[df_runs["paper_slug"] == paper]["approach"].unique())
        missing = sorted(set(all_approaches) - paper_approaches)
        f_approaches = sorted(
            df_runs[(df_runs["paper_slug"] == paper) & (df_runs["overall_grade"] == "F")]["approach"].unique()
        )
        rows_missing.append({
            "paper_slug": paper,
            "n_approaches_total": len(all_approaches),
            "n_approaches_present": len(paper_approaches),
            "n_approaches_missing": len(missing),
            "n_approaches_F": len(f_approaches),
            "missing_approaches": "; ".join(missing),
            "F_approaches": "; ".join(f_approaches),
        })
    df_missing = pd.DataFrame(rows_missing)
    df_missing.to_csv(target / "missingness_approaches_by_paper.csv", index=False)

    # --- 3. Per paper × approach: F tables, missing templates, extra tables ---
    rows_tables = []
    for _, run in df_runs.iterrows():
        paper = run["paper_slug"]
        approach = run["approach"]
        n_templates = run.get("n_table_templates", 0)
        n_jsons = run.get("n_table_jsons", 0)
        n_extra = run.get("n_extra_table_jsons", 0)
        n_missing_templates = max(0, n_templates - (n_jsons - n_extra))

        # F tables from items
        items = df_items[(df_items["paper_slug"] == paper) & (df_items["approach"] == approach)]
        table_items = items[items["item_type"] == "table"]
        n_f_tables = (table_items["grade"] == "F").sum() if len(table_items) > 0 else 0

        rows_tables.append({
            "paper_slug": paper,
            "approach": approach,
            "n_table_templates": int(n_templates),
            "n_table_jsons": int(n_jsons),
            "n_missing_from_templates": int(n_missing_templates),
            "n_extra_not_in_templates": int(n_extra),
            "n_f_tables": int(n_f_tables),
            "n_verified_tables": len(table_items),
        })
    df_tables = pd.DataFrame(rows_tables)
    df_tables.to_csv(target / "missingness_tables_detail.csv", index=False)

    # --- 4. Aggregated per paper: sum across approaches, filter to papers with gaps ---
    agg = df_tables.groupby("paper_slug").agg(
        n_approaches=("approach", "count"),
        n_table_templates_max=("n_table_templates", "max"),
        total_table_jsons=("n_table_jsons", "sum"),
        total_missing_from_templates=("n_missing_from_templates", "sum"),
        total_extra_not_in_templates=("n_extra_not_in_templates", "sum"),
        total_f_tables=("n_f_tables", "sum"),
        total_verified_tables=("n_verified_tables", "sum"),
    ).reset_index()
    # Add per-approach detail: which approaches have missing or F tables
    approach_issues = {}
    for paper, grp in df_tables.groupby("paper_slug"):
        missing_apps = grp[grp["n_missing_from_templates"] > 0]["approach"].tolist()
        f_apps = grp[grp["n_f_tables"] > 0]["approach"].tolist()
        approach_issues[paper] = {
            "approaches_with_missing_templates": "; ".join(sorted(missing_apps)),
            "approaches_with_f_tables": "; ".join(sorted(f_apps)),
        }
    agg["approaches_with_missing_templates"] = agg["paper_slug"].map(
        lambda p: approach_issues.get(p, {}).get("approaches_with_missing_templates", ""))
    agg["approaches_with_f_tables"] = agg["paper_slug"].map(
        lambda p: approach_issues.get(p, {}).get("approaches_with_f_tables", ""))

    agg_filtered = agg[
        (agg["total_missing_from_templates"] > 0) | (agg["total_f_tables"] > 0)
    ].copy()
    agg_filtered.to_csv(target / "missingness_tables_by_paper.csv", index=False)

    print(f"  Saved missingness_f_papers_by_approach.csv ({len(df_f)} rows)")
    print(f"  Saved missingness_approaches_by_paper.csv ({len(df_missing)} rows)")
    print(f"  Saved missingness_tables_detail.csv ({len(df_tables)} rows)")
    print(f"  Saved missingness_tables_by_paper.csv ({len(agg_filtered)} papers with gaps)")


# ============================================================================
# Section: Paper Level
# ============================================================================

def plot_overall_grade_distribution(df_runs: pd.DataFrame, output_dir: Path, subdir: str = "",
                                     f_mode: str = "all_f",
                                     grade_col: str = "overall_grade", name_suffix: str = ""):
    """Grouped bar chart of overall grades by approach (excludes NA).

    ``grade_col`` is expected to already encode the f_mode semantics (it should
    be a pre-aggregated column such as ``overall_grade_all_f``). ``f_mode`` here
    only affects which grade buckets are shown and the filename suffix.
    """
    if grade_col not in df_runs.columns:
        return
    df = df_runs[df_runs[grade_col] != "NA"]
    grades_shown = [g for g in GRADE_ORDER if g != "NA"]
    if f_mode == "no_f":
        grades_shown = [g for g in grades_shown if g != "F"]
    ct = pd.crosstab(df["approach"], df[grade_col], normalize="index") * 100
    ct = ct.reindex(columns=grades_shown, fill_value=0)

    fig, ax = plt.subplots(figsize=(7, 5))
    present = _approaches_in(df)
    x = np.arange(len(present))
    width = 0.15
    for i, grade in enumerate(grades_shown):
        vals = [ct.loc[a, grade] if a in ct.index else 0 for a in present]
        ax.bar(x + i * width, vals, width, label=grade, color=GRADE_COLORS[grade], edgecolor="white")

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([APPROACH_LABELS.get(a, a).replace("\n", " ") for a in present], fontsize=10, rotation=25, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("Share of Runs (%)", fontsize=18, fontweight="bold")
    place_legend(fig, ax, fontsize=14, ncol=6)
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig, output_dir, f"overall_grades{F_MODE_SUFFIX[f_mode]}{name_suffix}", subdir)


def plot_overall_grade_cumulative(df_runs: pd.DataFrame, output_dir: Path, subdir: str = "",
                                   f_mode: str = "all_f",
                                   grade_col: str = "overall_grade", name_suffix: str = ""):
    """Dot plot of cumulative paper-level grade shares: ≥A, ≥B, ≥C, ≥D, ≥E."""
    if grade_col not in df_runs.columns:
        return
    df = df_runs[df_runs[grade_col] != "NA"].copy()
    if df.empty:
        return

    grades_cum = [g for g in GRADE_ORDER if g != "NA"]
    if f_mode == "no_f":
        grades_cum = [g for g in grades_cum if g != "F"]

    approaches = _approaches_in(df)
    ct = pd.crosstab(df["approach"], df[grade_col], normalize="index") * 100
    ct = ct.reindex(columns=grades_cum, fill_value=0)

    cum = pd.DataFrame(index=ct.index, columns=[f"≥{g}" for g in grades_cum])
    for i, g in enumerate(grades_cum):
        cum[f"≥{g}"] = ct[grades_cum[:i + 1]].sum(axis=1)

    present = [a for a in approaches if a in cum.index]
    sort_key = cum.loc[present, "≥B"].values
    order = np.argsort(-sort_key)
    present = [present[i] for i in order]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    y_pos = np.arange(len(present))

    for g_idx, g in enumerate(grades_cum):
        col = f"≥{g}"
        vals = [cum.loc[a, col] for a in present]
        color = GRADE_COLORS.get(g, "#95a5a6")
        for i, (a, v) in enumerate(zip(present, vals)):
            ax.text(v, i, g, fontsize=14, fontweight="bold", color=color,
                    ha="center", va="center", zorder=5)
            if g_idx > 0:
                prev_col = f"≥{grades_cum[g_idx - 1]}"
                prev_v = cum.loc[a, prev_col]
                ax.plot([prev_v, v], [i, i], color="#cccccc", linewidth=1, zorder=1)
    ax.set_yticks(y_pos)
    labels = [APPROACH_LABELS.get(a, a).replace("\n", " ") for a in present]
    colors = [APPROACH_COLORS.get(a, "#95a5a6") for a in present]
    ax.set_yticklabels(labels, fontsize=10)
    for tick_label, color in zip(ax.get_yticklabels(), colors):
        tick_label.set_color(color)
        tick_label.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("Cumulative share of papers (%)", fontsize=12, fontweight="bold")
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig, output_dir, f"overall_grades_cumulative{F_MODE_SUFFIX[f_mode]}{name_suffix}", subdir)


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


def plot_paper_difficulty(df_runs: pd.DataFrame, output_dir: Path, subdir: str = "",
                          f_mode: str = "all_f"):
    df = _filter_f_mode(df_runs, f_mode, level="paper")
    df = df[df["overall_grade"] != "NA"]
    if df.empty:
        return
    agg = df.groupby(["paper_slug", "paper_title"])["overall_grade_num"].agg(["mean", "min", "max"])
    agg = agg.sort_values("mean", ascending=True)
    if len(agg) > 40:
        agg = pd.concat([agg.head(20), agg.tail(20)])
    labels = []
    for slug, title in agg.index:
        journal = _infer_journal(slug)
        # Use slug-based short ID for consistency (most papers lack metadata titles)
        short_id = slug.split("_", 1)[-1] if "_" in slug else slug
        labels.append(f"{journal} — {short_id}" if journal != "Other" else short_id)

    fig, ax = plt.subplots(figsize=(12, max(8, len(agg) * 0.35)))
    y_pos = range(len(agg))
    colors = [GRADE_COLORS.get(NUM_TO_GRADE.get(int(np.floor(v + 0.5)), "F"), "#95a5a6") for v in agg["mean"].values]
    ax.barh(y_pos, agg["mean"].values, color=colors, edgecolor="white", alpha=0.8)
    xerr_low = agg["mean"].values - agg["min"].values
    xerr_high = agg["max"].values - agg["mean"].values
    ax.errorbar(agg["mean"].values, y_pos, xerr=[xerr_low, xerr_high],
                fmt="none", ecolor="black", elinewidth=1.2, capsize=3, capthick=1.0)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=10)
    # Only show numeric grades on x-axis (A=5 down to F=0); NA has no numeric value
    numeric_grades = [g for g in GRADE_ORDER if g != "NA"]
    ax.set_xticks(range(len(numeric_grades)))
    ax.set_xticklabels(numeric_grades[::-1])
    ax.set_xlabel("Mean Grade (across approaches)", fontsize=18, fontweight="bold")
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig, output_dir, f"paper_difficulty{F_MODE_SUFFIX[f_mode]}", subdir)


def plot_scatter_vs_grade(df: pd.DataFrame, x_col: str, x_label: str, output_dir: Path,
                          name: str, log_x: bool = False, grade_col: str = "overall_grade_num",
                          subdir: str = "", f_mode: str = "all_f"):
    df_plot = df.dropna(subset=[x_col, grade_col]).copy()
    grade_str_col = "overall_grade" if grade_col == "overall_grade_num" else "grade"
    level = "paper" if grade_col == "overall_grade_num" else "item"
    df_plot = _filter_f_mode(df_plot, f_mode, level=level, grade_col=grade_str_col)
    if grade_str_col in df_plot.columns:
        df_plot = df_plot[df_plot[grade_str_col] != "NA"]
    if df_plot.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5.5))
    jitter_rng = np.random.default_rng(42)
    for approach in _approaches_in(df_plot):
        sub = df_plot[df_plot["approach"] == approach]
        if sub.empty:
            continue
        ax.scatter(sub[x_col], sub[grade_col] + jitter_rng.uniform(-0.15, 0.15, len(sub)),
                   color=APPROACH_COLORS.get(approach, "#95a5a6"), label=APPROACH_LABELS.get(approach, approach).replace("\n", " "),
                   alpha=0.6, s=60, edgecolor="white", linewidth=0.5)

    if log_x and df_plot[x_col].gt(0).any():
        ax.set_xscale("log")
    ax.set_xlabel(x_label, fontsize=18, fontweight="bold")
    ax.set_ylabel("Overall Grade", fontsize=18, fontweight="bold")
    # OLD: excluded F
    # grades_shown = [g for g in GRADE_ORDER if g != "F"] if exclude_f else GRADE_ORDER
    # NEW: exclude NA (no numeric value) from y-axis ticks always
    grades_shown = [g for g in GRADE_ORDER if g != "NA"]
    ax.set_yticks([GRADE_TO_NUM[g] for g in grades_shown])
    ax.set_yticklabels(grades_shown[::-1])
    place_legend(fig, ax, fontsize=12)
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig, output_dir, f"{name}{F_MODE_SUFFIX[f_mode]}", subdir)


def plot_grade_by_discipline(df_runs: pd.DataFrame, output_dir: Path, subdir: str = "",
                              f_mode: str = "all_f"):
    df = _filter_f_mode(df_runs, f_mode, level="paper")
    df = df[(df["overall_grade"] != "NA") & (df["discipline"] != "Other")]
    grades_shown = [g for g in GRADE_ORDER if g != "NA"]
    if f_mode == "no_f":
        grades_shown = [g for g in grades_shown if g != "F"]
    if df.empty:
        return
    ct = pd.crosstab(df["discipline"], df["overall_grade"], normalize="index") * 100
    ct = ct.reindex(columns=grades_shown, fill_value=0)
    ct["_mean"] = sum(ct[g] * GRADE_TO_NUM[g] for g in grades_shown if g in ct.columns) / 100
    ct = ct.sort_values("_mean", ascending=False).drop(columns="_mean")

    fig, ax = plt.subplots(figsize=(7, 5))
    ct.plot(kind="bar", stacked=True, ax=ax, color=[GRADE_COLORS[g] for g in ct.columns], edgecolor="white", width=0.7)
    ax.set_xlabel("Discipline", fontsize=18, fontweight="bold")
    ax.set_ylabel("Share (%)", fontsize=18, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    place_legend(fig, ax, fontsize=12, ncol=6)
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig, output_dir, f"grade_by_discipline{F_MODE_SUFFIX[f_mode]}", subdir)


def plot_grade_by_language(df_runs: pd.DataFrame, output_dir: Path, subdir: str = "",
                            f_mode: str = "all_f"):
    df = _filter_f_mode(df_runs, f_mode, level="paper")
    df = df[(df["overall_grade"] != "NA") & (df["original_language"] != "Unknown")]
    grades_shown = [g for g in GRADE_ORDER if g != "NA"]
    if f_mode == "no_f":
        grades_shown = [g for g in grades_shown if g != "F"]
    if df.empty:
        return

    # Primary language — standalone figure
    fig1, ax = plt.subplots(figsize=(7, 5))
    ct = pd.crosstab(df["original_language"], df["overall_grade"], normalize="index") * 100
    ct = ct.reindex(columns=grades_shown, fill_value=0)
    ct["_mean"] = sum(ct[g] * GRADE_TO_NUM[g] for g in grades_shown if g in ct.columns) / 100
    ct = ct.sort_values("_mean", ascending=False).drop(columns="_mean")
    lang_counts = df.drop_duplicates("paper_slug").groupby("original_language").size()
    ct.index = [f"{lang} (n={lang_counts.get(lang, 0)})" for lang in ct.index]
    ct.plot(kind="bar", stacked=True, ax=ax, color=[GRADE_COLORS[g] for g in ct.columns], edgecolor="white", width=0.7)
    ax.set_xlabel("Primary Language", fontsize=14, fontweight="bold")
    ax.set_ylabel("Share (%)", fontsize=14, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    place_legend(fig1, ax, fontsize=11, ncol=5)
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig1, output_dir, f"grade_by_language_primary{F_MODE_SUFFIX[f_mode]}", subdir)

    # Combined two-panel figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ct.plot(kind="bar", stacked=True, ax=ax, color=[GRADE_COLORS[g] for g in ct.columns], edgecolor="white", width=0.7)
    ax.set_xlabel("Primary Language", fontsize=14, fontweight="bold")
    ax.set_ylabel("Share (%)", fontsize=14, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.get_legend().remove()
    apply_style(ax)

    ax2 = axes[1]
    ct2 = pd.crosstab(df["original_languages_all"], df["overall_grade"], normalize="index") * 100
    ct2 = ct2.reindex(columns=grades_shown, fill_value=0)
    ct2["_mean"] = sum(ct2[g] * GRADE_TO_NUM[g] for g in grades_shown if g in ct2.columns) / 100
    ct2 = ct2.sort_values("_mean", ascending=False).drop(columns="_mean")
    combo_counts = df.drop_duplicates("paper_slug").groupby("original_languages_all").size()
    ct2.index = [f"{combo} (n={combo_counts.get(combo, 0)})" for combo in ct2.index]
    ct2.plot(kind="bar", stacked=True, ax=ax2, color=[GRADE_COLORS[g] for g in ct2.columns], edgecolor="white", width=0.7)
    ax2.set_xlabel("Language Combination", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Share (%)", fontsize=14, fontweight="bold")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, ha="right")
    place_legend(fig, ax2, fontsize=11, ncol=5)
    apply_style(ax2)

    plt.tight_layout()
    save_figure(fig, output_dir, f"grade_by_language{F_MODE_SUFFIX[f_mode]}", subdir)


def plot_tokens_vs_grade_within_paper(df_runs: pd.DataFrame, output_dir: Path, subdir: str = ""):
    """Two visualisations of whether more tokens → better grade, controlling for paper.

    (D) Scatter of pairwise differences across approaches within each paper:
        For each paper and each pair of approaches (A, B) with tokens_A < tokens_B,
        plot (Δ tokens, Δ grade). A positive trend means extra tokens yield better
        grades within the same paper.

    (E) Histogram of per-paper Spearman correlations between tokens and grade
        across the approaches available for that paper.
    """
    df = df_runs[
        (df_runs["total_tokens"] > 0) &
        df_runs["overall_grade_num"].notna() &
        (df_runs["overall_grade"] != "NA")
    ].copy()
    if df.empty or df["paper_slug"].nunique() < 2:
        print("  Skipping tokens_vs_grade_within_paper: not enough data")
        return

    target = output_dir / subdir if subdir else output_dir
    target.mkdir(parents=True, exist_ok=True)

    # ── (D) Pairwise differences scatter ─────────────────────────────────
    from itertools import combinations
    records = []
    for paper, grp in df.groupby("paper_slug", observed=True):
        rows = grp[["approach", "total_tokens", "overall_grade_num"]].dropna()
        if len(rows) < 2:
            continue
        for a, b in combinations(rows.itertuples(index=False), 2):
            if a.total_tokens == b.total_tokens:
                continue
            lo, hi = (a, b) if a.total_tokens < b.total_tokens else (b, a)
            records.append({
                "paper_slug": paper,
                "approach_lo": str(lo.approach),
                "approach_hi": str(hi.approach),
                "delta_tokens": hi.total_tokens - lo.total_tokens,
                "delta_grade": hi.overall_grade_num - lo.overall_grade_num,
            })
    if records:
        pairs = pd.DataFrame(records)
        pairs.to_csv(target / "tokens_vs_grade_pairs.csv", index=False)

        fig, ax = plt.subplots(figsize=(9, 5.5))
        # Millions of tokens on log x-axis for legibility
        x = pairs["delta_tokens"] / 1e6
        y = pairs["delta_grade"]
        ax.scatter(x, y, alpha=0.35, color="#3498db", s=18, edgecolor="none")
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)

        # LOESS-ish smoother via rolling median in x bins
        x_sorted = x.sort_values()
        if len(x_sorted) >= 20:
            nbins = min(15, max(5, len(x_sorted) // 50))
            q = np.linspace(0, 1, nbins + 1)
            edges = x_sorted.quantile(q).values
            mid_x = (edges[:-1] + edges[1:]) / 2
            mid_y = []
            for lo_e, hi_e in zip(edges[:-1], edges[1:]):
                mask = (x >= lo_e) & (x <= hi_e)
                mid_y.append(y[mask].median() if mask.any() else np.nan)
            ax.plot(mid_x, mid_y, color="#e67e22", linewidth=2.2,
                    label="Binned median Δ grade")

        # Overall share of positive Δ grades
        n_pos = (y > 0).sum()
        n_neg = (y < 0).sum()
        n_zero = (y == 0).sum()
        ax.text(0.02, 0.97,
                f"n pairs = {len(pairs)}\nΔ grade > 0: {n_pos} ({n_pos / len(pairs) * 100:.0f}%)\n"
                f"Δ grade = 0: {n_zero} ({n_zero / len(pairs) * 100:.0f}%)\n"
                f"Δ grade < 0: {n_neg} ({n_neg / len(pairs) * 100:.0f}%)",
                transform=ax.transAxes, fontsize=10, va="top", ha="left",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="#cccccc"))

        ax.set_xlabel("Extra tokens spent on same paper (millions)",
                      fontsize=11, fontweight="bold")
        ax.set_ylabel("Δ grade (higher-token approach − lower-token approach)",
                      fontsize=11, fontweight="bold")
        ax.legend(fontsize=10, loc="lower right")
        apply_style(ax)
        plt.tight_layout()
        save_figure(fig, output_dir, "tokens_vs_grade_pairwise", subdir)

    # ── (E) Per-paper correlation histogram ──────────────────────────────
    corrs = []
    for paper, grp in df.groupby("paper_slug", observed=True):
        rows = grp[["total_tokens", "overall_grade_num"]].dropna()
        if len(rows) < 3:
            continue
        # Spearman: rank-based, robust to monotonic non-linearity
        try:
            c = rows["total_tokens"].corr(rows["overall_grade_num"], method="spearman")
            if pd.notna(c):
                corrs.append({"paper_slug": paper, "spearman": c, "n_approaches": len(rows)})
        except Exception:
            pass
    if corrs:
        cdf = pd.DataFrame(corrs)
        cdf.to_csv(target / "tokens_vs_grade_per_paper_correlation.csv", index=False)

        fig, ax = plt.subplots(figsize=(8, 5))
        vals = cdf["spearman"].values
        ax.hist(vals, bins=np.arange(-1.05, 1.05 + 0.1, 0.1),
                color="#3498db", edgecolor="white", alpha=0.85)
        ax.axvline(0, color="black", linewidth=0.8, alpha=0.6)
        mean_c = vals.mean()
        median_c = np.median(vals)
        n_pos = (vals > 0).sum()
        n_neg = (vals < 0).sum()
        ax.axvline(median_c, color="#e67e22", linewidth=2, linestyle="--",
                    label=f"median = {median_c:.2f}")
        ax.text(0.02, 0.97,
                f"n papers = {len(cdf)}\n"
                f"mean ρ = {mean_c:.2f}\n"
                f"median ρ = {median_c:.2f}\n"
                f"ρ > 0: {n_pos} ({n_pos / len(cdf) * 100:.0f}%)\n"
                f"ρ < 0: {n_neg} ({n_neg / len(cdf) * 100:.0f}%)",
                transform=ax.transAxes, fontsize=10, va="top", ha="left",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="#cccccc"))
        ax.set_xlabel("Per-paper Spearman correlation (tokens vs grade)",
                      fontsize=11, fontweight="bold")
        ax.set_ylabel("Number of papers", fontsize=11, fontweight="bold")
        ax.legend(fontsize=10)
        apply_style(ax)
        plt.tight_layout()
        save_figure(fig, output_dir, "tokens_vs_grade_correlations", subdir)


def plot_duration_vs_grade(df_runs: pd.DataFrame, output_dir: Path, subdir: str = ""):
    """Left: duration distribution per approach. Right: duration vs grade. Both excl. NA."""
    # OLD: excluded F
    # df = df_runs[df_runs["duration_seconds"].notna() & df_runs["overall_grade_num"].notna() & (df_runs["overall_grade"] != "F")].copy()
    # NEW: exclude NA instead
    df = df_runs[df_runs["duration_seconds"].notna() & df_runs["overall_grade_num"].notna() & (df_runs["overall_grade"] != "NA")].copy()
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
    jitter_rng = np.random.default_rng(42)
    for approach in _approaches_in(df):
        sub = df[df["approach"] == approach]
        if sub.empty:
            continue
        ax2.scatter(sub["duration_seconds"] / 60, sub["overall_grade_num"] + jitter_rng.uniform(-0.15, 0.15, len(sub)),
                   color=APPROACH_COLORS.get(approach, "#95a5a6"), label=APPROACH_LABELS.get(approach, approach).replace("\n", " "),
                   alpha=0.6, s=60, edgecolor="white", linewidth=0.5)
    ax2.set_xlabel("Duration (minutes)", fontsize=18, fontweight="bold")
    ax2.set_ylabel("Overall Grade", fontsize=18, fontweight="bold")
    # OLD: excluded F
    # grades_shown = [g for g in GRADE_ORDER if g != "F"]
    # NEW: exclude NA instead
    grades_shown = [g for g in GRADE_ORDER if g != "NA"]
    ax2.set_yticks([GRADE_TO_NUM[g] for g in grades_shown])
    ax2.set_yticklabels(grades_shown[::-1])
    place_legend(fig, ax2, fontsize=12)
    apply_style(ax2)

    plt.tight_layout()
    save_figure(fig, output_dir, "duration_vs_grade", subdir)

    # Standalone horizontal duration plot (mean + 95% CI)
    fig2, ax3 = plt.subplots(figsize=(7, 5))
    durations = []
    for a in approaches:
        vals = df.loc[df["approach"] == a, "duration_seconds"].dropna() / 60
        if not vals.empty:
            mean = vals.mean()
            ci = 1.96 * vals.std() / np.sqrt(len(vals))
            durations.append((mean, ci, APPROACH_LABELS.get(a, a).replace("\n", " "),
                              APPROACH_COLORS.get(a, "#95a5a6")))
    durations.sort(key=lambda x: x[0])
    y = np.arange(len(durations))
    for i, (mean, ci, label, color) in enumerate(durations):
        ax3.errorbar(mean, i, xerr=ci, fmt="o", markersize=10, color=color,
                     ecolor="#cccccc", capsize=4, capthick=1.5, linewidth=1.5, zorder=5)
        ax3.text(mean + ci + 1, i, f"{mean:.0f}m", va="center", fontsize=10, fontweight="bold")
    ax3.set_yticks(y)
    ax3.set_yticklabels([d[2] for d in durations], fontsize=10)
    for tick_label, (_, _, _, color) in zip(ax3.get_yticklabels(), durations):
        tick_label.set_color(color)
        tick_label.set_fontweight("bold")
    ax3.set_xlabel("Duration (minutes)", fontsize=12, fontweight="bold")
    ax3.invert_yaxis()
    apply_style(ax3)
    plt.tight_layout()
    save_figure(fig2, output_dir, "duration_by_approach", subdir)

    # Token usage plot (mean + 95% CI, excl. NA)
    # OLD: excluded F
    # df_tok = df_runs[(df_runs["total_tokens"] > 0) & (df_runs["overall_grade"] != "F")].copy()
    # NEW: exclude NA instead
    df_tok = df_runs[(df_runs["total_tokens"] > 0) & (df_runs["overall_grade"] != "NA")].copy()
    if not df_tok.empty:
        fig3, ax4 = plt.subplots(figsize=(7, 5))
        tok_data = []
        for a in _approaches_in(df_tok):
            vals = df_tok.loc[df_tok["approach"] == a, "total_tokens"].dropna() / 1e6
            if not vals.empty:
                mean = vals.mean()
                ci = 1.96 * vals.std() / np.sqrt(len(vals))
                tok_data.append((mean, ci, APPROACH_LABELS.get(a, a).replace("\n", " "),
                                 APPROACH_COLORS.get(a, "#95a5a6")))
        tok_data.sort(key=lambda x: x[0])
        y = np.arange(len(tok_data))
        for i, (mean, ci, label, color) in enumerate(tok_data):
            ax4.errorbar(mean, i, xerr=ci, fmt="o", markersize=10, color=color,
                         ecolor="#cccccc", capsize=4, capthick=1.5, linewidth=1.5, zorder=5)
            ax4.text(mean + ci + 0.1, i, f"{mean:.1f}M", va="center", fontsize=10, fontweight="bold")
        ax4.set_yticks(y)
        ax4.set_yticklabels([d[2] for d in tok_data], fontsize=10)
        for tick_label, (_, _, _, color) in zip(ax4.get_yticklabels(), tok_data):
            tick_label.set_color(color)
            tick_label.set_fontweight("bold")
        ax4.set_xlabel("Total tokens (millions)", fontsize=12, fontweight="bold")
        ax4.set_xlim(0, 9)
        ax4.invert_yaxis()
        apply_style(ax4)
        plt.tight_layout()
        save_figure(fig3, output_dir, "tokens_by_approach", subdir)

    # Cost plot (mean + 95% CI, excl. NA)
    # OLD: excluded F
    # df_cost = df_runs[(df_runs["total_cost_usd"] > 0) & (df_runs["overall_grade"] != "F")].copy()
    # NEW: exclude NA instead
    df_cost = df_runs[(df_runs["total_cost_usd"] > 0) & (df_runs["overall_grade"] != "NA")].copy()
    if not df_cost.empty:
        fig4, ax5 = plt.subplots(figsize=(7, 5))
        cost_data = []
        for a in _approaches_in(df_cost):
            vals = df_cost.loc[df_cost["approach"] == a, "total_cost_usd"].dropna()
            if not vals.empty:
                mean = vals.mean()
                ci = 1.96 * vals.std() / np.sqrt(len(vals))
                cost_data.append((mean, ci, APPROACH_LABELS.get(a, a).replace("\n", " "),
                                  APPROACH_COLORS.get(a, "#95a5a6")))
        cost_data.sort(key=lambda x: x[0])
        y = np.arange(len(cost_data))
        for i, (mean, ci, label, color) in enumerate(cost_data):
            ax5.errorbar(mean, i, xerr=ci, fmt="o", markersize=10, color=color,
                         ecolor="#cccccc", capsize=4, capthick=1.5, linewidth=1.5, zorder=5)
            ax5.text(mean + ci + 0.1, i, f"${mean:.2f}", va="center", fontsize=10, fontweight="bold")
        ax5.set_yticks(y)
        ax5.set_yticklabels([d[2] for d in cost_data], fontsize=10)
        for tick_label, (_, _, _, color) in zip(ax5.get_yticklabels(), cost_data):
            tick_label.set_color(color)
            tick_label.set_fontweight("bold")
        ax5.set_xlabel("Cost per paper (USD)", fontsize=12, fontweight="bold")
        ax5.invert_yaxis()
        apply_style(ax5)
        plt.tight_layout()
        save_figure(fig4, output_dir, "cost_by_approach", subdir)


# ============================================================================
# Section: Computational Efficiency
# ============================================================================

# $/M tokens (input, output), March 2026 list prices. Used only to impute run
# cost where the scaffold did not report one (Codex CLI). The cached-input
# share of those runs is unknown, so imputed costs are upper bounds.
MODEL_PRICING_USD_PER_MTOK = {
    "gpt-5.3-codex": (1.75, 14.00),
    "gpt-5.4": (1.75, 14.00),
}

# A table counts as a "success" if graded at or above this set.
EFFICIENCY_SUCCESS_GRADES = {"A", "B"}

EFFICIENCY_EFFORT_DIMS = {
    # effort key -> (per-run column, axis label, figure name)
    "cost": ("cost_usd", "Mean cost per paper (USD, log scale)",
             "efficiency_frontier_cost"),
    "tokens": ("completion_ktok", "Mean completion tokens per paper (thousands, log scale)",
               "efficiency_frontier_tokens"),
    "time": ("duration_min", "Mean wall-clock time per paper (minutes, log scale)",
             "efficiency_frontier_time"),
}


def _effort_frame(df_runs: pd.DataFrame, df_items: pd.DataFrame,
                  f_mode: str = "all_f") -> pd.DataFrame:
    """One row per run with effort and success metrics.

    Success = number of table items graded in EFFICIENCY_SUCCESS_GRADES under
    the given f_mode. NA tables stay in the denominator: they are identical
    across approaches, so they shift all success rates equally.
    Cost is the reported run cost where available, otherwise imputed from the
    prompt/completion token split via MODEL_PRICING_USD_PER_MTOK.
    """
    grade_col = f"grade_{f_mode}"
    tables = df_items[df_items["item_type"] == "table"].copy()
    if grade_col not in tables.columns:
        grade_col = "grade"
    tables["_success"] = tables[grade_col].astype("object").isin(EFFICIENCY_SUCCESS_GRADES).astype(int)
    per_run = (
        tables.groupby(["paper_slug", "approach"], observed=True)
        .agg(n_tables=("_success", "size"), n_success=("_success", "sum"))
        .reset_index()
    )

    # Select only what we need from df_runs — it already has its own
    # n_tables/n_figures columns which would collide in the merge.
    grade_num_cols = [c for c in df_runs.columns
                      if c == "overall_grade_num" or (c.startswith("overall_grade_") and c.endswith("_num"))]
    run_cols = (["paper_slug", "approach", "model", "total_cost_usd",
                 "prompt_tokens", "completion_tokens", "duration_seconds"] + grade_num_cols)
    df = df_runs[run_cols].merge(per_run, on=["paper_slug", "approach"], how="left")
    df["n_tables"] = df["n_tables"].fillna(0).astype(int)
    df["n_success"] = df["n_success"].fillna(0).astype(int)

    reported = pd.to_numeric(df["total_cost_usd"], errors="coerce")
    prices = df["model"].map(MODEL_PRICING_USD_PER_MTOK)
    prompt_tok = pd.to_numeric(df["prompt_tokens"], errors="coerce").fillna(0)
    compl_tok = pd.to_numeric(df["completion_tokens"], errors="coerce").fillna(0)
    imputed = np.array([
        (p[0] * pt + p[1] * ct) / 1e6 if isinstance(p, tuple) and (pt + ct) > 0 else np.nan
        for p, pt, ct in zip(prices, prompt_tok, compl_tok)
    ])
    df["cost_imputed"] = reported.isna() | (reported <= 0)
    df["cost_usd"] = np.where(df["cost_imputed"], imputed, reported)
    df["duration_min"] = pd.to_numeric(df["duration_seconds"], errors="coerce") / 60
    df["completion_ktok"] = compl_tok.replace(0, np.nan) / 1e3
    return df


def _bootstrap_effort_ci(sub: pd.DataFrame, effort_col: str,
                         n_boot: int = 1000, seed: int = 42) -> tuple:
    """Paper-level bootstrap of (mean effort, success rate) for one approach.

    Returns ((x_lo, x_hi), (y_lo, y_hi)) 95% percentile intervals.
    """
    rng = np.random.default_rng(seed)
    n = len(sub)
    effort = sub[effort_col].values
    succ = sub["n_success"].values
    tabs = sub["n_tables"].values
    xs, ys = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        e = effort[idx]
        e = e[~np.isnan(e)]
        xs.append(e.mean() if len(e) else np.nan)
        denom = tabs[idx].sum()
        ys.append(succ[idx].sum() / denom if denom else np.nan)
    return ((np.nanpercentile(xs, 2.5), np.nanpercentile(xs, 97.5)),
            (np.nanpercentile(ys, 2.5), np.nanpercentile(ys, 97.5)))


def generate_efficiency_table(df_runs: pd.DataFrame, df_items: pd.DataFrame,
                              output_dir: Path, subdir: str = "",
                              f_mode: str = "all_f", n_boot: int = 1000):
    """Per-approach efficiency table: effort per run and cost-of-pass metrics.

    Cost-of-pass = E[cost per run] / E[successful tables per run], i.e. the
    expected spend to obtain one table replicated at grade B or better
    (Erol et al. 2025). Same construction for minutes and completion tokens.
    """
    target = output_dir / subdir if subdir else output_dir
    target.mkdir(parents=True, exist_ok=True)
    df = _effort_frame(df_runs, df_items, f_mode=f_mode)
    if df.empty:
        print("  Skipping efficiency_table: no data")
        return

    grade_num_col = f"overall_grade_{f_mode}_num"
    if grade_num_col not in df.columns:
        grade_num_col = "overall_grade_num"

    rng = np.random.default_rng(42)
    rows = []
    for approach in _approaches_in(df):
        sub = df[df["approach"] == approach]
        if sub.empty:
            continue
        n = len(sub)
        total_success = sub["n_success"].sum()
        total_tables = sub["n_tables"].sum()
        success_rate = total_success / total_tables * 100 if total_tables else np.nan
        any_imputed = sub["cost_imputed"].any()

        def _per_success(col):
            vals = sub[col]
            # Restrict both numerator and denominator to runs where the
            # effort measure is observed, so the ratio stays internally
            # consistent when a run is missing e.g. duration.
            ok = vals.notna()
            n_succ = sub.loc[ok, "n_success"].sum()
            return vals[ok].sum() / n_succ if n_succ else np.nan

        cop = _per_success("cost_usd")
        # Paper-level bootstrap CI for cost-of-pass
        cops = []
        sub_cost = sub[sub["cost_usd"].notna()]
        for _ in range(n_boot):
            idx = rng.integers(0, len(sub_cost), len(sub_cost))
            s = sub_cost.iloc[idx]
            denom = s["n_success"].sum()
            cops.append(s["cost_usd"].sum() / denom if denom else np.nan)
        cop_lo, cop_hi = np.nanpercentile(cops, 2.5), np.nanpercentile(cops, 97.5)

        dagger = r"$^\dagger$" if any_imputed else ""
        rows.append({
            "approach": approach,
            "label": APPROACH_LABELS.get(approach, approach).replace("\n", " "),
            "n_runs": n,
            "success_rate_pct": success_rate,
            "mean_grade": sub[grade_num_col].mean(),
            "median_min_per_run": sub["duration_min"].median(),
            "median_ktok_out_per_run": sub["completion_ktok"].median(),
            "median_cost_per_run": sub["cost_usd"].median(),
            "cost_imputed": any_imputed,
            "n_cost_imputed": int(sub["cost_imputed"].sum()),
            "cost_per_success": cop,
            "cost_per_success_ci_lo": cop_lo,
            "cost_per_success_ci_hi": cop_hi,
            "min_per_success": _per_success("duration_min"),
            "ktok_out_per_success": _per_success("completion_ktok"),
            "_dagger": dagger,
        })

    tab = pd.DataFrame(rows)
    tab.drop(columns=["_dagger"]).to_csv(target / "efficiency_table.csv", index=False)

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Computational effort and efficiency by scaffold/model. Success = table "
        r"replicated at grade B or better (" + f_mode.replace("_", r"\_") + r" grading). "
        r"Cost-of-pass = expected cost per successfully replicated table. "
        r"$^\dagger$Cost imputed from token counts at list prices for one or more runs "
        r"(upper bound: cached-input discounts unavailable). "
        r"95\% CIs from a paper-level bootstrap.}",
        r"\label{tab:efficiency}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r" & & & \multicolumn{3}{c}{Median per run} & \multicolumn{3}{c}{Per successful table} \\",
        r"\cmidrule(lr){4-6} \cmidrule(lr){7-9}",
        r"Scaffold/Model & Tables $\geq$B (\%) & Grade & Min & kTok out & USD & USD [95\% CI] & Min & kTok out \\",
        r"\midrule",
    ]
    for _, r_ in tab.iterrows():
        def _f(v, fmt="{:.1f}"):
            return fmt.format(v) if pd.notna(v) else "—"
        dag = r"$^\dagger$" if r_["cost_imputed"] else ""
        lines.append(
            f"{r_['label']} & {_f(r_['success_rate_pct'])} & {_f(r_['mean_grade'], '{:.2f}')} & "
            f"{_f(r_['median_min_per_run'])} & {_f(r_['median_ktok_out_per_run'], '{:.0f}')} & "
            f"{_f(r_['median_cost_per_run'], '{:.2f}')}{dag} & "
            f"{_f(r_['cost_per_success'], '{:.2f}')}{dag} "
            f"[{_f(r_['cost_per_success_ci_lo'], '{:.2f}')}, {_f(r_['cost_per_success_ci_hi'], '{:.2f}')}] & "
            f"{_f(r_['min_per_success'])} & {_f(r_['ktok_out_per_success'], '{:.0f}')} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (target / "efficiency_table.tex").write_text("\n".join(lines))
    print(f"  Saved {subdir + '/' if subdir else ''}efficiency_table (.csv + .tex)")


def generate_efficiency_regression(df_runs: pd.DataFrame, df_items: pd.DataFrame,
                                   output_dir: Path, subdir: str = "",
                                   f_mode: str = "all_f"):
    """LPM of replication success on scaffold dummies and effort, two panels.

    Panel A (table level): outcome = 100 x 1(table grade >= B); NA counts as
    failure, matching the efficiency table's success-rate definition.
    Panel B (run level): outcome = share of the run's tables >= B (in %).
    Effort telemetry only varies at the run level, so Panel B is the natural
    level for the effort coefficient; Panel A implicitly weights runs by their
    table count. Specs: (1) baseline with paper FE; (2) + log2 tokens, FE;
    (3) + log2 tokens WITHOUT paper FE (absorbs paper difficulty into the
    effort coefficient — the FE/no-FE contrast shows the difficulty confound);
    (4)-(6) linear effort with FE (per +10k completion tokens, +10 minutes,
    +1 USD). SEs clustered by paper. Common sample of runs with complete
    telemetry throughout.

    Descriptive, not causal: effort is chosen by the agent, not assigned.
    """
    import statsmodels.formula.api as smf

    target = output_dir / subdir if subdir else output_dir
    target.mkdir(parents=True, exist_ok=True)

    eff = _effort_frame(df_runs, df_items, f_mode=f_mode)
    grade_col = f"grade_{f_mode}"
    tables = df_items[df_items["item_type"] == "table"].copy()
    if grade_col not in tables.columns:
        grade_col = "grade"
    tables["success"] = tables[grade_col].astype("object").isin(
        EFFICIENCY_SUCCESS_GRADES).astype(float) * 100

    run_cols = ["paper_slug", "approach", "cost_usd", "duration_min", "completion_ktok"]
    dfA = tables.merge(eff[run_cols], on=["paper_slug", "approach"], how="left")
    n_all = len(dfA)
    dfA = dfA[dfA["cost_usd"].notna() & dfA["duration_min"].notna()
              & dfA["completion_ktok"].notna()].copy()
    if len(dfA) < n_all:
        print(f"  efficiency_regression: common sample {len(dfA)}/{n_all} table items "
              f"(dropped runs with incomplete effort telemetry)")
    if dfA.empty:
        print("  Skipping efficiency_regression: no data")
        return

    for d in (dfA,):
        d["approach"] = d["approach"].astype(str)
        d["log2_ktok"] = np.log2(d["completion_ktok"])
        d["ktok10"] = d["completion_ktok"] / 10
        d["min10"] = d["duration_min"] / 10
        d["cost1"] = d["cost_usd"]

    dfB = (dfA.groupby(["paper_slug", "approach"], observed=True)
           .agg(success=("success", "mean"),
                log2_ktok=("log2_ktok", "first"), ktok10=("ktok10", "first"),
                min10=("min10", "first"), cost1=("cost1", "first"))
           .reset_index())

    gnum = f"overall_grade_{f_mode}_num"
    if gnum not in eff.columns:
        gnum = "overall_grade_num"
    dfC = eff[eff["cost_usd"].notna() & eff["duration_min"].notna()
              & eff["completion_ktok"].notna() & eff[gnum].notna()].copy()
    dfC["approach"] = dfC["approach"].astype(str)
    dfC["success"] = dfC[gnum]
    dfC["log2_ktok"] = np.log2(dfC["completion_ktok"])
    dfC["ktok10"] = dfC["completion_ktok"] / 10
    dfC["min10"] = dfC["duration_min"] / 10
    dfC["cost1"] = dfC["cost_usd"]

    ref = "claude-code/claude-opus-4-6"
    scaff = f"C(approach, Treatment('{ref}'))"
    fe = " + C(paper_slug)"
    SPECS = [
        ("(1)", scaff + fe, None, True),
        ("(2)", scaff + fe + " + log2_ktok", "log2_ktok", True),
        ("(3)", scaff + " + log2_ktok", "log2_ktok", False),
        ("(4)", scaff + fe + " + ktok10", "ktok10", True),
        ("(5)", scaff + fe + " + min10", "min10", True),
        ("(6)", scaff + fe + " + cost1", "cost1", True),
    ]
    EFFORT_LABELS = {
        "log2_ktok": r"log$_2$(completion tokens)",
        "ktok10": r"Completion tokens (per +10k)",
        "min10": r"Minutes (per +10)",
        "cost1": r"Cost (per +\$1)",
    }

    def stars(p):
        return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))

    panels, rows_csv = [], []
    for panel_name, d in (("A. Table level, 1(table $\\geq$ B)", dfA),
                          ("B. Run level, share of tables $\\geq$ B", dfB),
                          ("C. Run level, paper grade (0--5)", dfC)):
        fitted = []
        for col_name, formula, effort_var, has_fe in SPECS:
            m = smf.ols("success ~ " + formula, data=d).fit(
                cov_type="cluster", cov_kwds={"groups": d["paper_slug"]})
            fitted.append((col_name, m, has_fe))
            for name, coef, se, p in zip(m.params.index, m.params, m.bse, m.pvalues):
                if name.startswith("C(paper_slug)") or name == "Intercept":
                    continue
                rows_csv.append({"panel": panel_name, "spec": col_name,
                                 "paper_fe": has_fe, "term": name, "coef": coef,
                                 "se": se, "pvalue": p, "n": int(m.nobs),
                                 "r2": m.rsquared})
        panels.append((panel_name, d, fitted))
    pd.DataFrame(rows_csv).to_csv(target / "efficiency_regression.csv", index=False)

    scaffold_terms = [(a, f"C(approach, Treatment('{ref}'))[T.{a}]")
                      for a in APPROACH_MODEL_ORDER if a != ref]

    def cell(m, term):
        if term not in m.params.index:
            return "", ""
        c, s, p = m.params[term], m.bse[term], m.pvalues[term]
        return f"{c:.2f}{stars(p)}", f"({s:.2f})"

    ncols = len(SPECS)
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Replication success and computational effort. Linear probability "
        r"models; outcome in Panel A $= 100 \times \mathbf{1}(\text{table grade} \geq B)$ "
        r"(one observation per table; " + f_mode.replace("_", r"\_") + r" grading, NA "
        r"counts as failure), in Panel B $=$ the share of a run's tables graded $\geq B$, "
        r"in Panel C $=$ the run's overall paper grade on a 0--5 scale (Panels B/C: one "
        r"observation per paper $\times$ scaffold run). Effort telemetry varies at "
        r"the run level only, so Panels B/C are the natural level for the effort "
        r"coefficients. Reference category: Claude Code Opus 4.6. Column (3) omits "
        r"paper fixed effects. Common sample of runs with complete telemetry; Codex CLI "
        r"costs imputed from tokens at list prices. Effort is agent-chosen: "
        r"coefficients are descriptive, not causal. SEs clustered by paper in "
        r"parentheses. $^{*}p<0.1$, $^{**}p<0.05$, $^{***}p<0.01$.}",
        r"\label{tab:efficiency_regression}",
        r"\begin{tabular}{l" + "c" * ncols + "}",
        r"\toprule",
        " & " + " & ".join(cn for cn, _, _, _ in SPECS) + r" \\",
    ]
    for panel_name, d, fitted in panels:
        lines += [
            r"\midrule",
            r"\multicolumn{" + str(ncols + 1) + r"}{l}{\textit{Panel " + panel_name + r"}} \\",
            r"\addlinespace",
        ]
        for a, term in scaffold_terms:
            label = APPROACH_MODEL_LABELS.get(a, a).replace("\n", " ")
            coefs = [cell(m, term) for _, m, _ in fitted]
            lines.append(label + " & " + " & ".join(c for c, _ in coefs) + r" \\")
            lines.append(" & " + " & ".join(s for _, s in coefs) + r" \\")
        lines.append(r"\addlinespace")
        for var, label in EFFORT_LABELS.items():
            coefs = [cell(m, var) for _, m, _ in fitted]
            if all(c == "" for c, _ in coefs):
                continue
            lines.append(label + " & " + " & ".join(c for c, _ in coefs) + r" \\")
            lines.append(" & " + " & ".join(s for _, s in coefs) + r" \\")
        lines.append(r"\addlinespace")
        lines.append(r"Paper FE & " + " & ".join("Yes" if h else "No" for _, _, h in fitted) + r" \\")
        lines.append(r"Observations & " + " & ".join(f"{int(m.nobs):,}" for _, m, _ in fitted) + r" \\")
        lines.append(r"R$^2$ & " + " & ".join(f"{m.rsquared:.3f}" for _, m, _ in fitted) + r" \\")
        lines.append(r"Mean dep.\ var.\ & "
                     + " & ".join([f"{d['success'].mean():.1f}"] * ncols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (target / "efficiency_regression.tex").write_text("\n".join(lines))

    # Standalone table: run-level paper-grade regression only (Panel C).
    _, dC, fittedC = panels[2]
    linesC = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Paper grade and computational effort (run level). OLS; outcome "
        r"$=$ the run's overall paper grade mapped to a 0--5 scale (A=5, \ldots, F=0; "
        + f_mode.replace("_", r"\_") + r" grading), one observation per paper $\times$ "
        r"scaffold run. Reference category: Claude Code Opus 4.6. Effort in column (2) "
        r"is $\log_2$, i.e.\ grade points per doubling; column (3) omits paper fixed "
        r"effects; columns (4)--(6) use linear effort (per $+$10k completion tokens, "
        r"$+$10 minutes, $+\$1$). Common sample of runs with complete telemetry; Codex "
        r"CLI costs imputed from tokens at list prices. Effort is agent-chosen: "
        r"coefficients are descriptive, not causal. SEs clustered by paper in "
        r"parentheses. $^{*}p<0.1$, $^{**}p<0.05$, $^{***}p<0.01$.}",
        r"\label{tab:efficiency_regression_papergrade}",
        r"\begin{tabular}{l" + "c" * ncols + "}",
        r"\toprule",
        " & " + " & ".join(cn for cn, _, _, _ in SPECS) + r" \\",
        r"\midrule",
    ]
    for a, term in scaffold_terms:
        label = APPROACH_MODEL_LABELS.get(a, a).replace("\n", " ")
        coefs = [cell(m, term) for _, m, _ in fittedC]
        linesC.append(label + " & " + " & ".join(c for c, _ in coefs) + r" \\")
        linesC.append(" & " + " & ".join(s for _, s in coefs) + r" \\")
    linesC.append(r"\midrule")
    for var, label in EFFORT_LABELS.items():
        coefs = [cell(m, var) for _, m, _ in fittedC]
        if all(c == "" for c, _ in coefs):
            continue
        linesC.append(label + " & " + " & ".join(c for c, _ in coefs) + r" \\")
        linesC.append(" & " + " & ".join(s for _, s in coefs) + r" \\")
    linesC += [
        r"\midrule",
        r"Paper FE & " + " & ".join("Yes" if h else "No" for _, _, h in fittedC) + r" \\",
        r"Observations & " + " & ".join(f"{int(m.nobs):,}" for _, m, _ in fittedC) + r" \\",
        r"R$^2$ & " + " & ".join(f"{m.rsquared:.3f}" for _, m, _ in fittedC) + r" \\",
        r"Mean dep.\ var.\ & " + " & ".join([f"{dC['success'].mean():.2f}"] * ncols) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (target / "efficiency_regression_papergrade.tex").write_text("\n".join(linesC))

    # Correlational table: outcome ~ ONE effort measure + paper FE, NO scaffold
    # dummies. Answers "do runs that spend more do better on the same paper,
    # pooling across scaffolds" — scaffold quality and scaffold spending policy
    # are deliberately NOT partialled out here.
    OUTCOMES = [
        (r"1(table $\geq$ B)", dfA),
        (r"Share tables $\geq$ B", dfB),
        (r"Paper grade (0--5)", dfC),
    ]
    corr_rows_csv = []
    corr_fits = {}
    for out_label, d in OUTCOMES:
        for var in EFFORT_LABELS:
            m = smf.ols(f"success ~ {var} + C(paper_slug)", data=d).fit(
                cov_type="cluster", cov_kwds={"groups": d["paper_slug"]})
            corr_fits[(out_label, var)] = m
            corr_rows_csv.append({"panel": "correlational_no_scaffold_fe",
                                  "spec": out_label, "paper_fe": True,
                                  "term": var, "coef": m.params[var],
                                  "se": m.bse[var], "pvalue": m.pvalues[var],
                                  "n": int(m.nobs), "r2": m.rsquared})
    pd.concat([pd.DataFrame(rows_csv), pd.DataFrame(corr_rows_csv)]).to_csv(
        target / "efficiency_regression.csv", index=False)

    linesX = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Within-paper correlation of effort and replication success, pooled "
        r"across scaffolds. Each cell is a separate regression of the column outcome on "
        r"the row effort measure and paper fixed effects, WITHOUT scaffold/model "
        r"dummies: the coefficient answers whether runs that spend more do better on "
        r"the same paper, regardless of which scaffold spends it (scaffold identity is "
        r"deliberately not held constant). Outcomes as in the main efficiency "
        r"regression: column (1) table level, columns (2)--(3) run level. Effort "
        r"units: log$_2$ (per doubling), or linear per $+$10k completion tokens, "
        r"$+$10 minutes, $+\$1$. Common telemetry sample; SEs clustered by paper in "
        r"parentheses. $^{*}p<0.1$, $^{**}p<0.05$, $^{***}p<0.01$.}",
        r"\label{tab:efficiency_correlational}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r" & " + " & ".join(f"{ol}" for ol, _ in OUTCOMES) + r" \\",
        r"\midrule",
    ]
    for var, vlabel in EFFORT_LABELS.items():
        cells_c, cells_s = [], []
        for out_label, _ in OUTCOMES:
            m = corr_fits[(out_label, var)]
            c, s = cell(m, var)
            cells_c.append(c)
            cells_s.append(s)
        linesX.append(vlabel + " & " + " & ".join(cells_c) + r" \\")
        linesX.append(" & " + " & ".join(cells_s) + r" \\")
    linesX += [
        r"\midrule",
        r"Paper FE & Yes & Yes & Yes \\",
        r"Scaffold FE & No & No & No \\",
        r"Observations & " + " & ".join(f"{len(d):,}" for _, d in OUTCOMES) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (target / "efficiency_regression_correlational.tex").write_text("\n".join(linesX))

    # Two-column contrast: paper grade on log2 tokens + paper FE,
    # (1) without vs (2) with scaffold/model dummies. The effort coefficient's
    # collapse between the columns is the "effort proxies scaffold quality"
    # result in its most compact form.
    m1 = corr_fits[(r"Paper grade (0--5)", "log2_ktok")]
    m2 = next(m for cn, m, _ in panels[2][2] if cn == "(2)")
    linesT = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Paper grade and completion tokens, within paper (run level). OLS; "
        r"outcome $=$ the run's overall paper grade on a 0--5 scale (A=5, \ldots, F=0; "
        + f_mode.replace("_", r"\_") + r" grading), one observation per paper $\times$ "
        r"scaffold run. Column (1): tokens and paper fixed effects only — the pooled "
        r"within-paper association. Column (2) adds scaffold/model dummies (reference: "
        r"Claude Code Opus 4.6). Tokens in $\log_2$: coefficients read as grade points "
        r"per doubling. Common telemetry sample. Effort is agent-chosen: coefficients "
        r"are descriptive, not causal. SEs clustered by paper in parentheses. "
        r"$^{*}p<0.1$, $^{**}p<0.05$, $^{***}p<0.01$.}",
        r"\label{tab:papergrade_tokens_contrast}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r" & (1) & (2) \\",
        r"\midrule",
    ]
    c1, s1 = cell(m1, "log2_ktok")
    c2, s2 = cell(m2, "log2_ktok")
    linesT.append(r"log$_2$(completion tokens) & " + c1 + " & " + c2 + r" \\")
    linesT.append(" & " + s1 + " & " + s2 + r" \\")
    linesT.append(r"\addlinespace")
    for a, term in scaffold_terms:
        label = APPROACH_MODEL_LABELS.get(a, a).replace("\n", " ")
        c2, s2 = cell(m2, term)
        linesT.append(label + r" &  & " + c2 + r" \\")
        linesT.append(r" &  & " + s2 + r" \\")
    linesT += [
        r"\midrule",
        r"Paper FE & Yes & Yes \\",
        r"Scaffold/model FE & No & Yes \\",
        r"Observations & " + f"{int(m1.nobs):,} & {int(m2.nobs):,}" + r" \\",
        r"R$^2$ & " + f"{m1.rsquared:.3f} & {m2.rsquared:.3f}" + r" \\",
        r"Mean dep.\ var.\ & " + f"{dfC['success'].mean():.2f} & {dfC['success'].mean():.2f}" + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (target / "efficiency_regression_tokens_contrast.tex").write_text("\n".join(linesT))
    print(f"  Saved {subdir + '/' if subdir else ''}efficiency_regression "
          f"(.tex + .csv) + papergrade.tex + correlational.tex + tokens_contrast.tex")
    return panels


def plot_efficiency_frontier(df_runs: pd.DataFrame, df_items: pd.DataFrame,
                             output_dir: Path, subdir: str = "",
                             f_mode: str = "all_f", effort: str = "cost",
                             n_boot: int = 1000):
    """Success rate vs per-run effort, one point per scaffold/model, with the
    Pareto frontier. Points below/right of the frontier are dominated."""
    effort_col, xlabel, fig_name = EFFICIENCY_EFFORT_DIMS[effort]
    df = _effort_frame(df_runs, df_items, f_mode=f_mode)
    if df.empty:
        print(f"  Skipping {fig_name}: no data")
        return

    pts = []
    for approach in _approaches_in(df):
        sub_all = df[df["approach"] == approach]
        sub_obs = sub_all[sub_all[effort_col].notna()]
        if sub_obs.empty or sub_all["n_tables"].sum() == 0:
            print(f"  {fig_name}: dropping {approach} (no {effort_col} data)")
            continue
        if len(sub_obs) < len(sub_all):
            print(f"  {fig_name}: {approach} effort from {len(sub_obs)}/{len(sub_all)} runs "
                  f"(missing {effort_col})")
        # Success rate over ALL runs (identical across effort dimensions and
        # to the efficiency table); mean effort over runs with telemetry.
        x = sub_obs[effort_col].mean()
        y = sub_all["n_success"].sum() / sub_all["n_tables"].sum()
        (x_lo, x_hi), (y_lo, y_hi) = _bootstrap_effort_ci(sub_all, effort_col, n_boot=n_boot)
        pts.append({
            "approach": approach, "x": x, "y": y,
            "x_lo": x_lo, "x_hi": x_hi, "y_lo": y_lo, "y_hi": y_hi,
            "n_runs": len(sub_all), "n_runs_with_effort": len(sub_obs),
            "cost_imputed": bool(sub_obs["cost_imputed"].any()) if effort == "cost" else False,
        })
    if not pts:
        print(f"  Skipping {fig_name}: no approaches with data")
        return

    pdf = pd.DataFrame(pts)
    # Pareto frontier: a point is dominated if some other point has
    # lower-or-equal effort and strictly higher success.
    pdf = pdf.sort_values(["x", "y"], ascending=[True, False]).reset_index(drop=True)
    best_y = -np.inf
    on_frontier = []
    for _, r_ in pdf.iterrows():
        on_frontier.append(r_["y"] > best_y)
        best_y = max(best_y, r_["y"])
    pdf["on_frontier"] = on_frontier

    target = output_dir / subdir if subdir else output_dir
    target.mkdir(parents=True, exist_ok=True)
    pdf.to_csv(target / f"{fig_name}.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.5, 6))
    front = pdf[pdf["on_frontier"]]
    ax.step(front["x"], front["y"] * 100, where="post", color="#95a5a6",
            linestyle="--", linewidth=1.8, zorder=1, label="Pareto frontier")

    # Label placement: above-right by default; when two points are close in
    # (log-x, y), the lower/leftmost one's label flips to below-left.
    log_x = np.log10(pdf["x"])
    label_pos = {}
    for i in pdf.index:
        dx, dy, ha = 8, 6, "left"
        for j in pdf.index:
            if j == i:
                continue
            close = (abs(log_x[i] - log_x[j]) < 0.15
                     and abs(pdf.at[i, "y"] - pdf.at[j, "y"]) * 100 < 6)
            if close and (pdf.at[i, "y"], pdf.at[i, "x"]) < (pdf.at[j, "y"], pdf.at[j, "x"]):
                dx, dy, ha = -10, -20, "right"
        label_pos[i] = (dx, dy, ha)

    for i, r_ in pdf.iterrows():
        color = APPROACH_COLORS.get(r_["approach"], "#95a5a6")
        label = APPROACH_LABELS.get(r_["approach"], r_["approach"]).replace("\n", " ")
        if r_["cost_imputed"]:
            label += " †"
        ax.errorbar(r_["x"], r_["y"] * 100,
                    xerr=[[r_["x"] - r_["x_lo"]], [r_["x_hi"] - r_["x"]]],
                    yerr=[[(r_["y"] - r_["y_lo"]) * 100], [(r_["y_hi"] - r_["y"]) * 100]],
                    fmt="o", markersize=11, color=color, ecolor=color,
                    elinewidth=1.3, capsize=3, alpha=0.9, zorder=5)
        dx, dy, ha = label_pos[i]
        ax.annotate(label, (r_["x"], r_["y"] * 100),
                    xytext=(dx, dy), textcoords="offset points", ha=ha,
                    fontsize=10, fontweight="bold", color=color)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=13, fontweight="bold")
    ax.set_ylabel("Tables replicated at grade $\\geq$ B (%)", fontsize=13, fontweight="bold")
    if pdf["cost_imputed"].any():
        ax.text(0.02, 0.02, "† ≥1 run's cost imputed from tokens (upper bound)",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=9,
                color="#7f8c8d")
    ax.legend(fontsize=10, loc="lower right")
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig, output_dir, fig_name, subdir)


# ============================================================================
# Section: Item Level — Tables
# ============================================================================

def plot_item_grade_by_type(df_items: pd.DataFrame, output_dir: Path, item_type: str | None, name: str,
                            subdir: str = "", f_mode: str = "all_f",
                            grade_col: str = "grade", name_suffix: str = ""):
    if grade_col not in df_items.columns:
        return
    df = df_items if item_type is None else df_items[df_items["item_type"] == item_type]
    df = df[df[grade_col] != "NA"]
    grades_shown = [g for g in GRADE_ORDER if g != "NA"]
    if f_mode == "no_f":
        grades_shown = [g for g in grades_shown if g != "F"]
    if df.empty:
        return

    ct = pd.crosstab(df["approach"], df[grade_col], normalize="index") * 100
    ct = ct.reindex(columns=grades_shown, fill_value=0)

    fig, ax = plt.subplots(figsize=(7, 5))
    present = _approaches_in(df)
    ct.loc[[a for a in present if a in ct.index]].plot(kind="bar", ax=ax, color=[GRADE_COLORS[g] for g in ct.columns],
                                edgecolor="white", width=0.8)
    ax.set_xticklabels([APPROACH_LABELS.get(a, a).replace("\n", " ") for a in present if a in ct.index], fontsize=10, rotation=25, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("Share of Items (%)", fontsize=18, fontweight="bold")
    place_legend(fig, ax, fontsize=14, ncol=6)
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig, output_dir, f"{name}{F_MODE_SUFFIX[f_mode]}{name_suffix}", subdir)


def plot_grade_distribution_by_table_type(df_items: pd.DataFrame, output_dir: Path, subdir: str = "",
                                           f_mode: str = "all_f"):
    """Grade distribution (stacked bar per approach), one panel per table category.

    Requires `table_category` column on df_items.
    """
    df = df_items[
        (df_items["item_type"] == "table") &
        df_items["table_category"].notna() &
        (df_items["grade"] != "NA")
    ].copy()
    df = _filter_f_mode(df, f_mode, level="item")
    if df.empty:
        print("  Skipping grade_distribution_by_table_type: no categorized tables")
        return

    category_order = ["main_results", "mechanism", "robustness", "descriptive", "other"]
    category_labels = {
        "main_results": "Main Results",
        "mechanism": "Mechanism",
        "robustness": "Robustness",
        "descriptive": "Descriptive",
        "other": "Other",
    }
    cats_present = [c for c in category_order if c in df["table_category"].unique()]
    grades_shown = [g for g in GRADE_ORDER if g != "NA"]
    if f_mode == "no_f":
        grades_shown = [g for g in grades_shown if g != "F"]
    approaches = _approaches_in(df)

    n_cats = len(cats_present)
    fig, axes = plt.subplots(1, n_cats, figsize=(4.5 * n_cats, 5.5), sharey=True)
    if n_cats == 1:
        axes = [axes]

    for ax_idx, cat in enumerate(cats_present):
        ax = axes[ax_idx]
        sub = df[df["table_category"] == cat]
        n_tables = len(sub)
        ct = pd.crosstab(sub["approach"], sub["grade"], normalize="index") * 100
        ct = ct.reindex(columns=grades_shown, fill_value=0)
        present = [a for a in approaches if a in ct.index]
        if not present:
            continue
        ct.loc[present].plot(
            kind="bar", ax=ax,
            color=[GRADE_COLORS.get(g, "#95a5a6") for g in ct.columns],
            edgecolor="white", width=0.8, legend=False,
        )
        ax.set_xticklabels(
            [APPROACH_LABELS.get(a, a).replace("\n", " ") for a in present],
            fontsize=8, rotation=35, ha="right",
        )
        ax.set_xlabel("")
        n_per_approach = sub.groupby("approach").size().reindex(present).median()
        apply_style(ax)

    axes[0].set_ylabel("Share of Tables (%)", fontsize=13, fontweight="bold")
    place_legend(fig, axes[0], fontsize=10, ncol=len(grades_shown))
    plt.tight_layout()
    save_figure(fig, output_dir, f"grade_distribution_by_table_type{F_MODE_SUFFIX[f_mode]}", subdir)


def plot_grade_cumulative_by_table_type(df_items: pd.DataFrame, output_dir: Path, subdir: str = "",
                                         f_mode: str = "all_f"):
    """Cumulative grade dot-plots (≥A, ≥B, … ≥F), one panel per table category.

    Requires `table_category` column on df_items.
    """
    df = df_items[
        (df_items["item_type"] == "table") &
        df_items["table_category"].notna() &
        (df_items["grade"] != "NA")
    ].copy()
    df = _filter_f_mode(df, f_mode, level="item")
    if df.empty:
        print("  Skipping grade_cumulative_by_table_type: no categorized tables")
        return

    category_order = ["main_results", "mechanism", "robustness", "descriptive", "other"]
    category_labels = {
        "main_results": "Main Results",
        "mechanism": "Mechanism",
        "robustness": "Robustness",
        "descriptive": "Descriptive",
        "other": "Other",
    }
    cats_present = [c for c in category_order if c in df["table_category"].unique()]
    grades_cum = ["A", "B", "C", "D", "E"] if f_mode == "no_f" else ["A", "B", "C", "D", "E", "F"]
    approaches = _approaches_in(df)

    n_cats = len(cats_present)
    fig, axes = plt.subplots(1, n_cats, figsize=(4.5 * n_cats, 5.5), sharey=True)
    if n_cats == 1:
        axes = [axes]

    for ax_idx, cat in enumerate(cats_present):
        ax = axes[ax_idx]
        sub = df[df["table_category"] == cat]
        if sub.empty:
            continue

        ct = pd.crosstab(sub["approach"], sub["grade"], normalize="index") * 100
        ct = ct.reindex(columns=grades_cum, fill_value=0)

        cum = pd.DataFrame(index=ct.index, columns=[f"≥{g}" for g in grades_cum])
        for i, g in enumerate(grades_cum):
            cum[f"≥{g}"] = ct[grades_cum[:i + 1]].sum(axis=1)

        present = [a for a in approaches if a in cum.index]
        sort_key = cum.loc[present, "≥B"].values
        order = np.argsort(-sort_key)
        present = [present[i] for i in order]

        y_pos = np.arange(len(present))
        for g_idx, g in enumerate(grades_cum):
            col = f"≥{g}"
            vals = [cum.loc[a, col] for a in present]
            color = GRADE_COLORS.get(g, "#95a5a6")
            for i, (a, v) in enumerate(zip(present, vals)):
                ax.text(v, i, g, fontsize=11, fontweight="bold", color=color,
                        ha="center", va="center", zorder=5)
                if g_idx > 0:
                    prev_col = f"≥{grades_cum[g_idx - 1]}"
                    prev_v = cum.loc[a, prev_col]
                    ax.plot([prev_v, v], [i, i], color="#cccccc", linewidth=1, zorder=1)

        ax.set_yticks(y_pos)
        if ax_idx == 0:
            labels = [APPROACH_LABELS.get(a, a).replace("\n", " ") for a in present]
            ax.set_yticklabels(labels, fontsize=9)
            for tick_label, a in zip(ax.get_yticklabels(), present):
                tick_label.set_color(APPROACH_COLORS.get(a, "#95a5a6"))
                tick_label.set_fontweight("bold")
        else:
            ax.set_yticklabels([])
        ax.invert_yaxis()
        ax.set_xlim(0, 105)
        n_tables = len(sub)
        apply_style(ax)

    axes[0].set_xlabel("Cumulative share of tables (%)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, output_dir, f"grade_cumulative_by_table_type{F_MODE_SUFFIX[f_mode]}", subdir)


def plot_item_grade_cumulative(df_items: pd.DataFrame, output_dir: Path, item_type: str | None, name: str,
                                subdir: str = "", f_mode: str = "all_f",
                                grade_col: str = "grade", name_suffix: str = ""):
    """Dot plot of cumulative grade shares: ≥A, ≥B, ≥C, ≥D, ≥E per approach."""
    if grade_col not in df_items.columns:
        return
    df = df_items if item_type is None else df_items[df_items["item_type"] == item_type]
    df = df[df[grade_col] != "NA"]
    if df.empty:
        return

    grades_cum = ["A", "B", "C", "D", "E"] if f_mode == "no_f" else ["A", "B", "C", "D", "E", "F"]

    approaches = _approaches_in(df)
    ct = pd.crosstab(df["approach"], df[grade_col], normalize="index") * 100
    ct = ct.reindex(columns=grades_cum, fill_value=0)

    cum = pd.DataFrame(index=ct.index, columns=[f"≥{g}" for g in grades_cum])
    for i, g in enumerate(grades_cum):
        cum[f"≥{g}"] = ct[grades_cum[:i + 1]].sum(axis=1)

    present = [a for a in approaches if a in cum.index]
    sort_key = cum.loc[present, "≥B"].values
    order = np.argsort(-sort_key)
    present = [present[i] for i in order]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    y_pos = np.arange(len(present))

    for g_idx, g in enumerate(grades_cum):
        col = f"≥{g}"
        vals = [cum.loc[a, col] for a in present]
        color = GRADE_COLORS.get(g, "#95a5a6")
        for i, (a, v) in enumerate(zip(present, vals)):
            ax.text(v, i, g, fontsize=14, fontweight="bold", color=color,
                    ha="center", va="center", zorder=5)
            if g_idx > 0:
                prev_col = f"≥{grades_cum[g_idx - 1]}"
                prev_v = cum.loc[a, prev_col]
                ax.plot([prev_v, v], [i, i], color="#cccccc", linewidth=1, zorder=1)
    ax.set_yticks(y_pos)
    labels = [APPROACH_LABELS.get(a, a).replace("\n", " ") for a in present]
    colors = [APPROACH_COLORS.get(a, "#95a5a6") for a in present]
    ax.set_yticklabels(labels, fontsize=10)
    for tick_label, color in zip(ax.get_yticklabels(), colors):
        tick_label.set_color(color)
        tick_label.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("Cumulative share of items (%)", fontsize=12, fontweight="bold")
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig, output_dir, f"{name}{F_MODE_SUFFIX[f_mode]}{name_suffix}", subdir)


def plot_item_number_vs_grade(df_items: pd.DataFrame, output_dir: Path, subdir: str = "",
                                f_mode: str = "all_f"):
    df = df_items[df_items["item_number"].notna() & (df_items["item_type"] == "table") & (df_items["grade"] != "NA")].copy()
    df = _filter_f_mode(df, f_mode, level="item")
    df["item_number"] = df["item_number"].astype(int)
    df = df[df["item_number"] <= 8]
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
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
    numeric_grades = [g for g in GRADE_ORDER if g != "NA"]
    ax.set_yticks(range(len(numeric_grades)))
    ax.set_yticklabels(numeric_grades[::-1])
    place_legend(fig, ax, fontsize=12)
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig, output_dir, f"item_number_vs_grade{F_MODE_SUFFIX[f_mode]}", subdir)


# ============================================================================
# Section: Cell Level
# ============================================================================


def _trim_pct(series: pd.Series, trim: float = 0.025) -> pd.Series:
    """Remove top and bottom trim% of values, dropping NaN and inf."""
    s = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 10:
        return s
    hi = s.quantile(1 - trim)
    lo = s.quantile(trim)
    return s[(s >= lo) & (s <= hi)]


PCT_DIFF_CAP = 300  # Cap for mean/histogram plots


def plot_pct_diff_by_cell_type_mean(df_cells: pd.DataFrame, output_dir: Path, subdir: str = "",
                                     pct_col: str = "percent_difference", name_suffix: str = "",
                                     f_mode: str = "all_f"):
    """Mean |% difference| by cell type and approach, with SD whiskers (capped at PCT_DIFF_CAP)."""
    CELL_TYPES = [
        ("coefficient", "Coefficients"),
        ("se", "Standard Errors"),
        ("statistic_r2", "R²"),
        ("p_value", "P-values"),
    ]
    if pct_col not in df_cells.columns:
        return
    cell_col = "cell_grade_rounded" if pct_col.endswith("_rounded") else "cell_grade"
    base = _cells_for_mode(df_cells, f_mode, cell_col=cell_col)
    df = base[
        base[pct_col].notna() &
        base["is_numeric"] &
        base["row_type"].isin([ct for ct, _ in CELL_TYPES])
    ].copy()
    if df.empty:
        return

    df["pct_abs"] = df[pct_col].abs().replace([np.inf], np.nan)
    df_unfiltered = df.copy()  # keep for <2% and >300% stats
    df = df[df["pct_abs"] <= PCT_DIFF_CAP].copy()

    approaches = _approaches_in(df)
    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    axes = axes.flatten()

    for idx, (rt, rt_label) in enumerate(CELL_TYPES):
        ax = axes[idx]
        ax.set_title(rt_label, fontsize=12, fontweight="bold")
        sub = df[df["row_type"] == rt]

        sub_unf = df_unfiltered[df_unfiltered["row_type"] == rt]
        means, ci95s, labels, colors, pct_lt2, pct_gt300 = [], [], [], [], [], []
        for a in approaches:
            vals = sub.loc[sub["approach"] == a, "pct_abs"].dropna()
            vals_unf = sub_unf.loc[sub_unf["approach"] == a, "pct_abs"].dropna()
            if len(vals) < 5:
                continue
            means.append(vals.mean())
            ci95s.append(1.96 * vals.std() / np.sqrt(len(vals)))
            labels.append(APPROACH_LABELS.get(a, a).replace("\n", " "))
            colors.append(APPROACH_COLORS.get(a, "#95a5a6"))
            pct_lt2.append((vals_unf < 2).mean() * 100)
            pct_gt300.append((vals_unf > 300).mean() * 100)

        if not means:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        # Sort by mean (best/lowest on top)
        order = np.argsort(means)
        means = [means[i] for i in order]
        ci95s = [ci95s[i] for i in order]
        labels = [labels[i] for i in order]
        colors = [colors[i] for i in order]
        pct_lt2 = [pct_lt2[i] for i in order]
        pct_gt300 = [pct_gt300[i] for i in order]

        y = np.arange(len(means))
        xerr_low = [min(m, ci) for m, ci in zip(means, ci95s)]
        xerr_high = ci95s
        ax.errorbar(means, y, xerr=[xerr_low, xerr_high], fmt="o", markersize=8,
                     capsize=5, capthick=1.5, linewidth=1.5,
                     color="black", ecolor="black")
        for i, (m, c) in enumerate(zip(means, colors)):
            ax.plot(m, i, "o", markersize=10, color=c, zorder=5)

        # Add stat columns
        for i, (lt2, gt300) in enumerate(zip(pct_lt2, pct_gt300)):
            ax.text(1.02, y[i], f"{lt2:.0f}%", transform=ax.get_yaxis_transform(),
                    fontsize=9, va="center", ha="left", color="#2ecc71", fontweight="bold")
            ax.text(1.17, y[i], f"{gt300:.0f}%", transform=ax.get_yaxis_transform(),
                    fontsize=9, va="center", ha="left", color="#e74c3c", fontweight="bold")
        if idx < 2:
            ax.text(1.02, -0.8, "<2%", transform=ax.get_yaxis_transform(),
                    fontsize=9, va="center", ha="left", color="#2ecc71", fontweight="bold")
            ax.text(1.17, -0.8, ">300%", transform=ax.get_yaxis_transform(),
                    fontsize=9, va="center", ha="left", color="#e74c3c", fontweight="bold")

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=10)
        for tick_label, color in zip(ax.get_yticklabels(), colors):
            tick_label.set_color(color)
            tick_label.set_fontweight("bold")
        ax.set_xlim(0, 80)
        ax.invert_yaxis()
        if idx >= 2:
            ax.set_xlabel("Mean |% difference|", fontsize=11)
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])
        apply_style(ax)

    plt.tight_layout()
    fig.subplots_adjust(right=0.82)
    save_figure(fig, output_dir, f"pct_diff_by_cell_type_mean_{f_mode}{name_suffix}", subdir)


def plot_pct_diff_exceedance(df_cells: pd.DataFrame, output_dir: Path, subdir: str = "",
                              pct_col: str = "percent_difference", name_suffix: str = "",
                              f_mode: str = "all_f"):
    """Share of cells exceeding the cap, by cell type and approach."""
    CELL_TYPES = [
        ("coefficient", "Coefficients"),
        ("se", "Standard Errors"),
        ("statistic_r2", "R²"),
        ("p_value", "P-values"),
    ]
    if pct_col not in df_cells.columns:
        return
    cell_col = "cell_grade_rounded" if pct_col.endswith("_rounded") else "cell_grade"
    base = _cells_for_mode(df_cells, f_mode, cell_col=cell_col)
    df = base[
        base[pct_col].notna() &
        base["is_numeric"] &
        base["row_type"].isin([ct for ct, _ in CELL_TYPES])
    ].copy()
    if df.empty:
        return

    df["pct_abs"] = df[pct_col].abs().replace([np.inf], np.nan)

    approaches = _approaches_in(df)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (rt, rt_label) in enumerate(CELL_TYPES):
        ax = axes[idx]
        ax.set_title(rt_label, fontsize=12, fontweight="bold")
        sub = df[df["row_type"] == rt]

        rates, labels, colors = [], [], []
        for a in approaches:
            vals = sub.loc[sub["approach"] == a, "pct_abs"].dropna()
            if len(vals) < 5:
                continue
            exceed = (vals > PCT_DIFF_CAP).mean() * 100
            rates.append(exceed)
            labels.append(APPROACH_LABELS.get(a, a).replace("\n", " "))
            colors.append(APPROACH_COLORS.get(a, "#95a5a6"))

        if not rates:
            continue

        x = np.arange(len(rates))
        ax.bar(x, rates, color=colors, edgecolor="white", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
        if idx % 2 == 0:
            ax.set_ylabel(f"% of cells > {PCT_DIFF_CAP}% difference", fontsize=11)
        for i, rate in enumerate(rates):
            ax.text(i, rate + 0.3, f"{rate:.1f}%", ha="center", fontsize=9, fontweight="bold")
        apply_style(ax)

    plt.tight_layout()
    save_figure(fig, output_dir, f"pct_diff_exceedance_{f_mode}{name_suffix}", subdir)


def plot_pct_diff_histograms(df_cells: pd.DataFrame, output_dir: Path, subdir: str = "",
                              pct_col: str = "percent_difference", name_suffix: str = "",
                              f_mode: str = "all_f"):
    """Overlaid histograms of |% difference| for 3 approaches, faceted by cell type."""
    CELL_TYPES = [
        ("coefficient", "Coefficients"),
        ("se", "Standard Errors"),
        ("statistic_r2", "R²"),
        ("p_value", "P-values"),
    ]
    SELECTED = [
        ("claude-code/claude-opus-4-6", "Claude Code Opus 4.6", "#E07B39"),
        ("codex/gpt-5.4", "Codex CLI GPT-5.4", "#10A37F"),
        ("opencode/z-ai_glm-5", "OpenCode GLM-5", "#0984E3"),
    ]
    if pct_col not in df_cells.columns:
        return
    cell_col = "cell_grade_rounded" if pct_col.endswith("_rounded") else "cell_grade"
    base = _cells_for_mode(df_cells, f_mode, cell_col=cell_col)
    df = base[
        base[pct_col].notna() &
        base["is_numeric"] &
        base["row_type"].isin([ct for ct, _ in CELL_TYPES]) &
        base["approach"].isin([a for a, _, _ in SELECTED])
    ].copy()
    if df.empty:
        return

    df["pct_abs"] = df[pct_col].abs().replace([np.inf], np.nan)
    df = df[df["pct_abs"] <= PCT_DIFF_CAP].copy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (rt, rt_label) in enumerate(CELL_TYPES):
        ax = axes[idx]
        ax.set_title(rt_label, fontsize=12, fontweight="bold")
        sub = df[df["row_type"] == rt]
        for a, label, color in SELECTED:
            vals = sub.loc[sub["approach"] == a, "pct_abs"].dropna()
            if len(vals) < 5:
                continue
            ax.hist(vals, bins=40, range=(0, PCT_DIFF_CAP), alpha=0.4,
                    color=color, label=label, density=True, edgecolor="none")
        ax.set_xlabel("|% difference|", fontsize=11)
        if idx % 2 == 0:
            ax.set_ylabel("Density", fontsize=11)
        if idx == 0:
            place_legend(fig, ax, fontsize=9)
        apply_style(ax)

    plt.tight_layout()
    save_figure(fig, output_dir, f"pct_diff_histograms_{f_mode}{name_suffix}", subdir)

    # Log-scale version
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
    axes2 = axes2.flatten()
    for idx, (rt, rt_label) in enumerate(CELL_TYPES):
        ax = axes2[idx]
        ax.set_title(rt_label, fontsize=12, fontweight="bold")
        sub = df[df["row_type"] == rt]
        for a, label, color in SELECTED:
            vals = sub.loc[sub["approach"] == a, "pct_abs"].dropna()
            if len(vals) < 5:
                continue
            ax.hist(vals, bins=40, range=(0, PCT_DIFF_CAP), alpha=0.4,
                    color=color, label=label, density=True, edgecolor="none")
        ax.set_yscale("log")
        ax.set_xlabel("|% difference|", fontsize=11)
        if idx % 2 == 0:
            ax.set_ylabel("Density (log scale)", fontsize=11)
        if idx == 0:
            place_legend(fig2, ax, fontsize=9)
        apply_style(ax)
    plt.tight_layout()
    save_figure(fig2, output_dir, f"pct_diff_histograms_log_{f_mode}{name_suffix}", subdir)


def plot_value_distributions(df_cells: pd.DataFrame, output_dir: Path, subdir: str = ""):
    """Histograms of original vs replicated *values* by cell type.

    Four panels: coefficients, standard errors, R-squared, and all other numeric
    (pooling all row_types not in the first three). Coefficients, SEs, and other
    numeric use log-scaled x-axes (on absolute values) since their distributions
    span orders of magnitude. R-squared is plotted on a linear [0, 1] scale.

    Two output figures:
      - value_distributions.pdf: 2×2 panels, original (blue) vs replicated (orange),
        pooled across all approaches.
      - value_distribution_{type}.pdf: per-type figures with per-approach step lines.
    """
    COEF_TYPES = {"coefficient"}
    SE_TYPES = {"se"}
    R2_TYPES = {"statistic_r2"}
    # Everything else that's numeric and not panel_header/string
    EXCLUDE_TYPES = {"string", "panel_header", ""}

    df = df_cells[
        df_cells["is_numeric"] &
        # OLD: excluded F
        # (df_cells["item_grade"] != "F")
        # NEW: exclude NA instead
        (df_cells["item_grade"] != "NA")
    ].copy()
    if df.empty:
        print("  Skipping value_distributions: empty data")
        return

    df["orig_num"] = pd.to_numeric(df["original_value"], errors="coerce")
    df["repl_num"] = pd.to_numeric(df["replicated_value"], errors="coerce")
    df["rt"] = df["row_type"].fillna("")

    # Assign plot category
    def _cat(rt):
        if rt in COEF_TYPES:
            return "coefficient"
        if rt in SE_TYPES:
            return "se"
        if rt in R2_TYPES:
            return "r2"
        if rt in EXCLUDE_TYPES:
            return None
        return "other_numeric"
    df["plot_cat"] = df["rt"].map(_cat)
    df = df[df["plot_cat"].notna()].copy()

    # (key, label, scale): "log" = log on |value|, "asinh" = inverse hyperbolic sine, "linear" = linear
    PANELS = [
        ("coefficient", "Coefficients", "asinh"),
        ("se", "Standard Errors", "log"),
        ("r2", "R-squared", "linear"),
        ("other_numeric", "Other Numeric", "asinh"),
    ]

    # ── 2×2 pooled figure ─────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    def _asinh_bins(vals, n=60):
        """Create evenly-spaced bins in asinh-transformed space, return edges in original scale."""
        lo, hi = vals.quantile(0.01), vals.quantile(0.99)
        t_lo, t_hi = np.arcsinh(lo), np.arcsinh(hi)
        t_edges = np.linspace(t_lo, t_hi, n + 1)
        return np.sinh(t_edges), lo, hi

    def _asinh_ticks(ax, lo, hi):
        """Set tick labels at nice values in original scale on an asinh-transformed axis."""
        from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
        # Pick ticks at 0 and powers of 10 in both directions
        candidates = [0]
        for exp in range(6):
            candidates.append(10 ** exp)
            candidates.append(-(10 ** exp))
        # Keep only ticks within data range (with some margin)
        margin = 0.5  # in asinh space
        ticks = [t for t in candidates
                 if np.arcsinh(lo) - margin <= np.arcsinh(t) <= np.arcsinh(hi) + margin]
        ticks = sorted(set(ticks))
        # Thin out if too many
        if len(ticks) > 9:
            # Keep 0, then every other tick on each side
            neg = sorted([t for t in ticks if t < 0])
            pos = sorted([t for t in ticks if t > 0])
            ticks = neg[::2] + [0] + pos[::2]
        ax.xaxis.set_major_locator(FixedLocator([np.arcsinh(t) for t in ticks]))
        ax.xaxis.set_minor_locator(NullLocator())
        def _fmt(x, _):
            v = np.sinh(x)
            # Find closest candidate
            closest = min(ticks, key=lambda t: abs(np.arcsinh(t) - x))
            v = closest
            if v == 0:
                return "0"
            if v == int(v) and abs(v) >= 1:
                return f"{int(v):,}"
            return f"{v:g}"
        ax.xaxis.set_major_formatter(FuncFormatter(_fmt))
        ax.tick_params(axis="x", which="both", rotation=45, labelsize=8)
        for label in ax.get_xticklabels():
            label.set_ha("right")

    for idx, (cat, cat_label, scale) in enumerate(PANELS):
        ax = axes[idx]
        sub = df[df["plot_cat"] == cat]
        orig_vals = sub["orig_num"].dropna()
        repl_vals = sub["repl_num"].dropna()
        if len(orig_vals) < 5:
            continue

        if scale == "log":
            orig_plot = orig_vals.abs().replace(0, np.nan).dropna()
            repl_plot = repl_vals.abs().replace(0, np.nan).dropna()
            if orig_plot.empty:
                continue
            all_abs = pd.concat([orig_plot, repl_plot.dropna()])
            lo = max(all_abs.quantile(0.01), all_abs[all_abs > 0].min())
            hi = all_abs.quantile(0.99)
            log_bins = np.logspace(np.log10(lo), np.log10(hi), 50)
            ax.hist(orig_plot.clip(lo, hi), bins=log_bins, alpha=0.5,
                    color="#3498db", label="Original", density=True, edgecolor="none")
            if len(repl_plot) >= 5:
                ax.hist(repl_plot.clip(lo, hi), bins=log_bins, alpha=0.5,
                        color="#e67e22", label="Replicated", density=True, edgecolor="none")
            ax.set_xscale("log")
            ax.set_xlabel("|Value| (log scale)", fontsize=11)

        elif scale == "asinh":
            all_vals = pd.concat([orig_vals, repl_vals.dropna()])
            bins_orig, lo, hi = _asinh_bins(all_vals, n=60)
            # Transform data to asinh space, bin in that space
            t_lo, t_hi = np.arcsinh(lo), np.arcsinh(hi)
            t_bins = np.linspace(t_lo, t_hi, 61)
            ax.hist(np.arcsinh(orig_vals.clip(lo, hi)), bins=t_bins, alpha=0.5,
                    color="#3498db", label="Original", density=True, edgecolor="none")
            if len(repl_vals) >= 5:
                ax.hist(np.arcsinh(repl_vals.clip(lo, hi)), bins=t_bins, alpha=0.5,
                        color="#e67e22", label="Replicated", density=True, edgecolor="none")
            _asinh_ticks(ax, lo, hi)
            ax.set_xlabel("Value (asinh scale)", fontsize=11)

        else:  # linear
            lo, hi = 0.0, 1.0
            bins = 50
            ax.hist(orig_vals.clip(lo, hi), bins=bins, range=(lo, hi), alpha=0.5,
                    color="#3498db", label="Original", density=True, edgecolor="none")
            if len(repl_vals) >= 5:
                ax.hist(repl_vals.clip(lo, hi), bins=bins, range=(lo, hi), alpha=0.5,
                        color="#e67e22", label="Replicated", density=True, edgecolor="none")
            ax.set_xlabel("Value", fontsize=11)

        if idx % 2 == 0:
            ax.set_ylabel("Density", fontsize=11)
        if idx == 0:
            ax.legend(fontsize=10)
        apply_style(ax)

    plt.tight_layout()
    save_figure(fig, output_dir, "value_distributions", subdir)

    # ── Per-type per-approach figures ──────────────────────────────────────
    for cat, cat_label, scale in PANELS:
        sub = df[df["plot_cat"] == cat]
        orig_vals = sub.drop_duplicates(
            subset=["paper_slug", "item_id", "row_label", "column_label"],
            keep="first",
        )["orig_num"].dropna()
        if len(orig_vals) < 5:
            continue

        fig2, ax2 = plt.subplots(figsize=(8, 5))

        if scale == "log":
            orig_plot = orig_vals.abs().replace(0, np.nan).dropna()
            all_abs = sub[["orig_num", "repl_num"]].stack().abs().replace(0, np.nan).dropna()
            lo = max(all_abs.quantile(0.01), all_abs[all_abs > 0].min())
            hi = all_abs.quantile(0.99)
            log_bins = np.logspace(np.log10(lo), np.log10(hi), 50)
            ax2.hist(orig_plot.clip(lo, hi), bins=log_bins, alpha=0.3,
                     color="#333333", label="Original", density=True, edgecolor="none")
            for approach in _approaches_in(sub):
                vals = sub.loc[sub["approach"] == approach, "repl_num"].dropna().abs().replace(0, np.nan).dropna()
                if len(vals) < 5:
                    continue
                color = APPROACH_COLORS.get(approach, "#95a5a6")
                label = APPROACH_LABELS.get(approach, approach).replace("\n", " ")
                ax2.hist(vals.clip(lo, hi), bins=log_bins, histtype="step",
                         color=color, label=label, density=True, linewidth=1.5)
            ax2.set_xscale("log")
            ax2.set_xlabel("|Value| (log scale)", fontsize=12)

        elif scale == "asinh":
            all_vals = pd.concat([orig_vals, sub["repl_num"].dropna()])
            _, lo, hi = _asinh_bins(all_vals, n=60)
            t_lo, t_hi = np.arcsinh(lo), np.arcsinh(hi)
            t_bins = np.linspace(t_lo, t_hi, 61)
            ax2.hist(np.arcsinh(orig_vals.clip(lo, hi)), bins=t_bins, alpha=0.3,
                     color="#333333", label="Original", density=True, edgecolor="none")
            for approach in _approaches_in(sub):
                vals = sub.loc[sub["approach"] == approach, "repl_num"].dropna()
                if len(vals) < 5:
                    continue
                color = APPROACH_COLORS.get(approach, "#95a5a6")
                label = APPROACH_LABELS.get(approach, approach).replace("\n", " ")
                ax2.hist(np.arcsinh(vals.clip(lo, hi)), bins=t_bins, histtype="step",
                         color=color, label=label, density=True, linewidth=1.5)
            _asinh_ticks(ax2, lo, hi)
            ax2.set_xlabel("Value (asinh scale)", fontsize=12)

        else:  # linear
            lo, hi = 0.0, 1.0
            bins = 50
            ax2.hist(orig_vals.clip(lo, hi), bins=bins, range=(lo, hi), alpha=0.3,
                     color="#333333", label="Original", density=True, edgecolor="none")
            for approach in _approaches_in(sub):
                vals = sub.loc[sub["approach"] == approach, "repl_num"].dropna()
                if len(vals) < 5:
                    continue
                color = APPROACH_COLORS.get(approach, "#95a5a6")
                label = APPROACH_LABELS.get(approach, approach).replace("\n", " ")
                ax2.hist(vals.clip(lo, hi), bins=bins, range=(lo, hi), histtype="step",
                         color=color, label=label, density=True, linewidth=1.5)
            ax2.set_xlabel("Value", fontsize=12)

        ax2.set_ylabel("Density", fontsize=12)
        place_legend(fig2, ax2, fontsize=9, ncol=3)
        apply_style(ax2)
        plt.tight_layout()
        save_figure(fig2, output_dir, f"value_distribution_{cat}", subdir)


def _normalize_item_id(x: str) -> str:
    """Normalize table IDs so 'Table 3', 'Table 3—Caption', 'Table 3\\nfoo' all match."""
    if not isinstance(x, str):
        return ""
    import re as _re
    # Strip whitespace/newlines, collapse inner whitespace
    s = _re.sub(r"\s+", " ", x).strip()
    # Keep only "Table <num>[letter]" prefix if present
    m = _re.match(r"^(Table\s+\w+[a-zA-Z]?)", s, flags=_re.IGNORECASE)
    return m.group(1).lower() if m else s.lower()


def load_table_categories(path: Path) -> dict[tuple[str, str], str]:
    """Load table_categories.json → {(paper_slug, normalized_item_id): category}."""
    if not path.exists():
        print(f"  table_categories.json not found at {path}")
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        print(f"  Failed to load {path}: {e}")
        return {}
    lookup: dict[tuple[str, str], str] = {}
    for entry in data:
        pid = entry.get("paper_id", "")
        for c in entry.get("classifications", []):
            tid = c.get("table_id", "")
            cat = c.get("category", "")
            if pid and tid and cat:
                lookup[(pid, _normalize_item_id(tid))] = cat
    return lookup


def attach_table_category(df: pd.DataFrame, lookup: dict[tuple[str, str], str]) -> pd.DataFrame:
    """Add a `table_category` column using (paper_slug, item_id) lookup."""
    if df.empty or not lookup:
        df = df.copy()
        df["table_category"] = pd.NA
        return df
    df = df.copy()
    norm_ids = df["item_id"].map(_normalize_item_id)
    df["table_category"] = [
        lookup.get((p, i), pd.NA) for p, i in zip(df["paper_slug"], norm_ids)
    ]
    return df


def plot_grade_by_table_category(df_items: pd.DataFrame, output_dir: Path, subdir: str = "",
                                  f_mode: str = "all_f"):
    """Grouped bar chart: share of tables grade A–B by approach × table category.

    Requires `table_category` column on df_items (attached via attach_table_category).
    """
    df = df_items[
        (df_items["item_type"] == "table") &
        df_items["table_category"].notna()
    ].copy()
    df = _filter_f_mode(df, f_mode, level="item")
    if df.empty:
        print("  Skipping grade_by_table_category: no categorized tables")
        return

    category_order = ["main_results", "mechanism", "robustness", "descriptive", "other"]
    category_labels = {
        "main_results": "Main results",
        "mechanism": "Mechanism",
        "robustness": "Robustness",
        "descriptive": "Descriptive",
        "other": "Other",
    }

    approaches = _approaches_in(df)

    # Compute %A-B per (approach, category)
    df["ab"] = df["grade"].isin(["A", "B"])
    pct = df.groupby(["approach", "table_category"])["ab"].mean().mul(100).unstack(fill_value=np.nan)
    pct = pct.reindex(columns=[c for c in category_order if c in pct.columns])

    # Keep approach ordering stable
    pct = pct.reindex(index=[a for a in approaches if a in pct.index])

    n_cats = len(pct.columns)
    n_apps = len(pct.index)
    x = np.arange(n_cats)
    width = 0.8 / max(n_apps, 1)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, approach in enumerate(pct.index):
        values = pct.loc[approach].values
        color = APPROACH_COLORS.get(approach, "#95a5a6")
        label = APPROACH_LABELS.get(approach, approach).replace("\n", " ")
        ax.bar(x + i * width - (n_apps - 1) * width / 2, values,
                width=width, color=color, label=label, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([category_labels.get(c, c) for c in pct.columns],
                        fontsize=12, fontweight="bold")
    ax.set_ylabel("Share grade A–B (%)", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylim(0, 105)
    # Annotate per-category table counts
    counts_per_cat = df.groupby("table_category")["item_id"].count().reindex(pct.columns, fill_value=0)
    for i, c in enumerate(pct.columns):
        ax.text(i, -8, f"n={counts_per_cat[c]}", fontsize=9, ha="center",
                color="#555555", transform=ax.get_xaxis_transform())
    place_legend(fig, ax, fontsize=10, ncol=4)
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig, output_dir, f"grade_by_table_category{F_MODE_SUFFIX[f_mode]}", subdir)

    # Also save the underlying data
    target = output_dir / subdir if subdir else output_dir
    target.mkdir(parents=True, exist_ok=True)
    pct.round(2).to_csv(target / f"grade_by_table_category{F_MODE_SUFFIX[f_mode]}.csv")


def plot_coefficient_se_cdf(df_cells: pd.DataFrame, output_dir: Path, subdir: str = "",
                             category_filter: list[str] | None = None,
                             name: str = "coefficient_se_cdf"):
    """Cumulative distribution of |coeff difference| / SE (right panel only).

    If `category_filter` is given (e.g. ['main_results', 'mechanism', 'robustness']),
    filter df_cells to cells whose table_category ∈ filter. Requires a `table_category`
    column on df_cells.
    """
    df = df_cells[
        (df_cells["row_type"] == "coefficient") &
        df_cells["original_value"].notna() &
        df_cells["replicated_value"].notna() &
        df_cells["is_numeric"] &
        # OLD: excluded F
        # (df_cells["item_grade"] != "F")
        # NEW: exclude NA instead
        (df_cells["item_grade"] != "NA")
    ].copy()
    if category_filter is not None:
        if "table_category" not in df.columns:
            print(f"  Skipping {name}: no table_category column")
            return
        df = df[df["table_category"].isin(category_filter)]
    if df.empty:
        return

    df["abs_diff"] = (df["original_value"].astype(float) - df["replicated_value"].astype(float)).abs()
    df["se"] = df["original_se"]
    mask = df["se"].isna()
    df.loc[mask, "se"] = df.loc[mask, "replicated_se"]
    df = df[df["se"].notna() & (df["se"].astype(float) > 0)].copy()
    if df.empty:
        return

    df["diff_over_se"] = df["abs_diff"] / df["se"].astype(float)
    approaches = _approaches_in(df)

    fig, ax = plt.subplots(figsize=(7, 5))
    x_max = 10
    # Collect end-points for label placement
    end_points = []  # (y_value, label, color)
    for a in approaches:
        vals = df.loc[df["approach"] == a, "diff_over_se"].sort_values().values
        if len(vals) == 0:
            continue
        cdf_y = np.arange(1, len(vals) + 1) / len(vals) * 100
        color = APPROACH_COLORS.get(a, "#95a5a6")
        ax.plot(vals, cdf_y, color=color, linewidth=2)
        # Get y-value at x_max for label placement
        y_at_end = np.interp(x_max, vals, cdf_y)
        label = APPROACH_LABELS.get(a, a).replace("\n", " ")
        end_points.append((y_at_end, label, color))

    ax.axvline(x=1.96, color="red", linestyle="--", alpha=0.5)
    ax.text(2.05, 5, "1.96", color="red", fontsize=10, alpha=0.7)

    # Sort end-points by y-value and spread to avoid overlap, centered on midpoint
    end_points.sort(key=lambda x: x[0])
    n = len(end_points)
    min_gap = 3.5  # minimum gap in percentage points between labels

    # First pass: spread from bottom up
    spread_y = []
    for i, (y_val, label, color) in enumerate(end_points):
        target_y = y_val
        for py in spread_y:
            if abs(target_y - py) < min_gap:
                target_y = py + min_gap
        spread_y.append(target_y)

    # Second pass: shift all labels so the group is centered on the midpoint
    # of the actual y-values (reduces drift from the lines)
    actual_mid = (end_points[0][0] + end_points[-1][0]) / 2
    spread_mid = (spread_y[0] + spread_y[-1]) / 2
    shift = actual_mid - spread_mid
    spread_y = [y + shift for y in spread_y]

    for i, (y_val, label, color) in enumerate(end_points):
        ax.annotate(
            label, xy=(x_max, y_val), xytext=(x_max + 0.15, spread_y[i]),
            fontsize=9, fontweight="bold", color=color, va="center",
            annotation_clip=False,
        )

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 105)
    ax.set_xlabel("|Coeff. difference| / SE", fontsize=14, fontweight="bold")
    ax.set_ylabel("Cumulative share of coefficients (%)", fontsize=14, fontweight="bold")
    apply_style(ax)
    plt.tight_layout()
    fig.subplots_adjust(right=0.72)
    save_figure(fig, output_dir, name, subdir)


def plot_pct_diff_cdf_by_cell_type(df_cells: pd.DataFrame, output_dir: Path, subdir: str = "",
                                    pct_col: str = "percent_difference", name_suffix: str = "",
                                    f_mode: str = "all_f"):
    """CDF of |% difference| by cell type, one panel per type, lines per approach."""
    CELL_TYPES = [
        ("coefficient", "Coefficients"),
        ("se", "Standard Errors"),
        ("statistic_r2", "R²"),
        ("statistic_n_obs", "N Observations"),
    ]
    if pct_col not in df_cells.columns:
        return
    cell_col = "cell_grade_rounded" if pct_col.endswith("_rounded") else "cell_grade"
    base = _cells_for_mode(df_cells, f_mode, cell_col=cell_col)
    df = base[
        base[pct_col].notna() &
        base["is_numeric"] &
        base["row_type"].isin([ct for ct, _ in CELL_TYPES])
    ].copy()
    if df.empty:
        return

    df["pct_abs"] = df[pct_col].abs().replace([np.inf], np.nan)
    df = df[df["pct_abs"].notna()].copy()

    approaches = _approaches_in(df)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (rt, rt_label) in enumerate(CELL_TYPES):
        ax = axes[idx]
        ax.set_title(rt_label, fontsize=12, fontweight="bold")
        sub = df[df["row_type"] == rt]

        for a in approaches:
            vals = sub.loc[sub["approach"] == a, "pct_abs"].sort_values().values
            if len(vals) < 5:
                continue
            cdf_y = np.arange(1, len(vals) + 1) / len(vals) * 100
            ax.plot(vals, cdf_y,
                    label=APPROACH_LABELS.get(a, a).replace("\n", " "),
                    color=APPROACH_COLORS.get(a, "#95a5a6"), linewidth=2)

        ax.set_xlim(0, 100)
        ax.axvline(x=2, color="gray", linestyle=":", alpha=0.5)
        ax.axvline(x=20, color="gray", linestyle=":", alpha=0.3)
        if idx >= 2:
            ax.set_xlabel("|% Difference|", fontsize=12)
        if idx % 2 == 0:
            ax.set_ylabel("Cumulative Share (%)", fontsize=12)
        apply_style(ax)

    # Single legend from last panel with data
    for ax in reversed(axes):
        h, l = ax.get_legend_handles_labels()
        if h:
            place_legend(fig, ax, fontsize=10, ncol=4)
            break

    plt.tight_layout()
    save_figure(fig, output_dir, f"pct_diff_cdf_by_cell_type_{f_mode}{name_suffix}", subdir)


def plot_pct_diff_by_cell_type(df_cells: pd.DataFrame, output_dir: Path, subdir: str = "",
                                pct_col: str = "percent_difference", name_suffix: str = "",
                                f_mode: str = "all_f"):
    """Median |% difference| by cell type and approach, with IQR whiskers."""
    CELL_TYPES = [
        ("coefficient", "Coefficients"),
        ("se", "Standard Errors"),
        ("statistic_r2", "R²"),
        ("p_value", "P-values"),
    ]
    if pct_col not in df_cells.columns:
        return
    cell_col = "cell_grade_rounded" if pct_col.endswith("_rounded") else "cell_grade"
    base = _cells_for_mode(df_cells, f_mode, cell_col=cell_col)
    df = base[
        base[pct_col].notna() &
        base["is_numeric"] &
        base["row_type"].isin([ct for ct, _ in CELL_TYPES])
    ].copy()
    if df.empty:
        print("  Skipping pct_diff_by_cell_type: no data")
        return

    df["pct_abs"] = df[pct_col].abs().replace([np.inf], np.nan)

    approaches = _approaches_in(df)
    n_types = len(CELL_TYPES)
    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    axes = axes.flatten()

    for idx, (rt, rt_label) in enumerate(CELL_TYPES):
        ax = axes[idx]
        ax.set_title(rt_label, fontsize=12, fontweight="bold")
        sub = df[df["row_type"] == rt]

        medians, q25s, q75s, labels, colors = [], [], [], [], []
        for a in approaches:
            vals = sub.loc[sub["approach"] == a, "pct_abs"].dropna()
            if len(vals) < 5:
                continue
            medians.append(vals.median())
            q25s.append(vals.quantile(0.25))
            q75s.append(vals.quantile(0.75))
            labels.append(APPROACH_LABELS.get(a, a).replace("\n", " "))
            colors.append(APPROACH_COLORS.get(a, "#95a5a6"))

        if not medians:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        # Sort by median (best on top)
        order = np.argsort(medians)
        medians = [medians[i] for i in order]
        q25s = [q25s[i] for i in order]
        q75s = [q75s[i] for i in order]
        labels = [labels[i] for i in order]
        colors = [colors[i] for i in order]

        y = np.arange(len(medians))
        xerr_low = [m - q for m, q in zip(medians, q25s)]
        xerr_high = [q - m for m, q in zip(medians, q75s)]
        ax.barh(y, medians, color=colors, alpha=0.7, edgecolor="white", height=0.6)
        ax.errorbar(medians, y, xerr=[xerr_low, xerr_high], fmt="none",
                    ecolor="black", capsize=4, capthick=1.5, linewidth=1.5)

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=10)
        for tick_label, color in zip(ax.get_yticklabels(), colors):
            tick_label.set_color(color)
            tick_label.set_fontweight("bold")
        ax.invert_yaxis()
        if idx >= 2:
            ax.set_xlabel("Median |% difference|", fontsize=11, fontweight="bold")
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])

        apply_style(ax)

    # Hide unused axes
    for idx in range(n_types, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    save_figure(fig, output_dir, f"pct_diff_by_cell_type_{f_mode}{name_suffix}", subdir)


def plot_coefficient_se_scaled(df_cells: pd.DataFrame, output_dir: Path, subdir: str = ""):
    df = df_cells[
        (df_cells["row_type"] == "coefficient") &
        df_cells["original_value"].notna() &
        df_cells["replicated_value"].notna() &
        df_cells["is_numeric"] &
        # OLD: excluded F
        # (df_cells["item_grade"] != "F")
        # NEW: exclude NA instead
        (df_cells["item_grade"] != "NA")
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
    ax.set_ylabel("|Coeff. difference| / SE", fontsize=14, fontweight="bold")
    ax.axhline(y=1.96, color="red", linestyle="--", alpha=0.5, label="1.96 (95% CI)")
    place_legend(fig, ax, fontsize=12)
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
    place_legend(fig, ax2, fontsize=9)
    apply_style(ax2)

    plt.tight_layout()
    save_figure(fig, output_dir, "coefficient_se_scaled", subdir)


def plot_same_significance(df_cells: pd.DataFrame, output_dir: Path, subdir: str = ""):
    df = df_cells[
        (df_cells["row_type"] == "coefficient") &
        df_cells["significance_stars_orig"].notna() &
        df_cells["significance_stars_repl"].notna() &
        # OLD: excluded F
        # (df_cells["item_grade"] != "F")
        # NEW: exclude NA instead
        (df_cells["item_grade"] != "NA")
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
    ct = ct.reindex(index=bin_labels, fill_value=0)
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
    place_legend(fig, ax2, fontsize=9)
    apply_style(ax2)

    plt.tight_layout()
    save_figure(fig, output_dir, "same_significance", subdir)


def plot_same_sign(df_cells: pd.DataFrame, output_dir: Path, subdir: str = ""):
    df = df_cells[
        (df_cells["row_type"] == "coefficient") &
        df_cells["sign_match"].notna() &
        # OLD: excluded F
        # (df_cells["item_grade"] != "F")
        # NEW: exclude NA instead
        (df_cells["item_grade"] != "NA")
    ].copy()
    if df.empty:
        return

    approaches = _approaches_in(df)
    fig, ax = plt.subplots(figsize=(7, 5.5))
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
    for tick_label, color in zip(ax.get_xticklabels(), colors):
        tick_label.set_color(color)
        tick_label.set_fontweight("bold")
    ax.set_ylabel("Coefficients with same sign (%)", fontsize=14, fontweight="bold")
    for i, (rate, color) in enumerate(zip(match_rates, colors)):
        ax.text(i, rate + 1, f"{rate:.1f}%", ha="center", fontsize=10, fontweight="bold", color=color)
    ax.set_ylim(0, 110)
    apply_style(ax)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig, output_dir, "same_sign", subdir)


def plot_same_sign_with_missing(df_cells: pd.DataFrame, output_dir: Path, subdir: str = ""):
    """Same-sign match share with missing replicated values in the denominator.

    Denominator = all coefficient cells with a numeric original value (i.e. the
    agent was supposed to produce them). Cells where the agent failed to produce
    a value count as sign mismatches, penalizing coverage gaps.
    """
    df = df_cells[
        (df_cells["row_type"] == "coefficient") &
        df_cells["original_value"].notna() &
        (df_cells["item_grade"] != "NA")
    ].copy()
    if df.empty:
        return

    # Effective sign-match: False if replicated missing; sign_match value otherwise
    df["sign_match_penalized"] = df["sign_match"].where(
        df["replicated_value"].notna(), other=False
    )

    approaches = _approaches_in(df)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    match_rates, xlabels, colors = [], [], []
    for a in approaches:
        sub = df[df["approach"] == a]
        if sub.empty:
            continue
        match_rates.append(sub["sign_match_penalized"].mean() * 100)
        xlabels.append(APPROACH_LABELS.get(a, a).replace("\n", " "))
        colors.append(APPROACH_COLORS.get(a, "#95a5a6"))

    x = np.arange(len(xlabels))
    ax.bar(x, match_rates, color=colors, edgecolor="white", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=10, rotation=25, ha="right")
    for tick_label, color in zip(ax.get_xticklabels(), colors):
        tick_label.set_color(color)
        tick_label.set_fontweight("bold")
    ax.set_ylabel("Coefficients with same sign (%)", fontsize=14, fontweight="bold")
    for i, (rate, color) in enumerate(zip(match_rates, colors)):
        ax.text(i, rate + 1, f"{rate:.1f}%", ha="center", fontsize=10, fontweight="bold", color=color)
    ax.set_ylim(0, 110)
    apply_style(ax)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig, output_dir, "same_sign_with_missing", subdir)


def plot_statistic_pct_difference(df_cells: pd.DataFrame, output_dir: Path,
                                   row_type_filter: str, name: str,
                                   ylabel: str = "", subdir: str = ""):
    df = df_cells[
        (df_cells["row_type"] == row_type_filter) &
        df_cells["percent_difference"].notna() &
        df_cells["is_numeric"] &
        # OLD: excluded F
        # (df_cells["item_grade"] != "F")
        # NEW: exclude NA instead
        (df_cells["item_grade"] != "NA")
    ].copy()
    if df.empty:
        return

    df["pct_capped"] = df["percent_difference"].clip(upper=200)
    approaches = _approaches_in(df)

    fig, ax = plt.subplots(figsize=(7, 5))
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
        # OLD: excluded F
        # (df_items["grade"] != "F")
        # NEW: exclude NA instead
        (df_items["grade"] != "NA")
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

    fig, ax = plt.subplots(figsize=(7, 5))
    present = [a for a in _approaches_in(df) if a in ct.index]
    ct_present = ct.loc[present].copy()
    ct_present.columns = [cat_labels.get(c, c) for c in ct_present.columns]
    display_colors = [cat_colors.get(c, "#95a5a6") for c in ct.columns]

    ct_present.plot(kind="bar", stacked=True, ax=ax, color=display_colors, edgecolor="white", width=0.7)
    ax.set_xticklabels([APPROACH_LABELS.get(a, a).replace("\n", " ") for a in present], fontsize=10, rotation=25, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("Share (%)", fontsize=18, fontweight="bold")
    place_legend(fig, ax, fontsize=10, title="Fault Category")
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig, output_dir, "fault_attribution", subdir)


def generate_fault_by_grade_table(df_items: pd.DataFrame, output_dir: Path, subdir: str = ""):
    df = df_items[
        df_items["primary_fault"].notna() &
        (df_items["primary_fault"] != "") &
        # OLD: excluded F
        # (df_items["grade"] != "F")
        # NEW: exclude NA instead
        (df_items["grade"] != "NA")
    ].copy()
    if df.empty:
        return

    ct = pd.crosstab(df["grade"], df["primary_fault"])
    # OLD: excluded F
    # grades_shown = [g for g in GRADE_ORDER if g != "F" and g in ct.index]
    # NEW: exclude NA instead
    grades_shown = [g for g in GRADE_ORDER if g != "NA" and g in ct.index]
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
        # OLD: excluded F
        # (df_items["grade"] != "F")
        # NEW: exclude NA instead
        (df_items["grade"] != "NA")
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
# Section: Discrepancy Analysis (from code_JE error_analysis pipeline)
# ============================================================================

_ROOT_CAUSE_TRIGGERS = {"contradicts", "omission"}

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


def _derive_root_cause(d: dict) -> str:
    """Derive root cause from consistency check verdicts (matches 03_summarize_errors.py)."""
    p_code = d.get("paper_vs_original_code", "unclear")
    p_sum = d.get("paper_vs_summary", "unclear")
    s_agent = d.get("summary_vs_agent", "unclear")
    data = d.get("data_available", None)

    if data == "missing":
        return "Data not in package"
    if p_code == "contradicts":
        return "Paper-code mismatch"
    if p_code == "omission":
        return "Paper underspecified"
    if p_sum == "contradicts":
        return "Summary gap (contradicts)"
    if p_sum == "omission":
        return "Summary gap (omission)"
    if s_agent == "contradicts":
        return "Agent contradicted summary"
    if s_agent == "omission":
        return "Agent missed summary info"
    if p_code == "unclear" or p_sum == "unclear" or s_agent == "unclear":
        return "Insufficient specification"
    return "Unexplained"


def _load_error_analysis(error_analysis_dir: Path) -> pd.DataFrame:
    """Load all divergences_enriched.json into a DataFrame."""
    rows = []
    if not error_analysis_dir.exists():
        return pd.DataFrame()
    for enriched_path in error_analysis_dir.rglob("divergences_enriched.json"):
        # only the live error_source/ dir — archived variants (e.g.
        # error_source.preunclear_20260414/) hold superseded verdicts for the
        # SAME divergences and would double-count every record
        if enriched_path.parent.name != "error_source":
            continue
        data = _load_json(enriched_path)
        if not data:
            continue
        paper_id = data.get("paper_id", "")
        agent_label = data.get("agent", "")
        # Parse approach from agent label (e.g. "claude-opus-4-6_claude-code" -> "claude-code")
        approach = agent_label
        for known in ["claude-code", "codex", "swe-agent", "opencode"]:
            if agent_label.endswith(f"_{known}"):
                model = agent_label[: -len(f"_{known}") - 1] if agent_label.endswith(f"_{known}") else ""
                approach = f"{known}/{model}" if model else known
                break
        # Fix: approach should be approach_raw/model format
        # e.g. "claude-opus-4-6_claude-code" -> "claude-code/claude-opus-4-6"
        for known in ["claude-code", "codex", "swe-agent", "opencode"]:
            if f"_{known}" in agent_label:
                model_part = agent_label.replace(f"_{known}", "")
                approach = f"{known}/{model_part}"
                break

        for div in data.get("divergences", []):
            root_cause = _derive_root_cause(div)
            desc = div.get("description", "") or ""
            is_parse_failed = "[Parse failed]" in desc
            rows.append({
                "paper_slug": paper_id,
                "approach": approach,
                "agent_label": agent_label,
                "div_id": div.get("id"),
                "output": div.get("output", ""),
                "description": desc,
                "parse_failed": is_parse_failed,
                "divergence_type": div.get("divergence_type", ""),
                "severity": div.get("severity", ""),
                "data_available": div.get("data_available", ""),
                "paper_vs_original_code": div.get("paper_vs_original_code", ""),
                "paper_vs_summary": div.get("paper_vs_summary", ""),
                "summary_vs_agent": div.get("summary_vs_agent", ""),
                "root_cause": root_cause,
            })
    return pd.DataFrame(rows)


def plot_divergence_types(df_div: pd.DataFrame, output_dir: Path, subdir: str = ""):
    """Stacked bar: divergence type (S-codes) distribution by approach."""
    if df_div.empty:
        print("  Skipping divergence_types: no data")
        return

    stype_order = ["S1", "S2", "S3", "S4", "S5", "S6", "S8", "S9", "S0"]
    stype_labels = {
        "S1": "S1: Wrong model spec", "S2": "S2: Wrong estimator",
        "S3": "S3: Data substitution", "S4": "S4: Wrong sample",
        "S5": "S5: Wrong variable", "S6": "S6: Missing component",
        "S8": "S8: Wrong merge/transform", "S9": "S9: Wrong sequencing",
        "S0": "S0: Other",
    }
    stype_colors = {
        "S1": "#e74c3c", "S2": "#c0392b", "S3": "#e67e22", "S4": "#f39c12",
        "S5": "#3498db", "S6": "#9b59b6", "S8": "#1abc9c", "S9": "#2ecc71",
        "S0": "#95a5a6",
    }

    ct = pd.crosstab(df_div["approach"], df_div["divergence_type"], normalize="index") * 100
    present_types = [s for s in stype_order if s in ct.columns]
    ct = ct.reindex(columns=present_types, fill_value=0)

    fig, ax = plt.subplots(figsize=(7, 5))
    approaches = _approaches_in(df_div)
    ct_plot = ct.loc[[a for a in approaches if a in ct.index]]
    ct_plot.columns = [stype_labels.get(s, s) for s in ct_plot.columns]
    colors = [stype_colors.get(s, "#95a5a6") for s in present_types]

    ct_plot.plot(kind="bar", stacked=True, ax=ax, color=colors, edgecolor="white", width=0.7)
    ax.set_xticklabels([APPROACH_LABELS.get(a, a).replace("\n", " ") for a in approaches if a in ct.index],
                       fontsize=10, rotation=25, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("Share of Divergences (%)", fontsize=16, fontweight="bold")
    place_legend(fig, ax, fontsize=8, ncol=3, title="Divergence Type")
    apply_style(ax)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.35)
    save_figure(fig, output_dir, "divergence_types", subdir)


def plot_divergence_types_aggregate(df_div: pd.DataFrame, output_dir: Path, subdir: str = ""):
    """Horizontal bar: divergence type share aggregated across all runs."""
    if df_div.empty:
        print("  Skipping divergence_types_aggregate: no data")
        return

    stype_order = ["S1", "S2", "S3", "S4", "S5", "S6", "S8", "S9", "S0"]
    stype_labels = {
        "S1": "S1: Wrong model spec", "S2": "S2: Wrong estimator",
        "S3": "S3: Data substitution", "S4": "S4: Wrong sample",
        "S5": "S5: Wrong variable", "S6": "S6: Missing component",
        "S8": "S8: Wrong merge/transform", "S9": "S9: Wrong sequencing",
        "S0": "S0: Other",
    }
    stype_colors = {
        "S1": "#e74c3c", "S2": "#c0392b", "S3": "#e67e22", "S4": "#f39c12",
        "S5": "#3498db", "S6": "#9b59b6", "S8": "#1abc9c", "S9": "#2ecc71",
        "S0": "#95a5a6",
    }

    counts = df_div["divergence_type"].value_counts()
    total = counts.sum()
    present = [s for s in stype_order if s in counts.index]
    vals = [counts[s] / total * 100 for s in present]
    abs_vals = [counts[s] for s in present]
    labels = [stype_labels.get(s, s) for s in present]
    colors = [stype_colors.get(s, "#95a5a6") for s in present]

    fig, ax = plt.subplots(figsize=(7, 5))
    y_pos = range(len(present))
    bars = ax.barh(y_pos, vals, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Share of Divergences (%)", fontsize=14, fontweight="bold")
    ax.invert_yaxis()

    for bar, pct, n in zip(bars, vals, abs_vals):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}% (n={n})", va="center", fontsize=9)

    ax.set_xlim(0, max(vals) * 1.25 if vals else 100)
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig, output_dir, "divergence_types_aggregate", subdir)


def plot_root_causes(df_div: pd.DataFrame, output_dir: Path, subdir: str = ""):
    """Stacked bar: root cause distribution by approach."""
    if df_div.empty:
        print("  Skipping root_causes: no data")
        return

    cause_order = [
        "Agent contradicted summary", "Agent missed summary info",
        "Summary gap (contradicts)", "Summary gap (omission)",
        "Paper underspecified", "Paper-code mismatch",
        "Data not in package", "Insufficient specification", "Unexplained",
    ]
    cause_colors = {
        "Agent contradicted summary": "#e74c3c",
        "Agent missed summary info": "#c0392b",
        "Summary gap (contradicts)": "#e67e22",
        "Summary gap (omission)": "#f39c12",
        "Paper underspecified": "#9b59b6",
        "Paper-code mismatch": "#8e44ad",
        "Data not in package": "#3498db",
        "Insufficient specification": "#1abc9c",
        "Unexplained": "#95a5a6",
    }

    ct = pd.crosstab(df_div["approach"], df_div["root_cause"], normalize="index") * 100
    present_causes = [c for c in cause_order if c in ct.columns]
    extra = [c for c in ct.columns if c not in cause_order]
    present_causes += extra
    ct = ct.reindex(columns=present_causes, fill_value=0)

    fig, ax = plt.subplots(figsize=(7, 5))
    approaches = _approaches_in(df_div)
    ct_plot = ct.loc[[a for a in approaches if a in ct.index]]
    colors = [cause_colors.get(c, "#bdc3c7") for c in present_causes]

    ct_plot.plot(kind="bar", stacked=True, ax=ax, color=colors, edgecolor="white", width=0.7)
    ax.set_xticklabels([APPROACH_LABELS.get(a, a).replace("\n", " ") for a in approaches if a in ct.index],
                       fontsize=10, rotation=25, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("Share of Divergences (%)", fontsize=16, fontweight="bold")
    place_legend(fig, ax, fontsize=8, ncol=3, title="Root Cause")
    apply_style(ax)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.35)
    save_figure(fig, output_dir, "root_causes", subdir)


def plot_root_causes_aggregate(df_div: pd.DataFrame, output_dir: Path, subdir: str = ""):
    """Horizontal bar: root cause share aggregated across all runs."""
    if df_div.empty:
        print("  Skipping root_causes_aggregate: no data")
        return

    cause_order = [
        "Agent contradicted summary", "Agent missed summary info",
        "Summary gap (contradicts)", "Summary gap (omission)",
        "Paper underspecified", "Paper-code mismatch",
        "Data not in package", "Insufficient specification", "Unexplained",
    ]
    cause_colors = {
        "Agent contradicted summary": "#e74c3c",
        "Agent missed summary info": "#c0392b",
        "Summary gap (contradicts)": "#e67e22",
        "Summary gap (omission)": "#f39c12",
        "Paper underspecified": "#9b59b6",
        "Paper-code mismatch": "#8e44ad",
        "Data not in package": "#3498db",
        "Insufficient specification": "#1abc9c",
        "Unexplained": "#95a5a6",
    }

    counts = df_div["root_cause"].value_counts()
    total = counts.sum()
    present = [c for c in cause_order if c in counts.index]
    extra = [c for c in counts.index if c not in cause_order]
    present += extra
    vals = [counts[c] / total * 100 for c in present]
    abs_vals = [counts[c] for c in present]
    colors = [cause_colors.get(c, "#bdc3c7") for c in present]

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = range(len(present))
    bars = ax.barh(y_pos, vals, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(present, fontsize=11)
    ax.set_xlabel("Share of Divergences (%)", fontsize=14, fontweight="bold")
    ax.invert_yaxis()

    for bar, pct, n in zip(bars, vals, abs_vals):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}% (n={n})", va="center", fontsize=9)

    ax.set_xlim(0, max(vals) * 1.25 if vals else 100)
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig, output_dir, "root_causes_aggregate", subdir)


def plot_root_causes_horizontal(df_div: pd.DataFrame, output_dir: Path, subdir: str = ""):
    """Horizontal stacked bar: absolute divergence counts by approach and root cause."""
    if df_div.empty:
        return

    cause_order = [
        "Agent contradicted summary", "Agent missed summary info",
        "Summary gap (contradicts)", "Summary gap (omission)",
        "Paper underspecified", "Paper-code mismatch",
        "Data not in package", "Unexplained",
    ]
    cause_colors = {
        "Agent contradicted summary": "#e74c3c",
        "Agent missed summary info": "#c0392b",
        "Summary gap (contradicts)": "#e67e22",
        "Summary gap (omission)": "#f39c12",
        "Paper underspecified": "#9b59b6",
        "Paper-code mismatch": "#8e44ad",
        "Data not in package": "#3498db",
        "Unexplained": "#95a5a6",
    }

    # Merge "Insufficient specification" into "Unexplained"
    df_div = df_div.copy()
    df_div.loc[df_div["root_cause"] == "Insufficient specification", "root_cause"] = "Unexplained"
    ct = pd.crosstab(df_div["approach"], df_div["root_cause"])
    present_causes = [c for c in cause_order if c in ct.columns]
    ct = ct.reindex(columns=present_causes, fill_value=0)

    approaches = _approaches_in(df_div)
    ct_plot = ct.loc[[a for a in approaches if a in ct.index]]

    # Sort by total divergences
    ct_plot = ct_plot.loc[ct_plot.sum(axis=1).sort_values().index]

    fig, ax = plt.subplots(figsize=(10, max(3.5, len(ct_plot) * 0.6 + 1.5)))
    y = np.arange(len(ct_plot))
    lefts = np.zeros(len(ct_plot))

    for cause in present_causes:
        vals = ct_plot[cause].values.astype(float)
        if vals.sum() == 0:
            continue
        ax.barh(y, vals, left=lefts, color=cause_colors.get(cause, "#bdc3c7"),
                label=cause, height=0.6, edgecolor="white", linewidth=0.5)
        lefts += vals

    # Total count labels
    for i, total in enumerate(lefts):
        ax.text(total + 0.5, i, str(int(total)), va="center", fontsize=8, fontweight="bold")

    ax.set_yticks(y)
    labels = [APPROACH_LABELS.get(a, a).replace("\n", " ") for a in ct_plot.index]
    colors = [APPROACH_COLORS.get(a, "#95a5a6") for a in ct_plot.index]
    ax.set_yticklabels(labels, fontsize=8)
    for tick_label, color in zip(ax.get_yticklabels(), colors):
        tick_label.set_color(color)
        tick_label.set_fontweight("bold")
    ax.set_xlabel("Number of divergences", fontsize=10, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False)
    ax.grid(visible=False, which="both")
    place_legend(fig, ax, fontsize=7, ncol=4, title="Root Cause")
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.25)
    save_figure(fig, output_dir, "root_causes_absolute", subdir)


_ROOT_CAUSE_COARSE_RENAMED = {
    "Extraction vs Agent": "Agent error",
    "Paper vs Extraction": "Extractor error",
    "Paper vs Code": "Original error",
    "Missing data": "Data missing",
    "Other": "Other / unknown error",
}


def plot_root_causes_coarse_absolute(df_div: pd.DataFrame, output_dir: Path, subdir: str = ""):
    """Horizontal stacked bar: absolute divergence counts by approach and coarse root cause.

    Like `plot_root_causes_horizontal` (root_causes_absolute.pdf) but using the 5 coarse
    categories with user-facing labels (Agent error, Extractor error, Original error,
    Data missing, Other / unknown error).
    """
    if df_div.empty:
        return

    df = df_div.copy()
    df["root_cause_coarse"] = df["root_cause"].map(_ROOT_CAUSE_COARSE_MAP).fillna("Other")
    df["root_cause_renamed"] = df["root_cause_coarse"].map(_ROOT_CAUSE_COARSE_RENAMED)

    cause_order = ["Data missing", "Original error", "Extractor error",
                   "Agent error", "Other / unknown error"]
    cause_colors = {
        "Data missing": "#3498db",
        "Original error": "#9b59b6",
        "Extractor error": "#e67e22",
        "Agent error": "#e74c3c",
        "Other / unknown error": "#95a5a6",
    }

    ct = pd.crosstab(df["approach"], df["root_cause_renamed"])
    present_causes = [c for c in cause_order if c in ct.columns]
    ct = ct.reindex(columns=present_causes, fill_value=0)

    approaches = _approaches_in(df)
    ct_plot = ct.loc[[a for a in approaches if a in ct.index]]
    ct_plot = ct_plot.loc[ct_plot.sum(axis=1).sort_values().index]

    fig, ax = plt.subplots(figsize=(10, max(2.6, len(ct_plot) * 0.38 + 1.0)))
    y = np.arange(len(ct_plot))
    lefts = np.zeros(len(ct_plot))

    for cause in present_causes:
        vals = ct_plot[cause].values.astype(float)
        if vals.sum() == 0:
            continue
        ax.barh(y, vals, left=lefts, color=cause_colors.get(cause, "#bdc3c7"),
                label=cause, height=0.55, edgecolor="white", linewidth=0.5)
        lefts += vals

    for i, total in enumerate(lefts):
        ax.text(total + 0.5, i, str(int(total)), va="center", fontsize=9, fontweight="bold")

    ax.set_yticks(y)
    labels = [APPROACH_LABELS.get(a, a).replace("\n", " ") for a in ct_plot.index]
    colors = [APPROACH_COLORS.get(a, "#95a5a6") for a in ct_plot.index]
    ax.set_yticklabels(labels, fontsize=9)
    for tick_label, color in zip(ax.get_yticklabels(), colors):
        tick_label.set_color(color)
        tick_label.set_fontweight("bold")
    ax.set_xlabel("Number of divergences", fontsize=11, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False)
    ax.grid(visible=False, which="both")
    place_legend(fig, ax, fontsize=12, ncol=5, title="Root Cause")
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.32)
    save_figure(fig, output_dir, "root_causes_coarse_absolute", subdir)


def plot_root_causes_coarse(df_div: pd.DataFrame, output_dir: Path, subdir: str = ""):
    """Stacked bar: coarse root cause (4 categories) by approach."""
    if df_div.empty:
        print("  Skipping root_causes_coarse: no data")
        return

    df = df_div.copy()
    df["root_cause_coarse"] = df["root_cause"].map(_ROOT_CAUSE_COARSE_MAP).fillna("Other")

    cause_order = ["Extraction vs Agent", "Paper vs Extraction", "Paper vs Code", "Missing data", "Other"]
    cause_colors = {
        "Extraction vs Agent": "#e74c3c",
        "Paper vs Extraction": "#e67e22",
        "Paper vs Code": "#9b59b6",
        "Missing data": "#3498db",
        "Other": "#95a5a6",
    }

    ct = pd.crosstab(df["approach"], df["root_cause_coarse"], normalize="index") * 100
    ct = ct.reindex(columns=[c for c in cause_order if c in ct.columns], fill_value=0)

    fig, ax = plt.subplots(figsize=(7, 5))
    approaches = _approaches_in(df)
    ct_plot = ct.loc[[a for a in approaches if a in ct.index]]
    colors = [cause_colors[c] for c in ct_plot.columns]

    ct_plot.plot(kind="bar", stacked=True, ax=ax, color=colors, edgecolor="white", width=0.7)
    ax.set_xticklabels([APPROACH_LABELS.get(a, a).replace("\n", " ") for a in approaches if a in ct.index],
                       fontsize=10, rotation=25, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("Share of Divergences (%)", fontsize=14, fontweight="bold")
    place_legend(fig, ax, fontsize=9, ncol=3, title="Root Cause")
    apply_style(ax)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.3)
    save_figure(fig, output_dir, "root_causes_coarse", subdir)


def plot_divergence_types_comparison(df_div: pd.DataFrame, output_dir: Path, subdir: str = ""):
    """Three-panel horizontal bar chart for divergence types (S-codes).

    Mirrors plot_root_causes_comparison: shows all agents combined vs
    OpenCode/GPT-5.4 vs SWE-Agent/GPT-5.4.
    """
    if df_div.empty:
        return

    stype_order = ["S1", "S2", "S3", "S4", "S5", "S6", "S8", "S9", "S0"]
    stype_labels = {
        "S1": "S1: Wrong model spec", "S2": "S2: Wrong estimator",
        "S3": "S3: Data substitution", "S4": "S4: Wrong sample",
        "S5": "S5: Wrong variable", "S6": "S6: Missing component",
        "S8": "S8: Wrong merge/transform", "S9": "S9: Wrong sequencing",
        "S0": "S0: Other",
    }

    df = df_div.copy()
    groups = [
        ("All agents", df, "#555555"),
        ("OpenCode GPT-5.4", df[df["approach"] == "opencode/gpt-5.4"],
         APPROACH_COLORS.get("opencode/gpt-5.4", "#0984E3")),
        ("SWE-Agent GPT-5.4", df[df["approach"] == "swe-agent/gpt-5.4"],
         APPROACH_COLORS.get("swe-agent/gpt-5.4", "#6C5CE7")),
    ]

    present = [s for s in stype_order if any(
        (sub["divergence_type"] == s).any() for _, sub, _ in groups
    )]

    y_base = np.arange(len(present), dtype=float) * 0.9

    fig, ax = plt.subplots(figsize=(10, 5.5))
    n_groups = len(groups)
    bar_height = 0.22

    max_val = 0
    for gi, (label, sub, color) in enumerate(groups):
        if sub.empty:
            continue
        counts = sub["divergence_type"].value_counts()
        total = counts.sum()
        vals = [counts.get(s, 0) / total * 100 for s in present]
        max_val = max(max_val, max(vals) if vals else 0)
        y_off = y_base + (gi - n_groups / 2 + 0.5) * bar_height
        ax.barh(y_off, vals, height=bar_height, color=color, alpha=0.85,
                label=label, edgecolor="white", linewidth=0.5)
        for i, v in enumerate(vals):
            if v > 1:
                ax.text(v + 0.8, y_off[i], f"{v:.0f}%", va="center", fontsize=9)

    ax.set_xlim(0, max_val * 1.12)
    ax.set_yticks(y_base)
    ax.set_yticklabels([stype_labels.get(s, s) for s in present], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Share of divergences (%)", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.grid(visible=False, which="both")
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3,
              frameon=False, handlelength=1.5, handletextpad=0.5, columnspacing=2)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    save_figure(fig, output_dir, "divergence_types_comparison", subdir)


def plot_root_causes_comparison(df_div: pd.DataFrame, output_dir: Path, subdir: str = ""):
    """Three-panel: all agents combined, OpenCode GPT-5.4, SWE-Agent GPT-5.4."""
    if df_div.empty:
        return

    # Internal keys for data grouping
    cause_order = [
        "Extraction vs Agent", "Paper vs Extraction",
        "Paper vs Code", "Missing data", "Other",
    ]
    cause_colors = {
        "Extraction vs Agent": "#e74c3c",
        "Paper vs Extraction": "#e67e22",
        "Paper vs Code": "#9b59b6",
        "Missing data": "#3498db",
        "Other": "#95a5a6",
    }
    # The first 3 are "Inconsistency" sub-items (indented); rest are top-level
    inconsistency_causes = {"Extraction vs Agent", "Paper vs Extraction", "Paper vs Code"}

    df = df_div.copy()
    df["root_cause_coarse"] = df["root_cause"].map(_ROOT_CAUSE_COARSE_MAP).fillna("Other")

    groups = [
        ("All agents", df, "#555555"),
        ("OpenCode GPT-5.4", df[df["approach"] == "opencode/gpt-5.4"], APPROACH_COLORS.get("opencode/gpt-5.4", "#0984E3")),
        ("SWE-Agent GPT-5.4", df[df["approach"] == "swe-agent/gpt-5.4"], APPROACH_COLORS.get("swe-agent/gpt-5.4", "#6C5CE7")),
    ]

    present = [c for c in cause_order if any(
        (sub["root_cause_coarse"] == c).any() for _, sub, _ in groups
    )]

    # Build y positions: add extra space for "Inconsistency" header row
    header_offset = 0.5  # space for the header
    y_positions = []
    for i, c in enumerate(present):
        if i == 0 and c in inconsistency_causes:
            y_positions.append(header_offset)
        elif i > 0:
            prev = y_positions[-1]
            # Extra gap before "Missing data" (first non-inconsistency item)
            if c not in inconsistency_causes and present[i - 1] in inconsistency_causes:
                y_positions.append(prev + 1.2)
            else:
                y_positions.append(prev + 0.9)
        else:
            y_positions.append(0)
    y_base = np.array(y_positions)

    fig, ax = plt.subplots(figsize=(10, 4.0))
    n_groups = len(groups)
    bar_height = 0.22

    max_val = 0
    for gi, (label, sub, color) in enumerate(groups):
        if sub.empty:
            continue
        counts = sub["root_cause_coarse"].value_counts()
        total = counts.sum()
        vals = [counts.get(c, 0) / total * 100 for c in present]
        max_val = max(max_val, max(vals) if vals else 0)
        y_off = y_base + (gi - n_groups / 2 + 0.5) * bar_height
        bars = ax.barh(y_off, vals, height=bar_height, color=color, alpha=0.85,
                       label=label, edgecolor="white", linewidth=0.5)
        for i, v in enumerate(vals):
            if v > 2:
                ax.text(v + 0.8, y_off[i], f"{v:.0f}%", va="center", fontsize=9)

    ax.set_xlim(0, max_val * 1.12)

    # Y-axis: "Inconsistency" as a header tick, sub-items indented, then normal items
    incon_indices = [i for i, c in enumerate(present) if c in inconsistency_causes]

    # Place y-axis labels manually: "Inconsistency" left-aligned, sub-items italic,
    # "Missing data" and "Other" right-aligned (normal tick label position)
    ax.set_yticks(y_base)
    ax.set_yticklabels([""] * len(present))  # clear default labels

    # All labels right-aligned to axis; sub-items italic; header bold
    for i, c in enumerate(present):
        y = y_base[i]
        display_label = _ROOT_CAUSE_COARSE_RENAMED.get(c, c)
        if c in inconsistency_causes:
            ax.text(-0.02, y, display_label, fontsize=10, fontstyle="italic",
                    va="center", ha="right", transform=ax.get_yaxis_transform())
        else:
            ax.text(-0.02, y, display_label, fontsize=10,
                    va="center", ha="right", transform=ax.get_yaxis_transform())

    # "Inconsistency" header: right-aligned like the others but bold
    if incon_indices:
        header_y = y_base[incon_indices[0]] - 0.45
        ax.text(-0.02, header_y, "Inconsistency", fontsize=10, fontweight="bold",
                va="center", ha="right", transform=ax.get_yaxis_transform())

    ax.invert_yaxis()

    ax.set_xlabel("Share of divergences (%)", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.grid(visible=False, which="both")
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3,
              frameon=False, handlelength=1.5, handletextpad=0.5, columnspacing=2)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    save_figure(fig, output_dir, "root_causes_comparison", subdir)


def plot_cross_approach_verdict_consistency(df_div: pd.DataFrame, output_dir: Path, subdir: str = ""):
    """For paper≠code and paper≠summary verdicts: if one approach finds an error,
    how often do other approaches find the same error on the same paper×item?"""
    if df_div.empty:
        print("  Skipping verdict_consistency: no data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    checks = [
        ("paper_vs_original_code", "Paper vs Code"),
        ("paper_vs_summary", "Paper vs Summary"),
    ]

    for ax, (check_col, check_title) in zip(axes, checks):
        # For each paper×output, check if at least one approach found an error
        df = df_div[df_div[check_col].isin(["consistent", "contradicts", "omission", "unclear"])].copy()
        df["is_error"] = df[check_col].isin(["contradicts", "omission"])

        # Group by paper×output: does this divergence exist across approaches?
        paper_items = df.groupby(["paper_slug", "output"]).agg(
            n_approaches=("approach", "nunique"),
            n_error=("is_error", "sum"),
            n_total=("is_error", "count"),
        ).reset_index()

        # Only look at paper×output combos seen by >=2 approaches
        multi = paper_items[paper_items["n_approaches"] >= 2]
        if multi.empty:
            ax.text(0.5, 0.5, "Not enough\nmulti-approach data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14)
            continue

        # Categorize: all agree error, all agree no error, mixed
        def _categorize(row):
            if row["n_error"] == row["n_total"]:
                return "All approaches: error"
            elif row["n_error"] == 0:
                return "All approaches: no error"
            else:
                return "Mixed (some error, some not)"

        multi["agreement"] = multi.apply(_categorize, axis=1)
        counts = multi["agreement"].value_counts()

        cat_order = ["All approaches: error", "Mixed (some error, some not)", "All approaches: no error"]
        cat_colors_local = {
            "All approaches: error": "#e74c3c",
            "Mixed (some error, some not)": "#f39c12",
            "All approaches: no error": "#2ecc71",
        }
        vals = [counts.get(c, 0) for c in cat_order]
        total = sum(vals)
        bars = ax.bar(range(len(cat_order)), vals,
                      color=[cat_colors_local[c] for c in cat_order], edgecolor="white")
        ax.set_xticks(range(len(cat_order)))
        ax.set_xticklabels(["All: error", "Mixed", "All: no error"], fontsize=11)
        ax.set_ylabel("Number of paper × table pairs", fontsize=14, fontweight="bold")
        for i, v in enumerate(vals):
            if total > 0:
                ax.text(i, v + 0.5, f"{v} ({v/total*100:.0f}%)", ha="center", fontsize=10, fontweight="bold")
        apply_style(ax)

    plt.tight_layout()
    save_figure(fig, output_dir, "verdict_consistency", subdir)


def plot_verdict_distribution(df_div: pd.DataFrame, output_dir: Path, subdir: str = ""):
    """For each of the 3 checks, show the distribution of verdicts (consistent/contradicts/omission/unclear)."""
    if df_div.empty:
        print("  Skipping verdict_distribution: no data")
        return

    checks = [
        ("paper_vs_original_code", "Paper vs Code"),
        ("paper_vs_summary", "Paper vs Summary"),
        ("summary_vs_agent", "Summary vs Agent"),
    ]
    verdict_order = ["consistent", "contradicts", "omission", "unclear"]
    verdict_colors = {
        "consistent": "#2ecc71",
        "contradicts": "#e74c3c",
        "omission": "#e67e22",
        "unclear": "#95a5a6",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for ax, (check_col, check_title) in zip(axes, checks):
        df = df_div[df_div[check_col].isin(verdict_order)].copy()
        if df.empty:
            ax.set_visible(False)
            continue

        ct = pd.crosstab(df["approach"], df[check_col], normalize="index") * 100
        ct = ct.reindex(columns=verdict_order, fill_value=0)

        approaches = _approaches_in(df)
        ct_plot = ct.loc[[a for a in approaches if a in ct.index]]

        x = np.arange(len(ct_plot))
        width = 0.18
        for i, verdict in enumerate(verdict_order):
            vals = ct_plot[verdict].values if verdict in ct_plot.columns else [0] * len(ct_plot)
            ax.bar(x + i * width, vals, width, label=verdict,
                   color=verdict_colors[verdict], edgecolor="white")

        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([APPROACH_LABELS.get(a, a).replace("\n", " ") for a in ct_plot.index],
                           fontsize=9, rotation=30, ha="right")
        if ax == axes[0]:
            ax.set_ylabel("Share of divergences (%)", fontsize=14, fontweight="bold")
        apply_style(ax)

    # Single legend
    place_legend(fig, axes[-1], fontsize=11)
    plt.tight_layout()
    save_figure(fig, output_dir, "verdict_distribution", subdir)


# ============================================================================
# Main
# ============================================================================

def filter_to_complete_tables(
    df_runs: pd.DataFrame,
    df_items: pd.DataFrame,
    df_cells: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Filter to only tables where all approaches produced a non-F/non-NA grade.

    For each (paper_slug, item_id), checks that every approach present for that
    paper produced a non-F, non-NA grade. Tables that any approach failed on
    (F, NA, or missing) are excluded. Paper-level grades in df_runs are
    recomputed from the surviving items.

    Returns filtered copies of (df_runs, df_items, df_cells).
    """
    table_items = df_items[df_items["item_type"] == "table"].copy()

    # For each paper, which approaches are present?
    approaches_per_paper = df_runs.groupby("paper_slug")["approach"].apply(set).to_dict()

    # For each (paper, item_id), collect which approaches have non-F grades
    item_approach_grades = table_items.groupby(
        ["paper_slug", "item_id"]
    ).apply(lambda g: dict(zip(g["approach"], g["grade"])), include_groups=False)

    # Keep a table only if every approach present for that paper has a non-F grade
    keep_items = set()
    for (paper, item_id), approach_grades in item_approach_grades.items():
        paper_approaches = approaches_per_paper.get(paper, set())
        # All approaches must have graded this item, and none with F
        if (
            set(approach_grades.keys()) == paper_approaches
            # OLD: excluded F
            # and all(g != "F" for g in approach_grades.values())
            # NEW: exclude both F and NA (both represent failure/non-assessable)
            and all(g not in ("F", "NA") for g in approach_grades.values())
        ):
            keep_items.add((paper, item_id))

    n_before = len(table_items[["paper_slug", "item_id"]].drop_duplicates())
    n_after = len(keep_items)
    print(f"  Complete-tables filter: {n_after}/{n_before} tables kept "
          f"({n_before - n_after} excluded)")

    # Filter items
    df_items_f = df_items[
        (~(df_items["item_type"] == "table"))  # keep non-table items as-is
        | df_items.set_index(["paper_slug", "item_id"]).index.isin(keep_items)
    ].copy()

    # Filter cells
    if not df_cells.empty and "item_id" in df_cells.columns:
        df_cells_f = df_cells[
            df_cells.set_index(["paper_slug", "item_id"]).index.isin(keep_items)
        ].copy()
    else:
        df_cells_f = df_cells.copy()

    # Recompute paper-level grades from surviving items
    df_runs_f = df_runs.copy()
    # Ensure columns can hold the right types
    df_runs_f["overall_grade_num"] = df_runs_f["overall_grade_num"].astype(float)
    df_runs_f["overall_grade"] = df_runs_f["overall_grade"].astype(str)
    for g in GRADE_ORDER:
        col = f"n_grade_{g}"
        if col in df_runs_f.columns:
            df_runs_f[col] = df_runs_f[col].astype(int)

    new_grades = []
    new_grade_nums = []
    new_n_items = []
    new_n_tables = []
    new_grade_counts = {g: [] for g in GRADE_ORDER}

    for _, row in df_runs_f.iterrows():
        paper, approach = row["paper_slug"], row["approach"]
        items = df_items_f[
            (df_items_f["paper_slug"] == paper) & (df_items_f["approach"] == approach)
        ]
        table_items_run = items[items["item_type"] == "table"]
        if table_items_run.empty:
            new_grades.append("F")
            new_grade_nums.append(float(GRADE_TO_NUM["F"]))
        else:
            mean_num = table_items_run["grade_num"].mean()
            grade = NUM_TO_GRADE.get(int(np.floor(mean_num + 0.5)), "F")
            new_grades.append(grade)
            new_grade_nums.append(float(mean_num))
        for g in GRADE_ORDER:
            new_grade_counts[g].append(int((table_items_run["grade"] == g).sum()))
        new_n_items.append(len(items))
        new_n_tables.append(len(table_items_run))

    df_runs_f["overall_grade"] = new_grades
    df_runs_f["overall_grade_num"] = new_grade_nums
    df_runs_f["n_items"] = new_n_items
    df_runs_f["n_tables"] = new_n_tables
    for g in GRADE_ORDER:
        df_runs_f[f"n_grade_{g}"] = new_grade_counts[g]

    # Reapply categorical
    df_runs_f["approach"] = pd.Categorical(
        df_runs_f["approach"],
        categories=[c for c in APPROACH_ORDER if c in df_runs_f["approach"].unique()],
        ordered=True,
    )
    df_items_f["approach"] = pd.Categorical(
        df_items_f["approach"],
        categories=[c for c in APPROACH_ORDER if c in df_items_f["approach"].unique()],
        ordered=True,
    )
    df_items_f["grade"] = pd.Categorical(df_items_f["grade"], categories=GRADE_ORDER, ordered=True)

    # Drop runs/papers with zero surviving tables (no items to grade)
    papers_with_items = set(df_items_f[df_items_f["item_type"] == "table"]["paper_slug"].unique())
    n_before = df_runs_f["paper_slug"].nunique()
    df_runs_f = df_runs_f[df_runs_f["paper_slug"].isin(papers_with_items)].copy()
    df_items_f = df_items_f[df_items_f["paper_slug"].isin(papers_with_items)].copy()
    df_cells_f = df_cells_f[df_cells_f["paper_slug"].isin(papers_with_items)].copy()
    n_after = df_runs_f["paper_slug"].nunique()
    if n_before != n_after:
        print(f"  Dropped {n_before - n_after} papers with no surviving tables")

    return df_runs_f, df_items_f, df_cells_f


def _load_data_sufficiency(results_dir: Path) -> dict[str, str]:
    """Load data_sufficiency per paper from the GPT audit JSON.

    Searches standard locations relative to results_dir.
    Returns {paper_id: 'sufficient' | 'partial' | 'insufficient' | ...}.
    """
    candidates = [
        results_dir.parent / "audit_replication_data_v2.json",
        results_dir / "audit_replication_data_v2.json",
        results_dir.parent.parent / "data" / "audit_replication_data_v2.json",
        results_dir.parent.parent / "audit_replication_data_v2.json",
    ]
    for p in candidates:
        if p.exists():
            data = json.loads(p.read_text())
            if isinstance(data, list):
                return {e["paper_id"]: e.get("data_sufficiency", "unknown") for e in data}
    return {}


def _run_data_sufficiency_analysis(
    df_runs: pd.DataFrame, df_items: pd.DataFrame, df_cells: pd.DataFrame,
    results_dir: Path, output_dir: Path, subdir: str,
):
    """Generate plots split by data sufficiency (sufficient vs partial)."""
    suf_lookup = _load_data_sufficiency(results_dir)
    if not suf_lookup:
        print("  Skipping data_sufficiency: audit JSON not found")
        return

    target = output_dir / subdir
    target.mkdir(parents=True, exist_ok=True)

    # Attach sufficiency
    df_runs = df_runs.copy()
    df_items = df_items.copy()
    df_cells = df_cells.copy()
    df_runs["data_sufficiency"] = df_runs["paper_slug"].map(suf_lookup).fillna("unknown")
    df_items["data_sufficiency"] = df_items["paper_slug"].map(suf_lookup).fillna("unknown")
    df_cells["data_sufficiency"] = df_cells["paper_slug"].map(suf_lookup).fillna("unknown")

    groups = {"sufficient": "Sufficient data", "partial": "Partial data"}
    n_by_group = {k: (df_runs["data_sufficiency"] == k).sum() for k in groups}
    n_papers_by_group = {k: df_runs.loc[df_runs["data_sufficiency"] == k, "paper_slug"].nunique() for k in groups}
    print(f"  Papers: {n_papers_by_group}")
    print(f"  Runs: {n_by_group}")

    approaches = _approaches_in(df_runs)

    # ── 1. Grade distribution by approach, faceted by data sufficiency ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    # OLD: excluded F
    # grades_shown = [g for g in GRADE_ORDER if g != "F"]
    # NEW: exclude NA instead
    grades_shown = [g for g in GRADE_ORDER if g != "NA"]
    for ax_idx, (suf_key, suf_label) in enumerate(groups.items()):
        ax = axes[ax_idx]
        # OLD: excluded F
        # sub_items = df_items[(df_items["data_sufficiency"] == suf_key) & (df_items["grade"] != "F")
        # NEW: exclude NA instead
        sub_items = df_items[(df_items["data_sufficiency"] == suf_key) & (df_items["grade"] != "NA")
                              & (df_items["item_type"] == "table")]
        if sub_items.empty:
            continue
        ct = pd.crosstab(sub_items["approach"], sub_items["grade"], normalize="index") * 100
        ct = ct.reindex(columns=grades_shown, fill_value=0)
        present = [a for a in approaches if a in ct.index]
        ct.loc[present].plot(
            kind="bar", ax=ax,
            color=[GRADE_COLORS[g] for g in ct.columns],
            edgecolor="white", width=0.8, legend=False,
        )
        ax.set_xticklabels(
            [APPROACH_LABELS.get(a, a).replace("\n", " ") for a in present],
            fontsize=9, rotation=30, ha="right",
        )
        n_p = n_papers_by_group[suf_key]
        ax.set_xlabel("")
        apply_style(ax)
    axes[0].set_ylabel("Share of table items (%)", fontsize=13, fontweight="bold")
    place_legend(fig, axes[0], fontsize=11, ncol=6)
    plt.tight_layout()
    save_figure(fig, output_dir, "grade_distribution_by_sufficiency", subdir)

    # ── 2. Mean grade by approach × sufficiency (grouped bar) ──
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(approaches))
    w = 0.35
    for i, (suf_key, suf_label) in enumerate(groups.items()):
        means = []
        for a in approaches:
            sub = df_items[
                (df_items["approach"] == a) &
                (df_items["data_sufficiency"] == suf_key) &
                (df_items["item_type"] == "table") &
                # OLD: excluded F
                # (df_items["grade"] != "F")
                # NEW: exclude NA instead
                (df_items["grade"] != "NA")
            ]
            means.append(sub["grade_num"].mean() if not sub.empty else np.nan)
        offset = -w / 2 + i * w
        color = "#2ecc71" if suf_key == "sufficient" else "#e67e22"
        ax.bar(x + offset, means, w, label=suf_label, color=color, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [APPROACH_LABELS.get(a, a).replace("\n", " ") for a in approaches],
        fontsize=9, rotation=30, ha="right",
    )
    ax.set_ylabel("Mean table grade (A=5 → F=0)", fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.legend(fontsize=11)
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig, output_dir, "mean_grade_by_sufficiency", subdir)

    # ── 3. % A-B by approach × sufficiency (grouped bar) ──
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (suf_key, suf_label) in enumerate(groups.items()):
        pcts = []
        for a in approaches:
            sub = df_items[
                (df_items["approach"] == a) &
                (df_items["data_sufficiency"] == suf_key) &
                (df_items["item_type"] == "table") &
                # OLD: excluded F
                # (df_items["grade"] != "F")
                # NEW: exclude NA instead
                (df_items["grade"] != "NA")
            ]
            pcts.append(sub["grade"].isin(["A", "B"]).mean() * 100 if not sub.empty else np.nan)
        offset = -w / 2 + i * w
        color = "#2ecc71" if suf_key == "sufficient" else "#e67e22"
        ax.bar(x + offset, pcts, w, label=suf_label, color=color, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [APPROACH_LABELS.get(a, a).replace("\n", " ") for a in approaches],
        fontsize=9, rotation=30, ha="right",
    )
    ax.set_ylabel("Share grade A–B (%)", fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=11)
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig, output_dir, "pct_ab_by_sufficiency", subdir)

    # ── 4. Coefficient / SE CDF split by sufficiency ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax_idx, (suf_key, suf_label) in enumerate(groups.items()):
        ax = axes[ax_idx]
        sub = df_cells[
            (df_cells["data_sufficiency"] == suf_key) &
            (df_cells["row_type"] == "coefficient") &
            df_cells["original_value"].notna() &
            df_cells["replicated_value"].notna() &
            df_cells["is_numeric"] &
            # OLD: excluded F
            # (df_cells["item_grade"] != "F")
            # NEW: exclude NA instead
            (df_cells["item_grade"] != "NA")
        ].copy()
        if sub.empty:
            continue
        sub["abs_diff"] = (sub["original_value"].astype(float) - sub["replicated_value"].astype(float)).abs()
        sub["se"] = sub["original_se"]
        mask = sub["se"].isna()
        sub.loc[mask, "se"] = sub.loc[mask, "replicated_se"]
        sub = sub[sub["se"].notna() & (sub["se"].astype(float) > 0)].copy()
        sub["diff_over_se"] = sub["abs_diff"] / sub["se"].astype(float)

        x_max = 10
        for a in approaches:
            vals = sub.loc[sub["approach"] == a, "diff_over_se"].sort_values().values
            if len(vals) == 0:
                continue
            cdf_y = np.arange(1, len(vals) + 1) / len(vals) * 100
            color = APPROACH_COLORS.get(a, "#95a5a6")
            label = APPROACH_LABELS.get(a, a).replace("\n", " ") if ax_idx == 0 else None
            ax.plot(vals, cdf_y, color=color, linewidth=2, label=label)

        ax.axvline(x=1.96, color="red", linestyle="--", alpha=0.5)
        n_p = n_papers_by_group[suf_key]
        n_cells = len(sub)
        ax.set_xlim(0, x_max)
        ax.set_xlabel("|Coeff. diff.| / SE", fontsize=12, fontweight="bold")
        apply_style(ax)
    axes[0].set_ylabel("Cumulative share (%)", fontsize=12, fontweight="bold")
    place_legend(fig, axes[0], fontsize=9, ncol=4)
    plt.tight_layout()
    save_figure(fig, output_dir, "coefficient_se_cdf_by_sufficiency", subdir)

    # ── 5. Summary CSV ──
    rows = []
    for suf_key, suf_label in groups.items():
        for a in approaches:
            sub = df_items[
                (df_items["approach"] == a) &
                (df_items["data_sufficiency"] == suf_key) &
                (df_items["item_type"] == "table")
            ]
            # OLD: excluded F
            # non_f = sub[sub["grade"] != "F"]
            # NEW: exclude NA instead
            non_f = sub[sub["grade"] != "NA"]
            rows.append({
                "data_sufficiency": suf_key,
                "approach": APPROACH_LABELS.get(a, a).replace("\n", " "),
                "n_papers": sub["paper_slug"].nunique(),
                "n_tables": len(sub),
                "n_tables_excl_NA": len(non_f),
                "pct_A_B": f"{non_f['grade'].isin(['A', 'B']).mean() * 100:.1f}" if len(non_f) else "—",
                "mean_grade": f"{non_f['grade_num'].mean():.2f}" if len(non_f) else "—",
            })
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(target / "summary_by_sufficiency.csv", index=False)
    latex = summary_df.to_latex(index=False, escape=True,
                                 column_format="l" * 2 + "r" * (len(summary_df.columns) - 2))
    (target / "summary_by_sufficiency.tex").write_text(latex)
    print(f"  Saved {subdir}/summary_by_sufficiency (CSV + LaTeX)")


def _run_stability_analysis(
    df_items_main: pd.DataFrame,
    df_cells_main: pd.DataFrame,
    results_dir: Path,
    output_dir: Path,
    subdir: str,
    stability_dirs: list[Path],
    papers_dir: Path | None = None,
):
    """Compare the main run against additional stability runs on the same paper sample.

    For each (paper, approach, item) triple that appears in all runs, evaluate the
    stability of the grade across runs. Produces:
      - grade_consistency_table.csv: one row per (paper, approach, item) with the
        grade from each run + agreement flag
      - grade_agreement_rate.pdf: share of items where all runs agreed, by approach
      - grade_distribution_by_run.pdf: grade distribution per approach × run
      - cell_pct_diff_spread.pdf: for cells present in all runs, CDF of the run-to-run
        standard deviation of percent_difference (measures cell-level variance)
    """
    target = output_dir / subdir
    target.mkdir(parents=True, exist_ok=True)

    # Load each stability run as its own (items, cells) pair
    runs = [("run_0_main", df_items_main, df_cells_main)]
    for i, sdir in enumerate(stability_dirs, start=1):
        if not sdir.exists():
            print(f"  stability dir not found: {sdir}")
            continue
        try:
            _, di, dc = load_results(sdir, papers_dir)
            if di.empty:
                print(f"  stability dir empty: {sdir}")
                continue
            runs.append((f"run_{i}", di, dc))
        except Exception as e:
            print(f"  failed to load {sdir}: {e}")
    if len(runs) < 2:
        print("  Skipping run_stability: need at least 2 runs")
        return

    # Restrict main run to the papers that are in the stability sample
    stability_papers = set()
    for (_, di, _) in runs[1:]:
        stability_papers |= set(di["paper_slug"].unique())
    print(f"  Stability sample: {len(stability_papers)} papers × {len(runs)} runs")

    # ── 1. Build item-level comparison table ──────────────────────────────
    #    Key: (paper_slug, approach, item_id) → list of grades across runs
    records = []
    for run_name, di, _dc in runs:
        sub = di[
            di["paper_slug"].isin(stability_papers) &
            (di["item_type"] == "table")
        ].copy()
        for _, row in sub.iterrows():
            records.append({
                "run": run_name,
                "paper_slug": row["paper_slug"],
                "approach": row["approach"],
                "item_id": row["item_id"],
                "grade": row["grade"],
                "grade_num": row.get("grade_num"),
            })
    long_df = pd.DataFrame(records)

    # Pivot: rows = (paper, approach, item), cols = run, values = grade
    wide = long_df.pivot_table(
        index=["paper_slug", "approach", "item_id"],
        columns="run",
        values="grade",
        aggfunc="first",
    )
    run_cols = [c for c in wide.columns if c.startswith("run_")]
    # Keep rows where at least 2 runs produced a grade (handles split batches
    # where not all papers appear in all stability runs).
    wide["n_runs_present"] = wide[run_cols].notna().sum(axis=1)
    wide = wide[wide["n_runs_present"] >= 2].copy()
    # Agreement = all non-null values are the same
    wide["all_agree"] = wide[run_cols].apply(
        lambda r: r.dropna().nunique() == 1, axis=1
    )
    wide["unique_grades"] = wide[run_cols].apply(
        lambda r: r.dropna().nunique(), axis=1
    )
    wide.to_csv(target / "grade_consistency_table.csv")
    print(f"  Saved {subdir}/grade_consistency_table.csv ({len(wide)} items across runs)")

    # ── 2. Grade agreement rate per approach ──────────────────────────────
    if not wide.empty:
        agreement = wide.reset_index().groupby("approach")["all_agree"].agg(["mean", "sum", "count"])
        agreement.columns = ["agreement_rate", "n_agree", "n_total"]
        agreement["agreement_rate"] = (agreement["agreement_rate"] * 100).round(1)
        agreement.to_csv(target / "grade_agreement_rate.csv")

        fig, ax = plt.subplots(figsize=(8, 5))
        approaches_present = [a for a in _approaches_in(long_df) if a in agreement.index]
        y_pos = np.arange(len(approaches_present))
        for i, a in enumerate(approaches_present):
            rate = agreement.loc[a, "agreement_rate"]
            n_total = int(agreement.loc[a, "n_total"])
            color = APPROACH_COLORS.get(a, "#95a5a6")
            ax.barh(i, rate, color=color, edgecolor="white")
            ax.text(rate + 1, i, f"{rate:.1f}% (n={n_total})",
                    va="center", fontsize=10)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([APPROACH_LABELS.get(a, a).replace("\n", " ") for a in approaches_present], fontsize=10)
        ax.invert_yaxis()
        ax.set_xlim(0, 105)
        ax.set_xlabel("Share of items with same grade across all runs (%)", fontsize=12, fontweight="bold")
        apply_style(ax)
        plt.tight_layout()
        save_figure(fig, output_dir, "grade_agreement_rate", subdir)

    # ── 3. Grade distribution per approach × run ──────────────────────────
    approaches_present = _approaches_in(long_df)
    grades_shown = [g for g in GRADE_ORDER if g != "NA"]
    n_app = len(approaches_present)
    fig, axes = plt.subplots(1, n_app, figsize=(4.5 * n_app, 5), sharey=True)
    if n_app == 1:
        axes = [axes]

    for ax_idx, approach in enumerate(approaches_present):
        ax = axes[ax_idx]
        sub = long_df[long_df["approach"] == approach]
        if sub.empty:
            continue
        ct = pd.crosstab(sub["run"], sub["grade"], normalize="index") * 100
        ct = ct.reindex(columns=grades_shown, fill_value=0)
        runs_present = [r for r in sorted(ct.index) if r in ct.index]
        ct.loc[runs_present].plot(
            kind="bar", ax=ax,
            color=[GRADE_COLORS.get(g, "#95a5a6") for g in ct.columns],
            edgecolor="white", width=0.8, legend=(ax_idx == 0),
        )
        ax.set_xticklabels([r.replace("_", " ") for r in runs_present], fontsize=9, rotation=30, ha="right")
        ax.set_xlabel("")
        apply_style(ax)
    axes[0].set_ylabel("Share of tables (%)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, output_dir, "grade_distribution_by_run", subdir)

    # ── 4a. Table-level grade spread ──────────────────────────────────────
    # For each (paper, approach, item), compute max-min of grade_num across runs
    GRADE_NUM = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1, "F": 0}
    table_spreads = wide.copy()
    for col in run_cols:
        table_spreads[f"{col}_num"] = table_spreads[col].map(GRADE_NUM)
    num_cols = [f"{c}_num" for c in run_cols]
    table_spreads["grade_range"] = table_spreads[num_cols].max(axis=1) - table_spreads[num_cols].min(axis=1)
    table_spreads = table_spreads.reset_index()

    # Plot distribution of grade_range per approach (stacked bar).
    # Shared color scale across table and paper spread plots, keyed to the
    # theoretical maximum range (A↔F = 5). Value i always maps to the same
    # color in both plots.
    SHARED_MAX_RANGE = 5
    SHARED_RANGE_COLORS = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, SHARED_MAX_RANGE + 1))

    approaches_present = _approaches_in(table_spreads)
    max_range = int(table_spreads["grade_range"].dropna().max()) if not table_spreads["grade_range"].dropna().empty else 0
    ranges_shown = list(range(max_range + 1))

    fig, ax = plt.subplots(figsize=(9, 5))
    ct = pd.crosstab(table_spreads["approach"], table_spreads["grade_range"], normalize="index") * 100
    ct = ct.reindex(columns=ranges_shown, fill_value=0)
    present = [a for a in approaches_present if a in ct.index]
    range_colors = SHARED_RANGE_COLORS[:len(ranges_shown)]
    ct.loc[present].plot(
        kind="bar", stacked=True, ax=ax,
        color=range_colors, edgecolor="white", width=0.6,
    )
    ax.set_xticklabels([APPROACH_LABELS.get(a, a).replace("\n", " ") for a in present],
                        fontsize=10, rotation=20, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("Share of tables (%)", fontsize=12, fontweight="bold")
    handles, labels = ax.get_legend_handles_labels()
    labels = [f"{l} (all agree)" if l == "0" else f"spread = {l}" for l in labels]
    ax.legend(handles, labels, fontsize=9, bbox_to_anchor=(1.02, 1), loc="upper left",
              title="grade range")
    ax.set_ylim(0, 100)
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig, output_dir, "table_grade_spread", subdir)

    # ── 4b. Paper-level grade spread ──────────────────────────────────────
    # For each (paper, approach), recompute the overall paper grade per run
    # from the table grades in that run, then measure spread across runs.
    paper_grades = []
    for run_name, di, _dc in runs:
        sub = di[
            di["paper_slug"].isin(stability_papers) &
            (di["item_type"] == "table") &
            (di["grade"] != "NA")
        ].copy()
        sub["grade_num"] = sub["grade"].map(GRADE_NUM)
        # Group by (paper, approach) → mean grade_num → overall letter grade
        grp = sub.groupby(["paper_slug", "approach"], observed=True)["grade_num"].mean().reset_index()
        def _num_to_grade(avg):
            if avg >= 4.5: return "A"
            if avg >= 3.5: return "B"
            if avg >= 2.5: return "C"
            if avg >= 1.5: return "D"
            if avg >= 0.5: return "E"
            return "F"
        grp["overall_grade"] = grp["grade_num"].apply(_num_to_grade)
        grp["run"] = run_name
        paper_grades.append(grp)
    paper_df = pd.concat(paper_grades, ignore_index=True)

    paper_wide = paper_df.pivot_table(
        index=["paper_slug", "approach"],
        columns="run",
        values="grade_num",
        aggfunc="first",
    )
    prun_cols = [c for c in paper_wide.columns if c.startswith("run_")]
    paper_wide["n_runs_present"] = paper_wide[prun_cols].notna().sum(axis=1)
    paper_wide = paper_wide[paper_wide["n_runs_present"] >= 2].copy()
    paper_wide["grade_range"] = paper_wide[prun_cols].max(axis=1) - paper_wide[prun_cols].min(axis=1)
    paper_wide = paper_wide.reset_index()

    # Also add the letter-grade form per run for inspection
    for col in prun_cols:
        paper_wide[f"{col}_letter"] = paper_wide[col].apply(
            lambda x: _num_to_grade(x) if pd.notna(x) else ""
        )
    paper_wide.to_csv(target / "paper_grade_consistency.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    # Use 0.5-step bins since paper-level averages are continuous
    bins = np.arange(0, paper_wide["grade_range"].max() + 0.5, 0.5)
    approaches_p = _approaches_in(paper_wide)
    for approach in approaches_p:
        vals = paper_wide.loc[paper_wide["approach"] == approach, "grade_range"]
        if vals.empty:
            continue
        color = APPROACH_COLORS.get(approach, "#95a5a6")
        label = APPROACH_LABELS.get(approach, approach).replace("\n", " ")
        ax.hist(vals, bins=bins, alpha=0.5, color=color, label=f"{label} (n={len(vals)})",
                edgecolor="white")
    ax.set_xlabel("Paper-level grade spread (max − min of mean grade across runs)",
                  fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of papers", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig, output_dir, "paper_grade_spread", subdir)

    # ── 4c. Paper-level grade spread on DISCRETE letter grades ───────────
    # Round each run's mean grade to a letter, then spread = max_letter_num - min_letter_num
    GRADE_VAL = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1, "F": 0}
    for col in prun_cols:
        paper_wide[f"{col}_letter_num"] = paper_wide[f"{col}_letter"].map(GRADE_VAL)
    letter_num_cols = [f"{c}_letter_num" for c in prun_cols]
    paper_wide["letter_grade_range"] = paper_wide[letter_num_cols].max(axis=1) - paper_wide[letter_num_cols].min(axis=1)
    paper_wide.to_csv(target / "paper_grade_consistency.csv", index=False)

    max_range = int(paper_wide["letter_grade_range"].dropna().max()) if not paper_wide["letter_grade_range"].dropna().empty else 0
    ranges_shown = list(range(max_range + 1))

    fig, ax = plt.subplots(figsize=(9, 5))
    ct = pd.crosstab(paper_wide["approach"], paper_wide["letter_grade_range"].dropna(), normalize="index") * 100
    ct = ct.reindex(columns=ranges_shown, fill_value=0)
    approaches_p_present = [a for a in _approaches_in(paper_wide) if a in ct.index]
    range_colors = SHARED_RANGE_COLORS[:len(ranges_shown)]
    ct.loc[approaches_p_present].plot(
        kind="bar", stacked=True, ax=ax,
        color=range_colors, edgecolor="white", width=0.6,
    )
    ax.set_xticklabels([APPROACH_LABELS.get(a, a).replace("\n", " ") for a in approaches_p_present],
                        fontsize=10, rotation=20, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("Share of papers (%)", fontsize=12, fontweight="bold")
    handles, labels = ax.get_legend_handles_labels()
    labels = [f"{l} (all agree)" if l == "0" else f"spread = {l}" for l in labels]
    ax.legend(handles, labels, fontsize=9, bbox_to_anchor=(1.02, 1), loc="upper left",
              title="grade range")
    ax.set_ylim(0, 100)
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig, output_dir, "paper_grade_spread_discrete", subdir)
    print(f"  Saved {subdir}/table_grade_spread and paper_grade_spread (continuous + discrete)")

    # ── 5. Cell-level variance: run-to-run spread of percent_difference ──
    # Build a long df of cell pct_diff across runs, keyed on (paper, approach, item, row, col)
    cell_records = []
    for run_name, _di, dc in runs:
        sub = dc[
            dc["paper_slug"].isin(stability_papers) &
            dc["percent_difference"].notna() &
            dc["is_numeric"]
        ].copy()
        for _, row in sub.iterrows():
            cell_records.append({
                "run": run_name,
                "paper_slug": row["paper_slug"],
                "approach": row["approach"],
                "item_id": row["item_id"],
                "row_label": row.get("row_label", ""),
                "column_label": row.get("column_label", ""),
                "row_type": row.get("row_type", ""),
                "pct_diff": row["percent_difference"],
            })
    if cell_records:
        cell_long = pd.DataFrame(cell_records)
        cell_wide = cell_long.pivot_table(
            index=["paper_slug", "approach", "item_id", "row_label", "column_label", "row_type"],
            columns="run",
            values="pct_diff",
            aggfunc="first",
        )
        run_cols = [c for c in cell_wide.columns if c.startswith("run_")]
        cell_wide["n_runs_present"] = cell_wide[run_cols].notna().sum(axis=1)
        cell_wide = cell_wide[cell_wide["n_runs_present"] >= 2].copy()

        if not cell_wide.empty:
            # Row-wise std (if only 2 runs, this is |a-b|/sqrt(2); if 3, proper std)
            cell_wide["std"] = cell_wide[run_cols].std(axis=1, ddof=0)
            cell_wide["mean"] = cell_wide[run_cols].mean(axis=1)
            cell_wide.to_csv(target / "cell_pct_diff_spread.csv", escapechar="\\")

            # CDF of row-wise std, per approach (coefficients only)
            coef = cell_wide.reset_index()
            coef = coef[coef["row_type"] == "coefficient"]
            if not coef.empty:
                fig, ax = plt.subplots(figsize=(8, 5))
                for approach in _approaches_in(coef):
                    vals = coef.loc[coef["approach"] == approach, "std"].dropna().sort_values()
                    if len(vals) < 5:
                        continue
                    color = APPROACH_COLORS.get(approach, "#95a5a6")
                    label = APPROACH_LABELS.get(approach, approach).replace("\n", " ")
                    cdf_y = np.arange(1, len(vals) + 1) / len(vals) * 100
                    ax.plot(vals, cdf_y, color=color, linewidth=2, label=label)
                ax.set_xlim(0, 100)
                ax.set_ylim(0, 105)
                ax.set_xlabel("Run-to-run std dev of |% difference| (coefficient cells)", fontsize=11, fontweight="bold")
                ax.set_ylabel("Cumulative share of cells (%)", fontsize=11, fontweight="bold")
                ax.legend(fontsize=10)
                apply_style(ax)
                plt.tight_layout()
                save_figure(fig, output_dir, "cell_pct_diff_spread_cdf", subdir)

            # Summary stats
            summary = coef.groupby("approach")["std"].agg(["mean", "median", "count"]).round(2)
            summary.to_csv(target / "cell_pct_diff_spread_summary.csv")
            print(f"  Saved {subdir}/cell_pct_diff_spread (CSV + CDF PDF)")

            # ── 6a. Option 1: CDF of |run_i - run_j| pairwise pct_diff ──
            # For each cell, use MAX pairwise absolute difference across all run pairs.
            # With 2 runs this is just |a-b|; with 3 runs it's max over the 3 pairs.
            if not coef.empty and len(run_cols) >= 2:
                fig, ax = plt.subplots(figsize=(8, 5))
                # Compute max pairwise absolute diff per cell
                from itertools import combinations
                coef_pairs = coef.copy()
                diffs = []
                for r1, r2 in combinations(run_cols, 2):
                    diffs.append((coef_pairs[r1] - coef_pairs[r2]).abs())
                coef_pairs["pairwise_max_diff"] = np.maximum.reduce(diffs) if len(diffs) > 1 else diffs[0]
                for approach in _approaches_in(coef_pairs):
                    vals = coef_pairs.loc[coef_pairs["approach"] == approach, "pairwise_max_diff"].dropna().sort_values()
                    if len(vals) < 5:
                        continue
                    color = APPROACH_COLORS.get(approach, "#95a5a6")
                    label = APPROACH_LABELS.get(approach, approach).replace("\n", " ")
                    cdf_y = np.arange(1, len(vals) + 1) / len(vals) * 100
                    ax.plot(vals, cdf_y, color=color, linewidth=2, label=label)
                ax.set_xlim(0, 100)
                ax.set_ylim(0, 105)
                ax.set_xlabel("Max run-to-run change in |% difference| (pp)", fontsize=11, fontweight="bold")
                ax.set_ylabel("Cumulative share of cells (%)", fontsize=11, fontweight="bold")
                ax.legend(fontsize=10)
                apply_style(ax)
                plt.tight_layout()
                save_figure(fig, output_dir, "cell_pct_diff_pairwise_max_cdf", subdir)

            # ── 6b. Option 5: CDF of max-min pct_diff range across runs ──
            if not coef.empty and len(run_cols) >= 2:
                fig, ax = plt.subplots(figsize=(8, 5))
                coef_range = coef.copy()
                coef_range["range"] = coef_range[run_cols].max(axis=1) - coef_range[run_cols].min(axis=1)
                for approach in _approaches_in(coef_range):
                    vals = coef_range.loc[coef_range["approach"] == approach, "range"].dropna().sort_values()
                    if len(vals) < 5:
                        continue
                    color = APPROACH_COLORS.get(approach, "#95a5a6")
                    label = APPROACH_LABELS.get(approach, approach).replace("\n", " ")
                    cdf_y = np.arange(1, len(vals) + 1) / len(vals) * 100
                    ax.plot(vals, cdf_y, color=color, linewidth=2, label=label)
                ax.set_xlim(0, 100)
                ax.set_ylim(0, 105)
                ax.set_xlabel("Range of |% difference| across runs (max − min, pp)", fontsize=11, fontweight="bold")
                ax.set_ylabel("Cumulative share of cells (%)", fontsize=11, fontweight="bold")
                ax.legend(fontsize=10)
                apply_style(ax)
                plt.tight_layout()
                save_figure(fig, output_dir, "cell_pct_diff_range_cdf", subdir)

            # ── 6b'. CDF of mean pct_diff across runs (accuracy, not stability) ──
            if not coef.empty and len(run_cols) >= 2:
                fig, ax = plt.subplots(figsize=(8, 5))
                coef_mean = coef.copy()
                coef_mean["mean_pct_diff"] = coef_mean[run_cols].mean(axis=1)
                for approach in _approaches_in(coef_mean):
                    vals = coef_mean.loc[coef_mean["approach"] == approach, "mean_pct_diff"].dropna().sort_values()
                    if len(vals) < 5:
                        continue
                    color = APPROACH_COLORS.get(approach, "#95a5a6")
                    label = APPROACH_LABELS.get(approach, approach).replace("\n", " ")
                    cdf_y = np.arange(1, len(vals) + 1) / len(vals) * 100
                    ax.plot(vals, cdf_y, color=color, linewidth=2, label=label)
                ax.set_xlim(0, 100)
                ax.set_ylim(0, 105)
                ax.set_xlabel("Mean |% difference| across runs (pp)", fontsize=11, fontweight="bold")
                ax.set_ylabel("Cumulative share of cells (%)", fontsize=11, fontweight="bold")
                ax.legend(fontsize=10)
                apply_style(ax)
                plt.tight_layout()
                save_figure(fig, output_dir, "cell_pct_diff_mean_cdf", subdir)

            # ── 6b0. CDF of pct_diff range — cells with ≥2 runs (pairwise) ──
            # Relax the "all 3 runs" requirement: include cells present in at least 2 runs.
            # For cells in 2 runs, range = |a-b|. For cells in 3 runs, range = max-min.
            if cell_records and len(run_cols) >= 2:
                coef_long_all = cell_long[cell_long["row_type"] == "coefficient"].copy()
                if not coef_long_all.empty:
                    key_cols = ["paper_slug", "approach", "item_id", "row_label", "column_label"]
                    # Pivot across runs
                    cw = coef_long_all.pivot_table(
                        index=key_cols + ["row_type"],
                        columns="run",
                        values="pct_diff",
                        aggfunc="first",
                    )
                    cw_run_cols = [c for c in cw.columns if c.startswith("run_")]
                    # Keep rows with at least 2 non-null values
                    cw = cw[cw[cw_run_cols].notna().sum(axis=1) >= 2].copy()
                    cw["range"] = cw[cw_run_cols].max(axis=1) - cw[cw_run_cols].min(axis=1)
                    cw["n_runs"] = cw[cw_run_cols].notna().sum(axis=1)
                    cw = cw.reset_index()

                    fig, ax = plt.subplots(figsize=(8, 5))
                    for approach in _approaches_in(cw):
                        vals = cw.loc[cw["approach"] == approach, "range"].dropna().sort_values().values
                        if len(vals) < 5:
                            continue
                        cdf_y = np.arange(1, len(vals) + 1) / len(vals) * 100
                        color = APPROACH_COLORS.get(approach, "#95a5a6")
                        label = APPROACH_LABELS.get(approach, approach).replace(chr(10), ' ')
                        ax.plot(vals, cdf_y, color=color, linewidth=2, label=label)
                    ax.set_xlim(0, 100)
                    ax.set_ylim(0, 105)
                    ax.set_xlabel("Range of |% difference| across runs (max − min, pp)", fontsize=11, fontweight="bold")
                    ax.set_ylabel("Cumulative share of cells (%)", fontsize=11, fontweight="bold")
                    ax.legend(fontsize=9)
                    apply_style(ax)
                    plt.tight_layout()
                    save_figure(fig, output_dir, "cell_pct_diff_range_cdf_min2runs", subdir)

            # ── 6b'. CDF of max-min range, with missing cells in denominator ─
            # For each (paper, approach, item, row, col, row_type) attempted in ANY run,
            # show the CDF of pct_diff range. Cells present in all runs appear at their
            # actual range; cells missing in ≥1 run are censored at +∞ (never contribute
            # to the numerator). The curve plateaus at (# cells in all runs) / (# attempted).
            if cell_records and len(run_cols) >= 2:
                # cell_long has one row per (cell key, run) with a pct_diff
                # Attempted = unique cell keys (regardless of how many runs)
                coef_long = cell_long[cell_long["row_type"] == "coefficient"].copy()
                if not coef_long.empty:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    key_cols = ["paper_slug", "approach", "item_id", "row_label", "column_label"]
                    # Total attempted per approach (unique cell keys)
                    attempted = coef_long.groupby("approach")[key_cols[0]].count()  # placeholder
                    # Compute using the wide version we already have (ranges available for survivors)
                    coef_range_denom = coef.copy()
                    coef_range_denom["range"] = coef_range_denom[run_cols].max(axis=1) - coef_range_denom[run_cols].min(axis=1)
                    for approach in _approaches_in(coef_long):
                        # Total attempted = unique cell keys across all runs for this approach
                        n_attempted = coef_long[coef_long["approach"] == approach].drop_duplicates(key_cols).shape[0]
                        vals = coef_range_denom.loc[coef_range_denom["approach"] == approach, "range"].dropna().sort_values().values
                        if n_attempted < 5 or len(vals) == 0:
                            continue
                        # Numerator: cumulative count; denominator: total attempted
                        cdf_y = np.arange(1, len(vals) + 1) / n_attempted * 100
                        color = APPROACH_COLORS.get(approach, "#95a5a6")
                        label = APPROACH_LABELS.get(approach, approach).replace("\n", " ")
                        plateau = len(vals) / n_attempted * 100
                        full_label = f"{label} (n_attempted={n_attempted}, plateau={plateau:.0f}%)"
                        ax.plot(vals, cdf_y, color=color, linewidth=2, label=full_label)
                        # Mark plateau with a dashed horizontal line
                        ax.axhline(plateau, color=color, linestyle=":", alpha=0.35, linewidth=1)
                    ax.set_xlim(0, 100)
                    ax.set_ylim(0, 105)
                    ax.set_xlabel("Range of |% difference| across runs (max − min, pp)", fontsize=11, fontweight="bold")
                    ax.set_ylabel("Share of attempted cells (%)", fontsize=11, fontweight="bold")
                    ax.legend(fontsize=9)
                    apply_style(ax)
                    plt.tight_layout()
                    save_figure(fig, output_dir, "cell_pct_diff_range_cdf_with_missing", subdir)

    # ── 6c. Per-run coefficient_se_cdf: one line per (approach, run) ────
    # Like the paper's coefficient_se_cdf but with a separate line per run.
    # Restricted to the two approaches that were re-run in the stability test.
    # Merge stability runs pairwise: run_1+run_3 → "stability run 1",
    # run_2+run_4 → "stability run 2". Main run stays as-is.
    # This pools batch A and batch B into combined 20-paper runs.
    STABILITY_APPROACHES = {"claude-code/claude-opus-4-6", "codex/gpt-5.4"}
    merged_runs = []  # list of (label, combined_dc)
    # Main run (run_0)
    if runs:
        merged_runs.append(("main run", runs[0][2]))
    # Pair odd+even stability runs: (run_1, run_3) → stability 1, (run_2, run_4) → stability 2
    stab_pairs = {}  # label → list of dc frames
    for i, (run_name, _di, dc) in enumerate(runs[1:], start=1):
        # Runs 1,3,5... → "stability run 1"; runs 2,4,6... → "stability run 2"
        pair_id = ((i - 1) % 2) + 1
        label = f"stability run {pair_id}"
        stab_pairs.setdefault(label, []).append(dc)
    for label in sorted(stab_pairs):
        combined = pd.concat(stab_pairs[label], ignore_index=True)
        merged_runs.append((label, combined))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    linestyles = ["-", "--", ":"]
    for run_idx, (run_label, dc) in enumerate(merged_runs):
        sub = dc[
            dc["paper_slug"].isin(stability_papers) &
            dc["approach"].astype(str).isin(STABILITY_APPROACHES) &
            (dc["row_type"] == "coefficient") &
            dc["original_value"].notna() &
            dc["replicated_value"].notna() &
            dc["is_numeric"] &
            (dc["item_grade"] != "F")
        ].copy()
        if sub.empty:
            continue
        sub["abs_diff"] = (sub["original_value"].astype(float) - sub["replicated_value"].astype(float)).abs()
        sub["se"] = sub["original_se"]
        mask = sub["se"].isna()
        sub.loc[mask, "se"] = sub.loc[mask, "replicated_se"]
        sub = sub[sub["se"].notna() & (sub["se"].astype(float) > 0)].copy()
        if sub.empty:
            continue
        sub["diff_over_se"] = sub["abs_diff"] / sub["se"].astype(float)
        linestyle = linestyles[run_idx % len(linestyles)]
        for approach in _approaches_in(sub):
            vals = sub.loc[sub["approach"] == approach, "diff_over_se"].dropna().sort_values().values
            if len(vals) < 5:
                continue
            cdf_y = np.arange(1, len(vals) + 1) / len(vals) * 100
            color = APPROACH_COLORS.get(approach, "#95a5a6")
            label = f"{APPROACH_LABELS.get(approach, approach).replace(chr(10), ' ')} ({run_label})"
            ax.plot(vals, cdf_y, color=color, linewidth=1.8, linestyle=linestyle, label=label)
    ax.axvline(x=1.96, color="red", linestyle="--", alpha=0.4)
    ax.text(2.05, 5, "1.96", color="red", fontsize=10, alpha=0.7)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 105)
    ax.set_xlabel("|Coeff. difference| / SE", fontsize=12, fontweight="bold")
    ax.set_ylabel("Cumulative share of coefficients (%)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, ncol=1, bbox_to_anchor=(1.02, 1), loc="upper left")
    apply_style(ax)
    plt.tight_layout()
    save_figure(fig, output_dir, "coefficient_se_cdf_by_run", subdir)
    print(f"  Saved {subdir}/coefficient_se_cdf_by_run")

    # ── 6d. Histogram of BETWEEN-RUN coefficient differences (in SE units) ─
    # For every coefficient cell with ≥2 runs producing a replicated value,
    # compute every pairwise |rep_i − rep_j| / SE and bin as a histogram.
    # Measures within-agent run-to-run instability, normalised by the
    # coefficient's SE (from the original paper, with replicated_se as fallback).
    STABILITY_APPROACH_LIST = ["claude-code/claude-opus-4-6", "codex/gpt-5.4"]
    inter_run_rows = []
    for approach in STABILITY_APPROACH_LIST:
        per_run = []
        for run_name, _di, dc in runs:
            sub = dc[
                dc["paper_slug"].isin(stability_papers) &
                (dc["approach"].astype(str) == approach) &
                (dc["row_type"] == "coefficient") &
                dc["replicated_value"].notna() &
                dc["is_numeric"] &
                (dc["item_grade"] != "F")
            ][[
                "paper_slug", "item_id", "row_label", "column_label",
                "replicated_value", "original_se", "replicated_se",
            ]].copy()
            sub["run"] = run_name
            per_run.append(sub)
        if not per_run:
            continue
        long = pd.concat(per_run, ignore_index=True)
        # Use ONLY the paper's published SE as the denominator. If the extractor
        # did not capture original_se for a cell (~28% of coefficients), drop
        # the cell: an agent's replicated_se is not a stable ground-truth scale.
        long = long[long["original_se"].notna() & (long["original_se"].astype(float) > 0)].copy()
        long["se"] = long["original_se"].astype(float)
        # Compute all pairwise |Δrep|/SE per cell key
        key = ["paper_slug", "item_id", "row_label", "column_label"]
        for key_vals, g in long.groupby(key, dropna=False):
            if len(g) < 2:
                continue
            vals = g["replicated_value"].astype(float).values
            runs_for = g["run"].tolist()
            se = float(g["se"].iloc[0])
            paper, item, row_lbl, col_lbl = key_vals
            for i in range(len(vals)):
                for j in range(i + 1, len(vals)):
                    inter_run_rows.append({
                        "approach": approach,
                        "paper": paper,
                        "item": item,
                        "row_label": row_lbl,
                        "column_label": col_lbl,
                        "run_i": runs_for[i],
                        "run_j": runs_for[j],
                        "rep_i": vals[i],
                        "rep_j": vals[j],
                        "se": se,
                        "diff_over_se": abs(vals[i] - vals[j]) / se,
                    })

    if inter_run_rows:
        df_ir = pd.DataFrame(inter_run_rows)
        df_ir.to_csv(target / "coefficient_se_between_runs.csv",
                     index=False, escapechar="\\")

        # Log-spaced bins over the full distribution. Floor zero / tiny values
        # at 1e-4 so they're visible on the log axis; the first bin then
        # represents "essentially identical across runs".
        lo, hi = 1e-4, 1e6
        bins = np.logspace(np.log10(lo), np.log10(hi), 61)
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        for approach in STABILITY_APPROACH_LIST:
            vals = df_ir.loc[df_ir["approach"] == approach, "diff_over_se"].values
            if len(vals) < 5:
                continue
            color = APPROACH_COLORS.get(approach, "#95a5a6")
            vals_plot = np.clip(vals, lo, hi)
            ax.hist(vals_plot, bins=bins, color=color, alpha=0.55,
                    edgecolor=color, linewidth=0.6)
        ax.axvline(1.96, color="red", linestyle="--", alpha=0.4)
        ax.text(2.2, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 1,
                "1.96", color="red", fontsize=10, alpha=0.7, va="top")
        ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        ax.set_xlabel("|Δ reproduced coefficient| / SE (between-run, log scale)",
                      fontsize=12, fontweight="bold")
        ax.set_ylabel("Count of pairwise run comparisons",
                      fontsize=12, fontweight="bold")
        apply_style(ax)
        plt.tight_layout()
        save_figure(fig, output_dir, "coefficient_se_hist_between_runs", subdir)
        print(f"  Saved {subdir}/coefficient_se_hist_between_runs")


# ============================================================================
# Rounded regrading: recompute cell/table/paper grades using adaptively-rounded
# values so that e.g. a reported p-value of 0.01 and a replicated 0.00521345
# both collapse to 0.01 before grading. Mirrors src/benchmark/grader.py but
# vectorised and applied to the *_rounded columns produced by
# apply_adaptive_rounding.
# ============================================================================


_NEAR_ZERO_THRESHOLDS_ROUNDED = [(0.002, "A"), (0.02, "B"), (0.05, "C"), (0.1, "D")]


def _grade_cells_rounded(df_cells: pd.DataFrame) -> np.ndarray:
    """Apply the cell-grading rules (grader.grade_cell) to the *_rounded cols."""
    orig = df_cells["original_value_rounded"].to_numpy()
    repl = df_cells["replicated_value_rounded"].to_numpy()
    abs_d = df_cells["absolute_difference_rounded"].to_numpy()
    pct = df_cells["percent_difference_rounded"].to_numpy()  # unsigned

    grades = np.full(len(df_cells), "F", dtype=object)
    missing = np.isnan(orig) | np.isnan(repl)
    both_zero = ~missing & (orig == 0) & (repl == 0)
    sign_mis = ~missing & ~both_zero & (orig != 0) & (repl != 0) & (np.sign(orig) != np.sign(repl))
    near_zero = ~missing & ~both_zero & ~sign_mis & (np.abs(orig) < 0.001)
    std = ~missing & ~both_zero & ~sign_mis & ~near_zero

    grades[both_zero] = "A"
    grades[sign_mis] = "E"

    # Near-zero: absolute-difference thresholds
    for thr, g in _NEAR_ZERO_THRESHOLDS_ROUNDED:
        sel = near_zero & (abs_d < thr) & (grades == "F")
        grades[sel] = g
    grades[near_zero & (grades == "F")] = "E"

    # Standard: percent-difference thresholds
    std_have_pct = std & ~np.isnan(pct)
    grades[std_have_pct & (pct < 2)] = "A"
    grades[std_have_pct & (pct >= 2) & (pct < 20)] = "B"
    grades[std_have_pct & (pct >= 20) & (pct < 40)] = "C"
    grades[std_have_pct & (pct >= 40) & (pct < 60)] = "D"
    grades[std_have_pct & (pct >= 60)] = "E"
    return grades


def _avg_to_grade(avg: float) -> str:
    if pd.isna(avg):
        return "NA"
    if avg >= 4.5: return "A"
    if avg >= 3.5: return "B"
    if avg >= 2.5: return "C"
    if avg >= 1.5: return "D"
    if avg >= 0.5: return "E"
    return "F"


def apply_rounded_regrading(df_runs: pd.DataFrame, df_items: pd.DataFrame,
                             df_cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute the cell-level rounded grade column.

    Adds ``df_cells["cell_grade_rounded"]``: grade assigned by
    :func:`_grade_cells_rounded` (grader.grade_cell rules applied to the
    ``*_rounded`` numeric columns). Table-level and paper-level rollups are
    *not* produced here — they come from :func:`apply_mode_grades` called
    with ``cell_col="cell_grade_rounded"``, so the rounded grades follow the
    same per-mode aggregation semantics as the base grades.
    """
    if df_cells.empty:
        return df_runs, df_items, df_cells
    df_cells = df_cells.copy()
    df_cells["cell_grade_rounded"] = _grade_cells_rounded(df_cells)
    # If regrade-na was applied to the base column, mirror it on the rounded
    # column so missing-original cells become NA in both.
    if "cell_grade" in df_cells.columns:
        mask = (df_cells["cell_grade"].astype(object) == "NA")
        df_cells.loc[mask, "cell_grade_rounded"] = "NA"
    present = [g for g in GRADE_ORDER if g in df_cells["cell_grade_rounded"].unique()]
    df_cells["cell_grade_rounded"] = pd.Categorical(
        df_cells["cell_grade_rounded"], categories=present, ordered=True
    )
    return df_runs, df_items, df_cells


def regrade_with_na(df_runs: pd.DataFrame, df_items: pd.DataFrame,
                     df_cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Relabel cells graded F with no original_value as NA (unassessable).

    This is purely a **cell-level relabel**; it does not recompute table or
    paper grades. Those are computed per-f_mode by :func:`apply_mode_grades`
    downstream using the relabelled cells.

    - F cell with original_value missing → NA (extractor couldn't extract it).
    - F cell with original_value present → stays F (real failure).
    - A–E cells are untouched.
    """
    if df_cells.empty:
        return df_runs, df_items, df_cells
    df_cells = df_cells.copy()
    mask_na = (df_cells["cell_grade"] == "F") & df_cells["original_value"].isna()
    n_regraded = int(mask_na.sum())
    df_cells.loc[mask_na, "cell_grade"] = "NA"
    print(f"  regrade-na: relabelled {n_regraded} cells F → NA (missing original)")

    # Refresh categorical so "NA" is an allowed category.
    present = [g for g in GRADE_ORDER if g in df_cells["cell_grade"].unique()]
    df_cells["cell_grade"] = pd.Categorical(df_cells["cell_grade"], categories=present, ordered=True)
    return df_runs, df_items, df_cells


# ============================================================================
# Mode-aware grade aggregation (user spec — see docstring on apply_mode_grades)
# ============================================================================


_GRADE_VAL = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1, "F": 0}


def _cells_for_mode(df_cells: pd.DataFrame, f_mode: str,
                    cell_col: str = "cell_grade") -> pd.DataFrame:
    """Return the subset of ``df_cells`` that counts toward grading under ``f_mode``.

    - ``no_f``: drop cells with grade F or NA.
    - ``all_f``: drop cells with grade NA.
    - ``at_least_one_non_f``: drop NA cells; within ``(paper_slug, item_id,
      row_label, column_label)`` groups (a single cell position across all
      approaches), drop every cell in groups where every approach is F. Cells
      in groups with at least one non-F survive (including the F ones).
    """
    if df_cells.empty or cell_col not in df_cells.columns:
        return df_cells.iloc[0:0].copy()
    cg = df_cells[cell_col].astype(object)
    if f_mode == "no_f":
        return df_cells[~cg.isin(["F", "NA"])].copy()
    if f_mode == "all_f":
        return df_cells[cg != "NA"].copy()
    if f_mode == "at_least_one_non_f":
        out = df_cells[cg != "NA"].copy()
        keys = ["paper_slug", "item_id", "row_label", "column_label"]
        grp = out.groupby(keys, observed=True, dropna=False)[cell_col]
        total = grp.transform("size")
        n_f = grp.transform(lambda s: (s == "F").sum())
        all_f = (total > 0) & (total == n_f)
        return out[~all_f].copy()
    raise ValueError(f"Unknown f_mode: {f_mode}")


def _avg_series_to_grade(s: pd.Series) -> str:
    """Mean of a grade letter series → letter grade. Empty → 'NA'."""
    v = s.map(_GRADE_VAL).dropna()
    if v.empty:
        return "NA"
    return _avg_to_grade(v.mean())


def _tables_for_mode(df_items: pd.DataFrame, grade_col: str, f_mode: str) -> pd.DataFrame:
    """Drop tables that don't count toward paper grading under ``f_mode``.

    Only operates on rows with ``item_type == "table"``. Uses the given
    ``grade_col`` (the mode-specific table grade column).
    """
    if df_items.empty:
        return df_items.iloc[0:0].copy()
    tbl = df_items[df_items["item_type"] == "table"].copy()
    g = tbl[grade_col].astype(object)
    if f_mode == "no_f":
        return tbl[~g.isin(["F", "NA"])].copy()
    if f_mode == "all_f":
        return tbl[g != "NA"].copy()
    if f_mode == "at_least_one_non_f":
        tbl = tbl[g != "NA"].copy()
        keys = ["paper_slug", "item_id"]
        grp = tbl.groupby(keys, observed=True, dropna=False)[grade_col]
        total = grp.transform("size")
        n_f = grp.transform(lambda s: (s == "F").sum())
        all_f = (total > 0) & (total == n_f)
        return tbl[~all_f].copy()
    raise ValueError(f"Unknown f_mode: {f_mode}")


def apply_mode_grades(df_runs: pd.DataFrame, df_items: pd.DataFrame, df_cells: pd.DataFrame,
                       cell_col: str, item_col_prefix: str, run_col_prefix: str
                       ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add per-f_mode table-level and paper-level grade columns.

    For each f_mode in :data:`F_MODES`:

    - ``df_items[f"{item_col_prefix}_{mode}"]`` = table grade computed as
      ``_avg_series_to_grade`` over the mode's filtered cells, grouped by
      ``(paper_slug, approach, item_id)``. Missing (no cells) → ``"NA"``.
    - ``df_runs[f"{run_col_prefix}_{mode}"]`` = paper grade from the mode's
      filtered tables (using the table grades just computed), grouped by
      ``(paper_slug, approach)``. Missing → ``"NA"``.

    The per-mode table filter is applied *to the just-computed table grades*,
    so ``at_least_one_non_f`` at the paper level looks at the already-filtered
    table grades. Both ``_num`` companion columns are also added.
    """
    if df_cells.empty or cell_col not in df_cells.columns:
        return df_runs, df_items, df_cells

    df_items = df_items.copy()
    df_runs = df_runs.copy()
    table_keys = ["paper_slug", "approach", "item_id"]
    paper_keys = ["paper_slug", "approach"]

    # Flag tables that have at least one cell in df_cells (before any mode filter).
    # Tables without cells (e.g. f_reason == "not_produced", non-numerical unverifiable)
    # fall back to the judge's base grade, since there's nothing to re-aggregate.
    has_cells = df_cells[table_keys].drop_duplicates()
    has_cells["_has_cells"] = True
    df_items = df_items.merge(has_cells, on=table_keys, how="left")
    df_items["_has_cells"] = df_items["_has_cells"].fillna(False)

    for mode in F_MODES:
        # Table grades: aggregate mode-filtered cells
        cells_mode = _cells_for_mode(df_cells, mode, cell_col=cell_col)
        t_col = f"{item_col_prefix}_{mode}"
        if cells_mode.empty:
            df_items[t_col] = np.nan
        else:
            tg = (cells_mode.groupby(table_keys, observed=True)[cell_col]
                  .apply(_avg_series_to_grade).rename(t_col).reset_index())
            df_items = df_items.merge(tg, on=table_keys, how="left")
        # Fill: had cells but all filtered → NA (user spec: "empty after filter → NA");
        #        never had cells → inherit base judge grade (so "not_produced" F penalties survive).
        # The fallback grade must still respect the mode: under `no_f`, an F
        # fallback is dropped (→ NA) so the mode consistently excludes F tables.
        mask_filt_out = df_items["_has_cells"] & df_items[t_col].isna()
        mask_no_cells = (~df_items["_has_cells"]) & df_items[t_col].isna()
        df_items.loc[mask_filt_out, t_col] = "NA"
        base_grades = df_items.loc[mask_no_cells, "grade"].astype(object)
        if mode == "no_f":
            base_grades = base_grades.where(~base_grades.isin(["F"]), "NA")
        df_items.loc[mask_no_cells, t_col] = base_grades
        df_items[f"{t_col}_num"] = df_items[t_col].map(
            lambda g: GRADE_TO_NUM.get(g) if GRADE_TO_NUM.get(g) is not None else np.nan
        )

        # Paper grades: aggregate mode-filtered tables
        tables_mode = _tables_for_mode(df_items, t_col, mode)
        p_col = f"{run_col_prefix}_{mode}"
        if tables_mode.empty:
            df_runs[p_col] = "NA"
        else:
            pg = (tables_mode.groupby(paper_keys, observed=True)[t_col]
                  .apply(_avg_series_to_grade).rename(p_col).reset_index())
            df_runs = df_runs.merge(pg, on=paper_keys, how="left")
            df_runs[p_col] = df_runs[p_col].fillna("NA")
        df_runs[f"{p_col}_num"] = df_runs[p_col].map(
            lambda g: GRADE_TO_NUM.get(g) if GRADE_TO_NUM.get(g) is not None else np.nan
        )

        # Categorical
        for d, c in [(df_items, t_col), (df_runs, p_col)]:
            present = [g for g in GRADE_ORDER if g in d[c].unique()]
            d[c] = pd.Categorical(d[c], categories=present, ordered=True)

    df_items = df_items.drop(columns=["_has_cells"])
    return df_runs, df_items, df_cells


def main():
    parser = argparse.ArgumentParser(description="Analyze i4rep benchmark results")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--papers-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="analysis_output")
    parser.add_argument("--sample-manifest", type=str, default=None,
                        help="Validate the exact paper/run set against a pinned JSON manifest")
    parser.add_argument("--complete-filter", action="store_true",
                        help="Enable filtering to only tables where all approaches succeeded")
    parser.add_argument("--complete-papers", action="store_true",
                        help="Only include papers where all approaches have a run")
    parser.add_argument("--error-analysis-dir", type=str, default=None,
                        help="Path to JE explainer workspace for discrepancy analysis")
    parser.add_argument("--stability-dirs", type=str, nargs="*", default=None,
                        help="Paths to additional stability run results dirs "
                             "(e.g. data/i4replicate/stability_run_1 stability_run_2). "
                             "Enables the run_stability analysis section.")
    parser.add_argument("--regrade-na", dest="regrade_na",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Relabel cells graded F to NA when the original value is "
                             "missing (extractor couldn't extract). Default: on. "
                             "Pass --no-regrade-na to disable.")
    args = parser.parse_args()

    if args.results_dir is None:
        if DEFAULT_BASE.exists():
            args.results_dir = str(DEFAULT_BASE / "results")
            if args.papers_dir is None:
                args.papers_dir = str(DEFAULT_BASE / "papers")
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
    if args.sample_manifest:
        validate_sample_manifest(df_runs, Path(args.sample_manifest))

    # Cell-level relabel F → NA where original value is missing (default on).
    if args.regrade_na:
        df_runs, df_items, df_cells = regrade_with_na(df_runs, df_items, df_cells)

    # Compute the rounded cell-grade column (mirrors regrade-na if applied).
    df_runs, df_items, df_cells = apply_rounded_regrading(df_runs, df_items, df_cells)

    # Per-f_mode rollups to table and paper grades for both base and rounded cells.
    # Produces columns like grade_all_f, overall_grade_no_f, grade_rounded_all_f, etc.
    print("\n  Computing per-mode grade rollups (base + rounded)")
    df_runs, df_items, df_cells = apply_mode_grades(
        df_runs, df_items, df_cells,
        cell_col="cell_grade", item_col_prefix="grade", run_col_prefix="overall_grade",
    )
    df_runs, df_items, df_cells = apply_mode_grades(
        df_runs, df_items, df_cells,
        cell_col="cell_grade_rounded",
        item_col_prefix="grade_rounded", run_col_prefix="overall_grade_rounded",
    )

    # Attach table_category from the GPT classification (if present). This enables
    # the grade_by_table_category plot and filtered coefficient/SE CDFs.
    category_lookup = load_table_categories(output_dir / "table_categories.json")
    if category_lookup:
        df_items = attach_table_category(df_items, category_lookup)
        df_cells = attach_table_category(df_cells, category_lookup)
        n_cats_items = df_items["table_category"].notna().sum()
        print(f"  Attached table_category to {n_cats_items}/{len(df_items)} item rows")
    else:
        df_items["table_category"] = pd.NA
        df_cells["table_category"] = pd.NA

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

    # Save raw (unfiltered) DataFrames
    df_runs.to_csv(output_dir / "df_runs_raw.csv", index=False)
    df_items.to_csv(output_dir / "df_items_raw.csv", index=False)
    df_cells.to_csv(output_dir / "df_cells_raw.csv", index=False, escapechar="\\")
    print(f"  Saved df_runs_raw.csv, df_items_raw.csv, df_cells_raw.csv ({len(df_cells)} cells)")

    # Apply complete-papers filter (default: off) — keep only papers with all approaches
    if args.complete_papers:
        all_approaches = set(df_runs["approach"].unique())
        paper_approaches = df_runs.groupby("paper_slug")["approach"].apply(set)
        complete_papers = set(paper_approaches[paper_approaches.apply(lambda x: x == all_approaches)].index)
        n_before = df_runs["paper_slug"].nunique()
        df_runs = df_runs[df_runs["paper_slug"].isin(complete_papers)].copy()
        df_items = df_items[df_items["paper_slug"].isin(complete_papers)].copy()
        df_cells = df_cells[df_cells["paper_slug"].isin(complete_papers)].copy()
        # Reapply categoricals
        for col_df in [df_runs, df_items]:
            if "approach" in col_df.columns:
                combos = [a for a in APPROACH_ORDER if a in col_df["approach"].unique()]
                col_df["approach"] = pd.Categorical(col_df["approach"], categories=combos, ordered=True)
        if "grade" in df_items.columns:
            df_items["grade"] = pd.Categorical(df_items["grade"], categories=GRADE_ORDER, ordered=True)
        print(f"  Complete-papers filter: {len(complete_papers)}/{n_before} papers kept "
              f"({n_before - len(complete_papers)} excluded)")

    # Apply complete-tables filter (default: off)
    if args.complete_filter:
        df_runs, df_items, df_cells = filter_to_complete_tables(df_runs, df_items, df_cells)
        # Print filtered summary
        print(f"\n{'='*60}")
        print(f"AFTER COMPLETE-TABLES FILTER:")
        print(f"Runs:  {len(df_runs)} ({df_runs['paper_slug'].nunique()} papers)")
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

    # Save (possibly filtered) DataFrames
    df_runs.to_csv(output_dir / "df_runs.csv", index=False)
    df_items.to_csv(output_dir / "df_items.csv", index=False)
    df_cells.to_csv(output_dir / "df_cells.csv", index=False, escapechar="\\")
    print(f"  Saved df_runs.csv, df_items.csv, df_cells.csv ({len(df_cells)} cells)")

    # ── Setup & Descriptives ──────────────────────────────────────
    SD = "setup_descriptives"
    print(f"\n{SD}")
    plot_extractor_row_type_distribution(df_cells, output_dir, results_dir=results_dir, subdir=SD)
    plot_first_fail_distribution(df_items, output_dir, subdir=SD)
    plot_extractor_cells(df_cells, output_dir, subdir=SD)
    plot_agent_loc_distribution(df_runs, output_dir, subdir=SD)
    plot_agent_libraries(df_runs, output_dir, subdir=SD)
    generate_summary_table(df_runs, df_items, output_dir, subdir=SD)
    generate_summary_stats(df_cells, df_items, output_dir, subdir=SD, df_runs=df_runs)
    generate_summary_stats_panels(df_cells, df_items, output_dir, subdir=SD, df_runs=df_runs)
    generate_journal_discipline_table(df_runs, output_dir, subdir=SD)
    generate_overview_csv(df_runs, df_items, output_dir, subdir=SD)
    generate_missingness_reports(df_runs, df_items, output_dir, subdir=SD)

    # ── Paper Level ───────────────────────────────────────────────
    df_items_tables = df_items[df_items["item_type"] == "table"]
    PL = "paper_level"
    print(f"\n{PL}")
    plot_agreement_matrix(df_items_tables, output_dir, subdir=PL)
    plot_duration_vs_grade(df_runs, output_dir, subdir=PL)
    plot_tokens_vs_grade_within_paper(df_runs, output_dir, subdir=PL)
    for f_mode in F_MODES:
        plot_overall_grade_distribution(df_runs, output_dir, subdir=PL, f_mode=f_mode,
                                        grade_col=f"overall_grade_{f_mode}")
        plot_overall_grade_cumulative(df_runs, output_dir, subdir=PL, f_mode=f_mode,
                                       grade_col=f"overall_grade_{f_mode}")
        # Adaptive-rounded paper grades (same per-mode semantics, rounded cells)
        plot_overall_grade_distribution(df_runs, output_dir, subdir=PL, f_mode=f_mode,
                                        grade_col=f"overall_grade_rounded_{f_mode}",
                                        name_suffix="_rounded")
        plot_overall_grade_cumulative(df_runs, output_dir, subdir=PL, f_mode=f_mode,
                                       grade_col=f"overall_grade_rounded_{f_mode}",
                                       name_suffix="_rounded")
        plot_paper_difficulty(df_runs, output_dir, subdir=PL, f_mode=f_mode)
        plot_scatter_vs_grade(df_runs, "total_data_size_bytes", "Total Data Size (bytes)",
                              output_dir, "data_size_vs_grade", log_x=True, subdir=PL, f_mode=f_mode)
        plot_grade_by_discipline(df_runs, output_dir, subdir=PL, f_mode=f_mode)
        plot_grade_by_language(df_runs, output_dir, subdir=PL, f_mode=f_mode)

    # ── Computational Efficiency ──────────────────────────────────
    EF = "efficiency"
    print(f"\n{EF}")
    generate_efficiency_table(df_runs, df_items, output_dir, subdir=EF)
    generate_efficiency_regression(df_runs, df_items, output_dir, subdir=EF)
    for _effort in EFFICIENCY_EFFORT_DIMS:
        plot_efficiency_frontier(df_runs, df_items, output_dir, subdir=EF, effort=_effort)

    # ── Item Level — Tables ───────────────────────────────────────
    IT = "item_tables"
    print(f"\n{IT}")
    for f_mode in F_MODES:
        plot_item_grade_by_type(df_items, output_dir, "table", "table_grade_distribution",
                                subdir=IT, f_mode=f_mode,
                                grade_col=f"grade_{f_mode}")
        plot_item_grade_cumulative(df_items, output_dir, "table", "table_grade_cumulative",
                                    subdir=IT, f_mode=f_mode,
                                    grade_col=f"grade_{f_mode}")
        # Adaptive-rounded table grades
        plot_item_grade_by_type(df_items, output_dir, "table", "table_grade_distribution",
                                subdir=IT, f_mode=f_mode,
                                grade_col=f"grade_rounded_{f_mode}", name_suffix="_rounded")
        plot_item_grade_cumulative(df_items, output_dir, "table", "table_grade_cumulative",
                                    subdir=IT, f_mode=f_mode,
                                    grade_col=f"grade_rounded_{f_mode}", name_suffix="_rounded")
        plot_item_number_vs_grade(df_items, output_dir, subdir=IT, f_mode=f_mode)
        plot_grade_by_table_category(df_items, output_dir, subdir=IT, f_mode=f_mode)
        plot_grade_distribution_by_table_type(df_items, output_dir, subdir=IT, f_mode=f_mode)
        plot_grade_cumulative_by_table_type(df_items, output_dir, subdir=IT, f_mode=f_mode)
        plot_scatter_vs_grade(df_runs, "methodology_summary_len", "Methodology Summary Length (chars)",
                              output_dir, "methodology_length_vs_grade", subdir=IT, f_mode=f_mode)
        plot_scatter_vs_grade(df_runs, "total_code_chars", "Total Code Size (chars)",
                              output_dir, "code_length_vs_grade", log_x=True, subdir=IT, f_mode=f_mode)

    # ── Item Level — Figures ──────────────────────────────────────
    IF_ = "item_figures"
    print(f"\n{IF_}")
    for f_mode in F_MODES:
        plot_item_grade_by_type(df_items, output_dir, "figure", "figure_grade_distribution",
                                subdir=IF_, f_mode=f_mode)

    # ── Cell Level ────────────────────────────────────────────────
    CL = "cell_level"
    print(f"\n{CL}")
    _pct_plots = (plot_pct_diff_by_cell_type, plot_pct_diff_cdf_by_cell_type,
                  plot_pct_diff_by_cell_type_mean, plot_pct_diff_exceedance,
                  plot_pct_diff_histograms)
    for _fn in _pct_plots:
        for _mode in F_MODES:
            _fn(df_cells, output_dir, subdir=CL, f_mode=_mode)
            _fn(df_cells, output_dir, subdir=CL, f_mode=_mode,
                pct_col="percent_difference_rounded", name_suffix="_rounded")
    plot_value_distributions(df_cells, output_dir, subdir=CL)
    plot_coefficient_se_cdf(df_cells, output_dir, subdir=CL)
    # Same CDF but restricted to tables classified as main/mechanism/robustness
    plot_coefficient_se_cdf(
        df_cells, output_dir, subdir=CL,
        category_filter=["main_results", "mechanism", "robustness"],
        name="coefficient_se_cdf_main_mech_rob",
    )
    plot_coefficient_se_scaled(df_cells, output_dir, subdir=CL)
    plot_same_significance(df_cells, output_dir, subdir=CL)
    plot_same_sign(df_cells, output_dir, subdir=CL)
    plot_same_sign_with_missing(df_cells, output_dir, subdir=CL)
    plot_statistic_pct_difference(df_cells, output_dir, "statistic_n_obs",
                                   "n_obs_pct_difference",
                                   ylabel="% difference (N observations)", subdir=CL)
    plot_statistic_pct_difference(df_cells, output_dir, "statistic_r2",
                                   "r2_pct_difference",
                                   ylabel="% difference (R²)", subdir=CL)

    # ── Data Sufficiency Split ────────────────────────────────────
    DS = "data_sufficiency"
    print(f"\n{DS}")
    _run_data_sufficiency_analysis(df_runs, df_items, df_cells, results_dir, output_dir, DS)

    # ── Run Stability (multi-run comparison on a fixed paper sample) ──
    if args.stability_dirs:
        RS = "run_stability"
        print(f"\n{RS}")
        stability_paths = [Path(d) for d in args.stability_dirs]
        _run_stability_analysis(
            df_items, df_cells, results_dir, output_dir, RS,
            stability_dirs=stability_paths, papers_dir=papers_dir,
        )

    # ── Error Analysis ────────────────────────────────────────────
    EA = "error_analysis"
    print(f"\n{EA}")
    plot_fault_attribution(df_items, output_dir, subdir=EA)
    generate_fault_by_grade_table(df_items, output_dir, subdir=EA)
    plot_within_table_error_agreement(df_items, output_dir, subdir=EA)

    # ── Discrepancy Analysis (from code_JE pipeline) ─────────────
    DA = "discrepancy_analysis"
    print(f"\n{DA}")
    error_analysis_dir = Path(args.error_analysis_dir) if args.error_analysis_dir else results_dir.parent / "error_analysis"
    df_div = _load_error_analysis(error_analysis_dir)
    if not df_div.empty:
        # Enforce approach ordering
        combos_present_div = [a for a in APPROACH_ORDER if a in df_div["approach"].values]
        extra_div = [a for a in df_div["approach"].unique() if a not in combos_present_div]
        df_div["approach"] = pd.Categorical(df_div["approach"],
                                            categories=combos_present_div + extra_div, ordered=True)

        n_parse_failed = df_div["parse_failed"].sum()
        n_valid = len(df_div) - n_parse_failed
        print(f"  Loaded {len(df_div)} divergences from {df_div['paper_slug'].nunique()} papers")
        print(f"  Valid: {n_valid}, Parse failed: {n_parse_failed} (excluded from plots)")
        df_div.to_csv(output_dir / "df_divergences.csv", index=False)
        print(f"  Saved df_divergences.csv")

        # Filter out parse failures for plotting
        df_div_valid = df_div[~df_div["parse_failed"]].copy()
        if not df_div_valid.empty:
            plot_divergence_types(df_div_valid, output_dir, subdir=DA)
            plot_divergence_types_aggregate(df_div_valid, output_dir, subdir=DA)
            plot_divergence_types_comparison(df_div_valid, output_dir, subdir=DA)
            plot_root_causes(df_div_valid, output_dir, subdir=DA)
            plot_root_causes_aggregate(df_div_valid, output_dir, subdir=DA)
            plot_root_causes_horizontal(df_div_valid, output_dir, subdir=DA)
            plot_root_causes_coarse(df_div_valid, output_dir, subdir=DA)
            plot_root_causes_coarse_absolute(df_div_valid, output_dir, subdir=DA)
            plot_root_causes_comparison(df_div_valid, output_dir, subdir=DA)
            plot_verdict_distribution(df_div_valid, output_dir, subdir=DA)
            plot_cross_approach_verdict_consistency(df_div_valid, output_dir, subdir=DA)
        else:
            print(f"  No valid divergences to plot (all parse failures)")
    else:
        print(f"  No error_analysis data found at {error_analysis_dir}")

    print(f"\nDone! All outputs in {output_dir}/")


if __name__ == "__main__":
    main()
