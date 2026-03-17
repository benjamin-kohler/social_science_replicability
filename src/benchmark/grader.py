"""Deterministic grading from objective comparison metrics.

Assigns per-cell and overall grades based on the numeric output of the
ComparatorAgent. No LLM calls — purely mechanical rules so that grades
are reproducible and not biased by LLM reasoning.
"""

import math

from ..models.schemas import CellComparison, ExtractedTable, TableComparison
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

# Near-zero absolute thresholds (when |original| < 0.001)
_NEAR_ZERO_THRESHOLDS = [
    (0.002, "A"),
    (0.02, "B"),
    (0.05, "C"),
    (0.1, "D"),
]


def grade_cell(cell: CellComparison) -> str:
    """Assign a grade to a single cell comparison.

    Grading scale:
        A: <2% difference (or both zero/near-zero)
        B: 2-20% difference, same sign
        C: 20-40% difference, same sign
        D: 40-60% difference, same sign
        E: >60% difference or sign flip
        F: missing, incomparable, or could not be aligned
    """
    # Missing / incomparable
    if cell.original_value is None or cell.replicated_value is None:
        return "F"

    # Both zero
    if cell.original_value == 0 and cell.replicated_value == 0:
        return "A"

    # Sign mismatch (only meaningful for non-zero values)
    if cell.sign_match is False:
        return "E"

    # Near-zero original: use absolute thresholds
    if abs(cell.original_value) < 0.001:
        abs_diff = cell.absolute_difference
        if abs_diff is None:
            abs_diff = abs((cell.replicated_value or 0) - (cell.original_value or 0))
        for threshold, grade in _NEAR_ZERO_THRESHOLDS:
            if abs_diff < threshold:
                return grade
        return "E"

    # Standard percentage-based grading
    pct = cell.percent_difference
    if pct is None:
        return "F"

    if pct < 2:
        return "A"
    elif pct < 20:
        return "B"
    elif pct < 40:
        return "C"
    elif pct < 60:
        return "D"
    else:
        return "E"


def grade_table(comparison: TableComparison) -> TableComparison:
    """Fill in per-cell grades and compute overall grade for a TableComparison.

    Mutates and returns the same TableComparison object.
    """
    for cell in comparison.cell_comparisons:
        cell.grade = grade_cell(cell)

    comparison.overall_grade = _compute_overall_grade(comparison.cell_comparisons)
    return comparison


def _compute_overall_grade(cells: list[CellComparison]) -> str:
    """Compute an overall table grade from individual cell grades.

    The overall grade reflects the typical quality across graded cells.
    Cells graded F (missing/incomparable) are excluded from the average
    unless all cells are F.
    """
    grade_values = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1, "F": 0}

    # Exclude F cells from the average (they're missing, not wrong)
    graded = [c for c in cells if c.grade != "F"]
    if not graded:
        return "F"

    avg = sum(grade_values[c.grade] for c in graded) / len(graded)

    if avg >= 4.5:
        return "A"
    if avg >= 3.5:
        return "B"
    if avg >= 2.5:
        return "C"
    if avg >= 1.5:
        return "D"
    if avg >= 0.5:
        return "E"
    return "F"


# ---------------------------------------------------------------------------
# Pre-aligned table grading (no comparator agent needed)
# ---------------------------------------------------------------------------


def grade_aligned_tables(
    original: ExtractedTable,
    replicated: ExtractedTable,
) -> TableComparison:
    """Grade pre-aligned tables by zipping cells at matching positions.

    Both tables share the same row/column structure (the replicated table
    was produced from a blinded copy of the original). Cells are matched
    by (row_label, column_label). String cells and panel headers are skipped.

    Includes lightweight scale detection: if the median ratio across all
    matched cells is approximately a power of 10, rescaling is applied.

    Returns a fully graded TableComparison.
    """
    import re

    # --- Strategy 1: Match by (row_index, col_index) when available ---
    # This is the most robust approach: both tables share the same grid
    # coordinates from the blinded template.
    _has_indices = (
        all(c.row_index is not None and c.col_index is not None for c in original.cells if not c.is_string and c.row_type != "panel_header")
        and any(c.row_index is not None and c.col_index is not None for c in replicated.cells if not c.is_string and c.row_type != "panel_header")
    )

    if _has_indices:
        repl_by_idx = {}
        for cell in replicated.cells:
            if cell.row_index is not None and cell.col_index is not None:
                repl_by_idx[(cell.row_index, cell.col_index)] = cell

        _orig_to_repl: dict[int, object] = {}
        for cell in original.cells:
            if cell.is_string or cell.row_type == "panel_header":
                continue
            if cell.row_index is not None and cell.col_index is not None:
                repl_cell = repl_by_idx.get((cell.row_index, cell.col_index))
                if repl_cell is not None:
                    _orig_to_repl[id(cell)] = repl_cell

        logger.info(f"Matched {len(_orig_to_repl)} cells by (row_index, col_index)")

    else:
        # --- Strategy 2: Column-position matching (fallback) ---
        # Groups cells by column, matches by position within column.
        # Handles different orderings, label mismatches, extra rows.

        def _group_by_column(cells):
            """Group non-string, non-panel cells by column_label, preserving order."""
            groups: dict[str, list] = {}
            for cell in cells:
                if cell.is_string or cell.row_type == "panel_header":
                    continue
                groups.setdefault(cell.column_label, []).append(cell)
            return groups

        def _normalize_col(label: str) -> str:
            """Strip leading '(N)' prefixes and normalize."""
            label = re.sub(r"^\(\d+\)\s*", "", label)
            return label.strip().lower()

        orig_by_col = _group_by_column(original.cells)
        repl_by_col = _group_by_column(replicated.cells)

        # Fuzzy column mapping
        _col_map: dict[str, str] = {}
        repl_norm_lookup = {_normalize_col(k): k for k in repl_by_col}
        for orig_col in orig_by_col:
            if orig_col in repl_by_col:
                _col_map[orig_col] = orig_col
            else:
                norm = _normalize_col(orig_col)
                if norm in repl_norm_lookup:
                    _col_map[orig_col] = repl_norm_lookup[norm]

        mapped_orig_by_col: dict[str, list] = {}
        for orig_col, cells in orig_by_col.items():
            mapped_col = _col_map.get(orig_col, orig_col)
            mapped_orig_by_col.setdefault(mapped_col, []).extend(cells)
        orig_by_col = mapped_orig_by_col

        _orig_to_repl: dict[int, object] = {}
        for col_label, orig_col_cells in orig_by_col.items():
            repl_col_cells = repl_by_col.get(col_label, [])
            for i, orig_cell in enumerate(orig_col_cells):
                if i < len(repl_col_cells):
                    _orig_to_repl[id(orig_cell)] = repl_col_cells[i]

        logger.info(f"Matched {len(_orig_to_repl)} cells by column-position (fallback)")

    def _find_repl_cell(idx, orig_cell):
        """Find matching replicated cell."""
        return _orig_to_repl.get(id(orig_cell))

    # Collect matched numeric pairs for scale detection
    matched_pairs: list[tuple[float, float]] = []  # (original, replicated)
    for idx, cell in enumerate(original.cells):
        if cell.is_string or cell.row_type == "panel_header":
            continue
        if cell.numeric_value is None:
            continue
        repl_cell = _find_repl_cell(idx, cell)
        if repl_cell is None or repl_cell.numeric_value is None:
            continue
        matched_pairs.append((cell.numeric_value, repl_cell.numeric_value))

    # Detect global scale factor (median ratio, must be ~power of 10)
    scale_factor = None
    scale_note = ""
    if matched_pairs:
        ratios = []
        for orig_val, repl_val in matched_pairs:
            if abs(orig_val) > 1e-10:
                ratios.append(repl_val / orig_val)
        if ratios:
            ratios.sort()
            median_ratio = ratios[len(ratios) // 2]
            if abs(median_ratio) > 1e-10:
                log10 = math.log10(abs(median_ratio))
                nearest_pow = round(log10)
                if nearest_pow != 0 and abs(log10 - nearest_pow) < 0.15:
                    scale_factor = 10.0 ** (-nearest_pow)
                    scale_note = (
                        f"Detected scale mismatch: median ratio ≈ 10^{nearest_pow}. "
                        f"Applied factor {scale_factor} to replicated values."
                    )

    # Build cell comparisons
    cell_comparisons: list[CellComparison] = []
    for idx, cell in enumerate(original.cells):
        if cell.is_string or cell.row_type == "panel_header":
            continue

        repl_cell = _find_repl_cell(idx, cell)
        orig_val = cell.numeric_value
        repl_val = repl_cell.numeric_value if repl_cell else None

        # Apply scale factor
        if repl_val is not None and scale_factor is not None:
            repl_val = repl_val * scale_factor

        # Compute metrics
        abs_diff = None
        pct_diff = None
        sign_match = None

        if orig_val is not None and repl_val is not None:
            abs_diff = abs(repl_val - orig_val)
            if abs(orig_val) > 1e-10:
                pct_diff = (abs_diff / abs(orig_val)) * 100.0
            elif abs_diff < 1e-10:
                pct_diff = 0.0
            else:
                pct_diff = None  # near-zero original handled by grade_cell
            sign_match = (orig_val >= 0) == (repl_val >= 0)

        cc = CellComparison(
            row_label=cell.row_label,
            column_label=cell.column_label,
            original_value=orig_val,
            replicated_value=repl_val,
            absolute_difference=abs_diff,
            percent_difference=pct_diff,
            sign_match=sign_match,
        )
        cell_comparisons.append(cc)

    comparison = TableComparison(
        table_id=original.table_id,
        cell_comparisons=cell_comparisons,
        summary=f"Pre-aligned comparison: {len(cell_comparisons)} cells matched by position.",
        alignment_notes="Cells matched by (row_label, column_label) from blinded template.",
        scale_factor=scale_factor,
        scale_note=scale_note,
    )

    # Apply per-cell and overall grading
    return grade_table(comparison)
