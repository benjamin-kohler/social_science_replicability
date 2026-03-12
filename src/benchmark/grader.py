"""Deterministic grading from objective comparison metrics.

Assigns per-cell and overall grades based on the numeric output of the
ComparatorAgent. No LLM calls — purely mechanical rules so that grades
are reproducible and not biased by LLM reasoning.
"""

from ..models.schemas import CellComparison, TableComparison
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

# Near-zero absolute thresholds (when |original| < 0.001)
_NEAR_ZERO_THRESHOLDS = [
    (0.001, "A"),
    (0.01, "B"),
    (0.05, "C"),
    (0.2, "D"),
]


def grade_cell(cell: CellComparison) -> str:
    """Assign a grade to a single cell comparison.

    Grading scale:
        A: <1% difference (or both zero/near-zero)
        B: 1-10% difference, same sign
        C: 10-20% difference, same sign
        D: 20-50% difference, same sign
        E: >50% difference, different sign, or significance changed
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

    if pct < 1:
        return "A"
    elif pct < 10:
        return "B"
    elif pct < 20:
        return "C"
    elif pct < 50:
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
