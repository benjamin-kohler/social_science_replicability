"""Utilities for comparing replicated results with originals."""

import re
from typing import Any, Optional

import numpy as np
import pandas as pd

from .logging_utils import get_logger

logger = get_logger(__name__)


def compare_tables(
    original: pd.DataFrame,
    replicated: pd.DataFrame,
    tolerance: float = 0.01,
) -> dict[str, Any]:
    """Compare two tables and quantify differences.

    Args:
        original: Original table from paper.
        replicated: Replicated table.
        tolerance: Tolerance for numerical differences (proportion, e.g., 0.01 = 1%).

    Returns:
        Dictionary containing:
        - 'match': Whether tables match within tolerance
        - 'numerical_differences': Dict of cell-level differences
        - 'max_difference': Maximum proportional difference
        - 'mean_difference': Mean proportional difference
        - 'structural_match': Whether structure matches
    """
    result = {
        "match": False,
        "numerical_differences": {},
        "max_difference": None,
        "mean_difference": None,
        "structural_match": True,
        "notes": [],
    }

    # Check structural match
    if original.shape != replicated.shape:
        result["structural_match"] = False
        result["notes"].append(
            f"Shape mismatch: original {original.shape} vs replicated {replicated.shape}"
        )
        return result

    # Compare each cell
    differences = []
    for i in range(original.shape[0]):
        for j in range(original.shape[1]):
            orig_val = original.iloc[i, j]
            repl_val = replicated.iloc[i, j]

            # Try to compare as numbers
            try:
                orig_num = _extract_number(orig_val)
                repl_num = _extract_number(repl_val)

                if orig_num is not None and repl_num is not None:
                    if orig_num != 0:
                        diff = abs(repl_num - orig_num) / abs(orig_num)
                    elif repl_num == 0:
                        diff = 0
                    else:
                        diff = float("inf")

                    differences.append(diff)

                    if diff > tolerance:
                        result["numerical_differences"][f"({i},{j})"] = {
                            "original": orig_num,
                            "replicated": repl_num,
                            "difference": diff,
                        }

            except (ValueError, TypeError):
                # Non-numeric comparison
                if str(orig_val).strip() != str(repl_val).strip():
                    result["notes"].append(f"Text mismatch at ({i},{j})")

    if differences:
        result["max_difference"] = max(differences)
        result["mean_difference"] = np.mean(differences)
        result["match"] = result["max_difference"] <= tolerance

    return result


def _extract_number(value: Any) -> Optional[float]:
    """Extract numeric value from a cell.

    Handles common formats like:
    - Plain numbers: 1.23, -0.45
    - Numbers with stars: 1.23**, 0.45***
    - Numbers in parentheses: (0.12)
    - Percentages: 12.3%
    """
    if value is None or pd.isna(value):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        # Remove common non-numeric characters
        cleaned = value.strip()

        # Handle parentheses (often used for standard errors or negative numbers)
        paren_match = re.match(r"\(([0-9.,\-]+)\)", cleaned)
        if paren_match:
            cleaned = paren_match.group(1)

        # Remove stars (significance markers)
        cleaned = re.sub(r"\*+", "", cleaned)

        # Remove percentage sign
        is_percentage = "%" in cleaned
        cleaned = cleaned.replace("%", "")

        # Remove commas
        cleaned = cleaned.replace(",", "")

        try:
            num = float(cleaned)
            if is_percentage:
                num /= 100
            return num
        except ValueError:
            return None

    return None


def compare_effect_directions(
    original_effects: list[float],
    replicated_effects: list[float],
) -> dict[str, Any]:
    """Compare effect directions between original and replicated results.

    Args:
        original_effects: List of effect sizes from original.
        replicated_effects: List of effect sizes from replication.

    Returns:
        Comparison results.
    """
    if len(original_effects) != len(replicated_effects):
        return {
            "comparable": False,
            "note": "Different number of effects",
        }

    same_direction = []
    for orig, repl in zip(original_effects, replicated_effects):
        # Check if signs match (or both are zero)
        same_sign = (orig > 0 and repl > 0) or (orig < 0 and repl < 0) or (orig == 0 and repl == 0)
        same_direction.append(same_sign)

    return {
        "comparable": True,
        "all_same_direction": all(same_direction),
        "proportion_same_direction": sum(same_direction) / len(same_direction),
        "details": same_direction,
    }


def assess_significance_match(
    original_pvals: list[float],
    replicated_pvals: list[float],
    threshold: float = 0.05,
) -> dict[str, Any]:
    """Assess whether significance conclusions match.

    Args:
        original_pvals: P-values from original.
        replicated_pvals: P-values from replication.
        threshold: Significance threshold (default 0.05).

    Returns:
        Assessment results.
    """
    if len(original_pvals) != len(replicated_pvals):
        return {
            "comparable": False,
            "note": "Different number of tests",
        }

    matches = []
    for orig, repl in zip(original_pvals, replicated_pvals):
        orig_sig = orig < threshold
        repl_sig = repl < threshold
        matches.append(orig_sig == repl_sig)

    return {
        "comparable": True,
        "all_match": all(matches),
        "proportion_match": sum(matches) / len(matches),
        "details": matches,
    }


def calculate_replication_grade(
    comparison_result: dict[str, Any],
    tolerance_a: float = 0.01,
    tolerance_b: float = 0.05,
    tolerance_c: float = 0.20,
) -> str:
    """Calculate replication grade based on comparison results.

    Args:
        comparison_result: Result from compare_tables or similar.
        tolerance_a: Max difference for grade A.
        tolerance_b: Max difference for grade B.
        tolerance_c: Max difference for grade C.

    Returns:
        Grade string: A, B, C, D, or F.
    """
    if not comparison_result.get("structural_match"):
        return "F"

    max_diff = comparison_result.get("max_difference")
    if max_diff is None:
        return "F"

    if max_diff <= tolerance_a:
        return "A"
    elif max_diff <= tolerance_b:
        return "B"
    elif max_diff <= tolerance_c:
        return "C"
    else:
        return "D"
