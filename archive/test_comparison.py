"""Tests for comparison utilities."""

import numpy as np
import pandas as pd
import pytest

from src.utils.comparison import (
    _extract_number,
    assess_significance_match,
    calculate_replication_grade,
    compare_effect_directions,
    compare_tables,
)


class TestExtractNumber:
    def test_plain_int(self):
        assert _extract_number(42) == 42.0

    def test_plain_float(self):
        assert _extract_number(3.14) == 3.14

    def test_string_number(self):
        assert _extract_number("1.23") == 1.23

    def test_negative(self):
        assert _extract_number("-0.45") == -0.45

    def test_with_stars(self):
        assert _extract_number("1.23***") == 1.23

    def test_with_single_star(self):
        assert _extract_number("0.05*") == 0.05

    def test_parentheses(self):
        assert _extract_number("(0.12)") == 0.12

    def test_percentage(self):
        assert _extract_number("50%") == 0.5

    def test_with_commas(self):
        assert _extract_number("1,234.56") == 1234.56

    def test_none(self):
        assert _extract_number(None) is None

    def test_nan(self):
        assert _extract_number(float("nan")) is None

    def test_non_numeric_string(self):
        assert _extract_number("hello") is None

    def test_empty_string(self):
        assert _extract_number("") is None


class TestCompareTables:
    def test_identical_tables(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        result = compare_tables(df, df.copy())
        assert result["match"] is True
        assert result["structural_match"] is True
        assert result["max_difference"] == 0.0

    def test_small_differences(self):
        original = pd.DataFrame({"a": [100.0, 200.0]})
        replicated = pd.DataFrame({"a": [100.5, 200.5]})
        result = compare_tables(original, replicated, tolerance=0.01)
        assert result["structural_match"] is True
        # 0.5/100 = 0.005 and 0.5/200 = 0.0025, both < 1%
        assert result["match"] is True

    def test_large_differences(self):
        original = pd.DataFrame({"a": [100.0]})
        replicated = pd.DataFrame({"a": [120.0]})
        result = compare_tables(original, replicated, tolerance=0.01)
        assert result["match"] is False
        assert result["max_difference"] == pytest.approx(0.2)

    def test_shape_mismatch(self):
        original = pd.DataFrame({"a": [1, 2]})
        replicated = pd.DataFrame({"a": [1, 2, 3]})
        result = compare_tables(original, replicated)
        assert result["structural_match"] is False

    def test_zero_in_original(self):
        original = pd.DataFrame({"a": [0.0]})
        replicated = pd.DataFrame({"a": [0.0]})
        result = compare_tables(original, replicated)
        assert result["match"] is True

    def test_zero_original_nonzero_replicated(self):
        original = pd.DataFrame({"a": [0.0]})
        replicated = pd.DataFrame({"a": [1.0]})
        result = compare_tables(original, replicated)
        assert result["match"] is False

    def test_string_values(self):
        original = pd.DataFrame({"a": ["1.23***", "0.45*"]})
        replicated = pd.DataFrame({"a": ["1.23", "0.45"]})
        result = compare_tables(original, replicated)
        assert result["match"] is True

    def test_text_values_no_numeric_match(self):
        original = pd.DataFrame({"a": ["hello"]})
        replicated = pd.DataFrame({"a": ["world"]})
        result = compare_tables(original, replicated)
        # Non-numeric values that fail to parse are handled via except branch
        assert result["structural_match"] is True


class TestCompareEffectDirections:
    def test_all_same(self):
        result = compare_effect_directions([1.0, -2.0, 3.0], [0.5, -1.0, 2.0])
        assert result["comparable"] is True
        assert result["all_same_direction"] is True
        assert result["proportion_same_direction"] == 1.0

    def test_one_different(self):
        result = compare_effect_directions([1.0, -2.0], [0.5, 1.0])
        assert result["all_same_direction"] is False
        assert result["proportion_same_direction"] == 0.5

    def test_length_mismatch(self):
        result = compare_effect_directions([1.0], [1.0, 2.0])
        assert result["comparable"] is False


class TestAssessSignificanceMatch:
    def test_all_match(self):
        result = assess_significance_match([0.01, 0.06], [0.03, 0.10])
        assert result["all_match"] is True

    def test_mismatch(self):
        result = assess_significance_match([0.01, 0.06], [0.10, 0.03])
        assert result["all_match"] is False
        assert result["proportion_match"] == 0.0

    def test_length_mismatch(self):
        result = assess_significance_match([0.01], [0.01, 0.02])
        assert result["comparable"] is False


class TestCalculateReplicationGrade:
    def test_grade_a(self):
        result = {"structural_match": True, "max_difference": 0.005}
        assert calculate_replication_grade(result) == "A"

    def test_grade_b(self):
        result = {"structural_match": True, "max_difference": 0.03}
        assert calculate_replication_grade(result) == "B"

    def test_grade_c(self):
        result = {"structural_match": True, "max_difference": 0.15}
        assert calculate_replication_grade(result) == "C"

    def test_grade_d(self):
        result = {"structural_match": True, "max_difference": 0.50}
        assert calculate_replication_grade(result) == "D"

    def test_grade_f_structural(self):
        result = {"structural_match": False, "max_difference": 0.0}
        assert calculate_replication_grade(result) == "F"

    def test_grade_f_no_difference(self):
        result = {"structural_match": True, "max_difference": None}
        assert calculate_replication_grade(result) == "F"
