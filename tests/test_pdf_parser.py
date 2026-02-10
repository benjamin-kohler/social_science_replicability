"""Tests for PDF parsing utilities."""

import pytest

from src.utils.pdf_parser import (
    extract_figure_captions,
    extract_table_captions,
    identify_sections,
)


class TestIdentifySections:
    def test_basic_sections(self):
        text = """Abstract
This is the abstract.
Introduction
This is the introduction.
Methods
This describes the methods.
Results
These are the results.
Conclusion
This is the conclusion."""
        sections = identify_sections(text)
        assert "abstract" in sections
        assert "introduction" in sections
        assert "methods" in sections
        assert "results" in sections
        assert "conclusion" in sections

    def test_no_sections(self):
        text = "Just some plain text without section headers."
        sections = identify_sections(text)
        assert "preamble" in sections

    def test_alternative_headers(self):
        text = """Literature Review
Prior work shows...
Empirical Strategy
We use an IV approach...
Discussion
Our findings suggest..."""
        sections = identify_sections(text)
        assert any("literature" in k for k in sections)
        assert any("empirical" in k for k in sections)
        assert "discussion" in sections


class TestExtractTableCaptions:
    def test_basic_caption(self):
        text = "Table 1: Summary Statistics\nThis table shows..."
        captions = extract_table_captions(text)
        assert len(captions) == 1
        assert captions[0]["table_number"] == "Table 1"
        assert "Summary Statistics" in captions[0]["caption"]

    def test_multiple_tables(self):
        text = """Table 1: First Results
Some text here.
Table 2: Second Results
More text."""
        captions = extract_table_captions(text)
        assert len(captions) == 2
        assert captions[0]["table_number"] == "Table 1"
        assert captions[1]["table_number"] == "Table 2"

    def test_table_with_period(self):
        text = "Table 3. Regression Output\nDetails follow."
        captions = extract_table_captions(text)
        assert len(captions) == 1
        assert captions[0]["table_number"] == "Table 3"

    def test_no_tables(self):
        text = "This text has no tables at all."
        captions = extract_table_captions(text)
        assert len(captions) == 0

    def test_appendix_table(self):
        # The regex expects a digit after "Table", so "Table A1" needs digit first
        text = "Table 1A: Robustness Check"
        captions = extract_table_captions(text)
        assert len(captions) == 1
        assert captions[0]["table_number"] == "Table 1A"


class TestExtractFigureCaptions:
    def test_basic_caption(self):
        text = "Figure 1: Treatment Effect Over Time\nShows the trend."
        captions = extract_figure_captions(text)
        assert len(captions) == 1
        assert captions[0]["figure_number"] == "Figure 1"

    def test_fig_abbreviation(self):
        text = "Fig. 2: Distribution of Outcomes"
        captions = extract_figure_captions(text)
        assert len(captions) == 1
        assert captions[0]["figure_number"] == "Figure 2"

    def test_multiple_figures(self):
        text = """Figure 1: First plot
Figure 2: Second plot
Figure 3: Third plot"""
        captions = extract_figure_captions(text)
        assert len(captions) == 3

    def test_no_figures(self):
        text = "No figures in this text."
        captions = extract_figure_captions(text)
        assert len(captions) == 0
