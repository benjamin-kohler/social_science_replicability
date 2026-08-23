"""Regression tests for occurrence-aware SE/significance enrichment."""

import json

from scripts.analyze_i4rep_results import (
    _coefficient_metadata_by_position,
    load_results,
)


def _panel_cells(se_values=(0.1, 0.2), stars=(1, 2)):
    cells = []
    for panel, (se_value, star_count) in enumerate(zip(se_values, stars)):
        coefficient_row = panel * 3 + 1
        cells.extend([
            {
                "row_label": f"Panel {panel + 1}",
                "column_label": "(1)",
                "row_index": coefficient_row - 1,
                "col_index": 0,
                "numeric_value": None,
                "row_type": "panel_header",
            },
            {
                "row_label": "Treatment",
                "column_label": "(1)",
                "row_index": coefficient_row,
                "col_index": 0,
                "numeric_value": panel + 1.0,
                "row_type": "coefficient",
                "significance_stars": star_count,
            },
            {
                "row_label": "",
                "column_label": "(1)",
                "row_index": coefficient_row + 1,
                "col_index": 0,
                "numeric_value": se_value,
                "row_type": "se",
                "is_standard_error": True,
                "refers_to": coefficient_row,
            },
        ])
    return cells


def test_coefficient_metadata_preserves_repeated_panel_rows():
    metadata = _coefficient_metadata_by_position("Table 1", _panel_cells())

    first = metadata[("Table 1", "Treatment", "(1)", 0)]
    second = metadata[("Table 1", "Treatment", "(1)", 1)]
    assert first == {"se": 0.1, "stars": 1}
    assert second == {"se": 0.2, "stars": 2}


def test_load_results_attaches_metadata_by_occurrence(tmp_path):
    paper = "paper"
    results_dir = tmp_path / "results"
    paper_dir = results_dir / paper
    run_dir = paper_dir / f"model_{paper}_codex"
    workspace = run_dir / "workspace"
    summaries = paper_dir / "summaries"
    workspace.mkdir(parents=True)
    summaries.mkdir()

    original_cells = _panel_cells(se_values=(0.1, 0.2), stars=(1, 2))
    replicated_cells = _panel_cells(se_values=(0.11, 0.22), stars=(0, 3))
    comparisons = [
        {
            "row_label": "Treatment",
            "column_label": "(1)",
            "original_value": 1.0,
            "replicated_value": 1.0,
            "grade": "A",
        },
        {
            "row_label": "Treatment",
            "column_label": "(1)",
            "original_value": 2.0,
            "replicated_value": 2.0,
            "grade": "A",
        },
    ]

    (summaries / f"{paper}_results.json").write_text(json.dumps({
        "tables": [{"table_id": "Table 1", "cells": original_cells}],
    }))
    (workspace / "table_1.json").write_text(json.dumps({
        "table_id": "Table 1", "cells": replicated_cells,
    }))
    (workspace / "methodology_summary.json").write_text(json.dumps({
        "extracted_tables": [{
            "table_id": "Table 1",
            "cells": [{
                "row_label": "Treatment",
                "column_label": "(1)",
                "row_type": "coefficient",
            }],
        }],
    }))
    (run_dir / "verification_report.json").write_text(json.dumps({
        "overall_grade": "A",
        "item_verifications": [{
            "item_id": "Table 1",
            "item_type": "table",
            "grade": "A",
            "table_comparison": {"cell_comparisons": comparisons},
        }],
    }))

    _, _, cells = load_results(results_dir)

    assert cells["original_se"].tolist() == [0.1, 0.2]
    assert cells["replicated_se"].tolist() == [0.11, 0.22]
    assert cells["significance_stars_orig"].tolist() == [1, 2]
    assert cells["significance_stars_repl"].tolist() == [0, 3]
