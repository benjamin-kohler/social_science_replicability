#!/usr/bin/env python3
"""Regenerate only plots and tables referenced by the active paper.

This is a narrow entry point over the shared loaders and plotting functions in
``analyze_i4rep_results.py``. It deliberately does not emit the exploratory CSVs,
diagnostics, and unused figure variants produced by the full analysis program.

The whitelist was audited against ``paper_emnlp_v2_rebuttal.tex`` on 2026-08-23.
Figures made by the separate tool-usage and guardrail programs, static example
figures/tables, and prompt listings are outside this entry point.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import analyze_i4rep_results as a
from scripts import analyze_pre_post_cutoff as cutoff
from scripts.analyze_extractor_variant import _run_extractor_variant_analysis
from scripts.plot_root_causes_narrow import plot_narrow


PAPER_OUTPUTS = {
    "cell_level/coefficient_se_cdf.pdf",
    "cell_level/coefficient_se_cdf_main_mech_rob.pdf",
    "cell_level/pct_diff_by_cell_type_mean_rounded.pdf",
    "cell_level/same_sign.pdf",
    "cell_level/same_sign_with_missing.pdf",
    "cell_level/value_distributions.pdf",
    "discrepancy_analysis/root_causes_absolute.png",
    "discrepancy_analysis/root_causes_coarse_absolute_narrow.pdf",
    "efficiency/efficiency_regression_tokens_contrast.tex",
    "extractor_variant/variant_coefficient_se_cdf.pdf",
    "extractor_variant/variant_grade_dumbbell.pdf",
    "item_tables/code_length_vs_grade_no_f.pdf",
    "item_tables/grade_cumulative_by_table_type_all_f.pdf",
    "item_tables/table_grade_cumulative_all_f_rounded.pdf",
    "item_tables/table_grade_cumulative_at_least_one_non_f_rounded.pdf",
    "item_tables/table_grade_cumulative_no_f_rounded.pdf",
    "paper_level/agreement_matrix.pdf",
    "paper_level/cost_by_approach.pdf",
    "paper_level/data_size_vs_grade_no_f.pdf",
    "paper_level/duration_by_approach.pdf",
    "paper_level/grade_by_discipline_no_f.pdf",
    "paper_level/grade_by_language_primary_no_f.pdf",
    "paper_level/overall_grades_cumulative_all_f_rounded.pdf",
    "paper_level/overall_grades_cumulative_at_least_one_non_f_rounded.pdf",
    "paper_level/overall_grades_cumulative_no_f_rounded.pdf",
    "paper_level/paper_difficulty.pdf",
    "paper_level/tokens_by_approach.pdf",
    "pre_post_cutoff/bootstrap_grade_comparison.pdf",
    "run_stability/cell_pct_diff_range_cdf.pdf",
    "run_stability/coefficient_se_cdf_between_runs.pdf",
    "run_stability/coefficient_se_cdf_by_run.pdf",
    "run_stability/paper_grade_spread_discrete.pdf",
    "run_stability/table_grade_spread.pdf",
    "setup_descriptives/agent_libraries_heatmap.pdf",
    "setup_descriptives/agent_loc_distribution.pdf",
}

ALIASES = {
    "cell_level/pct_diff_by_cell_type_mean_rounded.pdf":
        "cell_level/pct_diff_by_cell_type_mean_all_f_rounded.pdf",
    "paper_level/paper_difficulty.pdf": "paper_level/paper_difficulty_all_f.pdf",
    # The manuscript retained the older CDF-style filename after this plot was
    # changed to a log-binned histogram.  Publish the current figure under the
    # manuscript-facing name so a clean rebuild needs no manual rename.
    "run_stability/coefficient_se_cdf_between_runs.pdf":
        "run_stability/coefficient_se_hist_between_runs.pdf",
}

VARIANT_APPROACHES = [
    "claude-code/claude-opus-4-6",
    "codex/gpt-5.4",
]


def _prepare_frames(results_dir: Path, papers_dir: Path | None,
                    manifest: Path, table_categories: Path | None):
    runs, items, cells = a.load_results(results_dir, papers_dir)
    a.validate_sample_manifest(runs, manifest)
    runs, items, cells = a.regrade_with_na(runs, items, cells)
    runs, items, cells = a.apply_rounded_regrading(runs, items, cells)
    runs, items, cells = a.apply_mode_grades(
        runs, items, cells, cell_col="cell_grade",
        item_col_prefix="grade", run_col_prefix="overall_grade",
    )
    runs, items, cells = a.apply_mode_grades(
        runs, items, cells, cell_col="cell_grade_rounded",
        item_col_prefix="grade_rounded", run_col_prefix="overall_grade_rounded",
    )

    lookup = a.load_table_categories(table_categories) if table_categories else {}
    if lookup:
        items = a.attach_table_category(items, lookup)
        cells = a.attach_table_category(cells, lookup)
    else:
        items["table_category"] = pd.NA
        cells["table_category"] = pd.NA

    if "non_numerical" in items.columns:
        items = items[~items["non_numerical"]].copy()
    return runs, items, cells


def _generate_primary(runs, items, cells, results_dir: Path,
                      papers_dir: Path | None, error_analysis_dir: Path,
                      stability_dirs: list[Path], variant_dir: Path | None,
                      precutoff_dir: Path | None, postcutoff_dir: Path | None,
                      work: Path) -> None:
    a.setup_style()
    tables = items[items["item_type"] == "table"]

    a.plot_same_sign(cells, work, subdir="cell_level")
    a.plot_coefficient_se_cdf(cells, work, subdir="cell_level")
    a.plot_pct_diff_by_cell_type_mean(
        cells, work, subdir="cell_level", f_mode="all_f",
        pct_col="percent_difference_rounded", name_suffix="_rounded",
    )
    a.plot_same_sign_with_missing(cells, work, subdir="cell_level")
    a.plot_coefficient_se_cdf(
        cells, work, subdir="cell_level",
        category_filter=["main_results", "mechanism", "robustness"],
        name="coefficient_se_cdf_main_mech_rob",
    )
    a.plot_value_distributions(cells, work, subdir="cell_level")

    for mode in a.F_MODES:
        a.plot_item_grade_cumulative(
            items, work, "table", "table_grade_cumulative",
            subdir="item_tables", f_mode=mode,
            grade_col=f"grade_rounded_{mode}", name_suffix="_rounded",
        )
        a.plot_overall_grade_cumulative(
            runs, work, subdir="paper_level", f_mode=mode,
            grade_col=f"overall_grade_rounded_{mode}", name_suffix="_rounded",
        )

    a.plot_grade_cumulative_by_table_type(
        items, work, subdir="item_tables", f_mode="all_f",
    )
    a.plot_scatter_vs_grade(
        runs, "total_code_chars", "Total Code Size (chars)", work,
        "code_length_vs_grade", log_x=True, subdir="item_tables", f_mode="no_f",
    )
    a.plot_agreement_matrix(tables, work, subdir="paper_level")
    a.plot_paper_difficulty(runs, work, subdir="paper_level", f_mode="all_f")
    a.plot_grade_by_discipline(runs, work, subdir="paper_level", f_mode="no_f")
    a.plot_grade_by_language(runs, work, subdir="paper_level", f_mode="no_f")
    a.plot_scatter_vs_grade(
        runs, "total_data_size_bytes", "Total Data Size (bytes)", work,
        "data_size_vs_grade", log_x=True, subdir="paper_level", f_mode="no_f",
    )
    a.plot_duration_vs_grade(runs, work, subdir="paper_level")
    a.plot_agent_loc_distribution(runs, work, subdir="setup_descriptives")
    a.plot_agent_libraries(runs, work, subdir="setup_descriptives")
    a.generate_efficiency_regression(runs, items, work, subdir="efficiency")

    if stability_dirs:
        a._run_stability_analysis(
            items, cells, results_dir, work, "run_stability",
            stability_dirs=stability_dirs, papers_dir=papers_dir,
        )

    divergences = a._load_error_analysis(error_analysis_dir)
    if not divergences.empty:
        valid = divergences[~divergences["parse_failed"]].copy()
        a.plot_root_causes_horizontal(valid, work, subdir="discrepancy_analysis")
        a.plot_root_causes_coarse_absolute(valid, work, subdir="discrepancy_analysis")
        plot_narrow(valid, work / "discrepancy_analysis")

    if variant_dir:
        _run_extractor_variant_analysis(
            runs, items, cells, work, "extractor_variant", variant_dir,
            stability_dirs=stability_dirs, papers_dir=papers_dir,
            approaches=VARIANT_APPROACHES,
        )

    if precutoff_dir and postcutoff_dir:
        pre = cutoff.load_collection(precutoff_dir, "Pre-cutoff")
        post = cutoff.load_collection(postcutoff_dir, "Post-cutoff")
        combined = pd.concat([pre, post], ignore_index=True)
        if not combined.empty:
            cutoff.plot_bootstrap_comparison(combined, work / "pre_post_cutoff")


def _publish_whitelist(work: Path, output_dir: Path) -> None:
    missing = []
    for relative in sorted(PAPER_OUTPUTS):
        source_relative = ALIASES.get(relative, relative)
        source = work / source_relative
        if not source.is_file():
            missing.append(f"{relative} (expected source {source_relative})")
            continue
        target = output_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    if missing:
        raise RuntimeError("Paper outputs were not generated:\n- " + "\n- ".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--papers-dir", type=Path, default=None)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--table-categories", type=Path, default=None)
    parser.add_argument("--error-analysis-dir", type=Path, required=True)
    parser.add_argument("--stability-dirs", type=Path, nargs="*", default=[])
    parser.add_argument("--extractor-variant-dir", type=Path, default=None)
    parser.add_argument("--precutoff-results", type=Path, default=None)
    parser.add_argument("--postcutoff-results", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    frames = _prepare_frames(
        args.results_dir, args.papers_dir, args.sample_manifest,
        args.table_categories,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="paper-analysis-") as tmp:
        work = Path(tmp)
        _generate_primary(
            *frames, args.results_dir, args.papers_dir, args.error_analysis_dir,
            args.stability_dirs, args.extractor_variant_dir,
            args.precutoff_results, args.postcutoff_results, work,
        )
        _publish_whitelist(work, args.output_dir)
    print(f"Generated {len(PAPER_OUTPUTS)} manuscript outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
