"""Comparator Agent: aligns and compares original vs replicated tables.

This is a truly agentic comparator — it spawns a CLI tool (claude-code or
codex) that writes and executes Python code to:
1. Align rows and columns between the two tables (fuzzy label matching)
2. Detect and apply rescaling if one table uses different units
3. Compute per-cell: original_value, replicated_value, absolute_difference,
   percent_difference, sign_match

The agent produces OBJECTIVE metrics only — no grades. Grades are assigned
deterministically by the separate grader module.

The comparison code is saved as an artifact for auditability.
"""

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from ..models.schemas import (
    CellComparison,
    ExtractedTable,
    TableComparison,
)
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Task prompt template for the CLI agent
# ---------------------------------------------------------------------------

COMPARATOR_TASK = """You are comparing an original research table against a replicated version.

This directory contains two files: `original_table.json` (values extracted from
the published paper) and `replicated_table.csv` (produced by an AI replicator).
Explore both files to understand their structure.

## Goal

Write a Python script `compare.py` that aligns the two tables, detects any
scale mismatches, computes per-cell comparison metrics, and writes the results
to `comparison_results.json`. Execute the script and verify the output exists.

Do not install extra packages.

## 1. Alignment (the core challenge)

The two tables represent the same data but may differ substantially in
structure and labeling. You need to match rows and columns by meaning, not
position. Common differences include:

- Label variations: "Log GDP", "log_gdp", "Log(GDP)" are the same variable.
- Column naming: "(1)", "Column 1", "(1) OLS" may be the same column.
- Standard errors may appear as parenthesized values in separate CSV rows but
  as dedicated SE entries in the JSON.
- Panel tables may be structured differently (nested vs flat).

If an original row/column has no match in the replicated table, include it
with replicated_value=null. Ignore extra rows/columns in the replicated table.

## 2. Scale detection

Replicated values may be systematically off by a power of 10 (e.g. percentages
vs decimals, basis points vs normalized). Detect this at two levels:

**a. Global**: Compute the median ratio of |original/replicated| across all
matched numeric cells (excluding near-zero values where |original| < 0.001).
If the median ratio is roughly 10, 100, or 1000 (within a factor of 3),
apply that as a global `scale_factor` to all replicated values before
computing differences.
- Example: original ~50-70 and replicated ~0.5-0.7 → scale_factor=100.

**b. Per-row** (after global rescaling): For each row, compute the median
ratio across that row's numeric columns. If a row's ratio is roughly 10, 100,
or 1000, apply an additional per-row scale factor. Record these in
`row_scale_factors`.
- Example: most rows match, but "MOVE" has original ~0.9 and replicated ~91 → row_scale_factor=0.01.
- Only apply when the ratio clearly indicates a scale mismatch across multiple
  cells in the row, not for 1-2 ambiguous cells.

## 3. Per-cell metrics (after rescaling)

For each matched cell, compute:
- `original_value`: float or null
- `replicated_value`: float or null (raw value before any rescaling)
- `replicated_value_rescaled`: float or null (after rescaling; same as replicated_value if none)
- `absolute_difference`: |rescaled - original| or null
- `percent_difference`: |rescaled - original| / |original| * 100, or null if original is 0
- `sign_match`: true/false/null (null if either value is null or zero)

## 4. Output schema

Write `comparison_results.json` with exactly this structure:
```json
{{
  "cell_comparisons": [
    {{
      "row_label": "string",
      "column_label": "string",
      "original_value": 1.23,
      "replicated_value": 0.0123,
      "replicated_value_rescaled": 1.23,
      "absolute_difference": 0.0,
      "percent_difference": 0.0,
      "sign_match": true,
      "note": "brief note about this cell"
    }}
  ],
  "scale_factor": null,
  "scale_note": "",
  "row_scale_factors": {{}},
  "row_scale_notes": {{}},
  "alignment_notes": "description of how alignment was done",
  "summary": "brief summary of findings"
}}
```
- Handle edge cases: empty cells, non-numeric cells, missing alignments.
"""


# ---------------------------------------------------------------------------
# ComparatorAgent
# ---------------------------------------------------------------------------


class ComparatorAgent:
    """Compares replicated tables against extracted original values.

    Spawns a CLI tool (claude-code or codex) that writes Python code to
    flexibly align and compare the tables. The code is saved as an artifact.
    Produces objective metrics only — no grades.
    """

    def __init__(
        self,
        cli_tool: str = "claude-code",
        model: str = "claude-sonnet-4-6",
        timeout: int = 300,
        max_turns: int = 20,
        claude_binary: str = "claude",
        codex_binary: str = "codex",
    ):
        self.cli_tool = cli_tool.lower()
        self.model = model
        self.timeout = timeout
        self.max_turns = max_turns
        self.claude_binary = claude_binary
        self.codex_binary = codex_binary
        self._usage: list[dict] = []

    @property
    def usage_summary(self) -> dict:
        return {
            "num_calls": len(self._usage),
            "per_call": self._usage,
        }

    def compare_table(
        self,
        original: ExtractedTable,
        replicated_csv: str,
        workspace_dir: str | None = None,
    ) -> TableComparison:
        """Compare an extracted original table against a replicated CSV.

        Args:
            original: Extracted table values from the paper.
            replicated_csv: CSV string of the replicated table.
            workspace_dir: Directory for comparison workspace. If None,
                creates a temp directory.

        Returns:
            TableComparison with objective cell-level metrics (no grades).
        """
        table_id = original.table_id
        logger.info(f"Comparing {table_id}: {len(original.cells)} original cells")

        # Set up workspace
        cleanup = False
        if workspace_dir is None:
            workspace_dir = tempfile.mkdtemp(prefix=f"comparator_{table_id.replace(' ', '_')}_")
            cleanup = False  # keep for debugging; caller can clean up
        workspace = Path(workspace_dir)
        workspace.mkdir(parents=True, exist_ok=True)

        # Write input files
        original_json = self._serialize_original(original)
        (workspace / "original_table.json").write_text(original_json)
        (workspace / "replicated_table.csv").write_text(replicated_csv)

        # Write task prompt
        (workspace / "TASK.md").write_text(COMPARATOR_TASK)

        # Run CLI tool
        try:
            cli_result = self._run_cli(workspace)
        except Exception as e:
            logger.error(f"CLI tool failed for {table_id}: {e}")
            return TableComparison(
                table_id=table_id,
                summary=f"CLI comparison failed: {e}",
            )

        # Parse results
        results_path = workspace / "comparison_results.json"
        if not results_path.exists():
            logger.error(f"No comparison_results.json produced for {table_id}")
            return TableComparison(
                table_id=table_id,
                summary="CLI tool did not produce comparison_results.json",
                alignment_notes=f"CLI exit_code={cli_result.get('exit_code')}",
            )

        try:
            results = json.loads(results_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Could not parse comparison results for {table_id}: {e}")
            return TableComparison(
                table_id=table_id,
                summary=f"Could not parse comparison_results.json: {e}",
            )

        # Load the comparison code for auditability
        code_path = workspace / "compare.py"
        comparison_code = ""
        if code_path.exists():
            comparison_code = code_path.read_text()

        # Convert to schema
        return self._parse_results(table_id, results, comparison_code, cli_result)

    def _serialize_original(self, original: ExtractedTable) -> str:
        """Serialize an ExtractedTable to JSON for the CLI workspace."""
        return json.dumps(
            {
                "table_id": original.table_id,
                "column_labels": original.column_labels,
                "row_labels": original.row_labels,
                "significance_convention": original.significance_convention,
                "cells": [
                    {
                        "row_label": c.row_label,
                        "column_label": c.column_label,
                        "raw_text": c.raw_text,
                        "numeric_value": c.numeric_value,
                        "is_standard_error": c.is_standard_error,
                        "significance_stars": c.significance_stars,
                        "is_string": c.is_string,
                        "row_type": c.row_type,
                    }
                    for c in original.cells
                ],
            },
            indent=2,
        )

    def _run_cli(self, workspace: Path) -> dict:
        """Spawn the CLI tool to write and execute comparison code."""
        prompt_text = (
            "Read TASK.md for instructions. Write compare.py, execute it, "
            "and verify comparison_results.json was created. "
            "Only use files in this directory."
        )

        abs_workspace = str(workspace.resolve())
        start = time.time()

        if self.cli_tool == "claude-code":
            cmd = [
                self.claude_binary, "-p",
                "--output-format", "json",
                "--model", self.model,
                "--dangerously-skip-permissions",
                "--max-turns", str(self.max_turns),
                "--no-session-persistence",
                "--", prompt_text,
            ]
            env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
        elif self.cli_tool == "codex":
            cmd = [
                self.codex_binary, "exec",
                "--full-auto",
                "--json",
                "-m", self.model,
                "-C", abs_workspace,
                "--skip-git-repo-check",
                prompt_text,
            ]
            env = None
        else:
            raise ValueError(f"Unknown CLI tool: {self.cli_tool}")

        logger.info(f"Running {self.cli_tool} comparator: model={self.model}")

        try:
            result = subprocess.run(
                cmd,
                cwd=abs_workspace,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
            )
            stdout = result.stdout
            stderr = result.stderr
            exit_code = result.returncode
        except subprocess.TimeoutExpired:
            logger.warning(f"Comparator timed out after {self.timeout}s")
            stdout = ""
            stderr = f"Timed out after {self.timeout}s"
            exit_code = -1
        except FileNotFoundError:
            binary = self.claude_binary if self.cli_tool == "claude-code" else self.codex_binary
            logger.error(f"CLI binary not found: {binary}")
            raise

        duration = time.time() - start
        logger.info(f"Comparator {self.cli_tool} finished: exit={exit_code}, duration={duration:.1f}s")

        # Save CLI log
        log_path = workspace / "comparator_log.txt"
        log_path.write_text(
            f"CLI tool: {self.cli_tool}\n"
            f"Model: {self.model}\n"
            f"Exit code: {exit_code}\n"
            f"Duration: {duration:.1f}s\n\n"
            f"=== STDOUT ===\n{stdout[:50000]}\n\n"
            f"=== STDERR ===\n{stderr[:10000]}\n"
        )

        # Record usage
        usage = self._extract_usage(stdout)
        usage["duration_seconds"] = duration
        usage["exit_code"] = exit_code
        self._usage.append(usage)

        return {"exit_code": exit_code, "duration": duration, "stdout": stdout, "stderr": stderr}

    def _extract_usage(self, stdout: str) -> dict:
        """Extract token usage from CLI output."""
        if self.cli_tool == "claude-code":
            try:
                events = json.loads(stdout)
                if not isinstance(events, list):
                    events = [events]
                for event in events:
                    if event.get("type") == "result":
                        return {
                            "num_turns": event.get("num_turns", 0),
                            "total_cost_usd": event.get("total_cost_usd", 0),
                        }
            except (json.JSONDecodeError, TypeError):
                pass
        return {}

    @staticmethod
    def _parse_results(
        table_id: str,
        results: dict,
        comparison_code: str,
        cli_result: dict,
    ) -> TableComparison:
        """Convert CLI output JSON into a TableComparison."""
        cell_comparisons = []
        for c in results.get("cell_comparisons", []):
            cell_comparisons.append(CellComparison(
                row_label=c.get("row_label", ""),
                column_label=c.get("column_label", ""),
                original_value=c.get("original_value"),
                replicated_value=c.get("replicated_value"),
                absolute_difference=c.get("absolute_difference"),
                percent_difference=c.get("percent_difference"),
                sign_match=c.get("sign_match"),
                grade="",  # No grade — grader assigns these later
                note=c.get("note", ""),
            ))

        return TableComparison(
            table_id=table_id,
            cell_comparisons=cell_comparisons,
            overall_grade="",  # Grader assigns this later
            summary=results.get("summary", ""),
            alignment_notes=results.get("alignment_notes", ""),
            scale_factor=results.get("scale_factor"),
            scale_note=results.get("scale_note", ""),
            row_scale_factors=results.get("row_scale_factors", {}),
            row_scale_notes=results.get("row_scale_notes", {}),
            comparison_code=comparison_code,
        )
