"""Parse workspace output into structured ReplicationResults."""

import json
import re
from pathlib import Path

from ..models.schemas import (
    ExtractedTable,
    GeneratedCode,
    GeneratedFigure,
    GeneratedTable,
    ReplicationResults,
)
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

CODE_EXTENSIONS = {".py", ".r"}
TABLE_EXTENSIONS = {".csv", ".json"}
FIGURE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf", ".svg"}


class ArtifactParser:
    """Parses a workspace into ReplicationResults.

    Scans the workspace directory for generated code, tables, and figures.
    Filenames follow the controlled convention: table_2.1.csv, figure_3.1.png.
    """

    @staticmethod
    def parse(workspace_dir: Path, paper_id: str) -> ReplicationResults:
        """Parse workspace files into ReplicationResults."""
        code_files = []
        tables = []
        figures = []

        if not workspace_dir.exists():
            logger.warning(f"Workspace directory does not exist: {workspace_dir}")
            return ReplicationResults(paper_id=paper_id)

        # Directories to skip (e.g., table_templates/ contains blank templates, not results)
        skip_dirs = {"table_templates", "csv_export", "__pycache__", ".git"}

        order = 0
        for path in sorted(workspace_dir.rglob("*")):
            if not path.is_file():
                continue
            # Skip files inside excluded directories
            if any(part in skip_dirs for part in path.relative_to(workspace_dir).parts[:-1]):
                continue
            suffix = path.suffix.lower()

            if suffix in CODE_EXTENSIONS:
                code_files.append(
                    GeneratedCode(
                        language="python" if suffix == ".py" else "r",
                        code=path.read_text(errors="replace"),
                        dependencies=[],
                        execution_order=order,
                        description=path.name,
                    )
                )
                order += 1

            elif suffix in TABLE_EXTENSIONS and _is_table_file(path.stem):
                data = _load_table_data(path)
                table_number = _infer_item_number(path.stem, "Table")

                # Attempt to parse as ExtractedTable (JSON template output)
                replicated_et = None
                if suffix == ".json" and isinstance(data, dict):
                    try:
                        # Normalize non-standard JSON formats into ExtractedTable format
                        data = _normalize_table_json(data, table_number)
                        # Sanitize cells before parsing: coerce types that
                        # LLMs sometimes get wrong (e.g. float significance_stars)
                        for cell in data.get("cells", []):
                            if "significance_stars" in cell:
                                try:
                                    cell["significance_stars"] = int(cell["significance_stars"])
                                except (ValueError, TypeError):
                                    cell["significance_stars"] = 0
                        et = ExtractedTable(**data)
                        if et.cells:
                            replicated_et = et
                    except Exception:
                        pass  # treat as generic JSON table data

                tables.append(
                    GeneratedTable(
                        table_number=table_number,
                        data=data,
                        format="csv" if suffix == ".csv" else "json",
                        code_reference=path.name,
                        execution_success=True,
                        replicated_extracted_table=replicated_et,
                    )
                )

            elif suffix in FIGURE_EXTENSIONS and _is_figure_file(path.stem):
                figure_number = _infer_item_number(path.stem, "Figure")
                figures.append(
                    GeneratedFigure(
                        figure_number=figure_number,
                        file_path=str(path),
                        format=suffix.lstrip("."),
                        code_reference=path.name,
                        execution_success=True,
                    )
                )

        logger.info(
            f"Parsed workspace: {len(code_files)} code, "
            f"{len(tables)} tables, {len(figures)} figures"
        )

        return ReplicationResults(
            paper_id=paper_id,
            code_files=code_files,
            tables=tables,
            figures=figures,
        )


def _normalize_table_json(data: dict, table_number: str) -> dict:
    """Normalize non-standard replicator JSON formats into ExtractedTable format.

    Handles formats like:
      - {title, columns, rows: [{label, cells: [{raw_text, numeric_value}]}]}
      - {table_id, results: {col: {row: val}}}
      - {table_id, columns, rows: [{row_label, values: [...]}]}
    Converts them to the standard {table_id, cells: [{row_label, column_label, ...}]} format.
    """
    # Already in standard format with cells
    if "cells" in data and isinstance(data["cells"], list) and data["cells"]:
        # Check first cell has row_label/column_label (not just nested row format)
        first = data["cells"][0]
        if isinstance(first, dict) and ("row_label" in first or "column_label" in first):
            return data

    # Ensure table_id is set
    if "table_id" not in data or not data.get("table_id"):
        data["table_id"] = data.get("title", data.get("table_number", table_number))

    cells = []
    columns = data.get("columns", data.get("column_labels", data.get("column_names", [])))
    row_labels = data.get("row_labels", data.get("row_names", []))

    # Format: {cells: [[{raw_text, numeric_value}, ...], ...]} (row-major grid of cell dicts)
    if "cells" in data and isinstance(data["cells"], list) and data["cells"]:
        first = data["cells"][0]
        if isinstance(first, list):
            for ri, row in enumerate(data["cells"]):
                r_label = row_labels[ri] if ri < len(row_labels) else f"Row {ri}"
                for ci, cell in enumerate(row):
                    c_label = columns[ci] if ci < len(columns) else f"Col {ci}"
                    if isinstance(cell, dict):
                        cells.append({
                            "row_label": str(r_label),
                            "column_label": str(c_label),
                            "row_index": ri,
                            "col_index": ci,
                            "raw_text": cell.get("raw_text", str(cell.get("numeric_value", ""))),
                            "numeric_value": cell.get("numeric_value"),
                        })
                    elif cell is not None:
                        try:
                            nv = float(str(cell).replace(",", "").strip("*() "))
                        except (ValueError, TypeError):
                            nv = None
                        cells.append({
                            "row_label": str(r_label),
                            "column_label": str(c_label),
                            "row_index": ri,
                            "col_index": ci,
                            "raw_text": str(cell),
                            "numeric_value": nv,
                        })
            if cells:
                logger.info(f"Normalized row-major grid for {table_number}: {len(cells)} cells")
                return {
                    "table_id": data.get("table_id", table_number),
                    "column_labels": [str(c) for c in columns] if columns else [],
                    "cells": cells,
                }

    # Format: {rows: [{label, cells: [{raw_text, numeric_value}]}]}
    # Only use rows if they contain data dicts (not just label strings)
    rows_have_data = (
        "rows" in data
        and isinstance(data["rows"], list)
        and data["rows"]
        and isinstance(data["rows"][0], dict)
    )
    if rows_have_data:
        for ri, row in enumerate(data["rows"]):
            if not isinstance(row, dict):
                continue
            row_label = row.get("label", row.get("row_label", f"Row {ri}"))
            row_cells = row.get("cells", row.get("values", []))
            if isinstance(row_cells, list):
                for ci, cell in enumerate(row_cells):
                    col_label = columns[ci] if ci < len(columns) else f"Col {ci}"
                    if isinstance(cell, dict):
                        cells.append({
                            "row_label": str(row_label),
                            "column_label": str(col_label),
                            "row_index": ri,
                            "col_index": ci,
                            "raw_text": cell.get("raw_text", str(cell.get("numeric_value", ""))),
                            "numeric_value": cell.get("numeric_value"),
                        })
                    elif cell is not None:
                        # Scalar value
                        try:
                            nv = float(str(cell).replace(",", "").strip("*() "))
                        except (ValueError, TypeError):
                            nv = None
                        cells.append({
                            "row_label": str(row_label),
                            "column_label": str(col_label),
                            "row_index": ri,
                            "col_index": ci,
                            "raw_text": str(cell),
                            "numeric_value": nv,
                        })

    # Format: {results: [{outcome: ..., coef: ..., se: ...}, ...]} (list of row dicts)
    elif "results" in data and isinstance(data["results"], list):
        for ri, row in enumerate(data["results"]):
            if not isinstance(row, dict):
                continue
            # Use first string-valued field as row label, rest as columns
            row_label = None
            for k, v in row.items():
                if isinstance(v, str):
                    row_label = v
                    break
            if row_label is None:
                row_label = f"Row {ri}"
            ci = 0
            for k, v in row.items():
                if isinstance(v, str) and v == row_label:
                    continue  # skip the label field
                try:
                    nv = float(v) if v is not None else None
                except (ValueError, TypeError):
                    nv = None
                if nv is not None:
                    cells.append({
                        "row_label": str(row_label),
                        "column_label": str(k),
                        "row_index": ri,
                        "col_index": ci,
                        "raw_text": str(v),
                        "numeric_value": nv,
                    })
                    ci += 1

    # Format: {results: {col_label: {row_label: value_or_dict}}} (nested dict)
    elif "results" in data and isinstance(data["results"], dict):
        ri_map = {}
        for ci, (col_label, col_data) in enumerate(data["results"].items()):
            if not isinstance(col_data, dict):
                continue
            for row_label, value in col_data.items():
                # Flatten nested dicts: {coef: 0.5, se: 0.1} -> two cells
                if isinstance(value, dict):
                    for sub_key, sub_val in value.items():
                        combined_label = f"{row_label}_{sub_key}"
                        if combined_label not in ri_map:
                            ri_map[combined_label] = len(ri_map)
                        ri = ri_map[combined_label]
                        try:
                            nv = float(sub_val) if sub_val is not None else None
                        except (ValueError, TypeError):
                            nv = None
                        if nv is not None:
                            cells.append({
                                "row_label": str(combined_label),
                                "column_label": str(col_label),
                                "row_index": ri,
                                "col_index": ci,
                                "raw_text": str(sub_val),
                                "numeric_value": nv,
                            })
                else:
                    if row_label not in ri_map:
                        ri_map[row_label] = len(ri_map)
                    ri = ri_map[row_label]
                    try:
                        nv = float(str(value).replace(",", "").strip("*() "))
                    except (ValueError, TypeError):
                        nv = None
                    if nv is not None:
                        cells.append({
                            "row_label": str(row_label),
                            "column_label": str(col_label),
                            "row_index": ri,
                            "col_index": ci,
                            "raw_text": str(value),
                            "numeric_value": nv,
                        })

    if cells:
        logger.info(f"Normalized non-standard table JSON for {table_number}: {len(cells)} cells")
        return {
            "table_id": data.get("table_id", table_number),
            "column_labels": [str(c) for c in columns] if columns else [],
            "cells": cells,
        }

    return data


def _is_table_file(stem: str) -> bool:
    """Check if a filename stem looks like a table output."""
    return bool(re.match(r"table[_\-\s]?\w", stem, re.IGNORECASE))


def _is_figure_file(stem: str) -> bool:
    """Check if a filename stem looks like a figure output."""
    return bool(re.match(r"(figure|fig)[_\-\s]?\w", stem, re.IGNORECASE))


def _load_table_data(path: Path) -> dict:
    """Load table data from a CSV or JSON file."""
    try:
        if path.suffix.lower() == ".csv":
            import pandas as pd

            df = pd.read_csv(path)
            return json.loads(df.to_json(orient="split"))
        elif path.suffix.lower() == ".json":
            return json.loads(path.read_text())
    except Exception as e:
        logger.warning(f"Could not load table data from {path}: {e}")
    return {"raw": path.read_text(errors="replace")[:5000]}


def _infer_item_number(stem: str, default_prefix: str) -> str:
    """Infer an item number from a controlled filename stem.

    Filenames follow the convention: table_2.1, figure_3.1, table_a.1, etc.
    Also handles Roman numeral filenames: table_i, table_ii, table_viii, etc.
    """
    stem_lower = stem.lower()
    prefix = "Table" if "table" in stem_lower else "Figure"

    # Dotted or letter-prefixed numbering: table_2.1, table_a.1, figure_3.2
    dotted = re.search(
        r"(?:table|figure|fig)[_\-\s]?([a-z]?\d*\.?\d+)", stem_lower
    )
    if dotted:
        return f"{prefix} {dotted.group(1).upper()}"

    # Plain number: table_1, figure_2
    plain = re.search(r"(?:table|figure|fig)[_\-\s]?(\d+)", stem_lower)
    if plain:
        return f"{prefix} {plain.group(1)}"

    # Roman numeral: table_i, table_ii, table_iv, table_viii, etc.
    roman = re.search(
        r"(?:table|figure|fig)[_\-\s]?((?:x{0,3})(?:ix|iv|v?i{0,3}))$", stem_lower
    )
    if roman and roman.group(1):  # non-empty match
        return f"{prefix} {roman.group(1).upper()}"

    return f"{default_prefix} ({stem})"
