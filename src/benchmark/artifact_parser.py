"""Parse freestyle (opencode) workspace output into structured ReplicationResults."""

import json
from pathlib import Path

from ..models.schemas import (
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
    """Parses a freestyle workspace into ReplicationResults.

    Scans the workspace directory for generated code, tables, and figures
    using filename heuristics to match them to paper items.
    """

    @staticmethod
    def parse(workspace_dir: Path, paper_id: str) -> ReplicationResults:
        """Parse workspace files into ReplicationResults.

        Args:
            workspace_dir: Directory containing freestyle run output.
            paper_id: Paper identifier.

        Returns:
            ReplicationResults with discovered artifacts.
        """
        code_files = []
        tables = []
        figures = []

        if not workspace_dir.exists():
            logger.warning(f"Workspace directory does not exist: {workspace_dir}")
            return ReplicationResults(paper_id=paper_id)

        # Load number maps from methodology_summary.json if available
        table_number_map, figure_number_map = _load_number_maps(workspace_dir)

        # Scan for code files
        order = 0
        for path in sorted(workspace_dir.rglob("*")):
            if not path.is_file():
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

            elif suffix in TABLE_EXTENSIONS:
                data = _load_table_data(path)
                table_number = _infer_item_number(path.stem, "Table", table_number_map)
                tables.append(
                    GeneratedTable(
                        table_number=table_number,
                        data=data,
                        format="csv" if suffix == ".csv" else "json",
                        code_reference=path.name,
                        execution_success=True,
                    )
                )

            elif suffix in FIGURE_EXTENSIONS:
                figure_number = _infer_item_number(path.stem, "Figure", figure_number_map)
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


def _load_number_maps(workspace_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Load table/figure number maps from methodology_summary.json.

    Builds maps from sequential position ("1", "2", ...) to paper numbering
    ("Table 2.1", "Table 2.2", ...) by reading the tables/figures arrays.

    Returns:
        Tuple of (table_number_map, figure_number_map).
    """
    table_map: dict[str, str] = {}
    figure_map: dict[str, str] = {}

    summary_path = workspace_dir / "methodology_summary.json"
    if not summary_path.exists():
        return table_map, figure_map

    try:
        summary = json.loads(summary_path.read_text())

        for i, table in enumerate(summary.get("tables", []), start=1):
            paper_num = table.get("table_number", f"Table {i}")
            table_map[str(i)] = paper_num

        for i, figure in enumerate(summary.get("figures", []), start=1):
            paper_num = figure.get("figure_number", f"Figure {i}")
            figure_map[str(i)] = paper_num

        if table_map or figure_map:
            logger.info(
                f"Loaded number maps: {len(table_map)} tables, {len(figure_map)} figures"
            )
    except Exception as e:
        logger.warning(f"Could not load number maps from {summary_path}: {e}")

    return table_map, figure_map


def _infer_item_number(
    stem: str, default_prefix: str, number_map: dict[str, str] | None = None
) -> str:
    """Infer an item number (e.g., 'Table 2.1') from a filename stem.

    Heuristics (in priority order):
    1. Dotted paper numbering in filename: table_2.1 -> 'Table 2.1'
    2. Sequential number with number_map lookup: table_1 -> number_map["1"]
    3. Sequential number without map: table_1 -> 'Table 1'
    4. Fallback: use filename
    """
    import re

    stem_lower = stem.lower()

    # 1. Check for dotted numbering (e.g., table_2.1, table_3.2)
    table_dotted = re.search(r"table[_\-\s]?(\d+\.\d+)", stem_lower)
    if table_dotted:
        return f"Table {table_dotted.group(1)}"

    figure_dotted = re.search(r"(?:figure|fig)[_\-\s]?(\d+\.\d+)", stem_lower)
    if figure_dotted:
        return f"Figure {figure_dotted.group(1)}"

    # 2. Match sequential patterns like table_1, table1, table-1
    table_match = re.search(r"table[_\-\s]?(\d+)", stem_lower)
    if table_match:
        seq_num = table_match.group(1)
        if number_map and seq_num in number_map:
            return number_map[seq_num]
        return f"Table {seq_num}"

    figure_match = re.search(r"(?:figure|fig)[_\-\s]?(\d+)", stem_lower)
    if figure_match:
        seq_num = figure_match.group(1)
        if number_map and seq_num in number_map:
            return number_map[seq_num]
        return f"Figure {seq_num}"

    # 3. Fallback: use filename
    return f"{default_prefix} ({stem})"
