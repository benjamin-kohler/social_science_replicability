"""Parse workspace output into structured ReplicationResults."""

import json
import re
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

            elif suffix in TABLE_EXTENSIONS and _is_table_file(path.stem):
                data = _load_table_data(path)
                table_number = _infer_item_number(path.stem, "Table")
                tables.append(
                    GeneratedTable(
                        table_number=table_number,
                        data=data,
                        format="csv" if suffix == ".csv" else "json",
                        code_reference=path.name,
                        execution_success=True,
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
    """
    stem_lower = stem.lower()

    # Dotted or letter-prefixed numbering: table_2.1, table_a.1, figure_3.2
    dotted = re.search(
        r"(?:table|figure|fig)[_\-\s]?([a-z]?\d*\.?\d+)", stem_lower
    )
    if dotted:
        prefix = "Table" if "table" in stem_lower else "Figure"
        return f"{prefix} {dotted.group(1).upper()}"

    # Plain number: table_1, figure_2
    plain = re.search(r"(?:table|figure|fig)[_\-\s]?(\d+)", stem_lower)
    if plain:
        prefix = "Table" if "table" in stem_lower else "Figure"
        return f"{prefix} {plain.group(1)}"

    return f"{default_prefix} ({stem})"
