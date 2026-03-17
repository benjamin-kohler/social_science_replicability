"""Results Extractor: extracts original table values from paper PDFs.

Runs once per paper (like the methodology extractor), producing a PaperResults
object with structured numeric values for every table. These are later compared
against replicated outputs by the Comparator Agent.

Uses the same OpenAI Responses API pattern as the methodology extractor, with
structured outputs and vision support.
"""

import time
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field

from ..models.schemas import (
    CellValue,
    ExtractedTable,
    PaperResults,
    PaperSummary,
)
from ..utils.logging_utils import get_logger
from ..utils.pdf_parser import extract_text_from_pdf, pdf_to_base64_images

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Structured-output models for OpenAI responses.parse()
# ---------------------------------------------------------------------------

class CellValueResponse(BaseModel):
    """A single cell extracted from a paper table (LLM output)."""

    row_label: str = Field(..., description="Row label exactly as in the paper")
    column_label: str = Field(..., description="Column header exactly as in the paper")
    raw_text: str = Field(
        ...,
        description="Exact text in the cell as printed in the paper, "
        "including significance stars and parentheses",
    )
    numeric_value: Optional[float] = Field(
        default=None,
        description="Numeric value extracted from the cell. "
        "Strip significance stars and parentheses before converting. "
        "None if the cell is non-numeric (e.g., 'Yes', 'No', a label).",
    )
    is_standard_error: bool = Field(
        default=False,
        description="True if this cell contains a standard error (typically in parentheses)",
    )
    significance_stars: int = Field(
        default=0,
        description="Number of significance stars: 0, 1 (*), 2 (**), or 3 (***)",
    )
    significance_level: Optional[float] = Field(
        default=None,
        description="Inferred significance level from stars: 0.1 for *, 0.05 for **, 0.01 for ***. None if no stars.",
    )
    is_string: bool = Field(
        default=False,
        description="True if the cell is non-numeric text (e.g., 'Yes', 'No', a checkmark, a label)",
    )
    row_type: str = Field(
        default="coefficient",
        description="One of: 'coefficient' (main estimate), 'se' (standard error row), "
        "'statistic' (N, R-squared, F-stat, etc.), 'string' (text cell), "
        "'panel_header' (panel separator like 'Panel A: ...')",
    )


class ExtractedTableResponse(BaseModel):
    """Complete extraction of one table from the paper (LLM output)."""

    table_id: str = Field(..., description="Table identifier (e.g., 'Table 1')")
    column_labels: list[str] = Field(..., description="All column headers in order")
    row_labels: list[str] = Field(..., description="All row labels in order (including SE rows)")
    cells: list[CellValueResponse] = Field(
        ...,
        description="Every cell in the table. Include ALL rows and columns.",
    )
    significance_convention: Optional[str] = Field(
        default=None,
        description="The significance convention from the table notes, "
        "e.g., '*** p<0.01, ** p<0.05, * p<0.1'",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Any issues encountered during extraction",
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

RESULTS_EXTRACTOR_SYSTEM = """You are an expert at reading academic paper tables and extracting their numeric values.

Your task: Given pages from a paper containing a specific table, extract EVERY cell value into a structured format.

Rules:
1. Copy the EXACT text from each cell (e.g., "-0.174***", "(0.052)", "Yes", "3,456").
2. For numeric cells, also provide the parsed numeric_value:
   - Strip significance stars before converting: "-0.174***" → numeric_value = -0.174
   - Strip parentheses for standard errors: "(0.052)" → numeric_value = 0.052
   - Handle commas as thousands separators: "3,456" → numeric_value = 3456.0
   - Handle percentage signs: "50.2%" → numeric_value = 50.2
3. Count significance stars accurately: * = 1, ** = 2, *** = 3.
4. Mark standard error rows (in parentheses) with is_standard_error = true and row_type = "se".
5. Mark summary statistics rows (N, R², Observations, F-statistic) with row_type = "statistic".
6. Mark non-numeric cells (Yes/No, checkmarks, labels) with is_string = true and row_type = "string".
7. Include ALL rows and ALL columns — do not skip any cells.
8. If a cell is empty or contains a dash/dot placeholder, set raw_text to the placeholder and numeric_value to null.
9. For the significance_convention, copy it from the table notes (e.g., "*** p<0.01, ** p<0.05, * p<0.1")."""


RESULTS_EXTRACTOR_PROMPT = """Extract all values from {table_id} in this paper.

## Table Structure (from methodology summary):
- Caption: {caption}
- Expected columns: {column_names}
- Expected rows: {row_names}
{panel_structure}

## Paper pages containing this table are attached as images.

Extract EVERY cell. Include coefficient rows, standard error rows, and summary statistic rows (N, R², etc.).
For each cell, provide: raw_text (exact text from paper), numeric_value (parsed float), significance_stars, is_standard_error, row_type."""


RESULTS_EXTRACTOR_PROMPT_TEXT = """Extract all values from {table_id} in this paper.

## Table Structure (from methodology summary):
- Caption: {caption}
- Expected columns: {column_names}
- Expected rows: {row_names}
{panel_structure}

## Paper text (relevant pages):
{paper_pages}

Extract EVERY cell. Include coefficient rows, standard error rows, and summary statistic rows (N, R², etc.).
For each cell, provide: raw_text (exact text from paper), numeric_value (parsed float), significance_stars, is_standard_error, row_type."""


# ---------------------------------------------------------------------------
# ResultsExtractor
# ---------------------------------------------------------------------------

# Reasoning models that don't support temperature
_REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5-mini", "gpt-5-nano", "gpt-5-pro", "gpt-5.2", "gpt-5.3")


class ResultsExtractor:
    """Extracts original numeric results from paper tables.

    Uses the OpenAI Responses API with structured outputs and vision.
    One LLM call per table. Results are cached as PaperResults JSON.
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-5-mini",
        api_key: str = "",
        use_vision: bool = True,
        vision_dpi: int = 200,
    ):
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key
        self.use_vision = use_vision
        self.vision_dpi = vision_dpi
        self._client: Any = None
        self._usage: list[dict] = []
        self._is_reasoning = any(model.startswith(p) for p in _REASONING_PREFIXES)

    @property
    def client(self):
        if self._client is None:
            if self.provider == "openai":
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            else:
                raise ValueError(f"ResultsExtractor currently only supports OpenAI, got: {self.provider}")
        return self._client

    @property
    def usage_summary(self) -> dict:
        total_prompt = sum(u.get("prompt_tokens", 0) for u in self._usage)
        total_completion = sum(u.get("completion_tokens", 0) for u in self._usage)
        return {
            "num_calls": len(self._usage),
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "per_call": self._usage,
        }

    def _record_usage(self, resp) -> None:
        u = getattr(resp, "usage", None)
        if u:
            self._usage.append({
                "prompt_tokens": getattr(u, "input_tokens", 0),
                "completion_tokens": getattr(u, "output_tokens", 0),
                "total_tokens": getattr(u, "total_tokens", 0),
            })

    def run(
        self,
        paper_path: str,
        paper_summary: PaperSummary,
    ) -> PaperResults:
        """Extract original results from all tables in the paper.

        Args:
            paper_path: Path to the paper PDF.
            paper_summary: Methodology summary (for table structure context).

        Returns:
            PaperResults with extracted values for all tables.
        """
        logger.info(f"Extracting original results for {paper_summary.paper_id} ({len(paper_summary.tables)} tables)")

        paper_text = extract_text_from_pdf(paper_path)

        page_images: list[dict] = []
        if self.use_vision:
            try:
                page_images = pdf_to_base64_images(paper_path, dpi=self.vision_dpi)
                logger.info(f"Converted paper to {len(page_images)} page images")
            except Exception as e:
                logger.warning(f"Failed to convert paper to images: {e}")

        extracted_tables: list[ExtractedTable] = []

        for table_spec in paper_summary.tables:
            try:
                extracted = self._extract_table(
                    table_spec, paper_text, page_images,
                )
                extracted_tables.append(extracted)
                logger.info(
                    f"Extracted {table_spec.table_number}: "
                    f"{len(extracted.cells)} cells, "
                    f"{len(extracted.column_labels)} cols, "
                    f"{len(extracted.row_labels)} rows"
                )
            except Exception as e:
                logger.error(f"Failed to extract {table_spec.table_number}: {e}")
                # Add an empty table so we know extraction was attempted
                extracted_tables.append(ExtractedTable(
                    table_id=table_spec.table_number,
                    notes=f"Extraction failed: {e}",
                ))

        return PaperResults(
            paper_id=paper_summary.paper_id,
            tables=extracted_tables,
            extraction_model=self.model,
            extraction_timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _extract_table(
        self,
        table_spec,
        paper_text: str,
        page_images: list[dict],
    ) -> ExtractedTable:
        """Extract values from a single table using structured LLM output."""
        table_id = table_spec.table_number
        panel_info = f"- Panel structure: {table_spec.panel_structure}" if table_spec.panel_structure else ""

        # Find relevant page images
        item_page_images: list[dict] = []
        if page_images:
            from .pdf_page_utils import find_item_pages, select_page_images
            page_nums = find_item_pages(paper_text, table_id)
            item_page_images = select_page_images(page_images, page_nums)

        if item_page_images and self.use_vision:
            prompt = RESULTS_EXTRACTOR_PROMPT.format(
                table_id=table_id,
                caption=table_spec.caption,
                column_names=", ".join(table_spec.column_names),
                row_names=", ".join(table_spec.row_names),
                panel_structure=panel_info,
            )
            result = self._call_vision_structured(prompt, item_page_images)
        else:
            # Text-only fallback
            paper_pages = self._extract_pages(paper_text, table_id)
            prompt = RESULTS_EXTRACTOR_PROMPT_TEXT.format(
                table_id=table_id,
                caption=table_spec.caption,
                column_names=", ".join(table_spec.column_names),
                row_names=", ".join(table_spec.row_names),
                panel_structure=panel_info,
                paper_pages=paper_pages[:8000],
            )
            result = self._call_structured(prompt)

        # Convert response to schema
        cells = [
            CellValue(
                row_label=c.row_label,
                column_label=c.column_label,
                raw_text=c.raw_text,
                numeric_value=c.numeric_value,
                is_standard_error=c.is_standard_error,
                significance_stars=c.significance_stars,
                significance_level=c.significance_level,
                is_string=c.is_string,
                row_type=c.row_type,
            )
            for c in result.cells
        ]

        return ExtractedTable(
            table_id=result.table_id,
            column_labels=result.column_labels,
            row_labels=result.row_labels,
            cells=cells,
            significance_convention=result.significance_convention,
            notes=result.notes,
        )

    def _call_structured(self, prompt: str) -> ExtractedTableResponse:
        """Make a structured LLM call (text only)."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "instructions": RESULTS_EXTRACTOR_SYSTEM,
            "input": prompt,
            "text_format": ExtractedTableResponse,
        }
        if not self._is_reasoning:
            kwargs["temperature"] = 0.0
        resp = self.client.responses.parse(**kwargs)
        self._record_usage(resp)
        return resp.output_parsed

    def _call_vision_structured(
        self, prompt: str, images: list[dict],
    ) -> ExtractedTableResponse:
        """Make a vision + structured LLM call."""
        content_parts: list[dict] = [{"type": "input_text", "text": prompt}]
        for img in images:
            media = img.get("media_type", "image/png")
            content_parts.append({
                "type": "input_image",
                "image_url": f"data:{media};base64,{img['base64']}",
            })
        kwargs: dict[str, Any] = {
            "model": self.model,
            "instructions": RESULTS_EXTRACTOR_SYSTEM,
            "input": [{"type": "message", "role": "user", "content": content_parts}],
            "text_format": ExtractedTableResponse,
        }
        if not self._is_reasoning:
            kwargs["temperature"] = 0.0
        resp = self.client.responses.parse(**kwargs)
        self._record_usage(resp)
        return resp.output_parsed

    @staticmethod
    def _extract_pages(paper_text: str, item_id: str) -> str:
        """Extract text pages mentioning this item (reuses Judge's logic)."""
        from .pdf_page_utils import extract_table_pages
        return extract_table_pages(paper_text, item_id)
