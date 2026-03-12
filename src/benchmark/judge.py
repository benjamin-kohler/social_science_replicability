"""Unified judge for grading replication quality.

Uses plain OpenAI/Anthropic SDK calls — no LangChain, no LangGraph.
Each item (table or figure) is graded in a single LLM call that produces
both a verification grade and (if non-A) a discrepancy explanation.

For OpenAI providers, the judge uses structured outputs via ``responses.parse()``
with Pydantic models, guaranteeing valid JSON.  For Anthropic providers, it
falls back to prompt-based JSON generation with ``_parse_json()`` + retry.
"""

import base64
import json
import re
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from ..models.schemas import (
    CellComparison,
    DiscrepancyAnalysis,
    ExplanationReport,
    ExtractedTable,
    ItemVerification,
    PaperResults,
    PaperSummary,
    ReplicationGrade,
    ReplicationResults,
    TableComparison,
    VerificationReport,
)
from ..utils.logging_utils import get_logger
from ..utils.pdf_parser import extract_text_from_pdf, pdf_to_base64_images
from .comparator import ComparatorAgent
from .grader import grade_table

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Structured-output Pydantic models (used by OpenAI responses.parse)
# ---------------------------------------------------------------------------

class FigureJudgment(BaseModel):
    """Structured judge output for a figure replication."""
    grade: Literal["A", "B", "C", "D", "E", "F"] = Field(description="Replication grade")
    comparison_notes: str = Field(description="Detailed comparison of patterns, trends, and values")
    key_findings_match: bool = Field(description="Whether the main patterns and trends match")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """You are a judge evaluating how closely a replicated result matches the original.

Grading scale:
- A: Results match within numerical precision (< 1% difference).
- B: Same direction with small discrepancies (1-10% difference).
- C: Same direction with moderate discrepancies (10-20% difference).
- D: Same direction but large discrepancies (20-50% difference).
- E: Results differ meaningfully — different significance, direction, or >50% difference.
- F: Not comparable — missing output, incompatible format, or unable to verify.

Focus on substance, not formatting or presentation. For figures, compare
patterns and trends, not exact visual appearance."""

FIGURE_JUDGE_PROMPT = """How closely does the replicated {item_id} match the original?

{vision_note}

## Original Paper (relevant pages):
{paper_pages}

Compare the replicated figure against the original. Assess whether the patterns,
trends, axis ranges, and data values match. Assign a grade (A-F) based on the
grading scale.

Respond with ONLY this JSON (no other text):
{{
    "grade": "A/B/C/D/E/F",
    "comparison_notes": "Detailed comparison of patterns, trends, and values",
    "key_findings_match": true
}}"""


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------


_JSON_RETRY_HINT = (
    "\n\nIMPORTANT: Your previous response was not valid JSON and could not be parsed. "
    "Respond with ONLY the JSON object — no markdown fences, no explanation, no extra text."
)


_STRUCTURED_PROVIDERS = {"openai"}  # providers that support responses.parse()


class Judge:
    """Grades replication outputs against the original paper.

    Uses plain OpenAI or Anthropic SDK — one LLM call per item.
    Tracks token usage across all LLM calls.
    """

    # Reasoning models don't support temperature
    _REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5-mini", "gpt-5-nano", "gpt-5-pro", "gpt-5.2", "gpt-5.3")
    # All OpenAI calls use the Responses API

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str,
        use_vision: bool = True,
        comparator: ComparatorAgent | None = None,
    ):
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key
        self.use_vision = use_vision
        self._comparator = comparator
        self._client: Any = None
        self._usage: list[dict] = []  # per-call token usage log
        self._is_reasoning = any(model.startswith(p) for p in self._REASONING_PREFIXES)

    # -- SDK client ---------------------------------------------------------

    @property
    def client(self):
        if self._client is None:
            if self.provider == "openai":
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            elif self.provider == "anthropic":
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.api_key)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        return self._client

    def _call_llm(self, system: str, prompt: str) -> str:
        """Make a single LLM call and return the text response."""
        if self.provider == "openai":
            kwargs: dict[str, Any] = {
                "model": self.model,
                "instructions": system,
                "input": prompt,
            }
            if not self._is_reasoning:
                kwargs["temperature"] = 0.0
            resp = self.client.responses.create(**kwargs)
            self._record_usage_responses(resp)
            return resp.output_text
        else:  # anthropic
            resp = self.client.messages.create(
                model=self.model,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=16384,
            )
            self._record_usage_anthropic(resp)
            return resp.content[0].text

    def _call_llm_vision(
        self, system: str, prompt: str, image_b64: str, media_type: str,
    ) -> str:
        """Make a vision LLM call with an image."""
        if self.provider == "openai":
            kwargs: dict[str, Any] = {
                "model": self.model,
                "instructions": system,
                "input": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:{media_type};base64,{image_b64}",
                    },
                ],
            }
            if not self._is_reasoning:
                kwargs["temperature"] = 0.0
            resp = self.client.responses.create(**kwargs)
            self._record_usage_responses(resp)
            return resp.output_text
        else:  # anthropic
            resp = self.client.messages.create(
                model=self.model,
                system=system,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_b64,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                temperature=0.0,
                max_tokens=16384,
            )
            self._record_usage_anthropic(resp)
            return resp.content[0].text

    # -- Multi-image vision calls ----------------------------------------------

    def _call_llm_vision_multi(
        self, system: str, prompt: str, images: list[dict],
    ) -> str:
        """Make a vision LLM call with multiple images.

        Args:
            images: list of dicts with 'base64' and optionally 'media_type' keys.
                    Page-image dicts (from pdf_to_base64_images) are also accepted
                    — media_type defaults to 'image/png'.
        """
        if self.provider == "openai":
            content_parts: list[dict] = [{"type": "input_text", "text": prompt}]
            for img in images:
                media = img.get("media_type", "image/png")
                content_parts.append({
                    "type": "input_image",
                    "image_url": f"data:{media};base64,{img['base64']}",
                })
            kwargs: dict[str, Any] = {
                "model": self.model,
                "instructions": system,
                "input": [{"type": "message", "role": "user", "content": content_parts}],
            }
            if not self._is_reasoning:
                kwargs["temperature"] = 0.0
            resp = self.client.responses.create(**kwargs)
            self._record_usage_responses(resp)
            return resp.output_text
        else:  # anthropic
            content: list[dict] = []
            for img in images:
                media = img.get("media_type", "image/png")
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": media, "data": img["base64"]},
                })
            content.append({"type": "text", "text": prompt})
            resp = self.client.messages.create(
                model=self.model,
                system=system,
                messages=[{"role": "user", "content": content}],
                temperature=0.0,
                max_tokens=16384,
            )
            self._record_usage_anthropic(resp)
            return resp.content[0].text

    # -- Structured-output calls (OpenAI only) --------------------------------

    def _call_llm_structured(self, system: str, prompt: str, response_model: type) -> BaseModel:
        """Make an OpenAI responses.parse() call, returning a Pydantic model instance."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "instructions": system,
            "input": prompt,
            "text_format": response_model,
        }
        if not self._is_reasoning:
            kwargs["temperature"] = 0.0
        resp = self.client.responses.parse(**kwargs)
        self._record_usage_responses(resp)
        return resp.output_parsed

    def _call_llm_vision_structured(
        self, system: str, prompt: str, image_b64: str, media_type: str,
        response_model: type,
    ) -> BaseModel:
        """Make an OpenAI vision + structured-output call (single image)."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "instructions": system,
            "input": [{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": prompt},
                {
                    "type": "input_image",
                    "image_url": f"data:{media_type};base64,{image_b64}",
                },
            ]}],
            "text_format": response_model,
        }
        if not self._is_reasoning:
            kwargs["temperature"] = 0.0
        resp = self.client.responses.parse(**kwargs)
        self._record_usage_responses(resp)
        return resp.output_parsed

    def _call_llm_vision_multi_structured(
        self, system: str, prompt: str, images: list[dict],
        response_model: type,
    ) -> BaseModel:
        """Make an OpenAI vision + structured-output call with multiple images."""
        content_parts: list[dict] = [{"type": "input_text", "text": prompt}]
        for img in images:
            media = img.get("media_type", "image/png")
            content_parts.append({
                "type": "input_image",
                "image_url": f"data:{media};base64,{img['base64']}",
            })
        kwargs: dict[str, Any] = {
            "model": self.model,
            "instructions": system,
            "input": [{"type": "message", "role": "user", "content": content_parts}],
            "text_format": response_model,
        }
        if not self._is_reasoning:
            kwargs["temperature"] = 0.0
        resp = self.client.responses.parse(**kwargs)
        self._record_usage_responses(resp)
        return resp.output_parsed

    # -- Token usage tracking ------------------------------------------------

    def _record_usage_responses(self, resp) -> None:
        """Record token usage from an OpenAI Responses API response."""
        u = getattr(resp, "usage", None)
        if u:
            self._usage.append({
                "prompt_tokens": getattr(u, "input_tokens", 0),
                "completion_tokens": getattr(u, "output_tokens", 0),
                "total_tokens": getattr(u, "total_tokens", 0),
            })

    def _record_usage_anthropic(self, resp) -> None:
        """Record token usage from an Anthropic response."""
        u = getattr(resp, "usage", None)
        if u:
            self._usage.append({
                "prompt_tokens": getattr(u, "input_tokens", 0),
                "completion_tokens": getattr(u, "output_tokens", 0),
                "total_tokens": getattr(u, "input_tokens", 0) + getattr(u, "output_tokens", 0),
            })

    @property
    def usage_summary(self) -> dict:
        """Aggregate token usage across all LLM calls."""
        total_prompt = sum(u["prompt_tokens"] for u in self._usage)
        total_completion = sum(u["completion_tokens"] for u in self._usage)
        return {
            "num_calls": len(self._usage),
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "per_call": self._usage,
        }

    def _parse_json_with_retry(self, system: str, prompt: str) -> dict:
        """Call LLM and parse JSON, retrying once on parse failure (Anthropic path)."""
        last_error = None
        for attempt in range(2):
            try:
                call_prompt = prompt if attempt == 0 else prompt + _JSON_RETRY_HINT
                return self._parse_json(self._call_llm(system, call_prompt))
            except Exception as e:
                last_error = e
                if attempt == 0:
                    logger.warning(f"JSON parse failed, retrying: {e}")
        raise last_error  # type: ignore[misc]

    def _parse_json(self, text: str) -> dict:
        """Parse JSON from LLM response, stripping markdown fences."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```\s*$", "", cleaned)
            cleaned = cleaned.strip()
        if cleaned.startswith("{"):
            return json.loads(cleaned)
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return json.loads(match.group())
        raise ValueError("No JSON found in LLM response")

    # -- Main entry point ---------------------------------------------------

    def run(
        self,
        paper_path: str,
        paper_summary: PaperSummary,
        replication_results: ReplicationResults,
        paper_results: PaperResults | None = None,
        replication_package_path: str | None = None,
    ) -> tuple[VerificationReport, ExplanationReport | None]:
        """Judge all replicated items.

        Args:
            paper_path: Path to the original paper PDF.
            paper_summary: Methodology summary.
            replication_results: Replicated outputs to judge.
            paper_results: Extracted original table values (for programmatic
                comparison). If None, tables fall back to LLM-only judging.
            replication_package_path: Deprecated, kept for backward compat. Ignored.

        Returns:
            (VerificationReport, ExplanationReport or None if all grades are A).
        """
        logger.info(f"Judging replication for: {replication_results.paper_id}")

        paper_text = extract_text_from_pdf(paper_path)

        # Convert paper PDF to page images once (used for vision calls)
        page_images: list[dict] = []
        if self.use_vision:
            try:
                page_images = pdf_to_base64_images(paper_path, dpi=150)
                logger.info(f"Converted paper to {len(page_images)} page images for vision judging")
            except Exception as e:
                logger.warning(f"Failed to convert paper to images, vision disabled: {e}")

        # Build lookups
        table_specs = {t.table_number: t for t in paper_summary.tables}
        figure_specs = {f.figure_number: f for f in paper_summary.figures}

        item_verifications: list[ItemVerification] = []
        discrepancy_analyses: list[DiscrepancyAnalysis] = []

        # Judge tables (programmatic via comparator if paper_results available)
        for gen_table in replication_results.tables:
            spec = table_specs.get(gen_table.table_number)
            original_table = paper_results.get_table(gen_table.table_number) if paper_results else None

            verification, analysis = self._judge_table(
                gen_table, spec, paper_text, page_images,
                original_table=original_table,
            )
            item_verifications.append(verification)
            if analysis:
                discrepancy_analyses.append(analysis)

        # Judge figures (LLM vision — unchanged)
        for gen_figure in replication_results.figures:
            spec = figure_specs.get(gen_figure.figure_number)

            verification, analysis = self._judge_figure(
                gen_figure, spec, paper_text, page_images,
            )
            item_verifications.append(verification)
            if analysis:
                discrepancy_analyses.append(analysis)

        # Build reports
        overall_grade = self._calculate_overall_grade(item_verifications)
        verification_report = VerificationReport(
            paper_id=replication_results.paper_id,
            overall_grade=overall_grade,
            item_verifications=item_verifications,
            summary=self._generate_summary(item_verifications, overall_grade),
        )

        explanation_report = None
        if discrepancy_analyses:
            explanation_report = ExplanationReport(
                paper_id=replication_results.paper_id,
                analyses=discrepancy_analyses,
                overall_assessment=self._generate_overall_assessment(
                    discrepancy_analyses, verification_report,
                ),
                recommendations=self._generate_recommendations(discrepancy_analyses),
            )

        logger.info(
            f"Judging complete: overall={overall_grade.value}, "
            f"{len(discrepancy_analyses)} discrepancies"
        )
        return verification_report, explanation_report

    # -- Per-item judging ---------------------------------------------------

    def _judge_table(
        self,
        gen_table,
        spec,
        paper_text: str,
        page_images: list[dict] | None = None,
        original_table: ExtractedTable | None = None,
    ) -> tuple[ItemVerification, DiscrepancyAnalysis | None]:
        """Judge a single table.

        If original_table (from ResultsExtractor) and a comparator are available,
        uses programmatic cell-by-cell comparison. Otherwise falls back to the
        LLM-based approach (deprecated path for backward compat).
        """
        item_id = gen_table.table_number
        logger.info(f"Judging {item_id}")

        # Failed execution → unverifiable
        if not gen_table.execution_success:
            return (
                ItemVerification(
                    item_id=item_id, item_type="table",
                    grade=ReplicationGrade.F,
                    comparison_notes=f"Replication failed: {gen_table.error_message}",
                    unverifiable=True,
                ),
                None,
            )

        # --- Comparator path (preferred) ---
        if original_table and original_table.cells and self._comparator:
            return self._judge_table_comparator(
                gen_table, original_table, item_id,
            )

        # --- Fallback: LLM-only judging (no extracted results available) ---
        logger.warning(f"No extracted original values for {item_id}, using LLM-only judging")
        return self._judge_table_llm_fallback(
            gen_table, spec, paper_text, page_images,
        )

    def _judge_table_comparator(
        self,
        gen_table,
        original_table: ExtractedTable,
        item_id: str,
    ) -> tuple[ItemVerification, DiscrepancyAnalysis | None]:
        """Judge a table using the Comparator Agent for cell-by-cell comparison."""
        try:
            # Convert replicated table data to CSV string
            import pandas as pd
            import io

            data = gen_table.data
            if isinstance(data, dict):
                # pandas JSON orient="split" format
                if "columns" in data and "data" in data:
                    df = pd.DataFrame(**{k: data[k] for k in ("columns", "data", "index") if k in data})
                else:
                    df = pd.DataFrame(data)
            else:
                df = pd.DataFrame(data)

            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=True)
            replicated_csv = csv_buffer.getvalue()

            # Run comparator (objective metrics only)
            comparison = self._comparator.compare_table(original_table, replicated_csv)

            # Apply deterministic grading
            comparison = grade_table(comparison)

            # Convert overall grade to ReplicationGrade
            try:
                grade = ReplicationGrade(comparison.overall_grade)
            except ValueError:
                grade = ReplicationGrade.F

            # Build numerical_differences dict for backward compat
            num_diffs = {}
            if comparison.cell_comparisons:
                pct_diffs = [
                    c.percent_difference for c in comparison.cell_comparisons
                    if c.percent_difference is not None
                ]
                if pct_diffs:
                    num_diffs = {
                        "max_difference_percent": max(pct_diffs),
                        "mean_difference_percent": sum(pct_diffs) / len(pct_diffs),
                        "num_cells_compared": len(comparison.cell_comparisons),
                    }

            verification = ItemVerification(
                item_id=item_id,
                item_type="table",
                grade=grade,
                comparison_notes=comparison.summary,
                numerical_differences=num_diffs or None,
                key_findings_match=grade in (ReplicationGrade.A, ReplicationGrade.B),
                table_comparison=comparison,
            )

            analysis = None
            if grade != ReplicationGrade.A:
                analysis = DiscrepancyAnalysis(
                    item_id=item_id,
                    grade=grade,
                    description_of_discrepancy=comparison.summary,
                    likely_causes=[comparison.alignment_notes] if comparison.alignment_notes else [],
                    is_identifiable=True,
                    fault_attribution="unclear",
                    confidence="medium",
                    supporting_evidence=f"Cell-by-cell comparison: {len(comparison.cell_comparisons)} cells compared",
                )

            return verification, analysis

        except Exception as e:
            logger.error(f"Comparator failed for {item_id}: {e}")
            return (
                ItemVerification(
                    item_id=item_id, item_type="table",
                    grade=ReplicationGrade.F,
                    comparison_notes=f"Comparator error: {e}",
                    judge_error=True,
                ),
                None,
            )

    def _judge_table_llm_fallback(
        self,
        gen_table,
        spec,
        paper_text: str,
        page_images: list[dict] | None = None,
    ) -> tuple[ItemVerification, DiscrepancyAnalysis | None]:
        """Fallback: judge a table using LLM when no extracted values are available."""
        item_id = gen_table.table_number

        paper_pages = self._extract_table_pages(paper_text, item_id)
        template = spec.template_markdown if spec and spec.template_markdown else "Not available"
        replicated_data = json.dumps(gen_table.data, indent=2)[:5000]

        # Simplified prompt without code
        prompt = f"""How closely does the replicated {item_id} match the original?

## Original Paper (relevant pages):
{paper_pages[:8000]}

## Expected Table Structure (from methodology summary):
{template[:3000]}

## Replicated Output (CSV data):
{replicated_data}

Compare the replicated values against the original paper. Assign a grade (A-F)
based on the grading scale.

Respond with ONLY this JSON (no other text):
{{
    "grade": "A/B/C/D/E/F",
    "comparison_notes": "Detailed comparison of the results",
    "key_findings_match": true
}}"""

        # Select relevant page images for this table
        item_page_images: list[dict] = []
        if page_images:
            page_nums = self._find_item_pages(paper_text, item_id)
            item_page_images = self._select_page_images(page_images, page_nums)

        try:
            if item_page_images and self.use_vision:
                if self.provider in _STRUCTURED_PROVIDERS:
                    parsed = self._call_llm_vision_multi_structured(
                        JUDGE_SYSTEM_PROMPT, prompt, item_page_images, FigureJudgment,
                    )
                    resp = parsed.model_dump()
                else:
                    raw = self._call_llm_vision_multi(
                        JUDGE_SYSTEM_PROMPT, prompt, item_page_images,
                    )
                    resp = self._parse_json(raw)
            else:
                if self.provider in _STRUCTURED_PROVIDERS:
                    parsed = self._call_llm_structured(
                        JUDGE_SYSTEM_PROMPT, prompt, FigureJudgment,
                    )
                    resp = parsed.model_dump()
                else:
                    resp = self._parse_json_with_retry(JUDGE_SYSTEM_PROMPT, prompt)
            return self._parse_judge_response(resp, item_id, "table")
        except Exception as e:
            logger.error(f"Judge call failed for {item_id}: {e}")
            return (
                ItemVerification(
                    item_id=item_id, item_type="table",
                    grade=ReplicationGrade.F,
                    comparison_notes=f"Judge error: {e}",
                    judge_error=True,
                ),
                None,
            )

    def _judge_figure(
        self,
        gen_figure,
        spec,
        paper_text: str,
        page_images: list[dict] | None = None,
    ) -> tuple[ItemVerification, DiscrepancyAnalysis | None]:
        """Judge a single figure using LLM vision comparison."""
        item_id = gen_figure.figure_number
        logger.info(f"Judging {item_id}")

        if not gen_figure.execution_success:
            return (
                ItemVerification(
                    item_id=item_id, item_type="figure",
                    grade=ReplicationGrade.F,
                    comparison_notes=f"Replication failed: {gen_figure.error_message}",
                    unverifiable=True,
                ),
                None,
            )

        fig_path = Path(gen_figure.file_path)
        if not fig_path.exists():
            return (
                ItemVerification(
                    item_id=item_id, item_type="figure",
                    grade=ReplicationGrade.F,
                    comparison_notes="Figure file not found",
                    unverifiable=True,
                ),
                None,
            )

        paper_pages = self._extract_table_pages(paper_text, item_id)

        prompt = FIGURE_JUDGE_PROMPT.format(
            item_id=item_id,
            paper_pages=paper_pages[:8000],
            vision_note="The replicated figure image and original paper pages are attached for visual comparison.",
        )

        # Collect all images: original paper pages + replicated figure
        item_page_images: list[dict] = []
        if page_images:
            page_nums = self._find_item_pages(paper_text, item_id)
            item_page_images = self._select_page_images(page_images, page_nums)

        try:
            if not self.use_vision:
                raise RuntimeError("Vision disabled by configuration")
            # Read replicated figure
            with open(fig_path, "rb") as f:
                repl_img_b64 = base64.b64encode(f.read()).decode("utf-8")
            suffix = fig_path.suffix.lower()
            repl_media = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(
                suffix, "image/png"
            )

            # Build all vision images: paper pages first, then replicated figure
            all_images = [
                {"base64": img["base64"], "media_type": "image/png"}
                for img in item_page_images
            ]
            all_images.append({"base64": repl_img_b64, "media_type": repl_media})

            if self.provider in _STRUCTURED_PROVIDERS:
                parsed = self._call_llm_vision_multi_structured(
                    JUDGE_SYSTEM_PROMPT, prompt, all_images, FigureJudgment,
                )
                resp = parsed.model_dump()
            else:
                raw = self._call_llm_vision_multi(
                    JUDGE_SYSTEM_PROMPT, prompt, all_images,
                )
                resp = self._parse_json(raw)
        except Exception as e:
            logger.warning(f"Vision comparison failed for {item_id}, falling back to text: {e}")
            # Text-only fallback
            text_prompt = prompt.replace(
                "The replicated figure image and original paper pages are attached for visual comparison.",
                "Note: Visual comparison not available. Assess based on code and description.",
            )
            try:
                if self.provider in _STRUCTURED_PROVIDERS:
                    parsed = self._call_llm_structured(
                        JUDGE_SYSTEM_PROMPT, text_prompt, FigureJudgment,
                    )
                    resp = parsed.model_dump()
                else:
                    resp = self._parse_json_with_retry(JUDGE_SYSTEM_PROMPT, text_prompt)
            except Exception as e2:
                logger.error(f"Judge call failed for {item_id}: {e2}")
                return (
                    ItemVerification(
                        item_id=item_id, item_type="figure",
                        grade=ReplicationGrade.F,
                        comparison_notes=f"Judge error: {e2}",
                        judge_error=True,
                    ),
                    None,
                )

        return self._parse_judge_response(resp, item_id, "figure")

    # -- Response parsing ---------------------------------------------------

    @staticmethod
    def _parse_judge_response(
        resp: dict, item_id: str, item_type: str,
    ) -> tuple[ItemVerification, DiscrepancyAnalysis | None]:
        """Parse judge JSON into verification + optional stub analysis.

        The judge only assesses similarity (grade + notes). Detailed
        discrepancy analysis (fault attribution, causes) is left to the
        explainer, so we create only a minimal stub here for non-A items.
        """
        grade = ReplicationGrade(resp.get("grade", "F"))

        verification = ItemVerification(
            item_id=item_id,
            item_type=item_type,
            grade=grade,
            comparison_notes=resp.get("comparison_notes", ""),
            numerical_differences=resp.get("numerical_differences"),
            key_findings_match=resp.get("key_findings_match"),
        )

        analysis = None
        if grade != ReplicationGrade.A:
            analysis = DiscrepancyAnalysis(
                item_id=item_id,
                grade=grade,
                description_of_discrepancy=resp.get("comparison_notes", ""),
                likely_causes=[],
                is_identifiable=False,
                fault_attribution="unclear",
                confidence="low",
                supporting_evidence=None,
            )

        return verification, analysis

    # -- Helpers (migrated from verifier.py / explainer.py) -----------------

    @staticmethod
    def _find_item_pages(paper_text: str, item_id: str) -> list[int]:
        from .pdf_page_utils import find_item_pages
        return find_item_pages(paper_text, item_id)

    @staticmethod
    def _extract_table_pages(paper_text: str, item_id: str) -> str:
        from .pdf_page_utils import extract_table_pages
        return extract_table_pages(paper_text, item_id)

    @staticmethod
    def _select_page_images(
        page_images: list[dict], page_nums: list[int],
    ) -> list[dict]:
        from .pdf_page_utils import select_page_images
        return select_page_images(page_images, page_nums)

    @staticmethod
    def _calculate_overall_grade(
        verifications: list[ItemVerification],
    ) -> ReplicationGrade:
        """Average verifiable item grades to an overall grade.

        Items flagged as ``unverifiable`` or ``judge_error`` are excluded from
        the average so that execution failures or judge glitches don't drag
        down the overall score.  If *all* items are excluded, the grade is F.
        """
        grades = [
            v.grade for v in verifications
            if not v.unverifiable and not v.judge_error
        ]
        if not grades:
            return ReplicationGrade.F
        values = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1, "F": 0}
        avg = sum(values[g.value] for g in grades) / len(grades)
        if avg >= 4.5:
            return ReplicationGrade.A
        if avg >= 3.5:
            return ReplicationGrade.B
        if avg >= 2.5:
            return ReplicationGrade.C
        if avg >= 1.5:
            return ReplicationGrade.D
        if avg >= 0.5:
            return ReplicationGrade.E
        return ReplicationGrade.F

    @staticmethod
    def _generate_summary(
        verifications: list[ItemVerification], overall_grade: ReplicationGrade,
    ) -> str:
        """Generate a human-readable verification summary."""
        counts: dict[str, int] = {}
        for v in verifications:
            counts[v.grade.value] = counts.get(v.grade.value, 0) + 1

        parts = [
            f"Overall replication grade: {overall_grade.value}",
            f"Total items verified: {len(verifications)}",
            "Grade distribution:",
        ]
        for g in ["A", "B", "C", "D", "E", "F"]:
            if g in counts:
                parts.append(f"  - Grade {g}: {counts[g]} items")

        unverifiable = [v for v in verifications if v.unverifiable]
        if unverifiable:
            parts.append(f"\nUnverifiable items (excluded from overall grade): {len(unverifiable)}")
            for v in unverifiable:
                parts.append(f"  - {v.item_id}: {v.comparison_notes[:100]}...")

        judge_errors = [v for v in verifications if v.judge_error]
        if judge_errors:
            parts.append(f"\nJudge errors (excluded from overall grade): {len(judge_errors)}")
            for v in judge_errors:
                parts.append(f"  - {v.item_id}: {v.comparison_notes[:100]}...")

        issues = [v for v in verifications if v.grade.value in ("D", "E", "F") and not v.judge_error and not v.unverifiable]
        if issues:
            parts.append("\nItems with significant issues:")
            for v in issues:
                parts.append(f"  - {v.item_id}: {v.comparison_notes[:100]}...")

        return "\n".join(parts)

    @staticmethod
    def _generate_overall_assessment(
        analyses: list[DiscrepancyAnalysis], report: VerificationReport,
    ) -> str:
        """Generate overall assessment of discrepancies."""
        if not analyses:
            return "All items received grade A. Replication was fully successful."

        attributions: dict[str, int] = {}
        for a in analyses:
            attributions[a.fault_attribution] = attributions.get(a.fault_attribution, 0) + 1

        parts = [
            f"Of {len(report.item_verifications)} items verified:",
            f"- {len(analyses)} had discrepancies requiring explanation",
        ]
        for attr, count in sorted(attributions.items(), key=lambda x: -x[1]):
            parts.append(f"- {count} attributed to: {attr}")

        identifiable = sum(1 for a in analyses if a.is_identifiable)
        parts.append(f"\n{identifiable}/{len(analyses)} discrepancies have identifiable causes.")
        return "\n".join(parts)

    @staticmethod
    def _generate_recommendations(analyses: list[DiscrepancyAnalysis]) -> list[str]:
        """Generate recommendations based on discrepancy patterns."""
        all_causes = [c for a in analyses for c in a.likely_causes]
        recs = []
        if any("software" in c.lower() for c in all_causes):
            recs.append("Consider using the same statistical software as the original paper")
        if any("ambiguous" in c.lower() or "unclear" in c.lower() for c in all_causes):
            recs.append("Request clarification from original authors on ambiguous methodology")
        if any("data" in c.lower() for c in all_causes):
            recs.append("Verify that the same version of the data is being used")
        if not recs:
            recs = [
                "Review methodology descriptions for potential ambiguities",
                "Compare data processing steps in detail",
                "Check for version differences in statistical packages",
            ]
        return recs

