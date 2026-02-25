"""Agent 3: Verifier - Compares replicated results with original paper."""

import base64
import json
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from langchain_core.language_models import BaseChatModel

from ..models.schemas import (
    ReplicationResults,
    VerificationReport,
    ItemVerification,
    ReplicationGrade,
)
from ..models.config import Config
from ..utils.pdf_parser import extract_text_from_pdf, extract_tables_from_pdf
from ..utils.comparison import compare_tables, calculate_replication_grade
from ..utils.logging_utils import get_logger
from .base import BaseAgent

logger = get_logger(__name__)


VERIFIER_SYSTEM_PROMPT = """You are an expert at verifying research replications.

Your task is to compare replicated results with original paper results and assign grades.

Grading Scale:
- A: Fully replicated. Results match within numerical precision (< 1% difference).
- B: Same direction of effects with small discrepancies (1-5% difference).
- C: Same direction of effects with large discrepancies (5-20% difference).
- D: Results differ meaningfully - different significance, direction, or magnitude.
- F: Not comparable - missing output, incompatible format, or unable to verify.

Important:
- Focus on SUBSTANCE, not formatting or presentation
- Compare coefficients, standard errors, significance levels, and key statistics
- For figures, compare patterns and trends, not exact visual appearance
- Note any differences in sample size or methodology that could explain discrepancies"""


VERIFICATION_PROMPT = """Compare the replicated results with the original paper results.

## Original Paper Content:
{original_content}

## Replicated Result:
{replicated_content}

## Item Being Compared:
{item_type}: {item_id}

Provide your verification in this JSON format:
{{
    "item_id": "{item_id}",
    "item_type": "{item_type}",
    "grade": "A/B/C/D/F",
    "comparison_notes": "Detailed comparison of the results",
    "numerical_differences": {{
        "max_difference_percent": 0.0,
        "key_differences": ["list of specific differences"]
    }},
    "key_findings_match": true/false
}}

Focus on whether the substantive findings match, not formatting."""


GROUND_TRUTH_EXTRACTION_PROMPT = """Extract ALL numerical values from the following table in the paper.

## Table: {table_number}
{caption}

## Relevant pages from the paper:
{page_text}

## Expected structure:
- Columns: {columns}
- Rows: {rows}

Extract every value from the table: coefficients, standard errors, t-statistics, p-values,
significance stars, N (observations), R², and any other statistics.

Return a JSON object with this structure:
{{
    "table_number": "{table_number}",
    "values": {{
        "row_name": {{
            "column_name": "value_as_string (e.g., '0.234***', '(0.012)', '1,234')"
        }}
    }},
    "summary_stats": {{
        "column_name": {{
            "N": "value",
            "R_squared": "value"
        }}
    }},
    "notes": "any relevant notes about the table"
}}

Extract EXACTLY what appears in the paper. Preserve significance stars (*, **, ***).
Preserve parentheses around standard errors. Include ALL rows and columns."""


class VerifierAgent(BaseAgent):
    """Agent 3: Verifies replication results against original paper.

    This agent compares replicated tables and figures with the original
    paper and assigns grades based on how well the replication matches.

    Vision methods for figure comparison use raw API clients (OpenAI/Anthropic)
    since LangChain chat models don't support multimodal image inputs uniformly.
    """

    def __init__(self, config: Config, chat_model: Optional[BaseChatModel] = None):
        """Initialize the verifier agent.

        Args:
            config: Configuration object.
            chat_model: Optional LangChain chat model for DI/testing.
        """
        super().__init__(
            config=config,
            name="Verifier",
            role="replication verification specialist",
            goal="Compare replicated results with originals and assign grades",
            chat_model=chat_model,
        )
        self.tolerance = config.verification.numerical_tolerance
        self._vision_provider = config.langgraph.default_provider.lower()
        self._vision_client = None
        self._ground_truth_cache: dict[str, dict] = {}

    def _get_vision_client(self):
        """Get or create a raw API client for vision requests."""
        if self._vision_client is None:
            if self._vision_provider == "openai":
                from openai import OpenAI
                self._vision_client = OpenAI(api_key=self.config.openai_api_key)
            elif self._vision_provider == "anthropic":
                from anthropic import Anthropic
                self._vision_client = Anthropic(api_key=self.config.anthropic_api_key)
            else:
                raise ValueError(f"Vision not supported for provider: {self._vision_provider}")
        return self._vision_client

    def run(
        self,
        paper_path: str,
        replication_results: ReplicationResults,
    ) -> VerificationReport:
        """Verify replication results against original paper.

        Args:
            paper_path: Path to the original paper PDF.
            replication_results: Results from the replicator agent.

        Returns:
            VerificationReport with grades for each item.
        """
        logger.info(f"Verifying replication for: {replication_results.paper_id}")

        # Extract original paper content
        paper_text = extract_text_from_pdf(paper_path)
        paper_tables = extract_tables_from_pdf(paper_path)

        # Initialize verification results
        item_verifications = []
        grades = []

        # Verify each table
        for gen_table in replication_results.tables:
            verification = self._verify_table(
                gen_table, paper_text, paper_tables
            )
            item_verifications.append(verification)
            grades.append(verification.grade)

        # Verify each figure
        for gen_figure in replication_results.figures:
            verification = self._verify_figure(
                gen_figure, paper_text, paper_path
            )
            item_verifications.append(verification)
            grades.append(verification.grade)

        # Calculate overall grade
        overall_grade = self._calculate_overall_grade(grades)

        # Generate summary
        summary = self._generate_summary(item_verifications, overall_grade)

        report = VerificationReport(
            paper_id=replication_results.paper_id,
            overall_grade=overall_grade,
            item_verifications=item_verifications,
            summary=summary,
            methodology_notes=f"Numerical tolerance: {self.tolerance * 100}%",
        )

        logger.info(f"Verification complete. Overall grade: {overall_grade.value}")
        return report

    def _extract_table_pages(self, paper_text: str, table_number: str) -> str:
        """Find PDF pages containing a table and return surrounding page text.

        Uses the page markers (--- Page N ---) inserted by extract_text_from_pdf.
        """
        # Split text into pages using the markers
        pages = re.split(r"\n--- Page (\d+) ---\n", paper_text)
        # pages[0] is empty/preamble, then alternating: page_num, page_text, page_num, ...
        page_map: dict[int, str] = {}
        for i in range(1, len(pages) - 1, 2):
            try:
                page_num = int(pages[i])
                page_map[page_num] = pages[i + 1]
            except (ValueError, IndexError):
                continue

        # Find pages that mention this table
        table_pages: list[int] = []
        for page_num, page_text in page_map.items():
            if re.search(re.escape(table_number), page_text, re.IGNORECASE):
                table_pages.append(page_num)

        if not table_pages:
            # Fallback: try just the number part (e.g., "2.1" for "Table 2.1")
            num_part = table_number.replace("Table ", "").replace("Figure ", "")
            for page_num, page_text in page_map.items():
                if re.search(rf"(?:Table|table)\s*{re.escape(num_part)}", page_text):
                    table_pages.append(page_num)

        if not table_pages:
            return ""

        # Include surrounding pages for context
        all_pages: set[int] = set()
        for p in table_pages:
            all_pages.update(range(max(1, p - 1), p + 2))

        result_parts = []
        for p in sorted(all_pages):
            if p in page_map:
                result_parts.append(f"--- Page {p} ---\n{page_map[p]}")

        return "\n".join(result_parts)

    def _extract_original_values(
        self,
        paper_text: str,
        table_number: str,
        caption: str = "",
        columns: list[str] | None = None,
        rows: list[str] | None = None,
    ) -> dict | None:
        """Extract original table values from PDF using LLM.

        Returns extracted values dict or None on failure. Results are cached.
        """
        if table_number in self._ground_truth_cache:
            return self._ground_truth_cache[table_number]

        page_text = self._extract_table_pages(paper_text, table_number)
        if not page_text:
            logger.warning(f"Could not find pages for {table_number}")
            return None

        prompt = GROUND_TRUTH_EXTRACTION_PROMPT.format(
            table_number=table_number,
            caption=caption,
            page_text=page_text[:8000],
            columns=", ".join(columns) if columns else "Not specified",
            rows=", ".join(rows) if rows else "Not specified",
        )

        try:
            result = self.generate_json(
                prompt=prompt,
                system_prompt=VERIFIER_SYSTEM_PROMPT,
            )
            self._ground_truth_cache[table_number] = result
            logger.info(f"Extracted ground truth for {table_number}")
            return result
        except Exception as e:
            logger.warning(f"Ground truth extraction failed for {table_number}: {e}")
            return None

    def _verify_table(
        self,
        gen_table,
        paper_text: str,
        paper_tables: list[dict],
    ) -> ItemVerification:
        """Verify a replicated table against the original."""
        logger.info(f"Verifying {gen_table.table_number}")

        # Check if table execution was successful
        if not gen_table.execution_success:
            return ItemVerification(
                item_id=gen_table.table_number,
                item_type="table",
                grade=ReplicationGrade.F,
                comparison_notes=f"Replication failed: {gen_table.error_message}",
            )

        # Try LLM-based ground truth extraction first
        ground_truth = self._extract_original_values(
            paper_text, gen_table.table_number
        )

        if ground_truth and ground_truth.get("values"):
            original_content = json.dumps(ground_truth, indent=2)[:5000]
        else:
            # Fallback to text section extraction
            original_content = self._extract_table_section(
                paper_text, gen_table.table_number
            )[:5000]

        # Use LLM to compare
        prompt = VERIFICATION_PROMPT.format(
            original_content=original_content,
            replicated_content=json.dumps(gen_table.data, indent=2)[:5000],
            item_type="table",
            item_id=gen_table.table_number,
        )

        try:
            response = self.generate_json(
                prompt=prompt,
                system_prompt=VERIFIER_SYSTEM_PROMPT,
            )

            return ItemVerification(
                item_id=gen_table.table_number,
                item_type="table",
                grade=ReplicationGrade(response.get("grade", "F")),
                comparison_notes=response.get("comparison_notes", ""),
                numerical_differences=response.get("numerical_differences"),
                key_findings_match=response.get("key_findings_match"),
            )

        except Exception as e:
            logger.error(f"Verification failed for {gen_table.table_number}: {e}")
            return ItemVerification(
                item_id=gen_table.table_number,
                item_type="table",
                grade=ReplicationGrade.F,
                comparison_notes=f"Verification error: {e}",
            )

    def _verify_figure(
        self,
        gen_figure,
        paper_text: str,
        paper_path: str,
    ) -> ItemVerification:
        """Verify a replicated figure against the original."""
        logger.info(f"Verifying {gen_figure.figure_number}")

        # Check if figure was generated successfully
        if not gen_figure.execution_success:
            return ItemVerification(
                item_id=gen_figure.figure_number,
                item_type="figure",
                grade=ReplicationGrade.F,
                comparison_notes=f"Replication failed: {gen_figure.error_message}",
            )

        # Check if figure file exists
        fig_path = Path(gen_figure.file_path)
        if not fig_path.exists():
            return ItemVerification(
                item_id=gen_figure.figure_number,
                item_type="figure",
                grade=ReplicationGrade.F,
                comparison_notes="Figure file not found",
            )

        # Extract figure caption and description from paper
        figure_section = self._extract_figure_section(
            paper_text, gen_figure.figure_number
        )

        # Try vision-based comparison if configured, fall back to text-based
        use_vision = self.config.verification.use_vision_model
        if use_vision:
            try:
                return self._verify_figure_with_vision(
                    gen_figure, figure_section, paper_path, fig_path
                )
            except Exception as e:
                logger.warning(
                    f"Vision comparison failed for {gen_figure.figure_number}, "
                    f"falling back to text-based: {e}"
                )

        return self._verify_figure_text_only(gen_figure, figure_section)

    def _verify_figure_with_vision(
        self,
        gen_figure,
        figure_section: str,
        paper_path: str,
        fig_path: Path,
    ) -> ItemVerification:
        """Verify a figure using vision model API for visual comparison.

        Sends the replicated figure image to the LLM vision API along with
        the original paper's figure description for comparison.
        """
        # Read replicated figure as base64
        with open(fig_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("utf-8")

        # Determine media type from extension
        suffix = fig_path.suffix.lower()
        media_types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
        media_type = media_types.get(suffix, "image/png")

        prompt_text = f"""Compare the replicated figure shown in the image with the original figure description from the paper.

## Original Figure Description from Paper:
{figure_section}

## Figure Being Compared:
{gen_figure.figure_number}

Examine the replicated figure image and compare it against the description of the original.
Focus on:
- Whether the plot type matches (scatter, bar, line, etc.)
- Whether axes and labels are consistent
- Whether the overall patterns and trends match the description
- Whether the data range and scale appear reasonable

Provide verification in JSON format:
{{
    "item_id": "{gen_figure.figure_number}",
    "item_type": "figure",
    "grade": "A/B/C/D/F",
    "comparison_notes": "Detailed visual comparison assessment",
    "key_findings_match": true/false
}}"""

        if self._vision_provider == "openai":
            response = self._vision_openai(prompt_text, img_data, media_type)
        elif self._vision_provider == "anthropic":
            response = self._vision_anthropic(prompt_text, img_data, media_type)
        else:
            raise ValueError(f"Vision not supported for provider: {self._vision_provider}")

        return ItemVerification(
            item_id=gen_figure.figure_number,
            item_type="figure",
            grade=ReplicationGrade(response.get("grade", "F")),
            comparison_notes=response.get("comparison_notes", ""),
            key_findings_match=response.get("key_findings_match"),
        )

    def _vision_openai(self, prompt: str, img_b64: str, media_type: str) -> dict:
        """Send a vision request via the OpenAI API."""
        client = self._get_vision_client()

        messages = [
            {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt + "\n\nRespond with valid JSON only."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{img_b64}",
                        },
                    },
                ],
            },
        ]

        # Newer models (gpt-5.x) require max_completion_tokens; older ones use max_tokens
        model = self.config.langgraph.default_model
        token_kwarg = (
            {"max_completion_tokens": self.config.langgraph.max_tokens}
            if model.startswith("gpt-5") or model.startswith("o")
            else {"max_tokens": self.config.langgraph.max_tokens}
        )
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=self.config.langgraph.temperature,
            **token_kwarg,
        )

        text = response.choices[0].message.content
        if text.strip().startswith("{"):
            return json.loads(text)
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            return json.loads(json_match.group())
        raise ValueError("No JSON found in vision response")

    def _vision_anthropic(self, prompt: str, img_b64: str, media_type: str) -> dict:
        """Send a vision request via the Anthropic API."""
        client = self._get_vision_client()

        response = client.messages.create(
            model=self.config.langgraph.default_model,
            max_tokens=self.config.langgraph.max_tokens,
            system=VERIFIER_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": img_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt + "\n\nRespond with valid JSON only.",
                        },
                    ],
                }
            ],
        )

        text = response.content[0].text
        if text.strip().startswith("{"):
            return json.loads(text)
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            return json.loads(json_match.group())
        raise ValueError("No JSON found in vision response")

    def _verify_figure_text_only(
        self, gen_figure, figure_section: str
    ) -> ItemVerification:
        """Verify a figure using text-based comparison only (fallback)."""
        prompt = f"""Compare the replicated figure with the original description.

## Original Figure Description:
{figure_section}

## Replicated Figure:
{gen_figure.figure_number} (saved to {gen_figure.file_path})

Note: Visual comparison not available. Base your assessment on whether the
replicated figure appears to follow the same specifications as described.

Provide verification in JSON format:
{{
    "item_id": "{gen_figure.figure_number}",
    "item_type": "figure",
    "grade": "A/B/C/D/F",
    "comparison_notes": "Assessment notes",
    "key_findings_match": true/false
}}"""

        try:
            response = self.generate_json(
                prompt=prompt,
                system_prompt=VERIFIER_SYSTEM_PROMPT,
            )

            return ItemVerification(
                item_id=gen_figure.figure_number,
                item_type="figure",
                grade=ReplicationGrade(response.get("grade", "F")),
                comparison_notes=response.get("comparison_notes", ""),
                key_findings_match=response.get("key_findings_match"),
            )

        except Exception as e:
            logger.error(f"Verification failed for {gen_figure.figure_number}: {e}")
            return ItemVerification(
                item_id=gen_figure.figure_number,
                item_type="figure",
                grade=ReplicationGrade.F,
                comparison_notes=f"Verification error: {e}",
            )

    def _extract_table_section(self, paper_text: str, table_number: str) -> str:
        """Extract the section of paper text related to a specific table."""
        # Find table mention and surrounding context
        pattern = rf"({re.escape(table_number)}[^\n]*(?:\n(?![A-Z])[^\n]*)*)"
        matches = re.findall(pattern, paper_text, re.IGNORECASE)

        if matches:
            return "\n\n".join(matches[:3])  # Return first 3 matches

        # Fallback: search for table number anywhere
        lines = paper_text.split("\n")
        relevant_lines = []
        for i, line in enumerate(lines):
            if table_number.lower() in line.lower():
                # Get surrounding context
                start = max(0, i - 5)
                end = min(len(lines), i + 20)
                relevant_lines.extend(lines[start:end])

        return "\n".join(relevant_lines) if relevant_lines else "Table not found in paper"

    def _extract_figure_section(self, paper_text: str, figure_number: str) -> str:
        """Extract the section of paper text related to a specific figure."""
        pattern = rf"({re.escape(figure_number)}[^\n]*(?:\n(?![A-Z])[^\n]*)*)"
        matches = re.findall(pattern, paper_text, re.IGNORECASE)

        if matches:
            return "\n\n".join(matches[:3])

        return "Figure description not found in paper"

    def _calculate_overall_grade(self, grades: list[ReplicationGrade]) -> ReplicationGrade:
        """Calculate overall grade from individual grades."""
        if not grades:
            return ReplicationGrade.F

        # Convert to numeric for averaging
        grade_values = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}
        numeric_grades = [grade_values[g.value] for g in grades]

        avg = sum(numeric_grades) / len(numeric_grades)

        if avg >= 3.5:
            return ReplicationGrade.A
        elif avg >= 2.5:
            return ReplicationGrade.B
        elif avg >= 1.5:
            return ReplicationGrade.C
        elif avg >= 0.5:
            return ReplicationGrade.D
        else:
            return ReplicationGrade.F

    def _generate_summary(
        self,
        verifications: list[ItemVerification],
        overall_grade: ReplicationGrade,
    ) -> str:
        """Generate a summary of the verification results."""
        grade_counts = {}
        for v in verifications:
            grade = v.grade.value
            grade_counts[grade] = grade_counts.get(grade, 0) + 1

        summary_parts = [
            f"Overall replication grade: {overall_grade.value}",
            f"Total items verified: {len(verifications)}",
            "Grade distribution:",
        ]

        for grade in ["A", "B", "C", "D", "F"]:
            if grade in grade_counts:
                summary_parts.append(f"  - Grade {grade}: {grade_counts[grade]} items")

        # Add notes about specific issues
        issues = [v for v in verifications if v.grade.value in ["D", "F"]]
        if issues:
            summary_parts.append("\nItems with significant issues:")
            for v in issues:
                summary_parts.append(f"  - {v.item_id}: {v.comparison_notes[:100]}...")

        return "\n".join(summary_parts)
