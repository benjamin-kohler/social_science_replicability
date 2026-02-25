"""Unified judge for grading replication quality.

Uses plain OpenAI/Anthropic SDK calls — no LangChain, no LangGraph.
Each item (table or figure) is graded in a single LLM call that produces
both a verification grade and (if non-A) a discrepancy explanation.
"""

import base64
import json
import re
from pathlib import Path
from typing import Any, Optional

from ..models.schemas import (
    DiscrepancyAnalysis,
    ExplanationReport,
    ItemVerification,
    PaperSummary,
    ReplicationGrade,
    ReplicationResults,
    VerificationReport,
)
from ..utils.logging_utils import get_logger
from ..utils.pdf_parser import extract_text_from_pdf

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """You are an expert judge evaluating research replication quality.

You compare a replicated result against the original paper, grade the replication,
and explain any discrepancies — all in a single assessment.

Grading Scale:
- A: Fully replicated. Results match within numerical precision (< 1% difference).
- B: Same direction of effects with small discrepancies (1-5% difference).
- C: Same direction of effects with large discrepancies (5-20% difference).
- D: Results differ meaningfully — different significance, direction, or magnitude.
- F: Not comparable — missing output, incompatible format, or unable to verify.

Context:
- The replication was performed WITHOUT access to the original paper's results or any
  replication code. The replicator received only a methodological summary (extracted from
  the paper) and the dataset, and had to reconstruct all analyses from that description alone.
- Some discrepancies may therefore stem from ambiguity in the methodology description rather
  than errors by the replicator. When attributing fault, distinguish between:
  (a) clear replicator errors (e.g., wrong dataset, coding bugs, flipped signs despite clear
      definitions), and
  (b) underspecified methodology (e.g., control variable lists referenced but not provided,
      sign conventions not stated, sample selection criteria ambiguous).

Important:
- Focus on SUBSTANCE, not formatting or presentation.
- Compare coefficients, standard errors, significance levels, and key statistics.
- For figures, compare patterns and trends, not exact visual appearance.
- Note any differences in sample size or methodology that could explain discrepancies.
- When comparing with the replication package code, identify whether discrepancies
  come from the replicator's approach vs. ambiguity in the methodology description."""

TABLE_JUDGE_PROMPT = """Judge the replication of {item_id}.

## Original Paper (relevant pages):
{paper_pages}

## Expected Table Structure (from methodology summary):
{template}

## Replicated Output (CSV data):
{replicated_data}

## Replication Code Used:
{replication_code}

## Original Replication Package Code:
{package_code}

Compare the replicated output against the original paper. Then:
1. Assign a grade (A/B/C/D/F) based on how well the results match.
2. If the grade is NOT A, explain the discrepancy and attribute fault.

Respond with ONLY this JSON (no other text):
{{
    "grade": "A/B/C/D/F",
    "comparison_notes": "Detailed comparison of the results",
    "numerical_differences": {{
        "max_difference_percent": 0.0,
        "key_differences": ["list of specific differences"]
    }},
    "key_findings_match": true,
    "discrepancy": {{
        "description": "What differs (empty string if grade A)",
        "likely_causes": ["ordered list of possible causes"],
        "is_identifiable": true,
        "fault_attribution": "replicator/original_paper/unclear/data_limitation",
        "confidence": "high/medium/low",
        "supporting_evidence": "Evidence from code comparison or paper"
    }}
}}"""

FIGURE_JUDGE_PROMPT = """Judge the replication of {item_id}.

## Original Paper (relevant pages):
{paper_pages}

## Expected Figure Specification:
- Caption: {caption}
- Plot type: {plot_type}
- X-axis: {x_axis}
- Y-axis: {y_axis}
- Grouping: {grouping}
- Subplots: {subplots}

## Replication Code Used:
{replication_code}

## Original Replication Package Code:
{package_code}

{vision_note}

Compare the replicated figure against the original paper description. Then:
1. Assign a grade (A/B/C/D/F) based on how well the results match.
2. If the grade is NOT A, explain the discrepancy and attribute fault.

Respond with ONLY this JSON (no other text):
{{
    "grade": "A/B/C/D/F",
    "comparison_notes": "Detailed comparison assessment",
    "key_findings_match": true,
    "discrepancy": {{
        "description": "What differs (empty string if grade A)",
        "likely_causes": ["ordered list of possible causes"],
        "is_identifiable": true,
        "fault_attribution": "replicator/original_paper/unclear/data_limitation",
        "confidence": "high/medium/low",
        "supporting_evidence": "Evidence from code comparison or paper"
    }}
}}"""


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------


class Judge:
    """Grades replication outputs against the original paper.

    Uses plain OpenAI or Anthropic SDK — one LLM call per item.
    Tracks token usage across all LLM calls.
    """

    # Reasoning models don't support temperature
    _REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5-mini", "gpt-5-nano")

    def __init__(self, provider: str, model: str, api_key: str):
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key
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
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            }
            if not self._is_reasoning:
                kwargs["temperature"] = 0.0
            resp = self.client.chat.completions.create(**kwargs)
            self._record_usage_openai(resp)
            return resp.choices[0].message.content
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
                "messages": [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{image_b64}",
                                },
                            },
                        ],
                    },
                ],
            }
            if not self._is_reasoning:
                kwargs["temperature"] = 0.0
            resp = self.client.chat.completions.create(**kwargs)
            self._record_usage_openai(resp)
            return resp.choices[0].message.content
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

    # -- Token usage tracking ------------------------------------------------

    def _record_usage_openai(self, resp) -> None:
        """Record token usage from an OpenAI response."""
        u = getattr(resp, "usage", None)
        if u:
            self._usage.append({
                "prompt_tokens": u.prompt_tokens,
                "completion_tokens": u.completion_tokens,
                "total_tokens": u.total_tokens,
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
        replication_package_path: str | None = None,
    ) -> tuple[VerificationReport, ExplanationReport | None]:
        """Judge all replicated items.

        Returns:
            (VerificationReport, ExplanationReport or None if all grades are A).
        """
        logger.info(f"Judging replication for: {replication_results.paper_id}")

        paper_text = extract_text_from_pdf(paper_path)
        package = self._load_replication_package(replication_package_path)

        # Build lookups
        table_specs = {t.table_number: t for t in paper_summary.tables}
        figure_specs = {f.figure_number: f for f in paper_summary.figures}

        item_verifications: list[ItemVerification] = []
        discrepancy_analyses: list[DiscrepancyAnalysis] = []
        grades: list[ReplicationGrade] = []

        # Judge tables
        for gen_table in replication_results.tables:
            spec = table_specs.get(gen_table.table_number)
            repl_code = self._find_code(replication_results, gen_table.table_number)
            pkg_code = self._find_package_code(package, gen_table.table_number)

            verification, analysis = self._judge_table(
                gen_table, spec, paper_text, repl_code, pkg_code,
            )
            item_verifications.append(verification)
            grades.append(verification.grade)
            if analysis:
                discrepancy_analyses.append(analysis)

        # Judge figures
        for gen_figure in replication_results.figures:
            spec = figure_specs.get(gen_figure.figure_number)
            repl_code = self._find_code(replication_results, gen_figure.figure_number)
            pkg_code = self._find_package_code(package, gen_figure.figure_number)

            verification, analysis = self._judge_figure(
                gen_figure, spec, paper_text, repl_code, pkg_code,
            )
            item_verifications.append(verification)
            grades.append(verification.grade)
            if analysis:
                discrepancy_analyses.append(analysis)

        # Build reports
        overall_grade = self._calculate_overall_grade(grades)
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
                replication_package_comparison=(
                    self._compare_with_package(replication_results, package)
                    if package else None
                ),
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
        repl_code: str,
        pkg_code: str | None,
    ) -> tuple[ItemVerification, DiscrepancyAnalysis | None]:
        """Judge a single table."""
        item_id = gen_table.table_number
        logger.info(f"Judging {item_id}")

        # Failed execution → automatic F
        if not gen_table.execution_success:
            return (
                ItemVerification(
                    item_id=item_id, item_type="table",
                    grade=ReplicationGrade.F,
                    comparison_notes=f"Replication failed: {gen_table.error_message}",
                ),
                None,
            )

        paper_pages = self._extract_table_pages(paper_text, item_id)
        template = spec.template_markdown if spec and spec.template_markdown else "Not available"
        replicated_data = json.dumps(gen_table.data, indent=2)[:5000]

        prompt = TABLE_JUDGE_PROMPT.format(
            item_id=item_id,
            paper_pages=paper_pages[:8000],
            template=template[:3000],
            replicated_data=replicated_data,
            replication_code=repl_code[:3000],
            package_code=pkg_code[:3000] if pkg_code else "Not available",
        )

        try:
            resp = self._parse_json(self._call_llm(JUDGE_SYSTEM_PROMPT, prompt))
            return self._parse_judge_response(resp, item_id, "table")
        except Exception as e:
            logger.error(f"Judge call failed for {item_id}: {e}")
            return (
                ItemVerification(
                    item_id=item_id, item_type="table",
                    grade=ReplicationGrade.F,
                    comparison_notes=f"Judge error: {e}",
                ),
                None,
            )

    def _judge_figure(
        self,
        gen_figure,
        spec,
        paper_text: str,
        repl_code: str,
        pkg_code: str | None,
    ) -> tuple[ItemVerification, DiscrepancyAnalysis | None]:
        """Judge a single figure."""
        item_id = gen_figure.figure_number
        logger.info(f"Judging {item_id}")

        if not gen_figure.execution_success:
            return (
                ItemVerification(
                    item_id=item_id, item_type="figure",
                    grade=ReplicationGrade.F,
                    comparison_notes=f"Replication failed: {gen_figure.error_message}",
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
                ),
                None,
            )

        paper_pages = self._extract_table_pages(paper_text, item_id)

        prompt = FIGURE_JUDGE_PROMPT.format(
            item_id=item_id,
            paper_pages=paper_pages[:8000],
            caption=spec.caption if spec else "Not available",
            plot_type=spec.plot_type if spec else "Unknown",
            x_axis=spec.x_axis if spec else "Unknown",
            y_axis=spec.y_axis if spec else "Unknown",
            grouping=", ".join(spec.grouping_vars) if spec and spec.grouping_vars else "None",
            subplots=spec.subplot_structure if spec and spec.subplot_structure else "None",
            replication_code=repl_code[:3000],
            package_code=pkg_code[:3000] if pkg_code else "Not available",
            vision_note="The replicated figure image is attached for visual comparison.",
        )

        try:
            # Try vision comparison
            with open(fig_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
            suffix = fig_path.suffix.lower()
            media_type = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(
                suffix, "image/png"
            )
            raw = self._call_llm_vision(JUDGE_SYSTEM_PROMPT, prompt, img_b64, media_type)
            resp = self._parse_json(raw)
        except Exception as e:
            logger.warning(f"Vision comparison failed for {item_id}, falling back to text: {e}")
            # Text-only fallback
            prompt = prompt.replace(
                "The replicated figure image is attached for visual comparison.",
                "Note: Visual comparison not available. Assess based on code and description.",
            )
            try:
                resp = self._parse_json(self._call_llm(JUDGE_SYSTEM_PROMPT, prompt))
            except Exception as e2:
                logger.error(f"Judge call failed for {item_id}: {e2}")
                return (
                    ItemVerification(
                        item_id=item_id, item_type="figure",
                        grade=ReplicationGrade.F,
                        comparison_notes=f"Judge error: {e2}",
                    ),
                    None,
                )

        return self._parse_judge_response(resp, item_id, "figure")

    # -- Response parsing ---------------------------------------------------

    @staticmethod
    def _parse_judge_response(
        resp: dict, item_id: str, item_type: str,
    ) -> tuple[ItemVerification, DiscrepancyAnalysis | None]:
        """Parse the combined judge JSON into verification + optional analysis."""
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
            disc = resp.get("discrepancy", {})
            analysis = DiscrepancyAnalysis(
                item_id=item_id,
                grade=grade,
                description_of_discrepancy=disc.get("description", ""),
                likely_causes=disc.get("likely_causes", []),
                is_identifiable=disc.get("is_identifiable", False),
                fault_attribution=disc.get("fault_attribution", "unclear"),
                confidence=disc.get("confidence", "low"),
                supporting_evidence=disc.get("supporting_evidence"),
            )

        return verification, analysis

    # -- Helpers (migrated from verifier.py / explainer.py) -----------------

    @staticmethod
    def _extract_table_pages(paper_text: str, item_id: str) -> str:
        """Find PDF pages mentioning this item and return surrounding context."""
        pages = re.split(r"\n--- Page (\d+) ---\n", paper_text)
        page_map: dict[int, str] = {}
        for i in range(1, len(pages) - 1, 2):
            try:
                page_map[int(pages[i])] = pages[i + 1]
            except (ValueError, IndexError):
                continue

        # Find pages mentioning the item
        matching: list[int] = []
        for pnum, ptext in page_map.items():
            if re.search(re.escape(item_id), ptext, re.IGNORECASE):
                matching.append(pnum)

        if not matching:
            num_part = item_id.replace("Table ", "").replace("Figure ", "")
            for pnum, ptext in page_map.items():
                if re.search(rf"(?:Table|Figure|table|figure)\s*{re.escape(num_part)}", ptext):
                    matching.append(pnum)

        if not matching:
            return ""

        # Include surrounding pages for context
        all_pages: set[int] = set()
        for p in matching:
            all_pages.update(range(max(1, p - 1), p + 2))

        parts = []
        for p in sorted(all_pages):
            if p in page_map:
                parts.append(f"--- Page {p} ---\n{page_map[p]}")
        return "\n".join(parts)

    @staticmethod
    def _calculate_overall_grade(grades: list[ReplicationGrade]) -> ReplicationGrade:
        """Average item grades to an overall grade."""
        if not grades:
            return ReplicationGrade.F
        values = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}
        avg = sum(values[g.value] for g in grades) / len(grades)
        if avg >= 3.5:
            return ReplicationGrade.A
        if avg >= 2.5:
            return ReplicationGrade.B
        if avg >= 1.5:
            return ReplicationGrade.C
        if avg >= 0.5:
            return ReplicationGrade.D
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
        for g in ["A", "B", "C", "D", "F"]:
            if g in counts:
                parts.append(f"  - Grade {g}: {counts[g]} items")

        issues = [v for v in verifications if v.grade.value in ("D", "F")]
        if issues:
            parts.append("\nItems with significant issues:")
            for v in issues:
                parts.append(f"  - {v.item_id}: {v.comparison_notes[:100]}...")

        return "\n".join(parts)

    @staticmethod
    def _load_replication_package(path: str | None) -> dict | None:
        """Load and index the original replication package code files."""
        if not path:
            return None
        pkg_dir = Path(path)
        if not pkg_dir.exists():
            logger.warning(f"Replication package not found: {path}")
            return None

        files: dict[str, str] = {}
        for ext in (".py", ".r", ".R", ".do", ".sas", ".m"):
            for f in pkg_dir.rglob(f"*{ext}"):
                try:
                    files[str(f.relative_to(pkg_dir))] = f.read_text()
                except Exception as e:
                    logger.warning(f"Could not read {f}: {e}")

        logger.info(f"Loaded {len(files)} files from replication package")
        return {"path": path, "files": files}

    @staticmethod
    def _find_code(results: ReplicationResults, item_id: str) -> str:
        """Find the replication code for a specific item."""
        item_lower = item_id.lower()
        for code in results.code_files:
            if code.description and item_lower in code.description.lower():
                return code.code
        return "Code not found"

    @staticmethod
    def _find_package_code(package: dict | None, item_id: str) -> str | None:
        """Find code from the replication package relevant to an item."""
        if not package:
            return None
        item_lower = item_id.lower()
        relevant = []
        for filename, content in package.get("files", {}).items():
            if item_lower in content.lower():
                relevant.append(f"--- {filename} ---\n{content}")
        return "\n\n".join(relevant) if relevant else None

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

    @staticmethod
    def _compare_with_package(results: ReplicationResults, package: dict) -> str:
        """Compare replication code with original package code."""
        parts = []
        for code in results.code_files:
            if not code.description:
                continue
            for filename in package.get("files", {}):
                if any(kw in filename.lower() for kw in ("table", "figure", "regression")):
                    parts.append(f"Potential match: {filename} may correspond to {code.description}")
                    break
        return "\n".join(parts) if parts else "No direct code matches found"
