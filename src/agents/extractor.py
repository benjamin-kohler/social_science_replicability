"""Agent 1: Extractor - Extracts methodology from papers without revealing results.

Uses direct OpenAI API with structured outputs (no LangChain).
"""

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from openai import OpenAI
from pydantic import BaseModel, Field

from ..models.schemas import (
    PaperSummary,
    DataProcessingStep,
    RegressionSpec,
    TableSpec,
    PlotSpec,
)
from ..models.config import Config
from ..utils.pdf_parser import (
    extract_text_from_pdf,
    extract_figure_captions,
    extract_table_captions,
    pdf_to_base64_images,
)
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Usage tracking
# ---------------------------------------------------------------------------

@dataclass
class APICallRecord:
    """Record of a single API call."""
    step: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    duration_seconds: float = 0.0


@dataclass
class ExtractionUsage:
    """Accumulated usage across all API calls in an extraction run."""
    calls: list[APICallRecord] = field(default_factory=list)

    @property
    def total_input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.calls)

    @property
    def total_tokens(self) -> int:
        return sum(c.total_tokens for c in self.calls)

    @property
    def total_duration(self) -> float:
        return sum(c.duration_seconds for c in self.calls)

    def summary_dict(self) -> dict:
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_duration_seconds": round(self.total_duration, 2),
            "num_api_calls": len(self.calls),
            "calls": [
                {
                    "step": c.step,
                    "model": c.model,
                    "input_tokens": c.input_tokens,
                    "output_tokens": c.output_tokens,
                    "total_tokens": c.total_tokens,
                    "cached_tokens": c.cached_tokens,
                    "reasoning_tokens": c.reasoning_tokens,
                    "duration_seconds": round(c.duration_seconds, 2),
                }
                for c in self.calls
            ],
        }


# ---------------------------------------------------------------------------
# Structured‑output response models (OpenAI‑compatible Pydantic schemas)
# ---------------------------------------------------------------------------

class RegressionSpecResponse(BaseModel):
    model_type: str
    dependent_var: str
    independent_vars: list[str] = []
    controls: list[str] = []
    fixed_effects: list[str] = []
    clustering: Optional[str] = None
    sample_restrictions: Optional[str] = None
    equation_latex: Optional[str] = None
    variable_definitions: Optional[str] = None
    omitted_categories: Optional[dict[str, str]] = None
    additional_notes: Optional[str] = None


class DataProcessingStepResponse(BaseModel):
    step_number: int
    description: str
    variables_involved: list[str] = []


class TableSpecResponse(BaseModel):
    table_number: str
    caption: str
    column_names: list[str] = []
    row_names: list[str] = []
    regression_specs: list[RegressionSpecResponse] = []
    data_processing_steps: list[DataProcessingStepResponse] = []
    notes: Optional[str] = None
    data_source: Optional[str] = None
    panel_structure: Optional[str] = None


class PlotSpecResponse(BaseModel):
    figure_number: str
    caption: str
    plot_type: str
    x_axis: Optional[str] = None
    y_axis: Optional[str] = None
    grouping_vars: list[str] = []
    regression_specs: list[RegressionSpecResponse] = []
    data_processing_steps: list[DataProcessingStepResponse] = []
    notes: Optional[str] = None
    data_source: Optional[str] = None
    subplot_structure: Optional[str] = None


class ExtractionResponse(BaseModel):
    """Schema for the first API call: methodology extraction."""
    paper_id: str
    title: Optional[str] = None
    research_questions: list[str] = []
    data_description: str
    data_context: str
    data_source: Optional[str] = None
    sample_size: Optional[str] = None
    time_period: Optional[str] = None
    data_processing_steps: list[DataProcessingStepResponse] = []
    tables: list[TableSpecResponse] = []
    figures: list[PlotSpecResponse] = []


class TableTemplate(BaseModel):
    """A single table template."""
    table_number: str
    template_markdown: str


class FigureTemplate(BaseModel):
    """A single figure template."""
    figure_number: str
    template_code: str


class TemplateResponse(BaseModel):
    """Schema for the second API call: template generation."""
    table_templates: list[TableTemplate] = []
    figure_templates: list[FigureTemplate] = []


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """You are a methodological extraction specialist for social science research papers.

Your task: extract the methodology, structure, and specifications from academic papers so that
a replicator who cannot see the paper can reproduce every table and figure. Extract NO results.

## 1. What to extract

- Research questions
- Data description, source, sample size, time period
- Data processing steps: BOTH general steps (applied to all analyses) AND per-table/per-figure
  steps (sample restrictions, variable construction, or filtering specific to one table or figure)
- Per-table: exact column headers, exact row labels, panel structure, caption, notes,
  regression specifications, data source (if different tables use different datasets)
- Per-figure: full caption, plot type, axis labels, legend/series entries, subplot structure,
  visual details (approximate axis ranges, reference lines, line styles, color conventions)
- Per-regression-spec: model type, dependent variable, independent variables, controls,
  fixed effects, clustering, sample restrictions, equation, variable definitions

Extract ALL tables and ALL data-based figures. Skip only purely conceptual visualizations
(flow diagrams, frameworks, screenshots, photos, maps). Do not skip tables because they seem
simple — summary statistics, balance tables, and cross-tabulations must all be extracted.

## 2. What NOT to extract (results)

Never include: regression coefficients, standard errors, t-statistics, p-values, significance
stars, point estimates, confidence intervals, effect sizes, or descriptions of direction/magnitude.

DO include: table/figure structure (headers, labels, panels), design parameters (sample sizes,
thresholds, variable names), and descriptive labels ("N", "R-squared", "Controls", "Yes"/"No").

## 3. Data processing and cleaning steps

This is critical for replication. Extract every step needed to go from the raw data files to
each analysis-ready sample. Separate these into:

**General steps** (applied before any specific analysis):
- File loading, merging, and reshaping (which files, which keys)
- Variable construction (new columns derived from raw data)
- Data cleaning: missing value treatment, outlier removal, winsorization
- Sample filtering: time period restrictions, geographic restrictions, demographic restrictions
- Transformations: log transforms, standardization, encoding

**Per-table / per-figure steps** (use `sample_restrictions` on each table/figure spec):
- Additional filtering specific to one table (e.g., "Table 3 restricts to males aged 25-64")
- Subsample definitions for different columns (e.g., "Columns 1-3 use the full sample,
  columns 4-6 restrict to the treatment group")
- Variable construction specific to one analysis (e.g., "For Figure 2, compute rolling 30-day
  averages of daily returns")

Use the actual variable names from the dataset where possible.

## 4. Regression specifications

Attach each regression spec to the table or figure that displays its results. Include one spec
per distinct model variant (e.g., different controls or subsamples across columns).

For each spec:
- Use actual variable names from the paper, not generic descriptions.
  Write `dependent_var: "log(wage)"`, not `dependent_var: "outcome variable"`.
- `controls`: enumerate every control variable individually. Never write aggregate descriptions
  like "socio-demographic controls" — list each variable: ["age", "female", "education_level", ...].
- `equation_latex`: copy the exact equation from the paper in LaTeX notation, preserving all
  subscripts, Greek letters, and notation. If no explicit equation is given, write one from the
  prose description. Example:
  `Y_i = \\beta_0 + \\beta_1 X_i + \\gamma Z_i + \\delta_j + \\epsilon_i`
- `variable_definitions`: define every symbol verbally, separated by semicolons. Include all
  information the paper provides: definition, construction, coding, unit, sign convention,
  omitted categories, data source. Only include what is explicitly stated — do not infer.
  Example: `A_i^T: acceptance of targeted tax (1 if not "No", as defined in Section 2.3);
  c_i: income-threshold fixed effects (4 levels: bottom 20/30/40/50 percentile)`
- `sample_restrictions`: extract the exact condition including any mathematical formulation
  and expected sample size.
  BAD: "Among invalidated respondents"
  GOOD: "Among invalidated respondents, defined as sgn(g_i) != sgn(gamma_hat_i) (Section 4.1). N = 1,365."

## 5. Cross-reference resolution

When anything references content elsewhere ("see Appendix H", "controls listed in Table A3",
"as defined in Section 2"), go to that location and extract the actual content. Never leave
unresolved references — the replicator cannot see the paper. This is especially important for
control variable lists: find the appendix and enumerate every variable individually.

## 6. Variable completeness

Extract ALL variables: dependent, independent, controls, instruments, weights, and constructed
variables. For each, combine information from all locations in the paper (methodology section,
table notes, appendices). The goal: a replicator should be able to construct every variable
exactly as the authors did based solely on your extraction.

## 7. Table and figure precision

- Copy EXACT column headers and row labels as printed. Count columns and rows precisely.
- Include panel structure (Panel A / Panel B) when present.
- Copy full captions and table notes (excluding notes about specific coefficient values).
- For figures: note approximate axis ranges, reference lines, and line style conventions.
- If different tables use different datasets, specify the data source per table."""


EXTRACTION_USER_PROMPT = """Extract the methodology from this paper. Follow the system instructions.

## Detected Table Captions:
{table_captions}

## Detected Figure Captions:
{figure_captions}

## Paper Text:
{paper_text}"""


TEMPLATE_GENERATION_SYSTEM_PROMPT = """You are a structural template specialist for academic paper tables and figures.

Generate templates that faithfully reproduce the exact layout of tables and figures from a paper.

## Table templates

- EXACTLY the same number of columns and rows as the original table.
- Use the EXACT column headers and row labels from the paper.
- Cell content rules (check the ORIGINAL TABLE in the paper for each cell):
  - **XXX**: computed result (coefficient, statistic, count, mean, etc.)
  - **(XXX)**: standard error in parentheses
  - **Empty**: leave blank if blank in the original (do NOT use --- or placeholders)
  - **Literal text**: copy "Yes", "No", checkmarks, or labels as-is
- Include panel headers (Panel A, Panel B) as spanning rows when present.
- Include the full caption above the table.

## Figure templates

- Matplotlib code skeleton that makes the intended visual style unambiguous.
- Use the correct plot function: `ax.plot()`, `ax.bar()`, `ax.hist()`, `ax.scatter()`, etc.
- Set line styles, markers, colors for each series (use `tab:blue`, `tab:orange`, etc.).
- Set approximate axis ranges from the paper (`ax.set_xlim()`, `ax.set_ylim()`).
- Add reference lines (axvline, axhline) if the paper has them.
- Use actual series names / legend labels from the paper.
- Include the full caption as the title.
- Set up subplots correctly if the figure has panels.
- NO actual data arrays — use comments like `# TODO: fill with data`.
- The skeleton should be runnable (producing an empty styled plot) even without data."""


TEMPLATE_GENERATION_USER_PROMPT = """Generate structural templates for each table and figure.
Respond with valid JSON.

Go back to the ORIGINAL TABLE in the paper text and reproduce its structure EXACTLY.

Example regression table template (note how Variable C only appears in columns (3)-(4)):

| | (1) | (2) | (3) | (4) |
|---|---|---|---|---|
| Variable A | XXX | XXX | XXX | XXX |
| | (XXX) | (XXX) | (XXX) | (XXX) |
| Variable B | XXX | XXX | XXX | XXX |
| | (XXX) | (XXX) | (XXX) | (XXX) |
| Variable C | | | XXX | XXX |
| | | | (XXX) | (XXX) |
| Controls | No | Yes | No | Yes |
| Observations | XXX | XXX | XXX | XXX |

Example figure skeleton:

```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
# x_obj, density_obj = ... # TODO: compute KDE from data
# ax.plot(x_obj, density_obj, color="tab:blue", linestyle="-", linewidth=2, label="Objective")
ax.set_xlabel("Net gain (EUR/year)", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_xlim(-1500, 1500)
ax.axvline(x=0, color="black", linestyle=":", linewidth=0.8)
ax.legend()
plt.tight_layout()
plt.savefig("figure_X.png", dpi=150)
```

## Paper Text (excerpt):
{paper_text}

## Extracted Tables:
{table_specs_json}

## Extracted Figures:
{figure_specs_json}"""


# ---------------------------------------------------------------------------
# Non-replicable figure types (code-level safety net)
# ---------------------------------------------------------------------------

NON_REPLICABLE_KEYWORDS = [
    "flow diagram", "flow chart", "flowchart", "conceptual framework",
    "conceptual diagram", "diagram", "schematic", "screenshot", "photo",
    "photograph", "timeline",
]


# ---------------------------------------------------------------------------
# ExtractorAgent
# ---------------------------------------------------------------------------

class ExtractorAgent:
    """Extracts methodology from papers using direct OpenAI API with structured outputs.

    Two-step process:
    1. Extract methodology (tables, figures, processing steps, regression specs)
    2. Generate structural templates (markdown for tables, matplotlib for figures)

    When use_vision=True, PDF pages are sent as images instead of extracted text.
    This gives the model direct visual access to table layouts, improving cell-level
    accuracy for template generation.
    """

    def __init__(
        self,
        config: Config,
        model: str = "gpt-5.2",
        max_tokens: int = 128000,
        use_vision: bool = True,
        vision_dpi: int = 200,
    ):
        self.config = config
        self.model = model
        self.max_tokens = max_tokens
        self.use_vision = use_vision
        self.vision_dpi = vision_dpi
        self.client = OpenAI(api_key=config.openai_api_key)
        logger.info(f"Initialized ExtractorAgent with model={model}, use_vision={use_vision}")

    def run(
        self,
        paper_path: str,
        paper_id: Optional[str] = None,
    ) -> tuple[PaperSummary, ExtractionUsage]:
        """Extract methodology from a paper.

        Args:
            paper_path: Path to the PDF paper.
            paper_id: Optional identifier for the paper.

        Returns:
            Tuple of (PaperSummary, ExtractionUsage).
        """
        self._usage = ExtractionUsage()
        logger.info(f"Extracting methodology from: {paper_path} (vision={self.use_vision})")

        if paper_id is None:
            paper_id = Path(paper_path).stem

        # Parse PDF text (always needed for captions and as fallback)
        paper_text = extract_text_from_pdf(paper_path)
        table_captions = extract_table_captions(paper_text)
        figure_captions = extract_figure_captions(paper_text)

        table_captions_str = "\n".join(
            f"- {t['table_number']}: {t['caption']}" for t in table_captions
        ) or "No table captions detected"
        figure_captions_str = "\n".join(
            f"- {f['figure_number']}: {f['caption']}" for f in figure_captions
        ) or "No figure captions detected"

        # Convert PDF to images if vision mode
        page_images = None
        if self.use_vision:
            page_images = pdf_to_base64_images(paper_path, dpi=self.vision_dpi)
            logger.info(f"Converted PDF to {len(page_images)} page images")

        # ----- Step 1: Extract methodology (structured output) -----
        logger.info("Step 1: Extracting methodology...")

        # Build user prompt (same structure for both vision and text paths;
        # all extraction rules live in the system prompt)
        if self.use_vision:
            user_text = EXTRACTION_USER_PROMPT.format(
                paper_text="(See attached page images below.)",
                table_captions=table_captions_str,
                figure_captions=figure_captions_str,
            )
            input_content = self._build_vision_input(user_text, page_images)
        else:
            safe_paper_text = paper_text[:200000].replace("{", "{{").replace("}", "}}")
            safe_table_captions = table_captions_str.replace("{", "{{").replace("}", "}}")
            safe_figure_captions = figure_captions_str.replace("{", "{{").replace("}", "}}")
            user_prompt = EXTRACTION_USER_PROMPT.format(
                paper_text=safe_paper_text,
                table_captions=safe_table_captions,
                figure_captions=safe_figure_captions,
            )
            input_content = user_prompt

        extraction = self._call_structured(
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            input_content=input_content,
            response_model=ExtractionResponse,
            step_label="extraction",
        )

        # Override paper_id
        extraction.paper_id = paper_id

        # Convert to PaperSummary
        summary = self._to_paper_summary(extraction)

        # Validate no results leaked
        self._validate_no_results(summary)

        # ----- Step 2: Generate templates -----
        logger.info("Step 2: Generating structural templates...")
        self._generate_templates(paper_text, summary, page_images=page_images)

        logger.info(
            f"Extraction complete: {len(summary.tables)} tables, "
            f"{len(summary.figures)} figures | "
            f"tokens: {self._usage.total_input_tokens:,} in + "
            f"{self._usage.total_output_tokens:,} out = "
            f"{self._usage.total_tokens:,} total | "
            f"{self._usage.total_duration:.1f}s"
        )
        return summary, self._usage

    def _build_vision_input(self, text_prompt: str, page_images: list[dict]) -> list[dict]:
        """Build a multimodal input list with text + page images for the Responses API.

        The Responses API expects input as a list of message objects:
        [{"role": "user", "content": [{"type": "input_text", ...}, {"type": "input_image", ...}]}]
        """
        content = [
            {"type": "input_text", "text": text_prompt},
        ]
        for img in page_images:
            content.append({
                "type": "input_image",
                "image_url": f"data:image/png;base64,{img['base64']}",
                "detail": "high",
            })
        return [{"role": "user", "content": content}]

    def _call_structured(
        self,
        system_prompt: str,
        input_content,  # str or list[dict] for multimodal
        response_model: type[BaseModel],
        step_label: str = "",
    ) -> BaseModel:
        """Make a structured-output API call using the Responses API.

        Uses `client.responses.parse()` with a Pydantic model as `text_format`
        to guarantee valid, schema-conforming output.

        Args:
            system_prompt: System instructions.
            input_content: Either a string (text-only) or a list of content
                blocks (multimodal with images).
            response_model: Pydantic model for structured output.
            step_label: Label for usage tracking (e.g. "extraction", "templates").
        """
        t0 = time.time()
        # Reasoning models (o3*, o4*, gpt-5-mini, gpt-5-nano, etc.) don't support temperature
        is_reasoning = any(
            self.model.startswith(p)
            for p in ("o3", "o4", "gpt-5-mini", "gpt-5-nano", "gpt-5-pro")
        )
        kwargs = dict(
            model=self.model,
            instructions=system_prompt,
            input=input_content,
            max_output_tokens=self.max_tokens,
            text_format=response_model,
        )
        if not is_reasoning:
            kwargs["temperature"] = 0.1
        response = self.client.responses.parse(**kwargs)
        duration = time.time() - t0

        # Track usage
        usage = response.usage
        input_details = getattr(usage, "input_tokens_details", None)
        output_details = getattr(usage, "output_tokens_details", None)

        record = APICallRecord(
            step=step_label,
            model=response.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            cached_tokens=getattr(input_details, "cached_tokens", 0) if input_details else 0,
            reasoning_tokens=getattr(output_details, "reasoning_tokens", 0) if output_details else 0,
            duration_seconds=duration,
        )
        self._usage.calls.append(record)

        logger.info(
            f"[{step_label}] model={response.model} "
            f"in={usage.input_tokens:,} out={usage.output_tokens:,} "
            f"total={usage.total_tokens:,} time={duration:.1f}s"
        )

        parsed = response.output_parsed
        if parsed is None:
            raise ValueError("Model returned no parsed output (possible refusal or content filter)")
        return parsed

    def _to_paper_summary(self, extraction: ExtractionResponse) -> PaperSummary:
        """Convert extraction response to PaperSummary, filtering non-replicable figures."""

        tables = [
            TableSpec(
                table_number=t.table_number,
                caption=t.caption,
                column_names=t.column_names,
                row_names=t.row_names,
                regression_specs=[RegressionSpec(**s.model_dump()) for s in t.regression_specs],
                data_processing_steps=[DataProcessingStep(**s.model_dump()) for s in t.data_processing_steps],
                notes=t.notes,
                data_source=t.data_source,
                panel_structure=t.panel_structure,
            )
            for t in extraction.tables
        ]

        figures = []
        for f in extraction.figures:
            plot_type = f.plot_type.lower().strip()
            if any(kw in plot_type for kw in NON_REPLICABLE_KEYWORDS):
                logger.info(f"Skipping non-replicable figure {f.figure_number}: {plot_type}")
                continue
            figures.append(
                PlotSpec(
                    figure_number=f.figure_number,
                    caption=f.caption,
                    plot_type=f.plot_type,
                    x_axis=f.x_axis,
                    y_axis=f.y_axis,
                    grouping_vars=f.grouping_vars or None,
                    regression_specs=[RegressionSpec(**s.model_dump()) for s in f.regression_specs],
                    data_processing_steps=[DataProcessingStep(**s.model_dump()) for s in f.data_processing_steps],
                    notes=f.notes,
                    data_source=f.data_source,
                    subplot_structure=f.subplot_structure,
                )
            )

        return PaperSummary(
            paper_id=extraction.paper_id,
            title=extraction.title,
            research_questions=extraction.research_questions,
            data_description=extraction.data_description,
            data_context=extraction.data_context,
            data_source=extraction.data_source,
            sample_size=extraction.sample_size,
            time_period=extraction.time_period,
            data_processing_steps=[
                DataProcessingStep(**s.model_dump()) for s in extraction.data_processing_steps
            ],
            tables=tables,
            figures=figures,
        )

    def _generate_templates(
        self,
        paper_text: str,
        summary: PaperSummary,
        page_images: Optional[list[dict]] = None,
    ) -> None:
        """Generate structural templates for tables and figures.

        Modifies summary in-place. Non-fatal: logs warnings on failure.
        """
        if not summary.tables and not summary.figures:
            return

        table_specs_json = json.dumps(
            [t.model_dump(exclude={"template_markdown"}) for t in summary.tables],
            indent=2,
            default=str,
        )
        figure_specs_json = json.dumps(
            [f.model_dump(exclude={"template_code"}) for f in summary.figures],
            indent=2,
            default=str,
        )

        if self.use_vision and page_images:
            # Build multimodal input for template generation
            text_part = TEMPLATE_GENERATION_USER_PROMPT.format(
                paper_text="[See page images below]",
                table_specs_json=table_specs_json,
                figure_specs_json=figure_specs_json,
            )
            input_content = self._build_vision_input(text_part, page_images)
        else:
            safe_paper_text = paper_text[:200000].replace("{", "{{").replace("}", "}}")
            safe_table_specs = table_specs_json.replace("{", "{{").replace("}", "}}")
            safe_figure_specs = figure_specs_json.replace("{", "{{").replace("}", "}}")
            input_content = TEMPLATE_GENERATION_USER_PROMPT.format(
                paper_text=safe_paper_text,
                table_specs_json=safe_table_specs,
                figure_specs_json=safe_figure_specs,
            )

        try:
            result = self._call_structured(
                system_prompt=TEMPLATE_GENERATION_SYSTEM_PROMPT,
                input_content=input_content,
                response_model=TemplateResponse,
                step_label="templates",
            )

            # Build lookup dicts from the list responses
            table_tmpl_map = {t.table_number: t.template_markdown for t in result.table_templates}
            figure_tmpl_map = {f.figure_number: f.template_code for f in result.figure_templates}

            logger.info(f"Template keys: tables={list(table_tmpl_map.keys())}, "
                        f"figures={list(figure_tmpl_map.keys())}")

            for table_spec in summary.tables:
                template = table_tmpl_map.get(table_spec.table_number)
                if template:
                    table_spec.template_markdown = template
                    logger.info(f"Set template for {table_spec.table_number}")
                else:
                    logger.warning(f"No template found for {table_spec.table_number}")

            for figure_spec in summary.figures:
                template = figure_tmpl_map.get(figure_spec.figure_number)
                if template:
                    figure_spec.template_code = template
                    logger.info(f"Set template for {figure_spec.figure_number}")
                else:
                    logger.warning(f"No template found for {figure_spec.figure_number}")

        except Exception as e:
            logger.warning(f"Template generation failed (non-fatal): {e}")

    def _validate_no_results(self, summary: PaperSummary) -> None:
        """Warn if the summary appears to contain actual results."""
        result_patterns = [
            r"\b\d+\.\d+\s*\*+",  # Numbers with significance stars
            r"p\s*[<>=]\s*0\.\d+",  # P-values
            r"significant(ly)?\s+(positive|negative)",  # Significance statements
            r"(increases?|decreases?)\s+by\s+\d+",  # Effect descriptions
        ]

        text_to_check = " ".join([
            summary.data_description,
            summary.data_context,
            *[t.caption for t in summary.tables],
            *[f.caption for f in summary.figures],
        ])

        for pattern in result_patterns:
            if re.search(pattern, text_to_check, re.IGNORECASE):
                logger.warning(f"Potential results leak detected: {pattern}")
