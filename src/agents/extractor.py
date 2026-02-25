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


class TableSpecResponse(BaseModel):
    table_number: str
    caption: str
    column_names: list[str] = []
    row_names: list[str] = []
    regression_specs: list[RegressionSpecResponse] = []
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
    notes: Optional[str] = None
    data_source: Optional[str] = None
    subplot_structure: Optional[str] = None


class DataProcessingStepResponse(BaseModel):
    step_number: int
    description: str
    variables_involved: list[str] = []


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

Your task is to extract ONLY the methodology, structure, and specifications from academic papers -
NOT the actual results, findings, or numerical outcomes.

You must extract:
1. Research questions
2. Data description and context
3. Data processing and cleaning steps
4. Regression specifications (variables, models, controls) — attached to the table or figure they belong to
5. Table structures — EXACT column headers and row labels as they appear in the paper
6. Figure specifications — FULL captions, exact plot type, axis labels, legend entries

## IMPORTANT: Extract ALL tables and figures by default.
You must extract EVERY table and figure from the paper. The ONLY exceptions you may skip are:
- Figures that are purely conceptual visualizations (flow diagrams, conceptual frameworks,
  screenshots, photos, timelines, maps) that cannot be reproduced from data.
Do NOT skip tables or figures just because they seem simple — summary statistics tables,
design parameter tables, cross-tabulations, balance tables, and descriptive tables should
ALL be extracted. When in doubt, extract it.

## Cross-reference resolution (CRITICAL):
- When a table note, regression specification, or methodology description references content
  elsewhere in the paper (e.g., "see Appendix H for the list of controls", "variables defined
  in Section 2", "specification as in equation (3)"), you MUST go to that location in the paper
  and extract the actual referenced content.
- NEVER leave unresolved forward/backward references like "see Appendix H" or "controls listed
  in Table A3". Replace them with the actual variable list, equation, or definition found at
  that location.
- This is especially important for CONTROL VARIABLE LISTS. Papers commonly say "controls include
  socio-demographic variables (see Appendix X)." You must find Appendix X and list every single
  control variable individually.

## Regression specification rules:
- Every regression specification must be attached to the specific table or figure that displays
  its results. Infer this mapping from the paper text — the methodology sections typically say
  things like "Table 3 reports the results of estimating equation (2)" or "Figure 5 plots the
  coefficients from specification (1)".
- For regression TABLES: include one `regression_spec` describing the general model estimated.
  If the table has multiple columns with variations (e.g., different controls or subsamples),
  include one spec per distinct model variant. Describe model type, dependent variable,
  independent variables, controls, fixed effects, clustering, and sample restrictions.
- For regression-based FIGURES (coefficient plots, RDD plots, binned scatters with fit lines):
  include `regression_specs` on the figure describing the underlying estimation.
- Use the actual variable names from the paper's variable definitions or table notes, not
  generic descriptions. If the paper says "we regress log(wage) on education", write
  `dependent_var: "log(wage)"`, not `dependent_var: "outcome variable"`.

## Control variable enumeration (CRITICAL):
- In the `controls` field of each regression spec, enumerate EVERY control variable individually.
  NEVER use aggregate descriptions like "socio-demographic controls" or "standard controls."
  Instead, list each variable separately: ["age", "female", "education_level", "income", ...].
- If the paper lists controls only in an appendix or another section, go to that location and
  extract the complete list. This is essential — a replicator who cannot see the paper needs to
  know every single variable in the model.

## Regression equation extraction:
- For each regression spec, extract the **exact equation** from the paper in LaTeX notation
  and store it in `equation_latex`. Copy the equation as written in the paper, preserving all
  subscripts, superscripts, Greek letters, and notation. Examples:
    - `A_i^T = \\alpha + \\beta \\bar{G}_i^T + f(I_{1,i}, I_{2,i}) + c_i + \\varepsilon_i`
    - `Y_i = \\beta_0 + \\beta_1 X_i + \\gamma Z_i + \\delta_j + \\epsilon_i`
  If the paper has numbered equations (e.g., "equation (3)"), copy the exact equation.
  If no explicit equation is written but the specification is described in prose, write the
  equation yourself based on the prose description using standard econometric notation.
- For each equation, provide `variable_definitions`: a verbal mapping of every symbol to its
  meaning. Include ALL information the paper provides about each variable — definition,
  construction, coding, unit, sign convention, omitted categories, data source. Only include
  what is explicitly stated in the paper; do not infer or guess. Write one definition per
  variable, separated by semicolons. Example:
    - `A_i^T: acceptance of targeted tax, equals 1 if respondent did not answer "No" and 0
      otherwise (Section 2.3); \\bar{G}_i^T: believes does not lose under targeted reform,
      endogenous, instrumented by eligibility indicators T_1 and T_2; f(I_1, I_2): flexible
      polynomial in respondent income I_1 and second-adult income I_2 (both in €/month);
      c_i: income-threshold fixed effects (4 levels: bottom 20/30/40/50 percentile);
      \\varepsilon_i: error term`

## Comprehensive variable extraction (CRITICAL for replication):
- Extract ALL variables used to compute results — dependent variables, independent variables,
  controls, instruments, weights, and any intermediate/constructed variables.
- For each variable, extract ALL information the paper provides about it. This includes but is
  not limited to: how it is defined, how it is constructed from raw data, its coding or scaling,
  its unit of measurement, sign conventions, reference/omitted categories for dummies, sample
  restrictions it implies, and which survey question or data source it comes from.
- Do NOT infer or guess information the paper does not state. Only extract what is explicitly
  written. If the paper does not specify a sign convention or unit, do not invent one.
- DO be thorough: if the paper defines a variable in the methodology section, restates it
  differently in a table note, and adds detail in an appendix, combine all of that information.
  Follow every cross-reference to collect the complete picture.
- The goal: a replicator who cannot see the paper should have enough information to construct
  every variable exactly as the authors did, based solely on your extraction.

## Per-table data source:
- If different tables or figures use different datasets or subsamples from different surveys,
  specify in `sample_restrictions` or `additional_notes` which exact data source is required.
  Example: "Uses EL 2013 housing survey (N≈27,000), NOT the main survey sample (N≈3,000)"
- This is critical when a paper combines multiple datasets (e.g., a main survey plus
  administrative data or official statistics).

## What counts as "results" (NEVER include):
- Regression coefficients, standard errors, t-statistics, p-values
- Significance stars or statements about significance
- Point estimates, confidence intervals, or effect sizes
- Descriptions of direction or magnitude of effects

## What is NOT results (DO include):
- Table/figure structure: exact column headers, row labels, panel labels
- Design parameters: sample sizes, thresholds, treatment assignments, variable names
- Descriptive labels that identify what each cell SHOULD contain (e.g. "N", "R²", "Controls")

## Critical rules for table extraction:
- Copy the EXACT column headers from the paper (e.g., "(1)", "(2)", "(3)" or "OLS", "IV", etc.)
- Copy the EXACT row labels from the paper (e.g., the actual variable names used)
- Count columns and rows precisely. If the paper has 7 columns, you must list exactly 7.
- Include panel structure (Panel A / Panel B) when present

## Critical rules for figure extraction:
- Copy the FULL caption from the paper (including subtitles and notes)
- Skip ONLY figures that are purely conceptual visualizations (flow diagrams, frameworks, photos)
- For each data-based figure: specify the exact plot type, axis labels, series/legend entries,
  and subplot arrangement as shown in the paper

Your output must allow someone to replicate the analysis without knowing what results to expect."""


EXTRACTION_USER_PROMPT = """Analyze the following academic paper and extract its methodology.

## Paper Text:
{paper_text}

## Detected Table Captions:
{table_captions}

## Detected Figure Captions:
{figure_captions}

## Instructions:

Carefully read the paper and extract all methodological information.

### Table extraction rules:
- Extract ALL tables from the paper. Do not skip any tables.
- For each table, go to the actual table in the paper text and copy the EXACT headers and labels.
- `column_names`: list every column header exactly as printed. If columns are "(1)", "(2)", "(3) OLS", "(4) IV",
  list them as ["(1)", "(2)", "(3) OLS", "(4) IV"]. Count carefully.
- `row_names`: list every row label exactly as printed. Include variable names, statistics rows
  ("Observations", "R²", "Controls"), and panel headers ("Panel A: ...", "Panel B: ...").
- For regression tables, fill in `regression_specs` — one per distinct model variant.
  Use the actual variable names from the paper's variable descriptions or table notes.
- `caption`: copy the full caption from the paper.
- `notes`: copy the table notes (footnotes), excluding any that describe specific coefficient values.

### Regression equation rules:
- For every `regression_spec`, you MUST fill in `equation_latex` and `variable_definitions`.
- `equation_latex`: Copy the exact equation from the paper in LaTeX. If the paper writes
  "equation (3): A_i^T = α + β Ḡ_i^T + f(I₁,I₂) + cᵢ + εᵢ", write it as:
  `A_i^T = \\alpha + \\beta \\bar{G}_i^T + f(I_{1,i}, I_{2,i}) + c_i + \\varepsilon_i`
  If the paper does not state an explicit equation but describes the model in prose,
  write the equation yourself in standard econometric LaTeX notation.
- `variable_definitions`: Define every symbol in the equation verbally, separated by semicolons.
  Example: `A_i^T: acceptance of targeted tax (1 if not "No"); \\bar{G}_i^T: believes does not
  lose under targeted reform (binary, endogenous); c_i: threshold fixed effects`

### Figure extraction rules:
- Extract ALL figures EXCEPT purely conceptual visualizations (flow diagrams, conceptual
  frameworks, screenshots, photos) that cannot be reproduced from data.
- For each data-based figure: copy the full caption including any subtitle.
- Identify exact plot type (histogram, kernel density, CDF, scatter, bar, line, box, etc.).
- List the exact axis labels and all legend/series entries.
- If the figure has subplots (panels), describe the subplot_structure.
- In `notes`, include any visual details from the paper: approximate axis ranges, whether there
  are reference lines (e.g., vertical line at zero), color scheme or line style conventions
  (e.g., "solid for objective, dashed for subjective"), and any annotations visible in the figure.

### Data processing rules:
- Focus on steps that are needed to go from the raw dataset to the analysis sample.
- Use the actual variable names from the dataset where possible.
- Include sample restrictions, variable construction, and any transformations."""


TEMPLATE_GENERATION_SYSTEM_PROMPT = """You are a structural template specialist for academic paper tables and figures.

Your task is to generate STRUCTURAL TEMPLATES that faithfully reproduce the exact layout of
tables and figures from a paper.

## Table template rules:

1. The template must have EXACTLY the same number of columns and rows as the original table.
2. Use the EXACT column headers and row labels from the paper.
3. Cell content rules — look at the ORIGINAL TABLE in the paper for each cell:
   - **XXX**: for cells where the paper shows a computed result (coefficient, statistic, count,
     mean, std. dev., p-value, etc.) that must be produced by running code on data.
   - **(XXX)**: for cells where the paper shows a standard error in parentheses.
   - **Empty cell**: leave the cell empty if it is blank in the original table. This includes
     cells where a variable is not part of a particular column's specification. Do NOT use ---
     or any other placeholder for empty cells — just leave them blank.
   - **Literal text**: for cells that contain fixed text like "Yes", "No", checkmarks, or labels.
4. Include panel headers (Panel A, Panel B) as spanning rows when present.
5. Include the full caption above the table.

## Figure template rules:

1. Produce a matplotlib code skeleton that a replicator can fill in with computed data.
2. The skeleton must make the intended visual style unambiguous:
   - Use the correct plot function: `ax.plot()` for lines, `ax.bar()` for bars, `ax.hist()`
     for histograms, `ax.scatter()` for scatter plots, `scipy.stats.gaussian_kde` for KDEs, etc.
   - Set line styles (`linestyle=`, `linewidth=`), marker styles, or bar widths as appropriate.
   - Set colors for each series using a consistent scheme (e.g., `color="tab:blue"`, `color="tab:orange"`).
     Use distinct colors for each series and dashed/solid to distinguish categories (e.g.,
     solid for "objective", dashed for "subjective").
3. Set axis ranges (`ax.set_xlim()`, `ax.set_ylim()`) based on what is visible or described
   in the paper. Use approximate ranges — they don't need to be exact.
4. Use the actual series names / legend labels from the paper.
5. Include the full caption as the title.
6. Set up subplots correctly if the figure has panels.
7. NO actual data arrays — use empty placeholder arrays or comments like `# TODO: fill with data`.
   The skeleton should be runnable (no errors) even without data, just producing an empty styled plot."""


TEMPLATE_GENERATION_USER_PROMPT = """Given the following paper text and extracted table/figure specifications,
generate structural templates for each item. Respond with valid JSON.

## Paper Text (excerpt):
{paper_text}

## Extracted Tables:
{table_specs_json}

## Extracted Figures:
{figure_specs_json}

## Instructions:

Go back to the ORIGINAL TABLE in the paper text and reproduce its structure EXACTLY.

For regression tables, the template should look like this example. Note how Variable C only
appears in columns (3) and (4) — the other columns are left empty because the original table
is blank there:

**Table 3.1: Determinants of Y**

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
| R² | XXX | XXX | | |

Rules:
- XXX = cell where the paper shows a computed result (coefficient, statistic, count, mean, etc.)
- (XXX) = cell where the paper shows a standard error in parentheses
- Empty cell = cell that is blank or shows a dash in the original table. Just leave the cell
  empty in the markdown. Do NOT use --- or any other placeholder.
- "No", "Yes", actual labels = copy literally from the paper
- Count columns from the paper EXACTLY. Do not add or remove columns.
- Count rows from the paper EXACTLY. Include every variable, every statistics row.
- If the table has panels (Panel A, Panel B), include them.

For figures, produce a detailed matplotlib skeleton like this example:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Series 1: Objective gains (solid blue line, KDE)
# x_obj, density_obj = ... # TODO: compute KDE from data
# ax.plot(x_obj, density_obj, color="tab:blue", linestyle="-", linewidth=2, label="Objective")

# Series 2: Subjective gains (dashed red line, KDE)
# x_subj, density_subj = ... # TODO: compute KDE from data
# ax.plot(x_subj, density_subj, color="tab:red", linestyle="--", linewidth=2, label="Subjective")

ax.set_xlabel("Net gain (€/year)", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_xlim(-1500, 1500)
ax.set_ylim(0, 0.003)
ax.axvline(x=0, color="black", linestyle=":", linewidth=0.8)
ax.set_title("Figure 3.1: Distribution of objective vs. subjective net gains")
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig("figure_3.1.png", dpi=150)
```

Rules for figure skeletons:
- Use the correct plot function for the plot type (plot, bar, hist, scatter, step, fill_between, etc.)
- Set colors for each series (use tab:blue, tab:orange, tab:red, tab:green, etc.)
- Use linestyle solid vs dashed to distinguish categories where the paper does so
- Set approximate axis limits (xlim, ylim) based on what is shown in the paper
- Add reference lines (axvline, axhline) if the paper has them (e.g., zero line)
- Include savefig with the correct filename
- Comment out data-dependent lines with # TODO, but keep the styling arguments visible
- The skeleton should convey the full visual intent so the replicator knows exactly what to build"""


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
        use_vision: bool = False,
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

        extraction_text = (
            "Analyze the following academic paper and extract its methodology.\n\n"
            f"## Detected Table Captions:\n{table_captions_str}\n\n"
            f"## Detected Figure Captions:\n{figure_captions_str}\n\n"
        )

        if self.use_vision:
            # Build multimodal input: text instructions + page images
            input_content = self._build_vision_input(
                extraction_text + self._extraction_instructions(),
                page_images,
            )
        else:
            user_prompt = EXTRACTION_USER_PROMPT.format(
                paper_text=paper_text[:200000],
                table_captions=table_captions_str,
                figure_captions=figure_captions_str,
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

    @staticmethod
    def _extraction_instructions() -> str:
        """Return the extraction instructions portion of the user prompt."""
        return """
## Instructions:

Carefully read the paper and extract all methodological information.

### Table extraction rules:
- Extract ALL tables from the paper. Do not skip any tables.
- For each table, look at the actual table in the paper and copy the EXACT headers and labels.
- `column_names`: list every column header exactly as printed. Count carefully.
- `row_names`: list every row label exactly as printed. Include variable names, statistics rows
  ("Observations", "R²", "Controls"), and panel headers ("Panel A: ...", "Panel B: ...").
- For regression tables, fill in `regression_specs` — one per distinct model variant.
- `caption`: copy the full caption from the paper.
- `notes`: copy the table notes (footnotes), excluding any that describe specific coefficient values.

### Regression equation rules:
- For every `regression_spec`, you MUST fill in `equation_latex` and `variable_definitions`.
- `equation_latex`: Copy the exact equation from the paper in LaTeX. If the paper writes
  "equation (3): A_i^T = α + β Ḡ_i^T + f(I₁,I₂) + cᵢ + εᵢ", write it as:
  `A_i^T = \\alpha + \\beta \\bar{G}_i^T + f(I_{1,i}, I_{2,i}) + c_i + \\varepsilon_i`
  If the paper does not state an explicit equation but describes the model in prose,
  write the equation yourself in standard econometric LaTeX notation.
- `variable_definitions`: Define every symbol in the equation verbally, separated by semicolons.
  Include ALL information the paper provides about each variable — how it is defined,
  constructed, coded, scaled, which survey question or data source it comes from, its unit,
  sign convention, and reference/omitted categories. Only include what the paper explicitly
  states; do not infer or guess.
  Example: `A_i^T: acceptance of targeted tax (1 if not "No", 0 if "No", as defined in
  Section 2.3); \\bar{G}_i^T: believes does not lose under targeted reform (binary,
  endogenous, instrumented by eligibility indicators T_1, T_2); c_i: income-threshold
  fixed effects (4 levels: bottom 20/30/40/50 percentile)`

### Cross-reference resolution rules:
- CRITICAL: If any table note, footnote, or regression description says "see Appendix X",
  "controls listed in Table Y", or "as defined in Section Z", you MUST go to that location
  in the paper and extract the actual content. The replicator cannot see the paper.
- For `controls`: enumerate every control variable individually. NEVER write
  "socio-demographic controls (see Appendix H)". Instead, find Appendix H and list:
  ["age", "age_squared", "female", "education_level", "income_decile", ...].

### Sample restriction rules:
- For `sample_restrictions`, extract the exact condition as stated in the paper, including
  any mathematical formulation. Include the expected sample size when given.
  BAD: "Among invalidated respondents"
  GOOD: "Among invalidated respondents, defined as sgn(g_i) ≠ sgn(γ̂_i) (Section 4.1). N = 1,365."

### Figure extraction rules:
- Extract ALL figures EXCEPT purely conceptual visualizations (flow diagrams, conceptual
  frameworks, screenshots, photos) that cannot be reproduced from data.
- For each data-based figure: copy the full caption including any subtitle.
- Identify exact plot type (histogram, kernel density, CDF, scatter, bar, line, box, etc.).
- List the exact axis labels and all legend/series entries.
- If the figure has subplots (panels), describe the subplot_structure.

### Data processing rules:
- Focus on steps that are needed to go from the raw dataset to the analysis sample.
- Use the actual variable names from the dataset where possible.
- Include sample restrictions, variable construction, and any transformations."""

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
        response = self.client.responses.parse(
            model=self.model,
            instructions=system_prompt,
            input=input_content,
            max_output_tokens=self.max_tokens,
            temperature=0.1,
            text_format=response_model,
        )
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
            input_content = TEMPLATE_GENERATION_USER_PROMPT.format(
                paper_text=paper_text[:200000],
                table_specs_json=table_specs_json,
                figure_specs_json=figure_specs_json,
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
