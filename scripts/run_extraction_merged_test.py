"""Test script: single-step merged extraction (extraction + templates in one API call).

Compares against the two-step approach by producing the same output format.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Optional
from openai import OpenAI
from pydantic import BaseModel

from src.models.config import load_config
from src.models.schemas import (
    PaperSummary, DataProcessingStep, RegressionSpec, TableSpec, PlotSpec,
)
from src.utils.pdf_parser import (
    extract_text_from_pdf, extract_figure_captions, extract_table_captions,
    pdf_to_base64_images,
)
from src.benchmark.task_prompt import build_task_prompt


# ---------------------------------------------------------------------------
# Merged response schema: extraction + templates in one
# ---------------------------------------------------------------------------

class RegressionSpecMerged(BaseModel):
    model_type: str
    dependent_var: str
    independent_vars: list[str] = []
    controls: list[str] = []
    fixed_effects: list[str] = []
    clustering: Optional[str] = None
    sample_restrictions: Optional[str] = None
    equation_latex: Optional[str] = None
    variable_definitions: Optional[str] = None
    additional_notes: Optional[str] = None


class TableSpecMerged(BaseModel):
    table_number: str
    caption: str
    column_names: list[str] = []
    row_names: list[str] = []
    regression_specs: list[RegressionSpecMerged] = []
    notes: Optional[str] = None
    panel_structure: Optional[str] = None
    template_markdown: Optional[str] = None  # <-- merged: template inline


class PlotSpecMerged(BaseModel):
    figure_number: str
    caption: str
    plot_type: str
    x_axis: Optional[str] = None
    y_axis: Optional[str] = None
    grouping_vars: list[str] = []
    regression_specs: list[RegressionSpecMerged] = []
    notes: Optional[str] = None
    subplot_structure: Optional[str] = None
    template_code: Optional[str] = None  # <-- merged: template inline


class DataProcessingStepMerged(BaseModel):
    step_number: int
    description: str
    variables_involved: list[str] = []


class MergedExtractionResponse(BaseModel):
    """Single-call schema: methodology extraction + templates together."""
    paper_id: str
    title: Optional[str] = None
    research_questions: list[str] = []
    data_description: str
    data_context: str
    data_source: Optional[str] = None
    sample_size: Optional[str] = None
    time_period: Optional[str] = None
    data_processing_steps: list[DataProcessingStepMerged] = []
    tables: list[TableSpecMerged] = []
    figures: list[PlotSpecMerged] = []


# ---------------------------------------------------------------------------
# Merged prompt
# ---------------------------------------------------------------------------

MERGED_SYSTEM_PROMPT = """You are a methodological extraction and template specialist for social science research papers.

Your task is to extract the methodology, structure, and specifications from academic papers
AND generate structural templates for each table and figure — all in a single pass.

You must extract:
1. Research questions
2. Data description and context
3. Data processing and cleaning steps
4. Regression specifications (variables, models, controls) — attached to the table or figure they belong to
5. Table structures — EXACT column headers, row labels, AND a markdown template
6. Figure specifications — FULL captions, exact plot type, axis labels, AND a matplotlib code skeleton

## IMPORTANT: Extract ALL tables and figures by default.
You must extract EVERY table and figure from the paper. The ONLY exceptions you may skip are:
- Figures that are purely conceptual visualizations (flow diagrams, conceptual frameworks,
  screenshots, photos, timelines, maps) that cannot be reproduced from data.

## Regression specification rules:
- For regression TABLES: include one `regression_spec` per distinct model variant.
- For regression-based FIGURES: include `regression_specs` describing the underlying estimation.
- Use the actual variable names from the paper.

## Regression equation extraction:
- For each regression spec, fill in `equation_latex` with the exact equation from the paper in
  LaTeX notation. If no explicit equation exists, write one from the prose description.
- Fill in `variable_definitions` with a verbal mapping of every symbol, separated by semicolons.

## What counts as "results" (NEVER include):
- Regression coefficients, standard errors, t-statistics, p-values
- Significance stars or effect sizes

## What is NOT results (DO include):
- Table/figure structure, design parameters, variable names, sample sizes

## Table template rules (fill in `template_markdown` for each table):
1. EXACTLY the same number of columns and rows as the original table.
2. EXACT column headers and row labels from the paper.
3. Cell content:
   - **XXX**: computed result (coefficient, statistic, count, mean, etc.)
   - **(XXX)**: standard error in parentheses
   - **Empty cell**: blank in original table — just leave empty, no placeholder
   - **Literal text**: "Yes", "No", checkmarks — copy literally
4. Include panel headers when present.

Example table template:

| | (1) | (2) | (3) | (4) |
|---|---|---|---|---|
| Variable A | XXX | XXX | XXX | XXX |
| | (XXX) | (XXX) | (XXX) | (XXX) |
| Variable B | | | XXX | XXX |
| | | | (XXX) | (XXX) |
| Controls | No | Yes | No | Yes |
| Observations | XXX | XXX | XXX | XXX |
| R² | XXX | XXX | | |

## Figure template rules (fill in `template_code` for each figure):
1. Matplotlib code skeleton with correct plot functions, colors, line styles, axis labels.
2. Set approximate axis limits from what is shown in the paper.
3. Use actual series names / legend labels from the paper.
4. NO actual data — use comments like `# TODO: fill with data`.
5. The skeleton should be runnable even without data.

Your output must allow someone to replicate the analysis without knowing what results to expect."""


MERGED_VISION_INSTRUCTIONS = """
## Instructions:

Carefully examine every page of the paper (provided as images) and extract all
methodological information AND generate structural templates in a single pass.

### For each table:
- Copy EXACT column headers and row labels from the paper.
- Fill in `regression_specs` with equation_latex and variable_definitions.
- Fill in `template_markdown` by looking at the actual table layout:
  XXX for results, (XXX) for standard errors, empty for blank cells, literals for text.

### For each figure:
- Copy full caption, plot type, axis labels, legend entries.
- Fill in `template_code` with a matplotlib skeleton.

### For data processing:
- Focus on steps needed to go from raw data to analysis sample.
"""


# ---------------------------------------------------------------------------
# Non-replicable figure keywords
# ---------------------------------------------------------------------------

NON_REPLICABLE_KEYWORDS = [
    "flow diagram", "flow chart", "flowchart", "conceptual framework",
    "conceptual diagram", "diagram", "schematic", "screenshot", "photo",
    "photograph", "timeline",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    paper_id = "yellow_vests_carbon_tax"
    pdf_path = "data/input/yellow_vests_carbon_tax/paper.pdf"
    output_dir = Path("data/benchmark_results/extraction_test_merged")
    output_dir.mkdir(parents=True, exist_ok=True)

    use_vision = "--vision" in sys.argv
    mode = "VISION" if use_vision else "TEXT"
    print(f"Running MERGED extraction in {mode} mode")

    config = load_config()
    client = OpenAI(api_key=config.openai_api_key)
    model = "gpt-5.2"

    # Parse PDF
    paper_text = extract_text_from_pdf(pdf_path)
    table_captions = extract_table_captions(paper_text)
    figure_captions = extract_figure_captions(paper_text)

    table_captions_str = "\n".join(
        f"- {t['table_number']}: {t['caption']}" for t in table_captions
    ) or "No table captions detected"
    figure_captions_str = "\n".join(
        f"- {f['figure_number']}: {f['caption']}" for f in figure_captions
    ) or "No figure captions detected"

    # Build input
    text_prompt = (
        "Analyze the following academic paper and extract its methodology "
        "AND generate structural templates for every table and figure.\n\n"
        f"## Detected Table Captions:\n{table_captions_str}\n\n"
        f"## Detected Figure Captions:\n{figure_captions_str}\n\n"
        + MERGED_VISION_INSTRUCTIONS
    )

    if use_vision:
        page_images = pdf_to_base64_images(pdf_path, dpi=200)
        print(f"Converted PDF to {len(page_images)} page images")
        content = [{"type": "input_text", "text": text_prompt}]
        for img in page_images:
            content.append({
                "type": "input_image",
                "image_url": f"data:image/png;base64,{img['base64']}",
                "detail": "high",
            })
        input_content = [{"role": "user", "content": content}]
    else:
        input_content = text_prompt + f"\n\n## Paper Text:\n{paper_text[:200000]}"

    # Single API call
    print("Calling API (single merged call)...")
    t0 = time.time()
    response = client.responses.parse(
        model=model,
        instructions=MERGED_SYSTEM_PROMPT,
        input=input_content,
        max_output_tokens=128000,
        temperature=0.1,
        text_format=MergedExtractionResponse,
    )
    duration = time.time() - t0

    result = response.output_parsed
    result.paper_id = paper_id
    usage = response.usage

    # Filter non-replicable figures
    filtered_figures = []
    for f in result.figures:
        plot_type = f.plot_type.lower().strip()
        if any(kw in plot_type for kw in NON_REPLICABLE_KEYWORDS):
            print(f"  Skipping non-replicable: {f.figure_number} ({plot_type})")
            continue
        filtered_figures.append(f)

    # Convert to PaperSummary
    summary = PaperSummary(
        paper_id=result.paper_id,
        title=result.title,
        research_questions=result.research_questions,
        data_description=result.data_description,
        data_context=result.data_context,
        data_source=result.data_source,
        sample_size=result.sample_size,
        time_period=result.time_period,
        data_processing_steps=[
            DataProcessingStep(**s.model_dump()) for s in result.data_processing_steps
        ],
        tables=[
            TableSpec(
                table_number=t.table_number,
                caption=t.caption,
                column_names=t.column_names,
                row_names=t.row_names,
                regression_specs=[RegressionSpec(**s.model_dump()) for s in t.regression_specs],
                notes=t.notes,
                panel_structure=t.panel_structure,
                template_markdown=t.template_markdown,
            )
            for t in result.tables
        ],
        figures=[
            PlotSpec(
                figure_number=f.figure_number,
                caption=f.caption,
                plot_type=f.plot_type,
                x_axis=f.x_axis,
                y_axis=f.y_axis,
                grouping_vars=f.grouping_vars or None,
                regression_specs=[RegressionSpec(**s.model_dump()) for s in f.regression_specs],
                notes=f.notes,
                subplot_structure=f.subplot_structure,
                template_code=f.template_code,
            )
            for f in filtered_figures
        ],
    )

    # Save outputs
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary.model_dump(), indent=2, default=str))

    task_md = build_task_prompt(summary, "data/")
    task_path = output_dir / "TASK.md"
    task_path.write_text(task_md)

    usage_data = {
        "total_input_tokens": usage.input_tokens,
        "total_output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "total_duration_seconds": round(duration, 2),
        "num_api_calls": 1,
        "model": response.model,
        "calls": [{
            "step": "merged",
            "model": response.model,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
            "duration_seconds": round(duration, 2),
        }],
    }
    (output_dir / "usage.json").write_text(json.dumps(usage_data, indent=2))

    # Print results
    print(f"\nSaved to {output_dir}")
    print(f"  Tables: {len(summary.tables)}")
    for t in summary.tables:
        has_tmpl = "YES" if t.template_markdown else "NO"
        print(f"    - {t.table_number} (template: {has_tmpl}): {t.caption[:60]}")
    print(f"  Figures: {len(summary.figures)}")
    for f in summary.figures:
        has_tmpl = "YES" if f.template_code else "NO"
        print(f"    - {f.figure_number} (template: {has_tmpl}): {f.caption[:60]}")

    print(f"\nToken usage:")
    print(f"  merged: {usage.input_tokens:,} in + {usage.output_tokens:,} out "
          f"= {usage.total_tokens:,} ({duration:.1f}s) [{response.model}]")


if __name__ == "__main__":
    main()
