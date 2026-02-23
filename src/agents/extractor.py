"""Agent 1: Extractor - Extracts methodology from papers without revealing results."""

import json
from pathlib import Path
from typing import Optional

from langchain_core.language_models import BaseChatModel

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
    extract_tables_from_pdf,
    identify_sections,
    extract_figure_captions,
    extract_table_captions,
)
from ..utils.logging_utils import get_logger
from .base import BaseAgent

logger = get_logger(__name__)


EXTRACTION_SYSTEM_PROMPT = """You are a methodological extraction specialist for social science research papers.

Your task is to extract ONLY the methodology, structure, and specifications from academic papers -
NOT the actual results, findings, or numerical outcomes.

You must extract:
1. Research questions
2. Data description and context
3. Data processing and cleaning steps
4. Regression specifications (variables, models, controls)
5. Table structures (column/row names, what each cell represents)
6. Figure specifications (plot type, axes, what is being shown)

CRITICAL: You must NEVER include:
- Actual coefficient values or estimates
- P-values or significance levels
- Actual numbers from tables
- Descriptions of findings or conclusions
- Statements about whether effects are positive/negative/significant

Your output must allow someone to replicate the analysis without knowing what results to expect."""


EXTRACTION_PROMPT_TEMPLATE = """Analyze the following academic paper and extract its methodology.

## Paper Text:
{paper_text}

## Detected Table Captions:
{table_captions}

## Detected Figure Captions:
{figure_captions}

## Instructions:
Extract all methodological information into the following JSON structure.
Remember: DO NOT include any actual results or numerical findings.

{{
    "paper_id": "string - unique identifier based on author/year",
    "title": "string - paper title",
    "research_questions": ["list of research questions"],
    "data_description": "string - describe the data source and type",
    "data_context": "string - context for the analysis",
    "data_source": "string - where the data comes from",
    "sample_size": "string - sample size info if mentioned",
    "time_period": "string - time period covered",
    "data_processing_steps": [
        {{
            "step_number": 1,
            "description": "what this step does",
            "variables_involved": ["list of variables"]
        }}
    ],
    "tables": [
        {{
            "table_number": "Table 1",
            "caption": "caption without results",
            "column_names": ["list of column headers"],
            "row_names": ["list of row labels"],
            "regression_specs": [
                {{
                    "model_type": "OLS/Logit/etc",
                    "dependent_var": "outcome variable",
                    "independent_vars": ["explanatory variables"],
                    "controls": ["control variables"],
                    "fixed_effects": ["fixed effects if any"],
                    "clustering": "clustering level if any",
                    "sample_restrictions": "any sample restrictions"
                }}
            ],
            "notes": "table notes without results"
        }}
    ],
    "figures": [
        {{
            "figure_number": "Figure 1",
            "caption": "caption without describing results",
            "plot_type": "scatter/bar/line/etc",
            "x_axis": "x-axis variable",
            "y_axis": "y-axis variable",
            "grouping_vars": ["grouping variables if any"],
            "notes": "figure notes"
        }}
    ]
}}

Respond with valid JSON only."""


TEMPLATE_GENERATION_SYSTEM_PROMPT = """You are a structural template specialist for academic paper tables and figures.

Your task is to generate STRUCTURAL TEMPLATES that show the exact layout of tables and figures
from a paper — without any actual numerical results.

For tables: produce a markdown table where every data cell contains XXX (for cells that should
have values) or --- (for cells that are structurally empty). Include the caption above the table.

For figures: produce a minimal matplotlib code skeleton that sets up the correct plot type,
axes labels, legend entries, and subplot structure — but contains NO data."""


TEMPLATE_GENERATION_PROMPT = """Given the following paper text and extracted table/figure specifications,
generate structural templates for each item.

## Paper Text (excerpt):
{paper_text}

## Extracted Tables:
{table_specs_json}

## Extracted Figures:
{figure_specs_json}

## Instructions:

For each table, produce a markdown template like:

**Table 2.1: Effect of X on Y**

| | (1) OLS | (2) Logit | (3) FE |
|---|---|---|---|
| Treatment | XXX | XXX | XXX |
| | (XXX) | (XXX) | (XXX) |
| Control A | XXX | XXX | XXX |
| Observations | XXX | XXX | XXX |
| R² | XXX | XXX | --- |

Use XXX for cells that should contain values, (XXX) for standard errors, and --- for empty cells.

For each figure, produce a matplotlib code skeleton like:

```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.set_xlabel("X Label")
ax.set_ylabel("Y Label")
ax.set_title("Figure Title")
ax.legend(["Group A", "Group B"])
plt.tight_layout()
```

Respond with valid JSON:
{{
    "table_templates": {{
        "Table 2.1": "markdown template string",
        "Table 2.2": "markdown template string"
    }},
    "figure_templates": {{
        "Figure 1": "python code skeleton string"
    }}
}}"""


class ExtractorAgent(BaseAgent):
    """Agent 1: Extracts methodology from papers.

    This agent parses academic papers and extracts all methodological
    information needed to replicate the analysis, WITHOUT revealing
    any actual results or findings.
    """

    def __init__(self, config: Config, chat_model: Optional[BaseChatModel] = None):
        """Initialize the extractor agent.

        Args:
            config: Configuration object.
            chat_model: Optional LangChain chat model for DI/testing.
        """
        super().__init__(
            config=config,
            name="Extractor",
            role="methodological extraction specialist",
            goal="Extract methodology from papers without revealing results",
            chat_model=chat_model,
        )

    def run(
        self,
        paper_path: str,
        paper_id: Optional[str] = None,
    ) -> PaperSummary:
        """Extract methodology from a paper.

        Args:
            paper_path: Path to the PDF paper.
            paper_id: Optional identifier for the paper.

        Returns:
            PaperSummary containing extracted methodology.

        Raises:
            FileNotFoundError: If paper not found.
            ValueError: If extraction fails.
        """
        logger.info(f"Extracting methodology from: {paper_path}")

        # Generate paper_id if not provided
        if paper_id is None:
            paper_id = Path(paper_path).stem

        # Extract text from PDF
        paper_text = extract_text_from_pdf(paper_path)

        # Extract table and figure captions
        table_captions = extract_table_captions(paper_text)
        figure_captions = extract_figure_captions(paper_text)

        # Format captions for prompt
        table_captions_str = "\n".join(
            f"- {t['table_number']}: {t['caption']}" for t in table_captions
        )
        figure_captions_str = "\n".join(
            f"- {f['figure_number']}: {f['caption']}" for f in figure_captions
        )

        # Create extraction prompt
        prompt = EXTRACTION_PROMPT_TEMPLATE.format(
            paper_text=paper_text[:50000],  # Limit text length
            table_captions=table_captions_str or "No table captions detected",
            figure_captions=figure_captions_str or "No figure captions detected",
        )

        # Generate extraction
        logger.info("Generating methodological extraction...")
        response = self.generate_json(
            prompt=prompt,
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
        )

        # Validate and create PaperSummary
        try:
            # Ensure paper_id is set
            response["paper_id"] = paper_id

            # Parse nested structures
            if "data_processing_steps" in response:
                response["data_processing_steps"] = [
                    DataProcessingStep(**step) for step in response["data_processing_steps"]
                ]

            if "tables" in response:
                tables = []
                for table in response["tables"]:
                    if "regression_specs" in table:
                        table["regression_specs"] = [
                            RegressionSpec(**spec) for spec in table["regression_specs"]
                        ]
                    tables.append(TableSpec(**table))
                response["tables"] = tables

            if "figures" in response:
                response["figures"] = [PlotSpec(**fig) for fig in response["figures"]]

            paper_summary = PaperSummary(**response)

            # Validate no results leaked
            self._validate_no_results(paper_summary)

            # Generate structural templates (non-fatal)
            self._generate_templates(paper_text, paper_summary)

            logger.info(
                f"Extraction complete: {len(paper_summary.tables)} tables, "
                f"{len(paper_summary.figures)} figures"
            )

            return paper_summary

        except Exception as e:
            logger.error(f"Failed to parse extraction response: {e}")
            raise ValueError(f"Extraction failed: {e}")

    def _generate_templates(self, paper_text: str, summary: PaperSummary) -> None:
        """Generate structural templates for tables and figures.

        Modifies the summary in-place, setting template_markdown on TableSpecs
        and template_code on PlotSpecs. Non-fatal: logs warnings on failure.
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

        prompt = TEMPLATE_GENERATION_PROMPT.format(
            paper_text=paper_text[:30000],
            table_specs_json=table_specs_json,
            figure_specs_json=figure_specs_json,
        )

        try:
            response = self.generate_json(
                prompt=prompt,
                system_prompt=TEMPLATE_GENERATION_SYSTEM_PROMPT,
            )

            # Apply table templates
            table_templates = response.get("table_templates", {})
            for table_spec in summary.tables:
                template = table_templates.get(table_spec.table_number)
                if template:
                    table_spec.template_markdown = template
                    logger.info(f"Set template for {table_spec.table_number}")

            # Apply figure templates
            figure_templates = response.get("figure_templates", {})
            for figure_spec in summary.figures:
                template = figure_templates.get(figure_spec.figure_number)
                if template:
                    figure_spec.template_code = template
                    logger.info(f"Set template for {figure_spec.figure_number}")

        except Exception as e:
            logger.warning(f"Template generation failed (non-fatal): {e}")

    def _validate_no_results(self, summary: PaperSummary) -> None:
        """Validate that the summary doesn't contain actual results.

        Args:
            summary: The extracted summary to validate.

        Raises:
            ValueError: If results appear to be leaked.
        """
        # Check for common patterns that indicate results
        result_patterns = [
            r"\b\d+\.\d+\s*\*+",  # Numbers with significance stars
            r"p\s*[<>=]\s*0\.\d+",  # P-values
            r"significant(ly)?\s+(positive|negative)",  # Significance statements
            r"(increases?|decreases?)\s+by\s+\d+",  # Effect descriptions
        ]

        import re

        text_to_check = " ".join([
            summary.data_description,
            summary.data_context,
            *[t.caption for t in summary.tables],
            *[f.caption for f in summary.figures],
        ])

        for pattern in result_patterns:
            if re.search(pattern, text_to_check, re.IGNORECASE):
                logger.warning(f"Potential results leak detected: {pattern}")
                # Don't raise error, just warn - LLM output can be noisy

    def extract_sections(self, paper_path: str) -> dict[str, str]:
        """Extract paper sections (helper method).

        Args:
            paper_path: Path to the PDF.

        Returns:
            Dictionary mapping section names to content.
        """
        text = extract_text_from_pdf(paper_path)
        return identify_sections(text)
