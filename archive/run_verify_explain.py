"""Run verifier and explainer agents on the gpt-5.2-codex freestyle results."""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.models.schemas import (
    PaperSummary,
    ReplicationResults,
    GeneratedCode,
    GeneratedTable,
    GeneratedFigure,
)
from src.models.config import load_config
from src.agents.verifier import VerifierAgent
from src.agents.explainer import ExplainerAgent

# --- Paths ---
WORKSPACE = Path("data/benchmark_results/gpt-5.2-codex_yellow_vests_carbon_tax_freestyle/workspace")
SUMMARY_PATH = Path("data/benchmark_results/summaries/yellow_vests_carbon_tax_summary.json")
PDF_PATH = "data/input/yellow_vests_carbon_tax/paper.pdf"
REPLICATION_PACKAGE = "data/input/yellow_vests_carbon_tax/replication_package"
PAPER_ID = "yellow_vests_carbon_tax"

# --- Load paper summary ---
with open(SUMMARY_PATH) as f:
    paper_summary = PaperSummary(**json.load(f))

# --- Map workspace file numbers to paper table/figure numbers ---
# The paper uses "Table 2.1", "Table 2.2", etc. but opencode saved table_1.csv–table_7.csv
TABLE_MAP = {
    "1": "Table 2.1",  # Characteristics of targeted reform
    "2": "Table 2.2",  # Notations for reforms and gain notions
    "3": "Table 3.1",  # Determinants of bias in subjective gains
    "4": "Table 4.1",  # Share with new beliefs aligned with feedback
    "5": "Table 5.1",  # Effect of self-interest on acceptance
    "6": "Table 5.2",  # Effect of environmental effectiveness on acceptance
    "7": "Table 5.3",  # Effect of progressivity beliefs on acceptance
}
FIGURE_MAP = {
    "1": "Figure 2.1",  # Diagram of treatments and questions
    "2": "Figure 3.1",  # Distribution of objective vs subjective net gains
    "3": "Figure 3.2",  # CDF of objective vs subjective net gains
}

# --- Build ReplicationResults from workspace files ---
# Load the replication script
replicate_py = WORKSPACE / "replicate.py"
code_files = []
if replicate_py.exists():
    code_files.append(GeneratedCode(
        language="python",
        code=replicate_py.read_text(),
        dependencies=["pandas", "numpy", "statsmodels", "matplotlib"],
        execution_order=1,
        description="Full replication script for all tables and figures",
    ))

# Load table CSVs
tables = []
for csv_file in sorted(WORKSPACE.glob("table_*.csv")):
    file_num = csv_file.stem.replace("table_", "")
    paper_table_num = TABLE_MAP.get(file_num, f"Table {file_num}")
    try:
        # Read CSV and convert to dict
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        table_data = {"columns": list(rows[0].keys()) if rows else [], "rows": rows}
        tables.append(GeneratedTable(
            table_number=paper_table_num,
            data=table_data,
            code_reference="replicate.py",
            execution_success=True,
        ))
    except Exception as e:
        tables.append(GeneratedTable(
            table_number=paper_table_num,
            data={},
            code_reference="replicate.py",
            execution_success=False,
            error_message=str(e),
        ))

# Load figures
figures = []
for png_file in sorted(WORKSPACE.glob("figure_*.png")):
    file_num = png_file.stem.replace("figure_", "")
    paper_fig_num = FIGURE_MAP.get(file_num, f"Figure {file_num}")
    figures.append(GeneratedFigure(
        figure_number=paper_fig_num,
        file_path=str(png_file.resolve()),
        code_reference="replicate.py",
        execution_success=True,
    ))

replication_results = ReplicationResults(
    paper_id=PAPER_ID,
    code_files=code_files,
    tables=tables,
    figures=figures,
    execution_log="Opencode freestyle run with gpt-5.2-codex, exit code 0, duration 252.5s",
)

print(f"Built ReplicationResults:")
print(f"  Tables: {[t.table_number for t in tables]}")
print(f"  Figures: {[f.figure_number for f in figures]}")
print(f"  Code files: {len(code_files)}")

# --- Load config ---
config = load_config()
# Use gpt-5.2 as the judge (supports vision for figure comparison)
config.langgraph.default_model = "gpt-5.2"
config.langgraph.default_provider = "openai"

# --- Run Verifier ---
print(f"\n{'='*60}")
print("Running Verifier Agent...")
print(f"{'='*60}")

verifier = VerifierAgent(config)
verification_report = verifier.run(
    paper_path=PDF_PATH,
    replication_results=replication_results,
)

print(f"\nVerification Report:")
print(f"  Overall Grade: {verification_report.overall_grade.value}")
print(f"  Summary:\n{verification_report.summary}")
for v in verification_report.item_verifications:
    print(f"\n  {v.item_id} ({v.item_type}): Grade {v.grade.value}")
    print(f"    Notes: {v.comparison_notes[:200]}")
    if v.numerical_differences:
        print(f"    Differences: {v.numerical_differences}")

# --- Run Explainer ---
print(f"\n{'='*60}")
print("Running Explainer Agent...")
print(f"{'='*60}")

explainer = ExplainerAgent(config)
explanation_report = explainer.run(
    paper_path=PDF_PATH,
    paper_summary=paper_summary,
    replication_results=replication_results,
    verification_report=verification_report,
    replication_package_path=REPLICATION_PACKAGE,
)

print(f"\nExplanation Report:")
print(f"  Overall Assessment:\n{explanation_report.overall_assessment}")
print(f"\n  Recommendations:")
for r in explanation_report.recommendations:
    print(f"    - {r}")
print(f"\n  Discrepancy Analyses ({len(explanation_report.analyses)}):")
for a in explanation_report.analyses:
    print(f"\n  {a.item_id} (Grade {a.grade.value}):")
    print(f"    Discrepancy: {a.description_of_discrepancy[:200]}")
    print(f"    Causes: {a.likely_causes}")
    print(f"    Attribution: {a.fault_attribution} (confidence: {a.confidence})")
if explanation_report.replication_package_comparison:
    print(f"\n  Package Comparison:\n{explanation_report.replication_package_comparison}")

# --- Save reports ---
output_dir = WORKSPACE.parent
verification_path = output_dir / "verification_report.json"
explanation_path = output_dir / "explanation_report.json"

with open(verification_path, "w") as f:
    json.dump(verification_report.model_dump(), f, indent=2, default=str)
print(f"\nSaved verification report: {verification_path}")

with open(explanation_path, "w") as f:
    json.dump(explanation_report.model_dump(), f, indent=2, default=str)
print(f"Saved explanation report: {explanation_path}")
