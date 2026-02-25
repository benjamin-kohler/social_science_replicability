"""Run the judge on an existing workspace to grade replication quality."""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from src.benchmark.artifact_parser import ArtifactParser
from src.benchmark.judge import Judge
from src.models.schemas import PaperSummary

# --- Configuration ---
WORKSPACE = Path("data/benchmark_results/claude-opus-4-6_yellow_vests_carbon_tax_claude-code/workspace")
PDF_PATH = "data/input/yellow_vests_carbon_tax/paper.pdf"
REPLICATION_PACKAGE = "data/input/yellow_vests_carbon_tax/replication_package"
PAPER_ID = "yellow_vests_carbon_tax"
MODEL = "gpt-5-mini"

# --- Load paper summary from workspace ---
summary_path = WORKSPACE / "methodology_summary.json"
with open(summary_path) as f:
    paper_summary = PaperSummary(**json.load(f))

print(f"Loaded summary: {len(paper_summary.tables)} tables, {len(paper_summary.figures)} figures")

# --- Parse workspace artifacts ---
replication_results = ArtifactParser.parse(WORKSPACE, PAPER_ID)
print(f"Parsed workspace:")
print(f"  Tables: {[t.table_number for t in replication_results.tables]}")
print(f"  Figures: {[f.figure_number for f in replication_results.figures]}")
print(f"  Code files: {len(replication_results.code_files)}")

# --- Run Judge ---
judge = Judge(
    provider="openai",
    model=MODEL,
    api_key=os.environ.get("OPENAI_API_KEY", ""),
)

print(f"\n{'='*60}")
print(f"Running Judge ({MODEL})...")
print(f"{'='*60}")

t0 = time.time()
verification_report, explanation_report = judge.run(
    paper_path=PDF_PATH,
    paper_summary=paper_summary,
    replication_results=replication_results,
    replication_package_path=REPLICATION_PACKAGE,
)
elapsed = time.time() - t0

# --- Print results ---
print(f"\nVerification Report:")
print(f"  Overall Grade: {verification_report.overall_grade.value}")
print(f"  Summary:\n{verification_report.summary}")
for v in verification_report.item_verifications:
    print(f"\n  {v.item_id} ({v.item_type}): Grade {v.grade.value}")
    print(f"    Notes: {v.comparison_notes[:200]}")

if explanation_report:
    print(f"\n{'='*60}")
    print(f"Explanation Report ({len(explanation_report.analyses)} discrepancies):")
    print(f"{'='*60}")
    print(f"  Overall: {explanation_report.overall_assessment}")
    for a in explanation_report.analyses:
        print(f"\n  {a.item_id} (Grade {a.grade.value}):")
        print(f"    {a.description_of_discrepancy[:200]}")
        print(f"    Causes: {a.likely_causes}")
        print(f"    Attribution: {a.fault_attribution} ({a.confidence})")

# --- Token usage ---
usage = judge.usage_summary
print(f"\n{'='*60}")
print(f"Token Usage ({MODEL}):")
print(f"{'='*60}")
print(f"  LLM calls:        {usage['num_calls']}")
print(f"  Prompt tokens:     {usage['prompt_tokens']:,}")
print(f"  Completion tokens: {usage['completion_tokens']:,}")
print(f"  Total tokens:      {usage['total_tokens']:,}")
print(f"  Wall time:         {elapsed:.1f}s")

# --- Save reports ---
output_dir = WORKSPACE.parent
with open(output_dir / "verification_report.json", "w") as f:
    json.dump(verification_report.model_dump(), f, indent=2, default=str)
print(f"\nSaved: {output_dir / 'verification_report.json'}")

if explanation_report:
    with open(output_dir / "explanation_report.json", "w") as f:
        json.dump(explanation_report.model_dump(), f, indent=2, default=str)
    print(f"Saved: {output_dir / 'explanation_report.json'}")

# Save usage log
with open(output_dir / "judge_usage.json", "w") as f:
    json.dump({
        "model": MODEL,
        "provider": "openai",
        "wall_time_seconds": round(elapsed, 1),
        **usage,
    }, f, indent=2)
print(f"Saved: {output_dir / 'judge_usage.json'}")
