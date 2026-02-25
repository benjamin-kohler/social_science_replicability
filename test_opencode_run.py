"""Run opencode freestyle with gpt-5.2-codex on yellow_vests, save to benchmark_results."""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.models.schemas import PaperSummary
from src.benchmark.opencode_runner import OpencodeRunner
from src.benchmark.config import ModelSpec, PaperSpec

OPENCODE_BINARY = str(Path.home() / ".opencode" / "bin" / "opencode")
SUMMARY_PATH = Path("data/benchmark_results/summaries/yellow_vests_carbon_tax_summary.json")
DATA_PATH = "data/input/yellow_vests_carbon_tax/data"
PDF_PATH = "data/input/yellow_vests_carbon_tax/paper.pdf"

with open(SUMMARY_PATH) as f:
    paper_summary = PaperSummary(**json.load(f))

model = ModelSpec(
    provider="openai",
    model_name="gpt-5.2-codex",
    api_key_env="OPENAI_API_KEY",
    approaches=["freestyle"],
)
paper = PaperSpec(
    paper_id="yellow_vests_carbon_tax",
    pdf_path=PDF_PATH,
    data_path=DATA_PATH,
)

workspace = Path("data/benchmark_results/gpt-5.2-codex_yellow_vests_carbon_tax_freestyle/workspace")
if workspace.exists():
    shutil.rmtree(workspace)

runner = OpencodeRunner(opencode_binary=OPENCODE_BINARY, timeout=1800)
print(f"Running opencode freestyle: openai/gpt-5.2-codex on yellow_vests_carbon_tax")
print(f"  Workspace: {workspace.resolve()}")
print(f"  Timeout: {runner.timeout}s", flush=True)

artifacts = runner.run(model, paper, paper_summary, workspace)

print(f"\n{'='*60}")
print(f"Exit code: {artifacts.exit_code}")
print(f"Duration: {artifacts.duration_seconds:.1f}s")
print(f"\n--- STDOUT (last 3000 chars) ---")
print(artifacts.stdout[-3000:] if artifacts.stdout else "(empty)")
print(f"\n--- STDERR (last 1000 chars) ---")
print(artifacts.stderr[-1000:] if artifacts.stderr else "(empty)")

print(f"\n--- Workspace contents ---")
for p in sorted(workspace.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(workspace)}  ({p.stat().st_size:,} bytes)")

csv_files = sorted(workspace.glob("table_*.csv"))
png_files = sorted(workspace.glob("figure_*.png"))
py_files = sorted(workspace.glob("*.py"))
print(f"\n--- Key outputs ---")
print(f"  Table CSVs: {[f.name for f in csv_files]}")
print(f"  Figure PNGs: {[f.name for f in png_files]}")
print(f"  Python scripts: {[f.name for f in py_files]}")

if csv_files or png_files:
    print(f"\nSUCCESS: Generated {len(csv_files)} tables and {len(png_files)} figures!")
    # Print table contents
    for f in csv_files:
        print(f"\n=== {f.name} ===")
        print(f.read_text())
else:
    print(f"\nFAILED: No output files generated.")
