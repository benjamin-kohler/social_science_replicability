"""Launch Codex CLI (gpt-5.3-codex) on yellow_vests_carbon_tax using the vision extraction summary."""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.benchmark.codex_runner import CodexRunner
from src.benchmark.config import ModelSpec, PaperSpec
from src.models.schemas import PaperSummary

# Load the vision extraction summary
summary_path = Path("data/benchmark_results/extraction_test_vision/summary.json")
with open(summary_path) as f:
    summary_data = json.load(f)
paper_summary = PaperSummary(**summary_data)

print(f"Loaded summary: {len(paper_summary.tables)} tables, {len(paper_summary.figures)} figures")

# Configure the run
model = ModelSpec(
    provider="openai",
    model_name="gpt-5.3-codex",
    api_key_env="OPENAI_API_KEY",
    approaches=["codex"],
)

paper = PaperSpec(
    paper_id="yellow_vests_carbon_tax",
    pdf_path="data/input/yellow_vests_carbon_tax/paper.pdf",
    data_path="data/input/yellow_vests_carbon_tax/data",
    replication_package_path="data/input/yellow_vests_carbon_tax/replication_package",
)

workspace = Path("data/benchmark_results/gpt-5.3-codex_yellow_vests_carbon_tax_codex/workspace")

runner = CodexRunner(
    codex_binary="codex",
    timeout=36000,       # 10 hours
    allow_web_access=False,
)

print(f"Starting Codex run: model={model.model_name}, timeout=36000s")
print(f"Workspace: {workspace.resolve()}")
print("This will take a while...")

artifacts = runner.run(model, paper, paper_summary, workspace)

print(f"\nDone! exit_code={artifacts.exit_code}, duration={artifacts.duration_seconds:.1f}s")
print(f"Workspace: {artifacts.workspace_dir}")
print(f"Log: {workspace / 'run_log.txt'}")
