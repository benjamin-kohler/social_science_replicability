"""Launch Claude Code (opus 4.6) on yellow_vests_carbon_tax using the vision extraction summary."""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.benchmark.claude_code_runner import ClaudeCodeRunner
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
    provider="anthropic",
    model_name="claude-opus-4-6",
    api_key_env="ANTHROPIC_API_KEY",
    approaches=["claude-code"],
)

paper = PaperSpec(
    paper_id="yellow_vests_carbon_tax",
    pdf_path="data/input/yellow_vests_carbon_tax/paper.pdf",
    data_path="data/input/yellow_vests_carbon_tax/data",
    replication_package_path="data/input/yellow_vests_carbon_tax/replication_package",
)

workspace = Path("data/benchmark_results/claude-opus-4-6_yellow_vests_carbon_tax_claude-code/workspace")

runner = ClaudeCodeRunner(
    claude_binary="claude",
    timeout=36000,       # 10 hours
    max_turns=1000,      # 1000 agentic turns
    allow_web_access=False,
)

print(f"Starting Claude Code run: model={model.model_name}, timeout=36000s, max_turns=1000")
print(f"Workspace: {workspace.resolve()}")
print("This will take a while...")

artifacts = runner.run(model, paper, paper_summary, workspace)

print(f"\nDone! exit_code={artifacts.exit_code}, duration={artifacts.duration_seconds:.1f}s")
print(f"Workspace: {artifacts.workspace_dir}")
print(f"Log: {workspace / 'run_log.txt'}")
