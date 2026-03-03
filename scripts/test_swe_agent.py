"""Test SWE-agent runner on the postal systems paper."""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

from src.benchmark.config import BenchmarkConfig, ModelSpec, PaperSpec, JudgeConfig
from src.benchmark.runner import BenchmarkRunner
from src.utils.logging_utils import setup_logging

setup_logging()

config = BenchmarkConfig(
    models=[
        ModelSpec(
            provider="openai",
            model_name="gpt-5.2-codex",
            api_key_env="OPENAI_API_KEY",
            approaches=["swe-agent"],
        ),
    ],
    papers=[
        PaperSpec(
            paper_id="postal_systems",
            pdf_path="data/input/postal_systems/paper.pdf",
            data_path="data/input/postal_systems/data",
            replication_package_path="data/input/postal_systems/replication_package",
        ),
    ],
    approaches=["swe-agent"],
    judge=JudgeConfig(provider="openai", model_name="gpt-5-mini"),
    output_dir="data/benchmark_results/swe_agent_test",
    timeout_seconds=3600,
    allow_web_access=False,
)

runner = BenchmarkRunner(config)
print("Starting SWE-agent benchmark run on postal_systems with gpt-5.2-codex...")
results = runner.run()

print(f"\nBenchmark complete: {len(results.runs)} runs")
for run in results.runs:
    print(
        f"  {run.model.model_name:30s} | {run.paper.paper_id:20s} | "
        f"{run.approach:12s} | Grade: {run.evaluation.overall_grade} | "
        f"{run.duration_seconds:.0f}s"
    )
