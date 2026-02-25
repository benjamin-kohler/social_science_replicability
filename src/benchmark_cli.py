"""CLI entry point for running benchmarks."""

import argparse
import sys

import yaml
from dotenv import load_dotenv

from .benchmark.config import BenchmarkConfig
from .benchmark.runner import BenchmarkRunner
from .utils.logging_utils import setup_logging, get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark LLM models on paper replication tasks."
    )
    parser.add_argument(
        "--config", "-c",
        default="config/benchmark_config.yaml",
        help="Path to benchmark config YAML (default: config/benchmark_config.yaml)",
    )
    parser.add_argument(
        "--approaches",
        nargs="*",
        choices=["freestyle", "structured", "claude-code", "codex"],
        help="Override approaches to run (default: from config)",
    )
    parser.add_argument(
        "--papers",
        nargs="*",
        help="Filter to specific paper IDs (default: all from config)",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        help="Filter to specific model names (default: all from config)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        help="Override output directory (default: from config)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Override timeout in seconds (default: from config)",
    )
    parser.add_argument(
        "--allow-web-access",
        action="store_true",
        default=None,
        help="Allow models to use web search (default: blocked for information isolation)",
    )

    args = parser.parse_args()

    # Load .env file for API keys
    load_dotenv()

    setup_logging()

    # Load config
    try:
        with open(args.config) as f:
            data = yaml.safe_load(f)
        config = BenchmarkConfig(**data)
    except FileNotFoundError:
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

    # Apply CLI overrides
    if args.approaches:
        config.approaches = args.approaches
    if args.papers:
        config.papers = [p for p in config.papers if p.paper_id in args.papers]
    if args.models:
        config.models = [m for m in config.models if m.model_name in args.models]
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.timeout:
        config.timeout_seconds = args.timeout
    if args.allow_web_access is not None:
        config.allow_web_access = args.allow_web_access

    if not config.models:
        logger.error("No models selected. Check --models filter.")
        sys.exit(1)
    if not config.papers:
        logger.error("No papers selected. Check --papers filter.")
        sys.exit(1)

    # Run benchmark
    runner = BenchmarkRunner(config)
    results = runner.run()

    # Print summary
    print(f"\nBenchmark complete: {len(results.runs)} runs")
    for run in results.runs:
        print(
            f"  {run.model.model_name:30s} | {run.paper.paper_id:20s} | "
            f"{run.approach:12s} | Grade: {run.evaluation.overall_grade} | "
            f"{run.duration_seconds:.0f}s"
        )


if __name__ == "__main__":
    main()
