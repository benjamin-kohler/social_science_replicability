"""Main entry point for the Social Science Replicability System.

Usage:
    python -m src.main --paper data/input/paper.pdf --data data/input/dataset.csv
    python -m src.main --paper paper.pdf --data data.csv --model-provider anthropic --model-name claude-3-opus-20240229

For more options:
    python -m src.main --help
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from .orchestrator import ReplicationOrchestrator
from .models.config import load_config
from .utils.logging_utils import setup_logging, get_logger

logger = get_logger(__name__)


def run_replication(
    paper_path: str,
    data_path: str,
    replication_package_path: Optional[str] = None,
    output_dir: str = "reports",
    model_provider: str = "openai",
    model_name: str = "gpt-5.3-codex",
    config_path: Optional[str] = None,
    extraction_only: bool = False,
    log_level: str = "INFO",
) -> dict:
    """Run the full replication pipeline.

    Args:
        paper_path: Path to the PDF paper.
        data_path: Path to the data files.
        replication_package_path: Optional path to original replication package.
        output_dir: Directory for output reports.
        model_provider: LLM provider ("openai" or "anthropic").
        model_name: Specific model to use.
        config_path: Optional path to configuration file.
        extraction_only: Only run the extraction step.
        log_level: Logging level.

    Returns:
        Dictionary containing the run report.
    """
    # Setup logging
    setup_logging(level=log_level)

    logger.info("=" * 60)
    logger.info("Social Science Replicability System")
    logger.info("=" * 60)

    # Load configuration
    config = load_config(config_path)
    config.langgraph.default_provider = model_provider
    config.langgraph.default_model = model_name

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize orchestrator
    orchestrator = ReplicationOrchestrator(
        config=config,
        model_provider=model_provider,
        model_name=model_name,
    )

    # Run pipeline
    if extraction_only:
        logger.info("Running extraction only...")
        final_state = orchestrator.run_extraction_only(paper_path=paper_path)
    else:
        final_state = orchestrator.run(
            paper_path=paper_path,
            data_path=data_path,
            replication_package_path=replication_package_path,
            output_dir=str(output_path / "artifacts"),
        )

    # Generate report
    report = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "model_provider": model_provider,
            "model_name": model_name,
            "paper_path": paper_path,
            "data_path": data_path,
            "replication_package_path": replication_package_path,
        },
        "status": final_state.current_step,
        "results": {
            "paper_summary": (
                final_state.paper_summary.model_dump()
                if final_state.paper_summary
                else None
            ),
            "verification_report": (
                final_state.verification_report.model_dump()
                if final_state.verification_report
                else None
            ),
            "explanation_report": (
                final_state.explanation_report.model_dump()
                if final_state.explanation_report
                else None
            ),
        },
        "errors": final_state.errors,
        "warnings": final_state.warnings,
    }

    # Save report
    report_filename = f"replication_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path = output_path / report_filename

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Report saved to: {report_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("REPLICATION SUMMARY")
    print("=" * 60)

    if final_state.verification_report:
        print(f"Overall Grade: {final_state.verification_report.overall_grade.value}")
        print(f"Items Verified: {len(final_state.verification_report.item_verifications)}")
        print(f"\nSummary:\n{final_state.verification_report.summary}")

    if final_state.errors:
        print(f"\nErrors ({len(final_state.errors)}):")
        for error in final_state.errors:
            print(f"  - {error}")

    print(f"\nFull report: {report_path}")
    print("=" * 60)

    return report


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Social Science Replicability System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with OpenAI
  python -m src.main --paper paper.pdf --data dataset.csv

  # With Anthropic Claude
  python -m src.main --paper paper.pdf --data dataset.csv \\
      --model-provider anthropic --model-name claude-3-opus-20240229

  # Extraction only (no replication)
  python -m src.main --paper paper.pdf --extraction-only

  # With replication package for explanation
  python -m src.main --paper paper.pdf --data dataset.csv \\
      --replication-package original_code/
        """,
    )

    # Required arguments
    parser.add_argument(
        "--paper",
        required=True,
        help="Path to the PDF paper to replicate",
    )

    # Optional arguments
    parser.add_argument(
        "--data",
        default="",
        help="Path to data files (required unless --extraction-only)",
    )
    parser.add_argument(
        "--replication-package",
        help="Path to original replication package",
    )
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Output directory for reports (default: reports)",
    )
    parser.add_argument(
        "--model-provider",
        default="openai",
        choices=["openai", "anthropic"],
        help="LLM provider (default: openai)",
    )
    parser.add_argument(
        "--model-name",
        default="gpt-5.3-codex",
        help="Model name (default: gpt-5.3-codex)",
    )
    parser.add_argument(
        "--config",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--extraction-only",
        action="store_true",
        help="Only extract methodology, don't replicate",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.extraction_only and not args.data:
        parser.error("--data is required unless --extraction-only is specified")

    # Run
    run_replication(
        paper_path=args.paper,
        data_path=args.data,
        replication_package_path=args.replication_package,
        output_dir=args.output_dir,
        model_provider=args.model_provider,
        model_name=args.model_name,
        config_path=args.config,
        extraction_only=args.extraction_only,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
