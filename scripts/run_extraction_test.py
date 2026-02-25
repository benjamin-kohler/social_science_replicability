"""Quick script to run extraction on a single paper and save results."""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.config import load_config
from src.agents.extractor import ExtractorAgent
from src.benchmark.task_prompt import build_task_prompt


def main():
    paper_id = "yellow_vests_carbon_tax"
    pdf_path = "data/input/yellow_vests_carbon_tax/paper.pdf"

    # Toggle vision mode via command-line flag
    use_vision = "--vision" in sys.argv

    if use_vision:
        output_dir = Path("data/benchmark_results/extraction_test_vision")
        print("Running in VISION mode (PDF pages as images)")
    else:
        output_dir = Path("data/benchmark_results/extraction_test_v3")
        print("Running in TEXT mode")

    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config()

    extractor = ExtractorAgent(
        config=config,
        model="gpt-5.2",
        max_tokens=128000,
        use_vision=use_vision,
    )
    summary, usage = extractor.run(paper_path=pdf_path, paper_id=paper_id)

    # Save summary JSON
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary.model_dump(), indent=2, default=str)
    )
    print(f"\nSaved summary to {summary_path}")
    print(f"  Tables: {len(summary.tables)}")
    for t in summary.tables:
        print(f"    - {t.table_number}: {t.caption[:80]}")
    print(f"  Figures: {len(summary.figures)}")
    for f in summary.figures:
        print(f"    - {f.figure_number}: {f.caption[:80]}")

    # Build and save TASK.md
    task_md = build_task_prompt(summary, "data/")
    task_path = output_dir / "TASK.md"
    task_path.write_text(task_md)
    print(f"Saved TASK.md to {task_path}")

    # Save and print usage
    usage_path = output_dir / "usage.json"
    usage_path.write_text(json.dumps(usage.summary_dict(), indent=2))
    print(f"\nToken usage:")
    for call in usage.calls:
        print(f"  {call.step}: {call.input_tokens:,} in + {call.output_tokens:,} out "
              f"= {call.total_tokens:,} ({call.duration_seconds:.1f}s) [{call.model}]")
    print(f"  TOTAL: {usage.total_input_tokens:,} in + {usage.total_output_tokens:,} out "
          f"= {usage.total_tokens:,} ({usage.total_duration:.1f}s)")


if __name__ == "__main__":
    main()
