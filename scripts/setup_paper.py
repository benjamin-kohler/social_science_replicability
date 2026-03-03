#!/usr/bin/env python3
"""Set up a new paper for benchmarking.

Creates the standard directory structure, moves files into place, and
generates a benchmark config YAML.

Usage examples:

  # Minimal — just a paper PDF and data directory:
  python scripts/setup_paper.py my_paper --pdf ~/Downloads/paper.pdf --data ~/Downloads/data/

  # With replication package:
  python scripts/setup_paper.py my_paper --pdf paper.pdf --data data/ --replication-package code/

  # Custom models and approaches:
  python scripts/setup_paper.py my_paper --pdf paper.pdf --data data/ \
      --models gpt-5.2-codex:structured gpt-5.3-codex:codex claude-opus-4-6:claude-code \
      --judge-model gpt-5.2-mini

  # Dry run — show what would happen without doing it:
  python scripts/setup_paper.py my_paper --pdf paper.pdf --data data/ --dry-run
"""

import argparse
import shutil
import sys
from pathlib import Path

# Project root is the parent of scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "input"
CONFIG_DIR = PROJECT_ROOT / "config"
RESULTS_DIR = PROJECT_ROOT / "data" / "benchmark_results"

# Default model presets (provider:model_name:api_key_env -> approaches)
DEFAULT_MODELS = [
    ("openai", "gpt-5.2-codex", "OPENAI_API_KEY", ["freestyle", "structured"]),
    ("anthropic", "claude-opus-4-6", "ANTHROPIC_API_KEY", ["claude-code"]),
    ("openai", "gpt-5.3-codex", "OPENAI_API_KEY", ["codex"]),
]


def parse_model_spec(spec: str) -> tuple[str, str, str, list[str]]:
    """Parse a model spec string like 'gpt-5.2-codex:freestyle,structured'.

    Returns (provider, model_name, api_key_env, approaches).
    """
    parts = spec.split(":", 1)
    model_name = parts[0]
    approaches = parts[1].split(",") if len(parts) > 1 else None

    # Infer provider from model name
    if any(model_name.startswith(p) for p in ("claude",)):
        provider = "anthropic"
        api_key_env = "ANTHROPIC_API_KEY"
    else:
        provider = "openai"
        api_key_env = "OPENAI_API_KEY"

    # Infer default approach from model name if not specified
    if approaches is None:
        if "codex" in model_name and model_name.startswith("gpt-5.3"):
            approaches = ["codex"]
        elif "claude" in model_name:
            approaches = ["claude-code"]
        else:
            approaches = ["freestyle", "structured"]

    return provider, model_name, api_key_env, approaches


def build_config_yaml(
    paper_id: str,
    models: list[tuple[str, str, str, list[str]]],
    judge_provider: str,
    judge_model: str,
    judge_vision: bool,
    extractor_model: str,
    extractor_vision: bool,
    timeout: int,
    opencode_binary: str,
    claude_code_binary: str,
    codex_binary: str,
    allow_web_access: bool,
) -> str:
    """Build benchmark config YAML content."""
    lines = [
        f"## Benchmark Configuration — {paper_id}",
        "",
        "models:",
    ]

    all_approaches = set()
    for provider, model_name, api_key_env, approaches in models:
        lines.append(f"  - provider: {provider}")
        lines.append(f"    model_name: {model_name}")
        lines.append(f"    api_key_env: {api_key_env}")
        lines.append(f"    approaches:")
        for a in approaches:
            lines.append(f"      - {a}")
            all_approaches.add(a)
        lines.append("")

    data_dir = f"data/input/{paper_id}"
    lines.extend([
        "papers:",
        f"  - paper_id: {paper_id}",
        f"    pdf_path: {data_dir}/paper.pdf",
        f"    data_path: {data_dir}/data",
        f"    replication_package_path: {data_dir}/replication_package",
        "",
        "approaches:",
    ])
    for a in sorted(all_approaches):
        lines.append(f"  - {a}")

    lines.extend([
        "",
        "judge:",
        f"  provider: {judge_provider}",
        f"  model_name: {judge_model}",
        f"  use_vision: {'true' if judge_vision else 'false'}",
        "",
        "extractor:",
        f"  model: {extractor_model}",
        f"  use_vision: {'true' if extractor_vision else 'false'}",
        "",
        f"output_dir: data/benchmark_results/{paper_id}",
        f"opencode_binary: {opencode_binary}",
        f"claude_code_binary: {claude_code_binary}",
        f"codex_binary: {codex_binary}",
        f"timeout_seconds: {timeout}",
        f"allow_web_access: {'true' if allow_web_access else 'false'}",
        "",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Set up a new paper for benchmarking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s my_paper --pdf paper.pdf --data data/
  %(prog)s my_paper --pdf paper.pdf --data data/ --replication-package code/
  %(prog)s my_paper --pdf paper.pdf --data data/ --models gpt-5.2-codex:structured
  %(prog)s my_paper --pdf paper.pdf --data data/ --dry-run
""",
    )
    parser.add_argument(
        "paper_id",
        help="Unique identifier for the paper (e.g., 'yellow_vests_carbon_tax')",
    )
    parser.add_argument(
        "--pdf", required=True,
        help="Path to the paper PDF. Will be copied to data/input/{paper_id}/paper.pdf",
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to data file or directory. Will be copied to data/input/{paper_id}/data/",
    )
    parser.add_argument(
        "--replication-package", "--repl", default=None,
        help="Path to original replication code. Copied to data/input/{paper_id}/replication_package/",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Model specs as 'model_name:approach1,approach2'. "
             "Defaults to the standard 4-way setup. "
             "Examples: gpt-5.2-codex:structured claude-opus-4-6:claude-code",
    )
    parser.add_argument(
        "--judge-model", default="gpt-5.2-mini",
        help="Model for the judge (default: gpt-5.2-mini)",
    )
    parser.add_argument(
        "--judge-no-vision", action="store_true",
        help="Disable vision for the judge (default: vision enabled)",
    )
    parser.add_argument(
        "--extractor-model", default="gpt-5.2",
        help="Model for methodology extraction (default: gpt-5.2)",
    )
    parser.add_argument(
        "--extractor-no-vision", action="store_true",
        help="Disable vision for the extractor (default: vision enabled)",
    )
    parser.add_argument(
        "--timeout", type=int, default=3600,
        help="Timeout per run in seconds (default: 3600)",
    )
    parser.add_argument(
        "--opencode-binary", default="/Users/bkohler/.opencode/bin/opencode",
        help="Path to opencode binary",
    )
    parser.add_argument(
        "--claude-code-binary", default="claude",
        help="Path to claude CLI binary",
    )
    parser.add_argument(
        "--codex-binary", default="codex",
        help="Path to codex CLI binary",
    )
    parser.add_argument(
        "--allow-web-access", action="store_true",
        help="Allow web search during replication (default: disabled)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing directories and config",
    )
    parser.add_argument(
        "--config-name", default=None,
        help="Config filename (default: {paper_id}_benchmark.yaml)",
    )

    args = parser.parse_args()

    paper_id = args.paper_id
    paper_dir = INPUT_DIR / paper_id
    config_name = args.config_name or f"{paper_id}_benchmark.yaml"
    config_path = CONFIG_DIR / config_name

    pdf_src = Path(args.pdf).resolve()
    data_src = Path(args.data).resolve()
    repl_src = Path(args.replication_package).resolve() if args.replication_package else None

    # --- Validate inputs ---
    errors = []
    if not pdf_src.exists():
        errors.append(f"PDF not found: {pdf_src}")
    if not data_src.exists():
        errors.append(f"Data path not found: {data_src}")
    if repl_src and not repl_src.exists():
        errors.append(f"Replication package not found: {repl_src}")
    if paper_dir.exists() and not args.force:
        errors.append(f"Paper directory already exists: {paper_dir} (use --force to overwrite)")
    if config_path.exists() and not args.force:
        errors.append(f"Config already exists: {config_path} (use --force to overwrite)")
    if errors:
        for e in errors:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Parse models ---
    if args.models:
        models = [parse_model_spec(m) for m in args.models]
    else:
        models = DEFAULT_MODELS

    # Infer judge provider
    judge_model = args.judge_model
    if any(judge_model.startswith(p) for p in ("claude",)):
        judge_provider = "anthropic"
    else:
        judge_provider = "openai"

    # --- Plan actions ---
    pdf_dest = paper_dir / "paper.pdf"
    data_dest = paper_dir / "data"
    repl_dest = paper_dir / "replication_package"

    actions = []
    actions.append(f"Create directory: {paper_dir}")
    actions.append(f"Copy PDF: {pdf_src} -> {pdf_dest}")
    if data_src.is_dir():
        actions.append(f"Copy data directory: {data_src} -> {data_dest}")
    else:
        actions.append(f"Copy data file: {data_src} -> {data_dest}/{data_src.name}")
    if repl_src:
        if repl_src.is_dir():
            actions.append(f"Copy replication package: {repl_src} -> {repl_dest}")
        else:
            actions.append(f"Copy replication file: {repl_src} -> {repl_dest}/{repl_src.name}")
    else:
        actions.append(f"Create empty directory: {repl_dest}")
    actions.append(f"Write config: {config_path}")

    config_yaml = build_config_yaml(
        paper_id=paper_id,
        models=models,
        judge_provider=judge_provider,
        judge_model=judge_model,
        judge_vision=not args.judge_no_vision,
        extractor_model=args.extractor_model,
        extractor_vision=not args.extractor_no_vision,
        timeout=args.timeout,
        opencode_binary=args.opencode_binary,
        claude_code_binary=args.claude_code_binary,
        codex_binary=args.codex_binary,
        allow_web_access=args.allow_web_access,
    )

    # --- Show plan ---
    print(f"\n  Paper ID:  {paper_id}")
    print(f"  Input dir: {paper_dir}")
    print(f"  Config:    {config_path}")
    print(f"  Models:    {len(models)}")
    print()

    for a in actions:
        print(f"  {'[dry-run] ' if args.dry_run else ''}{a}")

    print(f"\n  Config preview:\n")
    for line in config_yaml.splitlines():
        print(f"    {line}")
    print()

    if args.dry_run:
        print("Dry run complete. No changes made.")
        return

    # --- Execute ---
    # Create directories
    paper_dir.mkdir(parents=True, exist_ok=True)

    # Copy PDF
    shutil.copy2(pdf_src, pdf_dest)

    # Copy data
    if data_src.is_dir():
        if data_dest.exists():
            shutil.rmtree(data_dest)
        shutil.copytree(data_src, data_dest)
    else:
        data_dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(data_src, data_dest / data_src.name)

    # Copy or create replication package
    if repl_src:
        if repl_src.is_dir():
            if repl_dest.exists():
                shutil.rmtree(repl_dest)
            shutil.copytree(repl_src, repl_dest)
        else:
            repl_dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repl_src, repl_dest / repl_src.name)
    else:
        repl_dest.mkdir(parents=True, exist_ok=True)

    # Write config
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config_yaml)

    # Create output directory
    output_dir = RESULTS_DIR / paper_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Done! To run the benchmark:\n")
    print(f"  benchmark --config {config_path.relative_to(PROJECT_ROOT)}")
    print()


if __name__ == "__main__":
    main()
