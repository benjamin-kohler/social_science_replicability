"""Prepare explainer_workspaces/ for the error analysis pipeline.

For each paper × agent combination found in results, creates:

  {output_dir}/{paper_id}/{agent_label}/
    code/
      agent_code/              <- .py files from the agent's workspace
      original_code/           <- full replication package
    error_source/
      paper_vs_original_code/
        paper.pdf
        original_code_files/   <- code files matching detected language
      paper_vs_summary/
        paper.pdf
        methodology_summary.json
      summary_vs_agent/
        methodology_summary.json
        agent_code/

Usage
-----
    python 00_prep_setup.py \\
        --papers-dir  /data/.../papers \\
        --results-dir /data/.../results \\
        [--output-dir ./explainer_workspaces] \\
        [--agents claude-code,codex]  \\
        [--papers 10.1257_aer.20190565,10.1086_714931] \\
        [--overwrite]
"""

import argparse
import shutil
import sys
from pathlib import Path

from language_info import detect_language, get_info

KNOWN_APPROACHES = {"claude-code", "codex", "swe-agent", "opencode"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_run_dir(dirname: str, paper_id: str) -> tuple[str, str] | None:
    """Parse {model}_{paper_id}_{approach} from a result directory name.

    Returns (model, approach) or None if not parseable.
    """
    idx = dirname.find(f"_{paper_id}_")
    if idx < 0:
        return None
    model = dirname[:idx]
    approach = dirname[idx + len(f"_{paper_id}_"):]
    if approach not in KNOWN_APPROACHES:
        return None
    return model, approach


def _copy_python_files(src_workspace: Path, dest_dir: Path, overwrite: bool) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for py_file in src_workspace.glob("*.py"):
        dest = dest_dir / py_file.name
        if dest.exists() and not overwrite:
            continue
        shutil.copy2(py_file, dest)
        count += 1
    return count


def _copy_tree(src: Path, dest: Path, overwrite: bool) -> None:
    if dest.exists() and not overwrite:
        return
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)


def _copy_file(src: Path, dest: Path, overwrite: bool) -> bool:
    if dest.exists() and not overwrite:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return True


def _copy_code_files(replication_pkg: Path, dest_dir: Path, language: str,
                     overwrite: bool) -> int:
    """Copy original code files matching detected language into dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    info = get_info(language)
    count = 0
    for pattern in info["file_patterns"]:
        for f in replication_pkg.rglob(pattern):
            rel = f.relative_to(replication_pkg)
            target = dest_dir / rel
            if target.exists() and not overwrite:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
            count += 1
    return count


def _find_paper_pdf(papers_dir: Path, results_dir: Path, paper_id: str,
                    run_folders: list[Path]) -> Path | None:
    """Find paper.pdf with fallback chain:
    1. papers_dir/{paper_id}/paper.pdf
    2. Any explainer_workspace/paper.pdf in result run folders
    """
    # Primary: papers dir
    pdf = papers_dir / paper_id / "paper.pdf"
    if pdf.exists():
        return pdf
    # Fallback: any explainer_workspace
    for rf in run_folders:
        pdf = rf / "explainer_workspace" / "paper.pdf"
        if pdf.exists():
            return pdf
    return None


def _setup_error_source(
    agent_out: Path,
    paper_pdf: Path,
    agent_workspace: Path,
    replication_pkg: Path,
    language: str,
    overwrite: bool,
) -> None:
    """Populate {agent_out}/error_source/ with three document-pair subfolders."""
    methodology_json = agent_workspace / "methodology_summary.json"
    if not methodology_json.exists():
        # Try explainer_workspace
        ew = agent_workspace.parent / "explainer_workspace"
        if (ew / "methodology_summary.json").exists():
            methodology_json = ew / "methodology_summary.json"

    agent_code_src = agent_out / "code" / "agent_code"
    es = agent_out / "error_source"

    missing = []
    if not paper_pdf.exists():
        missing.append(str(paper_pdf))
    if not methodology_json.exists():
        missing.append("methodology_summary.json")
    if missing:
        print(f"    [error_source] SKIP — missing: {', '.join(missing)}")
        return

    # 1. paper_vs_original_code/
    pvo = es / "paper_vs_original_code"
    _copy_file(paper_pdf, pvo / "paper.pdf", overwrite)
    n_code = _copy_code_files(replication_pkg, pvo / "original_code_files", language, overwrite)
    print(f"    [error_source] paper_vs_original_code: {n_code} {language} files")

    # 2. paper_vs_summary/
    pvs = es / "paper_vs_summary"
    _copy_file(paper_pdf, pvs / "paper.pdf", overwrite)
    _copy_file(methodology_json, pvs / "methodology_summary.json", overwrite)

    # 3. summary_vs_agent/
    sva = es / "summary_vs_agent"
    _copy_file(methodology_json, sva / "methodology_summary.json", overwrite)
    sva_agent = sva / "agent_code"
    sva_agent.mkdir(parents=True, exist_ok=True)
    n = 0
    for py_file in agent_code_src.glob("*.py"):
        if _copy_file(py_file, sva_agent / py_file.name, overwrite):
            n += 1
    print(f"    [error_source] created — {n} .py files in summary_vs_agent/agent_code/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Prepare explainer_workspaces/ for the evaluation pipeline."
    )
    parser.add_argument("--papers-dir", required=True,
        help="Root folder containing one subfolder per paper.")
    parser.add_argument("--results-dir", required=True,
        help="Root folder containing one subfolder per paper with run results.")
    parser.add_argument("--output-dir", default=str(here / "explainer_workspaces"),
        help="Root folder where workspaces will be created.")
    parser.add_argument("--agents", default="",
        help="Comma-separated list of approaches to include (default: all).")
    parser.add_argument("--papers", default="",
        help="Comma-separated list of paper IDs to include (default: all).")
    parser.add_argument("--overwrite", action="store_true",
        help="Re-copy files even if destination already exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    papers_dir = Path(args.papers_dir).expanduser().resolve()
    results_dir = Path(args.results_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    agent_filter = set(args.agents.split(",")) if args.agents else None
    paper_filter = set(args.papers.split(",")) if args.papers else None

    for p, label in [(papers_dir, "--papers-dir"), (results_dir, "--results-dir")]:
        if not p.is_dir():
            sys.exit(f"ERROR: {label} does not exist: {p}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Papers:   {papers_dir}")
    print(f"Results:  {results_dir}")
    print(f"Output:   {output_dir}")
    if agent_filter:
        print(f"Agents:   {sorted(agent_filter)}")
    print()

    paper_folders = sorted(d for d in papers_dir.iterdir() if d.is_dir())
    if not paper_folders:
        sys.exit("ERROR: no paper folders found in --papers-dir")

    for paper_dir in paper_folders:
        paper_id = paper_dir.name
        if paper_filter and paper_id not in paper_filter:
            continue

        replication_pkg = paper_dir / "replication_package"
        if not replication_pkg.is_dir():
            print(f"[{paper_id}] SKIP — no replication_package/")
            continue

        # Detect original code language
        language = detect_language(replication_pkg)

        # Find result run folders for this paper
        results_paper = results_dir / paper_id
        if not results_paper.is_dir():
            print(f"[{paper_id}] SKIP — no results folder")
            continue

        run_folders = [d for d in results_paper.iterdir()
                       if d.is_dir() and d.name not in ("judge_results", "summaries")]

        # Parse run dirs into agent entries
        agent_runs: dict[str, Path] = {}  # label -> run folder
        for rf in run_folders:
            parsed = _parse_run_dir(rf.name, paper_id)
            if parsed is None:
                continue
            model, approach = parsed
            if agent_filter and approach not in agent_filter:
                continue
            label = f"{model}_{approach}"
            if label not in agent_runs or rf.stat().st_mtime > agent_runs[label].stat().st_mtime:
                agent_runs[label] = rf

        if not agent_runs:
            print(f"[{paper_id}] SKIP — no matching run folders")
            continue

        print(f"[{paper_id}]  language={language}  agents: {sorted(agent_runs)}")

        # Find paper.pdf
        paper_pdf = _find_paper_pdf(papers_dir, results_dir, paper_id, list(agent_runs.values()))
        if not paper_pdf:
            print(f"  WARNING — paper.pdf not found; error_source will be skipped")

        for agent_label, run_folder in sorted(agent_runs.items()):
            workspace = run_folder / "workspace"
            if not workspace.is_dir():
                print(f"  [{agent_label}] SKIP — no workspace/")
                continue

            agent_out = output_dir / paper_id / agent_label

            # --- code/agent_code/ ---
            agent_code_dir = agent_out / "code" / "agent_code"
            n = _copy_python_files(workspace, agent_code_dir, args.overwrite)
            print(f"  [{agent_label}] code/agent_code/ <- {n} .py files")

            # --- code/original_code/ (symlink to replication_package) ---
            original_code_dest = agent_out / "code" / "original_code"
            if original_code_dest.is_symlink() or original_code_dest.exists():
                if args.overwrite:
                    if original_code_dest.is_symlink():
                        original_code_dest.unlink()
                    else:
                        shutil.rmtree(original_code_dest)
                else:
                    print(f"  [{agent_label}] code/original_code/ <- exists (skip)")
                    # still do error_source setup below
            if not original_code_dest.exists():
                original_code_dest.parent.mkdir(parents=True, exist_ok=True)
                original_code_dest.symlink_to(replication_pkg.resolve())
                print(f"  [{agent_label}] code/original_code/ -> {replication_pkg} (symlink)")

            # --- error_source/ ---
            if paper_pdf:
                _setup_error_source(
                    agent_out, paper_pdf, workspace, replication_pkg,
                    language, args.overwrite,
                )

    print(f"\nDone. Workspace root: {output_dir}")


if __name__ == "__main__":
    main()
