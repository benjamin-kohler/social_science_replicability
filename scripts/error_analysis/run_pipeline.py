#!/usr/bin/env python3
"""Orchestrator for the error analysis pipeline.

Discovers papers and agents from results, runs steps 00-04.

Usage:
    python run_pipeline.py --config config/error_analysis_i4rep.yaml
    python run_pipeline.py --papers-dir ... --results-dir ... --output-dir ...
    python run_pipeline.py --config ea.yaml --from-step 2 --papers 10.1257_aer.20190565
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from language_info import detect_language, get_info

KNOWN_APPROACHES = {"claude-code", "codex", "swe-agent", "opencode"}
HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    papers_dir: Path
    results_dir: Path
    output_dir: Path
    papers: list[str] = field(default_factory=list)
    agents: list[str] = field(default_factory=list)
    runner: str = "claude-code"
    model: str | None = None
    api_key: str | None = None
    from_step: int = 0
    to_step: int = 4
    rerun: bool = False
    rerun_checks: list[int] = field(default_factory=list)
    max_turns: int = 40
    timeout: int = 600
    parallel: int = 1


def load_config(config_path: Path | None, cli: argparse.Namespace) -> PipelineConfig:
    """Load YAML config, then override with any explicit CLI args."""
    cfg = {}
    if config_path and config_path.exists():
        cfg = yaml.safe_load(config_path.read_text()) or {}

    def _get(key, default=None):
        cli_val = getattr(cli, key, None)
        if cli_val is not None:
            return cli_val
        return cfg.get(key, default)

    papers_str = _get("papers", [])
    if isinstance(papers_str, str):
        papers_str = [p.strip() for p in papers_str.split(",") if p.strip()]

    agents_str = _get("agents", [])
    if isinstance(agents_str, str):
        agents_str = [a.strip() for a in agents_str.split(",") if a.strip()]

    return PipelineConfig(
        papers_dir=Path(_get("papers_dir", ".")).expanduser().resolve(),
        results_dir=Path(_get("results_dir", ".")).expanduser().resolve(),
        output_dir=Path(_get("output_dir", str(HERE / "workspaces"))).expanduser().resolve(),
        papers=papers_str,
        agents=agents_str,
        runner=_get("runner", "claude-code"),
        model=_get("model"),
        api_key=_get("api_key"),
        from_step=int(_get("from_step", 0)),
        to_step=int(_get("to_step", 4)),
        rerun=bool(_get("rerun", False)),
        rerun_checks=_get("rerun_checks", []),
        max_turns=int(_get("max_turns", 40)),
        timeout=int(_get("timeout", 600)),
        parallel=int(_get("parallel", 1)),
    )


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_papers(results_dir: Path) -> list[str]:
    """Return paper_ids that have at least one verification_report.json."""
    papers = []
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        has_vr = any(
            (rd / "verification_report.json").exists()
            for rd in d.iterdir() if rd.is_dir()
        )
        if has_vr:
            papers.append(d.name)
    return papers


def discover_agents(results_dir: Path, paper_id: str) -> list[dict]:
    """Parse result dir names and return list of {model, approach, label, result_dir}."""
    paper_results = results_dir / paper_id
    if not paper_results.is_dir():
        return []
    agents = []
    for rd in sorted(paper_results.iterdir()):
        if not rd.is_dir():
            continue
        if not (rd / "verification_report.json").exists():
            continue
        name = rd.name
        idx = name.find(f"_{paper_id}_")
        if idx < 0:
            continue
        model = name[:idx]
        approach = name[idx + len(f"_{paper_id}_"):]
        if approach not in KNOWN_APPROACHES:
            continue
        agents.append({
            "model": model,
            "approach": approach,
            "label": f"{model}_{approach}",
            "result_dir": rd,
        })
    return agents


# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------

def _run_script(script: str, args: list[str], cwd: Path | None = None,
                label: str = "") -> int:
    """Run a Python script via subprocess. Returns exit code."""
    cmd = [sys.executable, str(HERE / script)] + args
    print(f"\n{'='*60}")
    print(f"  {label or script}")
    print(f"  cmd: {' '.join(cmd[:4])} ...")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if result.returncode != 0:
        print(f"  WARNING: {script} exited with code {result.returncode}")
    return result.returncode


def run_step_00(cfg: PipelineConfig):
    """Prepare explainer_workspaces."""
    args = [
        "--papers-dir", str(cfg.papers_dir),
        "--results-dir", str(cfg.results_dir),
        "--output-dir", str(cfg.output_dir),
    ]
    if cfg.agents:
        args += ["--agents", ",".join(cfg.agents)]
    if cfg.papers:
        args += ["--papers", ",".join(cfg.papers)]
    if cfg.rerun:
        args.append("--overwrite")
    return _run_script("00_prep_setup.py", args, label="STEP 00 — workspace preparation")


def _run_step_01_single(cfg: PipelineConfig, paper_id: str, agent: dict,
                         language: str) -> int:
    """Run step 01 for a single paper × agent."""
    code_dir = cfg.output_dir / paper_id / agent["label"] / "code"
    vr_path = agent["result_dir"] / "verification_report.json"
    if not code_dir.is_dir() or not vr_path.exists():
        print(f"  SKIP {paper_id}/{agent['label']}: missing code_dir or VR")
        return 0

    args = [
        "--code-dir", str(code_dir),
        "--verification-report", str(vr_path),
        "--runner", cfg.runner,
        "--original-language", language,
        "--max-turns", str(cfg.max_turns),
        "--timeout", str(cfg.timeout),
    ]
    if cfg.model:
        args += ["--model", cfg.model]
    if cfg.api_key:
        args += ["--api-key", cfg.api_key]
    if cfg.rerun:
        args.append("--rerun")
    return _run_script("01_trace_failures.py", args,
                       label=f"STEP 01 — {paper_id} / {agent['label']}")


def _run_step_02_single(cfg: PipelineConfig, paper_id: str, agent: dict,
                         language: str) -> int:
    """Run step 02 for a single paper × agent."""
    agent_dir = cfg.output_dir / paper_id / agent["label"]
    divergences = agent_dir / "code" / "divergences.json"
    ws = agent_dir / "error_source"
    output = ws / "divergences_enriched.json"

    if not divergences.exists():
        print(f"  SKIP {paper_id}/{agent['label']}: no divergences.json")
        return 0

    args = [
        "--comparison", str(divergences),
        "--workspace", str(ws),
        "--output", str(output),
        "--runner", cfg.runner,
        "--original-language", language,
        "--max-turns", str(cfg.max_turns),
        "--timeout", str(cfg.timeout),
    ]
    # Optionally pass data dir for CHECK 4
    data_dir = cfg.papers_dir / paper_id / "replication_package"
    if data_dir.is_dir():
        args += ["--data-dir", str(data_dir)]
    if cfg.model:
        args += ["--model", cfg.model]
    if cfg.api_key:
        args += ["--api-key", cfg.api_key]
    if cfg.rerun:
        args.append("--rerun")
    if cfg.rerun_checks:
        args += ["--rerun-checks"] + [str(c) for c in cfg.rerun_checks]
    return _run_script("02_detect_error_source.py", args,
                       label=f"STEP 02 — {paper_id} / {agent['label']}")


def run_step_03(cfg: PipelineConfig):
    """Generate LaTeX summary tables."""
    args = [
        "--workspace-dir", str(cfg.output_dir),
        "--output-dir", str(cfg.output_dir / "summaries"),
    ]
    if cfg.rerun:
        args.append("--rerun")
    return _run_script("03_summarize_errors.py", args,
                       label="STEP 03 — LaTeX summary tables")


def run_step_04(cfg: PipelineConfig):
    """Generate overview plots."""
    args = [
        "--workspace-dir", str(cfg.output_dir),
        "--output-dir", str(cfg.output_dir / "plots"),
    ]
    if cfg.rerun:
        args.append("--rerun")
    return _run_script("04_overview_stats.py", args,
                       label="STEP 04 — overview plots")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run the error analysis pipeline.")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file.")
    parser.add_argument("--papers-dir", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--papers", type=str, default=None,
                        help="Comma-separated paper IDs (default: auto-discover).")
    parser.add_argument("--agents", type=str, default=None,
                        help="Comma-separated approaches (default: auto-discover).")
    parser.add_argument("--runner", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--from-step", type=int, default=None)
    parser.add_argument("--to-step", type=int, default=None)
    parser.add_argument("--rerun", action="store_true", default=None)
    parser.add_argument("--parallel", type=int, default=None)
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    cfg = load_config(config_path, args)

    # Validate paths
    for p, label in [(cfg.papers_dir, "papers_dir"), (cfg.results_dir, "results_dir")]:
        if not p.is_dir():
            sys.exit(f"ERROR: {label} does not exist: {p}")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Papers dir:  {cfg.papers_dir}")
    print(f"Results dir: {cfg.results_dir}")
    print(f"Output dir:  {cfg.output_dir}")
    print(f"Runner:      {cfg.runner} / {cfg.model or 'default'}")
    print(f"Steps:       {cfg.from_step} → {cfg.to_step}")
    print(f"Parallel:    {cfg.parallel}")

    # Discover papers
    paper_ids = cfg.papers or discover_papers(cfg.results_dir)
    print(f"\nPapers: {len(paper_ids)}")

    # Discover agents per paper and detect language
    work_items: list[tuple[str, dict, str]] = []  # (paper_id, agent_info, language)
    for paper_id in paper_ids:
        agents = discover_agents(cfg.results_dir, paper_id)
        if cfg.agents:
            agents = [a for a in agents if a["approach"] in cfg.agents]
        if not agents:
            print(f"  {paper_id}: no matching agents, skipping")
            continue

        pkg = cfg.papers_dir / paper_id / "replication_package"
        language = detect_language(pkg) if pkg.is_dir() else "unknown"
        lang_name = get_info(language)["name"]

        agent_labels = [a["label"] for a in agents]
        print(f"  {paper_id}: {lang_name}, {len(agents)} agents: {agent_labels}")

        for agent in agents:
            work_items.append((paper_id, agent, language))

    print(f"\nTotal work items: {len(work_items)}")

    # --- Step 00 ---
    if cfg.from_step <= 0 <= cfg.to_step:
        run_step_00(cfg)

    # --- Steps 01 & 02 (per paper × agent) ---
    def _run_01_02(item):
        paper_id, agent, language = item
        rc1, rc2 = 0, 0
        if cfg.from_step <= 1 <= cfg.to_step:
            rc1 = _run_step_01_single(cfg, paper_id, agent, language)
        if cfg.from_step <= 2 <= cfg.to_step:
            rc2 = _run_step_02_single(cfg, paper_id, agent, language)
        return paper_id, agent["label"], rc1, rc2

    if cfg.from_step <= 2 and cfg.to_step >= 1:
        if cfg.parallel > 1:
            with ThreadPoolExecutor(max_workers=cfg.parallel) as pool:
                futures = {pool.submit(_run_01_02, item): item for item in work_items}
                for fut in as_completed(futures):
                    paper_id, label, rc1, rc2 = fut.result()
                    status = "OK" if rc1 == 0 and rc2 == 0 else f"rc1={rc1} rc2={rc2}"
                    print(f"  Completed {paper_id}/{label}: {status}")
        else:
            for item in work_items:
                _run_01_02(item)

    # --- Step 03 ---
    if cfg.from_step <= 3 <= cfg.to_step:
        run_step_03(cfg)

    # --- Step 04 ---
    if cfg.from_step <= 4 <= cfg.to_step:
        run_step_04(cfg)

    print(f"\nDone. All outputs in {cfg.output_dir}/")


if __name__ == "__main__":
    main()
