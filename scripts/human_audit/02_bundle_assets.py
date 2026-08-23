#!/usr/bin/env python3
"""Bundle assets for the human audit of error attribution (runs on textlab).

Reads sample.json (from 01_sample_divergences.py), pulls the full enriched
divergence records plus the files an annotator needs, and writes everything
into a self-contained bundle directory:

    bundle/
      records.json                       full divergence records + asset paths
      papers/{paper}/paper.pdf
      papers/{paper}/methodology_summary.json
      papers/{paper}/original_code/...   code files only (no data), tree preserved
      workspaces/{paper}/{agent}/agent_code/*.py

Usage (on textlab):
    python3 02_bundle_assets.py --sample sample.json --output bundle \
        [--workspaces /data/individual/benjamin/social_science_replicability/src/code_JE/explainer_workspaces_all]
"""

import argparse
import json
import re
import shutil
from pathlib import Path

CODE_EXTS = {".do", ".ado", ".r", ".rmd", ".m", ".py", ".jl", ".sas", ".sh", ".txt", ".md", ".pdf"}
MAX_CODE_FILE_MB = 5  # skip oversized files (some packages ship huge logs/pdfs)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--workspaces",
                    default="/data/individual/benjamin/social_science_replicability/src/code_JE/explainer_workspaces_all")
    ap.add_argument("--results",
                    default="/data/individual/benjamin/social_science_replicability/data/i4replicate/results")
    return ap.parse_args()


KNOWN_APPROACHES = ("claude-code", "codex", "swe-agent", "opencode")


def run_dir_for(results_root: Path, paper: str, agent_label: str) -> Path | None:
    """results/{paper}/{model}_{paper}_{approach}/ for agent_label = {model}_{approach}."""
    for approach in KNOWN_APPROACHES:
        if agent_label.endswith(f"_{approach}"):
            model = agent_label[: -len(approach) - 1]
            d = results_root / paper / f"{model}_{paper}_{approach}"
            return d if d.exists() else None
    return None


def norm_id(s: str) -> str:
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())


def derive_root_cause(d: dict) -> str:
    """Same cascade as analyze_i4rep_results.py — derived from the record itself
    so the displayed label always matches the displayed verdicts."""
    p_code = d.get("paper_vs_original_code", "unclear")
    p_sum = d.get("paper_vs_summary", "unclear")
    s_agent = d.get("summary_vs_agent", "unclear")
    if d.get("data_available") == "missing":
        return "Data not in package"
    if p_code == "contradicts":
        return "Paper-code mismatch"
    if p_code == "omission":
        return "Paper underspecified"
    if p_sum == "contradicts":
        return "Summary gap (contradicts)"
    if p_sum == "omission":
        return "Summary gap (omission)"
    if s_agent == "contradicts":
        return "Agent contradicted summary"
    if s_agent == "omission":
        return "Agent missed summary info"
    if "unclear" in (p_code, p_sum, s_agent):
        return "Insufficient specification"
    return "Unexplained"


EXCLUDE_DIRS = {"__pycache__", ".git", "data"}


def copy_original_code(pkg_dir: Path, dest: Path) -> int:
    """Copy code/readme files from the replication package, preserving the tree."""
    n = 0
    for f in pkg_dir.rglob("*"):
        if not f.is_file():
            continue
        if any(part in EXCLUDE_DIRS for part in f.relative_to(pkg_dir).parts[:-1]):
            continue
        if f.suffix.lower() not in CODE_EXTS:
            continue
        if f.stat().st_size > MAX_CODE_FILE_MB * 1024 * 1024:
            continue
        rel = f.relative_to(pkg_dir)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(f, target)
            n += 1
    return n


def resolve_files(raw: str | None, prefix: str, src_root: Path,
                  out: Path, bundle_prefix: str) -> list[str]:
    """Parse a (possibly multi-file) location string, ensure each file is in the
    bundle (copying directly if the bulk filter missed it), return bundle paths."""
    if not raw or raw.strip().upper() == "ABSENT":
        return []

    def _try(rel: str) -> str | None:
        """Return the bundle path for `rel` if it exists / can be copied in."""
        rel = rel.strip().removeprefix(prefix).lstrip("/")
        if not rel or rel.upper() == "ABSENT":
            return None
        src = src_root / rel
        if not src.is_file():
            # fall back to a basename search in the source tree
            hits = [h for h in src_root.rglob(Path(rel).name) if h.is_file()]
            if len(hits) != 1:
                return None
            src = hits[0]
            rel = str(src.relative_to(src_root))
        bundle_path = f"{bundle_prefix}/{rel}"
        if not (out / bundle_path).exists():
            if src.stat().st_size >= 20 * 1024 * 1024:
                return None
            target = out / bundle_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
        return bundle_path

    # commas can appear inside real filenames — try the full string first,
    # then progressively finer splits
    for pattern in (None, r";| and ", r";|,| and "):
        parts = [raw] if pattern is None else re.split(pattern, raw)
        resolved = [bp for p in parts if (bp := _try(p))]
        if resolved:
            return resolved
    return []


def main():
    args = parse_args()
    sample = json.loads(Path(args.sample).read_text())
    ws_root = Path(args.workspaces)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    records = []
    done_papers = set()
    done_workspaces = set()
    done_tables = set()
    missing = []

    for s in sample["divergences"]:
        paper, agent, div_id = s["paper_slug"], s["agent_label"], s["div_id"]
        ws = ws_root / paper / agent
        enriched_path = ws / "error_source" / "divergences_enriched.json"
        if not enriched_path.exists():
            missing.append((paper, agent, div_id, "no enriched json"))
            continue
        data = json.loads(enriched_path.read_text())
        div = next((d for d in data.get("divergences", []) if d.get("id") == div_id), None)
        if div is None:
            missing.append((paper, agent, div_id, "div id not found"))
            continue

        # --- per-paper original extracted results (for table rendering) ---
        results_json = out / "papers" / paper / "original_results.json"
        if not results_json.exists():
            src = Path(args.results) / paper / "summaries" / f"{paper}_results.json"
            if src.exists():
                results_json.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, results_json)

        # --- per-workspace replicated table JSONs ---
        wtables_dir = out / "workspaces" / paper / agent / "tables"
        if (paper, agent) not in done_tables:
            rd = run_dir_for(Path(args.results), paper, agent)
            if rd is not None:
                for tf in sorted((rd / "workspace").glob("table_*.json")):
                    wtables_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(tf, wtables_dir / tf.name)
            done_tables.add((paper, agent))

        # map this divergence's output item to its replicated table file
        replicated_table = ""
        if wtables_dir.exists():
            target = norm_id(div.get("output", ""))
            for tf in sorted(wtables_dir.glob("table_*.json")):
                try:
                    tid = norm_id(json.loads(tf.read_text()).get("table_id", ""))
                except Exception:
                    continue
                if tid and (tid == target or tid in target or target in tid):
                    replicated_table = f"workspaces/{paper}/{agent}/tables/{tf.name}"
                    break

        # --- per-paper assets (paper.pdf, summary, original code) ---
        if paper not in done_papers:
            pdir = out / "papers" / paper
            pdir.mkdir(parents=True, exist_ok=True)
            pdf = ws / "error_source" / "paper_vs_original_code" / "paper.pdf"
            if pdf.exists():
                shutil.copy2(pdf, pdir / "paper.pdf")
            summ = ws / "error_source" / "paper_vs_summary" / "methodology_summary.json"
            if summ.exists():
                (pdir / "methodology_summary.json").write_text(
                    json.dumps(json.loads(summ.read_text()), indent=2))
            pkg = (ws / "code" / "original_code").resolve()
            if pkg.exists():
                n = copy_original_code(pkg, pdir / "original_code")
                print(f"  {paper}: {n} original code files")
            done_papers.add(paper)

        # --- per-workspace assets (agent code; filtered — some agents copied data in) ---
        wkey = (paper, agent)
        if wkey not in done_workspaces:
            adest = out / "workspaces" / paper / agent / "agent_code"
            asrc = ws / "code" / "agent_code"
            if asrc.exists() and not adest.exists():
                copy_original_code(asrc, adest)
            done_workspaces.add(wkey)

        # --- resolve referenced file paths within the bundle ---
        orig_files = resolve_files(
            (div.get("original_location") or {}).get("file"), "original_code/",
            (ws / "code" / "original_code").resolve(), out, f"papers/{paper}/original_code")
        agent_files = resolve_files(
            (div.get("agent_location") or {}).get("file"), "agent_code/",
            ws / "code" / "agent_code", out, f"workspaces/{paper}/{agent}/agent_code")

        root_cause = derive_root_cause(div)
        if root_cause != s.get("root_cause"):
            print(f"  WARNING {s['audit_id']}: sample root_cause "
                  f"{s.get('root_cause')!r} != derived {root_cause!r} (using derived)")
        records.append({
            **{k: s[k] for k in ("audit_id", "paper_slug", "agent_label", "approach", "div_id")},
            "root_cause": root_cause,
            "record": div,
            "assets": {
                "paper_pdf": f"papers/{paper}/paper.pdf",
                "summary": f"papers/{paper}/methodology_summary.json",
                "original_code_dir": f"papers/{paper}/original_code/",
                "agent_code_dir": f"workspaces/{paper}/{agent}/agent_code/",
                "original_files": orig_files,
                "agent_files": agent_files,
                "original_results": (f"papers/{paper}/original_results.json"
                                     if results_json.exists() else ""),
                "replicated_table": replicated_table,
            },
        })

    (out / "records.json").write_text(json.dumps({
        "sample_meta": {k: v for k, v in sample.items() if k != "divergences"},
        "records": records,
    }, indent=2))
    print(f"\n{len(records)} records bundled -> {out/'records.json'}")
    if missing:
        print(f"MISSING ({len(missing)}):")
        for m in missing:
            print("  ", m)


if __name__ == "__main__":
    main()
