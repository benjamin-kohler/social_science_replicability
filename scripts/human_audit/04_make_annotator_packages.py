#!/usr/bin/env python3
"""Create self-contained annotator packages for the error-attribution audit.

Draws N_PER_PACKAGE x N_PACKAGES divergences (disjoint, seeded) from the
existing 180-record bundle, and for each package builds a directory with:
  - index.html          (interface with only that package's records)
  - bundle/             (only the assets those records need)
  - README.md           (annotator instructions)
  - start.command / start.sh   (double-click / run to serve + open browser)
then zips it (human_audit/packages/error_audit_<X>.zip) ready to email.

Usage:
    python3 scripts/human_audit/04_make_annotator_packages.py [--seed 7] [--per-package 10]
"""

import argparse
import importlib.util
import json
import random
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PACKAGES = REPO / "human_audit" / "packages"
BUNDLE = None  # set in main() from --bundle

# import build_html from 03_build_interface.py (module name starts with a digit)
spec = importlib.util.spec_from_file_location(
    "build_interface", Path(__file__).parent / "03_build_interface.py")
build_interface = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_interface)

README = """\
# Error-Attribution Audit — Annotator Package {label}

Thanks for helping validate our LLM-based error attribution! This package
contains **{n} divergences** (cases where an AI agent's reproduction of a
published table differs from the original) together with everything needed to
judge them: the paper, the methodology summary the agent worked from, the
original and agent code, and both tables.

Your job is to **validate the labels our pipeline assigned** — not to
re-assign them yourself.
{extra}

## Setup (one time)

1. Unzip this folder anywhere.
2. Start the local viewer:
   - **macOS**: double-click `start.command`
     (if blocked: right-click -> Open, or run `bash start.sh` in a terminal)
   - **Linux/Windows**: run `bash start.sh`, or manually:
     `cd` into the folder and run `python3 -m http.server 8765`,
     then open http://localhost:8765/
3. Enter your name in the top-right "Annotator name" field.

## Annotating (per divergence, ~5 min)

For each of the {n} items in the left sidebar:

1. Read the **description** and open the **evidence snippets** toggle.
2. Use the **Open inputs** buttons as needed — everything opens in an inline
   panel (⌘/ctrl-click for a real tab): original & replicated table, paper
   PDF, methodology summary, and the original / agent scripts (auto-scrolled
   to the relevant lines).
3. Answer the four validation questions:
   - **Q1** Is this an actual, meaningful divergence/error?
   - **Q2-Q4** Are the assigned error source, divergence type, and severity
     correct? (Each label's definition is shown next to the question.)
   - If Q1 = "No", the rest is disabled — just move on.
   - Use **Unsure** when the evidence genuinely doesn't let you decide;
     use the **notes field** to say what the correct label would be whenever
     you answer "Incorrect".
4. Your answers save automatically (in your browser). The progress counter
   is top-left; the "To do" filter shows what's left.

## When you're done

Click **Export JSON** *and* **Export CSV** (top right) and email both files
back. That's it — thank you!

*Note: annotations are stored in your browser's local storage for this page,
so finish the audit on the same computer and browser you started with.*
"""

START_SH = """\
#!/bin/bash
cd "$(dirname "$0")"
echo "Serving audit interface at http://localhost:8765/ (Ctrl-C to stop)"
(sleep 1 && (open http://localhost:8765/ 2>/dev/null || xdg-open http://localhost:8765/ 2>/dev/null)) &
python3 -m http.server 8765
"""


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--per-package", type=int, default=10)
    ap.add_argument("--packages", nargs="+", default=["A", "B"])
    ap.add_argument("--bundle", default=str(REPO / "human_audit" / "bundle"),
                    help="bundle dir whose records.json is the sampling pool")
    ap.add_argument("--assign", action="append", default=None, metavar="LABEL=slug1,slug2",
                    help="assign ALL records of these papers to package LABEL "
                         "(record order preserved; overrides random sampling)")
    return ap.parse_args()


def copy_assets(records: list[dict], dest_bundle: Path):
    """Copy only the papers/ and workspaces/ trees these records reference."""
    papers = {r["paper_slug"] for r in records}
    workspaces = {(r["paper_slug"], r["agent_label"]) for r in records}
    for p in papers:
        shutil.copytree(BUNDLE / "papers" / p, dest_bundle / "papers" / p)
    for p, a in workspaces:
        src = BUNDLE / "workspaces" / p / a
        if src.exists():
            shutil.copytree(src, dest_bundle / "workspaces" / p / a)


def main():
    global BUNDLE
    args = parse_args()
    BUNDLE = Path(args.bundle)
    data = json.loads((BUNDLE / "records.json").read_text())
    all_records = data["records"]

    if args.assign:
        assignment = {}
        for spec in args.assign:
            label, slugs = spec.split("=", 1)
            assignment[label] = [s.strip() for s in slugs.split(",") if s.strip()]
        packages = {label: [r for r in all_records if r["paper_slug"] in slugs]
                    for label, slugs in assignment.items()}
        for label, slugs in assignment.items():
            found = {r["paper_slug"] for r in packages[label]}
            if missing := set(slugs) - found:
                raise SystemExit(f"ERROR: package {label}: no records for {missing}")
    else:
        rng = random.Random(args.seed)
        n_total = args.per_package * len(args.packages)
        picked = rng.sample(all_records, n_total)
        rng.shuffle(picked)
        packages = {label: picked[i * args.per_package:(i + 1) * args.per_package]
                    for i, label in enumerate(args.packages)}

    PACKAGES.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for label, recs in packages.items():
        manifest[label] = [r["audit_id"] for r in recs]

        pkg = PACKAGES / f"error_audit_{label}"
        if pkg.exists():
            shutil.rmtree(pkg)
        pkg.mkdir(parents=True)

        html = build_interface.build_html({"records": recs, "sample_meta": data.get("sample_meta", {})})
        (pkg / "index.html").write_text(html)
        (pkg / "bundle").mkdir()
        (pkg / "bundle" / "records.json").write_text(json.dumps(
            {"sample_meta": data.get("sample_meta", {}), "records": recs}, indent=2))
        copy_assets(recs, pkg / "bundle")
        papers_in_pkg = sorted({r["paper_slug"] for r in recs})
        extra = ""
        if args.assign:
            extra = (f"\nThis package covers **{len(papers_in_pkg)} papers in full** "
                     f"({', '.join(papers_in_pkg)}). Items are grouped by paper and table, "
                     "and the seven agent runs of a paper often diverge for the same "
                     "underlying reason — the first items of each paper take longest, "
                     "later ones go much faster.\n")
        (pkg / "README.md").write_text(README.format(label=label, n=len(recs), extra=extra))
        for name in ("start.sh", "start.command"):
            f = pkg / name
            f.write_text(START_SH)
            f.chmod(0o755)

        zip_path = PACKAGES / f"error_audit_{label}.zip"
        zip_path.unlink(missing_ok=True)
        subprocess.run(["zip", "-qr", str(zip_path), pkg.name], cwd=PACKAGES, check=True)
        size_mb = zip_path.stat().st_size / 1e6
        causes = sorted(r["root_cause"] for r in recs)
        print(f"Package {label}: {len(recs)} divergences, zip {size_mb:.1f} MB -> {zip_path}")
        print(f"  audit_ids: {', '.join(manifest[label])}")
        print(f"  root causes: {causes}")

    (PACKAGES / "packages_manifest.json").write_text(json.dumps(
        {"seed": args.seed, "per_package": args.per_package, "packages": manifest}, indent=2))
    print(f"\nManifest -> {PACKAGES / 'packages_manifest.json'}")


if __name__ == "__main__":
    main()
