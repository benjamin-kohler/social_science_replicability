#!/usr/bin/env python3
"""Draw a stratified sample of divergences for the human audit.

Strata: root_cause (9 categories), N per stratum (default 20), parse failures
excluded. Output: human_audit/sample.json — consumed by 02_bundle_assets.py
(on textlab) and 03_build_interface.py (local).

Usage:
    python scripts/human_audit/01_sample_divergences.py \
        [--csv analysis_output_regrade_na/df_divergences.csv] \
        [--per-stratum 20] [--seed 42] [--output human_audit/sample.json]
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(REPO / "analysis_output_regrade_na/df_divergences.csv"))
    ap.add_argument("--per-stratum", type=int, default=20)
    ap.add_argument("--uniform-n", type=int, default=None,
                    help="draw N divergences uniformly at random instead of stratifying")
    ap.add_argument("--papers", nargs="+", default=None,
                    help="take ALL divergences of these paper slugs (whole-paper review)")
    ap.add_argument("--id-prefix", default="d")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=str(REPO / "human_audit/sample.json"))
    return ap.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.csv)
    n_total = len(df)
    df = df[~df["parse_failed"]].copy()
    print(f"{n_total} divergences, {len(df)} after excluding parse failures")

    if args.papers:
        sample = df[df.paper_slug.isin(args.papers)].copy()
        missing = set(args.papers) - set(sample.paper_slug)
        if missing:
            sys.exit(f"ERROR: no divergences found for: {missing}")
        # group by paper -> table -> agent so annotators review papers as a unit
        sample = sample.sort_values(["paper_slug", "output", "agent_label", "div_id"])
        print(sample.groupby("paper_slug").size().to_string())
        meta = {"mode": "papers", "papers": args.papers}
    elif args.uniform_n:
        sample = df.sample(n=args.uniform_n, random_state=args.seed)
        print(f"uniform draw: {len(sample)} divergences")
        print(sample["root_cause"].value_counts().to_string())
        meta = {"mode": "uniform", "n_drawn": args.uniform_n}
    else:
        parts = []
        for cause, grp in df.groupby("root_cause"):
            n = min(args.per_stratum, len(grp))
            parts.append(grp.sample(n=n, random_state=args.seed))
            print(f"  {cause:<30s} {n:>3d} / {len(grp)}")
        sample = pd.concat(parts)
        meta = {"mode": "stratified", "per_stratum": args.per_stratum}
    if not args.papers:
        sample = sample.sample(frac=1, random_state=args.seed)  # shuffle order

    records = sample.to_dict(orient="records")
    for i, r in enumerate(records):
        r["audit_id"] = f"{args.id_prefix}{i+1:03d}"

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "seed": args.seed,
        **meta,
        "source_csv": str(args.csv),
        "n": len(records),
        "divergences": records,
    }, indent=2))
    print(f"\nWrote {len(records)} sampled divergences -> {out}")
    print(f"Papers: {sample['paper_slug'].nunique()}, "
          f"paper x agent workspaces: {sample.groupby(['paper_slug','agent_label']).ngroups}")


if __name__ == "__main__":
    main()
