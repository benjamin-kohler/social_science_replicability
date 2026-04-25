#!/usr/bin/env python3
"""Classify each extracted table in each i4rep paper into a category.

For each paper:
  1. Load {paper}_summary.json (methodology) and {paper}_results.json (ground-truth
     list of extracted tables with IDs + labels).
  2. Build a single prompt containing the methodology summary and the list of tables.
  3. Call gpt-5.4-mini via the OpenAI Responses API with a Pydantic structured
     output schema.
  4. Save classifications to analysis_output/table_categories.json (+ CSV).

Categories:
  - descriptive              : descriptive statistics, sample characteristics, balance
  - main_results             : primary regression/estimation tables for the main claims
  - robustness  : robustness checks, sensitivity, mechanism tests,
                                heterogeneity, placebo, alternative specs
  - other                    : anything that doesn't fit (e.g. survey instruments,
                                variable definitions, qualitative tables)

Each classification also carries an `uncertain` boolean flag and a short reason.

Usage:
    python scripts/classify_table_categories.py \\
      --results-dir data/i4replicate/results \\
      --output analysis_output/table_categories.json \\
      [--papers 10.1257_aer.20201333,...] \\
      [--concurrency 10] [--force]
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
from pathlib import Path
from typing import Literal

from openai import AsyncOpenAI
from pydantic import BaseModel, Field


MODEL = "gpt-5.4-mini"

CATEGORIES = ["descriptive", "main_results", "robustness", "mechanism", "other"]

SYSTEM_PROMPT = """You are an expert social science research assistant. Your task is to \
classify each table in an academic paper into exactly one of five categories:

1. **descriptive** — Tables reporting descriptive statistics, summary statistics, \
sample characteristics, balance tests, correlation matrices, or variable definitions \
(when given as a numeric summary rather than a codebook).

2. **main_results** — Tables reporting the primary regression or estimation results \
that test the paper's central hypotheses. These are the "headline" tables the paper's \
main claims rely on.

3. **robustness** — Tables whose purpose is to demonstrate that the main result is not \
an artefact of a particular choice. Includes sensitivity analyses, alternative \
specifications, alternative samples or estimators, placebo/falsification tests, \
different standard-error clusterings, winsorisation, bandwidth/weight variations, and \
instrumental-variable checks presented as "our result holds under X".

4. **mechanism** — Tables whose purpose is to explain *why* or *how* the main effect \
arises. Includes heterogeneity analyses that speak to the operative channel, \
mediator/moderator regressions, sub-group analyses designed to adjudicate between \
competing theories, decomposition tables, and tests on intermediate outcomes. \
Heterogeneity aimed at probing the mechanism is "mechanism"; heterogeneity aimed at \
showing the main result holds in different samples is "robustness". If a table clearly \
does both, pick the one the paper appears to emphasize and set `uncertain=true`.

5. **other** — Anything that does not fit the above (e.g. survey instruments given as \
a table, qualitative categorizations, experimental design tables, structural parameter \
calibrations not tied to main results).

For each table, output:
  - table_id: the exact table_id as given in the input
  - category: one of the five labels above
  - uncertain: true if you are not confident in the classification (e.g. the caption \
is ambiguous, multiple categories plausibly fit, or the context is insufficient)
  - reason: a one-sentence justification grounded in the caption, column/row labels, \
and methodology summary

Be decisive when possible. Set `uncertain=true` only when the evidence genuinely \
does not discriminate."""


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class TableClassification(BaseModel):
    table_id: str
    category: Literal["descriptive", "main_results", "robustness", "mechanism", "other"]
    uncertain: bool = Field(description="True if you are not confident in the classification.")
    reason: str = Field(description="One-sentence justification.")


class PaperTableAudit(BaseModel):
    paper_id: str
    classifications: list[TableClassification]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _summary_block(summary: dict) -> str:
    """Extract a compact methodology block from the paper summary."""
    parts = []
    if summary.get("title"):
        parts.append(f"**Title**: {summary['title']}")
    rqs = summary.get("research_questions")
    if rqs:
        if isinstance(rqs, list):
            rqs = "; ".join(str(r) for r in rqs)
        parts.append(f"**Research questions**: {rqs}")
    for key, label in [
        ("data_description", "Data description"),
        ("data_context", "Data context"),
        ("data_source", "Data source"),
        ("sample_size", "Sample size"),
        ("time_period", "Time period"),
    ]:
        v = summary.get(key)
        if v:
            parts.append(f"**{label}**: {v}")
    steps = summary.get("data_processing_steps")
    if steps:
        if isinstance(steps, list):
            steps = "\n".join(f"- {s}" for s in steps)
        parts.append(f"**Data processing**:\n{steps}")

    # Include the methodology-side table specs if present — gives the LLM the
    # paper's intent for each table (often richer than the caption alone).
    tbl_specs = summary.get("tables") or []
    if tbl_specs:
        spec_lines = ["**Table specs from methodology summary**:"]
        for t in tbl_specs:
            if not isinstance(t, dict):
                continue
            tid = t.get("table_number") or t.get("table_id") or "?"
            cap = t.get("caption", "") or ""
            spec_lines.append(f"- {tid}: {cap}")
            cols = t.get("column_names") or []
            rows = t.get("row_names") or []
            if cols:
                spec_lines.append(f"    columns: {cols}")
            if rows:
                spec_lines.append(f"    rows: {rows[:10]}{'...' if len(rows) > 10 else ''}")
        parts.append("\n".join(spec_lines))

    return "\n\n".join(parts)


def _tables_to_classify_block(results: dict) -> tuple[str, list[str]]:
    """Return the formatted table list and a list of table_ids to classify."""
    tables = results.get("tables") or []
    ids: list[str] = []
    lines = ["**Tables to classify** (ground-truth list from the extractor):"]
    for t in tables:
        if not isinstance(t, dict):
            continue
        tid = t.get("table_id", "")
        if not tid:
            continue
        ids.append(tid)
        cols = t.get("column_labels") or []
        rows = t.get("row_labels") or []
        notes = t.get("notes", "") or ""
        lines.append(f"\n- **table_id**: `{tid}`")
        if cols:
            lines.append(f"  - columns: {cols}")
        if rows:
            lines.append(f"  - rows: {rows[:12]}{'...' if len(rows) > 12 else ''}")
        if notes:
            lines.append(f"  - notes: {notes[:300]}")
    return "\n".join(lines), ids


def build_prompt(summary: dict, results: dict) -> tuple[str, list[str]]:
    summary_text = _summary_block(summary)
    tables_text, ids = _tables_to_classify_block(results)
    prompt = (
        f"{summary_text}\n\n"
        f"{tables_text}\n\n"
        "Classify every table listed above. Return one classification per table, "
        "using the exact table_id from the input."
    )
    return prompt, ids


# ---------------------------------------------------------------------------
# API call with retries
# ---------------------------------------------------------------------------

async def classify_paper(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    paper_id: str,
    summary: dict,
    results: dict,
) -> dict | None:
    prompt, expected_ids = build_prompt(summary, results)
    if not expected_ids:
        print(f"[{paper_id}] SKIP — no tables in results.json")
        return None

    async with semaphore:
        for attempt in range(1, 4):
            try:
                # temperature=0 is passed as a best-effort nudge toward determinism.
                # Reasoning-style models (gpt-5.x-mini) generally ignore the sampling
                # temperature and rely on internal reasoning tokens, so expect some
                # residual drift across runs even with this setting.
                resp = await client.responses.parse(
                    model=MODEL,
                    instructions=SYSTEM_PROMPT,
                    input=prompt,
                    text_format=PaperTableAudit,
                    temperature=0,
                )
                parsed = resp.output_parsed
                if parsed is None:
                    raise RuntimeError("Empty output_parsed")
                # Ensure paper_id is ours, not what the model invented
                parsed.paper_id = paper_id
                out = parsed.model_dump()
                out["n_tables_expected"] = len(expected_ids)
                out["n_tables_classified"] = len(out["classifications"])
                # Warn on missing tables
                got_ids = {c["table_id"] for c in out["classifications"]}
                missing = [i for i in expected_ids if i not in got_ids]
                if missing:
                    out["missing_table_ids"] = missing
                    print(f"[{paper_id}] warn: {len(missing)} tables not classified")
                print(f"[{paper_id}] OK — {len(out['classifications'])}/{len(expected_ids)} classified")
                return out
            except Exception as e:
                print(f"[{paper_id}] attempt {attempt}/3 failed: {e}")
                if attempt < 3:
                    await asyncio.sleep(2 * attempt)
        print(f"[{paper_id}] FAILED after 3 attempts")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_existing(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        if isinstance(data, list):
            return {d["paper_id"]: d for d in data if isinstance(d, dict) and d.get("paper_id")}
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def save_results(path: Path, results: dict[str, dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    as_list = list(results.values())
    path.write_text(json.dumps(as_list, indent=2, ensure_ascii=False))


def save_csv(path: Path, results: dict[str, dict]):
    rows = []
    for paper_id, res in results.items():
        for c in res.get("classifications", []):
            rows.append({
                "paper_id": paper_id,
                "table_id": c.get("table_id", ""),
                "category": c.get("category", ""),
                "uncertain": c.get("uncertain", False),
                "reason": c.get("reason", ""),
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="data/i4replicate/results")
    p.add_argument("--output", default="analysis_output/table_categories.json")
    p.add_argument("--csv", default="analysis_output/table_categories.csv")
    p.add_argument("--papers", default="",
                    help="Comma-separated list of paper IDs to classify (default: all).")
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--force", action="store_true",
                    help="Re-classify papers even if already in the output file.")
    return p.parse_args()


async def main_async():
    args = parse_args()
    results_dir = Path(args.results_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    csv_path = Path(args.csv).expanduser().resolve()

    if not results_dir.is_dir():
        sys.exit(f"ERROR: --results-dir does not exist: {results_dir}")
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ERROR: OPENAI_API_KEY not set")

    paper_filter = set(p.strip() for p in args.papers.split(",") if p.strip()) or None

    existing = load_existing(output_path)
    print(f"Loaded {len(existing)} existing classifications from {output_path}")

    # Discover papers that have both summary.json and results.json
    tasks_input: list[tuple[str, dict, dict]] = []
    for paper_dir in sorted(results_dir.iterdir()):
        if not paper_dir.is_dir():
            continue
        paper_id = paper_dir.name
        if paper_filter and paper_id not in paper_filter:
            continue
        if paper_id in existing and not args.force:
            continue
        summ_path = paper_dir / "summaries" / f"{paper_id}_summary.json"
        res_path = paper_dir / "summaries" / f"{paper_id}_results.json"
        if not summ_path.exists() or not res_path.exists():
            print(f"[{paper_id}] SKIP — missing summary or results json")
            continue
        try:
            summary = json.loads(summ_path.read_text())
            results_json = json.loads(res_path.read_text())
        except Exception as e:
            print(f"[{paper_id}] SKIP — JSON load failed: {e}")
            continue
        tasks_input.append((paper_id, summary, results_json))

    if not tasks_input:
        print("Nothing to do.")
        save_results(output_path, existing)
        save_csv(csv_path, existing)
        return

    print(f"Classifying {len(tasks_input)} papers with model={MODEL}, "
            f"concurrency={args.concurrency}")

    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(args.concurrency)
    coros = [
        classify_paper(client, semaphore, pid, summ, res)
        for pid, summ, res in tasks_input
    ]
    results = await asyncio.gather(*coros)

    # Merge
    for r in results:
        if r is None:
            continue
        existing[r["paper_id"]] = r

    save_results(output_path, existing)
    save_csv(csv_path, existing)
    print(f"\nSaved {len(existing)} papers to {output_path}")
    print(f"Saved CSV to {csv_path}")

    # Quick summary
    from collections import Counter
    cat_counts = Counter()
    n_uncertain = 0
    n_total = 0
    for res in existing.values():
        for c in res.get("classifications", []):
            cat_counts[c.get("category", "?")] += 1
            n_total += 1
            if c.get("uncertain"):
                n_uncertain += 1
    print(f"\nCategory distribution (n={n_total} tables, {n_uncertain} uncertain):")
    for cat, n in cat_counts.most_common():
        print(f"  {cat:30} {n:5}  ({100*n/n_total:.1f}%)")


if __name__ == "__main__":
    asyncio.run(main_async())
