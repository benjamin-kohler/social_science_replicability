#!/usr/bin/env python3
"""Audit replication package data directories for output contamination.

Scans all paper directories in postcutoff/ and i4replicate/ collections,
collects file trees and READMEs, and uses gpt-5-mini to classify whether
the data/ directory contains raw data, pre-computed outputs, or both.

Usage:
  python scripts/audit_replication_data.py                     # all papers
  python scripts/audit_replication_data.py --only postcutoff   # one collection
  python scripts/audit_replication_data.py --only i4rep        # one collection
  python scripts/audit_replication_data.py --papers 209827 209484  # specific papers
  python scripts/audit_replication_data.py --concurrency 5     # limit parallel requests
"""

import argparse
import asyncio
import base64
import csv
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
POSTCUTOFF_PAPERS = PROJECT_ROOT / "data" / "postcutoff" / "papers"
I4REP_PAPERS = PROJECT_ROOT / "data" / "i4replicate" / "papers"
POSTCUTOFF_ZIPS = Path("/data/individual/benjamin/openicpsr_aea/post_cutoff_sample")
I4REP_ZIPS = PROJECT_ROOT / "data" / "i4replicate" / "replication_packages"

OUTPUT_JSON = PROJECT_ROOT / "data" / "audit_replication_data.json"
OUTPUT_CSV = PROJECT_ROOT / "data" / "audit_replication_data.csv"

# ---------------------------------------------------------------------------
# Structured output model
# ---------------------------------------------------------------------------


class DataAuditResult(BaseModel):
    """Classification of a replication package's data directory."""

    paper_id: str = Field(description="Paper identifier")
    has_raw_data: bool = Field(
        description="Whether raw/source data files are present in the data directory"
    )
    has_output_files: bool = Field(
        description="Whether pre-computed results, regression outputs, or derived "
        "tables are present in the data directory"
    )
    raw_data_paths: list[str] = Field(
        description="Paths (relative to data/) that contain raw data only. "
        "Use directory paths where possible (e.g. 'Raw/' not individual files)"
    )
    output_paths: list[str] = Field(
        description="Paths (relative to data/) that contain pre-computed outputs, "
        "results, or derived data that should NOT be given to the replicator"
    )
    data_sufficiency: Literal["sufficient", "partial", "insufficient", "confidential"] = Field(
        description="Whether the raw data is sufficient to replicate all tables/figures. "
        "'sufficient': all raw data present. "
        "'partial': some data present but key variables/files missing. "
        "'insufficient': very little usable data. "
        "'confidential': raw data is restricted/confidential, only outputs provided."
    )
    sufficiency_explanation: str = Field(
        description="Brief explanation of why data is or isn't sufficient for replication"
    )
    readme_found: bool = Field(description="Whether a README was found in the package")
    notes: str = Field(
        description="Any other relevant observations about the data structure"
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are auditing replication packages for economics research papers. Your task
is to classify what is in each package's data directory.

## Key distinction: raw data vs outputs

**Raw data** (should be in data/):
- Original survey data, administrative records, census data
- Source datasets (.dta, .csv, .xlsx) that are inputs to analysis code
- Geographic data (shapefiles, etc.)

**Outputs/results** (should NOT be in data/):
- Pre-computed regression results (.RDS, .xlsx, .dta files in output/ or results/ directories)
- Generated tables (Table_1.xlsx, results.csv, etc.)
- Derived/intermediate datasets explicitly produced by analysis code
- Log files from statistical software runs
- Formatted tables (.tex, .rtf) produced by code

**Code** (should NOT be in data/):
- Analysis scripts (.do, .R, .py, .m)
- Shell scripts (.sh)
- Configuration files

## What to look for

1. Directory names like "output/", "results/", "tables/", "figures/" inside data/ — these contain outputs
2. Files that look like regression results (e.g., Regs_N_L1.RDS, Table_1.xlsx)
3. README descriptions of what is raw data vs what is generated
4. Whether the raw data alone would be sufficient to run regressions from scratch

## Data sufficiency

- "sufficient": The package contains the actual microdata/survey data needed to run the regressions described
- "partial": Some data is present but key datasets are missing (maybe confidential)
- "insufficient": Almost no usable raw data
- "confidential": The README or structure indicates raw data is restricted/confidential; only derived/output data is provided
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def collect_file_tree(directory: Path) -> str:
    """Collect a file tree with sizes for a directory."""
    if not directory.exists():
        return "(directory does not exist)"
    lines = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(directory)
        try:
            size = path.stat().st_size
            if size > 1_000_000:
                size_str = f"{size / 1_000_000:.1f}MB"
            elif size > 1000:
                size_str = f"{size / 1000:.0f}KB"
            else:
                size_str = f"{size}B"
        except OSError:
            size_str = "?"
        lines.append(f"  {rel}  ({size_str})")
    if not lines:
        return "(empty directory)"
    return "\n".join(lines)


def collect_zip_tree(paper_id: str, collection: str) -> str:
    """Get the file listing from the original ZIP (before data/code split)."""
    if collection == "postcutoff":
        zip_path = POSTCUTOFF_ZIPS / f"{paper_id}-V1.zip"
    else:
        zip_path = I4REP_ZIPS / f"{paper_id}.zip"

    if not zip_path.exists():
        return "(original ZIP not found)"

    try:
        result = subprocess.run(
            ["unzip", "-l", str(zip_path)],
            capture_output=True, text=True, timeout=30,
        )
        output = result.stdout
        if len(output) > 30000:
            lines = output.splitlines()
            output = "\n".join(lines[:500]) + f"\n... ({len(lines) - 500} more files truncated)"
        return output
    except Exception as e:
        return f"(error reading ZIP: {e})"


def find_readmes(paper_dir: Path) -> list[tuple[Path, str]]:
    """Find all README files in a paper directory. Returns [(path, type)].

    Prefers .md/.txt over .html (which can be huge). Deduplicates by stem
    so we don't read both README.md and README.html for the same content.
    """
    candidates = []
    for path in sorted(paper_dir.rglob("*")):
        if not path.is_file():
            continue
        name = path.name.lower()
        if name.startswith("readme") or name == "codebook.pdf":
            ext = path.suffix.lower()
            # Priority: .md=0, .txt=1, .pdf=2, .html=3
            priority = {".md": 0, ".txt": 1, ".pdf": 2, ".html": 3, ".htm": 3, ".rtf": 4, "": 1}
            candidates.append((priority.get(ext, 5), path, ext))

    # Deduplicate by stem+parent — keep highest priority (lowest number)
    seen = {}
    for pri, path, ext in sorted(candidates):
        key = (path.parent, path.stem.lower())
        if key not in seen:
            seen[key] = (path, ext)

    readmes = []
    for path, ext in seen.values():
        if ext == ".pdf":
            readmes.append((path, "pdf"))
        else:
            readmes.append((path, "text"))
    return readmes


def read_text_readme(path: Path, max_chars: int = 30000) -> str:
    """Read a text README file, capping at max_chars."""
    try:
        content = path.read_text(errors="replace")
        if len(content) > max_chars:
            content = content[:max_chars] + f"\n\n... (truncated, {len(content)} chars total)"
        return content
    except Exception as e:
        return f"(error reading {path.name}: {e})"


def pdf_to_base64_images(pdf_path: Path, dpi: int = 100, max_pages: int = 5) -> list[dict]:
    """Convert PDF pages to base64-encoded PNG images."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return []

    images = []
    try:
        doc = fitz.open(str(pdf_path))
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            images.append({
                "base64": base64.b64encode(img_bytes).decode("utf-8"),
                "media_type": "image/png",
                "page": i + 1,
            })
        doc.close()
    except Exception as e:
        log.warning(f"Failed to convert PDF {pdf_path}: {e}")
    return images


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


async def audit_paper(
    client,
    paper_id: str,
    paper_dir: Path,
    collection: str,
    semaphore: asyncio.Semaphore,
) -> DataAuditResult:
    """Audit a single paper's data directory using gpt-5-mini."""
    async with semaphore:
        log.info(f"Auditing {paper_id} ({collection})")

        # 1. Collect file trees
        data_tree = collect_file_tree(paper_dir / "data")
        repl_tree = collect_file_tree(paper_dir / "replication_package")
        zip_tree = collect_zip_tree(paper_id, collection)

        # 2. Find READMEs
        readmes = find_readmes(paper_dir)
        text_readmes = []
        pdf_images = []

        for readme_path, readme_type in readmes:
            if readme_type == "text":
                content = read_text_readme(readme_path)
                rel = readme_path.relative_to(paper_dir)
                text_readmes.append(f"=== {rel} ===\n{content}")
            elif readme_type == "pdf":
                images = pdf_to_base64_images(readme_path)
                if images:
                    rel = readme_path.relative_to(paper_dir)
                    pdf_images.extend(images)
                    text_readmes.append(f"=== {rel} === (attached as images)")

        readme_text = "\n\n".join(text_readmes) if text_readmes else "(no README found)"

        # 3. Build prompt
        prompt = f"""Audit the replication package for paper: {paper_id}

## Original replication package (ZIP listing before data/code split):
{zip_tree}

## Pipeline data/ directory (files given to the replicator):
{data_tree}

## Pipeline replication_package/ directory (code, NOT given to replicator):
{repl_tree}

## README content:
{readme_text}

Classify this paper's data directory. Are there pre-computed outputs mixed
in with raw data? Is the raw data sufficient for replication?"""

        # 4. Call LLM (try vision first for PDF READMEs, fall back to text-only)
        try:
            resp = None
            if pdf_images:
                try:
                    content_parts = [{"type": "input_text", "text": prompt}]
                    for img in pdf_images:
                        content_parts.append({
                            "type": "input_image",
                            "image_url": f"data:{img['media_type']};base64,{img['base64']}",
                        })
                    resp = await client.responses.parse(
                        model="gpt-5-mini",
                        instructions=SYSTEM_PROMPT,
                        input=[{"type": "message", "role": "user", "content": content_parts}],
                        text_format=DataAuditResult,
                    )
                except Exception as e:
                    if "context_length" in str(e) or "400" in str(e):
                        log.warning(f"  {paper_id}: vision call too large, retrying text-only")
                    else:
                        raise

            if resp is None:
                # Text-only call (no images or vision failed)
                resp = await client.responses.parse(
                    model="gpt-5-mini",
                    instructions=SYSTEM_PROMPT,
                    input=prompt,
                    text_format=DataAuditResult,
                )

            result = resp.output_parsed
            # Ensure paper_id is set correctly
            result.paper_id = paper_id
            result.readme_found = len(readmes) > 0

            usage = getattr(resp, "usage", None)
            tokens = getattr(usage, "total_tokens", 0) if usage else 0
            log.info(
                f"  {paper_id}: sufficiency={result.data_sufficiency}, "
                f"outputs={result.has_output_files}, tokens={tokens}"
            )
            return result

        except Exception as e:
            log.error(f"  {paper_id}: LLM call failed: {e}")
            return DataAuditResult(
                paper_id=paper_id,
                has_raw_data=False,
                has_output_files=False,
                raw_data_paths=[],
                output_paths=[],
                data_sufficiency="insufficient",
                sufficiency_explanation=f"Audit failed: {e}",
                readme_found=len(readmes) > 0,
                notes=f"LLM error: {e}",
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main():
    parser = argparse.ArgumentParser(description="Audit replication package data directories")
    parser.add_argument("--only", choices=["postcutoff", "i4rep"], default=None,
                        help="Only audit one collection")
    parser.add_argument("--papers", nargs="*", default=None,
                        help="Only audit specific paper IDs")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Max concurrent API requests (default: 10)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results (default: data/)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        log.error("OPENAI_API_KEY not set")
        return

    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=api_key)

    output_json = Path(args.output_dir) / "audit_replication_data.json" if args.output_dir else OUTPUT_JSON
    output_csv = Path(args.output_dir) / "audit_replication_data.csv" if args.output_dir else OUTPUT_CSV

    # Load existing results for resumption
    existing: dict[str, dict] = {}
    if output_json.exists():
        try:
            existing = {r["paper_id"]: r for r in json.loads(output_json.read_text())}
            log.info(f"Loaded {len(existing)} existing results for resumption")
        except Exception:
            pass

    # Collect papers to audit
    papers: list[tuple[str, Path, str]] = []  # (paper_id, paper_dir, collection)

    if args.only != "i4rep" and POSTCUTOFF_PAPERS.exists():
        for d in sorted(POSTCUTOFF_PAPERS.iterdir()):
            if d.is_dir():
                papers.append((d.name, d, "postcutoff"))

    if args.only != "postcutoff" and I4REP_PAPERS.exists():
        for d in sorted(I4REP_PAPERS.iterdir()):
            if d.is_dir():
                papers.append((d.name, d, "i4rep"))

    if args.papers:
        papers = [(pid, pdir, col) for pid, pdir, col in papers if pid in args.papers]

    # Filter out already-audited papers
    to_audit = [(pid, pdir, col) for pid, pdir, col in papers if pid not in existing]
    log.info(f"Papers to audit: {len(to_audit)} (skipping {len(papers) - len(to_audit)} already done)")

    if not to_audit:
        log.info("All papers already audited")
    else:
        # Run audits in parallel
        semaphore = asyncio.Semaphore(args.concurrency)
        tasks = [
            audit_paper(client, pid, pdir, col, semaphore)
            for pid, pdir, col in to_audit
        ]
        results = await asyncio.gather(*tasks)

        # Merge with existing
        for r in results:
            existing[r.paper_id] = r.model_dump()

    # Write JSON
    all_results = list(existing.values())
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(all_results, indent=2, default=str))
    log.info(f"Wrote {len(all_results)} results to {output_json}")

    # Write CSV
    if all_results:
        fieldnames = [
            "paper_id", "collection", "has_raw_data", "has_output_files",
            "data_sufficiency", "sufficiency_explanation", "readme_found",
            "raw_data_paths", "output_paths", "notes",
        ]
        # Add collection column
        paper_collections = {pid: col for pid, _, col in papers}
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in sorted(all_results, key=lambda x: x["paper_id"]):
                row = {k: r.get(k, "") for k in fieldnames}
                row["collection"] = paper_collections.get(r["paper_id"], "")
                # Flatten lists to semicolon-separated
                if isinstance(row["raw_data_paths"], list):
                    row["raw_data_paths"] = "; ".join(row["raw_data_paths"])
                if isinstance(row["output_paths"], list):
                    row["output_paths"] = "; ".join(row["output_paths"])
                writer.writerow(row)
        log.info(f"Wrote CSV to {output_csv}")

    # Summary
    from collections import Counter
    sufficiency = Counter(r.get("data_sufficiency", "?") for r in all_results)
    has_outputs = sum(1 for r in all_results if r.get("has_output_files"))
    print(f"\n{'=' * 60}")
    print(f"AUDIT SUMMARY ({len(all_results)} papers)")
    print(f"{'=' * 60}")
    print(f"\nData sufficiency:")
    for k in ["sufficient", "partial", "insufficient", "confidential"]:
        print(f"  {k:<15} {sufficiency.get(k, 0)}")
    print(f"\nHas output files in data/: {has_outputs}/{len(all_results)}")
    print(f"\nResults: {output_json}")
    print(f"         {output_csv}")


if __name__ == "__main__":
    asyncio.run(main())
