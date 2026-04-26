#!/usr/bin/env python3
"""Audit replication packages: classify every file and identify replication data.

Enhanced version of audit_replication_data.py. For each file in the package,
the model assigns a category (code, support, raw_data, intermediary_data,
final_data, results). Then it identifies which data files are needed to
replicate the paper's results (not just raw data — includes intermediary/final
data if required). Data sufficiency classification is preserved.

Usage:
  python scripts/audit_replication_data_v2.py                     # all papers
  python scripts/audit_replication_data_v2.py --only postcutoff   # one collection
  python scripts/audit_replication_data_v2.py --only i4rep        # one collection
  python scripts/audit_replication_data_v2.py --papers 209827 209484  # specific papers
  python scripts/audit_replication_data_v2.py --concurrency 5     # limit parallel requests
  python scripts/audit_replication_data_v2.py --force              # re-audit all (ignore cache)

  # Audit a single replication package (zip or directory) at any path:
  python scripts/audit_replication_data_v2.py --package /path/to/package.zip
  python scripts/audit_replication_data_v2.py --package ~/Downloads/my_paper_repo \\
      --paper-id my_paper
"""

import argparse
import asyncio
import base64
import csv
import json
import logging
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Literal

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
PRECUTOFF_PAPERS = PROJECT_ROOT / "data" / "precutoff" / "papers"
I4REP_PAPERS = PROJECT_ROOT / "data" / "i4replicate" / "papers"
POSTCUTOFF_ZIPS = Path(os.environ.get(
    "POSTCUTOFF_ZIPS_DIR",
    str(PROJECT_ROOT / "data" / "openicpsr_aea" / "post_cutoff_sample"),
))
PRECUTOFF_ZIPS = Path(os.environ.get(
    "PRECUTOFF_ZIPS_DIR",
    str(PROJECT_ROOT / "data" / "openicpsr_aea" / "pre_cutoff_sample"),
))
I4REP_ZIPS = PROJECT_ROOT / "data" / "i4replicate" / "replication_packages"

OUTPUT_JSON = PROJECT_ROOT / "data" / "audit_replication_data_v2.json"
OUTPUT_CSV = PROJECT_ROOT / "data" / "audit_replication_data_v2.csv"

# ---------------------------------------------------------------------------
# Structured output model
# ---------------------------------------------------------------------------


class FileClassification(BaseModel):
    """Classification of a single file in the replication package."""

    path: str = Field(description="File path relative to the package root")
    category: Literal["code", "support", "raw_data", "intermediary_data", "final_data", "results"] = Field(
        description="File category: "
        "'code' = analysis scripts (.do, .R, .py, .m, .sh); "
        "'support' = documentation (README, codebook, questionnaire, license); "
        "'raw_data' = original untouched source data (survey microdata, admin records, census); "
        "'intermediary_data' = data that has been partially processed/cleaned but is not the "
        "final analysis dataset (e.g. merged files, cleaned subsets); "
        "'final_data' = fully preprocessed analysis-ready datasets that are direct inputs to "
        "regression/estimation code; "
        "'results' = pre-computed outputs (regression tables, figures, logs, .tex/.rtf output)"
    )
    reason: str = Field(description="Brief reason for this classification")


class DataAuditResultV2(BaseModel):
    """Enhanced classification of a replication package."""

    paper_id: str = Field(description="Paper identifier")
    file_classifications: list[FileClassification] = Field(
        description="Classification of every file in the package"
    )
    replication_data_paths: list[str] = Field(
        description="Paths (relative to package root) of the MINIMUM set of data "
        "files sufficient to replicate from scratch. Prefer raw_data over "
        "intermediary/final data when code exists to derive them. Include "
        "final_data only when upstream data or construction code is missing. "
        "Do NOT include results/outputs, code, or support files. "
        "Do NOT include duplicate copies. Use individual file paths."
    )
    data_sufficiency: Literal["sufficient", "partial", "insufficient", "confidential"] = Field(
        description="Whether the available data is sufficient to replicate all tables/figures. "
        "'sufficient': all data needed to reproduce results is present. "
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
You are auditing replication packages for economics/social science research papers.
Your task is to classify every file in the package and identify which data files
are needed for replication.

## File categories

Classify each file into exactly one category:

1. **code** — Analysis scripts and programs (.do, .R, .py, .m, .sh, .jl, .sas, .ado, etc.)
2. **support** — Documentation and metadata (README, codebook, questionnaire, license, .bib,
   data dictionary, variable descriptions, project config files)
3. **raw_data** — Original, untouched source data. This is data as collected or obtained from
   an external source, before any manipulation by the authors. Examples: survey microdata,
   administrative records, census extracts, weather station data, satellite imagery,
   web-scraped data, experimental data as recorded.
4. **intermediary_data** — Data that has been partially processed, cleaned, or merged by the
   authors but is NOT the final analysis dataset. Examples: cleaned subsets, merged panels,
   geocoded records, imputed datasets, sample restrictions applied.
5. **final_data** — Fully preprocessed, analysis-ready datasets that are the direct inputs
   to regression/estimation code. These are the files that the analysis scripts load to
   produce tables and figures. Often a single .dta/.csv/.rds file per analysis.
6. **results** — Pre-computed outputs produced by analysis code. Examples: regression output
   tables (.tex, .rtf, .xlsx), figures (.pdf, .png, .eps), saved estimation objects (.RDS,
   .rda), log files (.log, .smcl), formatted results.

## Identifying replication data

After classifying files, identify the MINIMUM set of data files sufficient to
replicate the paper's results from scratch. Apply this priority order:

1. If raw_data is present AND the package contains code to construct
   intermediary/final data from it, include ONLY the raw_data. Do NOT include
   intermediary_data or final_data that can be derived from the raw data.
2. If raw_data is incomplete or missing but intermediary_data is available,
   include the intermediary_data (and any raw_data that is still needed).
3. Include final_data ONLY when neither raw nor intermediary data is available
   for that analysis step, or when the code to construct it from raw data is
   missing from the package.

The goal is to give the replicator the most upstream data possible, so they
reconstruct as much of the pipeline as possible rather than starting from
pre-computed datasets.

- Do NOT include results/outputs — these are what the replicator should reproduce
- Do NOT include code or support files
- Do NOT include duplicate copies of the same dataset

Think about what the analysis code actually loads as inputs. If the code loads a final
dataset, that's what the replicator needs. If the code starts from raw data and processes
it, the replicator needs the raw data. If some steps require intermediary data that can't
be reproduced from the raw data alone (e.g. because it involves proprietary processing),
include the intermediary data too.

## Data sufficiency

Assess whether the identified replication data is sufficient to reproduce ALL tables
and figures in the paper:
- "sufficient": All data needed to reproduce results is present in the package
- "partial": Some data present but key datasets are missing (e.g. one of several source
  datasets is confidential, or weather data is missing while mortality data is present)
- "insufficient": Very little usable data for replication
- "confidential": The README or structure indicates raw data is restricted/confidential;
  only derived/output data is provided

Pay close attention to whether ALL required data sources are present. A paper that uses
mortality data AND weather data AND population data is only "sufficient" if ALL three
are in the package — not just one of them.
"""


# ---------------------------------------------------------------------------
# Helpers (reused from v1)
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
    elif collection == "precutoff":
        zip_path = PRECUTOFF_ZIPS / f"{paper_id}-V1.zip"
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
    """Find all README files in a paper directory. Returns [(path, type)]."""
    candidates = []
    for path in sorted(paper_dir.rglob("*")):
        if not path.is_file():
            continue
        name = path.name.lower()
        if name.startswith("readme") or name == "codebook.pdf":
            ext = path.suffix.lower()
            priority = {".md": 0, ".txt": 1, ".pdf": 2, ".html": 3, ".htm": 3, ".rtf": 4, "": 1}
            candidates.append((priority.get(ext, 5), path, ext))

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
) -> DataAuditResultV2:
    """Audit a single paper's replication package using gpt-5-mini.

    Two layouts are supported:

    1. Pipeline layout (collection ∈ {postcutoff, precutoff, i4rep}):
       ``paper_dir`` contains ``data/`` and ``replication_package/`` subdirs
       that have already been split by ``setup_i4rep_batch.py``. The
       original (unsplit) ZIP listing is fetched from the corresponding
       collection's zips directory.
    2. Single-package layout (collection == "single"):
       ``paper_dir`` IS the unsplit replication package (or an extracted
       copy of one). No data/replication_package split has happened; the
       audit runs over the whole tree.
    """
    async with semaphore:
        log.info(f"Auditing {paper_id} ({collection})")

        # 1. Collect file trees
        if collection == "single":
            data_tree = "(no split — package is the whole directory)"
            repl_tree = "(no split — package is the whole directory)"
            zip_tree = collect_file_tree(paper_dir)
        else:
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

## Pipeline data/ directory (files currently given to the replicator):
{data_tree}

## Pipeline replication_package/ directory (code and other files, NOT given to replicator):
{repl_tree}

## README content:
{readme_text}

For this replication package:
1. Classify EVERY file listed above into one of the categories (code, support, raw_data,
   intermediary_data, final_data, results). Include files from both data/ and
   replication_package/ directories, using paths relative to the package root
   (e.g. "data/survey.dta", "replication_package/analysis.do").
2. Identify ALL data files needed to reproduce the paper's results. Include raw_data,
   intermediary_data, and final_data files that are required inputs — but NOT results.
3. Assess data sufficiency: is the identified replication data enough to reproduce
   all tables and figures?"""

        # 4. Call LLM
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
                        text_format=DataAuditResultV2,
                    )
                except Exception as e:
                    if "context_length" in str(e) or "400" in str(e):
                        log.warning(f"  {paper_id}: vision call too large, retrying text-only")
                    else:
                        raise

            if resp is None:
                resp = await client.responses.parse(
                    model="gpt-5-mini",
                    instructions=SYSTEM_PROMPT,
                    input=prompt,
                    text_format=DataAuditResultV2,
                )

            result = resp.output_parsed
            result.paper_id = paper_id
            result.readme_found = len(readmes) > 0

            usage = getattr(resp, "usage", None)
            tokens = getattr(usage, "total_tokens", 0) if usage else 0

            n_files = len(result.file_classifications)
            n_repl = len(result.replication_data_paths)
            cats = {}
            for fc in result.file_classifications:
                cats[fc.category] = cats.get(fc.category, 0) + 1
            cat_str = ", ".join(f"{k}={v}" for k, v in sorted(cats.items()))
            log.info(
                f"  {paper_id}: sufficiency={result.data_sufficiency}, "
                f"files={n_files} ({cat_str}), repl_data={n_repl}, tokens={tokens}"
            )
            return result

        except Exception as e:
            log.error(f"  {paper_id}: LLM call failed: {e}")
            return DataAuditResultV2(
                paper_id=paper_id,
                file_classifications=[],
                replication_data_paths=[],
                data_sufficiency="insufficient",
                sufficiency_explanation=f"Audit failed: {e}",
                readme_found=len(readmes) > 0,
                notes=f"LLM error: {e}",
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _resolve_single_package(package_path: Path, paper_id: str | None,
                            tmp_holder: list) -> tuple[str, Path]:
    """Resolve a --package argument to (paper_id, paper_dir).

    If ``package_path`` is a zip, it is extracted into a temp directory whose
    lifetime is tied to ``tmp_holder`` (the caller appends the
    TemporaryDirectory so it isn't garbage-collected mid-audit).
    """
    if not package_path.exists():
        raise FileNotFoundError(f"--package path not found: {package_path}")

    derived_id = package_path.stem
    if package_path.is_dir():
        return (paper_id or derived_id, package_path)

    if package_path.suffix.lower() == ".zip":
        tmp = tempfile.TemporaryDirectory(prefix="audit_pkg_")
        tmp_holder.append(tmp)
        extract_root = Path(tmp.name)
        with zipfile.ZipFile(package_path) as zf:
            zf.extractall(extract_root)
        # If the zip contains a single top-level directory, descend into it
        entries = list(extract_root.iterdir())
        if len(entries) == 1 and entries[0].is_dir():
            extract_root = entries[0]
        return (paper_id or derived_id, extract_root)

    raise ValueError(
        f"--package must be a directory or .zip (got: {package_path})"
    )


async def main():
    parser = argparse.ArgumentParser(description="Audit replication packages (v2: file-level classification)")
    parser.add_argument("--only", choices=["postcutoff", "precutoff", "i4rep"], default=None,
                        help="Only audit one collection")
    parser.add_argument("--papers", nargs="*", default=None,
                        help="Only audit specific paper IDs")
    parser.add_argument("--package", type=str, default=None,
                        help="Audit a single replication package at this path "
                             "(zip file or directory). Bypasses the collection-based "
                             "lookup; output is keyed by --paper-id (or the path stem).")
    parser.add_argument("--paper-id", type=str, default=None,
                        help="Paper ID to use when --package is supplied "
                             "(default: filename stem of the package).")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Max concurrent API requests (default: 10)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results (default: data/)")
    parser.add_argument("--force", action="store_true",
                        help="Re-audit all papers (ignore cached results)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        log.error("OPENAI_API_KEY not set")
        return

    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=api_key)

    output_json = Path(args.output_dir) / "audit_replication_data_v2.json" if args.output_dir else OUTPUT_JSON
    output_csv = Path(args.output_dir) / "audit_replication_data_v2.csv" if args.output_dir else OUTPUT_CSV

    # Load existing results for resumption
    existing: dict[str, dict] = {}
    if not args.force and output_json.exists():
        try:
            existing = {r["paper_id"]: r for r in json.loads(output_json.read_text())}
            log.info(f"Loaded {len(existing)} existing results for resumption")
        except Exception:
            pass

    # Collect papers to audit
    papers: list[tuple[str, Path, str]] = []  # (paper_id, paper_dir, collection)
    tmp_dirs: list = []  # holds TemporaryDirectory objects for --package mode

    if args.package:
        if args.only or args.papers:
            log.warning("--only / --papers are ignored when --package is supplied")
        pid, pdir = _resolve_single_package(Path(args.package).expanduser().resolve(),
                                            args.paper_id, tmp_dirs)
        papers.append((pid, pdir, "single"))
    else:
        if args.only not in ("i4rep", "precutoff") and POSTCUTOFF_PAPERS.exists():
            for d in sorted(POSTCUTOFF_PAPERS.iterdir()):
                if d.is_dir():
                    papers.append((d.name, d, "postcutoff"))

        if args.only not in ("postcutoff", "i4rep") and PRECUTOFF_PAPERS.exists():
            for d in sorted(PRECUTOFF_PAPERS.iterdir()):
                if d.is_dir():
                    papers.append((d.name, d, "precutoff"))

        if args.only not in ("postcutoff", "precutoff") and I4REP_PAPERS.exists():
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
        semaphore = asyncio.Semaphore(args.concurrency)
        tasks = [
            audit_paper(client, pid, pdir, col, semaphore)
            for pid, pdir, col in to_audit
        ]
        results = await asyncio.gather(*tasks)

        for r in results:
            existing[r.paper_id] = r.model_dump()

    # Write JSON
    all_results = list(existing.values())
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(all_results, indent=2, default=str))
    log.info(f"Wrote {len(all_results)} results to {output_json}")

    # Write CSV — one row per paper with summary columns
    if all_results:
        fieldnames = [
            "paper_id", "collection",
            "n_code", "n_support", "n_raw_data", "n_intermediary_data",
            "n_final_data", "n_results",
            "replication_data_paths",
            "data_sufficiency", "sufficiency_explanation",
            "readme_found", "notes",
        ]
        paper_collections = {pid: col for pid, _, col in papers}
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in sorted(all_results, key=lambda x: x["paper_id"]):
                cats = {}
                for fc in r.get("file_classifications", []):
                    cat = fc["category"] if isinstance(fc, dict) else fc.category
                    cats[cat] = cats.get(cat, 0) + 1
                repl_paths = r.get("replication_data_paths", [])
                if isinstance(repl_paths, list):
                    repl_paths = "; ".join(repl_paths)
                row = {
                    "paper_id": r["paper_id"],
                    "collection": paper_collections.get(r["paper_id"], ""),
                    "n_code": cats.get("code", 0),
                    "n_support": cats.get("support", 0),
                    "n_raw_data": cats.get("raw_data", 0),
                    "n_intermediary_data": cats.get("intermediary_data", 0),
                    "n_final_data": cats.get("final_data", 0),
                    "n_results": cats.get("results", 0),
                    "replication_data_paths": repl_paths,
                    "data_sufficiency": r.get("data_sufficiency", ""),
                    "sufficiency_explanation": r.get("sufficiency_explanation", ""),
                    "readme_found": r.get("readme_found", ""),
                    "notes": r.get("notes", ""),
                }
                writer.writerow(row)
        log.info(f"Wrote CSV to {output_csv}")

    # Write detailed file-level CSV
    file_csv = output_csv.with_name("audit_replication_data_v2_files.csv")
    with open(file_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["paper_id", "path", "category", "reason"])
        writer.writeheader()
        for r in sorted(all_results, key=lambda x: x["paper_id"]):
            for fc in r.get("file_classifications", []):
                if isinstance(fc, dict):
                    writer.writerow({
                        "paper_id": r["paper_id"],
                        "path": fc["path"],
                        "category": fc["category"],
                        "reason": fc["reason"],
                    })
                else:
                    writer.writerow({
                        "paper_id": r["paper_id"],
                        "path": fc.path,
                        "category": fc.category,
                        "reason": fc.reason,
                    })
    log.info(f"Wrote file-level CSV to {file_csv}")

    # Summary
    from collections import Counter
    sufficiency = Counter(r.get("data_sufficiency", "?") for r in all_results)
    all_cats = Counter()
    for r in all_results:
        for fc in r.get("file_classifications", []):
            cat = fc["category"] if isinstance(fc, dict) else fc.category
            all_cats[cat] += 1
    print(f"\n{'=' * 60}")
    print(f"AUDIT SUMMARY ({len(all_results)} papers)")
    print(f"{'=' * 60}")
    print(f"\nData sufficiency:")
    for k in ["sufficient", "partial", "insufficient", "confidential"]:
        print(f"  {k:<15} {sufficiency.get(k, 0)}")
    print(f"\nFile categories (total across all papers):")
    for k in ["code", "support", "raw_data", "intermediary_data", "final_data", "results"]:
        print(f"  {k:<20} {all_cats.get(k, 0)}")
    print(f"\nResults: {output_json}")
    print(f"         {output_csv}")
    print(f"         {file_csv}")


if __name__ == "__main__":
    asyncio.run(main())
