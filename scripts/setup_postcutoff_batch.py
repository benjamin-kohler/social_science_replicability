#!/usr/bin/env python3
"""Set up post-cutoff openICPSR paper directories for benchmarking.

Extracts replication package ZIPs into a standard directory layout
matching the i4replicate structure:

  {output_dir}/papers/{project_id}/
    ├── paper.pdf              — from local Downloads (scp'd separately)
    ├── data/                  — data files from replication package
    ├── replication_package/   — code, READMEs, and other files
    └── metadata.json          — title, DOI, project_id, publication date

Usage:
  python scripts/setup_postcutoff_batch.py

Or specify custom paths:
  python scripts/setup_postcutoff_batch.py \
    --zips-dir /path/to/zips \
    --output-dir /path/to/output \
    --force
"""

import argparse
import json
import logging
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults — override via env vars or --zips-dir / --output-dir CLI flags
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZIPS_DIR = os.environ.get(
    "POSTCUTOFF_ZIPS_DIR",
    str(PROJECT_ROOT / "data" / "openicpsr_aea" / "post_cutoff_sample"),
)
DEFAULT_OUTPUT_DIR = os.environ.get(
    "POSTCUTOFF_OUTPUT_DIR",
    str(PROJECT_ROOT / "data" / "postcutoff"),
)

# ---------------------------------------------------------------------------
# File classification (same rules as setup_i4rep_batch.py)
# ---------------------------------------------------------------------------
DATA_EXTENSIONS = {
    ".csv", ".dta", ".rds", ".rdata", ".rda", ".xlsx", ".xls",
    ".sav", ".shp", ".dbf", ".shx", ".prj", ".mat", ".tsv", ".tab",
    ".feather", ".parquet", ".json",
}

CODE_EXTENSIONS = {
    ".r", ".do", ".py", ".m", ".sas", ".jl",
    ".rmd", ".rproj", ".rnw", ".qmd",
}

OUTPUT_EXTENSIONS = {
    ".html", ".htm", ".docx", ".pptx", ".rtf",
}

SKIP_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".eps", ".tif", ".tiff",
    ".bmp", ".ico",
}


def classify_file(filepath: str) -> str:
    """Classify a file as 'data', 'code', 'pdf', 'readme', or 'other'."""
    name = os.path.basename(filepath).lower()
    _, ext = os.path.splitext(name)

    if name.startswith(".") or "__macosx" in filepath.lower():
        return "skip"
    if ext == ".pdf":
        return "pdf"
    if name.startswith("readme"):
        return "readme"
    if ext in DATA_EXTENSIONS:
        return "data"
    if ext in CODE_EXTENSIONS:
        return "code"
    if ext in OUTPUT_EXTENSIONS:
        return "code"
    if ext in SKIP_EXTENSIONS:
        return "skip"
    if ext in {".log", ".tex", ".bib", ".txt"}:
        return "code"
    if ext == ".md":
        return "readme"
    return "both"


def walk_files(directory: Path):
    """Walk a directory and yield (relative_path, absolute_path) for all files."""
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__MACOSX"]
        for fname in files:
            abs_path = Path(root) / fname
            rel_path = abs_path.relative_to(directory)
            yield str(rel_path), abs_path


def setup_paper_dir(
    project_id: str,
    zip_path: Path,
    output_dir: Path,
    metadata: dict,
    force: bool = False,
) -> dict:
    """Extract a replication package ZIP into the standard directory layout.

    Returns a status dict.
    """
    paper_dir = output_dir / "papers" / project_id
    result = {
        "project_id": project_id,
        "status": "ok",
        "has_pdf": False,
        "n_data": 0,
        "n_code": 0,
        "message": "",
    }

    if paper_dir.exists() and not force:
        result["status"] = "skipped"
        result["message"] = "already exists (use --force to overwrite)"
        result["has_pdf"] = (paper_dir / "paper.pdf").exists()
        data_dir = paper_dir / "data"
        if data_dir.exists():
            result["n_data"] = sum(1 for _ in data_dir.rglob("*") if _.is_file())
        return result

    tmp_extract = None
    try:
        # Extract ZIP
        tmp_extract = Path(tempfile.mkdtemp(prefix="postcutoff_"))
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_extract)

        # Also extract any inner ZIPs
        for inner_zip in list(tmp_extract.rglob("*.zip")):
            inner_dest = inner_zip.parent / inner_zip.stem
            inner_dest.mkdir(exist_ok=True)
            try:
                with zipfile.ZipFile(inner_zip, "r") as izf:
                    izf.extractall(inner_dest)
                inner_zip.unlink()
            except zipfile.BadZipFile:
                pass

        all_files = list(walk_files(tmp_extract))
        if not all_files:
            result["status"] = "error"
            result["message"] = "Package is empty after extraction"
            return result

        # Classify files
        data_files = []
        code_files = []
        readme_files = []
        pdf_files = []
        both_files = []

        for rel, abs_p in all_files:
            cat = classify_file(rel)
            if cat == "data":
                data_files.append((rel, abs_p))
            elif cat == "code":
                code_files.append((rel, abs_p))
            elif cat == "readme":
                readme_files.append((rel, abs_p))
            elif cat == "pdf":
                pdf_files.append((rel, abs_p))
            elif cat == "both":
                both_files.append((rel, abs_p))

        # Create paper directory
        if paper_dir.exists():
            shutil.rmtree(paper_dir)
        paper_dir.mkdir(parents=True)

        data_dest = paper_dir / "data"
        code_dest = paper_dir / "replication_package"
        data_dest.mkdir()
        code_dest.mkdir()

        # Copy data files
        for rel, abs_p in data_files + both_files:
            dest = data_dest / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(abs_p, dest)

        # Copy code + readmes + both
        for rel, abs_p in code_files + readme_files + both_files:
            dest = code_dest / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(abs_p, dest)

        # Copy PDFs to replication_package (not as paper.pdf)
        for rel, abs_p in pdf_files:
            dest = code_dest / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(abs_p, dest)

        # Write metadata
        (paper_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False)
        )

        result["n_data"] = sum(1 for _ in data_dest.rglob("*") if _.is_file())
        result["n_code"] = sum(1 for _ in code_dest.rglob("*") if _.is_file())

        if result["n_data"] == 0:
            result["status"] = "no_data"
            result["message"] = "No data files found in package"

    except Exception as e:
        result["status"] = "error"
        result["message"] = str(e)
        log.error(f"Error processing {project_id}: {e}")

    finally:
        if tmp_extract and tmp_extract.exists():
            try:
                shutil.rmtree(tmp_extract)
            except Exception:
                pass

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Set up post-cutoff openICPSR papers for benchmarking."
    )
    parser.add_argument(
        "--zips-dir", default=DEFAULT_ZIPS_DIR,
        help=f"Directory containing replication package ZIPs (default: {DEFAULT_ZIPS_DIR})",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for paper dirs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing paper directories",
    )
    parser.add_argument(
        "--only", nargs="*", default=None,
        help="Only process these project IDs",
    )
    args = parser.parse_args()

    zips_dir = Path(args.zips_dir)
    output_dir = Path(args.output_dir)

    if not zips_dir.exists():
        log.error(f"ZIPs directory does not exist: {zips_dir}")
        return

    # Find all ZIPs
    zips = sorted(zips_dir.glob("*-V*.zip"))
    log.info(f"Found {len(zips)} ZIPs in {zips_dir}")

    # Load metadata from catalog if available locally
    catalog = {}
    catalog_path = Path(__file__).resolve().parent.parent / "data" / "openicpsr_catalog.csv"
    if catalog_path.exists():
        import csv
        for row in csv.DictReader(open(catalog_path)):
            catalog[row["project_id"]] = row
        log.info(f"Loaded catalog with {len(catalog)} entries")

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for zip_path in zips:
        # Parse project ID from filename (e.g., 209827-V1.zip → 209827)
        import re
        m = re.match(r"(\d+)-V\d+\.zip", zip_path.name)
        if not m:
            log.warning(f"Skipping unrecognized filename: {zip_path.name}")
            continue

        project_id = m.group(1)
        if args.only and project_id not in args.only:
            continue

        cat = catalog.get(project_id, {})
        metadata = {
            "project_id": project_id,
            "doi": cat.get("doi", f"10.3886/E{project_id}V1"),
            "title": cat.get("title", ""),
            "publication_year": cat.get("publication_year", ""),
            "publication_month": cat.get("publication_month", ""),
            "source": "openicpsr_aea",
        }

        log.info(f"Processing {project_id}: {metadata['title'][:70]}")
        result = setup_paper_dir(
            project_id=project_id,
            zip_path=zip_path,
            output_dir=output_dir,
            metadata=metadata,
            force=args.force,
        )
        results.append(result)

        status_icon = {
            "ok": "OK", "skipped": "SKIP", "no_data": "NO_DATA", "error": "ERR",
        }.get(result["status"], "???")
        log.info(
            f"  [{status_icon}] data={result['n_data']} code={result['n_code']} "
            f"pdf={'Y' if result['has_pdf'] else 'N'} {result['message']}"
        )

    # Summary
    total = len(results)
    ok = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    no_data = sum(1 for r in results if r["status"] == "no_data")
    errors = sum(1 for r in results if r["status"] == "error")
    has_pdf = sum(1 for r in results if r["has_pdf"])
    ready = sum(
        1 for r in results
        if r["status"] in ("ok", "skipped") and r["has_pdf"] and r["n_data"] > 0
    )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total processed:     {total}")
    print(f"  Set up (new):        {ok}")
    print(f"  Skipped (existing):  {skipped}")
    print(f"  No data files:       {no_data}")
    print(f"  Errors:              {errors}")
    print(f"  Have PDF:            {has_pdf}")
    print(f"  Ready (PDF + data):  {ready}")
    print()

    if errors:
        print("Errors:")
        for r in results:
            if r["status"] == "error":
                print(f"  {r['project_id']}: {r['message']}")
        print()

    need_pdf = [r for r in results if r["status"] in ("ok", "skipped", "no_data") and not r["has_pdf"]]
    if need_pdf:
        print(f"Need paper.pdf ({len(need_pdf)}):")
        for r in need_pdf:
            print(f"  {r['project_id']}")
        print()


if __name__ == "__main__":
    main()
