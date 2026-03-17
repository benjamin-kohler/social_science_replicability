#!/usr/bin/env python3
"""Set up paper directories for benchmarking using audit-based data filtering.

Reads the audit CSV (from audit_replication_data.py) to determine which files
in a replication package are raw data vs pre-computed outputs. Only raw data
paths go into data/; the full unzipped package goes into replication_package/.

Directory layout per paper:
  {output_dir}/papers/{paper_id}/
    ├── paper.pdf              — must be added separately
    ├── data/                  — raw data files only (filtered by audit)
    ├── replication_package/   — full unzipped replication package
    └── metadata.json          — title, DOI, publication date, audit info

Usage:
  python scripts/setup_papers.py                           # all papers in audit
  python scripts/setup_papers.py --only postcutoff         # one collection
  python scripts/setup_papers.py --papers 209827 209484    # specific papers
  python scripts/setup_papers.py --force                   # overwrite existing
"""

import argparse
import csv
import json
import logging
import os
import re
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
# Paths (textlab defaults)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

POSTCUTOFF_PAPERS_DIR = PROJECT_ROOT / "data" / "postcutoff" / "papers"
I4REP_PAPERS_DIR = PROJECT_ROOT / "data" / "i4replicate" / "papers"

POSTCUTOFF_ZIPS = Path("/data/individual/benjamin/openicpsr_aea/post_cutoff_sample")
I4REP_ZIPS = PROJECT_ROOT / "data" / "i4replicate" / "replication_packages"

AUDIT_JSON = PROJECT_ROOT / "data" / "audit_replication_data.json"


def find_package(paper_id: str, collection: str) -> Path | None:
    """Locate the original package (ZIP file or directory) for a paper."""
    if collection == "postcutoff":
        # openICPSR format: {id}-V1.zip
        candidates = list(POSTCUTOFF_ZIPS.glob(f"{paper_id}-V*.zip"))
        if candidates:
            return candidates[0]
    else:
        # i4rep: try .zip first, then directory
        zip_path = I4REP_ZIPS / f"{paper_id}.zip"
        if zip_path.exists():
            return zip_path
        dir_path = I4REP_ZIPS / paper_id
        if dir_path.is_dir():
            return dir_path
    return None


def extract_zip(zip_path: Path, dest: Path) -> bool:
    """Extract a ZIP to dest. Returns True on success."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest)
        # Also extract inner ZIPs
        for inner in list(dest.rglob("*.zip")):
            inner_dest = inner.parent / inner.stem
            inner_dest.mkdir(exist_ok=True)
            try:
                with zipfile.ZipFile(inner, "r") as izf:
                    izf.extractall(inner_dest)
                inner.unlink()
            except zipfile.BadZipFile:
                pass
        return True
    except Exception as e:
        log.error(f"Failed to extract {zip_path}: {e}")
        return False


def matches_path_pattern(file_rel: str, patterns: list[str]) -> bool:
    """Check if a relative file path matches any of the audit's raw_data_paths.

    Patterns can be:
    - Directory paths ending in /: "Raw/", "Data/Raw/"
    - Specific files: "survey_data.dta", "data/census.csv"
    - Files within directories match if they're under a matching directory pattern
    """
    file_rel_lower = file_rel.lower()
    file_parts = Path(file_rel).parts

    for pattern in patterns:
        pattern_stripped = pattern.strip()
        if not pattern_stripped:
            continue
        pattern_lower = pattern_stripped.lower()

        if pattern_stripped.endswith("/"):
            # Directory pattern: match any file under this directory
            dir_name = pattern_lower.rstrip("/")
            # Check if any part of the path matches
            for i, part in enumerate(file_parts):
                if part.lower() == dir_name or part.lower().startswith(dir_name):
                    return True
            # Also check if the relative path starts with the pattern
            if file_rel_lower.startswith(dir_name):
                return True
        else:
            # File pattern: match by name or relative path
            if file_rel_lower == pattern_lower:
                return True
            if Path(file_rel).name.lower() == Path(pattern_stripped).name.lower():
                return True
            # Check if pattern appears as suffix of the path
            if file_rel_lower.endswith("/" + pattern_lower):
                return True
    return False


def walk_files(directory: Path):
    """Walk directory, yield (relative_path_str, absolute_path)."""
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__MACOSX"]
        for fname in files:
            if fname.startswith("."):
                continue
            abs_path = Path(root) / fname
            rel_path = abs_path.relative_to(directory)
            yield str(rel_path), abs_path


def setup_paper(
    paper_id: str,
    collection: str,
    audit: dict,
    output_dir: Path,
    force: bool = False,
) -> dict:
    """Set up a single paper directory.

    Returns status dict.
    """
    paper_dir = output_dir / paper_id
    result = {
        "paper_id": paper_id,
        "collection": collection,
        "status": "ok",
        "n_raw_data": 0,
        "n_repl_pkg": 0,
        "has_pdf": False,
        "message": "",
    }

    # Check existing
    if paper_dir.exists() and not force:
        has_data = (paper_dir / "data").exists()
        has_repl = (paper_dir / "replication_package").exists()
        if has_data and has_repl:
            result["status"] = "skipped"
            result["message"] = "already exists"
            result["has_pdf"] = (paper_dir / "paper.pdf").exists()
            result["n_raw_data"] = sum(1 for _ in (paper_dir / "data").rglob("*") if _.is_file())
            result["n_repl_pkg"] = sum(1 for _ in (paper_dir / "replication_package").rglob("*") if _.is_file())
            return result

    # Find package (ZIP or directory)
    pkg_path = find_package(paper_id, collection)
    if pkg_path is None:
        result["status"] = "error"
        result["message"] = "package not found"
        return result

    # Get raw data paths from audit
    raw_data_paths = audit.get("raw_data_paths", [])
    if isinstance(raw_data_paths, str):
        raw_data_paths = [p.strip() for p in raw_data_paths.split(";") if p.strip()]

    tmp_dir = None
    try:
        tmp_dir = Path(tempfile.mkdtemp(prefix="setup_papers_"))
        if pkg_path.is_file():
            # Extract ZIP to temp dir
            if not extract_zip(pkg_path, tmp_dir):
                result["status"] = "error"
                result["message"] = "extraction failed"
                return result
        else:
            # Directory package — may contain inner ZIPs that need extracting
            # Copy to temp dir first so we can extract in place
            for rel, abs_p in walk_files(pkg_path):
                dest = tmp_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(abs_p, dest)
            # Extract any inner ZIPs
            for inner in list(tmp_dir.rglob("*.zip")):
                inner_dest = inner.parent / inner.stem
                inner_dest.mkdir(exist_ok=True)
                try:
                    with zipfile.ZipFile(inner, "r") as izf:
                        izf.extractall(inner_dest)
                    inner.unlink()
                    # Also extract nested ZIPs inside the inner extraction
                    for nested in list(inner_dest.rglob("*.zip")):
                        nested_dest = nested.parent / nested.stem
                        nested_dest.mkdir(exist_ok=True)
                        try:
                            with zipfile.ZipFile(nested, "r") as nzf:
                                nzf.extractall(nested_dest)
                            nested.unlink()
                        except zipfile.BadZipFile:
                            pass
                except zipfile.BadZipFile:
                    pass
        source_dir = tmp_dir

        all_files = list(walk_files(source_dir))
        if not all_files:
            result["status"] = "error"
            result["message"] = "empty package"
            return result

        # Prepare paper directory
        if paper_dir.exists() and force:
            # Keep paper.pdf if it exists
            pdf_backup = None
            pdf_path = paper_dir / "paper.pdf"
            if pdf_path.exists():
                pdf_backup = Path(tempfile.mktemp(suffix=".pdf"))
                shutil.copy2(pdf_path, pdf_backup)
            shutil.rmtree(paper_dir)
            paper_dir.mkdir(parents=True)
            if pdf_backup:
                shutil.copy2(pdf_backup, paper_dir / "paper.pdf")
                pdf_backup.unlink()
        else:
            paper_dir.mkdir(parents=True, exist_ok=True)

        data_dir = paper_dir / "data"
        repl_dir = paper_dir / "replication_package"
        data_dir.mkdir(exist_ok=True)
        repl_dir.mkdir(exist_ok=True)

        # Copy full replication package (everything)
        for rel, abs_p in all_files:
            dest = repl_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(abs_p, dest)
            result["n_repl_pkg"] += 1

        # Copy only raw data files
        if raw_data_paths:
            for rel, abs_p in all_files:
                if matches_path_pattern(rel, raw_data_paths):
                    dest = data_dir / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(abs_p, dest)
                    result["n_raw_data"] += 1
        else:
            # No audit info — skip data dir (empty)
            result["message"] = "no raw_data_paths in audit"

        # Write metadata
        metadata = {
            "paper_id": paper_id,
            "collection": collection,
            "data_sufficiency": audit.get("data_sufficiency", "unknown"),
            "has_output_files": audit.get("has_output_files", False),
            "raw_data_paths": raw_data_paths,
            "output_paths": audit.get("output_paths", []),
        }
        (paper_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False)
        )

        result["has_pdf"] = (paper_dir / "paper.pdf").exists()

    except Exception as e:
        result["status"] = "error"
        result["message"] = str(e)
        log.error(f"Error processing {paper_id}: {e}")
    finally:
        if tmp_dir and tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Set up paper directories using audit-based data filtering."
    )
    parser.add_argument("--only", choices=["postcutoff", "i4rep"], default=None)
    parser.add_argument("--papers", nargs="*", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--audit-json", type=str, default=str(AUDIT_JSON))
    args = parser.parse_args()

    # Load audit results
    audit_path = Path(args.audit_json)
    if not audit_path.exists():
        log.error(f"Audit JSON not found: {audit_path}")
        log.error("Run scripts/audit_replication_data.py first.")
        return

    audit_list = json.loads(audit_path.read_text())
    audit_map = {r["paper_id"]: r for r in audit_list}
    log.info(f"Loaded audit for {len(audit_map)} papers")

    # Determine which papers to process
    papers_to_process: list[tuple[str, str, Path]] = []  # (paper_id, collection, output_dir)

    for paper_id, audit_entry in sorted(audit_map.items()):
        collection = audit_entry.get("collection", "")
        if not collection:
            # Infer from paper_id format
            if re.match(r"^\d+$", paper_id):
                collection = "postcutoff"
            else:
                collection = "i4rep"

        if args.only and args.only != collection:
            continue
        if args.papers and paper_id not in args.papers:
            continue

        if collection == "postcutoff":
            output_dir = POSTCUTOFF_PAPERS_DIR
        else:
            output_dir = I4REP_PAPERS_DIR

        papers_to_process.append((paper_id, collection, output_dir))

    log.info(f"Papers to process: {len(papers_to_process)}")

    # Process
    results = []
    for paper_id, collection, output_dir in papers_to_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        audit_entry = audit_map.get(paper_id, {})

        log.info(f"Setting up {paper_id} ({collection})")
        r = setup_paper(paper_id, collection, audit_entry, output_dir, force=args.force)
        results.append(r)

        status_icon = {"ok": "OK", "skipped": "SKIP", "error": "ERR"}.get(r["status"], "???")
        log.info(
            f"  [{status_icon}] raw_data={r['n_raw_data']} repl_pkg={r['n_repl_pkg']} "
            f"pdf={'Y' if r['has_pdf'] else 'N'} {r['message']}"
        )

    # Summary
    total = len(results)
    ok = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors = sum(1 for r in results if r["status"] == "error")
    has_pdf = sum(1 for r in results if r["has_pdf"])
    has_data = sum(1 for r in results if r["n_raw_data"] > 0)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total:        {total}")
    print(f"  Set up:       {ok}")
    print(f"  Skipped:      {skipped}")
    print(f"  Errors:       {errors}")
    print(f"  Have PDF:     {has_pdf}")
    print(f"  Have data:    {has_data}")
    print()

    if errors:
        print("Errors:")
        for r in results:
            if r["status"] == "error":
                print(f"  {r['paper_id']}: {r['message']}")
        print()

    no_data = [r for r in results if r["status"] == "ok" and r["n_raw_data"] == 0]
    if no_data:
        print(f"No raw data copied ({len(no_data)} papers):")
        for r in no_data:
            suf = audit_map.get(r["paper_id"], {}).get("data_sufficiency", "?")
            print(f"  {r['paper_id']}: sufficiency={suf} {r['message']}")
        print()


if __name__ == "__main__":
    main()
