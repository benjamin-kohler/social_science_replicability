#!/usr/bin/env python3
"""Set up i4replication paper directories for batch benchmarking.

Reads the i4replicate CSVs, extracts/organizes replication packages into
a standard directory layout under data/i4replicate/papers/{doi_slug}/.

Each paper directory gets:
  - data/                  — data files only
  - replication_package/   — code files and READMEs
  - paper.pdf              — if found in the package
  - metadata.json          — title, authors, journal, year, doi, perfect_replication

Usage:
  python scripts/setup_i4rep_batch.py
  python scripts/setup_i4rep_batch.py --dry-run
  python scripts/setup_i4rep_batch.py --force        # overwrite existing
  python scripts/setup_i4rep_batch.py --try-pdfs      # attempt PDF downloads
"""

import argparse
import csv
import json
import logging
import os
import shutil
import sys
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
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
I4REP_DIR = PROJECT_ROOT / "data" / "i4replicate"
PACKAGES_DIR = I4REP_DIR / "replication_packages"
PAPERS_DIR = I4REP_DIR / "papers"
PAPERS_CSV = I4REP_DIR / "successfully_replicated_papers.csv"
DOWNLOAD_CSV = I4REP_DIR / "download_results.csv"

# ---------------------------------------------------------------------------
# File classification
# ---------------------------------------------------------------------------
DATA_EXTENSIONS = {
    ".csv", ".dta", ".rds", ".rdata", ".rda", ".xlsx", ".xls",
    ".sav", ".shp", ".dbf", ".shx", ".prj", ".mat", ".tsv", ".tab",
    ".feather", ".parquet", ".json",
}

CODE_EXTENSIONS = {
    ".r", ".do", ".py", ".m", ".sas", ".jl",
    ".rmd", ".rproj", ".rnw", ".qmd",  # R project/notebook files
}

# Output/rendered files — treat as code artifacts, not data
OUTPUT_EXTENSIONS = {
    ".html", ".htm", ".docx", ".pptx", ".rtf",
}

# Files to skip entirely
SKIP_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".eps", ".tif", ".tiff",
    ".bmp", ".ico",
}

# DOI slugs to skip (no usable data)
SKIP_DOIS = {
    "10.1093_ej_ueac074",   # Teaching Norms — code only
    "10.1257_aer.20210867",  # Digital Addiction — massive package, no simple data
}


def doi_to_slug(doi: str) -> str:
    """Convert a DOI to a filesystem-safe slug (replace / with _)."""
    return doi.replace("/", "_")


def classify_file(filepath: str) -> str:
    """Classify a file as 'data', 'code', 'pdf', 'readme', or 'other'.

    Args:
        filepath: relative path within the extracted package

    Returns:
        One of: 'data', 'code', 'pdf', 'readme', 'other'
    """
    name = os.path.basename(filepath).lower()
    _, ext = os.path.splitext(name)

    # Skip directories, hidden files, __MACOSX junk
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
        return "code"  # rendered output, part of replication package

    if ext in SKIP_EXTENSIONS:
        return "skip"

    # .RData/.Rda (case-insensitive already handled above via .lower())
    # .log, .tex, .bib, .md files
    if ext in {".log", ".tex", ".bib", ".txt"}:
        return "code"  # treat as part of replication package

    if ext == ".md":
        return "readme"

    # Conservative: unknown files go to both
    return "both"


def extract_zip_to_dir(zip_path: Path, dest_dir: Path) -> bool:
    """Extract a zip file to dest_dir. Returns True on success."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
        return True
    except (zipfile.BadZipFile, Exception) as e:
        log.warning(f"Failed to extract {zip_path}: {e}")
        return False


def extract_inner_zips(directory: Path) -> Path:
    """If a directory contains only zip files (zenodo pattern), extract them.

    Some zenodo packages are directories containing inner .zip files.
    Extract all .zip files found inside the directory into a flat temp dir.

    Returns the directory with extracted contents (may be same or new temp dir).
    """
    contents = list(directory.iterdir())

    # Check if all files are archives
    zip_files = [f for f in contents if f.suffix.lower() == ".zip"]
    sevenz_files = [f for f in contents if f.suffix.lower() == ".7z"]
    other_files = [f for f in contents if f.suffix.lower() not in {".zip", ".7z"}]

    if not zip_files and not sevenz_files:
        # No inner archives — use directory as-is
        return directory

    if other_files:
        # Mix of archives and other files — extract zips alongside
        for zf in zip_files:
            extract_dir = directory / zf.stem
            extract_dir.mkdir(exist_ok=True)
            extract_zip_to_dir(zf, extract_dir)
        return directory

    # All contents are archives — extract into a temp dir
    tmp = Path(tempfile.mkdtemp(prefix="i4rep_inner_"))
    for zf in zip_files:
        extract_zip_to_dir(zf, tmp)

    if sevenz_files:
        log.warning(
            f"Found .7z files in {directory.name} — cannot auto-extract. "
            f"Please extract manually: {[f.name for f in sevenz_files]}"
        )

    return tmp


def walk_files(directory: Path):
    """Walk a directory and yield (relative_path, absolute_path) for all files."""
    for root, dirs, files in os.walk(directory):
        # Skip __MACOSX and hidden directories
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__MACOSX"]
        for fname in files:
            abs_path = Path(root) / fname
            rel_path = abs_path.relative_to(directory)
            yield str(rel_path), abs_path


def find_paper_pdf(files: list[tuple[str, Path]]) -> Path | None:
    """Find the most likely paper PDF from a list of files.

    Prefers files named 'paper.pdf' or containing 'manuscript', 'paper',
    or at the top level. Skips appendix/supplement PDFs.
    """
    pdfs = [(rel, abs_p) for rel, abs_p in files if abs_p.suffix.lower() == ".pdf"]
    if not pdfs:
        return None

    # Score each PDF
    scored = []
    for rel, abs_p in pdfs:
        name_lower = abs_p.name.lower()
        score = 0

        # Skip clearly non-paper PDFs
        if any(kw in name_lower for kw in ["appendix", "supplement", "codebook", "readme"]):
            score -= 10

        if name_lower == "paper.pdf":
            score += 20
        elif "paper" in name_lower:
            score += 10
        elif "manuscript" in name_lower:
            score += 10

        # Prefer files closer to root
        depth = rel.count(os.sep)
        score -= depth * 2

        # Prefer larger PDFs (more likely to be the actual paper)
        try:
            size = abs_p.stat().st_size
            if size > 500_000:  # > 500KB likely a real paper
                score += 5
        except OSError:
            pass

        scored.append((score, rel, abs_p))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, _, best_path = scored[0]

    # Only return if we have a reasonable candidate
    if best_score >= 0:
        return best_path
    return None


def setup_paper_dir(
    doi_slug: str,
    metadata: dict,
    package_path: Path,
    output_dir: Path,
    force: bool = False,
    dry_run: bool = False,
) -> dict:
    """Set up a single paper directory.

    Returns a status dict with keys: doi_slug, status, has_pdf, n_data, n_code, message
    """
    paper_dir = output_dir / doi_slug
    result = {
        "doi_slug": doi_slug,
        "status": "ok",
        "has_pdf": False,
        "n_data": 0,
        "n_code": 0,
        "message": "",
    }

    if paper_dir.exists() and not force:
        result["status"] = "skipped"
        result["message"] = "already exists (use --force to overwrite)"
        # Check what's already there
        result["has_pdf"] = (paper_dir / "paper.pdf").exists()
        data_dir = paper_dir / "data"
        if data_dir.exists():
            result["n_data"] = sum(1 for _ in data_dir.rglob("*") if _.is_file())
        return result

    if dry_run:
        result["status"] = "dry_run"
        result["message"] = "would be created"
        return result

    # Create temp extraction directory
    tmp_extract = None
    source_dir = None

    try:
        if package_path.is_file() and package_path.suffix.lower() == ".zip":
            # Dataverse/AJPS/APSR/JOP zip packages
            tmp_extract = Path(tempfile.mkdtemp(prefix="i4rep_"))
            if not extract_zip_to_dir(package_path, tmp_extract):
                result["status"] = "error"
                result["message"] = f"Failed to extract zip: {package_path.name}"
                return result
            source_dir = tmp_extract

        elif package_path.is_dir():
            # Zenodo/EJ/ReStud directory packages (may contain inner zips)
            source_dir = extract_inner_zips(package_path)
            if source_dir != package_path:
                tmp_extract = source_dir  # mark for cleanup
        else:
            result["status"] = "error"
            result["message"] = f"Unknown package format: {package_path}"
            return result

        # Collect all files
        all_files = list(walk_files(source_dir))

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
            # skip: do nothing

        # Find paper PDF
        paper_pdf = find_paper_pdf(pdf_files)

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

        # Copy code files + readmes + both
        for rel, abs_p in code_files + readme_files + both_files:
            dest = code_dest / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(abs_p, dest)

        # Also copy non-paper PDFs (codebooks etc.) to replication_package
        for rel, abs_p in pdf_files:
            if abs_p != paper_pdf:
                dest = code_dest / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(abs_p, dest)

        # Copy paper PDF
        if paper_pdf:
            shutil.copy2(paper_pdf, paper_dir / "paper.pdf")
            result["has_pdf"] = True

        # Write metadata
        meta_path = paper_dir / "metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

        result["n_data"] = sum(
            1 for _ in data_dest.rglob("*") if _.is_file()
        )
        result["n_code"] = sum(
            1 for _ in code_dest.rglob("*") if _.is_file()
        )

        if result["n_data"] == 0:
            result["status"] = "no_data"
            result["message"] = "No data files found in package"

    except Exception as e:
        result["status"] = "error"
        result["message"] = str(e)
        log.error(f"Error processing {doi_slug}: {e}")

    finally:
        # Clean up temp dirs
        if tmp_extract and tmp_extract.exists():
            try:
                shutil.rmtree(tmp_extract)
            except Exception:
                pass

    return result


def try_download_pdf(doi: str, paper_dir: Path) -> bool:
    """Attempt to download a paper PDF for the given DOI.

    Returns True if successful.
    """
    try:
        import urllib.request
        import urllib.error
        import re

        pdf_path = paper_dir / "paper.pdf"
        headers = {
            "User-Agent": "Mozilla/5.0 (Research Bot; replicability study)"
        }

        # SSRN
        if doi.startswith("10.2139/ssrn."):
            ssrn_id = doi.split(".")[-1]
            url = f"https://papers.ssrn.com/sol3/papers.cfm?abstract_id={ssrn_id}"
            log.info(f"  Trying SSRN: {url}")
            req = urllib.request.Request(url, headers=headers)
            try:
                resp = urllib.request.urlopen(req, timeout=30)
                html = resp.read().decode("utf-8", errors="replace")
                # Look for PDF download link
                match = re.search(
                    r'href="(https://papers\.ssrn\.com/sol3/Delivery\.cfm[^"]*)"',
                    html,
                )
                if match:
                    pdf_url = match.group(1)
                    req2 = urllib.request.Request(pdf_url, headers=headers)
                    resp2 = urllib.request.urlopen(req2, timeout=60)
                    pdf_path.write_bytes(resp2.read())
                    if pdf_path.stat().st_size > 10_000:
                        log.info(f"  Downloaded SSRN PDF for {doi}")
                        return True
                    pdf_path.unlink()
            except (urllib.error.URLError, TimeoutError):
                pass

        # NBER
        elif doi.startswith("10.3386/"):
            paper_id = doi.split("/")[-1]
            url = f"https://www.nber.org/system/files/working_papers/{paper_id}/{paper_id}.pdf"
            log.info(f"  Trying NBER: {url}")
            req = urllib.request.Request(url, headers=headers)
            try:
                resp = urllib.request.urlopen(req, timeout=60)
                pdf_path.write_bytes(resp.read())
                if pdf_path.stat().st_size > 10_000:
                    log.info(f"  Downloaded NBER PDF for {doi}")
                    return True
                pdf_path.unlink()
            except (urllib.error.URLError, TimeoutError):
                pass

        # Generic DOI resolution
        else:
            doi_url = f"https://doi.org/{doi}"
            log.info(f"  Trying DOI resolution: {doi_url}")
            req = urllib.request.Request(doi_url, headers=headers)
            try:
                resp = urllib.request.urlopen(req, timeout=30)
                landing_url = resp.url
                html = resp.read().decode("utf-8", errors="replace")
                # Look for PDF links
                patterns = [
                    r'href="([^"]*\.pdf)"',
                    r'href="([^"]*pdf[^"]*)"',
                    r'content="([^"]*\.pdf)"',
                ]
                for pat in patterns:
                    matches = re.findall(pat, html, re.IGNORECASE)
                    for match_url in matches:
                        if not match_url.startswith("http"):
                            # Make absolute
                            from urllib.parse import urljoin
                            match_url = urljoin(landing_url, match_url)
                        # Skip obviously wrong PDFs
                        if any(
                            kw in match_url.lower()
                            for kw in ["supplement", "appendix", "cover"]
                        ):
                            continue
                        try:
                            req2 = urllib.request.Request(match_url, headers=headers)
                            resp2 = urllib.request.urlopen(req2, timeout=60)
                            content = resp2.read()
                            if len(content) > 10_000 and content[:5] == b"%PDF-":
                                pdf_path.write_bytes(content)
                                log.info(f"  Downloaded PDF for {doi} from {match_url}")
                                return True
                        except (urllib.error.URLError, TimeoutError):
                            continue
                    if pdf_path.exists():
                        break
            except (urllib.error.URLError, TimeoutError):
                pass

    except Exception as e:
        log.warning(f"  PDF download failed for {doi}: {e}")

    return False


def load_paper_metadata(papers_csv: Path) -> dict[str, dict]:
    """Load paper metadata from successfully_replicated_papers.csv.

    Returns {doi: {title, authors, journal, year, doi, perfect_replication}}.
    """
    metadata = {}
    with open(papers_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doi = row["doi"].strip()
            metadata[doi] = {
                "title": row["title"].strip(),
                "authors": row["authors"].strip(),
                "journal": row["journal"].strip(),
                "year": row.get("year", "").strip(),
                "doi": doi,
                "perfect_replication": row["perfect_replication"].strip(),
            }
    return metadata


def load_download_results(download_csv: Path) -> dict[str, dict]:
    """Load download results.

    Returns {doi: {title, journal, repository, status, local_path}}.
    """
    results = {}
    with open(download_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doi = row["doi"].strip()
            results[doi] = {
                "title": row["title"].strip(),
                "journal": row.get("journal", "").strip(),
                "repository": row.get("repository", "").strip(),
                "status": row["status"].strip(),
                "local_path": row.get("local_path", "").strip(),
            }
    return results


def find_all_packages() -> dict[str, Path]:
    """Scan the replication_packages directory for all available packages.

    Returns {doi_slug: Path} where Path is a .zip file or a directory.
    """
    packages = {}
    if not PACKAGES_DIR.exists():
        return packages

    for entry in PACKAGES_DIR.iterdir():
        name = entry.name
        if entry.is_file() and name.endswith(".zip"):
            slug = name[:-4]  # remove .zip
            packages[slug] = entry
        elif entry.is_dir():
            packages[name] = entry

    return packages


def main():
    parser = argparse.ArgumentParser(
        description="Set up i4replication paper directories for batch benchmarking."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing paper directories",
    )
    parser.add_argument(
        "--try-pdfs", action="store_true",
        help="Attempt to download paper PDFs for papers that don't have one",
    )
    parser.add_argument(
        "--only", nargs="*", default=None,
        help="Only process these DOI slugs (e.g., 10.1111_ajps.12599)",
    )
    args = parser.parse_args()

    # Load CSVs
    log.info("Loading paper metadata...")
    paper_meta = load_paper_metadata(PAPERS_CSV)
    log.info(f"  {len(paper_meta)} papers in successfully_replicated_papers.csv")

    log.info("Loading download results...")
    download_results = load_download_results(DOWNLOAD_CSV)
    log.info(f"  {len(download_results)} entries in download_results.csv")

    # Find packages
    log.info("Scanning replication packages...")
    packages = find_all_packages()
    log.info(f"  {len(packages)} packages found")

    # Build work list: only packages we have AND have metadata for
    work_list = []
    for slug, pkg_path in sorted(packages.items()):
        if slug in SKIP_DOIS:
            log.info(f"  Skipping {slug} (in skip list)")
            continue

        if args.only and slug not in args.only:
            continue

        # Find metadata — convert slug back to DOI
        doi = slug.replace("_", "/", 1)  # first _ is the DOI separator
        # But DOIs can have multiple /, so we need a smarter approach
        # Try to find the DOI in our metadata
        matching_doi = None
        for meta_doi in paper_meta:
            if doi_to_slug(meta_doi) == slug:
                matching_doi = meta_doi
                break

        if not matching_doi:
            # Also check download_results
            for dl_doi in download_results:
                if doi_to_slug(dl_doi) == slug:
                    matching_doi = dl_doi
                    break

        if matching_doi and matching_doi in paper_meta:
            meta = paper_meta[matching_doi]
        elif matching_doi and matching_doi in download_results:
            # Partial metadata from download_results
            dl = download_results[matching_doi]
            meta = {
                "title": dl["title"],
                "authors": "",
                "journal": dl["journal"],
                "year": "",
                "doi": matching_doi,
                "perfect_replication": "unknown",
            }
        else:
            log.warning(f"  No metadata for {slug}, skipping")
            continue

        work_list.append((slug, matching_doi, meta, pkg_path))

    log.info(f"\n{len(work_list)} papers to process\n")

    # Process
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for slug, doi, meta, pkg_path in work_list:
        log.info(f"Processing {slug}...")
        log.info(f"  Title: {meta['title'][:80]}")

        result = setup_paper_dir(
            doi_slug=slug,
            metadata=meta,
            package_path=pkg_path,
            output_dir=PAPERS_DIR,
            force=args.force,
            dry_run=args.dry_run,
        )
        results.append(result)

        status_icon = {
            "ok": "OK",
            "skipped": "SKIP",
            "dry_run": "DRY",
            "no_data": "NO_DATA",
            "error": "ERR",
        }.get(result["status"], "???")

        log.info(
            f"  [{status_icon}] data={result['n_data']} code={result['n_code']} "
            f"pdf={'Y' if result['has_pdf'] else 'N'} {result['message']}"
        )

    # Try downloading PDFs
    if args.try_pdfs and not args.dry_run:
        need_pdfs = [
            r for r in results
            if r["status"] in ("ok", "skipped") and not r["has_pdf"]
        ]
        if need_pdfs:
            log.info(f"\nAttempting PDF downloads for {len(need_pdfs)} papers...\n")
            for r in need_pdfs:
                slug = r["doi_slug"]
                paper_dir = PAPERS_DIR / slug
                if not paper_dir.exists():
                    continue
                # Find DOI
                matching_doi = None
                for _, doi, _, _ in work_list:
                    if doi_to_slug(doi) == slug:
                        matching_doi = doi
                        break
                if matching_doi:
                    success = try_download_pdf(matching_doi, paper_dir)
                    if success:
                        r["has_pdf"] = True

    # Summary
    total = len(results)
    ok = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    no_data = sum(1 for r in results if r["status"] == "no_data")
    errors = sum(1 for r in results if r["status"] == "error")
    dry_runs = sum(1 for r in results if r["status"] == "dry_run")

    has_pdf = sum(1 for r in results if r["has_pdf"])
    needs_pdf = sum(
        1 for r in results
        if r["status"] in ("ok", "skipped") and not r["has_pdf"]
    )

    ready = sum(
        1 for r in results
        if r["status"] in ("ok", "skipped")
        and r["has_pdf"]
        and r["n_data"] > 0
    )
    need_pdf_only = sum(
        1 for r in results
        if r["status"] in ("ok", "skipped")
        and not r["has_pdf"]
        and r["n_data"] > 0
    )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total processed:     {total}")
    print(f"  Set up (new):        {ok}")
    print(f"  Skipped (existing):  {skipped}")
    print(f"  No data files:       {no_data}")
    print(f"  Errors:              {errors}")
    if dry_runs:
        print(f"  Dry run:             {dry_runs}")
    print()
    print(f"  Ready to benchmark (have PDF + data): {ready}")
    print(f"  Need PDF only (have data):            {need_pdf_only}")
    print(f"  Have PDF:            {has_pdf}")
    print(f"  Need PDF:            {needs_pdf}")
    print()

    # List papers needing PDFs
    if needs_pdf > 0:
        print("Papers needing PDF download:")
        for r in results:
            if r["status"] in ("ok", "skipped") and not r["has_pdf"]:
                # Find DOI
                for _, doi, meta, _ in work_list:
                    if doi_to_slug(doi) == r["doi_slug"]:
                        print(f"  {r['doi_slug']}: {meta['title'][:60]}")
                        break
        print()

    # List errors
    if errors > 0:
        print("Errors:")
        for r in results:
            if r["status"] == "error":
                print(f"  {r['doi_slug']}: {r['message']}")
        print()


if __name__ == "__main__":
    main()
