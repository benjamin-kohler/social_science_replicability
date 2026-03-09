#!/usr/bin/env python3
"""
Bulk download all replication packages from Harvard Dataverse collections.

Collections:
  - JPE  (Journal of Political Economy)           ~245 datasets
  - QJE  (Quarterly Journal of Economics)          ~401 datasets
  - AJPS (American Journal of Political Science)   ~814 datasets
  - APSR (American Political Science Review)       ~750 datasets
  - JOP  (Journal of Politics)                     ~1176 datasets

Usage:
    python scripts/download_dataverse_bulk.py --collections jpe qje
    python scripts/download_dataverse_bulk.py --collections all
    python scripts/download_dataverse_bulk.py --collections ajps --max-workers 4
"""

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent / "data" / "dataverse"
DATAVERSE_API = "https://dataverse.harvard.edu/api"
PER_PAGE = 25

COLLECTIONS = {
    "jpe":  {"subtree": "JPE",        "name": "Journal of Political Economy"},
    "qje":  {"subtree": "qje",        "name": "Quarterly Journal of Economics"},
    "ajps": {"subtree": "ajps",       "name": "American Journal of Political Science"},
    "apsr": {"subtree": "the_review", "name": "American Political Science Review"},
    "jop":  {"subtree": "jop",        "name": "Journal of Politics"},
}

# ── HTTP session ──────────────────────────────────────────────────────────────
def make_session():
    s = requests.Session()
    retries = Retry(total=5, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers["User-Agent"] = "ReplicationScraper/1.0 (academic research)"
    return s


# ── List all datasets in a collection ─────────────────────────────────────────
def list_datasets(session: requests.Session, subtree: str) -> list[dict]:
    """Paginate through Dataverse search API and return all dataset metadata."""
    datasets = []
    start = 0

    while True:
        resp = session.get(
            f"{DATAVERSE_API}/search",
            params={"q": "*", "subtree": subtree, "type": "dataset", "per_page": PER_PAGE, "start": start},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        items = data["items"]
        total = data["total_count"]

        for item in items:
            datasets.append({
                "global_id": item["global_id"],
                "name": item.get("name", ""),
                "url": item.get("url", ""),
                "published_at": item.get("published_at", ""),
                "file_count": item.get("fileCount", 0),
            })

        start += PER_PAGE
        if start >= total:
            break
        time.sleep(0.3)

    return datasets


# ── Download a single dataset ─────────────────────────────────────────────────
def download_dataset(session: requests.Session, dataset: dict, out_dir: Path) -> dict:
    """Download all files for a dataset as a ZIP bundle."""
    global_id = dataset["global_id"]
    # global_id looks like "doi:10.7910/DVN/ABCDEF" — extract the short ID
    short_id = global_id.split("/")[-1] if "/" in global_id else global_id
    safe_id = global_id.replace(":", "_").replace("/", "_")

    zip_path = out_dir / f"{safe_id}.zip"
    if zip_path.exists():
        return {"global_id": global_id, "status": "exists", "path": str(zip_path)}

    # Download dataset as ZIP via access API
    url = f"{DATAVERSE_API}/access/dataset/:persistentId?persistentId={global_id}"
    try:
        resp = session.get(url, timeout=600, stream=True)
        if resp.status_code == 200:
            with open(zip_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
            size_mb = zip_path.stat().st_size / (1024 * 1024)
            return {"global_id": global_id, "status": "downloaded", "path": str(zip_path), "size_mb": round(size_mb, 1)}
        elif resp.status_code == 403:
            return {"global_id": global_id, "status": "restricted", "path": ""}
        else:
            return {"global_id": global_id, "status": f"http_{resp.status_code}", "path": ""}
    except requests.exceptions.Timeout:
        return {"global_id": global_id, "status": "timeout", "path": ""}
    except Exception as e:
        return {"global_id": global_id, "status": f"error: {e}", "path": ""}


# ── Process one collection ────────────────────────────────────────────────────
def load_done_ids(done_dir: Path | None, collection_key: str) -> set[str]:
    """Load IDs of already-transferred datasets from a done directory.

    Looks for ZIP files named like doi_10.7910_DVN_ABCDEF.zip and converts
    back to global_id format (doi:10.7910/DVN/ABCDEF).
    """
    if not done_dir:
        return set()
    coll_dir = done_dir / collection_key
    if not coll_dir.exists():
        return set()
    done = set()
    for f in coll_dir.iterdir():
        if f.suffix == ".zip":
            # Reverse the sanitization: doi_10.7910_DVN_ABC -> doi:10.7910/DVN/ABC
            name = f.stem  # e.g. doi_10.7910_DVN_ABCDEF
            # First underscore separates "doi" from the rest
            parts = name.split("_", 1)
            if len(parts) == 2:
                global_id = parts[0] + ":" + parts[1].replace("_", "/", 2)
                done.add(global_id)
    return done


def process_collection(collection_key: str, max_workers: int, done_dir: Path | None = None):
    coll = COLLECTIONS[collection_key]
    subtree = coll["subtree"]
    coll_name = coll["name"]
    out_dir = BASE_DIR / collection_key
    out_dir.mkdir(parents=True, exist_ok=True)

    progress_file = out_dir / "progress.json"
    catalog_file = out_dir / "catalog.csv"

    print(f"\n{'='*72}")
    print(f"Collection: {coll_name} ({collection_key})")
    print(f"Output: {out_dir}")
    print(f"{'='*72}")

    # Step 1: List all datasets
    session = make_session()
    print(f"Listing datasets in '{subtree}' ...")
    datasets = list_datasets(session, subtree)
    print(f"Found {len(datasets)} datasets")

    # Save catalog
    with open(catalog_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["global_id", "name", "url", "published_at", "file_count"])
        writer.writeheader()
        writer.writerows(datasets)
    print(f"Catalog saved to {catalog_file.name}")

    # Load progress
    completed = set()
    if progress_file.exists():
        try:
            progress = json.loads(progress_file.read_text())
            completed = set(progress.get("completed", []))
            print(f"Resuming: {len(completed)} already processed")
        except json.JSONDecodeError:
            print(f"Warning: corrupted progress.json, starting fresh")

    # Load done IDs from already-transferred directory
    done_ids = load_done_ids(done_dir, collection_key)
    if done_ids:
        print(f"Skipping {len(done_ids)} already transferred to done-dir")
    skip_ids = completed | done_ids

    # Filter out already completed
    remaining = [d for d in datasets if d["global_id"] not in skip_ids]
    print(f"Remaining: {len(remaining)} datasets to download\n")

    if not remaining:
        print("Nothing to do.")
        return

    # Step 2: Download
    results = {"downloaded": 0, "exists": 0, "restricted": 0, "error": 0}
    errors = []

    def download_one(ds):
        # Each thread gets its own session
        thread_session = make_session()
        return download_dataset(thread_session, ds, out_dir)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(download_one, ds): ds for ds in remaining}

        for i, future in enumerate(as_completed(futures), 1):
            ds = futures[future]
            result = future.result()
            status = result["status"]

            # Bucket the status
            if status == "downloaded":
                results["downloaded"] += 1
                size_str = f" ({result.get('size_mb', '?')} MB)"
            elif status == "exists":
                results["exists"] += 1
                size_str = ""
            elif status == "restricted":
                results["restricted"] += 1
                size_str = ""
            else:
                results["error"] += 1
                errors.append(result)
                size_str = f" [{status}]"

            completed.add(ds["global_id"])

            # Print progress
            total = len(remaining)
            short_name = ds["name"][:60] + ("..." if len(ds["name"]) > 60 else "")
            print(f"  [{i}/{total}] {status:12s}{size_str}  {short_name}")

            # Save progress periodically
            if i % 10 == 0:
                progress_file.write_text(json.dumps({"completed": list(completed)}, indent=2))

    # Final save
    progress_file.write_text(json.dumps({"completed": list(completed)}, indent=2))

    # Save errors
    if errors:
        errors_file = out_dir / "errors.json"
        errors_file.write_text(json.dumps(errors, indent=2))

    print(f"\n--- {coll_name} Summary ---")
    print(f"  Downloaded:  {results['downloaded']}")
    print(f"  Already had: {results['exists']}")
    print(f"  Restricted:  {results['restricted']}")
    print(f"  Errors:      {results['error']}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Bulk download replication packages from Harvard Dataverse")
    parser.add_argument(
        "--collections", nargs="+", default=["all"],
        choices=list(COLLECTIONS.keys()) + ["all"],
        help="Which collections to download (default: all)",
    )
    parser.add_argument("--max-workers", type=int, default=3, help="Parallel download threads (default: 3)")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output base directory")
    parser.add_argument("--done-dir", type=str, default=None,
                        help="Directory with already-transferred data (e.g. Euler mount). "
                             "Datasets found here will be skipped.")
    args = parser.parse_args()

    global BASE_DIR
    if args.output_dir:
        BASE_DIR = Path(args.output_dir)

    done_dir = Path(args.done_dir) if args.done_dir else None
    collections = list(COLLECTIONS.keys()) if "all" in args.collections else args.collections

    print(f"Harvard Dataverse Bulk Downloader")
    print(f"Collections: {', '.join(collections)}")
    print(f"Workers: {args.max_workers}")
    print(f"Output: {BASE_DIR}")
    if done_dir:
        print(f"Done-dir: {done_dir}")

    for coll in collections:
        process_collection(coll, args.max_workers, done_dir)

    print(f"\n{'='*72}")
    print("All done.")


if __name__ == "__main__":
    main()
