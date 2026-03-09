#!/usr/bin/env python3
"""
Bulk download all replication packages from Zenodo journal communities.

Communities:
  - restud-replication            (Review of Economic Studies)  ~451 records
  - ej-replication-repository     (The Economic Journal)        ~501 records
  - es-replication-repository     (Econometric Society:         ~210 records
                                   Econometrica, QE, TE)

Usage:
    python scripts/download_zenodo_bulk.py --communities restud ej
    python scripts/download_zenodo_bulk.py --communities all
    python scripts/download_zenodo_bulk.py --communities econometrica --max-workers 2
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
BASE_DIR = Path(__file__).resolve().parent.parent / "data" / "zenodo"
ZENODO_API = "https://zenodo.org/api"
PAGE_SIZE = 25

COMMUNITIES = {
    "restud": {
        "community_id": "restud-replication",
        "name": "Review of Economic Studies",
    },
    "ej": {
        "community_id": "ej-replication-repository",
        "name": "The Economic Journal",
    },
    "econometrica": {
        "community_id": "es-replication-repository",
        "name": "Econometric Society (Econometrica, QE, TE)",
    },
}

# Zenodo rate limit: 30 requests per 60 seconds
REQUEST_DELAY = 2.1  # seconds between API calls (~28 req/min, safely under limit)

# ── HTTP session ──────────────────────────────────────────────────────────────
def make_session():
    s = requests.Session()
    retries = Retry(total=5, backoff_factor=2.0, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers["User-Agent"] = "ReplicationScraper/1.0 (academic research)"
    return s


def rate_limited_get(session: requests.Session, url: str, **kwargs) -> requests.Response:
    """GET with rate limit awareness — backs off on 429."""
    resp = session.get(url, **kwargs)
    if resp.status_code == 429:
        retry_after = int(resp.headers.get("retry-after", 60))
        print(f"  Rate limited, waiting {retry_after}s ...")
        time.sleep(retry_after + 1)
        resp = session.get(url, **kwargs)
    return resp


# ── List all records in a community ───────────────────────────────────────────
def list_records(session: requests.Session, community_id: str) -> list[dict]:
    """Paginate through a Zenodo community and return all record metadata."""
    records = []
    page = 1

    while True:
        resp = rate_limited_get(
            session,
            f"{ZENODO_API}/communities/{community_id}/records",
            params={"size": PAGE_SIZE, "page": page, "sort": "newest"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        hits = data["hits"]["hits"]
        total = data["hits"]["total"]

        for hit in hits:
            files = hit.get("files", [])
            total_size = sum(f.get("size", 0) for f in files)
            records.append({
                "record_id": hit["id"],
                "doi": hit.get("doi", ""),
                "title": hit.get("metadata", {}).get("title", ""),
                "published": hit.get("metadata", {}).get("publication_date", ""),
                "access": hit.get("metadata", {}).get("access_right", ""),
                "num_files": len(files),
                "total_size_bytes": total_size,
                "files": files,
            })

        if len(records) >= total:
            break
        page += 1
        time.sleep(REQUEST_DELAY)

    return records


# ── Download a single record ──────────────────────────────────────────────────
def download_record(session: requests.Session, record: dict, out_dir: Path) -> dict:
    """Download all files for a Zenodo record into a subdirectory."""
    record_id = record["record_id"]
    record_dir = out_dir / str(record_id)
    record_dir.mkdir(parents=True, exist_ok=True)

    files = record.get("files", [])
    if not files:
        return {"record_id": record_id, "status": "no_files", "downloaded": 0, "path": ""}

    downloaded = 0
    skipped = 0

    for finfo in files:
        fname = finfo.get("key", f"file_{downloaded}")
        # Sanitize: replace path separators
        safe_fname = fname.replace("/", "_").replace("\\", "_")
        local_path = record_dir / safe_fname

        if local_path.exists():
            skipped += 1
            continue

        link = finfo.get("links", {}).get("self")
        if not link:
            continue

        try:
            resp = session.get(link, timeout=600, stream=True)
            if resp.status_code == 200:
                with open(local_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        f.write(chunk)
                downloaded += 1
            elif resp.status_code == 429:
                retry_after = int(resp.headers.get("retry-after", 60))
                print(f"    Rate limited on file download, waiting {retry_after}s ...")
                time.sleep(retry_after + 1)
                # Retry once
                resp = session.get(link, timeout=600, stream=True)
                if resp.status_code == 200:
                    with open(local_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=65536):
                            f.write(chunk)
                    downloaded += 1
        except requests.exceptions.Timeout:
            pass
        except Exception:
            pass

    total_size_mb = record["total_size_bytes"] / (1024 * 1024)

    if downloaded > 0 or skipped > 0:
        return {
            "record_id": record_id,
            "status": "downloaded" if downloaded > 0 else "exists",
            "downloaded": downloaded,
            "skipped": skipped,
            "size_mb": round(total_size_mb, 1),
            "path": str(record_dir),
        }
    else:
        return {"record_id": record_id, "status": "error", "downloaded": 0, "path": ""}


# ── Process one community ────────────────────────────────────────────────────
def load_done_ids(done_dir: Path | None, community_key: str) -> set[int]:
    """Load record IDs of already-transferred datasets from a done directory.

    Looks for subdirectories named by record ID (e.g. 16755528/).
    """
    if not done_dir:
        return set()
    comm_dir = done_dir / community_key
    if not comm_dir.exists():
        return set()
    done = set()
    for d in comm_dir.iterdir():
        if d.is_dir() and d.name.isdigit():
            done.add(int(d.name))
    return done


def process_community(community_key: str, max_workers: int, done_dir: Path | None = None):
    comm = COMMUNITIES[community_key]
    community_id = comm["community_id"]
    comm_name = comm["name"]
    out_dir = BASE_DIR / community_key
    out_dir.mkdir(parents=True, exist_ok=True)

    progress_file = out_dir / "progress.json"
    catalog_file = out_dir / "catalog.csv"

    print(f"\n{'='*72}")
    print(f"Community: {comm_name} ({community_id})")
    print(f"Output: {out_dir}")
    print(f"{'='*72}")

    # Step 1: List all records
    session = make_session()
    print(f"Listing records in '{community_id}' ...")
    records = list_records(session, community_id)
    print(f"Found {len(records)} records")

    total_size_gb = sum(r["total_size_bytes"] for r in records) / (1024**3)
    print(f"Total size: {total_size_gb:.1f} GB")

    # Save catalog (without files column)
    with open(catalog_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "record_id", "doi", "title", "published", "access", "num_files", "total_size_bytes",
        ])
        writer.writeheader()
        for r in records:
            row = {k: v for k, v in r.items() if k != "files"}
            writer.writerow(row)
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
    done_ids = load_done_ids(done_dir, community_key)
    if done_ids:
        print(f"Skipping {len(done_ids)} already transferred to done-dir")
    skip_ids = completed | done_ids

    # Filter out completed
    remaining = [r for r in records if r["record_id"] not in skip_ids]
    print(f"Remaining: {len(remaining)} records to download\n")

    if not remaining:
        print("Nothing to do.")
        return

    # Step 2: Download
    # Zenodo has strict rate limits (30 req/min), so we use fewer workers
    # and add delays between downloads
    effective_workers = min(max_workers, 2)  # cap at 2 for Zenodo
    results = {"downloaded": 0, "exists": 0, "no_files": 0, "error": 0}
    errors = []

    def download_one(rec):
        thread_session = make_session()
        result = download_record(thread_session, rec, out_dir)
        time.sleep(REQUEST_DELAY)  # respect rate limits
        return result

    with ThreadPoolExecutor(max_workers=effective_workers) as pool:
        futures = {pool.submit(download_one, rec): rec for rec in remaining}

        for i, future in enumerate(as_completed(futures), 1):
            rec = futures[future]
            result = future.result()
            status = result["status"]

            if status in ("downloaded", "exists"):
                results[status] += 1
                size_str = f" ({result.get('size_mb', '?')} MB)"
            elif status == "no_files":
                results["no_files"] += 1
                size_str = ""
            else:
                results["error"] += 1
                errors.append(result)
                size_str = f" [{status}]"

            completed.add(rec["record_id"])

            total = len(remaining)
            short_title = rec["title"][:55] + ("..." if len(rec["title"]) > 55 else "")
            print(f"  [{i}/{total}] {status:12s}{size_str}  {short_title}")

            # Save progress periodically
            if i % 10 == 0:
                progress_file.write_text(json.dumps({"completed": list(completed)}, indent=2))

    # Final save
    progress_file.write_text(json.dumps({"completed": list(completed)}, indent=2))

    if errors:
        errors_file = out_dir / "errors.json"
        errors_file.write_text(json.dumps(errors, indent=2))

    print(f"\n--- {comm_name} Summary ---")
    print(f"  Downloaded:  {results['downloaded']}")
    print(f"  Already had: {results['exists']}")
    print(f"  No files:    {results['no_files']}")
    print(f"  Errors:      {results['error']}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Bulk download replication packages from Zenodo communities")
    parser.add_argument(
        "--communities", nargs="+", default=["all"],
        choices=list(COMMUNITIES.keys()) + ["all"],
        help="Which communities to download (default: all)",
    )
    parser.add_argument("--max-workers", type=int, default=2, help="Parallel download threads (default: 2, capped at 2 for rate limits)")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output base directory")
    parser.add_argument("--done-dir", type=str, default=None,
                        help="Directory with already-transferred data (e.g. Euler mount). "
                             "Records found here will be skipped.")
    args = parser.parse_args()

    global BASE_DIR
    if args.output_dir:
        BASE_DIR = Path(args.output_dir)

    done_dir = Path(args.done_dir) if args.done_dir else None
    communities = list(COMMUNITIES.keys()) if "all" in args.communities else args.communities

    print(f"Zenodo Bulk Downloader")
    print(f"Communities: {', '.join(communities)}")
    print(f"Workers: {min(args.max_workers, 2)} (capped for rate limits)")
    print(f"Output: {BASE_DIR}")
    if done_dir:
        print(f"Done-dir: {done_dir}")

    for comm in communities:
        process_community(comm, args.max_workers, done_dir)

    print(f"\n{'='*72}")
    print("All done.")


if __name__ == "__main__":
    main()
