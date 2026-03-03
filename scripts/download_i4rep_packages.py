#!/usr/bin/env python3
"""
Download replication packages for papers listed in successfully_replicated_papers.csv.

Repositories:
- Harvard Dataverse (AJPS, APSR, Journal of Politics): open API
- openICPSR (AER, AEJ:*, AER:Insights): requires auth -> log only
- Zenodo (EJ, ReStud, QJE, JPE): open API
- Unknown journal: infer from DOI prefix
"""

import csv
import os
import re
import sys
import time
import urllib.parse
from collections import Counter
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path("/Users/bkohler/Desktop/social_science_replicability/data/i4replicate")
CSV_PATH = BASE_DIR / "successfully_replicated_papers.csv"
PKG_DIR = BASE_DIR / "replication_packages"
RESULTS_CSV = BASE_DIR / "download_results.csv"

PKG_DIR.mkdir(parents=True, exist_ok=True)

# ── HTTP session with retry logic ──────────────────────────────────────────────
def make_session():
    s = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

SESSION = make_session()
TIMEOUT = 30
DOWNLOAD_TIMEOUT = 300

# ── Helpers ────────────────────────────────────────────────────────────────────
def sanitize_doi(doi: str) -> str:
    return re.sub(r'[^\w\-.]', '_', doi)


STOPWORDS = {
    'a','an','the','of','in','on','and','or','for','to','from','with','by',
    'at','as','is','it','its','are','was','were','be','been','do','does',
    'did','not','no','but','if','then','than','that','this','these','those',
    'how','what','when','where','which','who','whom','why','can','could',
    'may','might','must','shall','should','will','would','have','has','had',
    'having','replication','data'
}


def title_words(title: str) -> set:
    """Extract significant words (3+ chars) from title for validation."""
    words = set(w for w in re.findall(r'[a-z]+', title.lower()) if len(w) >= 3)
    return words - STOPWORDS


def validate_dataverse_match(paper_title: str, dataset_name: str) -> bool:
    """Check if a Dataverse result plausibly matches the paper."""
    pw = title_words(paper_title)
    dw = title_words(dataset_name)
    if not pw:
        return True
    overlap = pw & dw
    # Require at least 40% overlap
    return len(overlap) / len(pw) >= 0.4


def validate_zenodo_match(paper_title: str, record_title: str, files: list) -> bool:
    """
    Check if a Zenodo result is a real replication package for this paper.
    Filters out:
    1. Wrong paper matches (low title overlap)
    2. Paper-PDF-only records (single small PDF with generic name)
    """
    pw = title_words(paper_title)
    rw = title_words(record_title)
    if not pw:
        return True

    overlap = pw & rw
    ratio = len(overlap) / len(pw)

    # Require at least 50% title word overlap for Zenodo
    if ratio < 0.5:
        return False

    # Check if this looks like a replication package vs just a paper PDF
    if len(files) == 1:
        fname = files[0].get("key", "").lower()
        fsize = files[0].get("size", 0)
        # Single small PDF with generic name -> likely just the paper
        if fname.endswith(".pdf") and fsize < 5_000_000:  # < 5 MB
            generic_names = [
                "fulltext", "article", "paper", "manuscript", "preprint",
                "download", "accepted", "published",
            ]
            # Check if filename contains generic paper-PDF patterns
            if any(g in fname for g in generic_names):
                return False
            # Also skip if filename doesn't contain replication-related keywords
            replication_keywords = [
                "replication", "replic", "reppack", "code", "script",
                "data", "supplement", "appendix", "package",
            ]
            fname_base = os.path.splitext(fname)[0]
            if not any(k in fname_base for k in replication_keywords):
                # Could be a random unrelated PDF
                # Check more strictly: require higher overlap
                if ratio < 0.7:
                    return False

    return True


def classify_journal(journal: str, doi: str) -> str:
    j = journal.strip().lower()

    if "american journal of political science" in j:
        return "dataverse_ajps"
    if "american political science review" in j:
        return "dataverse_apsr"
    if "journal of politics" in j:
        return "dataverse_jop"
    if any(x in j for x in ["american economic review", "american economic journal"]):
        return "openicpsr"
    if "economic journal" in j:
        return "zenodo_ej"
    if "review of economic studies" in j:
        return "zenodo_restud"
    if "quarterly journal of economics" in j:
        return "zenodo_qje"
    if "journal of political economy" in j:
        return "zenodo_jpe"

    if not j:
        return infer_repo_from_doi(doi)

    return "unknown"


def infer_repo_from_doi(doi: str) -> str:
    d = doi.lower()
    if "10.1111/ajps" in d:
        return "dataverse_ajps"
    if "10.1017/s000305" in d:
        return "dataverse_apsr"
    if "10.1086/" in d:
        return "dataverse_jop_or_jpe"
    if "10.1257/" in d:
        return "openicpsr"
    if "10.1093/ej/" in d:
        return "zenodo_ej"
    if "10.1093/restud" in d:
        return "zenodo_restud"
    if "10.1093/qje" in d:
        return "zenodo_qje"
    if "10.3386/" in d:
        return "zenodo_nber"
    if "10.2139/" in d:
        return "unknown_ssrn"
    if "10.1163/" in d:
        return "zenodo_restud"
    return "unknown"


# ── Harvard Dataverse ──────────────────────────────────────────────────────────
DATAVERSE_BASE = "https://dataverse.harvard.edu"
DATAVERSE_SUBTREES = {
    "dataverse_ajps": "ajps",
    "dataverse_apsr": None,   # APSR datasets stored in various dataverses
    "dataverse_jop": "jop",
}


def search_dataverse(doi: str, title: str, subtree: str | None) -> str | None:
    """
    Search Harvard Dataverse for a dataset matching the paper.
    Returns the dataset persistent ID or None.

    Search strategy:
    1. Exact-match DOI (quoted) with subtree
    2. Exact-match DOI (quoted) globally
    3. Title keyword search in subtree
    4. Title keyword search globally
    """
    url = f"{DATAVERSE_BASE}/api/search"
    quoted_doi = f'"{doi}"'

    search_plans = []
    if subtree:
        search_plans.append({"q": quoted_doi, "subtree": subtree, "type": "dataset", "per_page": 5})
    search_plans.append({"q": quoted_doi, "type": "dataset", "per_page": 5})

    if title and subtree:
        words = re.findall(r'[A-Za-z]+', title)[:6]
        search_plans.append({"q": " ".join(words), "subtree": subtree, "type": "dataset", "per_page": 10})
    if title:
        words = re.findall(r'[A-Za-z]+', title)[:6]
        search_plans.append({"q": " ".join(words), "type": "dataset", "per_page": 10})

    for params in search_plans:
        try:
            resp = SESSION.get(url, params=params, timeout=TIMEOUT)
            if resp.status_code == 200:
                items = resp.json().get("data", {}).get("items", [])
                for item in items:
                    ds_name = item.get("name", "")
                    ds_id = item.get("global_id")
                    if ds_id and validate_dataverse_match(title, ds_name):
                        return ds_id
        except Exception as e:
            print(f"    Dataverse search error: {e}")

    return None


def download_dataverse_dataset(dataset_doi: str, local_path: Path) -> bool:
    encoded = urllib.parse.quote(dataset_doi, safe='')
    url = f"{DATAVERSE_BASE}/api/access/dataset/:persistentId?persistentId={encoded}"
    try:
        resp = SESSION.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True)
        if resp.status_code == 200:
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            size_mb = local_path.stat().st_size / (1024 * 1024)
            print(f"    Downloaded {size_mb:.1f} MB -> {local_path.name}")
            return True
        else:
            print(f"    Dataverse download HTTP {resp.status_code}")
            return False
    except Exception as e:
        print(f"    Dataverse download error: {e}")
        return False


def handle_dataverse(paper: dict, repo: str) -> dict:
    doi = paper["doi"]
    title = paper["title"]
    subtree = DATAVERSE_SUBTREES.get(repo)

    print(f"  Searching Dataverse (subtree={subtree or 'global'}) ...")
    dataset_doi = search_dataverse(doi, title, subtree)

    if not dataset_doi:
        print(f"  NOT FOUND on Dataverse")
        return {"repository": "harvard_dataverse", "status": "not_found", "local_path": ""}

    print(f"  Found dataset: {dataset_doi}")
    fname = sanitize_doi(doi) + ".zip"
    local_path = PKG_DIR / fname

    if local_path.exists():
        print(f"  Already downloaded: {fname}")
        return {"repository": "harvard_dataverse", "status": "downloaded", "local_path": str(local_path)}

    ok = download_dataverse_dataset(dataset_doi, local_path)
    if ok:
        return {"repository": "harvard_dataverse", "status": "downloaded", "local_path": str(local_path)}
    else:
        return {"repository": "harvard_dataverse", "status": "error", "local_path": ""}


# ── Zenodo ─────────────────────────────────────────────────────────────────────
ZENODO_API = "https://zenodo.org/api/records"


def search_zenodo(doi: str, title: str):
    """
    Search Zenodo for a record matching the paper.
    Returns (record_id, files_list) or (None, []).

    Strategy:
    1. Exact DOI search
    2. Title keyword search (validated strictly)
    """
    search_queries = [f'doi:"{doi}"']
    if title:
        words = re.findall(r'[A-Za-z]+', title)[:8]
        search_queries.append(" ".join(words))

    for query in search_queries:
        try:
            resp = SESSION.get(ZENODO_API, params={"q": query, "size": 5}, timeout=TIMEOUT)
            if resp.status_code == 200:
                hits = resp.json().get("hits", {}).get("hits", [])
                for hit in hits:
                    hit_title = hit.get("metadata", {}).get("title", "")
                    files = hit.get("files", [])
                    record_id = hit.get("id", "unknown")
                    if files and validate_zenodo_match(title, hit_title, files):
                        return record_id, files
        except Exception as e:
            print(f"    Zenodo search error: {e}")

    return None, []


def download_zenodo_files(doi: str, files: list, record_id) -> tuple:
    safe_doi = sanitize_doi(doi)
    out_dir = PKG_DIR / safe_doi
    out_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []
    for finfo in files:
        fname = finfo.get("key", f"file_{len(downloaded)}")
        link = finfo.get("links", {}).get("self") or finfo.get("links", {}).get("download")
        if not link:
            continue

        # Sanitize filename: replace path separators
        safe_fname = fname.replace("/", "_").replace("\\", "_")
        local_path = out_dir / safe_fname

        if local_path.exists():
            print(f"    Already exists: {safe_fname}")
            downloaded.append(str(local_path))
            continue
        try:
            resp = SESSION.get(link, timeout=DOWNLOAD_TIMEOUT, stream=True)
            if resp.status_code == 200:
                with open(local_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                size_mb = local_path.stat().st_size / (1024 * 1024)
                print(f"    Downloaded {safe_fname} ({size_mb:.1f} MB)")
                downloaded.append(str(local_path))
            else:
                print(f"    Failed to download {safe_fname}: HTTP {resp.status_code}")
        except Exception as e:
            print(f"    Error downloading {safe_fname}: {e}")

    return str(out_dir), len(downloaded)


def handle_zenodo(paper: dict, repo: str) -> dict:
    doi = paper["doi"]
    title = paper["title"]

    print(f"  Searching Zenodo ...")
    record_id, files = search_zenodo(doi, title)

    if not files:
        print(f"  NOT FOUND on Zenodo")
        return {"repository": "zenodo", "status": "not_found", "local_path": ""}

    print(f"  Found Zenodo record {record_id} with {len(files)} file(s)")
    out_dir, n = download_zenodo_files(doi, files, record_id)
    if n > 0:
        return {"repository": "zenodo", "status": "downloaded", "local_path": out_dir}
    else:
        return {"repository": "zenodo", "status": "error", "local_path": ""}


# ── Hybrid handler ─────────────────────────────────────────────────────────────
def handle_jop_or_jpe(paper: dict) -> dict:
    result = handle_dataverse(paper, "dataverse_jop")
    if result["status"] == "downloaded":
        return result
    return handle_zenodo(paper, "zenodo_jpe")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        papers = list(reader)

    print(f"Loaded {len(papers)} papers from {CSV_PATH.name}\n")

    results = []

    for i, paper in enumerate(papers, 1):
        title = paper["title"]
        doi = paper["doi"]
        journal = paper.get("journal", "")
        short_title = title[:70] + ("..." if len(title) > 70 else "")

        print(f"[{i}/{len(papers)}] {short_title}")
        print(f"  DOI: {doi}  |  Journal: {journal or '(empty)'}")

        repo = classify_journal(journal, doi)
        print(f"  Repository type: {repo}")

        if repo.startswith("dataverse_") and repo != "dataverse_jop_or_jpe":
            result = handle_dataverse(paper, repo)
        elif repo == "dataverse_jop_or_jpe":
            result = handle_jop_or_jpe(paper)
        elif repo == "openicpsr":
            print(f"  SKIPPED: requires openICPSR authentication")
            result = {"repository": "openicpsr", "status": "skipped", "local_path": ""}
        elif repo.startswith("zenodo_"):
            result = handle_zenodo(paper, repo)
        elif repo.startswith("unknown"):
            print(f"  SKIPPED: unknown repository / cannot determine source")
            result = {"repository": repo, "status": "skipped", "local_path": ""}
        else:
            print(f"  SKIPPED: unhandled repo type {repo}")
            result = {"repository": repo, "status": "skipped", "local_path": ""}

        results.append({
            "title": title,
            "doi": doi,
            "journal": journal,
            "repository": result["repository"],
            "status": result["status"],
            "local_path": result["local_path"],
        })

        print()
        time.sleep(0.5)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)

    status_counts = Counter(r["status"] for r in results)
    repo_counts = Counter(r["repository"] for r in results)

    print(f"\nBy status:")
    for s, n in status_counts.most_common():
        print(f"  {s:15s}  {n}")

    print(f"\nBy repository:")
    for r, n in repo_counts.most_common():
        print(f"  {r:25s}  {n}")

    openicpsr_papers = [r for r in results if r["repository"] == "openicpsr"]
    if openicpsr_papers:
        print(f"\n--- openICPSR papers (need authentication, {len(openicpsr_papers)} total) ---")
        for r in openicpsr_papers:
            print(f"  {r['doi']:40s}  {r['title'][:60]}")

    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "doi", "journal", "repository", "status", "local_path"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
