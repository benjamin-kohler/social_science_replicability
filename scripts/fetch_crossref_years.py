#!/usr/bin/env python3
"""Fetch publication years for the 48 sample papers from CrossRef.

Strategy:
  1. GET /works/{doi}  — canonical DOI lookup
  2. If that fails (404 or no year), fall back to a bibliographic search
     /works?query.title=...&query.container-title=... and pick the best match.

Results cached in a JSON file so repeated runs don't re-query.
"""
import csv, json, time, urllib.parse, urllib.request, sys
from pathlib import Path

CSV_PATH = Path("/Users/bkohler/Desktop/social_science_replicability/analysis_output/sample_papers.csv")
CACHE = Path("/Users/bkohler/Desktop/social_science_replicability/analysis_output/crossref_year_cache.json")
UA = "led-group-i4r (led-group@gess.ethz.ch)"  # polite identification

def _http_get(url: str, timeout: int = 15):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))

def _year_from_work(w: dict) -> int | None:
    for key in ("published-print", "issued", "published-online", "created"):
        v = w.get(key)
        if isinstance(v, dict):
            parts = v.get("date-parts") or []
            if parts and isinstance(parts[0], list) and parts[0]:
                try:
                    return int(parts[0][0])
                except (TypeError, ValueError):
                    continue
    return None

def fetch_by_doi(doi: str) -> tuple[int | None, str]:
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi, safe='/')}"
    try:
        js = _http_get(url)
        y = _year_from_work(js.get("message") or {})
        return (y, "doi") if y else (None, "doi-no-year")
    except Exception as e:
        return (None, f"doi-err:{type(e).__name__}")

def fetch_by_title(title: str, container: str | None = None) -> tuple[int | None, str]:
    params = {"query.bibliographic": title, "rows": 5}
    if container:
        params["query.container-title"] = container
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode(params)
    try:
        js = _http_get(url)
        items = (js.get("message") or {}).get("items") or []
    except Exception as e:
        return (None, f"search-err:{type(e).__name__}")
    if not items:
        return (None, "search-empty")
    # Prefer items whose title fuzzy-matches
    def _score(it):
        t = (it.get("title") or [""])[0].lower()
        return (title.lower() in t) or (t in title.lower())
    items.sort(key=lambda it: (_score(it), it.get("score", 0)), reverse=True)
    y = _year_from_work(items[0])
    return (y, "search") if y else (None, "search-no-year")

def main():
    rows = list(csv.DictReader(open(CSV_PATH)))
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}

    # Map AEA journal short names to canonical long names for better CrossRef match
    JOURNAL_FULL = {
        # keep original as-is for most
    }

    results = {}
    for i, r in enumerate(rows):
        slug = r["paper_slug"]
        doi = r["doi"]
        title = r["title"]
        journal = r["journal"]
        if slug in cache:
            results[slug] = cache[slug]
            continue
        year, src = fetch_by_doi(doi) if doi else (None, "no-doi")
        if year is None and title:
            time.sleep(0.3)
            year, src = fetch_by_title(title, journal)
        results[slug] = {"year": year, "source": src, "doi": doi, "title": title[:80]}
        print(f"[{i+1:2d}/{len(rows)}] {slug:38s} year={year}  src={src}")
        time.sleep(0.2)  # polite 5 req/sec
        if i % 5 == 0:
            CACHE.write_text(json.dumps(results, indent=2))

    CACHE.write_text(json.dumps(results, indent=2))
    # Merge back into CSV
    for r in rows:
        slug = r["paper_slug"]
        y = (results.get(slug) or {}).get("year")
        if y:
            r["year"] = str(y)
    # Preserve fieldnames
    out = CSV_PATH.with_name("sample_papers_with_years.csv")
    writer = csv.DictWriter(open(out, "w"), fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    print(f"\nWrote updated CSV → {out}")
    # Summary stats
    from collections import Counter
    yc = Counter(str(r["year"]) for r in rows)
    print(f"Year distribution: {dict(yc)}")
    missing = [r["paper_slug"] for r in rows if not str(r["year"]).strip()]
    print(f"Missing: {missing}")

if __name__ == "__main__":
    main()
