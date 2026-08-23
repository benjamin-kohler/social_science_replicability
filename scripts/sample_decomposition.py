#!/usr/bin/env python3
"""Build the sample decomposition table for the I4R replication analysis.

Starts from the full I4R Replicate universe and filters down step by step
to our final analysis sample, reporting counts and identifying papers lost
at each step.

Usage:
    python scripts/sample_decomposition.py
"""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEXTLAB_BASE = Path("/data/individual/benjamin/social_science_replicability/data/i4replicate")
LOCAL_BASE = PROJECT_ROOT / "data" / "i4replicate"
BASE = TEXTLAB_BASE if TEXTLAB_BASE.exists() else LOCAL_BASE

PAPERS_DIR = BASE / "papers"
RESULTS_DIR = BASE / "results"
ARCHIVE_DIR = BASE / "archive"
AUDIT_FILE = BASE.parent / "audit_replication_data_v2.json"
I4REP_CSV = BASE / "successfully_replicated_papers.csv"
COMP_REPRO_CSV = BASE / "Meta Database Public - Computational Reproducibility.csv"

SKIP_SUFFIXES = [".failed", ".bak", ".old", ".gap", ".earlystop", ".datafix",
                 ".f_", ".newdata", ".notprod"]

# Primary-language detection + naive LOC (line count) for a replication package.
# Extensions mapped here are the "code" we'll count; anything else is ignored.
_CODE_EXT_TO_LANG = {
    ".do": "Stata", ".ado": "Stata",
    ".R": "R", ".r": "R", ".Rmd": "R", ".rmd": "R",
    ".m": "MATLAB",
    ".py": "Python",
    ".jl": "Julia",
    ".sas": "SAS",
}


def detect_language_and_loc(repl_pkg: Path) -> tuple[str, int]:
    """Return (primary_language, total_lines_of_code) for a replication package.

    The primary language is the one with the most code files; if no single
    language holds >80% of files, we label the package ``mixed``. LOC is a
    naive ``wc -l`` sum across every file whose extension appears in
    ``_CODE_EXT_TO_LANG`` (not restricted to the primary language — gives a
    fair picture of codebase size for mixed packages).
    """
    if not repl_pkg.is_dir():
        return "unknown", 0
    by_lang: dict[str, list[Path]] = {}
    for f in repl_pkg.rglob("*"):
        if not f.is_file():
            continue
        lang = _CODE_EXT_TO_LANG.get(f.suffix)
        if lang:
            by_lang.setdefault(lang, []).append(f)
    if not by_lang:
        return "unknown", 0
    counts = {lang: len(fs) for lang, fs in by_lang.items()}
    total = sum(counts.values())
    top = max(counts, key=counts.get)
    primary = top if (counts[top] / total > 0.8) else ("mixed" if len(counts) > 1 else top)
    loc = 0
    for fs in by_lang.values():
        for f in fs:
            try:
                with open(f, "rb") as fh:
                    loc += sum(1 for _ in fh)
            except OSError:
                pass
    return primary, loc


def normalize_doi(doi: str) -> str:
    doi = doi.strip().lower()
    for prefix in ["https://doi.org/", "http://doi.org/", "doi:"]:
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
    return doi


def slug_to_dois(slug: str) -> list[str]:
    """Convert paper_slug to possible DOI variants for matching."""
    doi1 = slug.replace("_", "/", 1)
    candidates = [doi1.lower()]
    if slug.startswith("10.1093_"):
        candidates.append(slug.replace("_", "/", 2).lower())
    return candidates


def load_i4rep() -> tuple[dict[str, dict], int]:
    """Load I4R outcomes from both files.

    Uses the DOI-based success file as the primary key, enriched with
    computational reproducibility details. Also counts total unique papers
    from the comp repro file (which includes papers without DOIs).

    Returns:
        (dict keyed by normalized DOI, total unique papers in comp repro)
    """
    # Load DOI-based file
    outcomes = {}
    if I4REP_CSV.exists():
        with open(I4REP_CSV) as f:
            for row in csv.DictReader(f):
                doi = normalize_doi(row.get("doi", ""))
                if doi:
                    outcomes[doi] = row

    # Load comp repro file and merge details
    total_comp_repro = 0
    if COMP_REPRO_CSV.exists():
        title_to_doi = {}
        for doi, row in outcomes.items():
            title_to_doi[row.get("title", "").strip().lower()] = doi

        with open(COMP_REPRO_CSV) as f:
            comp_rows = list(csv.DictReader(f))

        # Count unique papers
        comp_titles = set()
        for row in comp_rows:
            comp_titles.add(row.get("paper_title", "").strip().lower())
        total_comp_repro = len(comp_titles)

        # Merge comp repro details into outcomes
        for row in comp_rows:
            title = row.get("paper_title", "").strip().lower()
            doi = title_to_doi.get(title)
            if doi and doi in outcomes:
                # Add comp repro fields
                outcomes[doi]["computational_reproduction"] = row.get("computational_reproduction", "")
                outcomes[doi]["why_not_perfect_repro"] = row.get("why_not_perfect_repro", "")
                if "perfect_reproduction" not in outcomes[doi] or not outcomes[doi]["perfect_reproduction"]:
                    outcomes[doi]["perfect_replication"] = row.get("perfect_reproduction", "")

    return outcomes, total_comp_repro


def load_audit() -> dict[str, dict]:
    """Load GPT audit data, keyed by paper_id."""
    if AUDIT_FILE.exists():
        with open(AUDIT_FILE) as f:
            return {r["paper_id"]: r for r in json.load(f)}
    return {}


def get_no_table_papers() -> set[str]:
    """Get paper slugs from the no-table archive."""
    nt_dir = ARCHIVE_DIR / "no-table-papers" / "papers"
    if nt_dir.is_dir():
        return {d.name for d in nt_dir.iterdir() if d.is_dir()}
    return set()


def build_doi_to_slug(slugs: set[str]) -> dict[str, str]:
    """Build DOI → slug lookup from a set of paper slugs."""
    mapping = {}
    for slug in slugs:
        for doi in slug_to_dois(slug):
            mapping[doi] = slug
    return mapping


def main():
    i4rep, total_comp_repro = load_i4rep()
    audit = load_audit()
    our_papers = {d.name for d in PAPERS_DIR.iterdir() if d.is_dir()}
    our_results = {d.name for d in RESULTS_DIR.iterdir()
                   if d.is_dir() and not d.name.startswith("_") and not d.name.endswith(".log")}
    no_table_papers = get_no_table_papers()
    all_known_papers = our_papers | no_table_papers

    # Build DOI lookups
    doi_to_slug = build_doi_to_slug(all_known_papers)
    audit_doi_lookup = {}
    for pid, a in audit.items():
        for d in slug_to_dois(pid):
            audit_doi_lookup[d] = (pid, a)

    # ── Step 0: I4R Universe ──
    total_i4rep_doi = len(i4rep)
    n_perfect = sum(1 for r in i4rep.values()
                    if r.get("perfect_replication", "").strip() == "Yes")
    n_not_perfect = total_i4rep_doi - n_perfect

    # ── Step 1: Perfect reproduction ──
    perfect_dois = {doi for doi, r in i4rep.items()
                    if r.get("perfect_replication", "").strip() == "Yes"}

    # ── Step 2: Audited with sufficient/partial data ──
    # Also check if paper is in no-table archive (means it was audited/downloaded)
    no_table_doi_lookup = {}
    for slug in no_table_papers:
        for d in slug_to_dois(slug):
            no_table_doi_lookup[d] = slug

    perfect_with_data = set()
    perfect_insuf_or_confid = set()
    perfect_not_audited = set()
    for doi in perfect_dois:
        if doi in audit_doi_lookup:
            pid, a = audit_doi_lookup[doi]
            suf = a.get("data_sufficiency", "")
            if suf in ["sufficient", "partial"]:
                perfect_with_data.add(doi)
            else:
                perfect_insuf_or_confid.add(doi)
        elif doi in no_table_doi_lookup:
            # Was downloaded and processed, but had no tables
            # Still count as "with data" so it gets dropped at step 3
            perfect_with_data.add(doi)
        else:
            perfect_not_audited.add(doi)

    # ── Step 3: Has extractable tables ──
    perfect_with_tables = set()
    perfect_no_tables = set()
    perfect_not_downloaded = set()
    for doi in perfect_with_data:
        if doi in doi_to_slug:
            slug = doi_to_slug[doi]
            if slug in no_table_papers:
                perfect_no_tables.add(doi)
            else:
                perfect_with_tables.add(doi)
        else:
            perfect_not_downloaded.add(doi)

    # ── Step 4: In final sample (has results) ──
    perfect_in_sample = set()
    perfect_missing = set()
    for doi in perfect_with_tables:
        slug = doi_to_slug[doi]
        if slug in our_results:
            perfect_in_sample.add(doi)
        else:
            perfect_missing.add(doi)

    # ── Not-perfect papers in our sample ──
    not_perfect_in_sample = []
    not_in_i4r = []
    for slug in sorted(our_results):
        matched_doi = None
        for d in slug_to_dois(slug):
            if d in i4rep:
                matched_doi = d
                break
        if matched_doi:
            if matched_doi not in perfect_dois:
                not_perfect_in_sample.append(slug)
        else:
            not_in_i4r.append(slug)

    # ── Print ──
    sep = "=" * 70
    print(sep)
    print("SAMPLE DECOMPOSITION (I4R → Our Sample)")
    print(sep)
    print()
    print(f"I4R Replicate universe:")
    print(f"  Unique papers (comp. repro. database):        {total_comp_repro}")
    print(f"  Papers with DOI (success file):               {total_i4rep_doi}")
    print(f"  Perfect computational reproduction:           {n_perfect}")
    print(f"  Not perfect reproduction:                     {n_not_perfect}")
    print()
    print(f"Step 1: Perfect reproduction                    {len(perfect_dois)}")
    print()
    print(f"Step 2: Sufficient/partial data (GPT audit)     {len(perfect_with_data)}")
    print(f"  Dropped: insufficient/confidential data:      {len(perfect_insuf_or_confid)}")
    print(f"  Dropped: not audited (not downloaded):        {len(perfect_not_audited)}")
    print()
    print(f"Step 3: Has extractable tables                  {len(perfect_with_tables)}")
    print(f"  Dropped: no tables (archived):                {len(perfect_no_tables)}")
    print(f"  Dropped: not downloaded:                      {len(perfect_not_downloaded)}")
    print()
    print(f"Step 4: In final sample (has results)           {len(perfect_in_sample)}")
    print(f"  Missing (no results):                         {len(perfect_missing)}")
    print()
    print(f"Additional papers in sample:")
    print(f"  Not-perfect reproduction:                     {len(not_perfect_in_sample)}")
    print(f"  Not in I4R:                                   {len(not_in_i4r)}")
    print()
    print(f"FINAL SAMPLE:                                   {len(our_results)}")
    print(f"  = {len(perfect_in_sample)} perfect + {len(not_perfect_in_sample)} not-perfect + {len(not_in_i4r)} not-in-I4R")

    # ── Details on dropped/extra papers ──
    if perfect_not_audited:
        print(f"\n--- Perfect but not audited ({len(perfect_not_audited)}) ---")
        for doi in sorted(perfect_not_audited):
            title = i4rep[doi].get("title", "?")[:70]
            print(f"  {doi}: {title}")

    if perfect_insuf_or_confid:
        print(f"\n--- Perfect but insufficient/confidential data ({len(perfect_insuf_or_confid)}) ---")
        for doi in sorted(perfect_insuf_or_confid):
            pid, a = audit_doi_lookup[doi]
            print(f"  {pid}: {a.get('data_sufficiency', '?')}")

    if perfect_no_tables:
        print(f"\n--- Perfect + data but no tables ({len(perfect_no_tables)}) ---")
        for doi in sorted(perfect_no_tables):
            slug = doi_to_slug.get(doi, "?")
            print(f"  {slug}")

    if perfect_missing:
        print(f"\n--- Perfect + data + tables but no results ({len(perfect_missing)}) ---")
        for doi in sorted(perfect_missing):
            slug = doi_to_slug.get(doi, "?")
            print(f"  {slug}")

    if not_perfect_in_sample:
        print(f"\n--- Not-perfect reproduction in sample ({len(not_perfect_in_sample)}) ---")
        for slug in not_perfect_in_sample:
            print(f"  {slug}")

    if not_in_i4r:
        print(f"\n--- Not in I4R ({len(not_in_i4r)}) ---")
        for slug in not_in_i4r:
            print(f"  {slug}")

    # ── Save CSV ──
    output_dir = BASE.parent / "analysis_output" if (BASE.parent / "analysis_output").exists() else PROJECT_ROOT / "analysis_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {"step": "I4R Replicate (comp. repro. database)", "n": total_comp_repro, "detail": ""},
        {"step": "I4R Replicate (with DOI)", "n": total_i4rep_doi, "detail": ""},
        {"step": "  Perfect reproduction", "n": n_perfect, "detail": ""},
        {"step": "  Not perfect reproduction", "n": n_not_perfect, "detail": ""},
        {"step": "Step 1: Perfect reproduction", "n": len(perfect_dois), "detail": ""},
        {"step": "Step 2: Sufficient/partial data", "n": len(perfect_with_data), "detail": ""},
        {"step": "  Dropped: insuff./confidential", "n": len(perfect_insuf_or_confid), "detail": ""},
        {"step": "  Dropped: not audited", "n": len(perfect_not_audited), "detail": ""},
        {"step": "Step 3: Has tables", "n": len(perfect_with_tables), "detail": ""},
        {"step": "  Dropped: no tables", "n": len(perfect_no_tables), "detail": "; ".join(sorted(doi_to_slug.get(d, d) for d in perfect_no_tables))},
        {"step": "  Dropped: not downloaded", "n": len(perfect_not_downloaded), "detail": ""},
        {"step": "Step 4: In final sample (perfect)", "n": len(perfect_in_sample), "detail": ""},
        {"step": "  Missing from sample", "n": len(perfect_missing), "detail": "; ".join(sorted(doi_to_slug.get(d, d) for d in perfect_missing))},
        {"step": "Additional: not-perfect", "n": len(not_perfect_in_sample), "detail": "; ".join(not_perfect_in_sample)},
        {"step": "Additional: not in I4R", "n": len(not_in_i4r), "detail": "; ".join(not_in_i4r)},
        {"step": "FINAL SAMPLE", "n": len(our_results), "detail": ""},
    ]
    out_csv = output_dir / "sample_decomposition.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "n", "detail"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {out_csv}")

    # ── Table 1: Sample construction (simple funnel) ──
    funnel_rows = [
        {"Description": "I4R Replicate universe (unique papers)", "N": total_comp_repro},
        {"Description": "Perfect computational reproduction", "N": len(perfect_dois)},
        {"Description": "Sufficient or partial data availability", "N": len(perfect_with_data)},
        {"Description": "Has extractable tables", "N": len(perfect_with_tables)},
        {"Description": "Final sample", "N": len(our_results)},
    ]
    funnel_csv = output_dir / "sample_funnel.csv"
    with open(funnel_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Description", "N"])
        writer.writeheader()
        writer.writerows(funnel_rows)
    # LaTeX
    funnel_tex = output_dir / "sample_funnel.tex"
    with open(funnel_tex, "w") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Sample Construction}\n")
        f.write("\\label{tab:sample_funnel}\n")
        f.write("\\begin{tabular}{lr}\n")
        f.write("\\toprule\n")
        f.write("Step & N \\\\\n")
        f.write("\\midrule\n")
        for r in funnel_rows:
            f.write(f"{r['Description']} & {r['N']} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"Saved: {funnel_csv}")
    print(f"Saved: {funnel_tex}")

    # ── Table 2: Final sample papers ──
    # Build paper details
    import os as _os
    # Load CrossRef year cache if present (produced by fetch_crossref_years.py)
    crossref_years = {}
    _cr_cache = output_dir / "crossref_year_cache.json"
    if _cr_cache.exists():
        try:
            crossref_years = json.loads(_cr_cache.read_text())
            print(f"Loaded CrossRef year cache: {len(crossref_years)} entries")
        except Exception as e:
            print(f"Could not read CrossRef cache ({e}); years may be missing")
    else:
        print(f"No CrossRef cache at {_cr_cache}. Run scripts/fetch_crossref_years.py "
              f"after this to populate years.")
    paper_rows = []
    for slug in sorted(our_results):
        # Get DOI
        doi = slug_to_dois(slug)[0]

        # Get title, journal, year, replication package URL from i4rep.
        # NOTE: the year column in successfully_replicated_papers.csv is the
        # I4R audit-cohort year (only 2022/2023), not the publication year.
        # We therefore prefer a CrossRef-derived year from
        # `analysis_output/crossref_year_cache.json` if it exists. Produce it
        # by running `scripts/fetch_crossref_years.py` once.
        title = ""
        journal = ""
        year = ""
        repl_url = ""
        for d in slug_to_dois(slug):
            if d in i4rep:
                title = i4rep[d].get("title", "")
                journal = i4rep[d].get("journal", "")
                repl_url = i4rep[d].get("replication_package_url", "")
                break
        cr_year = (crossref_years or {}).get(slug, {}).get("year")
        if cr_year:
            year = str(cr_year)

        # Count data files
        data_dir = PAPERS_DIR / slug / "data"
        n_data_files = 0
        if data_dir.exists():
            n_data_files = sum(1 for _ in data_dir.rglob("*") if _.is_file())

        # Count replication package files
        repl_dir = PAPERS_DIR / slug / "replication_package"
        n_repl_files = 0
        if repl_dir.exists():
            n_repl_files = sum(1 for _ in repl_dir.rglob("*") if _.is_file())

        language, loc = detect_language_and_loc(repl_dir)

        paper_rows.append({
            "paper_slug": slug,
            "doi": doi,
            "title": title,
            "journal": journal,
            "year": year,
            "language": language,
            "loc": loc,
            "n_data_files": n_data_files,
            "n_repl_package_files": n_repl_files,
            "replication_package_url": repl_url,
        })

    papers_csv = output_dir / "sample_papers.csv"
    with open(papers_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "paper_slug", "doi", "title", "journal", "year",
            "language", "loc",
            "n_data_files", "n_repl_package_files", "replication_package_url",
        ])
        writer.writeheader()
        writer.writerows(paper_rows)

    # LaTeX — 5 columns: Journal | Title | Year | Language | LOC
    papers_tex = output_dir / "sample_papers.tex"
    with open(papers_tex, "w") as f:
        f.write("\\begin{footnotesize}\n")
        f.write("\\begin{longtable}{p{1.8cm}p{8cm}rp{1.3cm}r}\n")
        f.write("\\caption{Papers in Final Sample}\n")
        f.write("\\label{tab:sample_papers} \\\\\n")
        f.write("\\toprule\n")
        f.write("Journal & Title & Year & Language & LOC \\\\\n")
        f.write("\\midrule\n")
        f.write("\\endfirsthead\n")
        f.write("\\toprule\n")
        f.write("Journal & Title & Year & Language & LOC \\\\\n")
        f.write("\\midrule\n")
        f.write("\\endhead\n")

        def _esc(s: str) -> str:
            return s.replace("&", "\\&").replace("_", "\\_").replace("%", "\\%").replace("#", "\\#")

        def _journal_short(j: str) -> str:
            for old, new in [
                ("American Economic Journal: Applied Economics", "AEJ: Applied"),
                ("American Economic Journal: Economic Policy", "AEJ: Policy"),
                ("American Economic Journal: Macroeconomics", "AEJ: Macro"),
                ("American Economic Journal: Microeconomics", "AEJ: Micro"),
                ("American Economic Review: Insights", "AER: Insights"),
                ("American Economic Review", "AER"),
                ("American Political Science Review", "APSR"),
                ("American Journal of Political Science", "AJPS"),
                ("Quarterly Journal of Economics", "QJE"),
                ("The Review of Economic Studies", "REStud"),
                ("The Economic Journal", "EJ"),
                ("The Journal of Politics", "JOP"),
                ("Journal of Political Economy", "JPE"),
            ]:
                j = j.replace(old, new)
            return j

        for r in paper_rows:
            title_esc = _esc(r["title"])
            journal_short = _esc(_journal_short(r["journal"]))
            year_str = _esc(r["year"]) if r["year"] else "--"
            lang_esc = _esc(r["language"])
            loc_str = f"{r['loc']:,}" if r["loc"] else "--"
            f.write(f"{journal_short} & {title_esc} & {year_str} & {lang_esc} & {loc_str} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{longtable}\n")
        f.write("\\end{footnotesize}\n")
    print(f"Saved: {papers_csv}")
    print(f"Saved: {papers_tex}")

    # ── Table 3: Language distribution + LOC summary ──
    lang_counts = Counter(r["language"] for r in paper_rows)
    lang_loc_totals: dict[str, list[int]] = {}
    for r in paper_rows:
        lang_loc_totals.setdefault(r["language"], []).append(r["loc"])
    # Stable ordering: known languages first, then mixed, then unknown
    _LANG_ORDER = ["Stata", "R", "MATLAB", "Python", "Julia", "SAS", "mixed", "unknown"]
    ordered_langs = [l for l in _LANG_ORDER if l in lang_counts] + \
                    sorted(l for l in lang_counts if l not in _LANG_ORDER)
    n_papers = len(paper_rows)
    total_loc = sum(r["loc"] for r in paper_rows)

    lang_dist_rows = []
    for lang in ordered_langs:
        n = lang_counts[lang]
        locs = lang_loc_totals.get(lang, [])
        mean_loc = int(sum(locs) / len(locs)) if locs else 0
        lang_dist_rows.append({
            "Language": lang,
            "N papers": n,
            "% papers": round(100 * n / n_papers, 1) if n_papers else 0.0,
            "Total LOC": sum(locs),
            "Mean LOC": mean_loc,
        })
    lang_dist_rows.append({
        "Language": "All",
        "N papers": n_papers,
        "% papers": 100.0 if n_papers else 0.0,
        "Total LOC": total_loc,
        "Mean LOC": int(total_loc / n_papers) if n_papers else 0,
    })
    lang_csv = output_dir / "language_distribution.csv"
    with open(lang_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Language", "N papers", "% papers", "Total LOC", "Mean LOC"])
        writer.writeheader()
        writer.writerows(lang_dist_rows)
    lang_tex = output_dir / "language_distribution.tex"
    with open(lang_tex, "w") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Primary language of replication packages in the final sample. "
                "LOC is a naive line-count across all code files (extensions: "
                ".do/.ado/.R/.r/.Rmd/.m/.py/.jl/.sas).}\n")
        f.write("\\label{tab:language_distribution}\n")
        f.write("\\begin{tabular}{lrrrr}\n")
        f.write("\\toprule\n")
        f.write("Language & N papers & \\% papers & Total LOC & Mean LOC \\\\\n")
        f.write("\\midrule\n")
        for r in lang_dist_rows:
            row = (f"{r['Language']} & {r['N papers']} & "
                   f"{r['% papers']:.1f}\\% & "
                   f"{r['Total LOC']:,} & {r['Mean LOC']:,}")
            if r["Language"] == "All":
                f.write("\\midrule\n")
            f.write(row + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"Saved: {lang_csv}")
    print(f"Saved: {lang_tex}")


if __name__ == "__main__":
    main()
