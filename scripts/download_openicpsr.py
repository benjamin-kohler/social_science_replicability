#!/usr/bin/env python3
"""
Download AEA replication packages from openICPSR.

Uses SeleniumBase UC mode to bypass Cloudflare protection,
logs into ICPSR, scrapes project metadata from AEA search results,
and downloads ZIP files via in-browser fetch() for each replication package.

Single-pass approach: scrape a page of results, download each project
immediately, then move to the next page. This keeps the browser session
fresh and avoids Cloudflare token expiry.

Usage:
    python scripts/download_openicpsr.py --max-pages 1          # test first page
    python scripts/download_openicpsr.py --resume               # resume downloads
    python scripts/download_openicpsr.py --start-page 50        # start from page 50

Parallel workers (each gets its own browser, progress file, and log):
    python scripts/download_openicpsr.py --worker-id 0 --start-page 0   --max-pages 58
    python scripts/download_openicpsr.py --worker-id 1 --start-page 58  --max-pages 58
    python scripts/download_openicpsr.py --worker-id 2 --start-page 116 --max-pages 58
    python scripts/download_openicpsr.py --worker-id 3 --start-page 174 --max-pages 58
"""

import argparse
import base64
import json
import logging
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from seleniumbase import SB

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AEA_SEARCH_URL = (
    "https://www.openicpsr.org/openicpsr/search/aea/studies"
    "?start={offset}"
    "&ARCHIVE=openicpsr"
    "&PUBLISH_STATUS=PUBLISHED"
    "&sort=DATEUPDATED+desc"
    "&rows=25"
    "&q="
)

OPENICPSR_BASE = "https://www.openicpsr.org/openicpsr/"
LOGIN_URL = "https://www.openicpsr.org/openicpsr/login"

PROJECT_PAGE_URL = "https://www.openicpsr.org/openicpsr/project/{pid}/version/V1"

PROJECT_DOWNLOAD_URL = (
    "https://www.openicpsr.org/openicpsr/project/{pid}"
    "/version/V1/download/project"
    "?dirPath=/openicpsr/{pid}/fcr:versions/V1"
)

TERMS_URL_TEMPLATE = (
    "https://www.openicpsr.org/openicpsr/project/{pid}"
    "/version/V1/download/terms"
    "?path=/openicpsr/{pid}/fcr:versions/V1&type=project"
)

ROWS_PER_PAGE = 25

logger = logging.getLogger("openicpsr")


# ---------------------------------------------------------------------------
# Progress tracking (simple JSON file with downloaded IDs)
# ---------------------------------------------------------------------------

def load_progress(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"downloaded": {}, "failed": [], "last_page": 0}


def save_progress(progress: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(progress, f, indent=2)


# ---------------------------------------------------------------------------
# Browser helpers
# ---------------------------------------------------------------------------

def do_login(sb, email: str, password: str, max_attempts: int = 3):
    """Navigate to openICPSR and complete the ICPSR Keycloak login."""
    for attempt in range(1, max_attempts + 1):
        try:
            logger.info("Login attempt %d/%d...", attempt, max_attempts)
            sb.open(OPENICPSR_BASE)
            time.sleep(3)

            sb.open(LOGIN_URL)
            time.sleep(5)

            sb.wait_for_element("#kc-emaillogin", timeout=20)
            sb.click("#kc-emaillogin")
            time.sleep(2)

            sb.wait_for_element("#username", timeout=10)
            sb.type("#username", email)
            time.sleep(0.3)
            sb.type("#password", password)
            time.sleep(0.3)
            logger.info("Submitting login form...")
            sb.execute_script(
                "document.querySelector('#kc-form-login').submit();"
            )
            time.sleep(10)

            url = sb.get_current_url()
            if "openicpsr.org" in url and "login" not in url:
                logger.info("Login successful")
                return
            else:
                logger.warning("Post-login URL: %s", url)
        except Exception as e:
            logger.warning("Login attempt %d failed: %s", attempt, e)

        if attempt < max_attempts:
            logger.info("Retrying login in 5s...")
            time.sleep(5)

    raise RuntimeError("Login failed after %d attempts" % max_attempts)


def accept_terms(sb, project_id: str):
    """Navigate to terms page and accept (required before first download)."""
    terms_url = TERMS_URL_TEMPLATE.format(pid=project_id)
    sb.open(terms_url)
    time.sleep(3)

    page = sb.get_page_source()
    if "completeProfile" in page and "display: block" in page:
        logger.info("Filling profile modal...")
        sb.select_option_by_value("#organizationDepartmentId", "8")
        sb.select_option_by_value("#organizationUserRoleId", "12")
        sb.click('input[type="submit"][value="Save"]')
        time.sleep(3)


def fetch_download(sb, project_id: str, download_dir: Path) -> str | None:
    """Use in-browser fetch() with streaming to download the project ZIP.

    Streams the response in 1 MB chunks to avoid holding the entire file
    in browser memory (which killed browsers on large files).
    Each chunk is base64-encoded and passed back via document.title.
    """
    dl_url = PROJECT_DOWNLOAD_URL.format(pid=project_id)
    filename = f"{project_id}-V1.zip"
    outpath = download_dir / filename

    if outpath.exists():
        logger.info("Already exists: %s", outpath)
        return filename

    # Navigate to the project page so fetch() has the right origin/cookies
    project_url = PROJECT_PAGE_URL.format(pid=project_id)
    sb.open(project_url)
    time.sleep(3)

    # Start the fetch and get total size + first signal
    js_start = """
    (async () => {
        try {
            const resp = await fetch("%s", {credentials: "include"});
            if (!resp.ok) {
                document.title = JSON.stringify({error: "HTTP " + resp.status});
                return;
            }
            // Store reader on window so we can read chunks later
            window._dlReader = resp.body.getReader();
            window._dlChunks = [];
            window._dlDone = false;
            window._dlSize = 0;

            // Read all chunks into memory in background
            async function readAll() {
                while (true) {
                    const {done, value} = await window._dlReader.read();
                    if (done) { window._dlDone = true; break; }
                    window._dlChunks.push(value);
                    window._dlSize += value.length;
                }
            }
            readAll();
            document.title = JSON.stringify({status: "streaming"});
        } catch(e) {
            document.title = JSON.stringify({error: e.toString()});
        }
    })();
    """ % dl_url

    sb.execute_script(js_start)

    # Wait for streaming to start or error
    for _ in range(30):
        time.sleep(2)
        title = sb.get_title()
        if title.startswith("{"):
            result = json.loads(title)
            break
    else:
        logger.error("Fetch start timed out for %s", project_id)
        return None

    if "error" in result:
        logger.error("Fetch error for %s: %s", project_id, result["error"])
        return None

    # Wait for download to complete (up to 3 min — longer risks Cloudflare expiry)
    for _ in range(90):
        time.sleep(2)
        done = sb.execute_script("return window._dlDone === true;")
        if done:
            break
        size = sb.execute_script("return window._dlSize || 0;")
        if size and size > 0:
            logger.debug("  %s: %.1f MB so far...", project_id, size / 1048576)
    else:
        logger.error("Download stream timed out for %s", project_id)
        return None

    total_size = sb.execute_script("return window._dlSize;")
    logger.info("  %s: fetched %.1f MB, extracting...", project_id, total_size / 1048576)

    # Extract data in chunks via base64 to avoid one giant string
    num_chunks = sb.execute_script("return window._dlChunks.length;")
    download_dir.mkdir(parents=True, exist_ok=True)

    with open(outpath, "wb") as f:
        for i in range(num_chunks):
            b64 = sb.execute_script(
                "return (function(){"
                "var c = window._dlChunks[%d];"
                "var b = '';"
                "for (var k = 0; k < c.length; k++) b += String.fromCharCode(c[k]);"
                "window._dlChunks[%d] = null;"
                "return btoa(b);"
                "})();" % (i, i)
            )
            f.write(base64.b64decode(b64))

    # Clean up browser memory
    sb.execute_script("delete window._dlReader; delete window._dlChunks; delete window._dlDone; delete window._dlSize;")

    actual_size = outpath.stat().st_size
    size_mb = actual_size / (1024 * 1024)

    # Validate: written size should match fetched size
    if abs(actual_size - total_size) > 1024:
        logger.error("Size mismatch for %s: fetched %d, wrote %d — deleting",
                      project_id, total_size, actual_size)
        outpath.unlink()
        return None

    logger.info("Saved: %s (%.1f MB)", outpath, size_mb)
    return filename


def parse_search_page(sb) -> list[dict]:
    """Extract project cards from the current search results page."""
    page_source = sb.get_page_source()

    project_links = re.findall(r'/openicpsr/project/(\d+)', page_source)
    unique_ids = list(dict.fromkeys(project_links))

    projects = []
    for pid in unique_ids:
        project = {"project_id": pid, "title": "", "doi": ""}

        try:
            link_el = sb.find_element(f"a[href*='/openicpsr/project/{pid}']")
            project["title"] = link_el.text.strip()
        except Exception:
            pass

        doi_match = re.search(rf'10\.3886/E{pid}V\d+', page_source)
        if doi_match:
            project["doi"] = doi_match.group(0)

        projects.append(project)

    logger.info("Found %d projects on this page", len(projects))
    return projects


# ---------------------------------------------------------------------------
# Main loop — single-pass: scrape page → download each project → next page
# ---------------------------------------------------------------------------

class SessionExpired(Exception):
    """Raised when too many consecutive downloads fail, indicating Cloudflare session death."""
    def __init__(self, last_page):
        self.last_page = last_page


def run(sb, output_dir: Path, max_pages: int | None, start_page: int,
        resume: bool, worker_id: int):
    """Scrape and download in a single pass."""
    download_dir = output_dir / "zips"
    download_dir.mkdir(parents=True, exist_ok=True)

    progress_path = output_dir / f"progress_w{worker_id}.json"
    progress = load_progress(progress_path)
    downloaded_ids = set(progress["downloaded"].keys())

    # Also check other workers' progress files and existing zips
    if resume:
        for pf in output_dir.glob("progress_w*.json"):
            if pf == progress_path:
                continue
            other = load_progress(pf)
            downloaded_ids.update(other["downloaded"].keys())

    terms_accepted = False
    page = start_page
    total_downloaded = len(downloaded_ids)
    consecutive_errors = 0
    consecutive_dl_failures = 0  # track session health

    logger.info("Starting from page %d (%d already downloaded)", page + 1, total_downloaded)

    while True:
        if max_pages is not None and (page - start_page) >= max_pages:
            logger.info("Reached max_pages limit (%d)", max_pages)
            break

        # --- Scrape one page of search results ---
        try:
            sb.open(AEA_SEARCH_URL.format(offset=page * ROWS_PER_PAGE))
            time.sleep(5)
            projects = parse_search_page(sb)
        except Exception as e:
            consecutive_errors += 1
            logger.warning("Error loading page %d (attempt %d): %s",
                           page + 1, consecutive_errors, e)
            if consecutive_errors >= 3:
                logger.error("Too many consecutive errors, stopping")
                break
            time.sleep(10)
            try:
                sb.open(OPENICPSR_BASE)
                time.sleep(3)
            except Exception:
                pass
            continue

        consecutive_errors = 0

        if not projects:
            logger.info("No projects on page %d — end of results", page + 1)
            break

        # --- Download each project from this page ---
        for proj in projects:
            pid = proj["project_id"]

            if resume and pid in downloaded_ids:
                continue

            # Accept terms on the very first download
            if not terms_accepted:
                logger.info("Accepting terms for first project %s...", pid)
                try:
                    accept_terms(sb, pid)
                    terms_accepted = True
                except Exception as e:
                    logger.warning("Terms acceptance failed: %s", e)
                    terms_accepted = True  # try downloading anyway

            logger.info("[%d] Downloading %s: %s",
                        total_downloaded + 1, pid, proj.get("title", "")[:60])

            success = False
            for attempt in range(1, 4):
                try:
                    filename = fetch_download(sb, pid, download_dir)
                    if filename:
                        progress["downloaded"][pid] = {
                            "title": proj.get("title", ""),
                            "doi": proj.get("doi", ""),
                            "filename": filename,
                            "downloaded_at": datetime.now(timezone.utc).isoformat(),
                        }
                        downloaded_ids.add(pid)
                        total_downloaded += 1
                        consecutive_dl_failures = 0
                        success = True
                        break
                    else:
                        logger.warning("  Attempt %d/3 failed for %s", attempt, pid)
                except Exception as e:
                    logger.error("  Attempt %d/3 error for %s: %s", attempt, pid, e)

                if attempt < 3:
                    # Refresh Cloudflare session before retry
                    logger.info("  Refreshing session before retry...")
                    try:
                        sb.open(OPENICPSR_BASE)
                        time.sleep(5)
                    except Exception:
                        pass
                    time.sleep(5 * attempt)

            if not success:
                if pid not in progress["failed"]:
                    progress["failed"].append(pid)
                logger.error("  Failed after 3 attempts: %s", pid)
                consecutive_dl_failures += 1

                # If 3+ projects in a row fail, session is likely dead
                if consecutive_dl_failures >= 3:
                    progress["last_page"] = page
                    save_progress(progress, progress_path)
                    raise SessionExpired(page)

            # Save progress after each project
            progress["last_page"] = page
            save_progress(progress, progress_path)

            # Polite delay between downloads
            time.sleep(random.uniform(5, 10))

        logger.info("Page %d complete. %d downloaded so far.", page + 1, total_downloaded)
        page += 1
        time.sleep(random.uniform(1, 3))

    logger.info("Done. %d downloaded, %d failed.",
                len(progress["downloaded"]), len(progress["failed"]))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download AEA replication packages from openICPSR"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/openicpsr_aea"),
        help="Download destination (default: data/openicpsr_aea/)",
    )
    parser.add_argument(
        "--max-pages", type=int, default=None,
        help="Limit number of search pages to process (default: unlimited)",
    )
    parser.add_argument(
        "--start-page", type=int, default=0,
        help="Page number to start from, 0-indexed (default: 0)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip projects already in progress.json",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run browser in headless mode (default: headed for debugging)",
    )
    parser.add_argument(
        "--worker-id", type=int, default=0,
        help="Worker ID for parallel runs (default: 0). Each worker gets its own progress file.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [W{args.worker_id}] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    script_dir = Path(__file__).resolve().parent
    load_dotenv(script_dir / ".env")
    load_dotenv()
    email = os.getenv("ICPSR_EMAIL")
    password = os.getenv("ICPSR_PASS")
    if not email or not password:
        logger.error("Set ICPSR_EMAIL and ICPSR_PASS in .env")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    current_start = args.start_page
    end_page = args.start_page + args.max_pages if args.max_pages else None
    restarts = 0
    login_backoff = 60  # seconds, doubles each retry
    max_login_backoff = 3600  # cap at 1 hour

    while True:
        remaining_pages = (end_page - current_start) if end_page else None
        try:
            with SB(uc=True, headless=args.headless) as sb:
                do_login(sb, email, password)
                login_backoff = 60  # reset on successful login
                run(sb, args.output_dir, remaining_pages, current_start,
                    args.resume, args.worker_id)
            break  # Normal completion
        except SessionExpired as e:
            restarts += 1
            current_start = e.last_page  # Resume from the page that was in progress
            logger.warning("Session expired on page %d. Restarting browser... (restart #%d)",
                           current_start + 1, restarts)
            time.sleep(10)
        except RuntimeError as e:
            if "Login failed" in str(e):
                logger.warning("Login failed. Retrying in %ds...", login_backoff)
                time.sleep(login_backoff)
                login_backoff = min(login_backoff * 2, max_login_backoff)
            else:
                logger.error("Fatal error: %s", e)
                break
        except Exception as e:
            logger.error("Fatal error: %s", e)
            break


if __name__ == "__main__":
    main()
