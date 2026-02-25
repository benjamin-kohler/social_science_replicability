#!/usr/bin/env python3
"""
Download a single openICPSR project using SeleniumBase UC mode.

Both www.openicpsr.org and test.openicpsr.org are behind Cloudflare,
so plain HTTP requests return 403. SeleniumBase UC mode bypasses
Cloudflare, and we use in-browser fetch() for the actual download
(since Chrome's CDP download commands don't work reliably in UC mode).

Usage:
    python scripts/download_openicpsr_single.py 113050
    python scripts/download_openicpsr_single.py 113050 --output-dir data/openicpsr_aea/zips
    python scripts/download_openicpsr_single.py 113050 --headless
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from seleniumbase import SB

OPENICPSR_BASE = "https://www.openicpsr.org/openicpsr/"
LOGIN_URL = f"{OPENICPSR_BASE}login"

DOWNLOAD_URL_TEMPLATE = (
    "https://www.openicpsr.org/openicpsr/project/{pid}"
    "/version/V1/download/project"
    "?dirPath=/openicpsr/{pid}/fcr:versions/V1"
)

TERMS_URL_TEMPLATE = (
    "https://www.openicpsr.org/openicpsr/project/{pid}"
    "/version/V1/download/terms"
    "?path=/openicpsr/{pid}/fcr:versions/V1&type=project"
)


def do_login(sb, email: str, password: str) -> None:
    """Navigate to openICPSR and complete the ICPSR Keycloak login."""
    print("Opening openICPSR...")
    sb.open(OPENICPSR_BASE)
    time.sleep(3)

    print("Navigating to login...")
    sb.open(LOGIN_URL)
    time.sleep(5)

    # The Keycloak login has two steps:
    # 1) Choose login method (email, Google, ORCID, etc.)
    # 2) Fill email + password form
    print("Selecting email login method...")
    sb.wait_for_element("#kc-emaillogin", timeout=30)
    sb.click("#kc-emaillogin")
    time.sleep(2)

    print("Filling credentials...")
    sb.wait_for_element("#username", timeout=10)
    sb.type("#username", email)
    time.sleep(0.3)
    sb.type("#password", password)
    time.sleep(0.3)
    print("Submitting login form...")
    sb.execute_script("document.querySelector('#kc-form-login').submit();")
    time.sleep(10)

    if "openicpsr.org" in sb.get_current_url() and "login" not in sb.get_current_url():
        print("Login successful.")
    else:
        print(f"WARNING: Post-login URL is {sb.get_current_url()}")


def accept_terms(sb, project_id: str) -> None:
    """Navigate to terms page and accept (required before first download)."""
    terms_url = TERMS_URL_TEMPLATE.format(pid=project_id)
    sb.open(terms_url)
    time.sleep(3)

    # Handle profile completion modal (first-time only)
    page = sb.get_page_source()
    if "completeProfile" in page and "display: block" in page:
        print("Filling profile modal...")
        sb.select_option_by_value("#organizationDepartmentId", "8")  # Economics
        sb.select_option_by_value("#organizationUserRoleId", "12")   # Research Org
        sb.click('input[type="submit"][value="Save"]')
        time.sleep(3)


def fetch_download(sb, project_id: str, output_dir: Path) -> str | None:
    """Use in-browser fetch() to download the project ZIP.

    CDP download commands don't work in UC mode, and passing browser
    cookies to requests fails because Cloudflare checks TLS fingerprints.
    The only reliable approach is fetch() inside the browser context.
    """
    dl_url = DOWNLOAD_URL_TEMPLATE.format(pid=project_id)

    js = """
    (async () => {
        try {
            const resp = await fetch("%s", {credentials: "include"});
            if (!resp.ok) {
                document.title = JSON.stringify({error: "HTTP " + resp.status});
                return;
            }
            const blob = await resp.blob();
            const reader = new FileReader();
            reader.onload = () => {
                document.title = JSON.stringify({
                    data: reader.result.split(",")[1],
                    size: blob.size,
                    type: blob.type,
                });
            };
            reader.readAsDataURL(blob);
        } catch(e) {
            document.title = JSON.stringify({error: e.toString()});
        }
    })();
    """ % dl_url

    print(f"Fetching project {project_id} via in-browser fetch()...")
    sb.execute_script(js)

    # Poll document.title for the result
    for _ in range(120):
        time.sleep(2)
        title = sb.get_title()
        if title.startswith("{"):
            result = json.loads(title)
            break
    else:
        print("Fetch timed out")
        return None

    if "error" in result:
        print(f"Fetch error: {result['error']}")
        return None

    print(f"Got {result['size']} bytes ({result['type']})")
    data = base64.b64decode(result["data"])

    filename = f"{project_id}-V1.zip"
    output_dir.mkdir(parents=True, exist_ok=True)
    outpath = output_dir / filename
    with open(outpath, "wb") as f:
        f.write(data)

    size_mb = outpath.stat().st_size / (1024 * 1024)
    print(f"Saved: {outpath} ({size_mb:.1f} MB)")
    return filename


def main():
    parser = argparse.ArgumentParser(
        description="Download a single openICPSR project ZIP"
    )
    parser.add_argument("project_id", help="openICPSR project ID (e.g. 113050)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/openicpsr_aea/zips"),
        help="Download destination (default: data/openicpsr_aea/zips)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode (default: headed)",
    )
    args = parser.parse_args()

    # Load .env from script directory or project root
    script_dir = Path(__file__).resolve().parent
    load_dotenv(script_dir / ".env")
    load_dotenv()
    email = os.getenv("ICPSR_EMAIL")
    password = os.getenv("ICPSR_PASS")
    if not email or not password:
        print("ERROR: Set ICPSR_EMAIL and ICPSR_PASS in .env")
        sys.exit(1)

    # Check if already downloaded
    output_path = args.output_dir / f"{args.project_id}-V1.zip"
    if output_path.exists():
        print(f"Already exists: {output_path}")
        return

    with SB(uc=True, headless=args.headless) as sb:
        do_login(sb, email, password)
        accept_terms(sb, args.project_id)
        result = fetch_download(sb, args.project_id, args.output_dir)
        if result:
            print(f"\nDone: {result}")
        else:
            print("\nDownload failed.")
            sys.exit(1)


if __name__ == "__main__":
    main()
