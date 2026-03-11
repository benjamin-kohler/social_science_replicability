"""Shared utilities for finding and selecting PDF pages by item ID.

These functions are used by both the Judge and ResultsExtractor to locate
pages containing specific tables or figures in a paper PDF.
"""

import re


def find_item_pages(paper_text: str, item_id: str) -> list[int]:
    """Find page numbers (1-indexed) mentioning an item, plus neighbours."""
    pages = re.split(r"\n--- Page (\d+) ---\n", paper_text)
    page_map: dict[int, str] = {}
    for i in range(1, len(pages) - 1, 2):
        try:
            page_map[int(pages[i])] = pages[i + 1]
        except (ValueError, IndexError):
            continue

    matching: list[int] = []
    for pnum, ptext in page_map.items():
        if re.search(re.escape(item_id), ptext, re.IGNORECASE):
            matching.append(pnum)

    if not matching:
        num_part = item_id.replace("Table ", "").replace("Figure ", "")
        for pnum, ptext in page_map.items():
            if re.search(rf"(?:Table|Figure|table|figure)\s*{re.escape(num_part)}", ptext):
                matching.append(pnum)

    if not matching:
        return []

    all_pages: set[int] = set()
    for p in matching:
        all_pages.update(range(max(1, p - 1), p + 2))

    return sorted(p for p in all_pages if p in page_map)


def extract_table_pages(paper_text: str, item_id: str) -> str:
    """Find PDF pages mentioning this item and return surrounding text context."""
    pages = re.split(r"\n--- Page (\d+) ---\n", paper_text)
    page_map: dict[int, str] = {}
    for i in range(1, len(pages) - 1, 2):
        try:
            page_map[int(pages[i])] = pages[i + 1]
        except (ValueError, IndexError):
            continue

    page_nums = find_item_pages(paper_text, item_id)
    parts = []
    for p in page_nums:
        if p in page_map:
            parts.append(f"--- Page {p} ---\n{page_map[p]}")
    return "\n".join(parts)


def select_page_images(
    page_images: list[dict], page_nums: list[int],
) -> list[dict]:
    """Select page images by page number."""
    by_page = {img["page"]: img for img in page_images}
    return [by_page[p] for p in page_nums if p in by_page]
