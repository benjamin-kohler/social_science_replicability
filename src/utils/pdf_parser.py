"""PDF parsing utilities for extracting text and tables from papers."""

import re
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import pdfplumber

from .logging_utils import get_logger

logger = get_logger(__name__)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file.

    Uses PyMuPDF for efficient text extraction with layout preservation.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Extracted text as a single string.

    Raises:
        FileNotFoundError: If the PDF file doesn't exist.
        ValueError: If the file is not a valid PDF.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"File is not a PDF: {pdf_path}")

    logger.info(f"Extracting text from: {pdf_path}")

    try:
        doc = fitz.open(pdf_path)
        text_parts = []

        for page_num, page in enumerate(doc):
            page_text = page.get_text("text")
            text_parts.append(f"\n--- Page {page_num + 1} ---\n")
            text_parts.append(page_text)

        doc.close()
        full_text = "".join(text_parts)

        logger.info(f"Extracted {len(full_text)} characters from {len(doc)} pages")
        return full_text

    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise


def extract_tables_from_pdf(pdf_path: str) -> list[dict]:
    """Extract tables from a PDF file using pdfplumber.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of dictionaries, each containing:
        - 'page': Page number (1-indexed)
        - 'table_index': Index of table on that page
        - 'data': Table data as list of lists
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    logger.info(f"Extracting tables from: {pdf_path}")

    tables = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_tables = page.extract_tables()

                for table_idx, table in enumerate(page_tables):
                    if table:  # Only add non-empty tables
                        tables.append({
                            "page": page_num + 1,
                            "table_index": table_idx,
                            "data": table,
                        })

        logger.info(f"Extracted {len(tables)} tables from PDF")
        return tables

    except Exception as e:
        logger.error(f"Error extracting tables from PDF: {e}")
        raise


def identify_sections(text: str) -> dict[str, str]:
    """Identify and extract major sections from paper text.

    Args:
        text: Full text of the paper.

    Returns:
        Dictionary mapping section names to their content.
    """
    # Common section headers in academic papers
    section_patterns = [
        r"(?i)\b(abstract)\b",
        r"(?i)\b(introduction)\b",
        r"(?i)\b(literature\s+review|related\s+work)\b",
        r"(?i)\b(data|data\s+and\s+methods?|methods?\s+and\s+data)\b",
        r"(?i)\b(methods?|methodology|empirical\s+strategy)\b",
        r"(?i)\b(results?|findings|empirical\s+results?)\b",
        r"(?i)\b(discussion)\b",
        r"(?i)\b(conclusion|conclusions|concluding\s+remarks)\b",
        r"(?i)\b(references|bibliography)\b",
        r"(?i)\b(appendix|appendices)\b",
    ]

    sections = {}
    lines = text.split("\n")
    current_section = "preamble"
    current_content = []

    for line in lines:
        # Check if line is a section header
        is_header = False
        for pattern in section_patterns:
            if re.match(pattern, line.strip()):
                # Save previous section
                if current_content:
                    sections[current_section] = "\n".join(current_content)

                # Start new section
                current_section = line.strip().lower()
                current_content = []
                is_header = True
                break

        if not is_header:
            current_content.append(line)

    # Save last section
    if current_content:
        sections[current_section] = "\n".join(current_content)

    return sections


def extract_figure_captions(text: str) -> list[dict]:
    """Extract figure captions from paper text.

    Args:
        text: Full text of the paper.

    Returns:
        List of dicts with 'figure_number' and 'caption'.
    """
    # Pattern for figure captions
    pattern = r"(?i)(Figure|Fig\.?)\s*(\d+[A-Za-z]?)[:\.]?\s*([^\n]+(?:\n(?![A-Z])[^\n]+)*)"

    captions = []
    for match in re.finditer(pattern, text):
        figure_num = f"Figure {match.group(2)}"
        caption = match.group(3).strip()
        captions.append({
            "figure_number": figure_num,
            "caption": caption,
        })

    return captions


def extract_table_captions(text: str) -> list[dict]:
    """Extract table captions from paper text.

    Args:
        text: Full text of the paper.

    Returns:
        List of dicts with 'table_number' and 'caption'.
    """
    # Pattern for table captions
    pattern = r"(?i)(Table)\s*(\d+[A-Za-z]?)[:\.]?\s*([^\n]+(?:\n(?![A-Z])[^\n]+)*)"

    captions = []
    for match in re.finditer(pattern, text):
        table_num = f"Table {match.group(2)}"
        caption = match.group(3).strip()
        captions.append({
            "table_number": table_num,
            "caption": caption,
        })

    return captions
