"""Shared language descriptors for the error analysis pipeline.

Used by 00_prep_setup.py, 01_trace_failures.py, and 02_detect_error_source.py
to handle multi-language replication packages (Stata, R, MATLAB, Python).
"""

from pathlib import Path

LANGUAGE_INFO = {
    "stata": {
        "name": "Stata",
        "file_patterns": ["*.do", "*.ado"],
        "file_glob": "*.do",
        "proof_label": "Stata snippet",
        "file_ext_example": "filename.do",
        "code_description": "Stata do-files",
    },
    "r": {
        "name": "R",
        "file_patterns": ["*.R", "*.r", "*.Rmd", "*.rmd"],
        "file_glob": "*.R",
        "proof_label": "R snippet",
        "file_ext_example": "filename.R",
        "code_description": "R scripts",
    },
    "matlab": {
        "name": "MATLAB",
        "file_patterns": ["*.m"],
        "file_glob": "*.m",
        "proof_label": "MATLAB snippet",
        "file_ext_example": "filename.m",
        "code_description": "MATLAB scripts",
    },
    "python": {
        "name": "Python",
        "file_patterns": ["*.py"],
        "file_glob": "*.py",
        "proof_label": "Python snippet",
        "file_ext_example": "filename.py",
        "code_description": "Python scripts",
    },
    "mixed": {
        "name": "mixed-language",
        "file_patterns": ["*.do", "*.R", "*.r", "*.Rmd", "*.m", "*.py"],
        "file_glob": "*",
        "proof_label": "code snippet",
        "file_ext_example": "filename.*",
        "code_description": "code files",
    },
    "unknown": {
        "name": "unknown",
        "file_patterns": ["*.do", "*.R", "*.r", "*.Rmd", "*.m", "*.py"],
        "file_glob": "*",
        "proof_label": "code snippet",
        "file_ext_example": "filename.*",
        "code_description": "code files",
    },
}

_EXT_TO_LANG = {
    ".do": "stata", ".ado": "stata",
    ".R": "r", ".r": "r", ".Rmd": "r", ".rmd": "r",
    ".m": "matlab",
    ".py": "python",
}


def detect_language(replication_pkg: Path) -> str:
    """Detect the primary coding language of a replication package.

    Returns one of: stata, r, matlab, python, mixed, unknown.
    """
    if not replication_pkg.is_dir():
        return "unknown"

    lang_counts: dict[str, int] = {}
    for f in replication_pkg.rglob("*"):
        if not f.is_file():
            continue
        lang = _EXT_TO_LANG.get(f.suffix)
        if lang:
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

    if not lang_counts:
        return "unknown"
    if len(lang_counts) == 1:
        return next(iter(lang_counts))

    # Multiple languages — return "mixed" unless one dominates (>80% of files)
    total = sum(lang_counts.values())
    top_lang = max(lang_counts, key=lang_counts.get)
    if lang_counts[top_lang] / total > 0.8:
        return top_lang
    return "mixed"


def get_info(language: str) -> dict:
    """Get the LANGUAGE_INFO entry for a language, defaulting to 'unknown'."""
    return LANGUAGE_INFO.get(language, LANGUAGE_INFO["unknown"])
