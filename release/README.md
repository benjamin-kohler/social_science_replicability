# Reproducing *Read the Paper, Write the Code*

This directory contains the release-specific documentation and dependency lock
for the code accompanying *Read the Paper, Write the Code: Agentic Reproduction
of Social-Science Results*. The repository's main `README.md` documents how to
run the benchmark on a new paper; this guide documents how the published
experiments and derived results are pinned and reproduced.

## Release components

The release consists of three linked components:

1. **Code (this repository).** The benchmark engine in `src/`, experiment and
   analysis programs in `scripts/`, the frozen sample manifest in `config/`, and
   tests in `tests/`.
2. **Derived artifacts (separate deposit).** Sanitized agent workspaces, run and
   verification records, error-attribution records, frozen usage metadata, and
   human-audit data. Generated plots and tables are deliberately not deposited.
3. **Third-party inputs (links only).** Published papers, replication packages,
   original code, and input datasets are not redistributed. The artifact deposit
   provides their source URLs and license metadata.

The exact production corpus comprises 335 runs across 48 papers and seven
approach/model combinations. `config/i4rep_release_manifest.json` enumerates the
runs included in that corpus; analysis fails if the loaded run set differs.

## Installation

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
pytest -q
```

`release/analysis-requirements-lock.txt` records the direct Python package
versions used for the artifact round-trip. Running new agent experiments also
requires the applicable external scaffold (Claude Code, OpenAI Codex, OpenCode,
or mini-SWE-agent) and provider credentials in a local `.env` copied from
`.env.example`.

## Reproducing the reported analysis

Download and unpack the accompanying derived-artifact deposit, then follow its
`REPRODUCE.md`. The analysis uses:

- the 335 production runs pinned by `config/i4rep_release_manifest.json`;
- the released stability, extractor-ablation, and pre/post-cutoff runs;
- derived error-attribution records;
- released metadata and usage sidecars; and
- `scripts/analyze_paper_outputs.py` from this repository. This entry point runs
  the underlying analyses but publishes only the plots and tables referenced by
  the manuscript.

The derived artifact is sufficient to regenerate the reported tables and
figures into a new output directory. It does not include generated analysis
outputs and does not require paper PDFs, replication packages, original author
code, input datasets, private provider databases, or unredacted local logs.

The artifact's agent logs preserve their substantive content. Credentials and
local operational identifiers are replaced with `[REDACTED]`, and the deposit
includes the redaction and validation reports.

## Error-attribution analysis

`scripts/error_analysis/` is the release copy of the error-attribution pipeline
used to produce the paper's divergence types and root-cause records. The
artifact deposit contains its derived `divergences_enriched.json` outputs, but
not copied original-author code or the large working directories used during
generation.

## Linking code and artifacts

The artifact metadata records the exact Git commit used for the release. A
paper-specific Git tag is not required.
