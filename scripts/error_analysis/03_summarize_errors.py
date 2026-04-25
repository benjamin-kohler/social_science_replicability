"""Step 03: produce a LaTeX overview table from enriched divergence results.

Each divergences_enriched.json (produced by 02_detect_error_source.py) covers one
paper × agent combination.  This script merges them into a single overview table per
paper, with one row per divergence and columns for the failure type, severity, three
consistency-check verdicts, and a derived root cause.

data_available per divergence comes from step 01 (01_trace_failures.py) and is
preserved unchanged through step 02.

Usage
-----
    # Auto-discover all enriched files under explainer_workspaces/
    python 03_summarize_errors.py --output-dir summaries/

    # Explicit inputs for one paper
    python 03_summarize_errors.py \\
        --paper-id 10.1257_aer.20190565 \\
        --output   summaries/10.1257_aer.20190565_errors.tex \\
        --inputs \\
          explainer_workspaces/10.1257_aer.20190565/codex/error_source/divergences_enriched.json codex \\
          explainer_workspaces/10.1257_aer.20190565/claude/error_source/divergences_enriched.json claude

--inputs accepts alternating pairs: PATH LABEL PATH LABEL ...
"""

import argparse
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Root cause derivation
# ---------------------------------------------------------------------------

_TRIGGERS = {"contradicts", "omission"}


def _derive_source(d: dict) -> str:
    """Derive a root cause title from the consistency check verdicts.

    Root cause cascade (first match wins):
      1. Data not in package      — required raw data absent
      2. Paper-code mismatch      — paper explicitly contradicts original code
      3. Paper underspecified      — original code does something the paper is silent about
      4. Summary gap (contradicts) — summary explicitly contradicts the paper
      5. Summary gap (omission)    — summary dropped information the paper provided
      6. Agent contradicted summary — agent explicitly deviated from summary instructions
      7. Agent missed summary info  — summary specified it but agent didn't implement it
      8. Unexplained               — no contradictions or omissions identified
    """
    p_code  = d.get("paper_vs_original_code", "unclear")
    p_sum   = d.get("paper_vs_summary",       "unclear")
    s_agent = d.get("summary_vs_agent",       "unclear")
    data    = d.get("data_available",          None)

    if data == "missing":
        return "Data not in package"
    # Paper vs original code
    if p_code == "contradicts":
        return "Paper-code mismatch"
    if p_code == "omission":
        return "Paper underspecified"
    # Paper vs summary
    if p_sum == "contradicts":
        return "Summary gap (contradicts)"
    if p_sum == "omission":
        return "Summary gap (omission)"
    # Summary vs agent
    if s_agent == "contradicts":
        return "Agent contradicted summary"
    if s_agent == "omission":
        return "Agent missed summary info"
    return "Unexplained"


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------

_VERDICT_SHORT = {
    "consistent":  r"\cmark",
    "contradicts": r"\xmark",
    "omission":    r"\omark",
    "unclear":     r"?",
    "available":   r"\cmark",
    "missing":     r"\xmark",
    None:          r"---",
}

_SEV_COLOR = {
    "critical": r"\cellcolor{red!15}critical",
    "medium":   r"\cellcolor{orange!20}medium",
    "minor":    r"\cellcolor{yellow!20}minor",
}

_AGENT_SHORT = {
    "claude-opus-4-6_claude-code": "Claude",
    "gpt-5.3-codex_codex":        "Codex 5.3",
    "gpt-5.4_codex":              "Codex 5.4",
    "gpt-5.4_opencode":           "OC 5.4",
    "gpt-5.4_swe-agent":          "SWE 5.4",
    "z-ai_glm-5_opencode":        "OC GLM",
    "z-ai_glm-5_swe-agent":       "SWE GLM",
}

_SOURCE_TO_RCODE = {
    "Agent ignored instructions": "R1",
    "Paper underspecified":       "R2",
    "Data not in package":        "R3",
    "Summary gap":                "R4",
    "Unexplained":                "R5",
}

_TABLE_NOTE = (
    r"\smallskip\noindent{\footnotesize"
    r"\textit{Notes:} "
    r"Fail stage: P\,↔\,C = paper vs.\ original code; "
    r"P\,↔\,S = paper vs.\ methodology summary; "
    r"S\,↔\,A = summary vs.\ agent code. "
    r"Fail type: \xmark~contradicts (direct conflict); \omark~omission (upstream specifies, downstream silent); ?~unclear. "
    r"Agents: Claude = Claude Code Opus~4.6; Codex~5.3/5.4 = Codex CLI; "
    r"OC = OpenCode; SWE = SWE-Agent; GLM = GLM-5.}"
)

def _esc(s: str | None) -> str:
    """Minimal LaTeX escaping for text cells."""
    if s is None:
        return ""
    return (
        s.replace("&",  r"\&")
         .replace("%",  r"\%")
         .replace("_",  r"\_")
         .replace("#",  r"\#")
         .replace("$",  r"\$")
         .replace("{",  r"\{")
         .replace("}",  r"\}")
         .replace("~",  r"\textasciitilde{}")
         .replace("^",  r"\textasciicircum{}")
    )

def _v(key: str, d: dict) -> str:
    return _VERDICT_SHORT.get(d.get(key), "?")


def _fail_stage_and_type(d: dict) -> tuple[str, str]:
    """Return (stage_macro, verdict_symbol) at the first non-consistent check.
    Data availability is checked first."""
    if d.get("data_available") == "missing":
        return r"\stageData", r"\xmark"
    for field, macro in [
        ("paper_vs_original_code", r"\stagePC"),
        ("paper_vs_summary",       r"\stagePS"),
        ("summary_vs_agent",       r"\stageSA"),
    ]:
        v = d.get(field, "unclear")
        if v in _TRIGGERS or v == "unclear":
            return macro, _VERDICT_SHORT.get(v, "?")
    return r"---", r"\cmark"   # all consistent — unexplained


def _output_label(d: dict) -> str:
    """Return the output label (Table/Figure number) for a divergence."""
    out = d.get("output")
    if out:
        return _esc(out)
    # Fallback: try to infer from agent_location filename
    loc = (d.get("agent_location") or {}).get("file", "")
    import re
    m = re.search(r"(table|figure|fig)[\s_]?(\d+)", loc, re.IGNORECASE)
    if m:
        kind = "Table" if m.group(1).lower() == "table" else "Figure"
        return f"{kind}~{m.group(2)}"
    return "General"


# ---------------------------------------------------------------------------
# Table builder
# ---------------------------------------------------------------------------

def _build_table(
    paper_id: str,
    datasets: list[tuple[str, list[dict]]],
) -> str:
    """Return the full LaTeX for one paper's overview table.

    Columns: Output | Sev. | Fail stage | Fail type | [Agent] | Description
    """
    n_agents  = len(datasets)
    has_multi = n_agents > 1

    # Output | Sev | Stage | Type | [Agent] | Description
    # Target total ≤ 16.5 cm on A4 portrait.
    if has_multi:
        col_spec = r"p{1.4cm} c p{2.2cm} c p{0.9cm} p{5.5cm}"
    else:
        col_spec = r"p{1.4cm} c p{2.2cm} c p{6.8cm}"

    n_cols = 6 if has_multi else 5

    agent_hdr = r"\textit{Agent} & " if has_multi else ""

    def _header_row() -> str:
        return (
            r"\textbf{Output} & \textbf{Sev.} & \textbf{Fail stage} & "
            r"\textbf{Fail type} & "
            + agent_hdr
            + r"\textbf{Description} \\"
        )

    lines = [
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{longtable}{" + col_spec + r"}",
        r"\caption{Replication Divergences: " + _esc(paper_id) + r"}"
        + r"\label{tab:errors_" + paper_id.replace(".", "_").replace("/", "_") + r"}\\",
        r"\toprule",
        _header_row(),
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{" + str(n_cols) + r"}{l}{\small\textit{(continued)}} \\",
        r"\toprule",
        _header_row(),
        r"\midrule",
        r"\endhead",
        r"\midrule \multicolumn{" + str(n_cols) + r"}{r}{\small\textit{continued on next page}} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\multicolumn{" + str(n_cols) + r"}{p{\linewidth}}{" + _TABLE_NOTE + r"} \\",
        r"\endlastfoot",
    ]

    all_ids: list[int] = []
    divs_by_id_by_label: dict[str, dict[int, dict]] = {}
    for label, divs in datasets:
        divs_by_id_by_label[label] = {d["id"]: d for d in divs}
        for d in divs:
            if d["id"] not in all_ids:
                all_ids.append(d["id"])
    all_ids.sort()

    for did in all_ids:
        for i, (label, divs) in enumerate(datasets):
            d = divs_by_id_by_label[label].get(did)
            if d is None:
                continue

            output_cell = _output_label(d)
            sev_cell    = _SEV_COLOR.get(d.get("severity", ""), _esc(d.get("severity", "")))
            stage, ftype = _fail_stage_and_type(d)
            description = _esc(d.get("description", ""))
            short_label = _AGENT_SHORT.get(label, label)
            agent_cell  = (r"\textit{" + _esc(short_label) + r"} & ") if has_multi else ""

            row = (
                f"  {output_cell} & "
                f"{sev_cell} & "
                f"{stage} & "
                f"{ftype} & "
                f"{agent_cell}"
                f"{description} \\\\"
            )
            lines.append(row)

        if has_multi and did != all_ids[-1]:
            lines.append(r"  \midrule[0.3pt]")

    lines.append(r"\end{longtable}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Preamble hint
# ---------------------------------------------------------------------------

_PREAMBLE = r"""% Add to your LaTeX preamble:
%   \usepackage{booktabs, longtable, colortbl, xcolor, pifont, textcomp, tikz}
%   \newcommand{\cmark}{\textcolor{green!60!black}{\ding{51}}}
%   \newcommand{\xmark}{\textcolor{red}{\ding{55}}}
%   \newcommand{\omark}{\textcolor{orange!80!black}{$\circ$}}
%   \setlength{\tabcolsep}{4pt}
%   % Pipeline stage badges (fail stage column)
%   \newcommand{\stageData}{%
%     \tikz[baseline=-0.6ex]\node[fill=gray!20,rounded corners=2pt,
%       inner xsep=3pt,inner ysep=1.5pt,font=\scriptsize\sffamily]{%
%       Data avail.};}
%   \newcommand{\stagePC}{%
%     \tikz[baseline=-0.6ex]\node[fill=blue!12,rounded corners=2pt,
%       inner xsep=3pt,inner ysep=1.5pt,font=\scriptsize\sffamily]{%
%       Code\,$\neq$\,Paper};}
%   \newcommand{\stagePS}{%
%     \tikz[baseline=-0.6ex]\node[fill=orange!18,rounded corners=2pt,
%       inner xsep=3pt,inner ysep=1.5pt,font=\scriptsize\sffamily]{%
%       Paper\,$\neq$\,Sum.};}
%   \newcommand{\stageSA}{%
%     \tikz[baseline=-0.6ex]\node[fill=green!15,rounded corners=2pt,
%       inner xsep=3pt,inner ysep=1.5pt,font=\scriptsize\sffamily]{%
%       Sum.\,$\neq$\,Agent};}
"""


# ---------------------------------------------------------------------------
# Auto-discovery
# ---------------------------------------------------------------------------

def _discover_inputs(workspace_root: Path) -> dict[str, list[tuple[str, Path]]]:
    """
    Scan workspace_root for divergences_enriched.json files.
    Returns {paper_id: [(agent_label, path), ...]} sorted by paper_id then label.
    """
    found: dict[str, list[tuple[str, Path]]] = {}
    pattern = "*/*/error_source/divergences_enriched.json"
    for p in sorted(workspace_root.glob(pattern)):
        parts = p.relative_to(workspace_root).parts
        # parts = (paper_id, agent, "error_source", "divergences_enriched.json")
        if len(parts) == 4:
            paper_id, agent = parts[0], parts[1]
            found.setdefault(paper_id, []).append((agent, p))
    return found


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Produce LaTeX overview tables from enriched divergence results."
    )
    parser.add_argument(
        "--inputs", nargs="+", default=None,
        metavar="PATH_OR_LABEL",
        help="Alternating pairs: path/to/divergences_enriched.json LABEL  (requires --paper-id and --output)."
    )
    parser.add_argument("--paper-id", default=None,
        help="Paper identifier (required when --inputs is used).")
    parser.add_argument("--output", default=None,
        help="Output .tex file path (required when --inputs is used).")
    parser.add_argument(
        "--workspace-dir",
        default=str(here / "explainer_workspaces"),
        help="Root of explainer_workspaces/ for auto-discovery (default: ./explainer_workspaces)."
    )
    parser.add_argument(
        "--output-dir",
        default=str(here / "summaries"),
        help="Output folder for auto-discovered tables (default: ./summaries)."
    )
    parser.add_argument("--rerun", action="store_true",
        help="Re-run even if output file already exists.")
    return parser.parse_args()


def _process_paper(
    paper_id: str,
    datasets: list[tuple[str, list[dict]]],
    output_path: Path,
    rerun: bool,
) -> None:
    if output_path.exists() and not rerun:
        print(f"SKIP: {output_path} already exists. Use --rerun to overwrite.")
        return

    table_tex = _build_table(paper_id, datasets)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_PREAMBLE + "\n" + table_tex + "\n", encoding="utf-8")
    print(f"-> Saved {output_path}")

    print(f"\n{'='*55}")
    for label, divs in datasets:
        by_sev: dict[str, int] = {}
        by_src: dict[str, int] = {}
        for d in divs:
            sev = d.get("severity", "?")
            by_sev[sev] = by_sev.get(sev, 0) + 1
            src = _derive_source(d)
            by_src[src] = by_src.get(src, 0) + 1
        print(f"  [{label}]  critical={by_sev.get('critical',0)}  "
              f"medium={by_sev.get('medium',0)}  minor={by_sev.get('minor',0)}")
        print(f"          sources: " + "  ".join(f"{k}={v}" for k, v in sorted(by_src.items())))
    print()


def main() -> None:
    args = parse_args()

    if args.inputs:
        # Explicit mode
        if not args.paper_id:
            sys.exit("ERROR: --paper-id is required when --inputs is used.")
        if not args.output:
            sys.exit("ERROR: --output is required when --inputs is used.")

        raw = args.inputs
        if len(raw) % 2 != 0:
            sys.exit("ERROR: --inputs must be pairs of PATH LABEL.")

        datasets: list[tuple[str, list[dict]]] = []
        for i in range(0, len(raw), 2):
            path_str, label = raw[i], raw[i + 1]
            p = Path(path_str).expanduser().resolve()
            if not p.exists():
                sys.exit(f"ERROR: file not found: {p}")
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                sys.exit(f"ERROR: malformed JSON in {p}: {e}")
            divs = data.get("divergences", data.get("discrepancies", []))
            print(f"  {label}: {len(divs)} divergences from {p.name}")
            datasets.append((label, divs))

        _process_paper(
            args.paper_id,
            datasets,
            Path(args.output).expanduser().resolve(),
            args.rerun,
        )

    else:
        # Auto-discovery mode
        ws_root = Path(args.workspace_dir).expanduser().resolve()
        out_dir = Path(args.output_dir).expanduser().resolve()

        if not ws_root.is_dir():
            sys.exit(f"ERROR: --workspace-dir does not exist: {ws_root}")

        discovered = _discover_inputs(ws_root)
        if not discovered:
            sys.exit(f"ERROR: no divergences_enriched.json files found under {ws_root}")

        print(f"Auto-discovered {sum(len(v) for v in discovered.values())} enriched files "
              f"across {len(discovered)} papers.\n")

        for paper_id, agent_paths in sorted(discovered.items()):
            datasets = []
            for label, p in sorted(agent_paths):
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                except json.JSONDecodeError as e:
                    print(f"  [{paper_id} / {label}]: SKIP — malformed JSON in {p.name}: {e}")
                    continue
                divs = data.get("divergences", data.get("discrepancies", []))
                print(f"  [{paper_id} / {label}]: {len(divs)} divergences")
                datasets.append((label, divs))

            output_path = out_dir / f"{paper_id}_errors.tex"
            _process_paper(paper_id, datasets, output_path, args.rerun)


if __name__ == "__main__":
    main()
