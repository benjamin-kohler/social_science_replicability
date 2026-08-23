"""Step 01: trace empirical output failures back to root code discrepancies.

Starts from cell-level failures in verification_report.json and traces each one
back to the specific code discrepancy that caused it. Supports cell-level tracking:
a table with multiple regression panels may have different root causes for different
columns, each traced separately.

Loop
----
  1. Sort failed outputs by severity (sign mismatches first, then mean % diff).
  2. For the first unresolved output: one focused agent call that
       - checks data availability for that specific output
       - locates the agent code producing those values
       - compares to original Stata code
       - identifies ALL distinct root causes within the output
       - maps each cause to the specific cells it explains (affected_cells)
       - lists which other outputs the same cause also explains (also_explains),
         either fully (string) or partially ({item_id, sections} object)
  3. Mark resolved outputs, record partial attributions for others.
  4. If a previously partial output is next, show already-attributed sections
     as context and ask the agent to find any remaining causes.
  5. Repeat until all outputs are resolved or the agent returns an empty list.

Output: divergences.json — consumed by 02_detect_error_source.py,
        03_summarize_errors.py, 04_overview_stats.py.
        Fields: affected_cells, explains_sections,
        also_explains (supports partial {item_id, sections} entries).

Usage
-----
    python3 01_trace_failures.py \\
        --code-dir            explainer_workspaces/10.1257_aer.20190565/codex/code \\
        --verification-report path/to/verification_report.json \\
        [--output             path/to/divergences.json] \\
        [--runner             codex] \\
        [--model              gpt-5.4] \\
        [--max-turns          40] \\
        [--timeout            600] \\
        [--rerun]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------

def _run_claude_code(workspace, prompt, model, max_turns, timeout, api_key=None):
    cmd = [
        "claude", "-p",
        "--output-format", "json",
        "--model", model,
        "--dangerously-skip-permissions",
        "--max-turns", str(max_turns),
        "--no-session-persistence",
        "--verbose",
        "--", prompt,
    ]
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    if api_key:
        env["ANTHROPIC_API_KEY"] = api_key
    try:
        result = subprocess.run(
            cmd, cwd=str(workspace),
            stdin=subprocess.DEVNULL, capture_output=True, text=True,
            timeout=timeout, env=env,
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", f"Timed out after {timeout}s", -1
    except FileNotFoundError:
        return "", "claude binary not found — is Claude Code installed?", -2


def _run_codex(workspace, prompt, model, max_turns, timeout, api_key=None):
    cmd = [
        "codex", "exec",
        "--full-auto", "--json",
        "-m", model,
        "-C", str(workspace),
        "--skip-git-repo-check",
        prompt,
    ]
    env = os.environ.copy()
    if api_key:
        env["OPENAI_API_KEY"] = api_key
    try:
        result = subprocess.run(
            cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True,
            timeout=timeout, env=env,
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = (exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        return stdout, f"Timed out after {timeout}s", -1
    except FileNotFoundError:
        return "", "codex binary not found", -2


def _run_agent(workspace, prompt, runner, model, max_turns, timeout, api_key=None):
    prompt = prompt.replace(chr(0), '')  # strip embedded null bytes from VR data
    if runner == "claude-code":
        return _run_claude_code(workspace, prompt, model, max_turns, timeout, api_key)
    return _run_codex(workspace, prompt, model, max_turns, timeout, api_key)


# ---------------------------------------------------------------------------
# Verification report helpers
# ---------------------------------------------------------------------------

def _failure_sort_key(item: dict) -> tuple:
    """Sort failures: sign mismatches first, then by mean % diff descending."""
    cells = (item.get("table_comparison") or {}).get("cell_comparisons") or []
    has_sign_mismatch = any(not c.get("sign_match", True) for c in cells)
    mean_diff = (item.get("numerical_differences") or {}).get("mean_difference_percent", 0)
    return (not has_sign_mismatch, -mean_diff)


def _format_cell_table(item: dict, max_rows: int = 15,
                       skip_columns: set[str] | None = None) -> str:
    """Format the worst-N cell comparisons as a readable table.

    skip_columns: set of column_label strings already attributed to a prior divergence.
    """
    cells = ((item.get("table_comparison") or {}).get("cell_comparisons") or [])
    if not cells:
        notes = item.get("comparison_notes", "")
        return f"  (no cell-level data available)\n  Notes: {notes}"

    if skip_columns:
        cells = [c for c in cells if c.get("column_label") not in skip_columns]
    if not cells:
        return "  (all failed cells already attributed to prior divergences)"

    sorted_cells = sorted(cells, key=lambda c: abs(c.get("percent_difference") or 0), reverse=True)
    shown = sorted_cells[:max_rows]
    n_total = len(cells)

    header = f"  {'Row':<35} {'Col':<30} {'Expected':>12} {'Actual':>12} {'%diff':>8} {'SignOK':>6}"
    sep = "  " + "-" * (35 + 30 + 12 + 12 + 8 + 6 + 10)
    rows = [header, sep]
    for c in shown:
        row_lbl = (c.get("row_label") or "")[:34]
        col_lbl = (c.get("column_label") or "")[:29]
        exp  = c.get("original_value")
        act  = c.get("replicated_value")
        pct  = c.get("percent_difference") or 0
        sign = "yes" if c.get("sign_match", True) else "NO"
        exp_s = f"{exp:>12.4g}" if isinstance(exp, (int, float)) else f"{'N/A':>12}"
        act_s = f"{act:>12.4g}" if isinstance(act, (int, float)) else f"{'N/A':>12}"
        rows.append(f"  {row_lbl:<35} {col_lbl:<30} {exp_s} {act_s} {pct:>7.1f}% {sign:>6}")
    if n_total > max_rows:
        rows.append(f"  ... ({n_total - max_rows} more cells not shown)")
    return "\n".join(rows)


def _format_already_attributed(prior: list[dict]) -> str:
    """Format a block describing sections already attributed to prior divergences."""
    if not prior:
        return ""
    lines = ["## Already attributed (do NOT re-explain these)"]
    for p in prior:
        lines.append(f"  Divergence {p['div_id']} ({p['divergence_type']}): "
                     f"{', '.join(p['sections'])}")
    lines.append("")
    return "\n".join(lines) + "\n"


def _format_remaining(failures: list[dict]) -> str:
    """Brief summary of each remaining unresolved failure."""
    if not failures:
        return "  (none — this is the last unresolved failure)"
    lines = []
    for item in failures:
        item_id = item.get("item_id", "?")
        grade   = item.get("grade", "?")
        nd      = item.get("numerical_differences") or {}
        mean_d  = nd.get("mean_difference_percent", 0)
        notes   = (item.get("comparison_notes") or "")[:150].replace("\n", " ")
        lines.append(f"  {item_id:<20} grade={grade}  mean_diff={mean_d:.0f}%  {notes}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-trace prompt
# ---------------------------------------------------------------------------

_TRACE_PROMPT_TEMPLATE = """\
You are an expert at tracing numerical replication failures back to their root code discrepancies.

Your working directory contains two subfolders:
  original_code/   -- original {original_language} replication package (code + data directory)
  agent_code/      -- agent-generated Python replicator

## Target failure
Item: {item_id}  |  Grade: {grade}  |  {n_failed} of {n_total} cells failed

{already_attributed_block}Failed cells still to explain (sorted by |% difference|):
{cell_table}

## Your tasks

1. DATA AVAILABILITY — Check whether the data files required to produce {item_id} exist in
   the data directory inside original_code/. If any required input file is absent from the
   replication package, set data_available = "missing" and describe the missing file(s).
   Otherwise set data_available = "available".

2. AGENT CODE — In agent_code/*.py, find the code that computes values for the REMAINING
   cells listed above (ignore already-attributed sections).
   If no such code exists, set agent_behavior = "<description of what is missing>".

3. ORIGINAL CODE — In original_code/{original_file_glob}, find the equivalent {original_language} code.

4. DISCREPANCIES — For the REMAINING cells only, identify all distinct root causes.
   Most outputs have one, but multi-panel tables with different specifications may have
   several (e.g. columns 1–3 OLS wrong clustering; columns 4–6 IV wrong instrument).
   If all remaining cells are already explained, return an empty divergences array.
   List one entry per distinct root cause — do NOT split trivial variations.

5. SECTION MAPPING — For each discrepancy, list which rows/columns/panels it explains
   (e.g. "All columns", "Columns 1–3 (OLS)", "Panel B"). Maps many cells to one cause.
   Also list each specific affected cell from the cell table above as
   {{"item_id": "{item_id}", "row_label": "...", "column_label": "..."}} using the exact
   labels shown. Include all cells you believe this discrepancy explains.

6. ALSO EXPLAINS — For each discrepancy, list which other failures it also explains.
   Each entry is either:
     - A plain string "<item_id>" if the discrepancy explains the entire other item, OR
     - An object {{"item_id": "<item_id>", "sections": "<which part>"}} if only partial.
   Be conservative: only include items you are confident share the exact same root cause.

## Other unresolved failures
{remaining_block}

## Divergence taxonomy (use exactly one code)
  S1  Wrong model specification      -- incorrect FE, clustering level, or SE type
  S2  Wrong estimator / inference    -- wrong estimator (OLS vs IV) or missing inference step
  S3  Data source substitution       -- proxy used / required dataset absent from package
  S4  Wrong sample restriction       -- filter missing, wrong condition, or wrong subset
  S5  Wrong variable construction    -- outcome/predictor coded differently than reference
  S6  Missing analysis component     -- required step entirely omitted
  S8  Wrong merge / transform logic  -- wrong join type, key, duplicate handling, or reshape
  S9  Wrong sequencing               -- steps in wrong order, changing results
  S0  Other                          -- does not fit any category above

## Severity
  minor    -- unlikely to materially affect point estimates or conclusions
  medium   -- could shift estimates noticeably; sign/significance probably stable
  critical -- likely changes sign, significance, or core conclusion of a main result

## Output
Write a JSON object with a "divergences" array to {output_path}.
One entry per distinct root cause found in {item_id} (usually 1, occasionally 2–3).
Write ONLY valid JSON — no markdown fences, no preamble, no comments.
Escape all special characters properly inside string values.

{{
  "divergences": [
    {{
      "id": {divergence_id},   (use sequential integers starting here for additional entries)
      "output": "{item_id}",
      "description": "<one sentence: what is different>",
      "original_behavior": "<what the original {original_language} code does on this specific point>",
      "agent_behavior": "<what Python does, or the ABSENT description>",
      "original_proof": "<short verbatim {original_proof_label}>",
      "agent_proof": "<short verbatim Python snippet, or ABSENT>",
      "original_location": {{"file": "<{original_file_ext}>", "line": "<line or range>"}},
      "agent_location": {{"file": "<filename.py or ABSENT>", "line": "<line or ABSENT>"}},
      "divergence_type": "<S0|S1|S2|S3|S4|S5|S6|S8|S9>",
      "severity": "<minor|medium|critical>",
      "data_available": "<available|missing>",
      "data_available_note": "<one sentence naming specific file(s) checked or found absent>",
      "explains_sections": ["<e.g. 'All columns', 'Columns 1-3 (OLS)', 'Panel B'>"],
      "affected_cells": [
        {{"item_id": "{item_id}", "row_label": "<exact row label from the cell table>", "column_label": "<exact column label>"}}
      ],
      "also_explains": ["<item_id>", {{"item_id": "<item_id>", "sections": "<which part>"}}]
    }}
  ]
}}
"""


def _build_trace_prompt(item: dict, remaining: list[dict],
                        divergence_id: int, output_path: Path,
                        prior_attributions: list[dict] | None = None,
                        lang_info: dict | None = None) -> str:
    """Build the per-trace prompt.

    prior_attributions: list of {div_id, divergence_type, sections, columns} dicts
        describing sections of this item already attributed to earlier divergences.
    lang_info: dict from language_info.LANGUAGE_INFO for the original code language.
    """
    item_id  = item.get("item_id", "?")
    grade    = item.get("grade", "?")
    nd       = item.get("numerical_differences") or {}
    n_total  = nd.get("num_cells_compared", 0)

    prior = prior_attributions or []
    # Build set of already-attributed column labels (best-effort from sections text)
    skip_cols: set[str] = set()
    cells_all = (item.get("table_comparison") or {}).get("cell_comparisons") or []
    for p in prior:
        # If sections says "All columns", skip everything
        if any("all" in s.lower() for s in p.get("sections", [])):
            skip_cols = {c.get("column_label", "") for c in cells_all}
            break
        # Otherwise collect explicit column labels mentioned in sections text
        for c in cells_all:
            col = c.get("column_label", "")
            if any(col and col in s for s in p.get("sections", [])):
                skip_cols.add(col)

    remaining_cells = [c for c in cells_all if c.get("column_label") not in skip_cols]
    n_failed = sum(1 for c in remaining_cells if c.get("grade", "A") not in ("A", "B"))
    cell_table = _format_cell_table(item, skip_columns=skip_cols)
    n_shown    = min(15, len(remaining_cells))

    li = lang_info or {}
    return _TRACE_PROMPT_TEMPLATE.format(
        item_id=item_id,
        grade=grade,
        n_failed=n_failed,
        n_total=n_total,
        n_shown=n_shown,
        cell_table=cell_table,
        already_attributed_block=_format_already_attributed(prior),
        remaining_block=_format_remaining(remaining),
        output_path=output_path,
        divergence_id=divergence_id,
        original_language=li.get("name", "Stata"),
        original_file_glob=li.get("file_glob", "*.do"),
        original_proof_label=li.get("proof_label", "Stata snippet"),
        original_file_ext=li.get("file_ext_example", "filename.do"),
    )


# ---------------------------------------------------------------------------
# JSON repair helpers  (from 03_detect_error_source.py pattern)
# ---------------------------------------------------------------------------

def _fix_json_escapes(text: str) -> str:
    def _fix_string(m):
        s = m.group(0)
        inner = s[1:-1]
        inner = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', inner)
        return '"' + inner + '"'
    return re.sub(r'"(?:[^"\\]|\\.)*"', _fix_string, text)


def _parse_trace_result(text: str) -> list[dict]:
    """Parse trace_result JSON; returns a list of divergence dicts."""
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)

    def _extract_list(parsed: dict | list) -> list[dict]:
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            # New format: {"divergences": [...]}
            if "divergences" in parsed:
                return parsed["divergences"] if isinstance(parsed["divergences"], list) else [parsed]
            # Old single-object format (backwards compat)
            return [parsed]
        return []

    for attempt in (text, _fix_json_escapes(text)):
        try:
            return _extract_list(json.loads(attempt))
        except json.JSONDecodeError:
            pass

    # Regex fallback — extract one entry (best-effort)
    def _first(pat):
        m = re.search(pat, text, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    also_m = re.search(r'"also_explains"\s*:\s*\[([^\]]*)\]', text)
    also = []
    if also_m:
        also = [s.strip().strip('"') for s in also_m.group(1).split(",") if s.strip().strip('"')]

    return [{
        "output":             _first(r'"output"\s*:\s*"([^"]*)"'),
        "description":        _first(r'"description"\s*:\s*"((?:[^"\\]|\\.)*)"'),
        "original_behavior":  _first(r'"original_behavior"\s*:\s*"((?:[^"\\]|\\.)*)"'),
        "agent_behavior":     _first(r'"agent_behavior"\s*:\s*"((?:[^"\\]|\\.)*)"'),
        "original_proof":     _first(r'"original_proof"\s*:\s*"((?:[^"\\]|\\.)*)"'),
        "agent_proof":        _first(r'"agent_proof"\s*:\s*"((?:[^"\\]|\\.)*)"'),
        "original_location":  {"file": _first(r'"original_location"[^}]*"file"\s*:\s*"([^"]*)"'), "line": "?"},
        "agent_location":     {"file": _first(r'"agent_location"[^}]*"file"\s*:\s*"([^"]*)"'),   "line": "?"},
        "divergence_type":    _first(r'"divergence_type"\s*:\s*"([^"]*)"'),
        "severity":           _first(r'"severity"\s*:\s*"([^"]*)"'),
        "data_available":     _first(r'"data_available"\s*:\s*"([^"]*)"'),
        "data_available_note":_first(r'"data_available_note"\s*:\s*"((?:[^"\\]|\\.)*)"'),
        "explains_sections":  [],
        "also_explains":      also,
        "_parse_fallback":    True,
    }]


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _make_summary(divergences: list[dict]) -> dict:
    by_sev  = {"critical": 0, "medium": 0, "minor": 0}
    by_type = {k: 0 for k in ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S8", "S9")}
    for d in divergences:
        s = d.get("severity", "")
        if s in by_sev:
            by_sev[s] += 1
        t = d.get("divergence_type", "")
        if t in by_type:
            by_type[t] += 1
    return {
        "by_severity": by_sev,
        "by_type": by_type,
        "overall_assessment": (
            f"{len(divergences)} divergence(s) identified via output-grounded tracing. "
            f"Critical: {by_sev['critical']}  Medium: {by_sev['medium']}  Minor: {by_sev['minor']}."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace empirical output failures back to code discrepancies."
    )
    parser.add_argument("--code-dir", required=True,
        help="Folder containing original_code/ and agent_code/ subfolders.")
    parser.add_argument("--verification-report", required=True,
        help="Path to verification_report.json with cell-level output comparisons.")
    parser.add_argument("--output", default=None,
        help="Output path for divergences.json (default: {code-dir}/divergences.json).")
    parser.add_argument("--paper-id", default=None)
    parser.add_argument("--agent",    default=None)
    parser.add_argument("--runner",   default="codex", choices=["claude-code", "codex"])
    parser.add_argument("--model",    default=None)
    parser.add_argument("--api-key",  default=None,
        help="API key for the chosen runner: Anthropic key for claude-code, "
             "OpenAI key for codex. Overrides any key set in the environment.")
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--timeout",   type=int, default=600,
        help="Per-call timeout in seconds (default: 600).")
    parser.add_argument("--original-language", default="auto",
        help="Language of original code: stata, r, matlab, python, mixed, auto (default: auto).")
    parser.add_argument("--rerun", action="store_true",
        help="Re-run even if divergences.json already exists.")
    args = parser.parse_args()

    model    = args.model or ("claude-opus-4-6" if args.runner == "claude-code" else "gpt-5.4")
    code_dir = Path(args.code_dir).expanduser().resolve()

    if not code_dir.is_dir():
        sys.exit(f"ERROR: --code-dir does not exist: {code_dir}")
    for sub in ("original_code", "agent_code"):
        if not (code_dir / sub).is_dir():
            sys.exit(f"ERROR: {code_dir / sub} not found — run 00_prep_setup.py first")

    # Infer paper_id / agent from path:  explainer_workspaces/{paper_id}/{agent}/code/
    parts    = code_dir.parts
    agent    = args.agent    or (parts[-2] if len(parts) >= 2 else "unknown")
    paper_id = args.paper_id or (parts[-3] if len(parts) >= 3 else "unknown")

    # Detect or use specified original code language
    from language_info import detect_language, get_info
    orig_lang = args.original_language
    if orig_lang == "auto":
        orig_lang = detect_language(code_dir / "original_code")
    lang_info = get_info(orig_lang)
    print(f"  Original language: {lang_info['name']}")

    output_path = Path(args.output).expanduser().resolve() if args.output else (code_dir / "divergences.json")

    if output_path.exists() and not args.rerun:
        # Check for parse failures — re-run if any placeholder divergences exist
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            has_failures = any(
                "[Parse failed]" in (d.get("description") or "")
                for d in existing.get("divergences", [])
            )
            if not has_failures:
                print(f"SKIP: {output_path} already exists (no parse failures). Use --rerun to overwrite.")
                return
            print(f"  Re-running: {output_path} has parse failures")
        except Exception:
            print(f"SKIP: {output_path} already exists. Use --rerun to overwrite.")
            return

    vr_path = Path(args.verification_report).expanduser().resolve()
    if not vr_path.exists():
        sys.exit(f"ERROR: --verification-report does not exist: {vr_path}")
    vr = json.loads(vr_path.read_text(encoding="utf-8"))

    # Collect failures (items where key_findings_match is False)
    all_items = vr.get("item_verifications", [])
    all_failures = sorted(
        [item for item in all_items if not item.get("key_findings_match", True)],
        key=_failure_sort_key,
    )

    n_passing = sum(1 for i in all_items if i.get("key_findings_match", False))
    print(f"\nRunner:    {args.runner} / {model}")
    print(f"Code dir:  {code_dir}")
    print(f"Paper:     {paper_id}  |  Agent: {agent}")
    print(f"VR:        {vr_path.name}  ({len(all_failures)} failed / {n_passing} passed items)")
    print(f"Output:    {output_path}")

    if not all_failures:
        print("\nNo failures to trace — all items passed. Writing empty divergences.json.")
        result = {
            "paper_id": paper_id, "agent": agent, "n_divergences": 0,
            "divergences": [],
            "summary": _make_summary([]),
        }
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return

    # ── Tracing loop ───────────────────────────────────────────────────────────
    resolved: set[str] = set()          # fully resolved item_ids
    # partial_attrs[item_id] = list of {div_id, divergence_type, sections, columns}
    partial_attrs: dict[str, list[dict]] = {}
    divergences: list[dict] = []
    failures = list(all_failures)       # mutable queue
    max_iters = len(all_failures) * 3   # allow re-visits for partial items

    for iteration in range(max_iters):
        if not failures:
            break

        target = failures[0]
        item_id = target.get("item_id", f"item_{iteration+1}")
        div_id  = len(divergences) + 1
        trace_out = code_dir / f"trace_result_{div_id}.json"

        prior = partial_attrs.get(item_id, [])
        prompt = _build_trace_prompt(target, failures[1:], div_id, trace_out,
                                     prior_attributions=prior,
                                     lang_info=lang_info)

        print(f"\n--- Trace {div_id}: {item_id} ({len(failures)} unresolved remaining) ---")
        print(f"  output: {trace_out.name}")
        t0 = time.time()

        stdout, stderr, exit_code = _run_agent(
            code_dir, prompt, args.runner, model, args.max_turns, args.timeout,
            api_key=args.api_key,
        )
        elapsed = time.time() - t0
        print(f"  done in {elapsed:.1f}s  (exit {exit_code})")
        if stderr:
            print(f"  STDERR: {stderr[:500]}")
        if stdout:
            print(f"  STDOUT (last 300): ...{stdout[-300:]}")

        # Parse result — returns a list of divergence dicts
        result_list = None
        if trace_out.exists():
            try:
                raw_text = trace_out.read_text(encoding="utf-8")
                result_list = _parse_trace_result(raw_text)
            except Exception as e:
                print(f"  WARNING: could not parse {trace_out.name}: {e}")

        if not result_list:
            print(f"  WARNING: no output for {item_id} — inserting placeholder")
            result_list = [{
                "output": item_id,
                "description": f"[Parse failed] Divergence for {item_id}",
                "original_behavior": "?", "agent_behavior": "?",
                "original_proof": "?", "agent_proof": "?",
                "original_location": {"file": "?", "line": "?"},
                "agent_location":    {"file": "?", "line": "?"},
                "divergence_type": "S0", "severity": "critical",
                "data_available": "available", "data_available_note": "parse failed — assumed available",
                "explains_sections": [], "also_explains": [],
            }]

        # Empty list → agent says all remaining cells already attributed
        if not result_list:
            resolved.add(item_id)
            print(f"  -> 0 new divergences — {item_id} fully attributed")
            failures = [f for f in failures if f.get("item_id") not in resolved]
            continue

        # Assign sequential IDs and accumulate
        fully_resolved_now = []
        for d in result_list:
            d["id"] = len(divergences) + 1
            d.setdefault("output", item_id)
            divergences.append(d)

            sections = d.get("explains_sections") or []

            # Record this attribution for the target item
            partial_attrs.setdefault(item_id, []).append({
                "div_id": d["id"],
                "divergence_type": d.get("divergence_type", "?"),
                "sections": sections,
            })

            # Process also_explains — support both strings and {item_id, sections} objects
            for entry in (d.get("also_explains") or []):
                if isinstance(entry, str):
                    # Full resolution of another item
                    resolved.add(entry)
                    if entry not in fully_resolved_now:
                        fully_resolved_now.append(entry)
                elif isinstance(entry, dict):
                    other_id = entry.get("item_id", "")
                    other_sections = entry.get("sections", "")
                    if other_id:
                        partial_attrs.setdefault(other_id, []).append({
                            "div_id": d["id"],
                            "divergence_type": d.get("divergence_type", "?"),
                            "sections": [other_sections] if other_sections else [],
                        })
                        # If sections = "All" or empty (meaning all), fully resolve
                        if not other_sections or "all" in other_sections.lower():
                            resolved.add(other_id)
                            if other_id not in fully_resolved_now:
                                fully_resolved_now.append(other_id)

        # Target item is fully resolved after processing all its divergences
        resolved.add(item_id)

        print(f"  -> {len(result_list)} divergence(s)  "
              f"fully resolved: {[item_id] + fully_resolved_now}")
        for d in result_list:
            print(f"     [{d['id']}] type={d.get('divergence_type','?')}  "
                  f"sev={d.get('severity','?')}  "
                  f"data={d.get('data_available','?')}  "
                  f"sections={d.get('explains_sections','?')}")

        failures = [f for f in failures if f.get("item_id") not in resolved]

    if failures:
        print(f"\nWARNING: {len(failures)} failure(s) still unresolved after {max_iters} iterations:")

        for f in failures:
            print(f"  - {f.get('item_id')}")

    # ── Write divergences.json ──────────────────────────────────────────────────
    # Clean up intermediate trace_result_N.json files
    for f in code_dir.glob("trace_result_*.json"):
        f.unlink(missing_ok=True)
    final = {
        "paper_id": paper_id,
        "agent":    agent,
        "n_divergences": len(divergences),
        "divergences": divergences,
        "summary":  _make_summary(divergences),
    }
    output_path.write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{'='*60}")
    print(f"  Divergences:  {len(divergences)}")
    print(f"  Critical: {final['summary']['by_severity']['critical']}  "
          f"Medium: {final['summary']['by_severity']['medium']}  "
          f"Minor: {final['summary']['by_severity']['minor']}")
    n_resolved = sum(1 for f in all_failures if f.get("item_id") in resolved)
    print(f"  Resolved {n_resolved} of {len(all_failures)} failed items")
    print(f"-> Output: {output_path}  ({output_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
