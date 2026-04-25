"""Step 02: detect the root cause of each replication divergence via isolated agent calls.

Runs THREE consistency checks — each in its own cwd — for divergences where data is
available. Divergences with data_available=missing are skipped (root cause already
determined by step 01).

Cost optimisations vs. the original design:
  - original_proof / agent_proof from step 01 are embedded directly in each prompt, so
    agents no longer need to explore code files.  CHECK 1 only reads paper.pdf;
    CHECK 2 reads paper.pdf + summary; CHECK 3 only reads summary.
  - Divergences with data_available=missing are excluded from all check prompts.
  - If ALL divergences are missing-data, all checks are skipped entirely.

CHECK 1  cwd=paper_vs_original_code/   documents: paper.pdf
         verifies: does the paper support `original_behavior`? (Stata proof provided)

CHECK 2  cwd=paper_vs_summary/         documents: paper.pdf + methodology_summary.json
         verifies: does the summary faithfully represent what the paper says?
                   (Stata proof provided; no need to read dofiles)

CHECK 3  cwd=summary_vs_agent/         documents: methodology_summary.json
         verifies: does the agent code implement what the summary says?
                   (Python proof provided; no need to read agent_code/)

Directory layout (--workspace)
-------------------------------
  paper_vs_original_code/
    paper.pdf
    dofiles/*.do
  paper_vs_summary/
    paper.pdf                         (same PDF)
    methodology_summary.json
  summary_vs_agent/
    methodology_summary.json          (same summary)
    agent_code/*.py

Usage
-----
    python 03_detect_error_source.py \\
        --comparison  PATH/divergences.json \\
        --workspace   PATH/error_source/ \\
        --output      PATH/error_source/divergences_enriched.json \\
        [--runner    codex]          # codex (default) or claude-code \\
        [--model     gpt-5.4] \\
        [--max-turns 30] \\
        [--timeout   600]

Output
------
    divergences_enriched.json — written by this script; original divergences.json
    fields preserved, each divergence extended with:
      paper_vs_original_code, paper_vs_original_code_note,
      paper_vs_summary, paper_vs_summary_note,
      summary_vs_agent, summary_vs_agent_note,
      [data_available, data_available_note]   # if data dir found
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------


def _run_claude_code(
    workspace: Path,
    prompt: str,
    model: str,
    max_turns: int,
    timeout: int,
    api_key: str | None = None,
) -> tuple[str, str, int]:
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
            cmd,
            cwd=str(workspace),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", f"Timed out after {timeout}s", -1
    except FileNotFoundError:
        return "", "claude binary not found — is Claude Code installed?", -2


def _run_codex(
    workspace: Path,
    prompt: str,
    model: str,
    max_turns: int,
    timeout: int,
    api_key: str | None = None,
) -> tuple[str, str, int]:
    cmd = [
        "codex",
        "exec",
        "--full-auto",
        "--json",
        "-m",
        model,
        "-C",
        str(workspace),
        "--skip-git-repo-check",
        prompt,
    ]
    env = os.environ.copy()
    if api_key:
        env["OPENAI_API_KEY"] = api_key
    try:
        result = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = (
            (exc.stdout or b"").decode(errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        return stdout, f"Timed out after {timeout}s", -1
    except FileNotFoundError:
        return "", "codex binary not found", -2


# ---------------------------------------------------------------------------
# Prompt builders — one per check type
# ---------------------------------------------------------------------------

_CHECK1_INSTRUCTIONS = """\
You are verifying whether the paper supports the original {original_language} code's behavior
for a set of replication divergences.

Your working directory contains:
  paper.pdf              — the published paper
  original_code_files/   — the original {original_language} replication code (for reference only)

For each divergence you are given:
  - `original_behavior`: what the {original_language} code does for this analysis step
  - `original_proof`: the exact {original_language} code snippet implementing this behavior
  - `original_location`: the file and line number in the original code

The original code is already provided — you do NOT need to re-read the code files.
Read paper.pdf to determine whether it explicitly states, implies, or is silent
about the behavior described in `original_behavior`.  Classify using exactly one
of these four verdicts:

  consistent   = paper and original code explicitly agree on this specific point
  contradicts  = direct contradiction: the paper explicitly says X AND the original
                 code explicitly does Y, where X ≠ Y
  omission     = the original code implements X, but the paper does not mention X at
                 all — no contradiction, the paper simply doesn't cover this detail

Rules:
- "contradicts" requires both documents to explicitly address the same point with
  different answers.  The paper being silent on an implementation detail is
  NEVER sufficient for "contradicts" — use "omission" instead.
- "omission" means the original code (the upstream document here) specifies something
  the paper (the downstream document) doesn't address.  Example: code computes
  monthly standardized values and averages them; paper describes the monthly formula
  but says nothing about aggregation → omission.
- "consistent" if both agree; prefer "omission" when one side is explicit and the other is silent.
- Your note must cite specific evidence: variable name, section heading, line
  number, or direct quote.
- You MUST produce exactly one of these three verdicts for every divergence id. Do not abstain or invent new labels.

EFFICIENCY: Do NOT read the entire paper exhaustively.  Use the divergence
description to identify the relevant section, check that section, form a
well-founded hypothesis, cite the evidence, and move on.
"""

_CHECK2_INSTRUCTIONS = """\
You are verifying whether the methodology summary accurately represents what
the paper says, for a set of replication divergences.

Your working directory contains:
  paper.pdf                  — the published paper
  methodology_summary.json   — the summary passed to the replicator agent

For each divergence you are given:
  - `original_behavior`: what the original code does for this analysis step
  - `original_proof`: the exact code snippet (for context on what to look for)
  - `original_location`: the file and line number in the original code

The original code is already provided — you do NOT need to read the code files.
Read paper.pdf and methodology_summary.json to determine whether the paper states
this behavior and whether the summary faithfully represents it.  Classify using
exactly one of these four verdicts:

  consistent   = summary faithfully represents the paper on this point, OR both
                 are silent (summary correctly captured the paper's silence)
  contradicts  = direct contradiction: paper explicitly says X AND summary
                 explicitly says Y ≠ X
  omission     = paper explicitly says X, but the summary does not mention X at
                 all — the summary dropped information the paper provided

Rules:
- "omission" means the paper (upstream) says something the summary (downstream)
  dropped.  This is the most common failure mode for this check.
- "contradicts" requires both documents to explicitly address the same point
  differently — not just one being silent where the other speaks.
- If neither the paper nor the summary addresses the point, mark "consistent"
  (they agree by silence — the break is not here).
- The summary is NOT expected to be more detailed than the paper.  If the paper
  states X at a conceptual level (e.g. "cluster at neighborhood level") and the
  summary conveys the same concept (e.g. "use neighborhood cluster identifiers"),
  mark "consistent" — even if the summary does not reproduce the exact variable
  names or implementation details that appear in the original code but not in the
  paper.  The standard is semantic faithfulness to what the paper says, not
  completeness of implementation detail.
- To distinguish "consistent" from "omission" at the conceptual level, ask:
  "Would a careful agent reading only the summary know that they should implement X
  (the concept the paper describes)?"  If yes → "consistent" (the concept was
  conveyed, even if less specific).  If no, because the summary says nothing about
  X at all → "omission".  Note: "omission" does not require that the agent would
  actively do the wrong thing; it is sufficient that the summary dropped a concept
  the paper stated, leaving the agent without guidance on that point.
- Your note must cite specific evidence from both documents.
- You MUST produce exactly one of these three verdicts for every divergence id. Do not abstain or invent new labels.

EFFICIENCY: Do NOT read documents exhaustively.  Use the divergence description
to locate the relevant section in each document, check it, form a well-founded
hypothesis, cite the evidence, and move on.
"""

_CHECK3_INSTRUCTIONS = """\
You are verifying whether the agent's Python code implements what the
methodology summary instructs, for a set of replication divergences.

Your working directory contains:
  methodology_summary.json   — the summary passed to the replicator agent
  agent_code/                — the agent's Python replication code (for reference only)

For each divergence you are given:
  - `agent_behavior`: what the Python code actually does for this analysis step
  - `agent_proof`: the exact Python code snippet implementing this behavior
  - `agent_location`: the file and line number in agent_code/

The Python code is already provided — you do NOT need to re-read agent_code/.
Read methodology_summary.json to determine whether it explicitly instructs,
implies, or is silent about the behavior described in `agent_behavior`.  Classify
using exactly one of these four verdicts:

  consistent   = agent follows the summary on this point, OR both are silent
                 (agent correctly followed the summary's omission)
  contradicts  = direct contradiction: summary explicitly says X AND agent code
                 explicitly does Y ≠ X
  omission     = summary explicitly instructs X, but the agent does not implement
                 X at all — agent omitted something the summary described

Rules:
- Each divergence also includes a `description` field explaining what the step
  is about.  Use this to find the relevant section in methodology_summary.json
  when agent_behavior is "ABSENT" or otherwise unclear.
- "omission" covers agent_behavior = "ABSENT": use `description` to identify
  what analysis step was supposed to be implemented, look it up in the summary,
  and if the summary describes it → omission.  Only mark "consistent" if the
  summary is also silent on that step.
- "contradicts" requires the summary to explicitly say X AND the agent to do
  something explicitly different — not just the agent being silent.
- IMPORTANT: if the summary does NOT mention this specific detail at all
  (e.g. the summary is silent about weighting, file choice, variable variant,
  clustering method, or any other implementation detail), mark "consistent" —
  the agent cannot be expected to follow guidance that was not provided.
  The agent making its own reasonable choice on an unspecified detail is NOT
  a contradiction or omission.
- Your note must cite specific evidence from both the summary and the Python code.
- You MUST produce exactly one of these three verdicts for every divergence id. Do not abstain or invent new labels.

EFFICIENCY: Work from the provided code snippets and the summary text.  Do NOT
exhaustively search through all files.  Form a well-founded hypothesis based on
the evidence given, cite it, and move on.
"""

_CHECK4_INSTRUCTIONS = """\
You are checking whether the data required by the original {original_language} code is
available to a Python replicator, for a set of divergences.

Your working directory is the replication data directory.  The original
{original_code_description} are in {code_files_path}.

## Step 1 — identify what the original code loads

For each divergence, read the original code files and find the data file(s)
loaded for that analysis step.

## Step 2 — distinguish raw inputs from code-constructed intermediates

Before checking whether a file exists, determine whether it is:

  (a) A RAW SOURCE FILE — brought in from outside (survey data, official
      statistics, downloaded datasets).  These must exist as-is for any
      replicator.

  (b) A CODE-CONSTRUCTED INTERMEDIATE — a file that is itself created by
      the original code in the same replication package.  A Python
      replicator would construct this from its own upstream steps, not load a
      pre-built file.

You can identify constructed intermediates by searching the original code for
save/export commands that write that filename.

## Step 3 — check availability

  • If the required file is a RAW SOURCE FILE: check whether it exists in the
    current directory.  If yes → available; if no → missing.

  • If the required file is a CODE-CONSTRUCTED INTERMEDIATE: instead check
    whether the raw source file(s) that feed into its construction are present.
    If those raw sources exist → available (the agent can construct the
    intermediate from them); if the raw sources are also absent → missing.

  available = the required data (raw or constructable from present sources)
              was accessible; the agent had what it needed and chose incorrectly
  missing   = the required raw data is absent; the agent could not have
              implemented this correctly regardless of effort

Every divergence must receive `available` or `missing` — no other values.
In your note, name the file(s) you checked and state whether each is a raw
source or a constructed intermediate.
"""

_OUTPUT_SCHEMA = """\
Write ONLY the following JSON to {output_path} — no markdown, no preamble.
IMPORTANT: any double-quote characters inside a string value MUST be escaped as \\".
Do NOT include raw double quotes inside note strings.

{{
  "verdicts": [
    {{"id": <int>, "verdict": "<consistent|contradicts|omission>", "note": "<one sentence citing specific evidence>"}},
    ...
  ]
}}
"""

_CHECK3_OUTPUT_SCHEMA = """\
Write ONLY the following JSON to {output_path} — no markdown, no preamble.
IMPORTANT: any double-quote characters inside a string value MUST be escaped as \\".
Do NOT include raw double quotes inside note strings.

{{
  "verdicts": [
    {{"id": <int>, "verdict": "<consistent|contradicts|omission>", "note": "<one sentence citing specific evidence>"}},
    ...
  ]
}}
"""

_DATA_OUTPUT_SCHEMA = """\
Write ONLY the following JSON to {output_path} — no markdown, no preamble:

{{
  "verdicts": [
    {{"id": <int>, "verdict": "<available|missing>", "note": "<one sentence naming the specific file(s) checked>"}},
    ...
  ]
}}
"""


def _build_check_prompt(
    instructions: str,
    divergences_subset: list[dict],
    output_path: Path,
    schema_template: str,
) -> str:
    disc_json = json.dumps(divergences_subset, indent=2)
    schema = schema_template.format(output_path=output_path)
    return (
        f"{instructions}\n\n"
        f"{schema}\n\n"
        f"Divergences:\n```json\n{disc_json}\n```"
    )


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

import re as _re

_VALID_ESCAPES = set('"\\\/bfnrtu')


def _fix_json_escapes(text: str) -> str:
    """Replace bare backslashes that form invalid JSON escape sequences with \\\\."""
    def _replace(m: "_re.Match") -> str:
        following = m.group(1)
        if following and following[0] in _VALID_ESCAPES:
            return m.group(0)   # valid escape — leave alone
        return "\\\\" + (following or "")
    return _re.sub(r"\\(.?)", _replace, text)


def _parse_verdicts_json(text: str) -> dict:
    """Parse verdicts JSON, retrying with progressively more aggressive repairs."""
    # 1. Normal parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Fix invalid backslash escapes
    try:
        return json.loads(_fix_json_escapes(text))
    except json.JSONDecodeError:
        pass

    # 3. Regex fallback — extract individual verdict objects even from broken JSON
    #    Captures: "id": N, "verdict": "...", "note": "..."  (note matched non-greedily)
    pattern = (
        r'"id"\s*:\s*(\d+)'
        r'.*?"verdict"\s*:\s*"([^"]+)"'
        r'.*?"note"\s*:\s*"((?:[^"\\]|\\.)*)"'
    )
    matches = _re.findall(pattern, text, _re.DOTALL)
    if matches:
        verdicts = [
            {"id": int(m[0]), "verdict": m[1], "note": m[2]}
            for m in matches
        ]
        print(f"  WARNING: JSON malformed — recovered {len(verdicts)} verdicts via regex")
        return {"verdicts": verdicts}

    raise json.JSONDecodeError("All repair attempts failed", text, 0)


# ---------------------------------------------------------------------------
# Check runner — executes one agent call, reads partial verdicts JSON
# ---------------------------------------------------------------------------


def _load_existing_verdicts(output_path: Path, check_name: str) -> dict[int, dict] | None:
    """Read an existing verdicts.json without running the agent."""
    if not output_path.exists():
        return None
    try:
        data = _parse_verdicts_json(output_path.read_text(encoding="utf-8"))
        verdicts = {v["id"]: {"verdict": v["verdict"], "note": v["note"]}
                    for v in data.get("verdicts", [])}
        print(f"\n--- {check_name} [SKIPPED — using existing {output_path.name}] ---")
        print(f"  -> {len(verdicts)} verdicts loaded")
        return verdicts
    except (json.JSONDecodeError, KeyError) as e:
        print(f"\n--- {check_name} [existing file unreadable: {e} — will rerun] ---")
        return None


def _run_check(
    check_name: str,
    cwd: Path,
    prompt: str,
    runner: str,
    model: str,
    max_turns: int,
    timeout: int,
    output_path: Path,
    api_key: str | None = None,
) -> dict[int, dict] | None:
    """
    Run one agent call and return {id: {verdict, note}} from the written JSON.
    Returns None if the agent failed to produce output.
    """
    print(f"\n--- {check_name} ---")
    print(f"  cwd:    {cwd}")
    print(f"  output: {output_path}")
    t0 = time.time()

    if runner == "claude-code":
        stdout, stderr, exit_code = _run_claude_code(cwd, prompt, model, max_turns, timeout, api_key)
    else:
        stdout, stderr, exit_code = _run_codex(cwd, prompt, model, max_turns, timeout, api_key)

    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s  (exit {exit_code})")
    if stderr:
        print(f"  stderr: {stderr[:300]}")

    if not output_path.exists():
        print(f"  WARNING: agent did not write {output_path.name}")
        fallback = output_path.with_suffix(".raw.txt")
        fallback.write_text(stdout, encoding="utf-8")
        print(f"  -> raw output saved to {fallback.name}")
        return None

    try:
        data = _parse_verdicts_json(output_path.read_text(encoding="utf-8"))
        verdicts = {v["id"]: {"verdict": v["verdict"], "note": v["note"]}
                    for v in data.get("verdicts", [])}
        print(f"  -> {len(verdicts)} verdicts read")
        return verdicts
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  WARNING: could not parse {output_path.name}: {e}")
        return None


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------


def _extract_divergences(comparison_data: dict) -> list[dict]:
    return comparison_data.get("divergences", comparison_data.get("discrepancies", []))


def _get_field(d: dict, new_name: str, old_name: str) -> str:
    return d.get(new_name, d.get(old_name, ""))


def _merge_verdicts(
    comparison_data: dict,
    check1: dict[int, dict] | None,
    check2: dict[int, dict] | None,
    check3: dict[int, dict] | None,
    check4: dict[int, dict] | None,
) -> dict:
    """Merge four partial verdict dicts into the enriched comparison_data."""
    enriched = dict(comparison_data)
    divs = _extract_divergences(enriched)
    key = "divergences" if "divergences" in enriched else "discrepancies"

    for d in divs:
        did = d["id"]
        if check1 and did in check1:
            d["paper_vs_original_code"]      = check1[did]["verdict"]
            d["paper_vs_original_code_note"] = check1[did]["note"]
        if check2 and did in check2:
            d["paper_vs_summary"]      = check2[did]["verdict"]
            d["paper_vs_summary_note"] = check2[did]["note"]
        if check3 and did in check3:
            d["summary_vs_agent"]      = check3[did]["verdict"]
            d["summary_vs_agent_note"] = check3[did]["note"]
        if check4 and did in check4:
            d["data_available"]      = check4[did]["verdict"]
            d["data_available_note"] = check4[did]["note"]

    enriched[key] = divs
    return enriched


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect the source of replication discrepancies using isolated agent calls."
    )
    parser.add_argument(
        "--comparison", required=True, help="Path to divergences.json."
    )
    parser.add_argument(
        "--workspace",
        required=True,
        help="Path to error_source/ directory (contains paper_vs_original_code/, etc.).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for the enriched output JSON.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Path to the full replication data directory (enables data availability check).",
    )
    parser.add_argument(
        "--runner",
        default="codex",
        choices=["claude-code", "codex"],
        help="Which CLI agent to use (default: codex).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (default: claude-opus-4-6 for claude-code, gpt-5.4 for codex).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (ANTHROPIC_API_KEY for claude-code, OPENAI_API_KEY for codex). "
             "Overrides any key already set in the environment.",
    )
    parser.add_argument(
        "--max-turns", type=int, default=30, help="Maximum agent turns per check (default: 30)."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Subprocess timeout in seconds per check (default: 600).",
    )
    parser.add_argument(
        "--rerun", action="store_true",
        help="Re-run all checks even if output already exists.",
    )
    parser.add_argument(
        "--rerun-checks", nargs="+", type=int, choices=[1, 2, 3, 4],
        metavar="N",
        help="Re-run only specific checks (e.g. --rerun-checks 2 3). "
             "Existing verdicts.json files for other checks are reused. "
             "Implies re-writing the final output.",
    )
    parser.add_argument(
        "--original-language", default="auto",
        help="Language of original code: stata, r, matlab, python, mixed, auto (default: auto).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = args.model or ("claude-opus-4-6" if args.runner == "claude-code" else "gpt-5.4")

    # Detect or use specified original code language
    from language_info import detect_language, get_info
    orig_lang = args.original_language
    if orig_lang == "auto":
        ws_path = Path(args.workspace).expanduser().resolve()
        orig_code = ws_path / "paper_vs_original_code" / "original_code_files"
        if orig_code.is_dir():
            orig_lang = detect_language(orig_code)
        else:
            orig_lang = "unknown"
    lang_info = get_info(orig_lang)
    print(f"  Original language: {lang_info['name']}")

    comparison_path = Path(args.comparison).expanduser().resolve()
    workspace = Path(args.workspace).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    for p, label in [
        (comparison_path, "--comparison"),
        (workspace, "--workspace"),
    ]:
        if not p.exists():
            sys.exit(f"ERROR: {label} does not exist: {p}")

    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else None
    if data_dir and not data_dir.exists():
        sys.exit(f"ERROR: --data-dir does not exist: {data_dir}")

    rerun_checks = set(args.rerun_checks or [])
    any_selective = bool(rerun_checks)

    if output_path.exists() and not args.rerun and not any_selective:
        # Re-run if input divergences.json is newer than output (step 01 re-ran)
        if comparison_path.stat().st_mtime > output_path.stat().st_mtime:
            print(f"  Re-running: input divergences.json is newer than enriched output")
        else:
            print(f"SKIP: {output_path} already exists. Use --rerun to overwrite.")
            return

    comparison_data = json.loads(comparison_path.read_text(encoding="utf-8"))
    discrepancies = _extract_divergences(comparison_data)
    print(f"\nLoaded {len(discrepancies)} divergences from {comparison_path.name}")
    print(f"Runner:    {args.runner} / {model}")
    print(f"Workspace: {workspace}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Early exit: all divergences already attributed via data_available=missing ---
    missing_ids = {d["id"] for d in discrepancies if d.get("data_available") == "missing"}
    checkable   = [d for d in discrepancies if d["id"] not in missing_ids]

    if not checkable:
        print(f"\nAll {len(discrepancies)} divergence(s) have data_available=missing — "
              f"root cause already determined. Skipping all checks.")
        enriched = _merge_verdicts(comparison_data, None, None, None, None)
        output_path.write_text(json.dumps(enriched, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n-> Enriched output written: {output_path}  ({output_path.stat().st_size:,} bytes)")
        return

    if missing_ids:
        print(f"  Note: {len(missing_ids)} divergence(s) with data_available=missing "
              f"excluded from checks (IDs: {sorted(missing_ids)}).")

    # Build per-check divergence lists — include proofs so agents don't need to re-read
    # code files.  Only include divergences where data is available (others are already
    # attributed).
    orig_beh_divs = [
        {
            "id": d["id"],
            "original_behavior": _get_field(d, "original_behavior", "reference_behavior"),
            "original_proof":    d.get("original_proof", ""),
            "original_location": d.get("original_location", {}),
        }
        for d in checkable
    ]
    agent_beh_divs = [
        {
            "id": d["id"],
            "description":   d.get("description", ""),
            "agent_behavior": _get_field(d, "agent_behavior", "candidate_behavior"),
            "agent_proof":    d.get("agent_proof", ""),
            "agent_location": d.get("agent_location", {}),
        }
        for d in checkable
    ]

    # Subdirectory paths
    dir1 = workspace / "paper_vs_original_code"
    dir2 = workspace / "paper_vs_summary"
    dir3 = workspace / "summary_vs_agent"

    for d, name in [(dir1, "paper_vs_original_code"), (dir2, "paper_vs_summary"),
                    (dir3, "summary_vs_agent")]:
        if not d.is_dir():
            sys.exit(f"ERROR: workspace subdirectory missing: {d}")

    # Intermediate output paths (written by each agent, read back by Python)
    out1 = dir1 / "verdicts.json"
    out2 = dir2 / "verdicts.json"
    out3 = dir3 / "verdicts.json"

    def _maybe_run(n: int, name: str, cwd: Path, prompt: str, out: Path) -> dict[int, dict] | None:
        if args.rerun or n in rerun_checks:
            return _run_check(name, cwd, prompt, args.runner, model, args.max_turns, args.timeout, out, args.api_key)
        return _load_existing_verdicts(out, name) or _run_check(
            name, cwd, prompt, args.runner, model, args.max_turns, args.timeout, out, args.api_key
        )

    # --- CHECK 1: paper vs original_behavior ---
    instructions1 = _CHECK1_INSTRUCTIONS.format(original_language=lang_info["name"])
    check1 = _maybe_run(
        1, "CHECK 1 — paper vs original code", dir1,
        _build_check_prompt(instructions1, orig_beh_divs, out1, _OUTPUT_SCHEMA), out1,
    )

    # --- CHECK 2: paper+summary vs original_behavior ---
    check2 = _maybe_run(
        2, "CHECK 2 — paper vs summary", dir2,
        _build_check_prompt(_CHECK2_INSTRUCTIONS, orig_beh_divs, out2, _OUTPUT_SCHEMA), out2,
    )

    # --- CHECK 3: summary+agent_code vs agent_behavior ---
    check3 = _maybe_run(
        3, "CHECK 3 — summary vs agent code", dir3,
        _build_check_prompt(_CHECK3_INSTRUCTIONS, agent_beh_divs, out3, _CHECK3_OUTPUT_SCHEMA), out3,
    )

    # --- CHECK 4: data availability (optional) ---
    check4 = None
    if data_dir:
        code_files_path = data_dir / "original_code_files"
        if not code_files_path.is_dir():
            code_files_path = data_dir / "dofiles"  # backward compat
        if not code_files_path.is_dir():
            code_files_path = data_dir
        out4 = data_dir / "verdicts_data.json"
        instructions4 = _CHECK4_INSTRUCTIONS.format(
            code_files_path=code_files_path,
            original_language=lang_info["name"],
            original_code_description=lang_info["code_description"],
        )
        check4 = _maybe_run(
            4, "CHECK 4 — data availability", data_dir,
            _build_check_prompt(instructions4, orig_beh_divs, out4, _DATA_OUTPUT_SCHEMA), out4,
        )

    # --- Merge and write final output ---
    enriched = _merge_verdicts(comparison_data, check1, check2, check3, check4)
    output_path.write_text(json.dumps(enriched, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> Enriched output written: {output_path}  ({output_path.stat().st_size:,} bytes)")

    # Summary table
    discs = _extract_divergences(enriched)
    print(f"\n{'=' * 84}")
    print(f"  {'ID':<4} {'Sev':<8} {'Data':<10} {'Paper↔Code':<14} {'Paper↔Summary':<16} {'Summary↔Agent':<16}")
    for d in discs:
        print(
            f"  {d['id']:<4} {d.get('severity', ''):<8} "
            f"{d.get('data_available', '?'):<10} "
            f"{d.get('paper_vs_original_code', '—'):<14} "
            f"{d.get('paper_vs_summary', '—'):<16} "
            f"{d.get('summary_vs_agent', '—'):<16}"
        )
    print()


if __name__ == "__main__":
    main()
