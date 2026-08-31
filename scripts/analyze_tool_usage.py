#!/usr/bin/env python3
"""Analyze tool-use behavior across benchmark runs.

Parses each harness's per-run log into a normalized event stream and reports
tool-call composition, main-vs-subagent split, and per-metric summaries

"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Shared with analyze_i4rep_results.py so both scripts agree on what counts
# as a valid run. "approach" there == "harness" here; same concept.
from analyze_i4rep_results import (
    _parse_approach_from_dirname,
    APPROACH_ORDER_RAW,
    APPROACH_LABELS,
    APPROACH_MODEL_ORDER,
)

RESULTS_DIR = Path(os.environ.get(
    "I4REPLICATE_RESULTS_DIR", "data/i4replicate/results",
))
DEFAULT_PLOTS_DIR = Path(os.environ.get(
    "TOOL_USAGE_PLOTS_DIR", "analysis_output/tool_usage",
))
HARNESSES: tuple[str, ...] = tuple(APPROACH_ORDER_RAW)


@dataclass
class Event:
    """One tool invocation, normalized across harnesses."""
    tool: str        # bucket from CANONICAL_TOOLS
    raw_tool: str    # tool name exactly as logged
    args: dict       # raw input/arguments dict
    agent: str = "main"  # "main" or a subagent_type (only claude-code exposes subagent internals)

    def command(self) -> str:
        return self.args.get("command") or "" if isinstance(self.args, dict) else ""

    def file_path(self) -> str:
        if not isinstance(self.args, dict):
            return ""
        return (
            self.args.get("file_path")       # claude-code
            or self.args.get("filePath")     # opencode
            or self.args.get("path")         # generic fallback
            or ""
        )


@dataclass
class Run:
    paper: str
    model: str
    harness: str
    workspace: Path
    events: list[Event] = field(default_factory=list)
    log_missing: bool = False
    excluded_reason: str = ""  # non-empty → skip this run in all aggregations



# Closed set. Unknown raw names become "other" and are surfaced in the summary
# so we never silently count an unknown tool as a read / edit / execution.
CANONICAL_TOOLS = ("bash", "read", "edit", "write", "grep", "glob", "planning", "subagent", "other")

_TOOL_ALIASES: dict[str, str] = {
    "bash": "bash",
    "shell": "bash",
    "command_execution": "bash",      # codex run_log.jsonl
    "read": "read",
    "edit": "edit",
    "multiedit": "edit",
    "str_replace_editor": "edit",
    "apply_patch": "edit",            # opencode/codex patch-based edit
    "write": "write",
    "grep": "grep",
    "glob": "glob",
    "todowrite": "planning",
    "todoread": "planning",
    "todo": "planning",
    "todo_list": "planning",          # codex run_log.jsonl item type
    "enterplanmode": "planning",
    "exitplanmode": "planning",
    "agent": "subagent",              # claude-code Agent
    "task": "subagent",               # claude-code Task, opencode task
    "taskoutput": "subagent",
    "taskstop": "subagent",
}


def _norm_tool(raw: str) -> str:
    return _TOOL_ALIASES.get((raw or "").strip().lower(), "other")


# ---------- parsers (one per harness) ----------

# Accumulates unknown event/block types encountered during parsing.
# Printed at end of main() so drops are never silent.
_unknown_events: Counter = Counter()

# Known non-tool event/block types per harness — explicitly skipped, not counted.
_CC_SKIP_ENTRY_TYPES  = frozenset({"system", "user", "result", "rate_limit_event"})
_CC_SKIP_BLOCK_TYPES  = frozenset({"text", "thinking"})
_OC_SKIP_EVENT_TYPES  = frozenset({"step_start", "step_finish", "text"})
_CDX_SKIP_ITEM_TYPES  = frozenset({"agent_message", "message", "reasoning"})
_SWE_SKIP_ROLES       = frozenset({"system", "user", "tool", "exit"})

def parse_claude_code(log_path: Path) -> list[Event]:
    """Two passes: record spawn_id->subagent_type from main-stream Agent/Task
    calls, then emit events tagging each by parent_tool_use_id lookup."""
    events: list[Event] = []
    try:
        data = json.loads(log_path.read_text())
    except Exception as e:
        print(f"[warn] claude-code parse failed: {log_path}: {e}")
        return events

    spawn_types: dict[str, str] = {}  # tool_use_id -> subagent_type
    for entry in data:
        etype = entry.get("type")
        if etype not in {"assistant"} | _CC_SKIP_ENTRY_TYPES:
            _unknown_events[f"claude-code:entry:{etype}"] += 1
        if etype != "assistant" or entry.get("parent_tool_use_id"):
            continue
        for b in (entry.get("message") or {}).get("content") or []:
            if b.get("type") == "tool_use" and b.get("name") in ("Agent", "Task"):
                sid = b.get("id")
                st = (b.get("input") or {}).get("subagent_type") or "subagent"
                if sid:
                    spawn_types[sid] = st

    for entry in data:
        if entry.get("type") != "assistant":
            continue
        ptu = entry.get("parent_tool_use_id")
        agent = spawn_types.get(ptu, "subagent") if ptu else "main"
        for block in (entry.get("message") or {}).get("content") or []:
            btype = block.get("type")
            if btype == "tool_use":
                raw = block.get("name", "")
                events.append(Event(
                    tool=_norm_tool(raw), raw_tool=raw,
                    args=block.get("input") or {}, agent=agent,
                ))
            elif btype not in _CC_SKIP_BLOCK_TYPES:
                _unknown_events[f"claude-code:block:{btype}"] += 1
    return events


def parse_opencode(log_path: Path) -> list[Event]:
    events: list[Event] = []
    with log_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            otype = obj.get("type")
            if otype == "tool_use":
                part = obj.get("part") or {}
                raw = part.get("tool", "")
                state = part.get("state") or {}
                events.append(
                    Event(tool=_norm_tool(raw), raw_tool=raw, args=state.get("input") or {})
                )
            elif otype not in _OC_SKIP_EVENT_TYPES:
                _unknown_events[f"opencode:event:{otype}"] += 1
    return events


def parse_codex(log_path: Path) -> list[Event]:
    """Codex run_log.jsonl: collect one item.completed snapshot per unique item ID.

    Every item (command_execution, todo_list, web_search) has both item.started
    and item.completed. file_change only has item.completed. Using completed for
    all items — keyed by ID to deduplicate — captures everything with no
    double-counting and no omissions.
    """
    # First pass: collect the item.completed snapshot for each ID (preserves order)
    completed: dict[str, dict] = {}
    with log_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("type") != "item.completed":
                continue
            item = obj.get("item") or {}
            iid = item.get("id")
            if iid:
                completed[iid] = item

    events: list[Event] = []
    for item in completed.values():
        raw = item.get("type", "")
        if not raw or raw in _CDX_SKIP_ITEM_TYPES:
            continue
        if raw == "file_change":
            # Codex built-in editor writes — content not in event, read from disk
            changes = [c for c in (item.get("changes") or [])
                       if c.get("kind") in ("add", "update")]
            if changes:
                events.append(Event(tool="write", raw_tool="file_change",
                                    args={"changes": changes}))
        elif raw == "command_execution":
            events.append(Event(tool=_norm_tool(raw), raw_tool=raw,
                                args={"command": item.get("command") or ""}))
        else:
            events.append(Event(tool=_norm_tool(raw), raw_tool=raw, args={}))
    return events


def parse_swe_agent(log_path: Path) -> list[Event]:
    """swe-agent: trajectory.json, tool_calls are typically a single 'bash' function."""
    events: list[Event] = []
    try:
        data = json.loads(log_path.read_text())
    except Exception as e:
        print(f"[warn] swe-agent parse failed: {log_path}: {e}")
        return events
    for msg in data.get("messages") or []:
        role = msg.get("role")
        if role == "assistant":
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                raw = fn.get("name", "")
                arg_str = fn.get("arguments") or "{}"
                try:
                    args = json.loads(arg_str) if isinstance(arg_str, str) else arg_str
                except json.JSONDecodeError:
                    args = {"_raw": arg_str}
                events.append(Event(tool=_norm_tool(raw), raw_tool=raw, args=args or {}))
        elif role not in _SWE_SKIP_ROLES:
            _unknown_events[f"swe-agent:role:{role}"] += 1
    return events


# ---------- discovery ----------

_NON_RUN_DIRS = frozenset({"summaries", "judge_results"})


def discover_runs(results_dir: Path) -> list[Run]:
    """Same directory walk as analyze_i4rep_results.load_results, minus the
    JSON loading. Uses the shared _parse_approach_from_dirname so both scripts
    agree on which dirs are valid runs (tagged alternates like
    ..._claude-code.contaminated_20260330 are rejected by the exact-match check).
    """
    runs: list[Run] = []
    for paper_dir in sorted(results_dir.iterdir()):
        if not paper_dir.is_dir():
            continue
        for run_dir in sorted(paper_dir.iterdir()):
            if not run_dir.is_dir() or run_dir.name in _NON_RUN_DIRS:
                continue
            parsed = _parse_approach_from_dirname(run_dir.name, paper_dir.name)
            if parsed is None:
                continue
            model, harness = parsed
            runs.append(Run(
                paper=paper_dir.name, model=model, harness=harness,
                workspace=run_dir / "workspace",
            ))
    return runs


_HARNESS_LOG_SPECS: dict[str, tuple[str, callable]] = {
    "claude-code": ("run_log.json", parse_claude_code),
    "opencode":    ("run_log.jsonl", parse_opencode),
    "codex":       ("run_log.jsonl", parse_codex),
    "swe-agent":   ("trajectory.json", parse_swe_agent),
}


def parse_run(run: Run) -> Run:
    log_name, parser = _HARNESS_LOG_SPECS[run.harness]
    p = run.workspace / log_name
    # Tier-2 releases keep the SWE-Agent trajectory with the other preserved
    # replicator logs in ``explainer_workspace``.  Accept that canonical copy
    # so the trace figures reproduce without duplicating a large JSON file.
    if run.harness == "swe-agent" and not p.exists():
        release_copy = run.workspace.parent / "explainer_workspace" / "replicator_trajectory.json"
        if release_copy.exists():
            p = release_copy
    if not p.exists():
        run.log_missing = True
        return run
    run.events = parser(p)
    # For codex, supplement run_log.jsonl with apply_patch content from
    # session_rollout.jsonl — this is the only way to measure model-generated
    # write chars for the built-in editor. See _augment_codex_with_rollout.
    if run.harness == "codex":
        _augment_codex_with_rollout(run)
    return run


# ---------- codex session_rollout augmentation ----------
#
# Why this exists:
# Codex exposes two distinct file-write mechanisms:
#   1. Bash heredocs like `cat > file.py <<'EOF' ... EOF` — the model's output
#      text IS the bash command text, so it's captured in run_log.jsonl's
#      command_execution items.
#   2. The built-in editor (apply_patch) — the runtime emits a file_change
#      event showing WHICH files were touched, but NOT the patch text the
#      model generated. That content lives only in session_rollout.jsonl
#      (the internal Codex session log).
#
# For Codex GPT-5.4 in particular, ~95% of file writes route through (2),
# so ignoring session_rollout means we vastly undercount model-generated
# write chars. This matters for any "how much did the model output"
# analysis.
#
# Why session_rollout is tricky:
# codex_runner.py copies session_rollout.jsonl into the workspace using an
# mtime heuristic (take the most recently modified rollout in ~/.codex/
# sessions/ after the run started). Under parallel execution, this
# occasionally captures the WRONG session — a different paper, a different
# harness's explainer agent, or a run that happened to touch the sessions
# dir at a similar time. Using an unverified rollout would silently mix
# content from a foreign run into our measurements.
#
# The verification protocol:
# The first line of session_rollout.jsonl is always a `session_meta` event
# with `payload.cwd` recording the working directory at session start.
# We check that cwd matches the run's (paper, run_dir_name) identity after
# stripping the parent-directory prefix (which may differ due to workspace
# migrations) and archive tag suffixes (e.g. .contaminated_YYYYMMDD).
# Audit across all 95 active codex runs shows 93 verify cleanly and 2 are
# contaminated with another run's session. We exclude the contaminated 2
# entirely — keeping them at zero write chars would be misleading noise,
# and any heuristic patch-up would introduce another source of error.


_ROLLOUT_SUFFIX_RE = re.compile(r"(?:\.[a-z_]+_\d{8})+")


def _rollout_identity(path: str) -> tuple[str, str] | None:
    """Extract (paper, run_dir_name) from a workspace path, stripping archive
    suffixes so migrated/renamed workspaces still match the run that created
    the rollout. Returns None if the path doesn't have the expected shape."""
    parts = Path(path).parts
    for i, part in enumerate(parts):
        # Run dir names look like: <model>_<paper>_<harness>[.<tag>_<date>]*
        if "_codex" in part and i > 0:
            paper = _ROLLOUT_SUFFIX_RE.sub("", parts[i - 1])
            run_dir = _ROLLOUT_SUFFIX_RE.sub("", part)
            return (paper, run_dir)
    return None


def _augment_codex_with_rollout(run: Run) -> None:
    """Verify session_rollout and emit write events from apply_patch content.

    On failure (rollout missing, malformed, or contaminated), mark the run
    excluded so it drops out of all downstream aggregation. A missing rollout
    is NOT by itself fatal — some runs may predate the rollout-copy logic
    or have had it fail — but contamination is: we'd rather lose those runs
    than silently attribute another run's output to this one.
    """
    rollout_path = run.workspace / "session_rollout.jsonl"
    if not rollout_path.exists():
        # No rollout available: we keep the run but write chars from
        # apply_patch won't be counted. This is an honest limitation rather
        # than a contamination risk.
        return

    try:
        first_line = rollout_path.open().readline()
        meta = json.loads(first_line)
    except (OSError, json.JSONDecodeError):
        run.excluded_reason = "session_rollout unreadable"
        return
    if meta.get("type") != "session_meta":
        run.excluded_reason = "session_rollout missing session_meta"
        return

    rollout_cwd = (meta.get("payload") or {}).get("cwd", "")
    rollout_id = _rollout_identity(rollout_cwd)
    workspace_id = _rollout_identity(str(run.workspace))
    if rollout_id is None and rollout_cwd == "[REDACTED]":
        # Public releases redact the private absolute cwd from session_meta.
        # A small sidecar records the identity check performed before
        # redaction and binds it to the sanitized rollout by SHA-256.
        identity_path = run.workspace / "session_rollout_identity.json"
        try:
            identity = json.loads(identity_path.read_text())
            rollout_sha = hashlib.sha256(rollout_path.read_bytes()).hexdigest()
            if (
                identity.get("identity_verified_before_redaction") is True
                and identity.get("sanitized_rollout_sha256") == rollout_sha
            ):
                rollout_id = (identity.get("paper"), identity.get("run_dir"))
        except (OSError, json.JSONDecodeError):
            pass
    if rollout_id is None or workspace_id is None or rollout_id != workspace_id:
        # Cross-run contamination — mtime heuristic picked a different run's
        # rollout. Drop the run entirely rather than using a mixed dataset.
        run.excluded_reason = (
            f"session_rollout contaminated (rollout_cwd={rollout_cwd!r})"
        )
        return

    # Verified: extract apply_patch tool calls from the rollout. Each call
    # represents the model output that drove a file edit — use it as our
    # canonical edit event. file_change items from run_log.jsonl describe the
    # SAME underlying operations (Codex runtime emits both for every
    # apply_patch), so we first drop the file_change-derived write events to
    # avoid double-counting in action totals.
    run.events = [
        e for e in run.events if e.raw_tool != "file_change"
    ]
    with rollout_path.open() as f:
        for line in f:
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            if evt.get("type") != "response_item":
                continue
            payload = evt.get("payload") or {}
            if payload.get("type") != "custom_tool_call":
                continue
            if payload.get("name") != "apply_patch":
                continue
            patch_text = payload.get("input") or ""
            if patch_text:
                run.events.append(Event(
                    tool="edit", raw_tool="apply_patch",
                    args={"patchText": patch_text},
                ))




# ---------- metrics ----------

# Source/script files only — instruction (.md), config, and data files are
# explicitly not code for the code_reads metric. All lowercase; _is_code_file
# lowercases before matching so ".r" covers both .r and .R on disk.
CODE_EXTENSIONS = frozenset({
    ".py", ".r", ".do", ".sh", ".ipynb", ".jl", ".m", ".sas", ".sql",
})

# Strict allowlist: bash first-tokens we consider an explicit content read.
BASH_READ_COMMANDS = frozenset({
    "cat", "sed", "head", "tail", "less", "more", "view", "nl", "bat",
})

# File-inspection-adjacent commands deliberately NOT counted as reads
# (search/metadata). Surfaced separately so we can audit what we're excluding.
BASH_READ_ADJACENT = frozenset({
    "grep", "rg", "awk", "wc", "file", "stat", "md5sum", "sha256sum", "diff",
})

# Bash sub-classification — used by _classify_bash and the plot.
BASH_EXEC_COMMANDS = frozenset({
    "python", "python3", "python2",
    "rscript", "r",
    "stata", "statamp",
    "julia", "node", "perl", "ruby",
    "jupyter", "ipython",
    "uv", "pixi", "make",
})
BASH_SEARCH_COMMANDS = frozenset({"grep", "rg", "ag", "egrep", "fgrep", "find"})
BASH_NAV_COMMANDS = frozenset({"ls", "cd", "pwd", "readlink", "which", "du", "dir", "tree"})
BASH_WRITE_COMMANDS = frozenset({
    "echo", "printf", "tee", "cp", "mv", "mkdir", "rm", "touch", "chmod", "rsync",
})


def _is_code_file(path: str) -> bool:
    if not isinstance(path, str) or not path:
        return False
    return any(path.strip().strip("'\"").lower().endswith(ext) for ext in CODE_EXTENSIONS)


def _split_compound(cmd: str, split_semicolon: bool = True) -> list[str]:
    """Split a shell command on && / || / ; at the top level only.

    Respects single and double quotes so that inline scripts like
    `python3 -c "x=1; y=2"` or `Rscript -e 'a <- 1; b <- 2'` are NOT split
    on the semicolons inside the quoted string.
    Set split_semicolon=False when heredocs are present to avoid splitting on
    bare ; inside the heredoc body.
    """
    parts: list[str] = []
    current: list[str] = []
    in_single = False
    in_double = False
    i = 0
    while i < len(cmd):
        c = cmd[i]
        if c == "'" and not in_double:
            in_single = not in_single
            current.append(c)
        elif c == '"' and not in_single:
            in_double = not in_double
            current.append(c)
        elif not in_single and not in_double:
            if c == ';' and split_semicolon:
                part = "".join(current).strip()
                if part:
                    parts.append(part)
                current = []
            elif c in ("&", "|") and i + 1 < len(cmd) and cmd[i + 1] == c:
                part = "".join(current).strip()
                if part:
                    parts.append(part)
                current = []
                i += 1  # consume second & or |
            else:
                current.append(c)
        else:
            current.append(c)
        i += 1
    part = "".join(current).strip()
    if part:
        parts.append(part)
    return parts or [""]


def _unwrap_bash_wrapper(cmd: str) -> str:
    """Strip a `/bin/bash -lc "..."` / `bash -c "..."` wrapper if present.
    Codex wraps every command this way; other harnesses typically don't.
    """
    if not cmd:
        return cmd
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        return cmd
    if (
        len(tokens) >= 3
        and tokens[0] in ("/bin/bash", "bash", "sh")
        and tokens[1] in ("-c", "-lc", "-cl")
    ):
        return tokens[2]
    return cmd


def _first_real_token(tokens: list[str]) -> tuple[str, list[str]]:
    """Skip leading env-var assignments (e.g. `PYTHONPATH=... cmd ...`) and
    return (first_command, remaining_args). Empty string if no command."""
    i = 0
    while i < len(tokens) and "=" in tokens[i] and not tokens[i].startswith("-"):
        i += 1
    if i >= len(tokens):
        return "", []
    return tokens[i], tokens[i + 1 :]



def _classify_one_bash(sub: str) -> str:
    """Classify a single non-compound bash command."""
    sub = sub.strip()
    if not sub:
        return "other"
    try:
        tokens = shlex.split(sub, posix=True)
    except ValueError:
        tokens = sub.split()
    first, rest = _first_real_token(tokens)
    if not first:
        return "other"
    fb = first.rsplit("/", 1)[-1].lower()
    if fb in BASH_EXEC_COMMANDS:
        return "exec"
    if fb in ("bash", "sh", "zsh"):
        return "exec" if any(a.endswith(".sh") for a in rest) else "other"
    if fb == "sed":
        # sed -i edits in place; sed without -i reads and streams to stdout
        return "write" if "-i" in rest else "read"
    if fb in BASH_READ_COMMANDS:
        # `cat > file` / `cat >> file` redirects stdout to a file — that's a write
        return "write" if ">" in rest else "read"
    if fb in BASH_NAV_COMMANDS:
        return "nav"
    if fb in BASH_SEARCH_COMMANDS:
        return "search"
    if fb in BASH_WRITE_COMMANDS:
        return "write"
    return "other"


def _classify_bash(cmd: str) -> list[str]:
    """Classify ALL sub-commands in a (possibly compound) bash command.

    Splits on && / || / ; and classifies each part independently.
    Pipes are not split — `cat x.py | head` is one read.
    Returns one classification per sub-command so compound calls like
    `python x.py && echo done` correctly contribute both exec and write.
    """
    if not cmd:
        return ["other"]
    return [_classify_one_bash(sub) for sub in _iter_subcommands(cmd)] or ["other"]


def _iter_subcommands(cmd: str) -> list[str]:
    """Unwrap and split a bash command into independently-classifiable sub-commands.

    Handles: bash -c / /bin/bash -lc wrappers, heredocs (<<), and compound
    operators (&& / || / ;). Always call this instead of _split_compound directly
    so the heredoc guard is consistently applied.

    When a heredoc is present (e.g. `cd workspace && cat > f.py <<'PY'\n...\nPY`),
    we still split on && / || so that `cd` and the heredoc-write are classified
    separately, but we skip ; splitting since the heredoc body contains bare
    semicolons that are not shell separators.
    """
    cmd = _unwrap_bash_wrapper(cmd)
    has_heredoc = "<<" in cmd
    return [sub for sub in _split_compound(cmd, split_semicolon=not has_heredoc)
            if sub.strip()] or [cmd]


@dataclass
class CodeReadStats:
    reads_main: int = 0
    reads_sub: int = 0
    unique_main: set[str] = field(default_factory=set)
    unique_sub: set[str] = field(default_factory=set)
    # Diagnostic: commands in BASH_READ_ADJACENT hitting a code file. Not
    # counted as reads, but surfaced so we know what we're excluding.
    adjacent_excluded: int = 0
    adjacent_breakdown: Counter = field(default_factory=Counter)


def _parse_bash_for_code_reads(cmd: str, stats: CodeReadStats, is_sub: bool) -> None:
    """Scan one bash command for code-file reads. Splits on && / || / ;
    (pipes are not split — first command's intent dominates)."""
    if not cmd:
        return
    cmd = _unwrap_bash_wrapper(cmd)
    for sub in re.split(r"\s*(?:&&|\|\||;)\s*", cmd):
        sub = sub.strip()
        if not sub:
            continue
        try:
            tokens = shlex.split(sub, posix=True)
        except ValueError:
            tokens = sub.split()
        first, rest = _first_real_token(tokens)
        if not first:
            continue
        first_base = first.rsplit("/", 1)[-1]
        code_args = [a for a in rest if _is_code_file(a)]
        if not code_args:
            continue
        if first_base in BASH_READ_COMMANDS:
            for a in code_args:
                if is_sub:
                    stats.reads_sub += 1
                    stats.unique_sub.add(a)
                else:
                    stats.reads_main += 1
                    stats.unique_main.add(a)
        elif first_base in BASH_READ_ADJACENT:
            stats.adjacent_excluded += len(code_args)
            stats.adjacent_breakdown[first_base] += len(code_args)


def compute_code_reads(run: Run) -> CodeReadStats:
    stats = CodeReadStats()
    for e in run.events:
        is_sub = e.agent != "main"
        if e.tool == "read":
            fp = e.file_path()
            if _is_code_file(fp):
                if is_sub:
                    stats.reads_sub += 1
                    stats.unique_sub.add(fp)
                else:
                    stats.reads_main += 1
                    stats.unique_main.add(fp)
        elif e.tool == "bash":
            _parse_bash_for_code_reads(e.command(), stats, is_sub)
    return stats


# ---------- reporting ----------

def summarize(runs: list[Run]) -> None:
    by_combo: dict[tuple[str, str], list[Run]] = defaultdict(list)
    for r in runs:
        by_combo[(r.model, r.harness)].append(r)

    header = f"{'model':<24} {'harness':<12} {'runs':>5} {'events':>8}  tool breakdown"
    print("\n" + header)
    print("-" * len(header))
    other_raw_by_combo: dict[tuple[str, str], Counter] = {}
    for (model, harness), rs in sorted(by_combo.items()):
        all_events = [e for r in rs for e in r.events]
        counter = Counter(e.tool for e in all_events)
        parts = [f"{t}:{counter.get(t, 0)}" for t in CANONICAL_TOOLS]
        n_missing = sum(1 for r in rs if r.log_missing)
        miss = f"  missing_log={n_missing}" if n_missing else ""
        print(f"{model:<24} {harness:<12} {len(rs):>5} {len(all_events):>8}  " + " ".join(parts) + miss)
        other_raw_by_combo[(model, harness)] = Counter(
            e.raw_tool for e in all_events if e.tool == "other"
        )

    if any(other_raw_by_combo.values()):
        print("\n'other' raw tool names (audit before classifying):")
        for (model, harness), c in sorted(other_raw_by_combo.items()):
            if not c:
                continue
            top = ", ".join(f"{name}:{n}" for name, n in c.most_common(10))
            print(f"  {model:<22} {harness:<12}  {top}")

    print("\nmain vs subagent (internals visible only for claude-code):")
    for (model, harness), rs in sorted(by_combo.items()):
        all_events = [e for r in rs for e in r.events]
        n_spawns = sum(1 for e in all_events if e.tool == "subagent")
        subagent_events = [e for e in all_events if e.agent != "main"]
        if n_spawns == 0 and not subagent_events:
            continue
        sub_by_type: Counter = Counter(e.agent for e in subagent_events)
        sub_tools: Counter = Counter(e.tool for e in subagent_events)
        visible = (
            f"subagent_events={len(subagent_events)} "
            f"types=[{', '.join(f'{k}:{v}' for k, v in sub_by_type.most_common())}] "
            f"tools=[{', '.join(f'{k}:{v}' for k, v in sub_tools.most_common(5))}]"
        )
        if harness == "opencode" and n_spawns > 0 and not subagent_events:
            visible += "   (note: opencode subagent internals are opaque)"
        print(f"  {model:<22} {harness:<12}  spawns={n_spawns}  {visible}")


def _sub_cell(harness: str, value: int, width: int) -> str:
    """Sub-agent column display: 'opq' for opencode (internals opaque),
    '-' for harnesses without subagents, else the numeric value."""
    if harness == "opencode":
        return "opq".rjust(width)
    if harness in ("codex", "swe-agent"):
        return "-".rjust(width)
    return f"{value:>{width}}"


def report_code_reads(runs: list[Run]) -> None:
    """Metric #1: reads of source-code files (.py, .R, .do, .sh, .ipynb, ...)."""
    by_combo: dict[tuple[str, str], list[Run]] = defaultdict(list)
    for r in runs:
        by_combo[(r.model, r.harness)].append(r)

    header = (
        f"{'model':<22} {'harness':<12} {'runs':>4} "
        f"{'main':>6} {'sub':>5} {'total':>6} "
        f"{'uniq_m':>6} {'uniq_s':>6} {'reread':>6} {'adj_excl':>8}"
    )
    print(f"\n=== metric #1: code_reads ({', '.join(sorted(CODE_EXTENSIONS))}) ===")
    print(header)
    print("-" * len(header))
    adj_breakdown_all: Counter = Counter()
    for (model, harness), rs in sorted(by_combo.items()):
        stats_list = [compute_code_reads(r) for r in rs]
        reads_main = sum(s.reads_main for s in stats_list)
        reads_sub = sum(s.reads_sub for s in stats_list)
        # Unique is per-run, summed — same file in a different run counts separately.
        uniq_main = sum(len(s.unique_main) for s in stats_list)
        uniq_sub = sum(len(s.unique_sub) for s in stats_list)
        total = reads_main + reads_sub
        uniq_total = uniq_main + uniq_sub
        reread = total / uniq_total if uniq_total else 0.0
        adj_excl = sum(s.adjacent_excluded for s in stats_list)
        for s in stats_list:
            adj_breakdown_all.update(s.adjacent_breakdown)

        print(
            f"{model:<22} {harness:<12} {len(rs):>4} "
            f"{reads_main:>6} {_sub_cell(harness, reads_sub, 5)} {total:>6} "
            f"{uniq_main:>6} {_sub_cell(harness, uniq_sub, 6)} {reread:>6.2f} {adj_excl:>8}"
        )

    if adj_breakdown_all:
        top = ", ".join(f"{k}:{v}" for k, v in adj_breakdown_all.most_common())
        print(f"\nadj_excl breakdown (code-file commands NOT counted as reads): {top}")
    print("legend: main/sub = read-tool calls; opq = opencode subagent internals opaque; "
          "- = no subagent mechanism; reread = total / unique per run")


def report_paper_citables(runs: list[Run]) -> None:
    """Print per-combo summary stats formatted for direct citation in the paper.

    Emits three blocks: tool-call counts, tool-call character volume, and
    model-level behavioral patterns (inline heredocs, subagent spawns). Numbers
    quoted in the main text should match these exactly.
    """
    by_combo: dict[tuple[str, str], list[Run]] = defaultdict(list)
    for r in runs:
        by_combo[(r.model, r.harness)].append(r)

    # --- Tool-call counts (raw events, not sub-command-expanded) -------------
    print("\n=== paper-citable: tool-call counts ===")
    print(f"{'model':<22} {'harness':<12}  {'N':>3}  {'events/run':>10}  {'rank':>5}")
    combo_events = []
    for (m, h), rs in by_combo.items():
        evts_per_run = sum(len(r.events) for r in rs) / max(len(rs), 1)
        combo_events.append(((m, h), len(rs), evts_per_run))
    combo_events.sort(key=lambda x: -x[2])
    for rank, ((m, h), n, epr) in enumerate(combo_events, 1):
        print(f"  {m:<20} {h:<12}  {n:>3}  {epr:>10.1f}  {rank:>5}")

    # --- Tool-call character volume -----------------------------------------
    print("\n=== paper-citable: tool-call characters per run ===")
    print(f"{'model':<22} {'harness':<12}  {'N':>3}  {'chars/run':>10}  "
          f"{'exec%':>6} {'write%':>6} {'read%':>6} {'nav%':>5} {'rank':>5}")
    combo_chars = []
    for (m, h), rs in by_combo.items():
        cat_chars: Counter = Counter()
        for r in rs:
            for e in r.events:
                for cat, n in _event_chars(e):
                    cat_chars[cat] += n
        total = sum(cat_chars.values())
        chars_per_run = total / max(len(rs), 1)
        combo_chars.append(((m, h), len(rs), chars_per_run, cat_chars, total))
    combo_chars.sort(key=lambda x: -x[2])
    for rank, ((m, h), n, cpr, cc, total) in enumerate(combo_chars, 1):
        pct = lambda k: 100 * cc.get(k, 0) / max(total, 1)
        print(f"  {m:<20} {h:<12}  {n:>3}  {cpr:>10,.0f}  "
              f"{pct('exec'):>5.1f}% {pct('write'):>5.1f}% "
              f"{pct('read'):>5.1f}% {pct('nav'):>4.1f}% {rank:>5}")

    # --- Inline-heredoc prevalence (what drives GPT-5.4's inline-exec signal) -
    print("\n=== paper-citable: inline python/R heredoc usage ===")
    print("(number of bash sub-commands matching `<lang> - <<'DELIM'...DELIM`)")
    print(f"{'model':<22} {'harness':<12}  {'N':>3}  {'heredoc_subs/run':>18}  "
          f"{'heredoc_chars/run':>18}")
    heredoc_re = re.compile(r"^(?:python|python3|rscript|node|julia|ruby|perl|ipython)"
                            r"\s+-\s+<<\S*")
    for (m, h), rs in sorted(by_combo.items()):
        total_subs = 0
        total_chars = 0
        for r in rs:
            for e in r.events:
                if e.tool != "bash":
                    continue
                for sub in _iter_subcommands(e.command()):
                    if heredoc_re.match(sub.strip().lower()):
                        total_subs += 1
                        total_chars += len(sub)
        n = len(rs)
        print(f"  {m:<20} {h:<12}  {n:>3}  {total_subs/max(n,1):>18.1f}  "
              f"{total_chars/max(n,1):>18,.0f}")

    # --- Subagent spawning ---------------------------------------------------
    print("\n=== paper-citable: subagent spawns (Claude Code and OpenCode only) ===")
    print(f"{'model':<22} {'harness':<12}  {'N':>3}  {'spawns/run':>10}  "
          f"{'sub_events/run':>14}")
    for (m, h), rs in sorted(by_combo.items()):
        spawns = sub_events = 0
        for r in rs:
            for e in r.events:
                if e.tool == "subagent":
                    spawns += 1
                if e.agent != "main":
                    sub_events += 1
        n = len(rs)
        if spawns == 0 and sub_events == 0 and h not in ("claude-code", "opencode"):
            continue  # skip harnesses that don't expose subagents
        print(f"  {m:<20} {h:<12}  {n:>3}  {spawns/max(n,1):>10.2f}  "
              f"{sub_events/max(n,1):>14.1f}")


# ---------- plots ----------

def _combo_key(model: str, harness: str) -> str:
    return f"{harness}/{model}"


def _combo_label(model: str, harness: str) -> str:
    return APPROACH_LABELS.get(_combo_key(model, harness), f"{model}\n{harness}")


def _combo_order(by_combo: dict) -> list[tuple[str, str]]:
    # Use canonical ordering from analyze_i4rep_results; unknowns go to the end.
    order = {key: i for i, key in enumerate(APPROACH_MODEL_ORDER)}
    return sorted(by_combo.keys(), key=lambda k: order.get(_combo_key(k[0], k[1]), 999))


# Categories ordered by action type so bash and tool variants of the same
# action sit adjacent. Legend order = left-to-right stacking order.
_PLOT_CATEGORIES = (
    "bash:exec",
    "bash:read", "read",
    "bash:nav",
    "bash:search", "grep",
    "bash:write", "edit", "write",
    "bash:other",
    "glob", "planning", "subagent", "other",
)

def _build_plot_colors() -> dict[str, tuple]:
    # All bash sub-types: dark→light blues (exec darkest = most active)
    bash_cats = [c for c in _PLOT_CATEGORIES if c.startswith("bash:")]
    blues = plt.cm.Blues(np.linspace(0.85, 0.28, len(bash_cats)))
    # Non-bash tools: tab10, skipping index 0 (blue) which is reserved for bash
    non_bash = [c for c in _PLOT_CATEGORIES if not c.startswith("bash:")]
    tab10 = plt.cm.tab10.colors
    return {
        **{cat: tuple(blues[i]) for i, cat in enumerate(bash_cats)},
        **{t: tab10[i + 1] for i, t in enumerate(non_bash)},
    }


def _event_categories(e: Event) -> list[str]:
    """Return one category label per action in the event.
    Bash events with compound commands (&&/||/;) return multiple labels."""
    if e.tool == "bash":
        return [f"bash:{c}" for c in _classify_bash(e.command())]
    return [e.tool]


def _build_combo_matrix(runs: list[Run], categories: tuple) -> tuple[dict, list, np.ndarray]:
    by_combo: dict[tuple[str, str], list[Run]] = defaultdict(list)
    for r in runs:
        by_combo[(r.model, r.harness)].append(r)
    combos = _combo_order(by_combo)
    mat = np.zeros((len(combos), len(categories)))
    for i, (m, h) in enumerate(combos):
        rs = by_combo[(m, h)]
        counter: Counter = Counter(cat for r in rs for e in r.events for cat in _event_categories(e))
        for j, cat in enumerate(categories):
            mat[i, j] = counter.get(cat, 0) / max(1, len(rs))
    return combos, mat


def plot_tool_composition(runs: list[Run], out_dir: Path) -> None:
    """Absolute + % composition stacked bars, bash expanded into sub-types."""
    colors = _build_plot_colors()
    combos, mat_abs = _build_combo_matrix(runs, _PLOT_CATEGORIES)
    labels = [_combo_label(m, h) for m, h in combos]
    row_sums = mat_abs.sum(axis=1, keepdims=True)
    mat_pct = np.where(row_sums > 0, mat_abs / row_sums * 100, 0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, mat, xlabel, xlim in [
        (axes[0], mat_abs, "tool calls / run", mat_abs.sum(axis=1).max() * 1.15),
        (axes[1], mat_pct, "% of actions",     100),
    ]:
        left = np.zeros(len(combos))
        for j, cat in enumerate(_PLOT_CATEGORIES):
            ax.barh(labels, mat[:, j], left=left, label=cat,
                    color=colors[cat], edgecolor="white", linewidth=0.5)
            left += mat[:, j]
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_xlim(0, xlim)
        ax.invert_yaxis()
        _style(ax)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
                   frameon=False, fontsize=8)
    fig.tight_layout()
    out = out_dir / "01_tool_composition.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  wrote {out}")
    _save_panels(fig, list(axes), out_dir, "01", dpi=150)
    plt.close(fig)


# Merged action categories — collapses bash + tool variants into one bucket.
_MERGED_CATEGORIES = ("exec", "read", "nav", "search", "write", "other")
# Human-readable names for plot legends (internal short names are used everywhere
# else for dict keys and matching). Spelled out for papers.
_MERGED_DISPLAY = {
    "exec": "execution",
    "read": "read",
    "nav": "navigation",
    "search": "search",
    "write": "write",
    "other": "other",
}
_MERGED_MAP = {
    "bash:exec":   "exec",
    "bash:read":   "read",   "read":     "read",
    "bash:nav":    "nav",
    "bash:search": "search", "grep":     "search",
    "bash:write":  "write",  "edit":     "write",  "write": "write",
    "bash:other":  "other",  "glob":     "other",  "planning": "other",
    "subagent":    "other",  "other":    "other",
}


def plot_merged_actions(runs: list[Run], out_dir: Path) -> None:
    """% action mix, bash + tool calls merged into exec/read/search/write/other."""
    merge_colors = {
        "exec":   tuple(plt.cm.Blues(0.75)),
        "read":   tuple(plt.cm.Oranges(0.65)),
        "nav":    tuple(plt.cm.Blues(0.40)),
        "search": tuple(plt.cm.Purples(0.60)),
        "write":  tuple(plt.cm.Reds(0.65)),
        "other":  tuple(plt.cm.Greys(0.35)),
    }
    by_combo: dict[tuple[str, str], list[Run]] = defaultdict(list)
    for r in runs:
        by_combo[(r.model, r.harness)].append(r)
    combos = _combo_order(by_combo)
    labels = [_combo_label(m, h) for m, h in combos]

    mat = np.zeros((len(combos), len(_MERGED_CATEGORIES)))
    for i, (m, h) in enumerate(combos):
        rs = by_combo[(m, h)]
        counter: Counter = Counter(
            _MERGED_MAP.get(cat, "other")
            for r in rs for e in r.events for cat in _event_categories(e)
        )
        for j, cat in enumerate(_MERGED_CATEGORIES):
            mat[i, j] = counter.get(cat, 0) / max(1, len(rs))

    row_sums = mat.sum(axis=1, keepdims=True)
    mat_pct = np.where(row_sums > 0, mat / row_sums * 100, 0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, data, xlabel, xlim in [
        (axes[0], mat,     "actions / run",  mat.sum(axis=1).max() * 1.15),
        (axes[1], mat_pct, "% of actions",   100),
    ]:
        left = np.zeros(len(combos))
        for j, cat in enumerate(_MERGED_CATEGORIES):
            ax.barh(labels, data[:, j], left=left, label=_MERGED_DISPLAY[cat],
                    color=merge_colors[cat], edgecolor="white", linewidth=0.5)
            left += data[:, j]
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_xlim(0, xlim)
        ax.invert_yaxis()
        _style(ax)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9)
    fig.tight_layout()
    out = out_dir / "02_action_mix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  wrote {out}")
    _save_panels(fig, list(axes), out_dir, "02", dpi=150)
    plt.close(fig)


_EXT_BUCKETS = {
    ".py": ".py",
    ".r": ".R",
    ".do": ".do (Stata)",
    ".sh": "other code", ".ipynb": "other code", ".jl": "other code",
    ".m": "other code", ".sas": "other code", ".sql": "other code",
    ".md": ".md (instructions)",
    ".json": ".json",
    ".csv": "data", ".dta": "data", ".xlsx": "data", ".xls": "data",
    ".parquet": "data", ".rds": "data", ".tsv": "data",
    ".txt": ".txt",
}
_EXT_BUCKET_ORDER = [".py", ".R", ".do (Stata)", "other code",
                     ".md (instructions)", ".json", "data", ".txt", "other"]


def plot_read_targets(runs: list[Run], out_dir: Path) -> None:
    """What file extensions the Read tool opens (codex/swe-agent have none — bash-only)."""
    # Muted qualitative palette — avoids Paired's garish lime/cyan
    _EXT_COLORS = [
        "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
        "#937860", "#DA8BC3", "#8C8C8C", "#CCB974",
    ]
    colors = dict(zip(_EXT_BUCKET_ORDER, _EXT_COLORS))

    by_combo: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for r in runs:
        by_combo[(r.model, r.harness)]  # ensure empty combos appear
        for e in r.events:
            if e.tool != "read":
                continue
            ext = os.path.splitext(e.file_path())[1].lower()
            by_combo[(r.model, r.harness)][_EXT_BUCKETS.get(ext, "other")] += 1

    combos = _combo_order(by_combo)
    labels = [_combo_label(m, h) for m, h in combos]
    mat = np.zeros((len(combos), len(_EXT_BUCKET_ORDER)))
    for i, c in enumerate(combos):
        for j, b in enumerate(_EXT_BUCKET_ORDER):
            mat[i, j] = by_combo[c].get(b, 0)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    left = np.zeros(len(combos))
    for j, b in enumerate(_EXT_BUCKET_ORDER):
        ax.barh(labels, mat[:, j], left=left, label=b,
                color=colors[b], edgecolor="white", linewidth=0.5)
        left += mat[:, j]
    ax.set_xlabel("Read-tool calls (total across all runs)", fontsize=10)
    ax.invert_yaxis()
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    out = out_dir / "03_read_targets.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def _save_panels(fig, axes: list, out_dir: Path, stem: str, dpi: int = 150) -> None:
    """Save each axis as its own PNG, with a uniform inside-legend for consistent width."""
    # Remove existing legends (may be outside-positioned, inflating tightbbox)
    existing: list[tuple] = []
    for ax in axes:
        leg = ax.get_legend()
        if leg is not None:
            h, l = ax.get_legend_handles_labels()
            existing.append((ax, h, l))
            leg.remove()
    added = []
    for ax in axes:
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="lower right", frameon=True, framealpha=0.85,
                      fontsize=8, edgecolor="none")
            added.append(ax)
    fig.canvas.draw()
    for ax, letter in zip(axes, "abcdefgh"):
        extent = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(
            fig.dpi_scale_trans.inverted()
        )
        out = out_dir / f"{stem}_{letter}.png"
        fig.savefig(out, dpi=dpi, bbox_inches=extent.expanded(1.02, 1.05))
        print(f"  wrote {out}")
    # Restore combined-figure state
    for ax in added:
        leg = ax.get_legend()
        if leg: leg.remove()
    for ax, h, l in existing:
        ax.legend(h, l, loc="center left", bbox_to_anchor=(1.01, 0.5),
                  frameon=False, fontsize=9)


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.25, linewidth=0.6)
    ax.tick_params(labelsize=9)


def _event_chars(e: Event) -> list[tuple[str, int]]:
    """Return (merged_category, char_count) pairs measuring chars the model OUTPUT.

    The intent is "how much did the model generate per action category":
    - bash: len of each sub-command string (what the model typed)
    - write: len of file content written (what the model produced)
    - edit: len of new_string / patchText (what the model changed)
    - read: len of file path (the model only output the path; content is input not output)
    - grep/glob/planning/subagent: skipped (args are small/structural)
    """
    if e.tool == "bash":
        subs = _iter_subcommands(e.command())
        cats = _classify_bash(e.command())
        return [(_MERGED_MAP.get(f"bash:{c}", "other"), len(s)) for s, c in zip(subs, cats)]
    if e.tool == "read":
        return []  # model generates only a path pointer; not meaningful generated content
    if e.tool == "write":
        a = e.args if isinstance(e.args, dict) else {}
        # Codex file_change: the model's output is the apply_patch text, which
        # lives in session_rollout.jsonl (unreliable) but NOT in run_log.jsonl.
        # We only count chars we can directly observe as model output from the
        # authoritative log — so file_change contributes to event counts (it's
        # still a write action) but not to character volume.
        if "changes" in a:
            return []
        # write tool (claude-code, opencode): content is in args
        content = a.get("content", "")
        return [("write", len(content))] if content else []
    if e.tool == "edit":
        # Count every char the model typed for this edit. patchText (Codex/OpenCode
        # apply_patch) embeds both old and new content inside the diff format, so
        # len(patchText) already covers both. Replace-style edits (claude-code
        # Edit, GLM-5 opencode edit) have separate old/new fields — sum them.
        a = e.args if isinstance(e.args, dict) else {}
        patch = a.get("patchText")
        if patch:
            return [("write", len(patch))]
        new = a.get("new_string") or a.get("newString") or ""
        old = a.get("old_string") or a.get("oldString") or ""
        total = len(new) + len(old)
        return [("write", total)] if total else []
    return []


def plot_command_chars(runs: list[Run], out_dir: Path) -> None:
    """Character volume per run, stacked by merged action category (same as plot 02).

    Counts: bash sub-command text, write tool content, edit new_string.
    Read tool content is not counted (only file path is available in args).
    """
    merge_colors = {
        "exec":   tuple(plt.cm.Blues(0.75)),
        "read":   tuple(plt.cm.Oranges(0.65)),
        "nav":    tuple(plt.cm.Blues(0.40)),
        "search": tuple(plt.cm.Purples(0.60)),
        "write":  tuple(plt.cm.Reds(0.65)),
        "other":  tuple(plt.cm.Greys(0.35)),
    }

    by_combo: dict[tuple[str, str], list[Run]] = defaultdict(list)
    for r in runs:
        by_combo[(r.model, r.harness)].append(r)
    combos = _combo_order(by_combo)
    labels = [_combo_label(m, h) for m, h in combos]

    mat = np.zeros((len(combos), len(_MERGED_CATEGORIES)))
    for i, (m, h) in enumerate(combos):
        rs = by_combo[(m, h)]
        cat_chars: Counter = Counter()
        for r in rs:
            for e in r.events:
                for merged_cat, n in _event_chars(e):
                    cat_chars[merged_cat] += n
        for j, cat in enumerate(_MERGED_CATEGORIES):
            mat[i, j] = cat_chars.get(cat, 0) / max(1, len(rs))

    row_sums = mat.sum(axis=1, keepdims=True)
    mat_pct = np.where(row_sums > 0, mat / row_sums * 100, 0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, data, xlabel, xlim in [
        (axes[0], mat,     "chars / run",  mat.sum(axis=1).max() * 1.15),
        (axes[1], mat_pct, "% of chars",   100),
    ]:
        left = np.zeros(len(combos))
        for j, cat in enumerate(_MERGED_CATEGORIES):
            ax.barh(labels, data[:, j], left=left, label=_MERGED_DISPLAY[cat],
                    color=merge_colors[cat], edgecolor="white", linewidth=0.5)
            left += data[:, j]
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_xlim(0, xlim)
        ax.invert_yaxis()
        _style(ax)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9)
    fig.tight_layout()
    out = out_dir / "04_command_chars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  wrote {out}")
    _save_panels(fig, list(axes), out_dir, "04", dpi=150)
    plt.close(fig)


def generate_plots(runs: list[Run], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelweight": "bold",   # match the paper's bold axis-label style
    })
    print(f"\nwriting plots to {out_dir}")
    plot_tool_composition(runs, out_dir)
    plot_merged_actions(runs, out_dir)
    plot_read_targets(runs, out_dir)
    plot_command_chars(runs, out_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--paper", help="limit to a single paper DOI dir")
    ap.add_argument("--harness", choices=HARNESSES, help="limit to one harness")
    ap.add_argument("--json", type=Path, help="also write per-run events dump to this JSON file")
    ap.add_argument(
        "--plots-dir",
        type=Path,
        nargs="?",
        const=DEFAULT_PLOTS_DIR,
        default=None,
        help=f"generate plots into this directory (default when flag passed with no value: {DEFAULT_PLOTS_DIR})",
    )
    args = ap.parse_args()

    runs = [
        r for r in discover_runs(args.results_dir)
        if (not args.paper or r.paper == args.paper)
        and (not args.harness or r.harness == args.harness)
    ]
    for run in runs:
        parse_run(run)
    # Drop excluded runs (e.g. codex with contaminated session_rollout) from
    # everything downstream; surface which runs and why so it's not silent.
    excluded = [r for r in runs if r.excluded_reason]
    runs = [r for r in runs if not r.excluded_reason]
    print(f"loaded {len(runs)} runs from {args.results_dir}"
          + (f"  (excluded {len(excluded)})" if excluded else ""))
    for r in excluded:
        print(f"  excluded: {r.model:<22} {r.harness:<12} {r.paper}  ({r.excluded_reason})")

    summarize(runs)
    report_code_reads(runs)
    report_paper_citables(runs)

    bash_other_tokens: Counter = Counter()
    for run in runs:
        for e in run.events:
            if e.tool != "bash":
                continue
            for sub in _iter_subcommands(e.command()):
                sub = sub.strip()
                if sub and _classify_one_bash(sub) == "other":
                    try:
                        toks = shlex.split(sub, posix=True)
                    except ValueError:
                        toks = sub.split()
                    first, _ = _first_real_token(toks)
                    if first:
                        bash_other_tokens[first.rsplit("/", 1)[-1].lower()] += 1
    if bash_other_tokens:
        print("\n'bash:other' first tokens (not matched by any classification set):")
        for token, n in bash_other_tokens.most_common():
            print(f"  {token:<30} {n:>5}")

    if _unknown_events:
        print("\n[warn] unknown event types encountered during parsing (not counted):")
        for key, n in _unknown_events.most_common():
            print(f"  {key:<50} {n:>5}")

    if args.plots_dir:
        generate_plots(runs, args.plots_dir)

    if args.json:
        payload = [
            {
                "paper": r.paper,
                "model": r.model,
                "harness": r.harness,
                "workspace": str(r.workspace),
                "log_missing": r.log_missing,
                "events": [
                    {"tool": e.tool, "raw_tool": e.raw_tool, "args": e.args}
                    for e in r.events
                ],
            }
            for r in runs
        ]
        args.json.write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nwrote per-run events to {args.json}")


if __name__ == "__main__":
    main()
