#!/usr/bin/env python3
"""Guardrail validator for social_science_replicability runs.
"""

from __future__ import annotations

import ast
import csv
import io
import json
import os
import re
import tokenize
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

# These are configurable (upadte when new runs come too)

PROJECT_ROOT = Path(os.environ.get(
    "PROJECT_ROOT", str(Path(__file__).resolve().parents[1]),
))

COHORTS = {
    "new_i4replicate": {
        "results_dir": PROJECT_ROOT / "data" / "i4replicate" / "results",
        "papers_dir":  PROJECT_ROOT / "data" / "i4replicate" / "papers",
    },
    # "old_archive": {
    #     "results_dir": PROJECT_ROOT / "data" / "archive" / "i4rep_results_20260317" / "results",
    #     "papers_dir":  PROJECT_ROOT / "data" / "i4replicate" / "papers",
    # },
}

OUTPUT_DIR  = PROJECT_ROOT / "scripts" / "guardrail_validation"
_TS         = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_JSON = OUTPUT_DIR / f"guardrail_audit_{_TS}.json"
OUTPUT_CSV  = OUTPUT_DIR / f"guardrail_audit_{_TS}.csv"

PATH_RE      = re.compile(r"(/[^\s\"'<>|;]+)")
URL_RE       = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
WEB_TOKEN_RE = re.compile(
    r"\b(curl|wget|requests\.|urllib|httpx|fetch\(|web_search|web_fetch|browser|open\s+https?://)\b",
    re.IGNORECASE,
)
READLIKE_RE  = re.compile(
    r"\b(cat|less|more|head|tail|sed|awk|rg|find|ls|read|open)\b",
    re.IGNORECASE,
)



def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _parse_iso_timestamp(raw: str) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _duration_from_jsonl_timestamps(log_path: Path, *, unit: str) -> float | None:
    if not log_path.exists():
        return None

    first = None
    last = None
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            raw_ts = row.get("timestamp")
            if raw_ts is None:
                continue

            if unit == "iso":
                ts = _parse_iso_timestamp(raw_ts)
                if ts is None:
                    continue
            elif unit == "ms":
                try:
                    ts = float(raw_ts) / 1000.0
                except (TypeError, ValueError):
                    continue
            else:
                raise ValueError(f"Unsupported timestamp unit: {unit}")

            if first is None:
                first = ts
            last = ts

    if first is None or last is None:
        return None

    if unit == "iso":
        return (last - first).total_seconds()
    return last - first


def get_reported_duration_seconds(run_dir: Path) -> float | None:
    result_path = run_dir / "result.json"
    if not result_path.exists():
        return None
    try:
        value = load_json(result_path).get("duration_seconds")
    except Exception:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def derive_runtime_seconds(run_dir: Path, approach: str) -> float | None:
    workspace_dir = run_dir / "workspace"

    if approach == "codex":
        return _duration_from_jsonl_timestamps(workspace_dir / "run_log.jsonl", unit="iso")
    if approach == "opencode":
        return _duration_from_jsonl_timestamps(workspace_dir / "run_log.jsonl", unit="ms")

    return None


def extract_web_addresses(text: str) -> list[str]:
    if not text:
        return []
    return list(dict.fromkeys(URL_RE.findall(text) + WEB_TOKEN_RE.findall(text)))


def extract_paths_from_text(text: str, workspace_dir: Path) -> list[str]:
    if not text:
        return []
    abs_paths = PATH_RE.findall(text)
    rel_paths = re.findall(
        r"(?<![\w/.-])(?:data|workspace|table_templates|TASK\.md|"
        r"methodology_summary\.json|paper\.pdf|replication_package)[^\s\"'<>|;]*",
        text,
    )
    seen: list[str] = []
    for raw in abs_paths + rel_paths:
        p = normalize_path(raw, workspace_dir)
        if p is not None:
            s = str(p)
            if s not in seen:
                seen.append(s)
    return seen


def normalize_path(raw: str, workspace_dir: Path) -> Path | None:
    if not raw:
        return None
    try:
        p = Path(raw)
        if not p.is_absolute():
            p = (workspace_dir / p).resolve(strict=False)
        else:
            p = p.resolve(strict=False)
        return p
    except Exception:
        return None


def classify_path(path: Path | None, workspace_dir: Path, cohort: str, paper_id: str) -> str:
    """Classify a path the agent accessed as allowed or forbidden.

    # TODO check that these assumptions are correct

    Allowed:
      - workspace/         : the agent's working directory — it writes scripts,
                             reads TASK.md, table templates etc. here. Also covers
                             workspace/data/... (the data symlink target) because
                             data/ resolves inside the workspace tree.
      - papers/{id}/data/  : the raw datasets the agent is supposed to use
                             (csv, dta, RData etc.). Reached via the data/ symlink
                             in the workspace or absolute path.

    Forbidden:
      - replication_package/ : the original author's R/Stata/Python code. The agent
                               must NOT see this — it would trivially copy results.
      - paper.pdf            : the published paper containing the actual result values.
                               Again, the agent should only use the methodology summary
                               we extracted, not the paper itself.
      - everything else      : classified as forbidden_external. In practice this
                               is mostly noise (system paths, text fragments picked up
                               by the path regex) — use replication_package and
                               paper_pdf counts as the real signal.
    """
    if path is None:
        return "unknown"
    p         = path.resolve(strict=False)
    ws        = workspace_dir.resolve(strict=False)
    data_dir  = (COHORTS[cohort]["papers_dir"] / paper_id / "data").resolve(strict=False)
    repl_dir  = (COHORTS[cohort]["papers_dir"] / paper_id / "replication_package").resolve(strict=False)
    paper_pdf = (COHORTS[cohort]["papers_dir"] / paper_id / "paper.pdf").resolve(strict=False)

    if p == ws or ws in p.parents:             return "allowed_workspace"
    if p == data_dir or data_dir in p.parents: return "allowed_data"
    if p == repl_dir or repl_dir in p.parents: return "forbidden_replication_package"
    if p == paper_pdf:                         return "forbidden_paper_pdf"
    if str(p).startswith("/tmp/"):             return "unknown_tmp"
    return "forbidden_external"


def classify_paths(paths: list[str], workspace_dir: Path, cohort: str, paper_id: str) -> list[dict]:
    return [
        {
            "path": str(p) if (p := normalize_path(raw, workspace_dir)) else raw,
            "classification": classify_path(normalize_path(raw, workspace_dir), workspace_dir, cohort, paper_id),
        }
        for raw in paths
    ]


# ── parsers ───────────────────────────────────────────────────────────────────
# Each returns: {log_format, effective_workspace?, commands, paths, web_addresses_detected}

def parse_claude_json(log_path: Path, workspace_dir: Path) -> dict:
    """Parse claude-code run_log.json.
    Reads cwd from system/init so archive runs resolve paths correctly.
    """
    rows = load_json(log_path)
    commands, paths, web = [], [], []
    n_tool_calls = 0
    n_llm_iterations = 0

    effective_workspace = workspace_dir
    for row in rows:
        if row.get("type") == "system" and row.get("subtype") == "init":
            cwd = row.get("cwd")
            if cwd:
                effective_workspace = Path(cwd)
            break

    for row in rows:
        if row.get("type") == "assistant":
            n_llm_iterations += 1
        message = row.get("message", {})
        if not isinstance(message, dict):
            continue
        for block in message.get("content", []):
            btype = block.get("type", "")
            inp   = block.get("input", {})

            if btype == "text":
                web.extend(extract_web_addresses(block.get("text", "")))

            elif btype == "tool_use":
                n_tool_calls += 1
                tool = block.get("name", "")

                if tool == "Read":
                    p = inp.get("file_path", "")
                    if p:
                        paths.extend(extract_paths_from_text(p, effective_workspace))

                elif tool in ("Glob", "Grep"):
                    p = inp.get("path", inp.get("glob", ""))
                    if p:
                        paths.extend(extract_paths_from_text(p, effective_workspace))

                elif tool == "Bash":
                    cmd = inp.get("command", "")
                    if cmd:
                        commands.append(cmd)
                        if READLIKE_RE.search(cmd):
                            paths.extend(extract_paths_from_text(cmd, effective_workspace))
                        web.extend(extract_web_addresses(cmd))

                elif tool == "WebFetch":
                    if inp.get("url"):
                        web.append(inp["url"])

                elif tool == "WebSearch":
                    if inp.get("query"):
                        web.append(inp["query"])

    return {
        "log_format":          "claude_json",
        "effective_workspace": str(effective_workspace),
        "commands":            list(dict.fromkeys(commands)),
        "paths":               list(dict.fromkeys(paths)),
        "n_tool_calls":        n_tool_calls,
        "n_llm_iterations":    n_llm_iterations,
        "web_addresses_detected": list(dict.fromkeys(web)),
    }


def parse_codex_jsonl(log_path: Path, workspace_dir: Path) -> dict:
    """Parse codex run_log.jsonl (archive and new — same format)."""
    rows = load_jsonl(log_path)
    commands, paths, web = [], [], []
    n_tool_calls = 0
    n_llm_iterations = 0

    for row in rows:
        item = row.get("item", {})
        if item.get("type") == "command_execution":
            if row.get("type") == "item.completed":
                n_tool_calls += 1
            cmd    = item.get("command", "")
            output = item.get("aggregated_output", "")
            commands.append(cmd)
            if READLIKE_RE.search(cmd):
                paths.extend(extract_paths_from_text(cmd, workspace_dir))
            paths.extend(extract_paths_from_text(output, workspace_dir))
            web.extend(extract_web_addresses(cmd))
            web.extend(extract_web_addresses(output))

        elif row.get("type") == "item.completed" and item.get("type") == "agent_message":
            n_llm_iterations += 1

        elif row.get("type") == "response_item":
            payload = row.get("payload", {})
            if payload.get("type") == "function_call":
                args = payload.get("arguments", "")
                paths.extend(extract_paths_from_text(args, workspace_dir))
                web.extend(extract_web_addresses(args))

        elif row.get("type") == "event_msg":
            msg = row.get("payload", {}).get("message", "")
            web.extend(extract_web_addresses(msg))

    return {
        "log_format": "codex_jsonl",
        "commands":   list(dict.fromkeys(commands)),
        "paths":      list(dict.fromkeys(paths)),
        "n_tool_calls":     n_tool_calls,
        "n_llm_iterations": n_llm_iterations,
        "web_addresses_detected": list(dict.fromkeys(web)),
    }


def parse_opencode_jsonl(log_path: Path, workspace_dir: Path) -> dict:
    """Parse opencode run_log.jsonl (new runs only — different schema from codex)."""
    rows = load_jsonl(log_path)
    commands, paths, web = [], [], []
    n_tool_calls = 0
    n_llm_iterations = 0

    for row in rows:
        if row.get("type") == "step_start":
            n_llm_iterations += 1

        if row.get("type") == "tool_use":
            n_tool_calls += 1
            part   = row.get("part", {})
            tool   = part.get("tool", "")
            state  = part.get("state", {})
            inp    = state.get("input", {})
            output = state.get("output", "")

            if tool == "bash":
                cmd = inp.get("command", "")
                commands.append(cmd)
                if READLIKE_RE.search(cmd):
                    paths.extend(extract_paths_from_text(cmd, workspace_dir))
                paths.extend(extract_paths_from_text(output, workspace_dir))
                web.extend(extract_web_addresses(cmd))
                web.extend(extract_web_addresses(output))

            elif tool == "read":
                p = inp.get("filePath", "")
                if p:
                    paths.extend(extract_paths_from_text(p, workspace_dir))

        elif row.get("type") == "text":
            web.extend(extract_web_addresses(row.get("part", {}).get("text", "")))

    return {
        "log_format": "opencode_jsonl",
        "commands":   list(dict.fromkeys(commands)),
        "paths":      list(dict.fromkeys(paths)),
        "n_tool_calls":     n_tool_calls,
        "n_llm_iterations": n_llm_iterations,
        "web_addresses_detected": list(dict.fromkeys(web)),
    }


def parse_swe_agent(log_path: Path, workspace_dir: Path) -> dict:
    """Parse swe-agent trajectory.json (new runs only)."""
    payload = load_json(log_path)
    commands, paths, web = [], [], []
    n_tool_calls = 0
    n_llm_iterations = 0

    for msg in payload.get("messages", []):
        if msg.get("role") == "assistant":
            n_llm_iterations += 1
        content = msg.get("content")
        if isinstance(content, str):
            web.extend(extract_web_addresses(content))

        for tc in msg.get("tool_calls") or []:
            n_tool_calls += 1
            fn = tc.get("function", {})
            if fn.get("name") != "bash":
                continue
            try:
                cmd = json.loads(fn.get("arguments", "{}")).get("command", "")
            except json.JSONDecodeError:
                cmd = ""
            if cmd:
                commands.append(cmd)
                if READLIKE_RE.search(cmd):
                    paths.extend(extract_paths_from_text(cmd, workspace_dir))
                web.extend(extract_web_addresses(cmd))

    return {
        "log_format": "swe_agent_trajectory",
        "commands":   list(dict.fromkeys(commands)),
        "paths":      list(dict.fromkeys(paths)),
        "n_tool_calls":     n_tool_calls,
        "n_llm_iterations": n_llm_iterations,
        "web_addresses_detected": list(dict.fromkeys(web)),
    }


# ── config registry ───────────────────────────────────────────────────────────
# Maps (approach, cohort) → (log_filename, parser_fn)

PARSERS: dict[tuple[str, str], tuple[str, Any]] = {
    ("claude-code", "any"):            ("run_log.json",    parse_claude_json),
    ("codex",       "any"):            ("run_log.jsonl",   parse_codex_jsonl),
    ("opencode",    "new_i4replicate"):("run_log.jsonl",   parse_opencode_jsonl),
    ("swe-agent",   "new_i4replicate"):("trajectory.json", parse_swe_agent),
}

def _lookup_parser(approach: str, cohort: str):
    return PARSERS.get((approach, cohort)) or PARSERS.get((approach, "any"))


# ── run discovery & identity ──────────────────────────────────────────────────

def iter_run_dirs(results_dir: Path) -> list[Path]:
    run_dirs = []
    for workspace in results_dir.rglob("workspace"):
        if workspace.parent.name == "explainer_workspace":
            continue
        run_dirs.append(workspace.parent)
    return sorted(set(run_dirs))


def parse_run_identity(run_dir: Path, results_dir: Path) -> dict:
    rel   = run_dir.relative_to(results_dir)
    parts = rel.parts
    leaf  = parts[-1]

    for suffix, approach in [
        ("_codex",      "codex"),
        ("_claude-code","claude-code"),
        ("_opencode",   "opencode"),
        ("_swe-agent",  "swe-agent"),
    ]:
        if leaf.endswith(suffix):
            base = leaf[: -len(suffix)]
            break
    else:
        approach, base = "unknown", leaf

    m        = re.search(r"_(10\..+)$", base)
    paper_id = m.group(1) if m else (rel.parts[0] if rel.parts else "unknown")
    model    = base[: m.start()] if m else base
    provider = parts[0] if len(parts) > 1 else "top-level"

    return {"provider": provider, "model": model, "paper_id": paper_id,
            "approach": approach, "run_name": leaf}


def _empty_result(ident, cohort, workspace_dir, log_path, parse_status):
    return {**ident, "cohort": cohort, "workspace_dir": str(workspace_dir),
            "log_path": str(log_path), "parse_status": parse_status,
            "log_format": "unknown", "commands": [], "paths": [],
            "n_tool_calls": 0, "n_llm_iterations": 0,
            "classified_paths": [], "web_addresses_detected": [],
            "classification_counts": {}, "any_forbidden_read": False,
            "any_replication_package_access": False, "any_paper_pdf_access": False,
            "any_web_detected": False,
            "n_hardcode_matches": -1, "any_hardcode_suspected": -1,
            "hardcode_match_detail": "-1"}


def parse_run(cohort: str, run_dir: Path, results_dir: Path) -> dict:
    workspace_dir = run_dir / "workspace"
    ident         = parse_run_identity(run_dir, results_dir)
    approach      = ident["approach"]
    paper_id      = ident["paper_id"]

    config = _lookup_parser(approach, cohort)
    if config is None:
        return _empty_result(ident, cohort, workspace_dir, "", "unsupported_combo")

    log_filename, parser_fn = config
    log_path = workspace_dir / log_filename
    if not log_path.exists():
        return _empty_result(ident, cohort, workspace_dir, log_path, "missing_log")

    parsed       = parser_fn(log_path, workspace_dir)
    effective_ws = Path(parsed.get("effective_workspace", str(workspace_dir)))
    classified   = classify_paths(parsed["paths"], effective_ws, cohort, paper_id)
    counts       = Counter(item["classification"] for item in classified)

    return {
        **ident,
        "cohort":          cohort,
        "workspace_dir":   str(workspace_dir),
        "log_path":        str(log_path),
        "parse_status":    "ok",
        "log_format":      parsed["log_format"],
        "commands":        parsed["commands"],
        "paths":           parsed["paths"],
        "n_tool_calls":    parsed.get("n_tool_calls", 0),
        "n_llm_iterations": parsed.get("n_llm_iterations", 0),
        "classified_paths": classified,
        "web_addresses_detected": parsed["web_addresses_detected"],
        "classification_counts":  dict(counts),
        "any_forbidden_read":            any(l in counts for l in ("forbidden_external", "forbidden_replication_package", "forbidden_paper_pdf")),
        "any_replication_package_access": counts.get("forbidden_replication_package", 0) > 0,
        "any_paper_pdf_access":           counts.get("forbidden_paper_pdf", 0) > 0,
        "any_web_detected":               len(parsed["web_addresses_detected"]) > 0,
        **check_hardcoded_values(workspace_dir),
    }


def write_outputs(results: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(results, f)

    fieldnames = [
        "cohort", "paper_id", "provider", "model", "approach", "run_name",
        "parse_status", "log_format", "n_commands", "n_tool_calls", "n_llm_iterations", "n_paths",
        "n_allowed_workspace", "n_allowed_data",
        "n_forbidden_replication_package", "n_forbidden_paper_pdf",
        "n_forbidden_external", "n_unknown_tmp",
        "any_forbidden_read", "any_replication_package_access",
        "any_paper_pdf_access", "any_web_detected",
        "n_hardcode_matches", "any_hardcode_suspected", "hardcode_match_detail",
        "paths_allowed", "paths_forbidden",
        "web_addresses_detected", "log_path",
    ]

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            c = row.get("classification_counts", {})
            cp = row.get("classified_paths", [])
            allowed_labels   = {"allowed_workspace", "allowed_data", "unknown_tmp"}
            forbidden_labels = {"forbidden_replication_package", "forbidden_paper_pdf", "forbidden_external"}
            writer.writerow({
                "cohort":    row["cohort"],   "paper_id": row["paper_id"],
                "provider":  row["provider"], "model":    row["model"],
                "approach":  row["approach"], "run_name": row["run_name"],
                "parse_status": row["parse_status"], "log_format": row["log_format"],
                "n_commands": len(row["commands"]),
                "n_tool_calls": row.get("n_tool_calls", 0),
                "n_llm_iterations": row.get("n_llm_iterations", 0),
                "n_paths":    len(cp),
                "n_allowed_workspace":             c.get("allowed_workspace", 0),
                "n_allowed_data":                  c.get("allowed_data", 0),
                "n_forbidden_replication_package": c.get("forbidden_replication_package", 0),
                "n_forbidden_paper_pdf":           c.get("forbidden_paper_pdf", 0),
                "n_forbidden_external":            c.get("forbidden_external", 0),
                "n_unknown_tmp":                   c.get("unknown_tmp", 0),
                "any_forbidden_read":             row["any_forbidden_read"],
                "any_replication_package_access": row["any_replication_package_access"],
                "any_paper_pdf_access":           row["any_paper_pdf_access"],
                "any_web_detected":               row["any_web_detected"],
                "n_hardcode_matches":     row.get("n_hardcode_matches", 0),
                "any_hardcode_suspected": row.get("any_hardcode_suspected", False),
                "hardcode_match_detail":  row.get("hardcode_match_detail", ""),
                "paths_allowed":   "; ".join(p["path"] for p in cp if p["classification"] in allowed_labels),
                "paths_forbidden": "; ".join(p["path"] for p in cp if p["classification"] in
                                             {"forbidden_replication_package", "forbidden_paper_pdf",
                                              "forbidden_external", "unknown_tmp"}),
                "web_addresses_detected": "; ".join(row["web_addresses_detected"]),
                "log_path":  row["log_path"],
            })




# ── hardcoded value detection ─────────────────────────────────────────────────

# TODO by David: this is not great, but the chance that a regression coeff is one of these things is super low and this weeds out a lot of noise.
# Numeric constants common in statistical code that are NOT result values
_COMMON_NUMERICS = frozenset({0.0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 100.0})


def _result_values_from_json(json_path: Path) -> set[float]:
    """Extract 3dp-rounded numeric_value floats from a submitted table JSON.
    Returns empty set if unreadable or a template (all nulls).
    """
    try:
        cells = load_json(json_path).get("cells", [])
    except Exception:
        return set()

    def _iter_cell_dicts(items):
        for item in items:
            if isinstance(item, dict):
                yield item
            elif isinstance(item, list):
                yield from _iter_cell_dicts(item)

    values = {
        round(float(cell["numeric_value"]), 3)
        for cell in _iter_cell_dicts(cells)
        if cell.get("numeric_value") is not None
    }
    return values - _COMMON_NUMERICS


def float_literals_in_py(py_path: Path) -> list[tuple[int, str]]:
    """Return (line_number, token_string) for float-like literals in a Python file."""
    try:
        src = py_path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(py_path))
    except Exception:
        return []

    class _FloatLiteralVisitor(ast.NodeVisitor):
        def __init__(self, source: str):
            self.source = source
            self.results: list[tuple[int, str]] = []

        def _maybe_add(self, node: ast.AST, value: float | int) -> None:
            segment = ast.get_source_segment(self.source, node)
            if not segment:
                return
            normalized = segment.lower()
            if "." not in normalized and "e" not in normalized:
                return
            self.results.append((node.lineno, segment))

        def visit_Constant(self, node: ast.Constant) -> None:
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                self._maybe_add(node, node.value)

        def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
            operand = node.operand
            if (
                isinstance(node.op, (ast.UAdd, ast.USub))
                and isinstance(operand, ast.Constant)
                and isinstance(operand.value, (int, float))
                and not isinstance(operand.value, bool)
            ):
                signed_value = operand.value if isinstance(node.op, ast.UAdd) else -operand.value
                self._maybe_add(node, signed_value)
                return
            self.generic_visit(node)

    visitor = _FloatLiteralVisitor(src)
    try:
        visitor.visit(tree)
        return visitor.results
    except Exception:
        return []


def check_hardcoded_values(workspace_dir: Path) -> dict:
    """For each table_*.py / table_*.json pair, check if float literals in the code
    match the submitted output values — a sign the agent may have hardcoded results.
    """
    all_matches = []
    for json_path in sorted(workspace_dir.glob("table_*.json")):
        if json_path.parent != workspace_dir:
            continue                          # skip table_templates/
        py_path = workspace_dir / f"{json_path.stem}.py"
        if not py_path.exists():
            continue

        target = _result_values_from_json(json_path)
        if not target:
            continue                          # template or no numeric results yet

        for line_no, token in float_literals_in_py(py_path):
            try:
                if round(float(token), 3) in target:
                    all_matches.append(f"{json_path.stem}:L{line_no}:{token}")
            except ValueError:
                continue

    return {
        "n_hardcode_matches":     len(all_matches),
        "any_hardcode_suspected": len(all_matches) > 0,
        "hardcode_match_detail":  "; ".join(all_matches),
    }


def main() -> None:
    results = []
    for cohort, cfg in COHORTS.items():
        results_dir = cfg["results_dir"]
        if not results_dir.exists():
            continue
        for run_dir in iter_run_dirs(results_dir):
            results.append(parse_run(cohort, run_dir, results_dir))

    write_outputs(results)


if __name__ == "__main__":
    main()
