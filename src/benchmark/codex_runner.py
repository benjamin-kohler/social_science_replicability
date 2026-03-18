"""Codex CLI runner for freestyle replication."""

import glob as glob_mod
import json as json_mod
import shutil
import subprocess
import time
from pathlib import Path

from ..models.schemas import PaperSummary
from ..utils.logging_utils import get_logger
from .base_runner import BaseReplicationRunner
from .config import ModelSpec, PaperSpec
from .results import RunArtifacts
from .task_prompt import setup_workspace, setup_workspace_paper_direct

logger = get_logger(__name__)


class CodexRunner(BaseReplicationRunner):
    """Runs a freestyle replication using the Codex CLI (codex exec).

    Creates an isolated workspace with only the methodology summary and data,
    then invokes `codex exec` in full-auto mode. The model does NOT receive
    the original paper PDF or replication package.
    """

    def __init__(
        self,
        codex_binary: str = "codex",
        timeout: int = 600,
        allow_web_access: bool = False,
        item_types: list[str] | None = None,
    ):
        super().__init__(timeout=timeout, allow_web_access=allow_web_access, item_types=item_types)
        self.codex_binary = codex_binary

    def run(
        self,
        model: ModelSpec,
        paper: PaperSpec,
        paper_summary: PaperSummary,
        workspace_dir: Path,
        paper_direct: bool = False,
    ) -> RunArtifacts:
        """Run a freestyle replication using Codex CLI.

        Args:
            model: Model specification.
            paper: Paper specification (used only for data_path).
            paper_summary: Pre-extracted methodology summary (no results).
            workspace_dir: Isolated workspace directory for this run.
            paper_direct: If True, give the replicator the paper PDF instead
                of the extracted methodology summary.

        Returns:
            RunArtifacts with workspace contents, stdout, stderr, exit code, duration.
        """
        if paper_direct:
            setup_workspace_paper_direct(paper, paper_summary, workspace_dir)
        else:
            setup_workspace(paper, paper_summary, workspace_dir, item_types=self.item_types)

        # Build the inline prompt — single sentence pointing to TASK.md.
        if paper_direct:
            prompt_text = "Read TASK.md for your full instructions. Start by examining paper.pdf."
        else:
            prompt_text = "Read TASK.md for your full instructions."

        web_status = "ALLOWED" if self.allow_web_access else "BLOCKED"
        logger.info(
            f"Running codex: model={model.model_name}, "
            f"paper={paper.paper_id}, web_access={web_status}"
        )
        start = time.time()

        try:
            abs_workspace = str(Path(workspace_dir).resolve())
            cmd = [
                self.codex_binary, "exec",
                "--full-auto",
                "--json",
                "-m", model.model_name,
                "-C", abs_workspace,
                "--skip-git-repo-check",
            ]
            if not self.allow_web_access:
                # Restrict sandbox: no full-disk read, no network access.
                cmd.extend(["-c", "sandbox_permissions=[]"])
            cmd.append(prompt_text)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            stdout = result.stdout
            stderr = result.stderr
            exit_code = result.returncode
        except subprocess.TimeoutExpired as exc:
            logger.warning(f"Codex timed out after {self.timeout}s")
            # Preserve partial JSONL output captured before the timeout
            stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else (exc.stdout or b"").decode(errors="replace")
            stderr_partial = (exc.stderr or "") if isinstance(exc.stderr, str) else (exc.stderr or b"").decode(errors="replace")
            stderr = f"Timed out after {self.timeout} seconds\n{stderr_partial}".strip()
            exit_code = -1
        except FileNotFoundError:
            logger.error(f"codex binary not found: {self.codex_binary}")
            stdout = ""
            stderr = f"codex binary not found: {self.codex_binary}"
            exit_code = -2

        duration = time.time() - start

        logger.info(
            f"Codex finished: exit_code={exit_code}, duration={duration:.1f}s"
        )

        # Save JSONL log and create human-readable text log
        log_jsonl_path = workspace_dir / "run_log.jsonl"
        log_jsonl_path.write_text(stdout)

        readable_log, usage = self._parse_jsonl_output(stdout)
        log_path = workspace_dir / "run_log.txt"
        log_path.write_text(
            f"=== CODEX RUN LOG ===\n"
            f"Model: {model.model_name}\n"
            f"Paper: {paper.paper_id}\n"
            f"Web access: {web_status}\n"
            f"Exit code: {exit_code}\n"
            f"Duration: {duration:.1f}s\n\n"
            f"=== CONVERSATION ===\n{readable_log}\n\n"
            f"=== STDERR ===\n{stderr}\n"
        )

        # Copy the most recent Codex session rollout as backup
        self._copy_session_rollout(workspace_dir, start)

        if usage:
            usage_path = workspace_dir.parent / "usage.json"
            usage_path.write_text(json_mod.dumps(usage, indent=2))
            logger.info(
                f"Token usage: {usage.get('total_tokens', 0):,} total, "
                f"{usage.get('num_turns', 0)} turns, "
                f"{usage.get('num_commands', 0)} commands"
            )

        return RunArtifacts(
            workspace_dir=str(workspace_dir),
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            duration_seconds=duration,
            usage=usage,
        )

    @staticmethod
    def _copy_session_rollout(workspace_dir: Path, run_start: float) -> None:
        """Copy the most recent Codex session rollout file into the workspace.

        Codex saves full conversation logs to ~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl.
        We find the rollout file created closest to (and after) run_start and copy it
        as session_rollout.jsonl for post-hoc analysis.
        """
        codex_home = Path.home() / ".codex" / "sessions"
        if not codex_home.is_dir():
            return

        # Find rollout files modified after the run started
        candidates = []
        for rollout in codex_home.rglob("rollout-*.jsonl"):
            try:
                mtime = rollout.stat().st_mtime
                if mtime >= run_start - 5:  # small tolerance
                    candidates.append((mtime, rollout))
            except OSError:
                continue

        if not candidates:
            return

        # Pick the most recently modified
        candidates.sort(key=lambda x: x[0], reverse=True)
        best = candidates[0][1]
        dest = workspace_dir / "session_rollout.jsonl"
        try:
            shutil.copy2(best, dest)
            logger.info(f"Copied Codex session rollout: {best.name} ({best.stat().st_size:,} bytes)")
        except OSError as e:
            logger.warning(f"Could not copy session rollout: {e}")

    @staticmethod
    def _parse_jsonl_output(stdout: str) -> tuple[str, dict | None]:
        """Parse JSONL events from codex --json output.

        Handles event types: thread.started, turn.started, turn.completed,
        turn.failed, item.started, item.updated, item.completed, error.
        Items include AgentMessageItem (text) and CommandExecutionItem (shell).

        Returns a human-readable log string and aggregated usage dict.
        """
        parts = []
        total_input = 0
        total_output = 0
        num_turns = 0
        per_turn: list[dict] = []
        tool_calls: list[str] = []
        num_commands = 0

        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json_mod.loads(line)
            except json_mod.JSONDecodeError:
                parts.append(f"[raw] {line[:500]}")
                continue

            etype = event.get("type", "")

            # -- Thread lifecycle -----------------------------------------
            if etype == "thread.started":
                parts.append("[thread.started]")

            # -- Turn lifecycle -------------------------------------------
            elif etype == "turn.started":
                parts.append(f"[turn.started]")

            elif etype == "turn.completed":
                num_turns += 1
                usage = event.get("usage", {})
                inp = usage.get("input_tokens", 0)
                out = usage.get("output_tokens", 0)
                total_input += inp
                total_output += out
                per_turn.append({"input_tokens": inp, "output_tokens": out})
                parts.append(f"[turn {num_turns} completed] tokens: in={inp}, out={out}")

            elif etype == "turn.failed":
                num_turns += 1
                err_msg = event.get("error", event.get("message", "unknown"))
                parts.append(f"[turn {num_turns} FAILED] {err_msg}")

            # -- Item events (messages, commands, tool calls) -------------
            elif etype in ("item.started", "item.updated", "item.completed"):
                item = event.get("item", event)
                item_type = item.get("type", "")
                phase = etype.split(".")[-1]  # started/updated/completed

                if item_type in ("agent_message", "message"):
                    text = item.get("text", "")
                    # Also check nested content blocks
                    if not text:
                        content = item.get("content", [])
                        if isinstance(content, list):
                            for block in content:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    text = block.get("text", "")
                                    break
                        elif isinstance(content, str):
                            text = content
                    if text and phase == "completed":
                        parts.append(f"[assistant] {text[:2000]}")

                elif item_type in ("command_execution", "command"):
                    cmd = item.get("command", "")
                    if phase == "started":
                        num_commands += 1
                        tool_calls.append(f"bash:{cmd[:100]}")
                        parts.append(f"[cmd {num_commands}] $ {cmd[:500]}")
                    elif phase == "completed":
                        exit_c = item.get("exit_code", "?")
                        output = item.get("aggregated_output", item.get("output", ""))
                        parts.append(
                            f"[cmd {num_commands} done] exit={exit_c}"
                            f" | {str(output)[:1000]}"
                        )

                elif item_type in ("tool_call", "function_call", "mcp_tool_call"):
                    name = item.get("name", "?")
                    if phase == "started":
                        tool_calls.append(name)
                        parts.append(f"[tool_call] {name}")
                    elif phase == "completed":
                        output = str(item.get("output", item.get("result", "")))
                        parts.append(f"[tool_result] {name}: {output[:1000]}")

                elif item_type == "file_change":
                    path = item.get("path", item.get("file", "?"))
                    action = item.get("action", "change")
                    if phase == "completed":
                        parts.append(f"[file_{action}] {path}")

                elif item_type == "reasoning":
                    # Internal reasoning — log but truncate
                    text = item.get("text", "")
                    if text and phase == "completed":
                        parts.append(f"[reasoning] {text[:500]}")

                else:
                    # Unknown item type — log for debugging
                    if phase == "completed":
                        parts.append(f"[item:{item_type}] {str(item)[:300]}")

            # -- Legacy message format (fallback) -------------------------
            elif etype == "message":
                role = event.get("role", "?")
                content = event.get("content", "")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            btype = block.get("type", "")
                            if btype == "text":
                                parts.append(f"[{role}] {block.get('text', '')}")
                            elif btype in ("tool_use", "function_call"):
                                name = block.get("name", "?")
                                tool_calls.append(name)
                                parts.append(f"[{role}:tool] {name}")
                            elif btype in ("tool_result", "function_call_output"):
                                output = str(block.get("output", block.get("content", "")))
                                parts.append(f"[tool_result] {output[:1000]}")
                elif isinstance(content, str) and content.strip():
                    parts.append(f"[{role}] {content}")

            # -- Exec completed -------------------------------------------
            elif etype == "exec.completed":
                parts.append(f"[exec.completed] exit_code={event.get('exit_code', '?')}")

            # -- Errors ---------------------------------------------------
            elif etype == "error":
                parts.append(f"[error] {event.get('message', str(event))}")

        readable = "\n".join(parts)

        if num_turns > 0 or total_input > 0:
            usage_dict: dict | None = {
                "prompt_tokens": total_input,
                "completion_tokens": total_output,
                "total_tokens": total_input + total_output,
                "num_turns": num_turns,
                "num_commands": num_commands,
                "tool_calls": tool_calls,
                "per_turn": per_turn,
            }
        else:
            usage_dict = None

        return readable, usage_dict
