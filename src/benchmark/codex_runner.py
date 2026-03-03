"""Codex CLI runner for freestyle replication."""

import json as json_mod
import subprocess
import time
from pathlib import Path

from ..models.schemas import PaperSummary
from ..utils.logging_utils import get_logger
from .config import ModelSpec, PaperSpec
from .results import RunArtifacts
from .task_prompt import setup_workspace

logger = get_logger(__name__)


class CodexRunner:
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
    ):
        self.codex_binary = codex_binary
        self.timeout = timeout
        self.allow_web_access = allow_web_access

    def run(
        self,
        model: ModelSpec,
        paper: PaperSpec,
        paper_summary: PaperSummary,
        workspace_dir: Path,
    ) -> RunArtifacts:
        """Run a freestyle replication using Codex CLI.

        Args:
            model: Model specification.
            paper: Paper specification (used only for data_path).
            paper_summary: Pre-extracted methodology summary (no results).
            workspace_dir: Isolated workspace directory for this run.

        Returns:
            RunArtifacts with workspace contents, stdout, stderr, exit code, duration.
        """
        setup_workspace(paper, paper_summary, workspace_dir)

        # Build the inline prompt — action-oriented to minimize wasted turns
        prompt_text = (
            "Read TASK.md for your full instructions and constraints. "
            "IMPORTANT: Only access files inside this workspace directory. "
            "Do NOT read files outside this directory or search for the paper or its results. "
            "TASK.md already describes the variables and data structure in detail. "
            "Run ONE quick command to check actual column names, then immediately start "
            "writing code. Write and execute each table/figure script ONE AT A TIME — "
            "write, run, fix errors, then move to the next. "
            "You MUST execute every script with bash and verify the output file exists. "
            "Use the exact output filenames specified in TASK.md for each item."
        )

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
                "--ephemeral",
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
        except subprocess.TimeoutExpired:
            logger.warning(f"Codex timed out after {self.timeout}s")
            stdout = ""
            stderr = f"Timed out after {self.timeout} seconds"
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

        if usage:
            usage_path = workspace_dir.parent / "usage.json"
            usage_path.write_text(json_mod.dumps(usage, indent=2))
            logger.info(
                f"Token usage: {usage.get('total_tokens', 0):,} total"
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
    def _parse_jsonl_output(stdout: str) -> tuple[str, dict | None]:
        """Parse JSONL events from codex --json output.

        Returns a human-readable log string and aggregated usage dict.
        """
        parts = []
        total_input = 0
        total_output = 0
        num_turns = 0
        per_turn: list[dict] = []

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

            if etype == "message":
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
                                parts.append(f"[{role}:tool] {name}")
                            elif btype in ("tool_result", "function_call_output"):
                                output = str(block.get("output", block.get("content", "")))
                                parts.append(f"[tool_result] {output[:1000]}")
                elif isinstance(content, str):
                    parts.append(f"[{role}] {content}")

            elif etype == "turn.completed":
                num_turns += 1
                usage = event.get("usage", {})
                inp = usage.get("input_tokens", 0)
                out = usage.get("output_tokens", 0)
                total_input += inp
                total_output += out
                per_turn.append({"input_tokens": inp, "output_tokens": out})
                parts.append(f"[turn {num_turns}] tokens: in={inp}, out={out}")

            elif etype == "exec.completed":
                parts.append(f"[exec.completed] exit_code={event.get('exit_code', '?')}")

            elif etype == "error":
                parts.append(f"[error] {event.get('message', str(event))}")

        readable = "\n".join(parts)

        if num_turns > 0 or total_input > 0:
            usage_dict = {
                "prompt_tokens": total_input,
                "completion_tokens": total_output,
                "total_tokens": total_input + total_output,
                "num_turns": num_turns,
                "per_turn": per_turn,
            }
        else:
            usage_dict = None

        return readable, usage_dict
