"""Claude Code CLI runner for freestyle replication."""

import json as json_mod
import os
import subprocess
import time
from pathlib import Path

from ..models.schemas import PaperSummary
from ..utils.logging_utils import get_logger
from .config import ModelSpec, PaperSpec
from .results import RunArtifacts
from .task_prompt import setup_workspace

logger = get_logger(__name__)


class ClaudeCodeRunner:
    """Runs a freestyle replication using the Claude Code CLI (claude -p).

    Creates an isolated workspace with only the methodology summary and data,
    then invokes `claude -p` in headless mode. The model does NOT receive
    the original paper PDF or replication package.
    """

    def __init__(
        self,
        claude_binary: str = "claude",
        timeout: int = 600,
        max_turns: int = 50,
        allow_web_access: bool = False,
    ):
        self.claude_binary = claude_binary
        self.timeout = timeout
        self.max_turns = max_turns
        self.allow_web_access = allow_web_access

    def run(
        self,
        model: ModelSpec,
        paper: PaperSpec,
        paper_summary: PaperSummary,
        workspace_dir: Path,
    ) -> RunArtifacts:
        """Run a freestyle replication using Claude Code CLI.

        Args:
            model: Model specification.
            paper: Paper specification (used only for data_path).
            paper_summary: Pre-extracted methodology summary (no results).
            workspace_dir: Isolated workspace directory for this run.

        Returns:
            RunArtifacts with workspace contents, stdout, stderr, exit code, duration.
        """
        setup_workspace(paper, paper_summary, workspace_dir)

        # Build the inline prompt — includes isolation constraints
        prompt_text = (
            "Read TASK.md for your full instructions and constraints. "
            "IMPORTANT: Only access files inside this workspace directory. "
            "Do NOT read files outside this directory or search for the paper or its results. "
            "First explore the data files in this workspace to learn the actual "
            "column names. Then write Python scripts to replicate each table and figure. "
            "You MUST execute the scripts with bash and fix any errors until they run "
            "successfully. Use the exact output filenames specified in TASK.md for each item."
        )

        web_status = "ALLOWED" if self.allow_web_access else "BLOCKED"
        logger.info(
            f"Running claude-code: model={model.model_name}, "
            f"paper={paper.paper_id}, web_access={web_status}"
        )
        start = time.time()

        try:
            abs_workspace = str(Path(workspace_dir).resolve())
            cmd = [
                self.claude_binary, "-p",
                "--output-format", "json",
                "--model", model.model_name,
                "--dangerously-skip-permissions",
                "--max-turns", str(self.max_turns),
                "--no-session-persistence",
                "--verbose",
            ]
            if not self.allow_web_access:
                cmd.extend(["--disallowedTools", "WebSearch,WebFetch"])
            # Use -- to separate options from the positional prompt argument.
            # --disallowedTools is variadic and would otherwise consume the prompt.
            cmd.extend(["--", prompt_text])
            # Remove CLAUDECODE env var so the subprocess doesn't refuse
            # to launch when called from within a Claude Code session.
            env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
            result = subprocess.run(
                cmd,
                cwd=abs_workspace,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
            )
            stdout = result.stdout
            stderr = result.stderr
            exit_code = result.returncode
        except subprocess.TimeoutExpired:
            logger.warning(f"Claude Code timed out after {self.timeout}s")
            stdout = ""
            stderr = f"Timed out after {self.timeout} seconds"
            exit_code = -1
        except FileNotFoundError:
            logger.error(f"claude binary not found: {self.claude_binary}")
            stdout = ""
            stderr = f"claude binary not found: {self.claude_binary}"
            exit_code = -2

        duration = time.time() - start

        logger.info(
            f"Claude Code finished: exit_code={exit_code}, duration={duration:.1f}s"
        )

        # Save full CLI logs to workspace
        # JSON output contains the full conversation including tool use
        log_json_path = workspace_dir / "run_log.json"
        log_json_path.write_text(stdout)

        # Also create a human-readable text log
        log_txt_path = workspace_dir / "run_log.txt"
        readable_log = self._format_readable_log(
            stdout, stderr, model.model_name, paper.paper_id,
            exit_code, duration, web_status,
        )
        log_txt_path.write_text(readable_log)

        return RunArtifacts(
            workspace_dir=str(workspace_dir),
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            duration_seconds=duration,
        )

    @staticmethod
    def _format_readable_log(
        stdout: str,
        stderr: str,
        model_name: str,
        paper_id: str,
        exit_code: int,
        duration: float,
        web_status: str = "BLOCKED",
    ) -> str:
        """Convert JSON conversation output to a human-readable log."""
        header = (
            f"=== CLAUDE CODE RUN LOG ===\n"
            f"Model: {model_name}\n"
            f"Paper: {paper_id}\n"
            f"Web access: {web_status}\n"
            f"Exit code: {exit_code}\n"
            f"Duration: {duration:.1f}s\n\n"
        )

        # Try to parse JSON and extract conversation messages.
        # claude -p --output-format json emits an array of event objects:
        #   {"type": "assistant", "message": {"content": [...]}}
        #   {"type": "user", "message": {"content": [...]}}
        #   {"type": "result", "subtype": "success", "result": "...", "num_turns": N}
        #   {"type": "system", "subtype": "init"|"status"|"compact_boundary"}
        try:
            events = json_mod.loads(stdout)
            if not isinstance(events, list):
                events = [events]
            parts = []
            for event in events:
                etype = event.get("type", "")

                # Final result summary
                if etype == "result":
                    num_turns = event.get("num_turns", "?")
                    dur_ms = event.get("duration_ms", 0)
                    result_text = event.get("result", "")
                    parts.append(
                        f"[RESULT] turns={num_turns}, duration={dur_ms/1000:.1f}s\n"
                        f"{result_text}"
                    )
                    continue

                # System events (init, status, compact_boundary)
                if etype == "system":
                    subtype = event.get("subtype", "")
                    if subtype == "init":
                        parts.append(f"[system:init] model={event.get('model', '?')}")
                    continue

                # Assistant and user messages
                if etype not in ("assistant", "user"):
                    continue

                msg = event.get("message", event)
                role = msg.get("role", etype)
                content = msg.get("content", "")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            btype = block.get("type", "")
                            if btype == "text":
                                parts.append(f"[{role}] {block.get('text', '')}")
                            elif btype == "tool_use":
                                inp = json_mod.dumps(block.get("input", {}), indent=2)
                                if len(inp) > 500:
                                    inp = inp[:500] + "..."
                                parts.append(
                                    f"[{role}:tool_use] {block.get('name', '?')}({inp})"
                                )
                            elif btype == "tool_result":
                                result_content = block.get("content", "")
                                if isinstance(result_content, list):
                                    result_content = " ".join(
                                        b.get("text", "") for b in result_content if isinstance(b, dict)
                                    )
                                parts.append(f"[{role}:tool_result] {str(result_content)[:2000]}")
                            else:
                                parts.append(f"[{role}:{btype}] {json_mod.dumps(block)[:500]}")
                        else:
                            parts.append(f"[{role}] {block}")
                elif isinstance(content, str) and content:
                    parts.append(f"[{role}] {content}")
            conversation = "\n\n".join(parts)
        except (json_mod.JSONDecodeError, TypeError, KeyError):
            conversation = f"(Could not parse JSON output)\n\n{stdout}"

        return header + f"=== CONVERSATION ===\n{conversation}\n\n=== STDERR ===\n{stderr}\n"
