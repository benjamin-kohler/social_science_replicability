"""Freestyle approach runner using opencode."""

import json as json_mod
import os
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


class OpencodeRunner(BaseReplicationRunner):
    """Runs a freestyle replication using the opencode CLI.

    Creates an isolated workspace with only the methodology summary and data,
    then invokes opencode to let the model figure out the replication.
    The model does NOT receive the original paper PDF or replication package.
    """

    def __init__(
        self,
        opencode_binary: str = "opencode",
        timeout: int = 600,
        allow_web_access: bool = False,
    ):
        super().__init__(timeout=timeout, allow_web_access=allow_web_access)
        self.opencode_binary = opencode_binary

    def run(
        self,
        model: ModelSpec,
        paper: PaperSpec,
        paper_summary: PaperSummary,
        workspace_dir: Path,
        paper_direct: bool = False,
    ) -> RunArtifacts:
        """Run a freestyle replication from a methodology summary.

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
            setup_workspace(paper, paper_summary, workspace_dir)

        # Build the inline prompt — single sentence pointing to TASK.md.
        if paper_direct:
            prompt_text = "Read TASK.md for your full instructions. Start by examining paper.pdf."
        else:
            prompt_text = "Read TASK.md for your full instructions."

        web_status = "ALLOWED" if self.allow_web_access else "BLOCKED"
        logger.info(
            f"Running opencode freestyle: model={model.model_name}, "
            f"paper={paper.paper_id}, web_access={web_status}"
        )
        start = time.time()

        try:
            # opencode CLI syntax: opencode run -m provider/model --dir workspace "message"
            model_id = f"{model.provider}/{model.model_name}"
            abs_workspace = str(Path(workspace_dir).resolve())
            result = subprocess.run(
                [
                    self.opencode_binary, "run",
                    "--print-logs",
                    "--log-level", "DEBUG",
                    "--format", "json",
                    "-m", model_id,
                    "--dir", abs_workspace,
                    "-f", "TASK.md",
                    "--",
                    prompt_text,
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={**os.environ, "PATH": f"{Path.home() / '.opencode' / 'bin'}:{os.environ.get('PATH', '')}"},
            )
            stdout = result.stdout
            stderr = result.stderr
            exit_code = result.returncode
        except subprocess.TimeoutExpired:
            logger.warning(f"Opencode timed out after {self.timeout}s")
            stdout = ""
            stderr = f"Timed out after {self.timeout} seconds"
            exit_code = -1
        except FileNotFoundError:
            logger.error(f"opencode binary not found: {self.opencode_binary}")
            stdout = ""
            stderr = f"opencode binary not found: {self.opencode_binary}"
            exit_code = -2

        duration = time.time() - start

        logger.info(
            f"Opencode finished: exit_code={exit_code}, duration={duration:.1f}s"
        )

        # Save raw JSONL output and parsed human-readable log
        log_jsonl_path = workspace_dir / "run_log.jsonl"
        log_jsonl_path.write_text(stdout)

        readable_log, usage = self._parse_jsonl_output(stdout)
        log_path = workspace_dir / "run_log.txt"
        log_path.write_text(
            f"=== OPENCODE RUN LOG ===\n"
            f"Model: {model.provider}/{model.model_name}\n"
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
                f"Token usage: {usage.get('total_tokens', 0):,} total, "
                f"{usage.get('num_steps', 0)} steps"
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
        """Parse JSONL events from opencode --format json output.

        Returns a human-readable log string and aggregated usage dict.
        """
        parts = []
        total_input = 0
        total_output = 0
        num_steps = 0
        tool_calls: list[str] = []

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

            if etype == "text":
                text = event.get("content", "")
                if text.strip():
                    parts.append(f"[assistant] {text[:2000]}")

            elif etype == "tool_call":
                name = event.get("name", "?")
                tool_calls.append(name)
                args = event.get("arguments", event.get("input", ""))
                if isinstance(args, dict):
                    args = json_mod.dumps(args)
                parts.append(f"[tool_call] {name}({str(args)[:500]})")

            elif etype == "tool_result":
                output = str(event.get("content", event.get("output", "")))
                parts.append(f"[tool_result] {output[:1000]}")

            elif etype == "step":
                num_steps += 1
                step_usage = event.get("usage", {})
                inp = step_usage.get("input_tokens", 0)
                out = step_usage.get("output_tokens", 0)
                total_input += inp
                total_output += out
                parts.append(f"[step {num_steps}] tokens: in={inp}, out={out}")

            elif etype == "summary" or etype == "done":
                usage_data = event.get("usage", {})
                if usage_data:
                    total_input = usage_data.get("input_tokens", total_input)
                    total_output = usage_data.get("output_tokens", total_output)
                parts.append(f"[{etype}] {json_mod.dumps(event)[:500]}")

            elif etype == "error":
                parts.append(f"[error] {event.get('message', str(event))}")

        readable = "\n".join(parts)

        if num_steps > 0 or total_input > 0:
            usage_dict: dict | None = {
                "prompt_tokens": total_input,
                "completion_tokens": total_output,
                "total_tokens": total_input + total_output,
                "num_steps": num_steps,
                "tool_calls": tool_calls,
            }
        else:
            usage_dict = None

        return readable, usage_dict
