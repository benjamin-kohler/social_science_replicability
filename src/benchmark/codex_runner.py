"""Codex CLI runner for freestyle replication."""

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

        # Save full CLI logs (including tool use steps) to workspace
        log_path = workspace_dir / "run_log.txt"
        log_path.write_text(
            f"=== CODEX RUN LOG ===\n"
            f"Model: {model.model_name}\n"
            f"Paper: {paper.paper_id}\n"
            f"Web access: {web_status}\n"
            f"Exit code: {exit_code}\n"
            f"Duration: {duration:.1f}s\n\n"
            f"=== STDOUT ===\n{stdout}\n\n"
            f"=== STDERR ===\n{stderr}\n"
        )

        return RunArtifacts(
            workspace_dir=str(workspace_dir),
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            duration_seconds=duration,
        )
