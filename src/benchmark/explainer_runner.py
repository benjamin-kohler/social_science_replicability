"""Agentic Explainer runner — uses Claude Code or Codex CLI to investigate discrepancies.

Runs AFTER the judge. The explainer gets full access to all materials (paper PDF,
replication package, replicator code, logs, judge results) and produces a deep
root-cause analysis for each non-A item.
"""

import json as json_mod
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

from ..models.schemas import (
    AgenticDiscrepancyAnalysis,
    AgenticExplanationReport,
    CodeComparison,
    PaperSummary,
    ReplicationGrade,
)
from ..utils.logging_utils import get_logger
from .config import ModelSpec, PaperSpec
from .explainer_task_prompt import setup_explainer_workspace
from .results import EvaluationResult

logger = get_logger(__name__)


class ExplainerRunner:
    """Runs agentic explanation using Claude Code or Codex CLI.

    Sets up a workspace with all materials (paper, replication package,
    replicator code/outputs, methodology summary, logs, judge results),
    then launches a CLI subprocess to investigate discrepancies.
    """

    def __init__(
        self,
        runner_type: str = "claude-code",
        claude_binary: str = "claude",
        codex_binary: str = "codex",
        timeout: int = 600,
        max_turns: int = 30,
    ):
        if runner_type not in ("claude-code", "codex"):
            raise ValueError(f"Unsupported explainer runner_type: {runner_type}")
        self.runner_type = runner_type
        self.claude_binary = claude_binary
        self.codex_binary = codex_binary
        self.timeout = timeout
        self.max_turns = max_turns

    def run(
        self,
        model: ModelSpec,
        paper: PaperSpec,
        paper_summary: PaperSummary,
        replicator_workspace: Path,
        evaluation: EvaluationResult,
        workspace_dir: Path,
    ) -> AgenticExplanationReport:
        """Run the agentic explainer.

        Args:
            model: Model specification for the explainer.
            paper: Paper specification (pdf_path, data_path, replication_package_path).
            paper_summary: Pre-extracted methodology summary.
            replicator_workspace: Path to the replicator's workspace.
            evaluation: Judge evaluation result.
            workspace_dir: Target directory for the explainer workspace.

        Returns:
            AgenticExplanationReport with per-item analyses and fault categorization.
        """
        # Set up workspace with all materials
        setup_explainer_workspace(
            paper=paper,
            paper_summary=paper_summary,
            evaluation=evaluation,
            replicator_workspace=replicator_workspace,
            workspace_dir=workspace_dir,
        )

        prompt_text = (
            "Read TASK.md for your full instructions. "
            "You are investigating why an AI replicator's outputs differ from "
            "the original paper's results. You have access to ALL materials: "
            "paper PDF, replication package code, replicator code, outputs, logs, "
            "and judge grades. "
            "For each non-A item, diagnose the root cause and categorize fault. "
            "You MUST write explainer_report.json and explanation.md as specified in TASK.md."
        )

        logger.info(
            f"Running explainer ({self.runner_type}): model={model.model_name}, "
            f"paper={paper.paper_id}"
        )
        start = time.time()

        if self.runner_type == "claude-code":
            stdout, stderr, exit_code = self._run_claude_code(
                model, workspace_dir, prompt_text,
            )
        else:
            stdout, stderr, exit_code = self._run_codex(
                model, workspace_dir, prompt_text,
            )

        duration = time.time() - start
        logger.info(
            f"Explainer finished: exit_code={exit_code}, duration={duration:.1f}s"
        )

        # Save logs
        self._save_logs(workspace_dir, stdout, stderr, model, paper, exit_code, duration)

        # Extract usage
        usage = self._extract_usage(stdout)

        # Parse explainer_report.json from workspace
        report = self._parse_report(
            workspace_dir, paper_summary.paper_id, model.model_name,
            duration, usage,
        )

        return report

    # -- CLI subprocess invocation ------------------------------------------

    def _run_claude_code(
        self, model: ModelSpec, workspace_dir: Path, prompt_text: str,
    ) -> tuple[str, str, int]:
        """Launch Claude Code CLI subprocess."""
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
                "--", prompt_text,
            ]
            env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
            result = subprocess.run(
                cmd, cwd=abs_workspace, capture_output=True, text=True,
                timeout=self.timeout, env=env,
            )
            return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            logger.warning(f"Claude Code explainer timed out after {self.timeout}s")
            return "", f"Timed out after {self.timeout} seconds", -1
        except FileNotFoundError:
            logger.error(f"claude binary not found: {self.claude_binary}")
            return "", f"claude binary not found: {self.claude_binary}", -2

    def _run_codex(
        self, model: ModelSpec, workspace_dir: Path, prompt_text: str,
    ) -> tuple[str, str, int]:
        """Launch Codex CLI subprocess."""
        try:
            abs_workspace = str(Path(workspace_dir).resolve())
            cmd = [
                self.codex_binary, "exec",
                "--full-auto", "--json",
                "-m", model.model_name,
                "-C", abs_workspace,
                "--skip-git-repo-check",
                prompt_text,
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout,
            )
            return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired as exc:
            logger.warning(f"Codex explainer timed out after {self.timeout}s")
            stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else (exc.stdout or b"").decode(errors="replace")
            stderr_partial = (exc.stderr or "") if isinstance(exc.stderr, str) else (exc.stderr or b"").decode(errors="replace")
            return stdout, f"Timed out after {self.timeout} seconds\n{stderr_partial}".strip(), -1
        except FileNotFoundError:
            logger.error(f"codex binary not found: {self.codex_binary}")
            return "", f"codex binary not found: {self.codex_binary}", -2

    # -- Log saving ---------------------------------------------------------

    def _save_logs(
        self, workspace_dir: Path, stdout: str, stderr: str,
        model: ModelSpec, paper: PaperSpec, exit_code: int, duration: float,
    ) -> None:
        """Save raw and human-readable logs."""
        if self.runner_type == "claude-code":
            (workspace_dir / "run_log.json").write_text(stdout)
            readable = self._format_claude_log(stdout, stderr, model, paper, exit_code, duration)
        else:
            (workspace_dir / "run_log.jsonl").write_text(stdout)
            readable = self._format_codex_log(stdout, stderr, model, paper, exit_code, duration)
        (workspace_dir / "run_log.txt").write_text(readable)

    @staticmethod
    def _format_claude_log(
        stdout: str, stderr: str, model: ModelSpec, paper: PaperSpec,
        exit_code: int, duration: float,
    ) -> str:
        """Format Claude Code JSON output as human-readable log."""
        # Reuse the same pattern as ClaudeCodeRunner._format_readable_log
        from .claude_code_runner import ClaudeCodeRunner
        return ClaudeCodeRunner._format_readable_log(
            stdout, stderr, model.model_name, paper.paper_id, exit_code, duration,
        )

    @staticmethod
    def _format_codex_log(
        stdout: str, stderr: str, model: ModelSpec, paper: PaperSpec,
        exit_code: int, duration: float,
    ) -> str:
        """Format Codex JSONL output as human-readable log."""
        from .codex_runner import CodexRunner
        readable, _ = CodexRunner._parse_jsonl_output(stdout)
        return (
            f"=== CODEX EXPLAINER LOG ===\n"
            f"Model: {model.model_name}\n"
            f"Paper: {paper.paper_id}\n"
            f"Exit code: {exit_code}\n"
            f"Duration: {duration:.1f}s\n\n"
            f"=== CONVERSATION ===\n{readable}\n\n"
            f"=== STDERR ===\n{stderr}\n"
        )

    # -- Usage extraction ---------------------------------------------------

    def _extract_usage(self, stdout: str) -> Optional[dict]:
        """Extract token usage from CLI output."""
        if self.runner_type == "claude-code":
            from .claude_code_runner import ClaudeCodeRunner
            return ClaudeCodeRunner._extract_usage(stdout)
        else:
            from .codex_runner import CodexRunner
            _, usage = CodexRunner._parse_jsonl_output(stdout)
            return usage

    # -- Report parsing -----------------------------------------------------

    def _parse_report(
        self, workspace_dir: Path, paper_id: str, model_name: str,
        duration: float, usage: Optional[dict],
    ) -> AgenticExplanationReport:
        """Parse explainer_report.json from the workspace.

        Falls back to a minimal report if the file is missing or malformed.
        """
        report_path = workspace_dir / "explainer_report.json"
        if not report_path.exists():
            logger.warning("explainer_report.json not found in workspace")
            return self._fallback_report(paper_id, model_name, duration, usage,
                                         "Explainer did not produce explainer_report.json")

        try:
            raw = report_path.read_text()
            data = json_mod.loads(raw)
        except (json_mod.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to parse explainer_report.json: {e}")
            return self._fallback_report(paper_id, model_name, duration, usage,
                                         f"Failed to parse explainer_report.json: {e}")

        # Parse analyses
        analyses = []
        for item in data.get("analyses", []):
            code_comp = None
            if item.get("code_comparison"):
                cc = item["code_comparison"]
                code_comp = CodeComparison(
                    item_id=cc.get("item_id", item.get("item_id", "")),
                    replicator_approach=cc.get("replicator_approach", ""),
                    original_approach=cc.get("original_approach", ""),
                    key_differences=cc.get("key_differences", []),
                )

            grade_str = item.get("grade", "F")
            try:
                grade = ReplicationGrade(grade_str)
            except ValueError:
                grade = ReplicationGrade.F

            analyses.append(AgenticDiscrepancyAnalysis(
                item_id=item.get("item_id", "Unknown"),
                grade=grade,
                verbal_explanation=item.get("verbal_explanation", ""),
                code_comparison=code_comp,
                fault_category=item.get("fault_category", "unclear"),
                fault_explanation=item.get("fault_explanation", ""),
                confidence=item.get("confidence", "low"),
                supporting_evidence=item.get("supporting_evidence", []),
                suggested_fix=item.get("suggested_fix"),
            ))

        # Build fault summary
        fault_summary: dict[str, int] = {}
        for a in analyses:
            fault_summary[a.fault_category] = fault_summary.get(a.fault_category, 0) + 1

        return AgenticExplanationReport(
            paper_id=data.get("paper_id", paper_id),
            analyses=analyses,
            overall_assessment=data.get("overall_assessment", ""),
            methodology_quality_notes=data.get("methodology_quality_notes", ""),
            fault_summary=fault_summary,
            runner_model=model_name,
            runner_type=self.runner_type,
            duration_seconds=duration,
            usage=usage,
        )

    def _fallback_report(
        self, paper_id: str, model_name: str, duration: float,
        usage: Optional[dict], error_msg: str,
    ) -> AgenticExplanationReport:
        """Create a minimal report when the explainer fails to produce output."""
        return AgenticExplanationReport(
            paper_id=paper_id,
            analyses=[],
            overall_assessment=f"Explainer failed: {error_msg}",
            methodology_quality_notes="",
            fault_summary={},
            runner_model=model_name,
            runner_type=self.runner_type,
            duration_seconds=duration,
            usage=usage,
        )
