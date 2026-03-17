"""SWE-agent (mini-swe-agent) runner for freestyle replication."""

import json as json_mod
import logging
import signal
import time
from io import StringIO
from pathlib import Path

import yaml

from ..models.schemas import PaperSummary
from ..utils.logging_utils import get_logger
from .base_runner import BaseReplicationRunner
from .config import ModelSpec, PaperSpec
from .results import RunArtifacts
from .task_prompt import setup_workspace, setup_workspace_paper_direct

logger = get_logger(__name__)


def _to_litellm_model(model: ModelSpec) -> str:
    """Convert ModelSpec to litellm model name format.

    litellm uses 'provider/model' for non-OpenAI providers.
    OpenAI models use just the model name.
    """
    if model.provider.lower() == "openai":
        return model.model_name
    return f"{model.provider}/{model.model_name}"


class SweAgentRunner(BaseReplicationRunner):
    """Runs a freestyle replication using mini-swe-agent's Python API.

    Creates an isolated workspace with only the methodology summary and data,
    then invokes DefaultAgent with a LocalEnvironment pointed at the workspace.
    The model does NOT receive the original paper PDF or replication package.
    """

    def __init__(
        self,
        timeout: int = 600,
        step_limit: int = 0,
        cost_limit: float = 3.0,
        allow_web_access: bool = False,
    ):
        super().__init__(timeout=timeout, allow_web_access=allow_web_access)
        self.step_limit = step_limit
        self.cost_limit = cost_limit

    def run(
        self,
        model: ModelSpec,
        paper: PaperSpec,
        paper_summary: PaperSummary,
        workspace_dir: Path,
        paper_direct: bool = False,
    ) -> RunArtifacts:
        """Run a freestyle replication using mini-swe-agent.

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

        if paper_direct:
            prompt_text = (
                "Read TASK.md for your full instructions and constraints. "
                "IMPORTANT: Only access files inside this workspace directory. "
                "Do NOT search for the paper or its results online. "
                "Read the paper PDF (paper.pdf) to understand the methodology, "
                "then replicate all tables and figures using the provided data. "
                "Write and execute each script ONE AT A TIME. "
                "You MUST execute every script with bash and verify the output file exists. "
                "Use the naming convention: table_N.py -> table_N.csv, figure_N.py -> figure_N.png."
            )
        else:
            prompt_text = (
                "Read TASK.md for your full instructions and constraints. "
                "IMPORTANT: Only access files inside this workspace directory. "
                "Do NOT read files outside this directory or search for the paper or its results. "
                "First explore the data files in this workspace to learn the actual "
                "column names. Then write Python scripts to replicate each table and figure. "
                "You MUST execute the scripts with bash and fix any errors until they run "
                "successfully. Use the exact output filenames specified in TASK.md for each item."
            )

        litellm_model = _to_litellm_model(model)
        web_status = "ALLOWED" if self.allow_web_access else "BLOCKED"
        logger.info(
            f"Running swe-agent: model={litellm_model}, "
            f"paper={paper.paper_id}, web_access={web_status}"
        )
        start = time.time()

        # Capture log output during the agent run
        log_buffer = StringIO()
        log_handler = logging.StreamHandler(log_buffer)
        log_handler.setLevel(logging.DEBUG)
        log_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
        root_logger = logging.getLogger()
        root_logger.addHandler(log_handler)

        agent = None
        try:
            from minisweagent.agents.default import DefaultAgent
            from minisweagent.models.litellm_model import LitellmModel
            from minisweagent.environments.local import LocalEnvironment

            import minisweagent

            abs_workspace = str(Path(workspace_dir).resolve())

            # Load default agent config (system/instance templates)
            default_config_path = minisweagent.package_dir / "config" / "default.yaml"
            with open(default_config_path) as f:
                default_config = yaml.safe_load(f)
            agent_defaults = default_config.get("agent", {})
            env_defaults = default_config.get("environment", {})

            trajectory_path = workspace_dir / "trajectory.json"
            agent = DefaultAgent(
                LitellmModel(model_name=litellm_model),
                LocalEnvironment(
                    cwd=abs_workspace,
                    **{k: v for k, v in env_defaults.items() if k != "cwd"},
                ),
                system_template=agent_defaults["system_template"],
                instance_template=agent_defaults["instance_template"],
                step_limit=self.step_limit or agent_defaults.get("step_limit", 0),
                cost_limit=self.cost_limit,
                output_path=trajectory_path,
            )

            result = self._run_with_timeout(agent, prompt_text)

            exit_status = result.get("exit_status", "unknown") if isinstance(result, dict) else "unknown"
            submission = result.get("submission", "") if isinstance(result, dict) else str(result)
            exit_code = 0 if exit_status not in ("LimitsExceeded", "error") else 1
            stdout = json_mod.dumps({
                "exit_status": exit_status,
                "submission": submission,
                "cost": getattr(agent, "cost", 0),
                "n_calls": getattr(agent, "n_calls", 0),
            }, indent=2)
            stderr = ""

        except _TimeoutError:
            logger.warning(f"SWE-agent timed out after {self.timeout}s")
            stdout = ""
            stderr = f"Timed out after {self.timeout} seconds"
            exit_code = -1
        except ImportError as e:
            logger.error(f"mini-swe-agent not installed: {e}")
            stdout = ""
            stderr = f"mini-swe-agent not installed: {e}"
            exit_code = -2
        except Exception as e:
            logger.error(f"SWE-agent failed: {e}")
            stdout = ""
            stderr = str(e)
            exit_code = 1
        finally:
            root_logger.removeHandler(log_handler)
            log_handler.close()

        duration = time.time() - start
        captured_log = log_buffer.getvalue()

        logger.info(
            f"SWE-agent finished: exit_code={exit_code}, duration={duration:.1f}s"
        )

        # Save run log
        log_path = workspace_dir / "run_log.txt"
        log_path.write_text(
            f"=== SWE-AGENT RUN LOG ===\n"
            f"Model: {litellm_model}\n"
            f"Paper: {paper.paper_id}\n"
            f"Web access: {web_status}\n"
            f"Exit code: {exit_code}\n"
            f"Duration: {duration:.1f}s\n\n"
            f"=== AGENT LOG ===\n{captured_log}\n\n"
            f"=== STDOUT ===\n{stdout}\n\n"
            f"=== STDERR ===\n{stderr}\n"
        )

        # Save usage if agent ran
        usage = None
        if agent is not None:
            cost = getattr(agent, "cost", 0)
            n_calls = getattr(agent, "n_calls", 0)
            usage = {
                "total_cost_usd": cost,
                "num_calls": n_calls,
            }

            # Extract token counts from agent stats or trajectory file
            stats = getattr(agent, "stats", None)
            if stats is not None:
                usage["prompt_tokens"] = getattr(stats, "tokens_sent", 0)
                usage["completion_tokens"] = getattr(stats, "tokens_received", 0)
                usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

            # Fallback: parse trajectory file for model_stats
            if "prompt_tokens" not in usage or usage.get("prompt_tokens", 0) == 0:
                traj_path = workspace_dir / "trajectory.json"
                if traj_path.exists():
                    try:
                        traj = json_mod.loads(traj_path.read_text())
                        ms = (traj.get("info", {}) or {}).get("model_stats", {})
                        ts = ms.get("tokens_sent", 0)
                        tr = ms.get("tokens_received", 0)
                        if ts > 0 or tr > 0:
                            usage["prompt_tokens"] = ts
                            usage["completion_tokens"] = tr
                            usage["total_tokens"] = ts + tr
                    except Exception:
                        pass

            usage_path = workspace_dir.parent / "usage.json"
            usage_path.write_text(json_mod.dumps(usage, indent=2))

        return RunArtifacts(
            workspace_dir=str(workspace_dir),
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            duration_seconds=duration,
            usage=usage,
        )

    def _run_with_timeout(self, agent, task: str):
        """Run agent.run() with a wall-clock timeout via SIGALRM."""
        if self.timeout <= 0:
            return agent.run(task)

        def _alarm_handler(signum, frame):
            raise _TimeoutError(f"Timed out after {self.timeout}s")

        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(self.timeout)
        try:
            return agent.run(task)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


class _TimeoutError(Exception):
    """Internal timeout exception for SWE-agent runs."""
