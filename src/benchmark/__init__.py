"""Benchmark framework for comparing LLM models on paper replication."""

from .base_runner import BaseReplicationRunner
from .config import BenchmarkConfig, ModelSpec, PaperSpec, JudgeConfig
from .claude_code_runner import ClaudeCodeRunner
from .codex_runner import CodexRunner
from .explainer_runner import ExplainerRunner
from .swe_agent_runner import SweAgentRunner
from .runner import BenchmarkRunner, run_benchmark

__all__ = [
    "BaseReplicationRunner",
    "BenchmarkConfig",
    "BenchmarkRunner",
    "ClaudeCodeRunner",
    "CodexRunner",
    "ExplainerRunner",
    "SweAgentRunner",
    "JudgeConfig",
    "ModelSpec",
    "PaperSpec",
    "run_benchmark",
]
