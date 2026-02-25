"""Benchmark framework for comparing LLM models on paper replication."""

from .config import BenchmarkConfig, ModelSpec, PaperSpec, JudgeConfig
from .claude_code_runner import ClaudeCodeRunner
from .codex_runner import CodexRunner
from .runner import BenchmarkRunner, run_benchmark

__all__ = [
    "BenchmarkConfig",
    "BenchmarkRunner",
    "ClaudeCodeRunner",
    "CodexRunner",
    "JudgeConfig",
    "ModelSpec",
    "PaperSpec",
    "run_benchmark",
]
