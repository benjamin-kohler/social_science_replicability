"""Benchmark framework for comparing LLM models on paper replication."""

from .config import BenchmarkConfig, ModelSpec, PaperSpec, JudgeConfig
from .runner import BenchmarkRunner, run_benchmark

__all__ = [
    "BenchmarkConfig",
    "BenchmarkRunner",
    "JudgeConfig",
    "ModelSpec",
    "PaperSpec",
    "run_benchmark",
]
