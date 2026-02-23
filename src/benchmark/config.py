"""Benchmark configuration models."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class ModelSpec(BaseModel):
    """Specification for a model to benchmark."""

    provider: str = Field(..., description="LLM provider: 'openai' or 'anthropic'")
    model_name: str = Field(..., description="Model name (e.g., 'gpt-4o')")
    api_key_env: str = Field(..., description="Environment variable name for the API key")
    approaches: Optional[list[str]] = Field(
        default=None,
        description="Restrict this model to specific approaches. None means all approaches.",
    )


class PaperSpec(BaseModel):
    """Specification for a paper to use in benchmarking."""

    paper_id: str = Field(..., description="Unique identifier for the paper")
    pdf_path: str = Field(..., description="Path to the paper PDF")
    data_path: str = Field(..., description="Path to the data file(s)")
    replication_package_path: Optional[str] = Field(
        default=None, description="Path to original replication package"
    )


class JudgeConfig(BaseModel):
    """Configuration for the judge model used for evaluation."""

    provider: str = Field(default="openai", description="LLM provider for the judge")
    model_name: str = Field(default="gpt-4o", description="Model name for the judge")


class BenchmarkConfig(BaseModel):
    """Top-level benchmark configuration."""

    models: list[ModelSpec] = Field(..., description="Models to benchmark")
    papers: list[PaperSpec] = Field(..., description="Papers to replicate")
    approaches: list[str] = Field(
        default=["freestyle", "structured"],
        description="Approaches to benchmark: 'freestyle' and/or 'structured'",
    )
    judge: JudgeConfig = Field(default_factory=JudgeConfig, description="Judge model config")
    output_dir: str = Field(default="data/benchmark_results", description="Output directory")
    opencode_binary: str = Field(default="opencode", description="Path to opencode binary")
    timeout_seconds: int = Field(default=600, description="Timeout per run in seconds")
