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
    use_vision: bool = Field(default=True, description="Use vision for figure comparison")


class ExtractorConfig(BaseModel):
    """Configuration for the extractor model used for methodology extraction."""

    model: str = Field(default="gpt-5.2", description="Model for extraction API calls")
    max_tokens: int = Field(default=128000, description="Max output tokens per call")
    use_vision: bool = Field(default=True, description="Send PDF pages as images")
    vision_dpi: int = Field(default=200, description="DPI for PDF-to-image conversion")


# ---------------------------------------------------------------------------
# Paper-direct approach helpers
# ---------------------------------------------------------------------------

PAPER_DIRECT_SUFFIX = "-paper"
PAPER_DIRECT_INCOMPATIBLE = {"structured"}


def parse_approach(approach: str) -> tuple[str, bool]:
    """Parse an approach string into (base_approach, is_paper_direct).

    Examples:
        "claude-code"       -> ("claude-code", False)
        "claude-code-paper" -> ("claude-code", True)
        "freestyle-paper"   -> ("freestyle", True)
    """
    if approach.endswith(PAPER_DIRECT_SUFFIX):
        return approach[: -len(PAPER_DIRECT_SUFFIX)], True
    return approach, False


class BenchmarkConfig(BaseModel):
    """Top-level benchmark configuration."""

    models: list[ModelSpec] = Field(..., description="Models to benchmark")
    papers: list[PaperSpec] = Field(..., description="Papers to replicate")
    approaches: list[str] = Field(
        default=["freestyle", "structured"],
        description="Approaches to benchmark: 'freestyle' and/or 'structured'",
    )
    judge: JudgeConfig = Field(default_factory=JudgeConfig, description="Judge model config")
    extractor: ExtractorConfig = Field(default_factory=ExtractorConfig, description="Extractor model config")
    output_dir: str = Field(default="data/benchmark_results", description="Output directory")
    opencode_binary: str = Field(default="opencode", description="Path to opencode binary")
    claude_code_binary: str = Field(default="claude", description="Path to claude CLI binary")
    codex_binary: str = Field(default="codex", description="Path to codex CLI binary")
    timeout_seconds: int = Field(default=600, description="Timeout per run in seconds")
    allow_web_access: bool = Field(
        default=False,
        description="Allow models to use web search during replication. "
        "Default False blocks WebSearch/WebFetch for information isolation.",
    )

    # Explainer configuration
    run_explainer: bool = Field(
        default=False,
        description="Run agentic explainer after judge to investigate discrepancies",
    )
    explainer_model: Optional[ModelSpec] = Field(
        default=None,
        description="Model for the explainer agent. Defaults to first model in models list.",
    )
    explainer_runner_type: str = Field(
        default="claude-code",
        description="CLI runner for explainer: 'claude-code' or 'codex'",
    )
    explainer_timeout_seconds: int = Field(
        default=600, description="Timeout for explainer runs in seconds"
    )
    explainer_max_turns: int = Field(
        default=30, description="Max turns for explainer agent"
    )
