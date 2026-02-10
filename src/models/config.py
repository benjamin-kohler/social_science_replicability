"""Configuration management for the replication system."""

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field
from dotenv import load_dotenv


class OpenAgentConfig(BaseModel):
    """Configuration for the agent framework."""

    default_provider: str = Field(default="openai", description="LLM provider to use")
    default_model: str = Field(default="gpt-4-turbo-preview", description="Default model")
    temperature: float = Field(default=0.1, description="Temperature for LLM calls")
    max_tokens: int = Field(default=4000, description="Max tokens per response")


class ExecutionConfig(BaseModel):
    """Configuration for code execution."""

    timeout_seconds: int = Field(default=300, description="Timeout for code execution")
    max_retries: int = Field(default=3, description="Max retries on failure")
    sandbox_type: str = Field(default="jupyter", description="Type of sandbox to use")


class ExtractionConfig(BaseModel):
    """Configuration for paper extraction."""

    focus_sections: list[str] = Field(
        default=["Methods", "Results", "Data"], description="Sections to focus on"
    )
    extract_appendix: bool = Field(default=False, description="Whether to extract appendix")


class VerificationConfig(BaseModel):
    """Configuration for result verification."""

    numerical_tolerance: float = Field(
        default=0.01, description="Tolerance for numerical differences (1% = 0.01)"
    )
    use_vision_model: bool = Field(
        default=True, description="Use vision model for figure comparison"
    )


class OutputConfig(BaseModel):
    """Configuration for outputs."""

    save_intermediate_results: bool = Field(
        default=True, description="Save intermediate results"
    )
    reports_dir: str = Field(default="reports", description="Directory for reports")
    figures_format: str = Field(default="png", description="Format for saved figures")


class Config(BaseModel):
    """Main configuration class."""

    open_agent: OpenAgentConfig = Field(default_factory=OpenAgentConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    # API Keys (loaded from environment)
    openai_api_key: Optional[str] = Field(default=None, exclude=True)
    anthropic_api_key: Optional[str] = Field(default=None, exclude=True)


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file and environment.

    Args:
        config_path: Path to YAML config file. If None, uses default config.

    Returns:
        Loaded Config object.
    """
    # Load environment variables
    load_dotenv()

    # Start with default config
    config_dict = {}

    # Load from file if provided
    if config_path:
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file) as f:
                config_dict = yaml.safe_load(f) or {}
    else:
        # Try to load default config
        default_path = Path(__file__).parent.parent.parent / "config" / "default_config.yaml"
        if default_path.exists():
            with open(default_path) as f:
                config_dict = yaml.safe_load(f) or {}

    # Create config with file values
    config = Config(**config_dict)

    # Override with environment variables
    config.openai_api_key = os.getenv("OPENAI_API_KEY")
    config.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

    return config


def get_llm_client(config: Config):
    """Get the appropriate LLM client based on configuration.

    Args:
        config: Configuration object.

    Returns:
        Configured LLM client (OpenAI or Anthropic).

    Raises:
        ValueError: If no API key is configured for the selected provider.
    """
    provider = config.open_agent.default_provider.lower()

    if provider == "openai":
        if not config.openai_api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")
        from openai import OpenAI
        return OpenAI(api_key=config.openai_api_key)

    elif provider == "anthropic":
        if not config.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in environment")
        from anthropic import Anthropic
        return Anthropic(api_key=config.anthropic_api_key)

    else:
        raise ValueError(f"Unsupported provider: {provider}")
