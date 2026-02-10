"""Tests for configuration management."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from src.models.config import (
    Config,
    ExecutionConfig,
    ExtractionConfig,
    OpenAgentConfig,
    OutputConfig,
    VerificationConfig,
    load_config,
)


class TestConfig:
    def test_default_values(self):
        config = Config()
        assert config.open_agent.default_provider == "openai"
        assert config.open_agent.temperature == 0.1
        assert config.execution.timeout_seconds == 300
        assert config.verification.numerical_tolerance == 0.01
        assert config.output.save_intermediate_results is True

    def test_custom_values(self):
        config = Config(
            open_agent=OpenAgentConfig(
                default_provider="anthropic",
                default_model="claude-3-opus-20240229",
                temperature=0.5,
            ),
            execution=ExecutionConfig(timeout_seconds=60),
        )
        assert config.open_agent.default_provider == "anthropic"
        assert config.execution.timeout_seconds == 60

    def test_api_keys_excluded_from_dump(self):
        config = Config(openai_api_key="secret", anthropic_api_key="secret2")
        d = config.model_dump()
        assert "openai_api_key" not in d
        assert "anthropic_api_key" not in d


class TestLoadConfig:
    def test_load_default(self):
        config = load_config()
        assert isinstance(config, Config)

    def test_load_from_yaml(self, tmp_path):
        yaml_content = {
            "open_agent": {
                "default_provider": "anthropic",
                "default_model": "claude-3-opus-20240229",
                "temperature": 0.3,
            },
            "execution": {"timeout_seconds": 120},
        }
        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(yaml_content, f)

        config = load_config(str(config_path))
        assert config.open_agent.default_provider == "anthropic"
        assert config.execution.timeout_seconds == 120

    def test_load_nonexistent_file(self):
        config = load_config("/nonexistent/path.yaml")
        # Should fall back to defaults
        assert isinstance(config, Config)

    def test_env_vars_override(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
        config = load_config()
        assert config.openai_api_key == "test-key-123"
