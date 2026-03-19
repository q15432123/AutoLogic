"""
Tests for autologic.config — Configuration loading and resolution.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from autologic.config import AutoLogicConfig
from autologic.exceptions import ConfigurationError


# ──────────────────────────────────────────────
# YAML loading
# ──────────────────────────────────────────────

_SAMPLE_YAML = """\
autologic:
  version: "0.2.0"

gemini:
  api_key: "yaml-key-123"
  model: "gemini-2.0-flash"

whisper:
  model_size: "small"

pipeline:
  workspace_dir: "_test_workspaces"
  max_concurrent_nodes: 8
  node_timeout_seconds: 300
  stop_on_error: true

server:
  host: "127.0.0.1"
  port: 9000

deploy:
  firebase_project_id: "test-project"

logging:
  level: "DEBUG"
  file: "test.log"
"""


def test_config_from_yaml():
    """Loading from a YAML file should populate all properties."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as f:
        f.write(_SAMPLE_YAML)
        yaml_path = f.name

    try:
        config = AutoLogicConfig.from_file(yaml_path)

        assert config.gemini_api_key == "yaml-key-123"
        assert config.gemini_model == "gemini-2.0-flash"
        assert config.whisper_model_size == "small"
        assert config.workspace_dir == "_test_workspaces"
        assert config.max_concurrent_nodes == 8
        assert config.node_timeout_seconds == 300
        assert config.stop_on_error is True
        assert config.server_host == "127.0.0.1"
        assert config.server_port == 9000
        assert config.firebase_project_id == "test-project"
        assert config.log_level == "DEBUG"
        assert config.log_file == "test.log"
    finally:
        os.unlink(yaml_path)


def test_config_from_yaml_missing_file():
    """from_file with a nonexistent path should raise ConfigurationError."""
    with pytest.raises(ConfigurationError):
        AutoLogicConfig.from_file("/nonexistent/path/config.yaml")


def test_config_from_yaml_invalid_yaml():
    """from_file with invalid YAML should raise ConfigurationError."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as f:
        f.write("{{{{invalid yaml: [")
        yaml_path = f.name

    try:
        with pytest.raises(ConfigurationError):
            AutoLogicConfig.from_file(yaml_path)
    finally:
        os.unlink(yaml_path)


# ──────────────────────────────────────────────
# Environment variable overrides
# ──────────────────────────────────────────────

def test_config_env_override(monkeypatch):
    """Environment variables should override YAML values."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as f:
        f.write(_SAMPLE_YAML)
        yaml_path = f.name

    try:
        monkeypatch.setenv("GEMINI_API_KEY", "env-key-456")
        monkeypatch.setenv("AUTOLOGIC_LOG_LEVEL", "ERROR")
        monkeypatch.setenv("AUTOLOGIC_PORT", "4321")
        monkeypatch.setenv("AUTOLOGIC_STOP_ON_ERROR", "false")

        config = AutoLogicConfig.from_file(yaml_path)

        # Env vars win over YAML
        assert config.gemini_api_key == "env-key-456"
        assert config.log_level == "ERROR"
        assert config.server_port == 4321
        assert config.stop_on_error is False

        # YAML values still used where no env override
        assert config.whisper_model_size == "small"
        assert config.server_host == "127.0.0.1"
    finally:
        os.unlink(yaml_path)


def test_config_from_env(monkeypatch):
    """from_env should work with environment variables alone."""
    monkeypatch.setenv("GEMINI_API_KEY", "pure-env-key")
    monkeypatch.setenv("FIREBASE_PROJECT_ID", "env-project")
    monkeypatch.setenv("AUTOLOGIC_LOG_LEVEL", "WARNING")

    config = AutoLogicConfig.from_env()

    assert config.gemini_api_key == "pure-env-key"
    assert config.firebase_project_id == "env-project"
    assert config.log_level == "WARNING"


# ──────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────

def test_config_defaults():
    """An empty config should fall back to all defaults."""
    config = AutoLogicConfig({})

    assert config.gemini_api_key == ""
    assert config.gemini_model == "gemini-1.5-pro-latest"
    assert config.workspace_dir == "_workspaces"
    assert config.log_level == "INFO"
    assert config.log_file is None
    assert config.whisper_model_size == "base"
    assert config.max_concurrent_nodes == 4
    assert config.node_timeout_seconds == 120
    assert config.server_host == "0.0.0.0"
    assert config.server_port == 8000
    assert config.stop_on_error is False


# ──────────────────────────────────────────────
# Utility methods
# ──────────────────────────────────────────────

def test_config_to_dict():
    """to_dict should return a complete copy of all config values."""
    config = AutoLogicConfig({"gemini_api_key": "abc123"})
    data = config.to_dict()

    assert isinstance(data, dict)
    assert data["gemini_api_key"] == "abc123"
    assert "workspace_dir" in data  # defaults should be present


def test_config_get_arbitrary_key():
    """get() should retrieve arbitrary keys including custom ones."""
    config = AutoLogicConfig({"custom_setting": 42})
    assert config.get("custom_setting") == 42
    assert config.get("missing", "fallback") == "fallback"


def test_config_repr_masks_keys():
    """__repr__ should mask keys/secrets for safety."""
    config = AutoLogicConfig({"gemini_api_key": "super_secret"})
    r = repr(config)
    assert "super_secret" not in r
    assert "***" in r
