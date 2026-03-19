"""
autologic.config — Configuration loader for AutoLogic.

Loads settings with the following precedence (highest wins):
  1. Environment variables
  2. .env file (via python-dotenv)
  3. config.yaml file
  4. Built-in defaults
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv

from .exceptions import ConfigurationError


# ──────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────

_DEFAULTS: dict[str, Any] = {
    "gemini_api_key": "",
    "gemini_model": "gemini-1.5-pro-latest",
    "firebase_project_id": "",
    "workspace_dir": "_workspaces",
    "log_level": "INFO",
    "log_file": None,
    "whisper_model_size": "base",
    "max_concurrent_nodes": 4,
    "node_timeout_seconds": 120,
    "server_host": "0.0.0.0",
    "server_port": 8000,
    "stop_on_error": False,
}


class AutoLogicConfig:
    """
    Immutable configuration container for AutoLogic.

    Instantiate via the factory class methods :meth:`from_file` or
    :meth:`from_env` rather than calling ``__init__`` directly.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self._data: dict[str, Any] = {**_DEFAULTS, **data}

    # ── Factory methods ──────────────────────────

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        env_file: str | Path | None = None,
    ) -> AutoLogicConfig:
        """
        Load configuration from a YAML file, then overlay env-var overrides.

        Args:
            path: Path to a ``config.yaml`` file.
            env_file: Optional path to a ``.env`` file. If *None* the loader
                      looks for ``.env`` next to the YAML file.

        Returns:
            A fully-resolved :class:`AutoLogicConfig`.

        Raises:
            ConfigurationError: If the YAML file cannot be read or parsed.
        """
        yaml_path = Path(path).resolve()
        if not yaml_path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {yaml_path}",
                key="config_file",
            )

        try:
            with open(yaml_path, "r", encoding="utf-8") as fh:
                raw: dict[str, Any] = yaml.safe_load(fh) or {}
        except yaml.YAMLError as exc:
            raise ConfigurationError(
                f"Failed to parse YAML config: {exc}",
                key="config_file",
            ) from exc

        # Flatten nested YAML structure into our flat key space
        flat = cls._flatten_yaml(raw)

        # Load .env (does not overwrite existing env vars)
        env_path = Path(env_file) if env_file else yaml_path.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=False)

        # Overlay environment variables
        flat = cls._apply_env_overrides(flat)

        return cls(flat)

    @classmethod
    def from_env(cls) -> AutoLogicConfig:
        """
        Load configuration purely from environment variables and ``.env``.

        Useful for containerized deployments where no YAML file is mounted.

        Returns:
            A fully-resolved :class:`AutoLogicConfig`.
        """
        load_dotenv(override=False)
        return cls(cls._apply_env_overrides({}))

    # ── Properties ───────────────────────────────

    @property
    def gemini_api_key(self) -> str:
        """Google Gemini API key."""
        return str(self._data["gemini_api_key"])

    @property
    def gemini_model(self) -> str:
        """Gemini model identifier."""
        return str(self._data["gemini_model"])

    @property
    def firebase_project_id(self) -> str:
        """Firebase project ID for deployment."""
        return str(self._data["firebase_project_id"])

    @property
    def workspace_dir(self) -> str:
        """Directory where generated code is written."""
        return str(self._data["workspace_dir"])

    @property
    def log_level(self) -> str:
        """Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)."""
        return str(self._data["log_level"]).upper()

    @property
    def log_file(self) -> Optional[str]:
        """Optional log file path. None means console-only."""
        val = self._data.get("log_file")
        return str(val) if val else None

    @property
    def whisper_model_size(self) -> str:
        """OpenAI Whisper model size (tiny, base, small, medium, large)."""
        return str(self._data["whisper_model_size"])

    @property
    def max_concurrent_nodes(self) -> int:
        """Maximum number of nodes that may execute concurrently."""
        return int(self._data["max_concurrent_nodes"])

    @property
    def node_timeout_seconds(self) -> int:
        """Per-node execution timeout in seconds."""
        return int(self._data["node_timeout_seconds"])

    @property
    def server_host(self) -> str:
        """Host address for the FastAPI server."""
        return str(self._data["server_host"])

    @property
    def server_port(self) -> int:
        """Port number for the FastAPI server."""
        return int(self._data["server_port"])

    @property
    def stop_on_error(self) -> bool:
        """Whether the pipeline should abort on the first node failure."""
        return bool(self._data["stop_on_error"])

    # ── Utility ──────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        """Generic getter for arbitrary config keys."""
        return self._data.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        """Return a copy of all configuration values."""
        return dict(self._data)

    def __repr__(self) -> str:
        safe = {k: ("***" if "key" in k.lower() or "secret" in k.lower() else v)
                for k, v in self._data.items()}
        return f"AutoLogicConfig({safe})"

    # ── Internal helpers ─────────────────────────

    @staticmethod
    def _flatten_yaml(raw: dict[str, Any]) -> dict[str, Any]:
        """
        Map the nested YAML structure to our flat config keys.

        Expected YAML sections:
            gemini.api_key       -> gemini_api_key
            gemini.model         -> gemini_model
            whisper.model_size   -> whisper_model_size
            pipeline.*           -> workspace_dir, max_concurrent_nodes, ...
            server.*             -> server_host, server_port
            deploy.*             -> firebase_project_id
            logging.*            -> log_level, log_file
        """
        flat: dict[str, Any] = {}

        gemini = raw.get("gemini", {})
        if isinstance(gemini, dict):
            if gemini.get("api_key"):
                flat["gemini_api_key"] = gemini["api_key"]
            if gemini.get("model"):
                flat["gemini_model"] = gemini["model"]

        whisper_cfg = raw.get("whisper", {})
        if isinstance(whisper_cfg, dict):
            if whisper_cfg.get("model_size"):
                flat["whisper_model_size"] = whisper_cfg["model_size"]

        pipeline = raw.get("pipeline", {})
        if isinstance(pipeline, dict):
            for key in ("workspace_dir", "max_concurrent_nodes",
                        "node_timeout_seconds", "stop_on_error"):
                if key in pipeline:
                    flat[key] = pipeline[key]

        server = raw.get("server", {})
        if isinstance(server, dict):
            if "host" in server:
                flat["server_host"] = server["host"]
            if "port" in server:
                flat["server_port"] = server["port"]

        deploy = raw.get("deploy", {})
        if isinstance(deploy, dict):
            if deploy.get("firebase_project_id"):
                flat["firebase_project_id"] = deploy["firebase_project_id"]

        log_cfg = raw.get("logging", {})
        if isinstance(log_cfg, dict):
            if log_cfg.get("level"):
                flat["log_level"] = log_cfg["level"]
            if log_cfg.get("file"):
                flat["log_file"] = log_cfg["file"]

        return flat

    @staticmethod
    def _apply_env_overrides(flat: dict[str, Any]) -> dict[str, Any]:
        """Override flat config values with environment variables when set."""
        env_map: dict[str, str] = {
            "GEMINI_API_KEY": "gemini_api_key",
            "GEMINI_MODEL": "gemini_model",
            "FIREBASE_PROJECT_ID": "firebase_project_id",
            "AUTOLOGIC_WORKSPACE_DIR": "workspace_dir",
            "AUTOLOGIC_LOG_LEVEL": "log_level",
            "AUTOLOGIC_LOG_FILE": "log_file",
            "WHISPER_MODEL_SIZE": "whisper_model_size",
            "AUTOLOGIC_MAX_CONCURRENT_NODES": "max_concurrent_nodes",
            "AUTOLOGIC_NODE_TIMEOUT": "node_timeout_seconds",
            "AUTOLOGIC_HOST": "server_host",
            "AUTOLOGIC_PORT": "server_port",
            "AUTOLOGIC_STOP_ON_ERROR": "stop_on_error",
        }

        for env_var, config_key in env_map.items():
            value = os.getenv(env_var)
            if value is not None and value != "":
                # Coerce booleans
                if config_key == "stop_on_error":
                    flat[config_key] = value.lower() in ("1", "true", "yes")
                # Coerce integers
                elif config_key in ("max_concurrent_nodes", "node_timeout_seconds", "server_port"):
                    try:
                        flat[config_key] = int(value)
                    except ValueError:
                        pass  # Keep existing value if env var is not a valid int
                else:
                    flat[config_key] = value

        return flat
