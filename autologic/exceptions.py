"""
autologic.exceptions — Custom exception hierarchy for AutoLogic.

All AutoLogic-specific exceptions inherit from AutoLogicError,
making it easy to catch any framework error with a single except clause.
"""

from __future__ import annotations


class AutoLogicError(Exception):
    """Base exception for all AutoLogic errors."""

    def __init__(self, message: str = "", *, details: dict | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


class NodeExecutionError(AutoLogicError):
    """Raised when a pipeline node fails during execution."""

    def __init__(
        self,
        message: str = "",
        *,
        node_name: str = "",
        original_error: Exception | None = None,
        details: dict | None = None,
    ) -> None:
        super().__init__(message, details=details)
        self.node_name = node_name
        self.original_error = original_error


class ConfigurationError(AutoLogicError):
    """Raised when configuration is invalid or missing."""

    def __init__(
        self,
        message: str = "",
        *,
        key: str = "",
        details: dict | None = None,
    ) -> None:
        super().__init__(message, details=details)
        self.key = key


class PipelineError(AutoLogicError):
    """Raised for pipeline-level failures (orchestration, sequencing, etc.)."""

    def __init__(
        self,
        message: str = "",
        *,
        failed_nodes: list[str] | None = None,
        details: dict | None = None,
    ) -> None:
        super().__init__(message, details=details)
        self.failed_nodes = failed_nodes or []


class PluginLoadError(AutoLogicError):
    """Raised when a plugin cannot be discovered, imported, or instantiated."""

    def __init__(
        self,
        message: str = "",
        *,
        plugin_path: str = "",
        details: dict | None = None,
    ) -> None:
        super().__init__(message, details=details)
        self.plugin_path = plugin_path
