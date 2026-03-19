"""
autologic.logger — Centralized logging setup with colored console output.

Usage:
    from autologic.logger import setup_logger
    logger = setup_logger("engine", level="DEBUG", log_file="autologic.log")
    logger.info("Pipeline started")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Union

# ──────────────────────────────────────────────
# ANSI color codes for console output
# ──────────────────────────────────────────────

_COLORS: dict[int, str] = {
    logging.DEBUG: "\033[36m",     # Cyan
    logging.INFO: "\033[32m",      # Green
    logging.WARNING: "\033[33m",   # Yellow
    logging.ERROR: "\033[31m",     # Red
    logging.CRITICAL: "\033[1;31m",  # Bold Red
}
_RESET = "\033[0m"


class _ColoredFormatter(logging.Formatter):
    """Formatter that injects ANSI color codes based on log level."""

    FMT = "[{asctime}] [{levelname}] [{name}] {message}"
    DATE_FMT = "%Y-%m-%d %H:%M:%S"

    def __init__(self, use_colors: bool = True) -> None:
        super().__init__(fmt=self.FMT, datefmt=self.DATE_FMT, style="{")
        self._use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        # Pad level name for alignment
        record.levelname = f"{record.levelname:<8}"
        formatted = super().format(record)
        if self._use_colors:
            color = _COLORS.get(record.levelno, "")
            return f"{color}{formatted}{_RESET}"
        return formatted


class _PlainFormatter(logging.Formatter):
    """Non-colored formatter for file output."""

    FMT = "[{asctime}] [{levelname}] [{name}] {message}"
    DATE_FMT = "%Y-%m-%d %H:%M:%S"

    def __init__(self) -> None:
        super().__init__(fmt=self.FMT, datefmt=self.DATE_FMT, style="{")

    def format(self, record: logging.LogRecord) -> str:
        record.levelname = f"{record.levelname:<8}"
        return super().format(record)


def setup_logger(
    name: str,
    level: Union[str, int] = "INFO",
    log_file: Optional[str] = None,
    use_colors: bool = True,
) -> logging.Logger:
    """
    Configure and return a named logger with console (and optional file) output.

    Args:
        name: Logger name. Typically a module identifier like ``"engine"`` or
              ``"node.ingest"``. The logger is namespaced under ``autologic.``.
        level: Logging level — accepts strings (``"DEBUG"``, ``"INFO"``, etc.)
               or integer constants from :mod:`logging`.
        log_file: Optional filesystem path for a file handler.
        use_colors: Whether to apply ANSI colors to the console handler.
                    Automatically disabled on non-TTY streams.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    # Resolve level
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Fully-qualified logger name
    qualified_name = f"autologic.{name}" if not name.startswith("autologic") else name
    logger = logging.getLogger(qualified_name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    # Disable colors if stderr is not a real terminal
    effective_colors = use_colors and hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
    console_handler.setFormatter(_ColoredFormatter(use_colors=effective_colors))
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(_PlainFormatter())
        logger.addHandler(file_handler)

    # Prevent log propagation to the root logger (avoids duplicate messages)
    logger.propagate = False

    return logger
