"""
autologic.plugins.loader — Dynamic plugin discovery and loading.

Scans a directory for Python files, imports them, and collects any classes
that are subclasses of :class:`~autologic.nodes.base.LogicNode`.

Usage::

    from autologic.plugins.loader import discover_plugins

    plugins = discover_plugins("./my_plugins")
    for PluginCls in plugins:
        engine.add_node(PluginCls(name=PluginCls.__name__.lower()))
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
import sys
from pathlib import Path
from typing import Type

from ..exceptions import PluginLoadError
from ..nodes.base import LogicNode

logger = logging.getLogger("autologic.plugins.loader")


def load_plugin(path: str | Path) -> Type[LogicNode]:
    """
    Load a single Python file and return the first :class:`LogicNode` subclass
    found in it.

    Args:
        path: Filesystem path to a ``.py`` file containing a LogicNode subclass.

    Returns:
        The discovered :class:`LogicNode` subclass (not an instance).

    Raises:
        PluginLoadError: If the file cannot be imported or contains no
            LogicNode subclass.
    """
    file_path = Path(path).resolve()

    if not file_path.exists():
        raise PluginLoadError(
            f"Plugin file not found: {file_path}",
            plugin_path=str(file_path),
        )

    if not file_path.suffix == ".py":
        raise PluginLoadError(
            f"Plugin must be a .py file: {file_path}",
            plugin_path=str(file_path),
        )

    module_name = f"autologic_plugin_{file_path.stem}"

    try:
        spec = importlib.util.spec_from_file_location(module_name, str(file_path))
        if spec is None or spec.loader is None:
            raise PluginLoadError(
                f"Cannot create module spec for: {file_path}",
                plugin_path=str(file_path),
            )

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except PluginLoadError:
        raise
    except Exception as exc:
        raise PluginLoadError(
            f"Failed to import plugin {file_path}: {exc}",
            plugin_path=str(file_path),
        ) from exc

    # Find LogicNode subclasses (excluding LogicNode itself)
    node_classes = [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, LogicNode) and obj is not LogicNode
    ]

    if not node_classes:
        raise PluginLoadError(
            f"No LogicNode subclass found in: {file_path}",
            plugin_path=str(file_path),
        )

    chosen = node_classes[0]
    logger.info("Loaded plugin: %s from %s", chosen.__name__, file_path.name)
    return chosen


def discover_plugins(directory: str | Path) -> list[Type[LogicNode]]:
    """
    Scan a directory for ``.py`` files and collect all :class:`LogicNode`
    subclasses found in them.

    Files whose names start with ``_`` (e.g., ``__init__.py``) are skipped.

    Args:
        directory: Path to the plugin directory to scan.

    Returns:
        A list of discovered :class:`LogicNode` subclasses. The list may be
        empty if no valid plugins are found.
    """
    dir_path = Path(directory).resolve()

    if not dir_path.is_dir():
        logger.warning("Plugin directory does not exist: %s", dir_path)
        return []

    plugins: list[Type[LogicNode]] = []

    for py_file in sorted(dir_path.glob("*.py")):
        # Skip private / dunder files
        if py_file.name.startswith("_"):
            continue

        try:
            plugin_cls = load_plugin(py_file)
            plugins.append(plugin_cls)
        except PluginLoadError as exc:
            logger.warning("Skipping plugin %s: %s", py_file.name, exc)
        except Exception as exc:
            logger.warning("Unexpected error loading %s: %s", py_file.name, exc)

    logger.info(
        "Discovered %d plugin(s) in %s",
        len(plugins), dir_path,
    )
    return plugins
