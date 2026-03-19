"""
autologic.plugins — Dynamic plugin system for third-party LogicNode extensions.
"""

from .loader import discover_plugins, load_plugin

__all__ = ["discover_plugins", "load_plugin"]
