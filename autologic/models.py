"""
autologic.models — Data models for the AutoLogic pipeline.

Provides structured containers for node results, shared pipeline context,
and overall pipeline execution results.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class NodeStatus(str, Enum):
    """Possible outcomes of a single node execution."""

    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class PipelineStatus(str, Enum):
    """Possible outcomes of the full pipeline execution."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass
class NodeResult:
    """Result of a single LogicNode execution."""

    node_name: str
    status: NodeStatus
    output: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    duration_seconds: float = 0.0

    @property
    def is_success(self) -> bool:
        """Check whether this node completed successfully."""
        return self.status == NodeStatus.SUCCESS

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "node_name": self.node_name,
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "duration_seconds": round(self.duration_seconds, 4),
        }


class PipelineContext:
    """
    Shared, thread-safe context that pipeline nodes read from and write to.

    Every node receives the same PipelineContext instance. Keys are arbitrary
    strings; values are any serializable Python object.

    Uses an asyncio.Lock so concurrent nodes (if ever scheduled in parallel)
    cannot corrupt the internal state.
    """

    def __init__(self, initial: dict[str, Any] | None = None) -> None:
        self._data: dict[str, Any] = dict(initial) if initial else {}
        self._lock: asyncio.Lock = asyncio.Lock()
        self._created_at: float = time.time()

    async def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from the context."""
        async with self._lock:
            return self._data.get(key, default)

    async def set(self, key: str, value: Any) -> None:
        """Store a value in the context."""
        async with self._lock:
            self._data[key] = value

    async def update(self, mapping: dict[str, Any]) -> None:
        """Merge multiple key-value pairs into the context at once."""
        async with self._lock:
            self._data.update(mapping)

    async def has(self, key: str) -> bool:
        """Check whether a key exists in the context."""
        async with self._lock:
            return key in self._data

    async def remove(self, key: str) -> Any:
        """Remove and return a key from the context. Returns None if missing."""
        async with self._lock:
            return self._data.pop(key, None)

    async def to_dict(self) -> dict[str, Any]:
        """Return a shallow copy of all context data."""
        async with self._lock:
            return dict(self._data)

    @property
    def created_at(self) -> float:
        """Timestamp when this context was created."""
        return self._created_at

    def __repr__(self) -> str:
        keys = list(self._data.keys())
        return f"PipelineContext(keys={keys})"


@dataclass
class PipelineResult:
    """Aggregated result of a full pipeline run."""

    node_results: list[NodeResult] = field(default_factory=list)
    total_duration: float = 0.0
    status: PipelineStatus = PipelineStatus.SUCCESS
    workspace_dir: str = ""

    @property
    def is_success(self) -> bool:
        """Check whether every node succeeded."""
        return self.status == PipelineStatus.SUCCESS

    @property
    def failed_nodes(self) -> list[str]:
        """Return names of nodes that failed."""
        return [
            nr.node_name
            for nr in self.node_results
            if nr.status == NodeStatus.FAILED
        ]

    @property
    def summary(self) -> str:
        """Human-readable one-line summary."""
        total = len(self.node_results)
        ok = sum(1 for nr in self.node_results if nr.is_success)
        return (
            f"Pipeline {self.status.value}: {ok}/{total} nodes succeeded "
            f"in {self.total_duration:.2f}s"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "status": self.status.value,
            "total_duration": round(self.total_duration, 4),
            "workspace_dir": self.workspace_dir,
            "node_results": [nr.to_dict() for nr in self.node_results],
        }
