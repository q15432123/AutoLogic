"""
autologic.nodes.base — Abstract Base Class for all pipeline nodes.

Every pipeline node must subclass :class:`LogicNode` and implement the
:meth:`execute` coroutine. The :meth:`run` method is the public entry point
called by the engine; it wraps *execute* with validation, timing, and
error handling.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

from ..exceptions import NodeExecutionError
from ..models import NodeResult, NodeStatus, PipelineContext


class LogicNode(ABC):
    """
    Abstract base class for all AutoLogic pipeline nodes.

    Subclasses **must** override :meth:`execute` and may optionally override
    :meth:`validate` (pre-execution checks) and :meth:`on_error` (custom
    error recovery).

    The engine calls :meth:`run`, which orchestrates the full lifecycle::

        validate -> execute -> NodeResult(success)
                     |
                     +---> on_error -> NodeResult(failed)
    """

    def __init__(self, name: str, description: str = "") -> None:
        self.name: str = name
        self.description: str = description
        self._logger: logging.Logger = logging.getLogger(f"autologic.node.{name}")

    @abstractmethod
    async def execute(self, context: PipelineContext) -> NodeResult:
        """
        Execute this node's core logic.

        Implementations should read inputs from *context* via
        ``await context.get(key)`` and write outputs via
        ``await context.set(key, value)``.

        Args:
            context: The shared pipeline context.

        Returns:
            A :class:`NodeResult` describing the outcome.
        """
        ...

    async def validate(self, context: PipelineContext) -> bool:
        """
        Optional pre-execution validation.

        Override this to verify that required keys exist in *context* or that
        external services are reachable. Return ``False`` to skip execution
        (the node result will have status :attr:`NodeStatus.SKIPPED`).

        Args:
            context: The shared pipeline context.

        Returns:
            ``True`` if the node is ready to execute, ``False`` to skip.
        """
        return True

    async def on_error(self, error: Exception, context: PipelineContext) -> None:
        """
        Optional error handler invoked when :meth:`execute` raises.

        Override for custom recovery logic (e.g. writing partial results,
        sending alerts, or setting context flags for downstream nodes).

        Args:
            error: The exception that was raised.
            context: The shared pipeline context.
        """
        pass

    async def run(self, context: PipelineContext) -> NodeResult:
        """
        Main entry point called by the engine.

        Orchestrates the full node lifecycle:

        1. Calls :meth:`validate`. If it returns ``False``, the node is
           skipped and a :attr:`NodeStatus.SKIPPED` result is returned.
        2. Times and calls :meth:`execute`.
        3. On success, returns the :class:`NodeResult` from *execute*.
        4. On failure, calls :meth:`on_error` and returns a
           :attr:`NodeStatus.FAILED` result containing the error message.

        Args:
            context: The shared pipeline context.

        Returns:
            A :class:`NodeResult` with timing and status information.
        """
        # Validation
        try:
            is_valid = await self.validate(context)
        except Exception as exc:
            self._logger.warning("Validation raised an exception: %s", exc)
            is_valid = False

        if not is_valid:
            self._logger.info("Node '%s' skipped (validation returned False)", self.name)
            return NodeResult(
                node_name=self.name,
                status=NodeStatus.SKIPPED,
                output={},
                error="Validation failed or returned False",
                duration_seconds=0.0,
            )

        # Execution with timing
        start = time.perf_counter()
        try:
            self._logger.info("Node '%s' starting execution", self.name)
            result = await self.execute(context)
            elapsed = time.perf_counter() - start
            result.duration_seconds = elapsed
            self._logger.info(
                "Node '%s' completed in %.3fs (status=%s)",
                self.name, elapsed, result.status.value,
            )
            return result

        except Exception as exc:
            elapsed = time.perf_counter() - start
            self._logger.error(
                "Node '%s' failed after %.3fs: %s", self.name, elapsed, exc,
            )

            # Give the subclass a chance to handle the error
            try:
                await self.on_error(exc, context)
            except Exception as recovery_exc:
                self._logger.error(
                    "on_error handler for '%s' also raised: %s",
                    self.name, recovery_exc,
                )

            raise NodeExecutionError(
                f"Node '{self.name}' failed: {exc}",
                node_name=self.name,
                original_error=exc,
            ) from exc

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
