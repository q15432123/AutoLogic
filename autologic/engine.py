"""
autologic.engine — Core async orchestration engine.

The :class:`AutoLogicEngine` executes a sequence of :class:`LogicNode`
instances, passing a shared :class:`PipelineContext` through each one.
It emits lifecycle events that external code can subscribe to.

Usage::

    from autologic.config import AutoLogicConfig
    from autologic.engine import AutoLogicEngine

    config = AutoLogicConfig.from_file("config.yaml")
    engine = AutoLogicEngine.default_pipeline(config)
    result = await engine.run()
    print(result.summary)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Coroutine, Dict, List, Optional, Union

from .config import AutoLogicConfig
from .exceptions import NodeExecutionError, PipelineError
from .logger import setup_logger
from .models import (
    NodeResult,
    NodeStatus,
    PipelineContext,
    PipelineResult,
    PipelineStatus,
)
from .nodes.base import LogicNode
from .nodes.codegen_node import CodeGenNode
from .nodes.deploy_node import DeployNode
from .nodes.ingest_node import IngestNode
from .nodes.planning_node import PlanningNode
from .nodes.preprocess_node import PreprocessNode
from .nodes.verifier_node import VerifierNode
from .reflection import LogicValidator, ReflectiveExecutor
from .reasoning import ConcurrentReasoner

# Type alias for event handlers: can be sync or async callables
EventHandler = Union[
    Callable[..., None],
    Callable[..., Coroutine[Any, Any, None]],
]


class AutoLogicEngine:
    """
    Async pipeline engine that orchestrates :class:`LogicNode` execution.

    Nodes are executed sequentially in the order they were added. Each node
    receives the same :class:`PipelineContext` instance so it can read
    outputs from upstream nodes and write its own outputs.

    **Events** (subscribe with :meth:`on`)::

        node_start       — fires before each node; kwargs: node, context
        node_complete    — fires after each node succeeds; kwargs: node, result, context
        node_error       — fires when a node fails; kwargs: node, error, context
        pipeline_start   — fires once at the beginning; kwargs: context
        pipeline_complete — fires once at the end; kwargs: pipeline_result, context
    """

    def __init__(
        self,
        config: AutoLogicConfig,
        reflective: bool = False,
        concurrent_reasoning: bool = False,
    ) -> None:
        self.config: AutoLogicConfig = config
        self.nodes: List[LogicNode] = []
        self._logger: logging.Logger = setup_logger(
            "engine",
            level=config.log_level,
            log_file=config.log_file,
        )
        self._event_handlers: Dict[str, List[EventHandler]] = {}

        # ── Reflection & Reasoning (Gemini v2 upgrade) ──
        self.reflective = reflective
        self.concurrent_reasoning = concurrent_reasoning
        self._validator = LogicValidator(confidence_threshold=0.6)
        self._reflective_executor = ReflectiveExecutor(
            validator=self._validator, max_retries=3
        ) if reflective else None
        self._concurrent_reasoner = ConcurrentReasoner(
            validator=self._validator, num_branches=3
        ) if concurrent_reasoning else None
        # Set of node names that should use concurrent reasoning
        self._critical_nodes: set[str] = set()

    # ── Fluent node management ───────────────────

    def add_node(self, node: LogicNode) -> AutoLogicEngine:
        """
        Append a node to the pipeline. Returns *self* for chaining::

            engine.add_node(A()).add_node(B()).add_node(C())
        """
        self.nodes.append(node)
        self._logger.debug("Added node: %s", node.name)
        return self

    def insert_node(self, index: int, node: LogicNode) -> AutoLogicEngine:
        """Insert a node at a specific position."""
        self.nodes.insert(index, node)
        self._logger.debug("Inserted node '%s' at index %d", node.name, index)
        return self

    def remove_node(self, name: str) -> AutoLogicEngine:
        """Remove the first node matching *name*."""
        self.nodes = [n for n in self.nodes if n.name != name]
        return self

    def mark_critical(self, *node_names: str) -> AutoLogicEngine:
        """
        Mark nodes as critical. Critical nodes use concurrent reasoning
        (multiple parallel branches) when ``concurrent_reasoning=True``.

        Usage::

            engine.mark_critical("planning", "codegen")
        """
        self._critical_nodes.update(node_names)
        return self

    # ── Event system ─────────────────────────────

    def on(self, event: str, handler: EventHandler) -> None:
        """
        Register an event handler.

        Supported events:
            ``node_start``, ``node_complete``, ``node_error``,
            ``pipeline_start``, ``pipeline_complete``

        Handlers may be synchronous functions or async coroutines.
        """
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)

    async def _emit(self, event: str, **kwargs: Any) -> None:
        """Emit an event, calling all registered handlers."""
        handlers = self._event_handlers.get(event, [])
        for handler in handlers:
            try:
                result = handler(**kwargs)
                # Await if the handler is a coroutine
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                self._logger.warning(
                    "Event handler for '%s' raised: %s", event, exc
                )

    # ── Pipeline execution ───────────────────────

    async def run(
        self,
        context: Optional[PipelineContext] = None,
    ) -> PipelineResult:
        """
        Execute all nodes in sequence.

        Args:
            context: An optional pre-populated context. If *None*, a fresh
                     :class:`PipelineContext` is created with the workspace
                     directory from config.

        Returns:
            A :class:`PipelineResult` summarizing the entire run.

        Raises:
            PipelineError: If ``config.stop_on_error`` is ``True`` and a node
                           fails, or if the pipeline itself encounters an
                           unrecoverable error.
        """
        if context is None:
            context = PipelineContext({"workspace_dir": self.config.workspace_dir})

        pipeline_start_time = time.perf_counter()
        node_results: List[NodeResult] = []
        has_failure = False

        self._logger.info(
            "Pipeline starting with %d node(s): %s",
            len(self.nodes),
            " -> ".join(n.name for n in self.nodes),
        )
        await self._emit("pipeline_start", context=context)

        original_goal = await context.get("text_prompt", "")

        for node in self.nodes:
            # ── Emit node_start ──
            await self._emit("node_start", node=node, context=context)

            try:
                # ── Choose execution strategy ──
                if (
                    self.concurrent_reasoning
                    and self._concurrent_reasoner
                    and node.name in self._critical_nodes
                ):
                    # Concurrent reasoning: run N parallel branches
                    self._logger.info(
                        "Using concurrent reasoning for critical node '%s'",
                        node.name,
                    )
                    result, branches = await asyncio.wait_for(
                        self._concurrent_reasoner.reason(
                            node, context, original_goal
                        ),
                        timeout=self.config.node_timeout_seconds,
                    )
                    await self._emit(
                        "node_reasoning",
                        node=node,
                        branches=branches,
                        context=context,
                    )

                elif self.reflective and self._reflective_executor:
                    # Reflective execution: validate + retry with critique
                    result, attempts = await asyncio.wait_for(
                        self._reflective_executor.execute_with_reflection(
                            node, context, original_goal
                        ),
                        timeout=self.config.node_timeout_seconds,
                    )
                    if len(attempts) > 1:
                        await self._emit(
                            "node_reflection",
                            node=node,
                            attempts=attempts,
                            context=context,
                        )

                else:
                    # Standard execution
                    result = await asyncio.wait_for(
                        node.run(context),
                        timeout=self.config.node_timeout_seconds,
                    )

                # Store node result in context for VerifierNodes
                await context.set(
                    f"_node_result_{node.name}", result.output
                )

                node_results.append(result)
                await self._emit(
                    "node_complete", node=node, result=result, context=context
                )

            except asyncio.TimeoutError:
                error_msg = (
                    f"Node '{node.name}' timed out after "
                    f"{self.config.node_timeout_seconds}s"
                )
                self._logger.error(error_msg)
                failed_result = NodeResult(
                    node_name=node.name,
                    status=NodeStatus.FAILED,
                    error=error_msg,
                    duration_seconds=float(self.config.node_timeout_seconds),
                )
                node_results.append(failed_result)
                has_failure = True
                await self._emit(
                    "node_error",
                    node=node,
                    error=TimeoutError(error_msg),
                    context=context,
                )

                if self.config.stop_on_error:
                    self._logger.error("stop_on_error is set — aborting pipeline")
                    break

            except NodeExecutionError as exc:
                self._logger.error("Node '%s' failed: %s", node.name, exc)
                failed_result = NodeResult(
                    node_name=node.name,
                    status=NodeStatus.FAILED,
                    error=str(exc),
                    duration_seconds=0.0,
                )
                node_results.append(failed_result)
                has_failure = True
                await self._emit(
                    "node_error", node=node, error=exc, context=context
                )

                if self.config.stop_on_error:
                    self._logger.error("stop_on_error is set — aborting pipeline")
                    break

            except Exception as exc:
                self._logger.exception(
                    "Unexpected error in node '%s': %s", node.name, exc
                )
                failed_result = NodeResult(
                    node_name=node.name,
                    status=NodeStatus.FAILED,
                    error=f"Unexpected: {exc}",
                    duration_seconds=0.0,
                )
                node_results.append(failed_result)
                has_failure = True
                await self._emit(
                    "node_error", node=node, error=exc, context=context
                )

                if self.config.stop_on_error:
                    break

        # ── Build final result ───────────────────
        total_duration = time.perf_counter() - pipeline_start_time
        workspace_dir = await context.get("workspace_dir", self.config.workspace_dir)

        # Determine overall status
        all_ok = all(
            r.status in (NodeStatus.SUCCESS, NodeStatus.SKIPPED)
            for r in node_results
        )
        if all_ok:
            status = PipelineStatus.SUCCESS
        elif has_failure and any(r.is_success for r in node_results):
            status = PipelineStatus.PARTIAL
        else:
            status = PipelineStatus.FAILED

        pipeline_result = PipelineResult(
            node_results=node_results,
            total_duration=total_duration,
            status=status,
            workspace_dir=str(workspace_dir),
        )

        self._logger.info(pipeline_result.summary)
        await self._emit(
            "pipeline_complete",
            pipeline_result=pipeline_result,
            context=context,
        )

        return pipeline_result

    # ── Factory ──────────────────────────────────

    @classmethod
    def default_pipeline(cls, config: AutoLogicConfig) -> AutoLogicEngine:
        """
        Factory method that creates an engine pre-loaded with the standard
        five-node pipeline::

            PreprocessNode -> IngestNode -> PlanningNode -> CodeGenNode -> DeployNode

        Args:
            config: The application configuration.

        Returns:
            A fully-configured :class:`AutoLogicEngine`.
        """
        engine = cls(config)
        engine.add_node(PreprocessNode("preprocess"))
        engine.add_node(IngestNode("ingest"))
        engine.add_node(PlanningNode("planning"))
        engine.add_node(CodeGenNode("codegen"))
        engine.add_node(DeployNode("deploy"))
        return engine

    @classmethod
    def reflective_pipeline(cls, config: AutoLogicConfig) -> AutoLogicEngine:
        """
        Factory: standard pipeline with self-critique verification after
        planning and codegen nodes.

        Pipeline::

            Preprocess -> Ingest -> Planning -> VerifyPlan
            -> CodeGen -> VerifyCode -> Deploy

        Critical nodes (planning, codegen) use concurrent reasoning if enabled.
        """
        engine = cls(config, reflective=True, concurrent_reasoning=True)
        engine.add_node(PreprocessNode("preprocess"))
        engine.add_node(IngestNode("ingest"))
        engine.add_node(PlanningNode("planning"))
        engine.add_node(VerifierNode("verify_plan", target_node="planning"))
        engine.add_node(CodeGenNode("codegen"))
        engine.add_node(VerifierNode("verify_code", target_node="codegen"))
        engine.add_node(DeployNode("deploy"))
        engine.mark_critical("planning", "codegen")
        return engine

    def __repr__(self) -> str:
        node_names = [n.name for n in self.nodes]
        return f"AutoLogicEngine(nodes={node_names})"
