"""
Tests for autologic.engine — Core async orchestration engine.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from autologic.engine import AutoLogicEngine
from autologic.config import AutoLogicConfig
from autologic.exceptions import NodeExecutionError
from autologic.models import (
    NodeResult,
    NodeStatus,
    PipelineContext,
    PipelineStatus,
)
from autologic.nodes.base import LogicNode


# ──────────────────────────────────────────────
# Helpers: lightweight mock nodes
# ──────────────────────────────────────────────

class PassNode(LogicNode):
    """A node that always succeeds."""

    async def execute(self, context: PipelineContext) -> NodeResult:
        await context.set(f"{self.name}_ran", True)
        return NodeResult(
            node_name=self.name,
            status=NodeStatus.SUCCESS,
            output={"message": f"{self.name} completed"},
        )


class FailNode(LogicNode):
    """A node that always raises."""

    async def execute(self, context: PipelineContext) -> NodeResult:
        raise RuntimeError(f"{self.name} exploded")


class SkipNode(LogicNode):
    """A node whose validation always returns False."""

    async def validate(self, context: PipelineContext) -> bool:
        return False

    async def execute(self, context: PipelineContext) -> NodeResult:
        # Should never be reached
        return NodeResult(
            node_name=self.name,
            status=NodeStatus.SUCCESS,
        )


def _make_config(**overrides: Any) -> AutoLogicConfig:
    """Build a minimal config for testing."""
    defaults = {
        "log_level": "WARNING",
        "node_timeout_seconds": 5,
        "stop_on_error": False,
        "workspace_dir": "/tmp/autologic_test",
    }
    defaults.update(overrides)
    return AutoLogicConfig(defaults)


# ──────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_engine_empty_pipeline():
    """An engine with no nodes should complete successfully with an empty result."""
    config = _make_config()
    engine = AutoLogicEngine(config)

    result = await engine.run()

    assert result.status == PipelineStatus.SUCCESS
    assert result.node_results == []
    assert result.total_duration >= 0
    assert result.is_success is True


@pytest.mark.asyncio
async def test_engine_single_node():
    """A single passing node should run and appear in results."""
    config = _make_config()
    engine = AutoLogicEngine(config)
    engine.add_node(PassNode("alpha"))

    result = await engine.run()

    assert result.status == PipelineStatus.SUCCESS
    assert len(result.node_results) == 1
    assert result.node_results[0].node_name == "alpha"
    assert result.node_results[0].status == NodeStatus.SUCCESS


@pytest.mark.asyncio
async def test_engine_fluent_api():
    """add_node should return the engine for chaining."""
    config = _make_config()
    engine = AutoLogicEngine(config)
    ret = engine.add_node(PassNode("a")).add_node(PassNode("b"))

    assert ret is engine
    assert len(engine.nodes) == 2


@pytest.mark.asyncio
async def test_engine_multiple_nodes_sequence():
    """Nodes execute in order and each can see earlier nodes' context writes."""
    config = _make_config()
    engine = AutoLogicEngine(config)
    engine.add_node(PassNode("first"))
    engine.add_node(PassNode("second"))
    engine.add_node(PassNode("third"))

    context = PipelineContext()
    result = await engine.run(context)

    assert result.status == PipelineStatus.SUCCESS
    assert len(result.node_results) == 3

    # Verify all nodes wrote to context
    assert await context.get("first_ran") is True
    assert await context.get("second_ran") is True
    assert await context.get("third_ran") is True


@pytest.mark.asyncio
async def test_engine_node_failure_continues():
    """With stop_on_error=False, later nodes still execute after a failure."""
    config = _make_config(stop_on_error=False)
    engine = AutoLogicEngine(config)
    engine.add_node(PassNode("before"))
    engine.add_node(FailNode("bad"))
    engine.add_node(PassNode("after"))

    result = await engine.run()

    assert result.status == PipelineStatus.PARTIAL
    assert len(result.node_results) == 3
    assert result.node_results[0].status == NodeStatus.SUCCESS
    assert result.node_results[1].status == NodeStatus.FAILED
    assert result.node_results[2].status == NodeStatus.SUCCESS
    assert "bad" in result.failed_nodes


@pytest.mark.asyncio
async def test_engine_node_failure_stops():
    """With stop_on_error=True, the pipeline aborts on the first failure."""
    config = _make_config(stop_on_error=True)
    engine = AutoLogicEngine(config)
    engine.add_node(PassNode("before"))
    engine.add_node(FailNode("bad"))
    engine.add_node(PassNode("never_runs"))

    result = await engine.run()

    assert result.status in (PipelineStatus.PARTIAL, PipelineStatus.FAILED)
    assert len(result.node_results) == 2  # "never_runs" was not executed
    assert result.node_results[1].status == NodeStatus.FAILED


@pytest.mark.asyncio
async def test_engine_skipped_node():
    """A node whose validate() returns False should be SKIPPED, not FAILED."""
    config = _make_config()
    engine = AutoLogicEngine(config)
    engine.add_node(SkipNode("optional"))

    result = await engine.run()

    assert result.status == PipelineStatus.SUCCESS
    assert len(result.node_results) == 1
    assert result.node_results[0].status == NodeStatus.SKIPPED


@pytest.mark.asyncio
async def test_engine_event_emission():
    """Verify that pipeline_start, node_start, node_complete, and pipeline_complete events fire."""
    config = _make_config()
    engine = AutoLogicEngine(config)
    engine.add_node(PassNode("evented"))

    events_received: list[str] = []

    async def on_pipeline_start(**kwargs):
        events_received.append("pipeline_start")

    async def on_node_start(**kwargs):
        events_received.append(f"node_start:{kwargs['node'].name}")

    async def on_node_complete(**kwargs):
        events_received.append(f"node_complete:{kwargs['node'].name}")

    async def on_pipeline_complete(**kwargs):
        events_received.append("pipeline_complete")

    engine.on("pipeline_start", on_pipeline_start)
    engine.on("node_start", on_node_start)
    engine.on("node_complete", on_node_complete)
    engine.on("pipeline_complete", on_pipeline_complete)

    await engine.run()

    assert events_received == [
        "pipeline_start",
        "node_start:evented",
        "node_complete:evented",
        "pipeline_complete",
    ]


@pytest.mark.asyncio
async def test_engine_node_error_event():
    """The node_error event should fire when a node fails."""
    config = _make_config()
    engine = AutoLogicEngine(config)
    engine.add_node(FailNode("broken"))

    error_events: list[str] = []

    async def on_node_error(**kwargs):
        error_events.append(f"error:{kwargs['node'].name}")

    engine.on("node_error", on_node_error)

    await engine.run()

    assert error_events == ["error:broken"]


@pytest.mark.asyncio
async def test_engine_sync_event_handler():
    """Synchronous (non-async) event handlers should also work."""
    config = _make_config()
    engine = AutoLogicEngine(config)
    engine.add_node(PassNode("sync_test"))

    called = []

    def on_start(**kwargs):
        called.append("sync_start")

    engine.on("pipeline_start", on_start)

    await engine.run()

    assert "sync_start" in called


@pytest.mark.asyncio
async def test_pipeline_context_get_set():
    """PipelineContext basic get/set/to_dict operations."""
    ctx = PipelineContext({"initial_key": 42})

    assert await ctx.get("initial_key") == 42
    assert await ctx.get("missing", "default") == "default"

    await ctx.set("new_key", "hello")
    assert await ctx.get("new_key") == "hello"

    data = await ctx.to_dict()
    assert data == {"initial_key": 42, "new_key": "hello"}


@pytest.mark.asyncio
async def test_pipeline_context_update_and_has():
    """PipelineContext update and has methods."""
    ctx = PipelineContext()

    assert await ctx.has("x") is False
    await ctx.update({"x": 1, "y": 2})
    assert await ctx.has("x") is True
    assert await ctx.get("y") == 2


@pytest.mark.asyncio
async def test_pipeline_context_remove():
    """PipelineContext remove returns the value and deletes the key."""
    ctx = PipelineContext({"temp": "data"})

    val = await ctx.remove("temp")
    assert val == "data"
    assert await ctx.has("temp") is False

    # Removing a missing key returns None
    assert await ctx.remove("nonexistent") is None


@pytest.mark.asyncio
async def test_pipeline_result_summary():
    """PipelineResult.summary should produce a human-readable string."""
    from autologic.models import PipelineResult

    pr = PipelineResult(
        node_results=[
            NodeResult(node_name="a", status=NodeStatus.SUCCESS),
            NodeResult(node_name="b", status=NodeStatus.FAILED, error="boom"),
        ],
        total_duration=3.14,
        status=PipelineStatus.PARTIAL,
    )

    assert "1/2" in pr.summary
    assert "partial" in pr.summary.lower()
    assert pr.failed_nodes == ["b"]
