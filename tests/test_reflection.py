"""Tests for the Reflective Logic Loop — validator, reflection, reasoning."""

import asyncio

import pytest

from autologic.models import NodeResult, NodeStatus, PipelineContext
from autologic.nodes.base import LogicNode
from autologic.reflection import (
    ConfidenceScore,
    CritiqueResult,
    LogicValidator,
    ReflectiveExecutor,
)
from autologic.reasoning import ConcurrentReasoner
from autologic.nodes.verifier_node import VerifierNode


# ── Test helpers ──────────────────────────────────


class SuccessNode(LogicNode):
    async def execute(self, context: PipelineContext) -> NodeResult:
        return NodeResult(
            node_name=self.name,
            status=NodeStatus.SUCCESS,
            output={"result": "built a landing page with hero section"},
        )


class FailingNode(LogicNode):
    async def execute(self, context: PipelineContext) -> NodeResult:
        return NodeResult(
            node_name=self.name,
            status=NodeStatus.FAILED,
            error="Something went wrong",
            output={},
        )


class ImprovingNode(LogicNode):
    """Node that fails on first attempt, succeeds on second."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._call_count = 0

    async def execute(self, context: PipelineContext) -> NodeResult:
        self._call_count += 1
        if self._call_count == 1:
            return NodeResult(
                node_name=self.name,
                status=NodeStatus.FAILED,
                error="First attempt failed",
                output={},
            )
        return NodeResult(
            node_name=self.name,
            status=NodeStatus.SUCCESS,
            output={"result": "improved output", "landing": "page"},
        )


# ── ConfidenceScore ──────────────────────────────


def test_confidence_score_acceptable():
    score = ConfidenceScore(value=0.8, reasoning="good")
    assert score.is_acceptable
    assert "0.80" in repr(score)


def test_confidence_score_unacceptable():
    score = ConfidenceScore(value=0.3, reasoning="bad")
    assert not score.is_acceptable


# ── LogicValidator ───────────────────────────────


@pytest.mark.asyncio
async def test_validator_success_node():
    validator = LogicValidator(confidence_threshold=0.5)
    ctx = PipelineContext({"text_prompt": "build a landing page"})

    result = NodeResult(
        node_name="test",
        status=NodeStatus.SUCCESS,
        output={"landing": "page", "hero": "section", "build": True},
    )

    critique = await validator.check_consistency(
        original_goal="build a landing page",
        node_result=result,
        context=ctx,
    )

    assert critique.is_valid
    assert critique.confidence.value > 0.5
    assert critique.attempt == 1


@pytest.mark.asyncio
async def test_validator_failed_node():
    validator = LogicValidator(confidence_threshold=0.6)
    ctx = PipelineContext()

    result = NodeResult(
        node_name="test",
        status=NodeStatus.FAILED,
        error="crash",
        output={},
    )

    critique = await validator.check_consistency(
        original_goal="build something",
        node_result=result,
        context=ctx,
    )

    assert not critique.is_valid
    assert critique.confidence.value < 0.6
    assert len(critique.suggestions) > 0


@pytest.mark.asyncio
async def test_validator_empty_output():
    validator = LogicValidator(confidence_threshold=0.6)
    ctx = PipelineContext()

    result = NodeResult(
        node_name="test",
        status=NodeStatus.SUCCESS,
        output={},
    )

    critique = await validator.check_consistency(
        original_goal="build app",
        node_result=result,
        context=ctx,
    )

    assert "empty output" in critique.suggestions[0].lower() or critique.confidence.value < 1.0


@pytest.mark.asyncio
async def test_validator_with_model_checker():
    async def mock_model(prompt: str) -> str:
        return "0.9"

    validator = LogicValidator(confidence_threshold=0.5, model_checker=mock_model)
    ctx = PipelineContext()

    result = NodeResult(
        node_name="test",
        status=NodeStatus.SUCCESS,
        output={"data": "present"},
    )

    critique = await validator.check_consistency(
        original_goal="test goal",
        node_result=result,
        context=ctx,
    )

    assert critique.is_valid
    assert "model_validation" in critique.confidence.factors


# ── ReflectiveExecutor ───────────────────────────


@pytest.mark.asyncio
async def test_reflective_executor_passes_first_try():
    validator = LogicValidator(confidence_threshold=0.5)
    executor = ReflectiveExecutor(validator=validator, max_retries=3)

    node = SuccessNode("test_node")
    ctx = PipelineContext({"text_prompt": "build a landing page"})

    result, attempts = await executor.execute_with_reflection(
        node, ctx, "build a landing page"
    )

    assert result.status == NodeStatus.SUCCESS
    assert len(attempts) == 1
    assert attempts[0].critique.is_valid


@pytest.mark.asyncio
async def test_reflective_executor_retries_on_failure():
    validator = LogicValidator(confidence_threshold=0.5)
    executor = ReflectiveExecutor(validator=validator, max_retries=3)

    node = ImprovingNode("improving_node")
    ctx = PipelineContext({"text_prompt": "build a landing page"})

    result, attempts = await executor.execute_with_reflection(
        node, ctx, "build a landing page"
    )

    assert len(attempts) == 2
    assert not attempts[0].critique.is_valid
    assert attempts[1].critique.is_valid
    assert result.status == NodeStatus.SUCCESS


@pytest.mark.asyncio
async def test_reflective_executor_exhausts_retries():
    validator = LogicValidator(confidence_threshold=0.99)  # Very high threshold
    executor = ReflectiveExecutor(validator=validator, max_retries=2)

    node = SuccessNode("test")
    ctx = PipelineContext({"text_prompt": "test"})

    result, attempts = await executor.execute_with_reflection(node, ctx, "test")

    assert len(attempts) == 2  # Exhausted all retries


@pytest.mark.asyncio
async def test_reflective_executor_callback():
    critiques_received = []

    def on_critique(node_name, attempt, critique):
        critiques_received.append((node_name, attempt))

    validator = LogicValidator(confidence_threshold=0.99)
    executor = ReflectiveExecutor(
        validator=validator, max_retries=2, on_critique=on_critique
    )

    node = SuccessNode("cb_test")
    ctx = PipelineContext({"text_prompt": "test"})

    await executor.execute_with_reflection(node, ctx, "test")

    assert len(critiques_received) >= 1
    assert critiques_received[0][0] == "cb_test"


# ── ConcurrentReasoner ───────────────────────────


@pytest.mark.asyncio
async def test_concurrent_reasoner_picks_best():
    validator = LogicValidator(confidence_threshold=0.3)
    reasoner = ConcurrentReasoner(
        validator=validator, num_branches=3, timeout_seconds=10.0
    )

    node = SuccessNode("reasoning_test")
    ctx = PipelineContext({"text_prompt": "build app"})

    result, branches = await reasoner.reason(node, ctx, "build app")

    assert result.status == NodeStatus.SUCCESS
    assert len(branches) == 3
    # All branches should have confidence scores
    for b in branches:
        assert b.confidence.value >= 0.0


@pytest.mark.asyncio
async def test_concurrent_reasoner_handles_failures():
    validator = LogicValidator(confidence_threshold=0.3)
    reasoner = ConcurrentReasoner(
        validator=validator, num_branches=2, timeout_seconds=10.0
    )

    node = FailingNode("fail_test")
    ctx = PipelineContext({"text_prompt": "test"})

    result, branches = await reasoner.reason(node, ctx, "test")

    # Should still return results (even if all branches failed)
    assert len(branches) == 2


# ── VerifierNode ─────────────────────────────────


@pytest.mark.asyncio
async def test_verifier_node_passes():
    ctx = PipelineContext({
        "text_prompt": "build a landing page",
        "_node_result_planning": {"plan": "build landing", "steps": 3},
    })

    verifier = VerifierNode("verify", target_node="planning")
    result = await verifier.run(ctx)

    assert result.status == NodeStatus.SUCCESS
    assert "confidence" in result.output


@pytest.mark.asyncio
async def test_verifier_node_skips_when_no_target():
    ctx = PipelineContext({"text_prompt": "build app"})

    verifier = VerifierNode("verify", target_node="nonexistent")
    result = await verifier.run(ctx)

    assert result.status == NodeStatus.SKIPPED


@pytest.mark.asyncio
async def test_verifier_node_stores_result_in_context():
    ctx = PipelineContext({
        "text_prompt": "test",
        "_node_result_codegen": {"files": ["index.html"]},
    })

    verifier = VerifierNode("verify", target_node="codegen")
    await verifier.run(ctx)

    verification = await ctx.get("_verification_codegen")
    assert verification is not None
    assert "is_valid" in verification


# ── CritiqueResult ───────────────────────────────


def test_critique_result_to_dict():
    cr = CritiqueResult(
        is_valid=True,
        confidence=ConfidenceScore(value=0.85),
        critique="",
        suggestions=[],
        attempt=1,
    )
    d = cr.to_dict()
    assert d["is_valid"] is True
    assert d["confidence"] == 0.85
    assert d["attempt"] == 1
