"""
autologic.reasoning — Concurrent Reasoning Engine.

For critical logic points, spawns multiple parallel reasoning branches
and selects the best result based on confidence scoring.

Architecture by Google Gemini | Implementation by Anthropic Claude Opus
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .models import NodeResult, NodeStatus, PipelineContext
from .nodes.base import LogicNode
from .reflection import ConfidenceScore, LogicValidator


@dataclass
class ReasoningBranch:
    """A single reasoning branch and its outcome."""

    branch_id: int
    node_name: str
    result: NodeResult
    confidence: ConfidenceScore
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "node_name": self.node_name,
            "status": self.result.status.value,
            "confidence": self.confidence.value,
            "duration": round(self.duration_seconds, 4),
        }


class ConcurrentReasoner:
    """
    Runs multiple reasoning branches in parallel and picks the best.

    For critical pipeline nodes, instead of running once and hoping for the
    best, the reasoner creates N independent execution branches (each with
    a slightly different context variation), runs them concurrently, and
    selects the branch with the highest confidence score.

    Usage::

        reasoner = ConcurrentReasoner(validator=LogicValidator(), num_branches=3)
        best_result, all_branches = await reasoner.reason(node, context, goal)
    """

    def __init__(
        self,
        validator: LogicValidator,
        num_branches: int = 3,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.validator = validator
        self.num_branches = max(2, num_branches)
        self.timeout_seconds = timeout_seconds
        self._logger = logging.getLogger("autologic.reasoning")

    async def _run_branch(
        self,
        branch_id: int,
        node: LogicNode,
        context: PipelineContext,
        original_goal: str,
    ) -> ReasoningBranch:
        """Execute a single reasoning branch."""
        start = time.perf_counter()

        # Create an isolated context copy for this branch
        branch_context = PipelineContext(await context.to_dict())
        await branch_context.set("_reasoning_branch_id", branch_id)
        await branch_context.set(
            "_reasoning_variation",
            f"Branch {branch_id}: Explore alternative approach #{branch_id}",
        )

        try:
            result = await node.run(branch_context)
        except Exception as exc:
            self._logger.warning(
                "Branch %d for '%s' failed: %s", branch_id, node.name, exc
            )
            result = NodeResult(
                node_name=node.name,
                status=NodeStatus.FAILED,
                error=str(exc),
                duration_seconds=time.perf_counter() - start,
            )

        # Score the branch
        critique = await self.validator.check_consistency(
            original_goal=original_goal,
            node_result=result,
            context=branch_context,
            attempt=branch_id,
        )

        elapsed = time.perf_counter() - start

        return ReasoningBranch(
            branch_id=branch_id,
            node_name=node.name,
            result=result,
            confidence=critique.confidence,
            duration_seconds=elapsed,
        )

    async def reason(
        self,
        node: LogicNode,
        context: PipelineContext,
        original_goal: str = "",
    ) -> tuple[NodeResult, list[ReasoningBranch]]:
        """
        Run multiple reasoning branches concurrently and select the best.

        Args:
            node: The LogicNode to execute across branches.
            context: The shared pipeline context.
            original_goal: The user's goal for alignment scoring.

        Returns:
            Tuple of (best NodeResult, all ReasoningBranch records).
        """
        if not original_goal:
            original_goal = await context.get("text_prompt", "")

        self._logger.info(
            "Starting %d concurrent reasoning branches for '%s'",
            self.num_branches, node.name,
        )

        # Launch all branches concurrently
        tasks = [
            asyncio.create_task(
                self._run_branch(i, node, context, original_goal)
            )
            for i in range(1, self.num_branches + 1)
        ]

        # Wait with timeout
        done, pending = await asyncio.wait(
            tasks, timeout=self.timeout_seconds
        )

        # Cancel any stragglers
        for task in pending:
            task.cancel()

        # Collect results
        branches: list[ReasoningBranch] = []
        for task in done:
            try:
                branches.append(task.result())
            except Exception as exc:
                self._logger.warning("Branch task failed: %s", exc)

        if not branches:
            self._logger.error("All reasoning branches failed for '%s'", node.name)
            return (
                NodeResult(
                    node_name=node.name,
                    status=NodeStatus.FAILED,
                    error="All concurrent reasoning branches failed",
                ),
                [],
            )

        # Sort by confidence (highest first)
        branches.sort(key=lambda b: b.confidence.value, reverse=True)

        best = branches[0]
        self._logger.info(
            "Best branch for '%s': branch %d (confidence=%.2f)",
            node.name, best.branch_id, best.confidence.value,
        )

        # Log all branch results
        for b in branches:
            self._logger.debug(
                "  Branch %d: confidence=%.2f, status=%s, time=%.2fs",
                b.branch_id, b.confidence.value, b.result.status.value,
                b.duration_seconds,
            )

        # Store reasoning history in the main context
        await context.set(
            f"_reasoning_branches_{node.name}",
            [b.to_dict() for b in branches],
        )

        return best.result, branches
