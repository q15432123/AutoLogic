"""
autologic.reflection — Reflective Logic Loop (Self-Critique & Dynamic Correction).

Implements the core "think → execute → verify → correct" cycle. When a node
produces output, the :class:`ReflectiveExecutor` asks a :class:`LogicValidator`
whether the output is logically consistent with the original goal. If not,
the node is re-executed with critique feedback injected into the context.

Architecture by Google Gemini | Implementation by Anthropic Claude Opus
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .exceptions import NodeExecutionError
from .models import NodeResult, NodeStatus, PipelineContext
from .nodes.base import LogicNode


# ────────────────────────────────────────────────
# Confidence Scoring
# ────────────────────────────────────────────────

@dataclass
class ConfidenceScore:
    """Weighted confidence assessment for a node's output."""

    value: float  # 0.0 to 1.0
    reasoning: str = ""
    factors: dict[str, float] = field(default_factory=dict)

    @property
    def is_acceptable(self) -> bool:
        return self.value >= 0.6

    def __repr__(self) -> str:
        return f"ConfidenceScore({self.value:.2f}, acceptable={self.is_acceptable})"


@dataclass
class CritiqueResult:
    """Result of a validation/critique pass on a node's output."""

    is_valid: bool
    confidence: ConfidenceScore
    critique: str = ""
    suggestions: list[str] = field(default_factory=list)
    attempt: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "confidence": self.confidence.value,
            "critique": self.critique,
            "suggestions": self.suggestions,
            "attempt": self.attempt,
        }


# ────────────────────────────────────────────────
# Logic Validator
# ────────────────────────────────────────────────

class LogicValidator:
    """
    Validates node outputs against the original goal using structured checks.

    The validator can operate in two modes:
    - **rule-based** (default): Uses heuristic checks (output non-empty,
      required keys present, no error flags).
    - **model-based**: Optionally delegates to an LLM for semantic validation
      via a pluggable ``model_checker`` callable.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        model_checker: Optional[Callable] = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.model_checker = model_checker
        self._logger = logging.getLogger("autologic.validator")

    async def check_consistency(
        self,
        original_goal: str,
        node_result: NodeResult,
        context: PipelineContext,
        attempt: int = 1,
    ) -> CritiqueResult:
        """
        Check whether a node's output is consistent with the original goal.

        Args:
            original_goal: The user's original requirement text.
            node_result: The result to validate.
            context: Current pipeline context for additional state.
            attempt: Current attempt number (for logging).

        Returns:
            A CritiqueResult with validity, confidence, and critique text.
        """
        factors: dict[str, float] = {}
        suggestions: list[str] = []

        # ── Factor 1: Execution status ──
        if node_result.status == NodeStatus.SUCCESS:
            factors["execution_status"] = 1.0
        elif node_result.status == NodeStatus.SKIPPED:
            factors["execution_status"] = 0.5
        else:
            factors["execution_status"] = 0.0
            suggestions.append("Node execution failed — check error logs")

        # ── Factor 2: Output completeness ──
        output = node_result.output or {}
        if output:
            factors["output_completeness"] = min(1.0, len(output) / 3)
        else:
            factors["output_completeness"] = 0.0
            suggestions.append("Node produced empty output")

        # ── Factor 3: No error flag ──
        if node_result.error:
            factors["error_free"] = 0.0
            suggestions.append(f"Error present: {node_result.error}")
        else:
            factors["error_free"] = 1.0

        # ── Factor 4: Goal alignment (keyword overlap) ──
        if original_goal and output:
            goal_words = set(original_goal.lower().split())
            output_text = str(output).lower()
            matches = sum(1 for w in goal_words if w in output_text)
            factors["goal_alignment"] = min(1.0, matches / max(len(goal_words), 1))
        else:
            factors["goal_alignment"] = 0.5  # neutral if no goal

        # ── Factor 5: Model-based check (optional) ──
        if self.model_checker and node_result.is_success:
            try:
                model_score = await self._run_model_check(
                    original_goal, node_result, context
                )
                factors["model_validation"] = model_score
            except Exception as exc:
                self._logger.warning("Model checker failed: %s", exc)
                factors["model_validation"] = 0.5  # neutral on failure

        # ── Compute weighted confidence ──
        if factors:
            weights = {
                "execution_status": 0.30,
                "output_completeness": 0.20,
                "error_free": 0.25,
                "goal_alignment": 0.15,
                "model_validation": 0.10,
            }
            total_weight = sum(weights.get(k, 0.1) for k in factors)
            weighted_sum = sum(
                factors[k] * weights.get(k, 0.1) for k in factors
            )
            confidence_value = weighted_sum / total_weight if total_weight > 0 else 0.0
        else:
            confidence_value = 0.0

        confidence = ConfidenceScore(
            value=round(confidence_value, 4),
            reasoning=self._build_reasoning(factors),
            factors=factors,
        )

        is_valid = confidence.value >= self.confidence_threshold

        critique = ""
        if not is_valid:
            critique = (
                f"Confidence {confidence.value:.2f} is below threshold "
                f"{self.confidence_threshold:.2f}. "
                f"Issues: {'; '.join(suggestions) if suggestions else 'Low overall score'}"
            )

        self._logger.info(
            "[Attempt %d] Node '%s': confidence=%.2f, valid=%s",
            attempt, node_result.node_name, confidence.value, is_valid,
        )

        return CritiqueResult(
            is_valid=is_valid,
            confidence=confidence,
            critique=critique,
            suggestions=suggestions,
            attempt=attempt,
        )

    async def _run_model_check(
        self,
        original_goal: str,
        node_result: NodeResult,
        context: PipelineContext,
    ) -> float:
        """Delegate to the pluggable model checker. Returns a 0-1 score."""
        if not self.model_checker:
            return 0.5

        prompt = (
            f"Goal: {original_goal}\n"
            f"Node: {node_result.node_name}\n"
            f"Output: {node_result.output}\n"
            f"Is this output logically consistent and progressing toward the goal? "
            f"Rate confidence 0.0 to 1.0."
        )
        # model_checker can be sync or async
        result = self.model_checker(prompt)
        if asyncio.iscoroutine(result):
            result = await result

        # Try to extract a float from the response
        try:
            return float(str(result).strip())
        except (ValueError, TypeError):
            return 0.5

    def _build_reasoning(self, factors: dict[str, float]) -> str:
        parts = [f"{k}={v:.2f}" for k, v in sorted(factors.items())]
        return " | ".join(parts)


# ────────────────────────────────────────────────
# Reflective Executor
# ────────────────────────────────────────────────

@dataclass
class ReflectionAttempt:
    """Record of a single execution attempt within the reflection loop."""

    attempt: int
    node_result: NodeResult
    critique: CritiqueResult
    duration_seconds: float = 0.0


class ReflectiveExecutor:
    """
    Wraps node execution with a self-critique reflection loop.

    When a node produces output that doesn't pass validation, the executor
    injects the critique feedback into the context and re-runs the node,
    up to ``max_retries`` times.

    Usage::

        executor = ReflectiveExecutor(validator=LogicValidator())
        result = await executor.execute_with_reflection(node, context, goal)
    """

    def __init__(
        self,
        validator: LogicValidator,
        max_retries: int = 3,
        on_critique: Optional[Callable] = None,
    ) -> None:
        self.validator = validator
        self.max_retries = max_retries
        self.on_critique = on_critique  # callback(node_name, attempt, critique)
        self._logger = logging.getLogger("autologic.reflection")

    async def execute_with_reflection(
        self,
        node: LogicNode,
        context: PipelineContext,
        original_goal: str = "",
    ) -> tuple[NodeResult, list[ReflectionAttempt]]:
        """
        Execute a node with the reflective logic loop.

        1. Run the node
        2. Validate the output
        3. If invalid and retries remain, inject critique and re-run
        4. Return the best result and full attempt history

        Args:
            node: The LogicNode to execute.
            context: The shared pipeline context.
            original_goal: The user's original goal for alignment checking.

        Returns:
            Tuple of (final NodeResult, list of all attempts).
        """
        if not original_goal:
            original_goal = await context.get("text_prompt", "")

        attempts: list[ReflectionAttempt] = []
        best_result: Optional[NodeResult] = None
        best_confidence: float = -1.0

        for attempt_num in range(1, self.max_retries + 1):
            start_time = time.perf_counter()

            self._logger.info(
                "Reflection attempt %d/%d for node '%s'",
                attempt_num, self.max_retries, node.name,
            )

            # ── Execute the node ──
            try:
                result = await node.run(context)
            except NodeExecutionError as exc:
                result = NodeResult(
                    node_name=node.name,
                    status=NodeStatus.FAILED,
                    error=str(exc),
                    duration_seconds=time.perf_counter() - start_time,
                )

            # ── Validate the output ──
            critique = await self.validator.check_consistency(
                original_goal=original_goal,
                node_result=result,
                context=context,
                attempt=attempt_num,
            )

            elapsed = time.perf_counter() - start_time
            attempt_record = ReflectionAttempt(
                attempt=attempt_num,
                node_result=result,
                critique=critique,
                duration_seconds=elapsed,
            )
            attempts.append(attempt_record)

            # Track best result by confidence
            if critique.confidence.value > best_confidence:
                best_confidence = critique.confidence.value
                best_result = result

            # ── Pass? ──
            if critique.is_valid:
                self._logger.info(
                    "Node '%s' passed validation on attempt %d (confidence=%.2f)",
                    node.name, attempt_num, critique.confidence.value,
                )
                break

            # ── Fail: inject critique and retry ──
            self._logger.warning(
                "Node '%s' failed validation on attempt %d: %s",
                node.name, attempt_num, critique.critique,
            )

            if self.on_critique:
                try:
                    callback_result = self.on_critique(
                        node.name, attempt_num, critique
                    )
                    if asyncio.iscoroutine(callback_result):
                        await callback_result
                except Exception:
                    pass

            # Inject critique feedback into context for the next attempt
            if attempt_num < self.max_retries:
                prev_critique = await context.get("_reflection_critique", "")
                feedback = (
                    f"{prev_critique}\n"
                    f"[Attempt {attempt_num} failed] {critique.critique}\n"
                    f"Suggestions: {'; '.join(critique.suggestions)}"
                ).strip()
                await context.set("_reflection_critique", feedback)
                await context.set("_reflection_attempt", attempt_num + 1)

                self._logger.info(
                    "Retrying node '%s' with critique feedback injected",
                    node.name,
                )

        # Clean up reflection state
        await context.remove("_reflection_critique")
        await context.remove("_reflection_attempt")

        # Store reflection history in context
        history_key = f"_reflection_history_{node.name}"
        await context.set(history_key, [a.critique.to_dict() for a in attempts])

        final_result = best_result or result
        return final_result, attempts
