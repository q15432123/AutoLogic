"""
autologic.nodes.verifier_node — Post-execution verification node.

Can be inserted after any node in the pipeline to validate that node's
output is consistent with the original goal. If validation fails, it
sets context flags that downstream nodes can use to adjust behavior.

Architecture by Google Gemini | Implementation by Anthropic Claude Opus
"""

from __future__ import annotations

import logging
from typing import Optional

from ..models import NodeResult, NodeStatus, PipelineContext
from ..reflection import CritiqueResult, LogicValidator
from .base import LogicNode


class VerifierNode(LogicNode):
    """
    A pipeline node that validates the output of a preceding node.

    Insert after any critical node to add self-critique::

        engine.add_node(PlanningNode("planning"))
        engine.add_node(VerifierNode("verify_plan", target_node="planning"))
        engine.add_node(CodeGenNode("codegen"))

    If validation fails, the node writes a warning to context but does NOT
    block the pipeline (downstream nodes can check ``_verification_passed``).
    """

    def __init__(
        self,
        name: str,
        target_node: str = "",
        confidence_threshold: float = 0.6,
        validator: Optional[LogicValidator] = None,
        description: str = "",
    ) -> None:
        super().__init__(
            name=name,
            description=description or f"Verify output of '{target_node}'",
        )
        self.target_node = target_node
        self.confidence_threshold = confidence_threshold
        self.validator = validator or LogicValidator(
            confidence_threshold=confidence_threshold
        )

    async def validate(self, context: PipelineContext) -> bool:
        """Skip if no target node output to verify."""
        if self.target_node:
            result_key = f"_node_result_{self.target_node}"
            return await context.has(result_key)
        return True

    async def execute(self, context: PipelineContext) -> NodeResult:
        """Run validation against the target node's output."""
        original_goal = await context.get("text_prompt", "")

        # Get the target node's result from context
        result_key = f"_node_result_{self.target_node}"
        target_result_data = await context.get(result_key, {})

        # Reconstruct a NodeResult for validation
        target_result = NodeResult(
            node_name=self.target_node,
            status=NodeStatus.SUCCESS if target_result_data else NodeStatus.FAILED,
            output=target_result_data if isinstance(target_result_data, dict) else {"data": target_result_data},
        )

        # Run the validator
        critique = await self.validator.check_consistency(
            original_goal=original_goal,
            node_result=target_result,
            context=context,
        )

        # Store verification result in context
        await context.set(f"_verification_{self.target_node}", critique.to_dict())
        await context.set(
            f"_verification_passed_{self.target_node}",
            critique.is_valid,
        )

        if critique.is_valid:
            self._logger.info(
                "Verification PASSED for '%s' (confidence=%.2f)",
                self.target_node, critique.confidence.value,
            )
            return NodeResult(
                node_name=self.name,
                status=NodeStatus.SUCCESS,
                output={
                    "target_node": self.target_node,
                    "confidence": critique.confidence.value,
                    "is_valid": True,
                    "reasoning": critique.confidence.reasoning,
                },
            )
        else:
            self._logger.warning(
                "Verification FAILED for '%s' (confidence=%.2f): %s",
                self.target_node, critique.confidence.value, critique.critique,
            )
            return NodeResult(
                node_name=self.name,
                status=NodeStatus.SUCCESS,  # Verifier itself succeeded
                output={
                    "target_node": self.target_node,
                    "confidence": critique.confidence.value,
                    "is_valid": False,
                    "critique": critique.critique,
                    "suggestions": critique.suggestions,
                    "reasoning": critique.confidence.reasoning,
                },
            )
