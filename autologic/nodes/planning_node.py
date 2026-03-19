"""
autologic.nodes.planning_node — Gemini-powered planning node.

Wraps the legacy ``core_gen.orchestrate_planning()`` function as an
async :class:`LogicNode`. Sends the consolidated context to Google Gemini
and produces a structured multi-agent task list.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from ..models import NodeResult, NodeStatus, PipelineContext
from .base import LogicNode


class PlanningNode(LogicNode):
    """
    Pipeline node that uses Gemini to decompose requirements into tasks.

    **Reads from context:**
        - ``consolidated_context`` (str): merged requirements text.
        - ``image_path`` (str | None): optional sketch for visual analysis.
        - ``gemini_api_key`` (str | None): API key override. Falls back to
          the ``GEMINI_API_KEY`` environment variable.

    **Writes to context:**
        - ``task_plan`` (list[dict]): ordered list of agent task dicts.
    """

    def __init__(self, name: str = "planning", description: str = "") -> None:
        super().__init__(
            name=name,
            description=description or "Gemini-powered requirement decomposition and planning",
        )

    async def validate(self, context: PipelineContext) -> bool:
        """Consolidated context must be present and non-empty."""
        consolidated = await context.get("consolidated_context", "")
        if not consolidated:
            self._logger.warning("No consolidated_context — cannot plan")
            return False

        # Check for an API key
        api_key = await context.get("gemini_api_key") or os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            self._logger.warning("No Gemini API key available — cannot plan")
            return False

        return True

    async def execute(self, context: PipelineContext) -> NodeResult:
        """
        Delegate to the legacy ``orchestrate_planning`` function in a thread.
        """
        consolidated_context: str = await context.get("consolidated_context", "")
        image_path: str | None = await context.get("image_path")
        api_key: str = (
            await context.get("gemini_api_key")
            or os.getenv("GEMINI_API_KEY", "")
        )

        # Build the multimodal context dict expected by the legacy function
        multimodal_context: dict[str, Any] = {
            "consolidated_context": consolidated_context,
            "image_path": image_path,
        }

        orchestrate_fn = self._get_planning_function()

        tasks: list[dict[str, Any]] = await asyncio.to_thread(
            orchestrate_fn,
            multimodal_context=multimodal_context,
            api_key=api_key,
            image_path=image_path,
        )

        await context.set("task_plan", tasks)

        self._logger.info("Planning complete — %d tasks generated", len(tasks))
        for i, t in enumerate(tasks):
            deps = ", ".join(t.get("depends_on", [])) or "none"
            self._logger.info(
                "  [%d] %-8s | %s (deps: %s)", i + 1, t["agent"], t["task"], deps
            )

        return NodeResult(
            node_name=self.name,
            status=NodeStatus.SUCCESS,
            output={
                "task_count": len(tasks),
                "agents": list({t.get("agent", "?") for t in tasks}),
            },
        )

    @staticmethod
    def _get_planning_function():
        """Dynamically import the legacy ``core_gen.orchestrate_planning``."""
        project_root = str(Path(__file__).resolve().parents[2])
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from core_gen import orchestrate_planning  # type: ignore[import-untyped]
        return orchestrate_planning
