"""
autologic.nodes.codegen_node — Code generation node.

Wraps the legacy ``core_gen.execute_all_tasks()`` function as an async
:class:`LogicNode`. Takes the task plan from the planning node and
orchestrates Gemini-powered agents to generate actual files.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from ..models import NodeResult, NodeStatus, PipelineContext
from .base import LogicNode


class CodeGenNode(LogicNode):
    """
    Pipeline node that executes agent tasks to generate code files.

    **Reads from context:**
        - ``task_plan`` (list[dict]): task list produced by the planning node.
        - ``gemini_api_key`` (str | None): API key override.

    **Writes to context:**
        - ``generated_files`` (list[dict]): per-agent execution results with
          file lists, commands, and notes.
    """

    def __init__(self, name: str = "codegen", description: str = "") -> None:
        super().__init__(
            name=name,
            description=description or "Multi-agent code generation via Gemini",
        )

    async def validate(self, context: PipelineContext) -> bool:
        """A non-empty task plan must exist."""
        task_plan = await context.get("task_plan")
        if not task_plan:
            self._logger.warning("No task_plan in context — nothing to generate")
            return False

        api_key = await context.get("gemini_api_key") or os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            self._logger.warning("No Gemini API key available — cannot generate code")
            return False

        return True

    async def execute(self, context: PipelineContext) -> NodeResult:
        """
        Delegate to the legacy ``execute_all_tasks`` function in a thread.
        """
        task_plan: list[dict[str, Any]] = await context.get("task_plan", [])
        api_key: str = (
            await context.get("gemini_api_key")
            or os.getenv("GEMINI_API_KEY", "")
        )

        execute_fn = self._get_codegen_function()

        results: list[dict[str, Any]] = await asyncio.to_thread(
            execute_fn, task_plan, api_key
        )

        await context.set("generated_files", results)

        # Summarize
        total_files = sum(len(r.get("files", [])) for r in results)
        agents_involved = list({r.get("agent", "?") for r in results})

        self._logger.info(
            "Code generation complete — %d agents produced %d total files",
            len(results), total_files,
        )
        for r in results:
            files = r.get("files", [])
            agent = r.get("agent", "?")
            for f in files:
                self._logger.info("  -> %s wrote: %s", agent, f)

        return NodeResult(
            node_name=self.name,
            status=NodeStatus.SUCCESS,
            output={
                "agents_executed": len(results),
                "total_files_generated": total_files,
                "agents": agents_involved,
            },
        )

    @staticmethod
    def _get_codegen_function():
        """Dynamically import the legacy ``core_gen.execute_all_tasks``."""
        project_root = str(Path(__file__).resolve().parents[2])
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from core_gen import execute_all_tasks  # type: ignore[import-untyped]
        return execute_all_tasks
