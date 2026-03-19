"""
autologic.nodes.deploy_node — Deployment node.

Wraps the legacy ``auto_deploy.deploy_to_firebase()`` function as an async
:class:`LogicNode`. Packages the generated workspace and deploys to
Firebase Hosting.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from ..models import NodeResult, NodeStatus, PipelineContext
from .base import LogicNode


class DeployNode(LogicNode):
    """
    Pipeline node that packages and deploys generated code.

    **Reads from context:**
        - ``workspace_dir`` (str): directory containing generated files.
        - ``firebase_project_id`` (str | None): Firebase project ID override.
          Falls back to the ``FIREBASE_PROJECT_ID`` environment variable.

    **Writes to context:**
        - ``deploy_result`` (dict): deployment status, hosting URL, and logs.
    """

    def __init__(self, name: str = "deploy", description: str = "") -> None:
        super().__init__(
            name=name,
            description=description or "Package and deploy to Firebase Hosting",
        )

    async def validate(self, context: PipelineContext) -> bool:
        """
        Skip deployment when no Firebase project ID is configured.
        """
        project_id = (
            await context.get("firebase_project_id")
            or os.getenv("FIREBASE_PROJECT_ID", "")
        )
        if not project_id:
            self._logger.info(
                "No Firebase project ID configured — skipping deployment"
            )
            return False

        workspace_dir = await context.get("workspace_dir", "_workspaces")
        ws_path = Path(workspace_dir)
        if not ws_path.exists():
            self._logger.warning("Workspace directory does not exist: %s", ws_path)
            return False

        return True

    async def execute(self, context: PipelineContext) -> NodeResult:
        """
        Delegate to the legacy ``deploy_to_firebase`` function in a thread.
        """
        workspace_dir: str = await context.get("workspace_dir", "_workspaces")
        project_id: str = (
            await context.get("firebase_project_id")
            or os.getenv("FIREBASE_PROJECT_ID", "")
        )

        deploy_fn, report_fn = self._get_deploy_functions()

        deploy_result: dict[str, Any] = await asyncio.to_thread(
            deploy_fn, workspace_dir, project_id
        )

        # Generate the markdown report as well
        await asyncio.to_thread(report_fn, deploy_result, workspace_dir)

        await context.set("deploy_result", deploy_result)

        status = deploy_result.get("deployment_status", "unknown")
        hosting_url = deploy_result.get("hosting_url", "")

        if status == "success":
            self._logger.info("Deployment successful: %s", hosting_url)
        else:
            self._logger.warning(
                "Deployment finished with status '%s': %s",
                status, deploy_result.get("logs", "no logs"),
            )

        return NodeResult(
            node_name=self.name,
            status=NodeStatus.SUCCESS if status == "success" else NodeStatus.FAILED,
            output={
                "deployment_status": status,
                "hosting_url": hosting_url,
            },
            error=None if status == "success" else deploy_result.get("logs"),
        )

    @staticmethod
    def _get_deploy_functions():
        """Dynamically import legacy deployment functions."""
        project_root = str(Path(__file__).resolve().parents[2])
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from auto_deploy import deploy_to_firebase, generate_deploy_report  # type: ignore[import-untyped]
        return deploy_to_firebase, generate_deploy_report
