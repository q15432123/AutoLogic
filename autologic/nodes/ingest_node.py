"""
autologic.nodes.ingest_node — Multimodal ingestion node.

Wraps the legacy ``multi_ingest.ingest_requirements()`` function as an
async :class:`LogicNode`. Reads image, audio, and text inputs from the
pipeline context and writes the consolidated context back.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from ..models import NodeResult, NodeStatus, PipelineContext
from .base import LogicNode


class IngestNode(LogicNode):
    """
    Pipeline node that processes multimodal inputs (text, image, audio).

    **Reads from context:**
        - ``image_path`` (str | None): filesystem path to a sketch image.
        - ``audio_path`` (str | None): filesystem path to a voice note.
        - ``text_prompt`` (str | None): raw text description.

    **Writes to context:**
        - ``consolidated_context`` (str): merged Markdown-style context string.
        - ``image_base64`` (str | None): base64-encoded image data.
        - ``transcription`` (str): audio transcription (empty string if no audio).
        - ``raw_text_prompt`` (str): cleaned text prompt.
        - ``ingest_result`` (dict): full result dict from the ingest function.
    """

    def __init__(self, name: str = "ingest", description: str = "") -> None:
        super().__init__(
            name=name,
            description=description or "Multimodal input ingestion (text, image, audio)",
        )

    async def validate(self, context: PipelineContext) -> bool:
        """At least one input source must be present."""
        has_text = bool(await context.get("text_prompt"))
        has_image = bool(await context.get("image_path"))
        has_audio = bool(await context.get("audio_path"))
        if not any([has_text, has_image, has_audio]):
            self._logger.warning("No text, image, or audio in context — cannot ingest")
            return False
        return True

    async def execute(self, context: PipelineContext) -> NodeResult:
        """
        Delegate to the legacy ``ingest_requirements`` function in a thread.
        """
        image_path: str | None = await context.get("image_path")
        audio_path: str | None = await context.get("audio_path")
        text_prompt: str | None = await context.get("text_prompt")

        ingest_fn = self._get_ingest_function()

        result: dict[str, Any] = await asyncio.to_thread(
            ingest_fn,
            image_path=image_path,
            audio_path=audio_path,
            text_prompt=text_prompt,
        )

        # Propagate outputs into the shared context
        await context.set("consolidated_context", result.get("consolidated_context", ""))
        await context.set("image_base64", result.get("image_base64"))
        await context.set("transcription", result.get("transcription", ""))
        await context.set("raw_text_prompt", result.get("raw_text_prompt", ""))
        await context.set("ingest_result", result)

        ctx_len = len(result.get("consolidated_context", ""))
        self._logger.info("Ingestion complete — consolidated context is %d chars", ctx_len)

        return NodeResult(
            node_name=self.name,
            status=NodeStatus.SUCCESS,
            output={
                "consolidated_context_length": ctx_len,
                "has_image": result.get("image_base64") is not None,
                "has_transcription": bool(result.get("transcription")),
                "has_text": bool(result.get("raw_text_prompt")),
            },
        )

    @staticmethod
    def _get_ingest_function():
        """
        Dynamically import the legacy ``multi_ingest.ingest_requirements``.

        This allows the node to work even if the legacy module sits at the
        project root (outside the autologic package). The project root is
        added to ``sys.path`` if it is not already present.
        """
        # Ensure the project root is importable
        project_root = str(Path(__file__).resolve().parents[2])
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from multi_ingest import ingest_requirements  # type: ignore[import-untyped]
        return ingest_requirements
