"""
Tests for autologic.nodes — Individual node behavior.
"""

from __future__ import annotations

import pytest

from autologic.models import NodeResult, NodeStatus, PipelineContext
from autologic.nodes.base import LogicNode
from autologic.nodes.ingest_node import IngestNode
from autologic.nodes.preprocess_node import PreprocessNode


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

class _MockIngestNode(IngestNode):
    """
    IngestNode subclass that overrides the legacy import with a pure-Python
    fake so that tests don't require OpenCV or Whisper.
    """

    @staticmethod
    def _get_ingest_function():
        """Return a mock ingest function that returns canned data."""

        def _mock_ingest(
            image_path=None,
            audio_path=None,
            text_prompt=None,
        ):
            parts = []
            result = {
                "raw_text_prompt": "",
                "transcription": "",
                "sketch_analysis_placeholder": "",
                "image_base64": None,
                "image_path": None,
                "consolidated_context": "",
            }

            if text_prompt:
                result["raw_text_prompt"] = text_prompt.strip()
                parts.append(f"## User Requirements (Text)\n{text_prompt.strip()}")

            if audio_path:
                result["transcription"] = "Mock transcription from audio"
                parts.append(
                    "## User Requirements (Voice Transcription)\n"
                    "Mock transcription from audio"
                )

            if image_path:
                result["image_path"] = image_path
                result["image_base64"] = "bW9ja19iYXNlNjQ="  # "mock_base64"
                parts.append(
                    "## Sketch Input\n[Sketch attached]"
                )

            result["consolidated_context"] = "\n\n".join(parts)
            return result

        return _mock_ingest


class _MockPreprocessNode(PreprocessNode):
    """
    PreprocessNode subclass that stubs out OpenCV calls.
    """

    @staticmethod
    def _preprocess_image(image_path: str) -> str:
        return f"{image_path}.preprocessed"

    @staticmethod
    def _preprocess_audio(audio_path: str) -> str:
        return audio_path


# ──────────────────────────────────────────────
# IngestNode tests
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ingest_node_text_only():
    """IngestNode with only a text prompt should produce consolidated context."""
    node = _MockIngestNode("ingest")
    context = PipelineContext({"text_prompt": "Build a todo app"})

    result = await node.run(context)

    assert result.status == NodeStatus.SUCCESS
    assert result.node_name == "ingest"

    consolidated = await context.get("consolidated_context")
    assert "Build a todo app" in consolidated
    assert result.output["has_text"] is True
    assert result.output["has_image"] is False
    assert result.output["has_transcription"] is False


@pytest.mark.asyncio
async def test_ingest_node_all_inputs():
    """IngestNode with all three input types should consolidate them all."""
    node = _MockIngestNode("ingest")
    context = PipelineContext({
        "text_prompt": "Build a chat app",
        "image_path": "/fake/sketch.png",
        "audio_path": "/fake/notes.mp3",
    })

    result = await node.run(context)

    assert result.status == NodeStatus.SUCCESS
    assert result.output["has_text"] is True
    assert result.output["has_image"] is True
    assert result.output["has_transcription"] is True

    consolidated = await context.get("consolidated_context")
    assert "Build a chat app" in consolidated
    assert "Sketch" in consolidated
    assert "Transcription" in consolidated


@pytest.mark.asyncio
async def test_ingest_node_no_inputs_skips():
    """IngestNode should skip (not crash) when no inputs are provided."""
    node = _MockIngestNode("ingest")
    context = PipelineContext()

    result = await node.run(context)

    assert result.status == NodeStatus.SKIPPED


@pytest.mark.asyncio
async def test_ingest_node_writes_image_base64():
    """IngestNode should write image_base64 to context when an image is given."""
    node = _MockIngestNode("ingest")
    context = PipelineContext({
        "text_prompt": "Portfolio site",
        "image_path": "/fake/wireframe.png",
    })

    await node.run(context)

    image_b64 = await context.get("image_base64")
    assert image_b64 is not None
    assert isinstance(image_b64, str)


# ──────────────────────────────────────────────
# PreprocessNode tests
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_preprocess_node_skip_no_image():
    """PreprocessNode should skip when there is no image or audio in context."""
    node = _MockPreprocessNode("preprocess")
    context = PipelineContext({"text_prompt": "Just text"})

    result = await node.run(context)

    assert result.status == NodeStatus.SKIPPED


@pytest.mark.asyncio
async def test_preprocess_node_image_only():
    """PreprocessNode should preprocess the image and update context."""
    node = _MockPreprocessNode("preprocess")
    context = PipelineContext({"image_path": "/fake/sketch.png"})

    result = await node.run(context)

    assert result.status == NodeStatus.SUCCESS
    assert result.output["image_preprocessed"] is True

    new_path = await context.get("image_path")
    assert new_path == "/fake/sketch.png.preprocessed"


@pytest.mark.asyncio
async def test_preprocess_node_audio_only():
    """PreprocessNode should validate audio and update context."""
    node = _MockPreprocessNode("preprocess")
    context = PipelineContext({"audio_path": "/fake/voice.mp3"})

    result = await node.run(context)

    assert result.status == NodeStatus.SUCCESS
    assert result.output["audio_validated"] is True


@pytest.mark.asyncio
async def test_preprocess_node_both():
    """PreprocessNode should handle both image and audio."""
    node = _MockPreprocessNode("preprocess")
    context = PipelineContext({
        "image_path": "/fake/sketch.png",
        "audio_path": "/fake/voice.mp3",
    })

    result = await node.run(context)

    assert result.status == NodeStatus.SUCCESS
    assert await context.get("preprocess_applied") is True


# ──────────────────────────────────────────────
# LogicNode base class tests
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_base_node_run_timing():
    """The run() wrapper should record a non-negative duration."""

    class TimedNode(LogicNode):
        async def execute(self, context: PipelineContext) -> NodeResult:
            return NodeResult(
                node_name=self.name,
                status=NodeStatus.SUCCESS,
            )

    node = TimedNode("timed")
    context = PipelineContext()
    result = await node.run(context)

    assert result.duration_seconds >= 0


@pytest.mark.asyncio
async def test_base_node_error_handler_called():
    """on_error should be called when execute raises."""

    error_seen = []

    class ErrorHandledNode(LogicNode):
        async def execute(self, context: PipelineContext) -> NodeResult:
            raise ValueError("test error")

        async def on_error(self, error: Exception, context: PipelineContext) -> None:
            error_seen.append(str(error))

    node = ErrorHandledNode("handled")
    context = PipelineContext()

    from autologic.exceptions import NodeExecutionError

    with pytest.raises(NodeExecutionError):
        await node.run(context)

    assert len(error_seen) == 1
    assert "test error" in error_seen[0]
