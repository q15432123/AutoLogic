"""
autologic.nodes.preprocess_node — Image and audio preprocessing node.

Runs *before* the ingest node to enhance sketches (CLAHE contrast +
Gaussian blur) and validate audio files. Wraps the preprocessing helpers
from the legacy ``multi_ingest`` module.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from ..models import NodeResult, NodeStatus, PipelineContext
from .base import LogicNode

if TYPE_CHECKING:
    pass


class PreprocessNode(LogicNode):
    """
    Pipeline node for image and audio preprocessing.

    **Reads from context:**
        - ``image_path`` (str | None): filesystem path to a sketch image.
        - ``audio_path`` (str | None): filesystem path to a voice note.

    **Writes to context:**
        - ``image_path``: overwritten with the path to the preprocessed image
          (or left unchanged if no image was provided).
        - ``audio_path``: overwritten with the validated audio path
          (or left unchanged if no audio was provided).
        - ``preprocess_applied`` (bool): whether any preprocessing was done.
    """

    def __init__(self, name: str = "preprocess", description: str = "") -> None:
        super().__init__(
            name=name,
            description=description or "Image CLAHE enhancement + audio validation",
        )

    async def validate(self, context: PipelineContext) -> bool:
        """Skip entirely when there is neither an image nor an audio file."""
        has_image = await context.get("image_path") is not None
        has_audio = await context.get("audio_path") is not None
        if not has_image and not has_audio:
            self._logger.info("No image or audio in context — skipping preprocessing")
            return False
        return True

    async def execute(self, context: PipelineContext) -> NodeResult:
        """
        Run preprocessing in a background thread (OpenCV is synchronous).
        """
        image_path: str | None = await context.get("image_path")
        audio_path: str | None = await context.get("audio_path")
        applied = False

        # --- Image preprocessing ---
        if image_path:
            preprocessed_path = await asyncio.to_thread(
                self._preprocess_image, image_path
            )
            await context.set("image_path", preprocessed_path)
            self._logger.info("Image preprocessed: %s -> %s", image_path, preprocessed_path)
            applied = True

        # --- Audio preprocessing (validation) ---
        if audio_path:
            validated_path = await asyncio.to_thread(
                self._preprocess_audio, audio_path
            )
            await context.set("audio_path", validated_path)
            self._logger.info("Audio validated: %s", validated_path)
            applied = True

        await context.set("preprocess_applied", applied)

        return NodeResult(
            node_name=self.name,
            status=NodeStatus.SUCCESS,
            output={
                "image_preprocessed": image_path is not None,
                "audio_validated": audio_path is not None,
            },
        )

    # ──────────────────────────────────────────────
    # Synchronous helpers (run via asyncio.to_thread)
    # ──────────────────────────────────────────────

    @staticmethod
    def _preprocess_image(image_path: str) -> str:
        """
        Enhance a sketch image with CLAHE and Gaussian blur.

        This replicates the ``_preprocess_image`` logic from the legacy
        ``multi_ingest`` module.
        """
        import tempfile
        from pathlib import Path

        import cv2

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image for preprocessing: {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

        suffix = Path(image_path).suffix or ".png"
        tmp = tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False, prefix="autologic_preproc_"
        )
        cv2.imwrite(tmp.name, blurred)
        tmp.close()

        return tmp.name

    @staticmethod
    def _preprocess_audio(audio_path: str) -> str:
        """
        Validate an audio file (existence + non-empty).

        This replicates the ``_preprocess_audio`` logic from the legacy
        ``multi_ingest`` module.
        """
        from pathlib import Path

        p = Path(audio_path)
        if not p.exists():
            raise FileNotFoundError(f"Audio file not found for preprocessing: {audio_path}")
        if p.stat().st_size == 0:
            raise ValueError(f"Audio file is empty: {audio_path}")

        return audio_path
