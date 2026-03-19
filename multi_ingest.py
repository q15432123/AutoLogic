"""
Module 1: multi_ingest — Multimodal Ingestion Engine

Accepts image (sketch), audio (voice note), and text inputs.
Processes them locally and returns a consolidated context object.
"""

import base64
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import whisper

# Lazy-load whisper model (downloaded once, cached)
_whisper_model = None

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SUPPORTED_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm"}


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        print("[multi_ingest] Loading Whisper 'base' model (first run downloads ~140MB)...")
        _whisper_model = whisper.load_model("base")
    return _whisper_model


def _validate_image(image_path: str) -> str:
    """Validate image file exists and is readable by OpenCV. Returns abs path."""
    p = Path(image_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    if p.suffix.lower() not in SUPPORTED_IMAGE_EXTS:
        raise ValueError(f"Unsupported image format: {p.suffix} (supported: {SUPPORTED_IMAGE_EXTS})")
    img = cv2.imread(str(p))
    if img is None:
        raise ValueError(f"OpenCV failed to read image: {p}")
    h, w = img.shape[:2]
    print(f"[multi_ingest] Image validated: {p.name} ({w}x{h})")
    return str(p)


def _validate_audio(audio_path: str) -> str:
    """Validate audio file exists. Returns abs path."""
    p = Path(audio_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Audio not found: {p}")
    if p.suffix.lower() not in SUPPORTED_AUDIO_EXTS:
        raise ValueError(f"Unsupported audio format: {p.suffix} (supported: {SUPPORTED_AUDIO_EXTS})")
    print(f"[multi_ingest] Audio validated: {p.name} ({p.stat().st_size / 1024:.1f} KB)")
    return str(p)


def _transcribe_audio(audio_path: str) -> str:
    """Transcribe audio file to text using Whisper."""
    model = _get_whisper_model()
    print(f"[multi_ingest] Transcribing audio...")
    result = model.transcribe(audio_path, fp16=False)
    text = result.get("text", "").strip()
    print(f"[multi_ingest] Transcription complete ({len(text)} chars)")
    return text


def _encode_image_base64(image_path: str) -> str:
    """Encode image to base64 for Gemini API multimodal input."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def ingest_requirements(
    image_path: Optional[str] = None,
    audio_path: Optional[str] = None,
    text_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Processes multimodal inputs and returns consolidated context.

    Args:
        image_path: Path to a sketch/wireframe image (JPEG, PNG, etc.)
        audio_path: Path to a voice note (MP3, WAV, etc.)
        text_prompt: Direct text description of requirements

    Returns:
        Consolidated context dict with all processed inputs.
    """
    if not any([image_path, audio_path, text_prompt]):
        raise ValueError("At least one input (image, audio, or text) is required.")

    result = {
        "raw_text_prompt": "",
        "transcription": "",
        "sketch_analysis_placeholder": "",
        "image_base64": None,
        "image_path": None,
        "consolidated_context": "",
    }

    # --- Process text ---
    if text_prompt:
        result["raw_text_prompt"] = text_prompt.strip()
        print(f"[multi_ingest] Text prompt received ({len(text_prompt)} chars)")

    # --- Process audio ---
    if audio_path:
        validated_path = _validate_audio(audio_path)
        result["transcription"] = _transcribe_audio(validated_path)

    # --- Process image ---
    if image_path:
        validated_path = _validate_image(image_path)
        result["image_path"] = validated_path
        result["image_base64"] = _encode_image_base64(validated_path)
        result["sketch_analysis_placeholder"] = (
            "[Sketch attached — Gemini will analyze this image in Module 2]"
        )

    # --- Build consolidated context ---
    parts = []
    if result["raw_text_prompt"]:
        parts.append(f"## User Requirements (Text)\n{result['raw_text_prompt']}")
    if result["transcription"]:
        parts.append(f"## User Requirements (Voice Transcription)\n{result['transcription']}")
    if result["sketch_analysis_placeholder"]:
        parts.append(f"## Sketch Input\n{result['sketch_analysis_placeholder']}")

    result["consolidated_context"] = "\n\n".join(parts)

    print(f"[multi_ingest] Context consolidated ({len(result['consolidated_context'])} chars)")
    print(f"  - Text:  {'Yes' if result['raw_text_prompt'] else 'No'}")
    print(f"  - Audio: {'Yes' if result['transcription'] else 'No'}")
    print(f"  - Image: {'Yes' if result['image_path'] else 'No'}")

    return result


# --- CLI test ---
if __name__ == "__main__":
    import sys

    ctx = ingest_requirements(
        image_path=sys.argv[1] if len(sys.argv) > 1 else None,
        audio_path=sys.argv[2] if len(sys.argv) > 2 else None,
        text_prompt="Build a landing page with a hero section, feature cards, and a contact form.",
    )
    print("\n=== Consolidated Context ===")
    print(ctx["consolidated_context"])
