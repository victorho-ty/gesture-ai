"""Download and cache the pretrained MediaPipe hand landmarker task file."""

from __future__ import annotations

import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker"
    "/hand_landmarker/float16/latest/hand_landmarker.task"
)

_DOWNLOAD_TIMEOUT_SECONDS = 60
_CHUNK_SIZE = 1024 * 1024


def ensure_model(model_path: Path) -> Path:
    """Return ``model_path``, downloading the official model when missing."""
    model_path = Path(model_path)
    if model_path.is_file() and model_path.stat().st_size > 0:
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = model_path.with_name(model_path.name + ".download")

    try:
        request = urllib.request.Request(
            MODEL_URL, headers={"User-Agent": "gesture-ai/0.1.0"}
        )
        downloaded = 0
        with urllib.request.urlopen(
            request, timeout=_DOWNLOAD_TIMEOUT_SECONDS
        ) as response, temp_path.open("wb") as handle:
            while True:
                chunk = response.read(_CHUNK_SIZE)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)

        if downloaded == 0:
            raise OSError("downloaded model was empty")

        temp_path.replace(model_path)
    except OSError as error:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to download the hand landmarker model from {MODEL_URL} "
            f"to {model_path}: {error}"
        ) from error

    return model_path
