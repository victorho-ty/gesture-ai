"""Threaded camera capture and hand tracking.

The renderer wants 60 fps; MediaPipe inference costs 15-25 ms on this CPU and
caps out near 30. Running both in one loop would pin the whole application to
the slower of the two and feel like mud, so tracking lives on its own thread and
publishes into a single slot the renderer samples whenever it likes.

The slot holds exactly one pose -- the newest. A queue would be the wrong shape
here: under load it would grow a backlog and the puppet would drift further and
further behind the hand. Dropping stale poses instead keeps latency flat.

This works in Python because the two expensive calls in the loop,
``VideoCapture.read()`` and MediaPipe inference, are native code that releases
the GIL, so the render thread is not starved while they run.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp

from gesture_ai.features import EulerUnwrapper, HandFeatures, hand_features


@dataclass(frozen=True)
class TrackerFrame:
    """One published tracking result."""

    features: HandFeatures | None
    """Latest features, or the last known ones while the hand is missing."""

    present: bool
    """Whether a hand was actually detected in this frame."""

    frame: Any
    """The BGR camera frame, already mirrored. Treat as read-only."""

    timestamp_ms: int
    seq: int


class HandTracker:
    """Runs capture and inference on a background thread."""

    def __init__(
        self,
        model_path: Path,
        camera: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: float = 30.0,
        mirror: bool = True,
        detection_confidence: float = 0.5,
        presence_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
        keep_frame: bool = True,
    ) -> None:
        self._model_path = Path(model_path)
        self._camera = camera
        self._width = width
        self._height = height
        self._fps = fps
        self._mirror = mirror
        self._detection_confidence = detection_confidence
        self._presence_confidence = presence_confidence
        self._tracking_confidence = tracking_confidence
        self._keep_frame = keep_frame

        self._lock = threading.Lock()
        self._slot: TrackerFrame | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error: str | None = None
        self._measured_fps = 0.0
        self._infer_ms = 0.0
        self._seq = 0

    # -- lifecycle ---------------------------------------------------------

    def start(self, timeout: float = 20.0) -> None:
        """Start the thread and block until the camera and model are live."""
        self._thread = threading.Thread(
            target=self._run, name="hand-tracker", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(timeout):
            self.stop()
            raise RuntimeError("Tracker did not start within the timeout.")
        if self._error:
            self.stop()
            raise RuntimeError(self._error)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def __enter__(self) -> "HandTracker":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- consumer side -----------------------------------------------------

    def latest(self) -> TrackerFrame | None:
        """Return the newest pose, or ``None`` before the first one arrives."""
        with self._lock:
            return self._slot

    @property
    def fps(self) -> float:
        """Measured tracker rate, for the HUD."""
        return self._measured_fps

    @property
    def inference_ms(self) -> float:
        return self._infer_ms

    @property
    def mirror(self) -> bool:
        return self._mirror

    @mirror.setter
    def mirror(self, value: bool) -> None:
        self._mirror = value

    # -- worker ------------------------------------------------------------

    def _run(self) -> None:
        capture = None
        try:
            capture = cv2.VideoCapture(self._camera, cv2.CAP_DSHOW)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            capture.set(cv2.CAP_PROP_FPS, self._fps)
            # A 1-frame driver buffer keeps read() from handing back stale
            # frames, which would show up as latency no filter can remove.
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not capture.isOpened():
                self._error = f"Could not open camera index {self._camera}."
                self._ready.set()
                return

            vision = mp.tasks.vision
            options = vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=str(self._model_path)
                ),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=self._detection_confidence,
                min_hand_presence_confidence=self._presence_confidence,
                min_tracking_confidence=self._tracking_confidence,
            )

            unwrapper = EulerUnwrapper()
            last_features: HandFeatures | None = None
            start = time.perf_counter()
            previous = start
            last_timestamp_ms = -1

            with vision.HandLandmarker.create_from_options(options) as landmarker:
                self._ready.set()
                while not self._stop.is_set():
                    ok, frame = capture.read()
                    if not ok:
                        time.sleep(0.005)
                        continue

                    if self._mirror:
                        frame = cv2.flip(frame, 1)

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                    now = time.perf_counter()
                    timestamp_ms = int((now - start) * 1000.0)
                    if timestamp_ms <= last_timestamp_ms:
                        timestamp_ms = last_timestamp_ms + 1
                    last_timestamp_ms = timestamp_ms

                    infer_start = time.perf_counter()
                    result = landmarker.detect_for_video(image, timestamp_ms)
                    self._infer_ms = (time.perf_counter() - infer_start) * 1000.0

                    features = hand_features(result, unwrapper)
                    present = features is not None
                    if present:
                        last_features = features
                    else:
                        # Do not let a hand that reappears elsewhere inherit the
                        # old rotation winding.
                        unwrapper.reset()

                    self._seq += 1
                    published = TrackerFrame(
                        features=features if present else last_features,
                        present=present,
                        frame=frame if self._keep_frame else None,
                        timestamp_ms=timestamp_ms,
                        seq=self._seq,
                    )
                    with self._lock:
                        self._slot = published

                    elapsed = now - previous
                    previous = now
                    if elapsed > 0:
                        # Smoothed so the HUD number is readable.
                        self._measured_fps += (
                            1.0 / elapsed - self._measured_fps
                        ) * 0.1
        except Exception as error:  # surfaced to the caller via start()
            self._error = f"{type(error).__name__}: {error}"
        finally:
            if capture is not None:
                capture.release()
            self._ready.set()
