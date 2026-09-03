"""Live webcam hand-landmark application."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, TextIO

import cv2
import mediapipe as mp

from gesture_ai.landmarks import LANDMARK_COUNT, result_to_record
from gesture_ai.model import ensure_model
from gesture_ai.render import draw_overlay

DEFAULT_MODEL_PATH = Path("models") / "hand_landmarker.task"
DEFAULT_OUTPUT_PATH = Path("captures") / "hand_landmarks.jsonl"

WINDOW_NAME = "gesture-ai"
_KEY_ESCAPE = 27


class JsonlRecorder:
    """Append-only JSON Lines writer that opens its target on first write."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._stream: TextIO | None = None

    @property
    def path(self) -> Path:
        return self._path

    def write(self, record: dict[str, Any]) -> None:
        if self._stream is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = self._path.open("a", encoding="utf-8")
        json.dump(record, self._stream, separators=(",", ":"))
        self._stream.write("\n")
        self._stream.flush()

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.close()
            finally:
                self._stream = None


def _joint_index(value: str) -> int:
    try:
        index = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from error
    if not 0 <= index < LANDMARK_COUNT:
        raise argparse.ArgumentTypeError(
            f"joint index must be in 0..{LANDMARK_COUNT - 1}, got {index}"
        )
    return index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gesture-ai",
        description="Live MediaPipe hand-landmark webcam demo.",
    )
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--width", type=int, default=1280, help="Requested frame width")
    parser.add_argument(
        "--height", type=int, default=720, help="Requested frame height"
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Requested camera FPS")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the hand_landmarker.task model file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="JSON Lines capture destination",
    )
    parser.add_argument(
        "--joint",
        type=_joint_index,
        default=8,
        help="Landmark index (0..20) shown on screen; default is the index fingertip",
    )
    parser.add_argument(
        "--mirror",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mirror the preview horizontally",
    )
    parser.add_argument(
        "--detection-confidence",
        type=float,
        default=0.5,
        help="Minimum hand detection confidence in [0, 1]",
    )
    parser.add_argument(
        "--presence-confidence",
        type=float,
        default=0.5,
        help="Minimum hand presence confidence in [0, 1]",
    )
    parser.add_argument(
        "--tracking-confidence",
        type=float,
        default=0.5,
        help="Minimum hand tracking confidence in [0, 1]",
    )
    parser.add_argument(
        "--download-model",
        action="store_true",
        help="Ensure the model file exists, print its path, and exit",
    )
    return parser


def _validate_confidences(parser: argparse.ArgumentParser, args) -> None:
    for name in ("detection_confidence", "presence_confidence", "tracking_confidence"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            parser.error(
                f"--{name.replace('_', '-')} must be in [0, 1], got {value}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_confidences(parser, args)

    try:
        model_path = ensure_model(args.model)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1

    if args.download_model:
        print(model_path.resolve())
        return 0

    capture = cv2.VideoCapture(args.camera)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    capture.set(cv2.CAP_PROP_FPS, args.fps)
    if not capture.isOpened():
        capture.release()
        print(f"Could not open camera index {args.camera}.", file=sys.stderr)
        return 1

    vision = mp.tasks.vision
    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=args.detection_confidence,
        min_hand_presence_confidence=args.presence_confidence,
        min_tracking_confidence=args.tracking_confidence,
    )

    recorder = JsonlRecorder(args.output)
    mirror = args.mirror
    selected_landmark = args.joint
    recording = False
    fps = 0.0
    start = time.perf_counter()
    previous = start
    last_timestamp_ms = -1

    try:
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = capture.read()
                if not ok:
                    print("Camera frame read failed; stopping.", file=sys.stderr)
                    break

                if mirror:
                    frame = cv2.flip(frame, 1)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=rgb_frame
                )

                now = time.perf_counter()
                timestamp_ms = int((now - start) * 1000.0)
                if timestamp_ms <= last_timestamp_ms:
                    timestamp_ms = last_timestamp_ms + 1
                last_timestamp_ms = timestamp_ms

                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if recording:
                    height, width = frame.shape[:2]
                    record = result_to_record(result, timestamp_ms, width, height)
                    if record is not None:
                        recorder.write(record)

                elapsed = now - previous
                previous = now
                if elapsed > 0:
                    fps = 1.0 / elapsed

                draw_overlay(frame, result, selected_landmark, fps, recording)
                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), _KEY_ESCAPE):
                    break
                if key == ord("m"):
                    mirror = not mirror
                elif key == ord("r"):
                    recording = not recording
                    if recording:
                        print(f"Recording to {recorder.path}")
                    else:
                        print("Recording stopped.")
                elif key == ord("["):
                    selected_landmark = (selected_landmark - 1) % LANDMARK_COUNT
                elif key == ord("]"):
                    selected_landmark = (selected_landmark + 1) % LANDMARK_COUNT
    except KeyboardInterrupt:
        pass
    finally:
        recorder.close()
        capture.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
