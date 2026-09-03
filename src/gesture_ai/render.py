"""OpenCV overlay drawing for the live hand-landmark preview."""

from __future__ import annotations

from typing import Any

import cv2

from gesture_ai.landmarks import (
    HAND_CONNECTIONS,
    count_raised_fingers,
    landmark_name,
    normalized_to_pixel,
)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_TEXT_COLOR = (255, 255, 255)
_RECORDING_COLOR = (0, 0, 255)
_READY_COLOR = (0, 200, 0)
_CONNECTION_COLOR = (0, 255, 0)
_JOINT_COLOR = (0, 128, 255)
_SELECTED_COLOR = (0, 255, 255)
_CONTROLS = "r record | m mirror | [] joint | q quit"


def _put_text(frame, text: str, origin: tuple[int, int], color, scale=0.55) -> None:
    cv2.putText(frame, text, origin, _FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, origin, _FONT, scale, color, 1, cv2.LINE_AA)


def draw_overlay(
    frame,
    result: Any,
    selected_landmark: int,
    fps: float,
    recording: bool,
) -> None:
    """Draw the status line and hand skeleton onto ``frame`` in place."""
    height, width = frame.shape[:2]

    state = "REC" if recording else "READY"
    state_color = _RECORDING_COLOR if recording else _READY_COLOR
    _put_text(frame, f"FPS {fps:5.1f}", (10, 24), _TEXT_COLOR)
    _put_text(frame, state, (110, 24), state_color)
    _put_text(frame, _CONTROLS, (170, 24), _TEXT_COLOR)

    hand_landmarks = getattr(result, "hand_landmarks", None) or []
    if not hand_landmarks:
        _put_text(frame, "No hand detected", (10, 52), _TEXT_COLOR)
        return

    landmarks = hand_landmarks[0]
    points = [
        normalized_to_pixel(lm.x, lm.y, width, height) for lm in landmarks
    ]

    for start, end in HAND_CONNECTIONS:
        if start < len(points) and end < len(points):
            cv2.line(
                frame, points[start], points[end], _CONNECTION_COLOR, 2, cv2.LINE_AA
            )

    for point in points:
        cv2.circle(frame, point, 4, _JOINT_COLOR, -1, cv2.LINE_AA)

    if 0 <= selected_landmark < len(points):
        cv2.circle(
            frame,
            points[selected_landmark],
            9,
            _SELECTED_COLOR,
            2,
            cv2.LINE_AA,
        )

    handedness = getattr(result, "handedness", None) or []
    if handedness and handedness[0]:
        category = handedness[0][0]
        _put_text(
            frame,
            f"Hand: {category.category_name} ({category.score:.2f})",
            (10, 52),
            _TEXT_COLOR,
        )

    fingers_up = count_raised_fingers(landmarks)
    _put_text(
        frame,
        f"Fingers up: {fingers_up}",
        (10, 78),
        _SELECTED_COLOR,
        scale=0.8,
    )

    if 0 <= selected_landmark < len(landmarks):
        selected = landmarks[selected_landmark]
        name = landmark_name(selected_landmark)
        _put_text(
            frame,
            f"[{selected_landmark}] {name}",
            (10, 108),
            _SELECTED_COLOR,
        )
        _put_text(
            frame,
            f"x={selected.x:.3f}  y={selected.y:.3f}  z={selected.z:.3f}",
            (10, 134),
            _SELECTED_COLOR,
        )
