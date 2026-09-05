"""Model-independent hand-landmark naming, geometry, and serialization."""

from __future__ import annotations

import math
from typing import Any

LANDMARK_NAMES: tuple[str, ...] = (
    "WRIST",
    "THUMB_CMC",
    "THUMB_MCP",
    "THUMB_IP",
    "THUMB_TIP",
    "INDEX_FINGER_MCP",
    "INDEX_FINGER_PIP",
    "INDEX_FINGER_DIP",
    "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP",
    "MIDDLE_FINGER_PIP",
    "MIDDLE_FINGER_DIP",
    "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP",
    "RING_FINGER_PIP",
    "RING_FINGER_DIP",
    "RING_FINGER_TIP",
    "PINKY_MCP",
    "PINKY_PIP",
    "PINKY_DIP",
    "PINKY_TIP",
)

LANDMARK_COUNT = len(LANDMARK_NAMES)

HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

FINGER_NAMES: tuple[str, ...] = ("thumb", "index", "middle", "ring", "pinky")

# Four-joint chain per finger, ordered proximal to distal. The thumb has no PIP
# or DIP, so its chain is (CMC, MCP, IP, TIP) while the others are
# (MCP, PIP, DIP, TIP); both are four joints, which lets callers walk every
# finger with one loop.
FINGER_CHAINS: tuple[tuple[int, int, int, int], ...] = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
)

SCHEMA_VERSION = 1

# Landmark indices used to count raised fingers.
_WRIST = 0
_THUMB_TIP = 4
_THUMB_IP = 3
_PINKY_MCP = 17
# (tip, pip) pairs for index, middle, ring, and pinky fingers.
_FINGERS: tuple[tuple[int, int], ...] = (
    (8, 6),
    (12, 10),
    (16, 14),
    (20, 18),
)


def landmark_name(index: int) -> str:
    """Return the canonical name of a hand landmark index."""
    if not 0 <= index < LANDMARK_COUNT:
        raise ValueError(
            f"landmark index must be in 0..{LANDMARK_COUNT - 1}, got {index}"
        )
    return LANDMARK_NAMES[index]


def landmark_index(name: str) -> int:
    """Return the index of a landmark by its canonical name."""
    try:
        return LANDMARK_NAMES.index(name)
    except ValueError as error:
        raise ValueError(f"unknown landmark name {name!r}") from error


def normalized_to_pixel(
    x: float, y: float, width: int, height: int
) -> tuple[int, int]:
    """Convert normalized coordinates to clamped integer pixel coordinates."""
    px = round(x * (width - 1))
    py = round(y * (height - 1))
    px = min(max(px, 0), max(width - 1, 0))
    py = min(max(py, 0), max(height - 1, 0))
    return px, py


def _distance(a: Any, b: Any) -> float:
    """2D distance between two normalized landmarks."""
    return math.hypot(a.x - b.x, a.y - b.y)


def count_raised_fingers(landmarks: Any) -> int:
    """Return how many fingers are extended, from ``0`` to ``5``.

    The index, middle, ring, and pinky fingers are raised when their tip is
    farther from the wrist than their PIP joint; this distance heuristic is
    orientation independent and catches a finger folded into the palm at any
    joint.

    The thumb folds *across* the palm rather than toward the wrist, so radial
    distance from the wrist cannot tell its state. Instead the thumb is counted
    as raised when its tip is farther from the pinky MCP (the opposite, ulnar
    side of the hand) than its IP joint is — i.e. the thumb is abducted away
    from the palm rather than tucked against it.
    """
    if not landmarks or len(landmarks) < 21:
        return 0

    wrist = landmarks[_WRIST]
    count = 0

    for tip, pip in _FINGERS:
        if _distance(wrist, landmarks[tip]) > _distance(wrist, landmarks[pip]):
            count += 1

    pinky_mcp = landmarks[_PINKY_MCP]
    if _distance(pinky_mcp, landmarks[_THUMB_TIP]) > _distance(
        pinky_mcp, landmarks[_THUMB_IP]
    ):
        count += 1

    return count


def _point(landmark: Any) -> dict[str, float]:
    return {
        "x": float(landmark.x),
        "y": float(landmark.y),
        "z": float(landmark.z),
    }


def result_to_record(
    result: Any,
    timestamp_ms: int,
    image_width: int,
    image_height: int,
) -> dict[str, Any] | None:
    """Build a JSONL-ready record, or ``None`` when no hand was detected."""
    hand_landmarks = getattr(result, "hand_landmarks", None) or []
    if not hand_landmarks:
        return None

    world_landmarks = getattr(result, "hand_world_landmarks", None) or []
    handedness = getattr(result, "handedness", None) or []

    hands: list[dict[str, Any]] = []
    for index, landmarks in enumerate(hand_landmarks):
        label: str | None = None
        score: float | None = None
        if index < len(handedness) and handedness[index]:
            category = handedness[index][0]
            label = category.category_name
            score = float(category.score)

        world = world_landmarks[index] if index < len(world_landmarks) else []
        hands.append(
            {
                "handedness": {"label": label, "score": score},
                "normalized_landmarks": [_point(lm) for lm in landmarks],
                "world_landmarks": [_point(lm) for lm in world],
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp_ms": int(timestamp_ms),
        "image_size": {"width": int(image_width), "height": int(image_height)},
        "hands": hands,
    }
