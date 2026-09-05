"""Derived hand parameters for driving an external renderer.

The landmarker emits 21 raw points per frame, which is the wrong granularity for
puppeteering a rig: nobody wants to bind a robot elbow to ``landmark[6].y``. This
module reduces a frame to roughly a dozen semantic parameters -- palm
orientation, per-finger curl, pinch, spread -- that map onto joint rotations
directly.

Angular quantities come from ``hand_world_landmarks`` (metric, centred on the
hand) rather than the normalized landmarks, whose ``z`` is wrist-relative and
noisy. Screen position keeps using the normalized landmarks, because that is the
space an overlay needs.

Deliberately excluded: any mapping from these parameters onto a particular rig.
Hand-to-humanoid is an artistic choice rather than an anatomical one, and it
wants to be retuned live in TouchDesigner rather than edited and restarted here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from gesture_ai.landmarks import FINGER_CHAINS, LANDMARK_COUNT, landmark_index

Vec = tuple[float, float, float]

WRIST = landmark_index("WRIST")
THUMB_TIP = landmark_index("THUMB_TIP")
INDEX_FINGER_MCP = landmark_index("INDEX_FINGER_MCP")
INDEX_FINGER_TIP = landmark_index("INDEX_FINGER_TIP")
MIDDLE_FINGER_MCP = landmark_index("MIDDLE_FINGER_MCP")
PINKY_MCP = landmark_index("PINKY_MCP")

_EPSILON = 1e-9

# Summed flexion across a finger's three joints when fully curled into a fist.
# The thumb chain bends far less than the others, so it gets its own limit;
# treating them alike would leave thumb curl pinned near zero.
_CURL_FULL = math.radians(240.0)
_THUMB_CURL_FULL = math.radians(110.0)

# Thumb-tip to index-tip separation, as a multiple of wrist-to-middle-MCP
# length, at which the pinch counts as fully open.
_PINCH_OPEN = 1.6

# Mean angle between adjacent proximal phalanges at maximum finger splay.
_SPREAD_FULL = math.radians(18.0)


def _xyz(landmark: Any) -> Vec:
    return (float(landmark.x), float(landmark.y), float(landmark.z))


def _sub(a: Vec, b: Vec) -> Vec:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a: Vec, b: Vec) -> Vec:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: Vec, b: Vec) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _length(a: Vec) -> float:
    return math.sqrt(_dot(a, a))


def _scale(a: Vec, k: float) -> Vec:
    return (a[0] * k, a[1] * k, a[2] * k)


def _normalize(a: Vec) -> Vec:
    length = _length(a)
    if length < _EPSILON:
        return (0.0, 0.0, 0.0)
    return (a[0] / length, a[1] / length, a[2] / length)


def _angle_between(a: Vec, b: Vec) -> float:
    """Unsigned angle between two vectors, in radians."""
    scale = _length(a) * _length(b)
    if scale < _EPSILON:
        return 0.0
    return math.acos(min(1.0, max(-1.0, _dot(a, b) / scale)))


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _to_yup(point: Vec) -> Vec:
    """Flip MediaPipe's Y-down, -Z-forward space into a Y-up, +Z-toward-viewer one.

    This is what makes an upright hand the *rest* pose. Without it the basis of a
    hand held up with the palm facing the camera is a 180-degree rotation about
    X rather than the identity, so every angle is measured from a rest pose
    nobody adopts, and the euler decomposition sits next to gimbal lock where
    small hand movements produce huge angle swings.
    """
    return (point[0], -point[1], -point[2])


def palm_basis(world: Sequence[Vec]) -> tuple[Vec, Vec, Vec]:
    """Return an orthonormal ``(right, up, normal)`` frame for the palm.

    ``up`` runs from the wrist toward the middle-finger MCP, along the fingers,
    and ``right`` runs across the knuckles from index to pinky. The palm normal
    is their cross product, and ``right`` is then recomputed from it so the frame
    comes out exactly orthonormal despite the two source vectors not being
    perpendicular on a real hand.

    A hand held upright with the palm toward the camera yields the identity,
    which is the pose all rig angles are relative to.
    """
    wrist = _to_yup(world[WRIST])
    up = _normalize(_sub(_to_yup(world[MIDDLE_FINGER_MCP]), wrist))
    across = _sub(_to_yup(world[PINKY_MCP]), _to_yup(world[INDEX_FINGER_MCP]))
    normal = _normalize(_cross(across, up))
    right = _cross(up, normal)
    return right, up, normal


def orthonormalize(basis: tuple[Vec, Vec, Vec]) -> tuple[Vec, Vec, Vec]:
    """Gram-Schmidt a nearly-orthonormal frame back into an exact one.

    Needed after any per-component operation on a rotation -- filtering or
    blending the nine numbers independently produces something close to a
    rotation but not quite one, and feeding that to a renderer shears the mesh.
    """
    right, up, normal = basis
    up = _normalize(up)
    right = _normalize(_sub(right, _scale(up, _dot(right, up))))
    normal = _cross(right, up)
    return right, up, normal


def blend_to_identity(basis: tuple[Vec, Vec, Vec], amount: float) -> tuple[Vec, Vec, Vec]:
    """Scale a rotation down toward no rotation at all.

    ``amount`` of 1 keeps the rotation, 0 removes it. Interpolating the matrix
    componentwise and re-orthonormalizing is the cheap stand-in for a proper
    slerp; over the angles a wrist can reach it is indistinguishable.
    """
    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    blended = tuple(
        tuple(i + (b - i) * amount for i, b in zip(iv, bv))
        for iv, bv in zip(identity, basis)
    )
    return orthonormalize(blended)  # type: ignore[arg-type]


def palm_euler(basis: tuple[Vec, Vec, Vec]) -> Vec:
    """Decompose a palm basis into ``(pitch, yaw, roll)`` degrees.

    Uses the ``Rz * Ry * Rx`` convention, matching TouchDesigner's default XYZ
    rotate order, so the values can be exported straight onto a transform.
    """
    right, up, normal = basis
    # Rotation matrix whose columns are the basis vectors.
    r00, r10, r20 = right[0], right[1], right[2]
    r01, r11, r21 = up[0], up[1], up[2]
    r02, r12, r22 = normal[0], normal[1], normal[2]

    cos_yaw = math.sqrt(r00 * r00 + r10 * r10)
    if cos_yaw < _EPSILON:
        # Gimbal lock: pitch and roll become degenerate, so fold both into pitch.
        pitch = math.atan2(-r12, r11)
        yaw = math.atan2(-r20, cos_yaw)
        roll = 0.0
    else:
        pitch = math.atan2(r21, r22)
        yaw = math.atan2(-r20, cos_yaw)
        roll = math.atan2(r10, r00)

    return (math.degrees(pitch), math.degrees(yaw), math.degrees(roll))


def finger_curl(world: Sequence[Vec], finger: int) -> float:
    """Return how curled one finger is, from ``0`` (straight) to ``1`` (fisted).

    Summing the interior angles at the three joints makes this invariant to hand
    size and to distance from the camera, unlike a tip-to-palm distance would be.
    """
    proximal, middle, distal, tip = (world[i] for i in FINGER_CHAINS[finger])
    wrist = world[WRIST]
    total = (
        _angle_between(_sub(proximal, wrist), _sub(middle, proximal))
        + _angle_between(_sub(middle, proximal), _sub(distal, middle))
        + _angle_between(_sub(distal, middle), _sub(tip, distal))
    )
    limit = _THUMB_CURL_FULL if finger == 0 else _CURL_FULL
    return _clamp01(total / limit)


def pinch(world: Sequence[Vec]) -> float:
    """Thumb-to-index separation: ``0`` when touching, ``1`` when wide open."""
    scale = _length(_sub(world[MIDDLE_FINGER_MCP], world[WRIST]))
    if scale < _EPSILON:
        return 0.0
    separation = _length(_sub(world[THUMB_TIP], world[INDEX_FINGER_TIP])) / scale
    return _clamp01(separation / _PINCH_OPEN)


def spread(world: Sequence[Vec]) -> float:
    """Finger splay: ``0`` when fingers are together, ``1`` when fanned.

    Measured between proximal phalanges rather than fingertips, because tip
    directions converge as fingers curl, which would read a fist as maximally
    splayed.
    """
    directions = []
    for finger in range(1, 5):
        chain = FINGER_CHAINS[finger]
        directions.append(_normalize(_sub(world[chain[1]], world[chain[0]])))
    total = sum(
        _angle_between(directions[i], directions[i + 1])
        for i in range(len(directions) - 1)
    )
    return _clamp01((total / 3.0) / _SPREAD_FULL)


def hand_scale(normalized: Sequence[Vec]) -> float:
    """Apparent hand size in normalized screen units; a usable depth proxy."""
    wrist = normalized[WRIST]
    mcp = normalized[MIDDLE_FINGER_MCP]
    return math.hypot(mcp[0] - wrist[0], mcp[1] - wrist[1])


class EulerUnwrapper:
    """Keeps euler angles continuous across the +/-180 degree seam.

    Without this, a palm rotating past 180 degrees emits a 360-degree jump, and
    any downstream smoothing filter renders that as a full spin rather than the
    small movement it actually was.
    """

    def __init__(self) -> None:
        self._previous: Vec | None = None

    def __call__(self, angles: Vec) -> Vec:
        if self._previous is None:
            self._previous = angles
            return angles

        unwrapped = []
        for previous, current in zip(self._previous, angles):
            unwrapped.append(current - 360.0 * round((current - previous) / 360.0))

        self._previous = (unwrapped[0], unwrapped[1], unwrapped[2])
        return self._previous

    def reset(self) -> None:
        """Forget history, so a reappearing hand does not inherit old winding."""
        self._previous = None


@dataclass(frozen=True)
class HandFeatures:
    """One frame of hand state, ready to serialize."""

    handedness: float
    """Signed confidence: negative for a left hand, positive for a right one."""

    wrist: Vec
    """Normalized screen x and y, plus apparent hand scale."""

    palm: Vec
    """Palm pitch, yaw, and roll in degrees."""

    curl: tuple[float, ...]
    """Per-finger curl, thumb first, each 0..1."""

    pinch: float
    spread: float

    basis: tuple[float, ...]
    """Palm rotation as 9 floats: right, up, normal.

    Carried alongside ``palm`` because euler angles hit gimbal lock when the
    palm turns edge-on -- which happens constantly in ordinary use -- and the
    decomposition then swings wildly for tiny real movements. Anything driving
    a 3D rotation should use this; ``palm`` is for display and for OSC.
    """

    pose: tuple[float, ...]
    """63 world-landmark floats, flattened xyz."""

    screen: tuple[float, ...]
    """42 normalized floats, flattened xy."""


def _flatten3(points: Iterable[Vec]) -> tuple[float, ...]:
    return tuple(value for point in points for value in point)


def _flatten2(points: Iterable[Vec]) -> tuple[float, ...]:
    return tuple(value for point in points for value in point[:2])


def hand_features(
    result: Any,
    unwrapper: EulerUnwrapper | None = None,
) -> HandFeatures | None:
    """Reduce a landmarker result to one parameter block, or ``None`` if no hand.

    Mirrors ``result_to_record`` in shape, but returns the derived parameters a
    rig wants rather than the raw record the capture file wants.
    """
    hand_landmarks = getattr(result, "hand_landmarks", None) or []
    world_landmarks = getattr(result, "hand_world_landmarks", None) or []
    if not hand_landmarks or not world_landmarks:
        return None

    normalized = [_xyz(lm) for lm in hand_landmarks[0]]
    world = [_xyz(lm) for lm in world_landmarks[0]]
    if len(normalized) < LANDMARK_COUNT or len(world) < LANDMARK_COUNT:
        return None

    handedness_value = 0.0
    handedness = getattr(result, "handedness", None) or []
    if handedness and handedness[0]:
        category = handedness[0][0]
        sign = -1.0 if category.category_name == "Left" else 1.0
        handedness_value = sign * float(category.score)

    basis = palm_basis(world)
    palm = palm_euler(basis)
    if unwrapper is not None:
        palm = unwrapper(palm)

    wrist = normalized[WRIST]
    return HandFeatures(
        handedness=handedness_value,
        wrist=(wrist[0], wrist[1], hand_scale(normalized)),
        palm=palm,
        curl=tuple(finger_curl(world, index) for index in range(5)),
        pinch=pinch(world),
        spread=spread(world),
        basis=_flatten3(basis),
        pose=_flatten3(world),
        screen=_flatten2(normalized),
    )
