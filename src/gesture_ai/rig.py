"""The puppet: a humanoid mecha skeleton and the hand-to-robot mapping.

Two separate things live here, and the split matters.

``SKELETON`` is anatomy -- which bone hangs off which, how long each one is.
It changes only when the robot's shape changes.

``RigMapping`` is choreography -- which hand parameter drives which joint, and
through what range. A hand landmarker reports no elbow and no shoulder, so
"the robot mimics my hand" is an artistic choice rather than an anatomical
truth: index curl driving the right arm is a decision, not a measurement. Those
decisions need many small adjustments before the puppet stops looking like a
marionette, so every constant is a field on one dataclass that can be retuned
live rather than a number buried in draw code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from gesture_ai.features import blend_to_identity
from gesture_ai.motion import SmoothedPose

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class Bone:
    """One rigid segment of the puppet."""

    name: str
    parent: str | None
    offset: Vec3
    """Joint position relative to the parent's joint."""

    size: Vec3
    """Box dimensions."""

    center: Vec3
    """Box centre relative to this bone's joint, so bones pivot at their end."""

    color: Vec3
    """RGB 0-255."""

    glow: bool = False
    """Drawn at full brightness and pulsed by motion energy."""


_BODY = (150, 146, 205)
_PLATE = (176, 172, 226)
_DARK = (74, 72, 104)
_ACCENT = (240, 130, 55)
_VISOR = (90, 210, 235)

# Parents always precede their children, so one forward pass resolves the
# hierarchy without recursion or sorting.
SKELETON: tuple[Bone, ...] = (
    Bone("root", None, (0, 0, 0), (0, 0, 0), (0, 0, 0), _BODY),

    # The pelvis hangs off the root, and the legs off the pelvis rather than
    # off the torso. That keeps the feet planted when the palm turns: rotating
    # the torso would otherwise swing the whole lower body and the puppet would
    # look like it was toppling instead of twisting.
    Bone("pelvis", "root", (0, 0, 0), (1.35, 0.62, 1.0), (0, 0, 0), _DARK),
    Bone("thighL", "pelvis", (-0.42, -0.3, 0), (0.5, 1.15, 0.5), (0, -0.57, 0), _BODY),
    Bone("shinL", "thighL", (0, -1.15, 0), (0.44, 1.05, 0.44), (0, -0.52, 0), _PLATE),
    Bone("footL", "shinL", (0, -1.05, 0), (0.55, 0.26, 0.85), (0, -0.13, 0.16), _DARK),
    Bone("thighR", "pelvis", (0.42, -0.3, 0), (0.5, 1.15, 0.5), (0, -0.57, 0), _BODY),
    Bone("shinR", "thighR", (0, -1.15, 0), (0.44, 1.05, 0.44), (0, -0.52, 0), _PLATE),
    Bone("footR", "shinR", (0, -1.05, 0), (0.55, 0.26, 0.85), (0, -0.13, 0.16), _DARK),

    Bone("torso", "pelvis", (0, 0.3, 0), (1.45, 1.5, 0.92), (0, 0.72, 0), _BODY),
    Bone("chest", "torso", (0, 1.5, 0), (2.1, 1.1, 1.15), (0, 0.32, 0), _PLATE),
    Bone("collar", "chest", (0, 0.8, 0.1), (1.0, 0.28, 0.55), (0, 0, 0), _DARK),
    Bone("head", "chest", (0, 0.92, 0), (1.02, 0.92, 1.0), (0, 0.4, 0), _PLATE),
    Bone("visor", "head", (0, 0.44, 0.5), (0.76, 0.3, 0.12), (0, 0, 0), _VISOR, True),
    Bone("antenna", "head", (0.3, 0.84, 0), (0.07, 0.6, 0.07), (0, 0.3, 0), _ACCENT),
    Bone("antenna_tip", "antenna", (0, 0.62, 0), (0.15, 0.15, 0.15), (0, 0, 0), _ACCENT, True),

    Bone("shoulderL", "chest", (-1.18, 0.42, 0), (0.6, 0.6, 0.6), (0, 0, 0), _DARK),
    Bone("upperArmL", "shoulderL", (0, -0.12, 0), (0.42, 1.2, 0.42), (0, -0.6, 0), _BODY),
    Bone("foreArmL", "upperArmL", (0, -1.2, 0), (0.37, 1.1, 0.37), (0, -0.55, 0), _PLATE),
    Bone("clawL", "foreArmL", (0, -1.1, 0), (0.19, 0.48, 0.42), (0, -0.24, 0), _ACCENT),
    Bone("clawL2", "foreArmL", (0, -1.1, 0), (0.19, 0.48, 0.42), (0, -0.24, 0), _ACCENT),

    Bone("shoulderR", "chest", (1.18, 0.42, 0), (0.6, 0.6, 0.6), (0, 0, 0), _DARK),
    Bone("upperArmR", "shoulderR", (0, -0.12, 0), (0.42, 1.2, 0.42), (0, -0.6, 0), _BODY),
    Bone("foreArmR", "upperArmR", (0, -1.2, 0), (0.37, 1.1, 0.37), (0, -0.55, 0), _PLATE),
    Bone("clawR", "foreArmR", (0, -1.1, 0), (0.19, 0.48, 0.42), (0, -0.24, 0), _ACCENT),
    Bone("clawR2", "foreArmR", (0, -1.1, 0), (0.19, 0.48, 0.42), (0, -0.24, 0), _ACCENT),
)

BONE_INDEX = {bone.name: i for i, bone in enumerate(SKELETON)}


@dataclass
class RigMapping:
    """Every tunable in the hand-to-robot mapping.

    Ranges are ``(at_zero, at_one)`` pairs in degrees, so reversing a joint
    means swapping the pair rather than editing a formula.
    """

    # Where the puppet sits, from wrist position and apparent hand size.
    base_height: float = 2.7
    """Root height that puts the feet on the ground plane. The skeleton hangs
    downward from the root, so without this the puppet renders half-buried."""
    pan: float = 7.0
    """World units of horizontal travel across the full frame width."""
    rise: float = 2.6
    depth_near: float = 0.30
    """Hand scale that reads as closest to the camera."""
    depth_far: float = 0.10
    depth_range: tuple[float, float] = (-5.0, 3.5)

    # Body orientation from palm orientation.
    torso_follow: float = 0.7
    """How much of the palm's rotation the torso takes, 0 upright to 1 exact.
    Under 1 keeps the puppet readable when the hand turns right over."""
    head_follow: float = 0.45
    """Head yaw as a fraction of torso yaw; under 1 makes the head lag, which
    reads as a neck rather than a welded block."""
    head_lead: float = 0.05
    """Degrees of extra head turn per degree/second of palm rotation."""

    # Limbs from finger curl. Arms hang at rest and lift as fingers extend.
    arm_raise: tuple[float, float] = (8.0, 105.0)
    elbow: tuple[float, float] = (-4.0, -125.0)
    arm_swing: tuple[float, float] = (2.0, -22.0)
    leg_bend: tuple[float, float] = (-2.0, 38.0)
    knee: tuple[float, float] = (2.0, -55.0)
    stance: tuple[float, float] = (-1.0, 16.0)
    claw: tuple[float, float] = (2.0, 38.0)

    # Secondary motion: the puppet leans into its own movement.
    lean: float = 26.0
    bob: float = 0.5
    idle_sway: float = 6.0
    idle_speed: float = 1.1


@dataclass
class JointPose:
    """Resolved state for one frame: rotations per bone, plus root placement."""

    rotations: dict[str, Vec3] = field(default_factory=dict)
    matrices: dict[str, tuple[Vec3, Vec3, Vec3]] = field(default_factory=dict)
    """Bones oriented by a rotation matrix instead of euler angles, applied
    before the euler term. Used for the torso, whose orientation comes straight
    from the palm and must not pass through a gimbal-locked decomposition."""
    root_translation: Vec3 = (0.0, 0.0, 0.0)
    glow: float = 0.0
    """0..1 brightness for emissive parts."""


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _remap(value: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
    if abs(hi - lo) < 1e-9:
        return out_lo
    t = (value - lo) / (hi - lo)
    return out_lo + (out_hi - out_lo) * max(0.0, min(1.0, t))


def solve(pose: SmoothedPose, elapsed: float, m: RigMapping) -> JointPose:
    """Map one smoothed hand pose onto puppet joint angles.

    ``pose.present`` is a 0..1 blend weight rather than a boolean, so the puppet
    eases between the tracked pose and an idle animation instead of snapping
    when tracking drops for a frame.
    """
    live = pose.present
    idle = 1.0 - live

    breath = math.sin(elapsed * m.idle_speed)
    sway = math.sin(elapsed * m.idle_speed * 0.7) * m.idle_sway
    energy = pose.motion.energy

    # -- placement --------------------------------------------------------
    # Screen x/y are 0..1 with y pointing down; the world is centred and y-up.
    wx, wy, scale = pose.wrist if pose.wrist else (0.5, 0.5, 0.2)
    tx = (wx - 0.5) * 2.0 * m.pan * live
    ty = m.base_height + (0.5 - wy) * 2.0 * m.rise * live
    tz = _remap(scale, m.depth_far, m.depth_near, *m.depth_range) * live
    ty += breath * m.bob * (0.4 + 0.6 * idle)

    # -- orientation ------------------------------------------------------
    # The torso takes the palm's rotation matrix directly, damped toward
    # upright. Euler angles are not usable here: yaw is the middle axis of the
    # decomposition, so it gimbal-locks whenever the palm turns edge-on, and
    # the puppet flips for no real movement.
    b = pose.basis
    torso_basis = blend_to_identity(
        ((b[0], b[1], b[2]), (b[3], b[4], b[5]), (b[6], b[7], b[8])),
        m.torso_follow * live,
    )

    # A stable yaw for the secondary bones, read off the direction the palm
    # faces rather than from the euler decomposition.
    normal = (b[6], b[7], b[8])
    facing = math.degrees(math.atan2(normal[0], max(1e-6, abs(normal[2])) * (1 if normal[2] >= 0 else -1)))
    torso_yaw = facing * live + sway * idle
    pitch = pose.palm[0] if pose.palm else 0.0

    # Lean into travel, so a fast sweep tips the body the way it is going.
    dx, dy = pose.motion.direction
    lean_roll = -dx * energy * m.lean * live
    lean_pitch = dy * energy * m.lean * 0.5 * live

    curl = pose.curl if len(pose.curl) == 5 else (0.0,) * 5
    thumb, index, middle, ring, pinky = curl

    # -- limbs ------------------------------------------------------------
    # Extending a finger raises the matching arm, so an open palm is arms-up.
    raise_r = _lerp(*m.arm_raise, 1.0 - index) * live
    raise_l = _lerp(*m.arm_raise, 1.0 - pinky) * live
    elbow = _lerp(*m.elbow, middle) * live
    swing = _lerp(*m.arm_swing, pose.spread) * live
    leg = _lerp(*m.leg_bend, ring) * live
    knee_angle = _lerp(*m.knee, ring) * live
    stance = _lerp(*m.stance, pose.spread) * live
    claw_open = _lerp(*m.claw, pose.pinch) * live

    # Idle: arms drift down, a slow breathing sway.
    raise_r = _lerp(12.0 + breath * 4.0, raise_r, live)
    raise_l = _lerp(12.0 - breath * 4.0, raise_l, live)
    elbow = _lerp(-14.0 - breath * 5.0, elbow, live)

    head_yaw = torso_yaw * m.head_follow + pose.motion.turn_speed * m.head_lead
    head_pitch = -pitch * 0.25 * live + breath * 2.0

    rotations: dict[str, Vec3] = {
        "torso": (lean_pitch, 0.0, lean_roll),
        "chest": (0.0, torso_yaw * 0.2, 0.0),
        "head": (head_pitch, head_yaw, -lean_roll * 0.4),
        "antenna": (0.0, 0.0, breath * 7.0 + energy * 12.0),
        "shoulderL": (0.0, 0.0, -raise_l - swing),
        "upperArmL": (0.0, 0.0, 0.0),
        "foreArmL": (elbow, 0.0, 0.0),
        "clawL": (0.0, 0.0, claw_open),
        "clawL2": (0.0, 0.0, -claw_open),
        "shoulderR": (0.0, 0.0, raise_r + swing),
        "upperArmR": (0.0, 0.0, 0.0),
        "foreArmR": (elbow, 0.0, 0.0),
        "clawR": (0.0, 0.0, claw_open),
        "clawR2": (0.0, 0.0, -claw_open),
        "pelvis": (0.0, torso_yaw * 0.3, 0.0),
        "thighL": (leg, 0.0, -stance),
        "shinL": (knee_angle, 0.0, 0.0),
        "thighR": (leg, 0.0, stance),
        "shinR": (knee_angle, 0.0, 0.0),
    }

    # Thumb curl tilts the head, which gives the thumb something to do and
    # reads as the robot cocking its head.
    pitch_extra, yaw_extra, roll_extra = rotations["head"]
    rotations["head"] = (pitch_extra, yaw_extra, roll_extra + (thumb - 0.5) * 22.0 * live)

    return JointPose(
        rotations=rotations,
        matrices={"torso": torso_basis},
        root_translation=(tx, ty, tz),
        glow=0.35 + 0.65 * energy,
    )
