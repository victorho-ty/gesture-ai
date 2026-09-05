"""Offline checks for the hand-to-puppet pipeline. No camera, no window.

Runs synthetic hands through features -> smoothing -> rig and asserts the
puppet responds the way the mapping claims. Catches the failures that a
screenshot cannot: a joint wired to the wrong finger, a reversed range, a
rotation that flips near gimbal lock.

    uv run python scripts/selfcheck.py
"""

from __future__ import annotations

import math
import sys

from gesture_ai import features as F
from gesture_ai.landmarks import FINGER_CHAINS
from gesture_ai.motion import PoseSmoother
from gesture_ai.rig import RigMapping, solve

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    ok = bool(condition)
    if not ok:
        FAILURES.append(name)
    suffix = f"   {detail}" if detail else ""
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{suffix}")


class _LM:
    def __init__(self, xyz):
        self.x, self.y, self.z = xyz


class _Cat:
    def __init__(self, name, score):
        self.category_name, self.score = name, score


class _Result:
    """Stands in for a HandLandmarkerResult."""

    def __init__(self, points, hand="Right"):
        if points is None:
            self.hand_landmarks = []
            self.hand_world_landmarks = []
            self.handedness = []
        else:
            self.hand_landmarks = [[_LM(p) for p in points]]
            self.hand_world_landmarks = [[_LM(p) for p in points]]
            self.handedness = [[_Cat(hand, 0.98)]]


def make_hand(curls=(0.0,) * 5, roll=0.0, spread_deg=8.0, pinch_gap=0.9):
    """Build a synthetic hand in MediaPipe convention: metres, Y grows downward.

    ``curls`` bends each finger by rotating successive phalanges toward the
    palm, which is what a real finger does and what ``finger_curl`` measures.
    """
    pts = [None] * 21
    pts[0] = (0.0, 0.0, 0.0)
    mcp_x = {5: -0.10, 9: 0.0, 13: 0.10, 17: 0.20}

    for finger, chain in enumerate(FINGER_CHAINS):
        if finger == 0:
            gap = pinch_gap
            for j, idx in enumerate(chain):
                pts[idx] = (-0.05 - 0.055 * j * gap, -(0.06 + 0.05 * j), 0.0)
            continue
        base_x = mcp_x[chain[0]]
        fan = math.radians(spread_deg * (finger - 2.5) / 1.5)
        # Curl bends each joint progressively; 1.0 folds the finger right in.
        bend = math.radians(80.0 * curls[finger])
        x, y, z = base_x, -0.35, 0.0
        pts[chain[0]] = (x, y, z)
        heading = 0.0
        for j in range(1, 4):
            heading += bend
            x += math.sin(fan) * 0.12 * math.cos(heading)
            y -= 0.12 * math.cos(heading)
            z += 0.12 * math.sin(heading)
            pts[chain[j]] = (x, y, z)

    a = math.radians(roll)
    c, s = math.cos(a), math.sin(a)
    return [(x * c - y * s, x * s + y * c, z) for (x, y, z) in pts]


def settle(features, steps=140, dt=1 / 60):
    """Run the smoother to convergence so filter lag does not skew a check."""
    smoother = PoseSmoother()
    pose = None
    for _ in range(steps):
        pose = smoother.update(features, True, dt)
    return pose


def rig_for(**kwargs):
    features = F.hand_features(_Result(make_hand(**kwargs)))
    assert features is not None
    return solve(settle(features), 0.0, RigMapping()), features


print("== features: rest pose and rotation ==")
flat, _ = None, None
f_open = F.hand_features(_Result(make_hand()))
e = F.palm_euler(F.palm_basis([(l.x, l.y, l.z) for l in _Result(make_hand()).hand_world_landmarks[0]]))
check("upright open hand is the rest pose", max(abs(v) for v in e) < 0.5,
      f"euler={tuple(round(v, 2) for v in e)}")

for angle in (30, 90, 150, 210, 300):
    pts = make_hand(roll=angle)
    b = F.palm_basis(pts)
    d = [abs(F._dot(b[i], b[j])) for i in range(3) for j in range(3) if i < j]
    lens = [abs(F._length(v) - 1.0) for v in b]
    check(f"basis stays orthonormal at {angle} deg roll",
          max(d) < 1e-9 and max(lens) < 1e-9)

print()
print("== features: curl responds to bending ==")
straight = F.hand_features(_Result(make_hand(curls=(0.0,) * 5)))
fisted = F.hand_features(_Result(make_hand(curls=(1.0,) * 5)))
check("open hand curls near 0", max(straight.curl[1:]) < 0.15,
      f"curls={tuple(round(c, 2) for c in straight.curl[1:])}")
check("closed hand curls near 1", min(fisted.curl[1:]) > 0.80,
      f"curls={tuple(round(c, 2) for c in fisted.curl[1:])}")
check("curl is monotonic",
      all(a < b for a, b in zip(straight.curl[1:], fisted.curl[1:])))

print()
print("== rig: the puppet follows the hand ==")
mapping = RigMapping()

# Arms: extending a finger should raise the matching arm.
open_rig, _ = rig_for(curls=(0.0,) * 5)
fist_rig, _ = rig_for(curls=(1.0,) * 5)
open_arm = abs(open_rig.rotations["shoulderR"][2])
fist_arm = abs(fist_rig.rotations["shoulderR"][2])
check("open hand raises the arms", open_arm > fist_arm + 20.0,
      f"open={open_arm:.0f} deg  fist={fist_arm:.0f} deg")

# The right arm should answer to the index finger, the left to the pinky.
only_index, _ = rig_for(curls=(0.0, 0.0, 1.0, 1.0, 1.0))
only_pinky, _ = rig_for(curls=(0.0, 1.0, 1.0, 1.0, 0.0))
check("right arm tracks the index finger",
      abs(only_index.rotations["shoulderR"][2])
      > abs(only_pinky.rotations["shoulderR"][2]) + 20.0)
check("left arm tracks the pinky",
      abs(only_pinky.rotations["shoulderL"][2])
      > abs(only_index.rotations["shoulderL"][2]) + 20.0)

# Elbow follows the middle finger.
bent, _ = rig_for(curls=(0.0, 0.0, 1.0, 0.0, 0.0))
check("elbow bends with the middle finger",
      abs(bent.rotations["foreArmR"][0]) > abs(open_rig.rotations["foreArmR"][0]) + 40.0)

print()
print("== rig: no gimbal flip through a full roll ==")
smoother = PoseSmoother()
previous = None
worst = 0.0
worst_at = 0
for deg in range(0, 361, 3):
    feats = F.hand_features(_Result(make_hand(roll=float(deg))))
    pose = smoother.update(feats, True, 1 / 60)
    joints = solve(pose, 0.0, RigMapping())
    right = joints.matrices["torso"][0]
    if previous is not None:
        delta = math.degrees(
            math.acos(max(-1.0, min(1.0, sum(a * b for a, b in zip(right, previous)))))
        )
        if delta > worst:
            worst, worst_at = delta, deg
    previous = right
check("torso rotation never jumps", worst < 12.0,
      f"largest step {worst:.1f} deg at roll {worst_at} (euler would spike ~180)")

print()
print("== rig: presence fades to idle instead of freezing ==")
smoother = PoseSmoother()
feats = F.hand_features(_Result(make_hand()))
for _ in range(120):
    tracked = smoother.update(feats, True, 1 / 60)
check("presence reaches 1 while tracked", tracked.present > 0.99)
for _ in range(60):
    lost = smoother.update(feats, False, 1 / 60)
check("presence decays after the hand goes", lost.present < 0.05,
      f"present={lost.present:.3f}")
idle_rig = solve(lost, 0.0, RigMapping())
check("idle puppet stays standing",
      abs(idle_rig.root_translation[1] - RigMapping().base_height) < 1.0)
check("idle puppet is upright",
      abs(idle_rig.matrices["torso"][1][1] - 1.0) < 0.05,
      "torso up vector is vertical")

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
    sys.exit(1)
print("all checks passed")
