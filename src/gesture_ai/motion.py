"""Temporal smoothing and motion derivatives.

``HandFeatures`` is purely per-frame: it knows where the hand *is*, never how it
is moving or how noisy the estimate was. This module adds the time axis.

Two jobs, both needed for a puppet that reads as alive:

*Smoothing.* Raw landmarks jitter by a degree or two even on a motionless hand,
and the tracker produces about 30 poses per second against a 60 fps render. A
One Euro filter fixes both -- it denoises hard when the hand is nearly still and
gets out of the way when it moves fast, which is exactly the tradeoff a fixed
exponential average gets wrong in one direction or the other.

*Derivatives.* Speed of movement is an input in its own right: it drives lean,
overshoot, and glow, so the puppet looks like it is reacting rather than merely
being posed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from gesture_ai.features import HandFeatures, orthonormalize

# Defaults tuned for hand tracking. Raising min_cutoff reduces lag and adds
# jitter; raising beta makes the filter yield sooner to fast movement.
DEFAULT_MIN_CUTOFF = 1.2
DEFAULT_BETA = 0.02
DEFAULT_D_CUTOFF = 1.0

_EPSILON = 1e-9


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * max(cutoff, _EPSILON))
    return 1.0 / (1.0 + tau / max(dt, _EPSILON))


class OneEuroFilter:
    """Adaptive low-pass filter: smooth when slow, responsive when fast.

    See Casiez, Roussel and Vogel (2012). The cutoff frequency rises with the
    signal's own rate of change, so a still hand is filtered heavily while a
    fast gesture passes through almost untouched.
    """

    def __init__(
        self,
        min_cutoff: float = DEFAULT_MIN_CUTOFF,
        beta: float = DEFAULT_BETA,
        d_cutoff: float = DEFAULT_D_CUTOFF,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x: float | None = None
        self._dx = 0.0

    def __call__(self, value: float, dt: float) -> float:
        if self._x is None or dt <= 0.0:
            self._x = value
            return value

        derivative = (value - self._x) / dt
        a_d = _alpha(self.d_cutoff, dt)
        self._dx += a_d * (derivative - self._dx)

        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a = _alpha(cutoff, dt)
        self._x += a * (value - self._x)
        return self._x

    @property
    def value(self) -> float | None:
        return self._x

    @property
    def speed(self) -> float:
        """Filtered rate of change, in units per second."""
        return self._dx

    def reset(self) -> None:
        self._x = None
        self._dx = 0.0


class _FilterBank:
    """One OneEuroFilter per element of a fixed-length vector."""

    def __init__(self, size: int, min_cutoff: float, beta: float) -> None:
        self._filters = [
            OneEuroFilter(min_cutoff, beta) for _ in range(size)
        ]

    def __call__(self, values, dt: float) -> tuple[float, ...]:
        return tuple(f(v, dt) for f, v in zip(self._filters, values))

    def speeds(self) -> tuple[float, ...]:
        return tuple(f.speed for f in self._filters)

    def reset(self) -> None:
        for f in self._filters:
            f.reset()


@dataclass
class MotionState:
    """How the hand is moving, as opposed to where it is."""

    wrist_speed: float = 0.0
    """Wrist travel in normalized screen units per second."""

    turn_speed: float = 0.0
    """Palm angular speed in degrees per second."""

    energy: float = 0.0
    """Overall movement intensity, 0..1, smoothed for display use."""

    direction: tuple[float, float] = (0.0, 0.0)
    """Unit vector of wrist travel; the direction to lean into."""


@dataclass
class SmoothedPose:
    """A filtered pose plus its motion, at render time rather than capture time."""

    present: float
    """Blend weight 0..1, eased. Not a boolean: it crossfades to idle."""

    wrist: tuple[float, ...] = (0.0, 0.0, 0.0)
    palm: tuple[float, ...] = (0.0, 0.0, 0.0)
    curl: tuple[float, ...] = (0.0,) * 5
    pinch: float = 0.0
    spread: float = 0.0
    screen: tuple[float, ...] = ()
    pose: tuple[float, ...] = ()
    basis: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    """Smoothed palm rotation, 9 floats. Gimbal-lock free, unlike ``palm``."""
    motion: MotionState = field(default_factory=MotionState)


# Seconds to cross-fade between tracked and idle. Long enough that a one-frame
# detection dropout does not visibly twitch the puppet.
PRESENCE_FADE = 0.4

# Normalizers turning raw rates into a 0..1 energy value. Set from observed
# rates: ordinary gesturing turns the palm far faster than first assumed, and
# under-scaling these pinned energy at 1.0, which left the glow and lean
# permanently maxed out and killed the whole point of a motion signal.
_SPEED_FULL = 2.5      # normalized screen units per second
_TURN_FULL = 600.0     # degrees per second


class PoseSmoother:
    """Turns the tracker's ~30 Hz pose stream into a smooth 60 Hz pose.

    Ticked once per rendered frame with the newest available features, so it
    both denoises and interpolates. Feeding it the same features twice is
    harmless -- it simply continues easing toward them.
    """

    def __init__(
        self,
        min_cutoff: float = DEFAULT_MIN_CUTOFF,
        beta: float = DEFAULT_BETA,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self._wrist = _FilterBank(3, min_cutoff, beta)
        self._palm = _FilterBank(3, min_cutoff, beta)
        self._curl = _FilterBank(5, min_cutoff, beta)
        self._pinch = OneEuroFilter(min_cutoff, beta)
        self._spread = OneEuroFilter(min_cutoff, beta)
        self._screen = _FilterBank(42, min_cutoff, beta)
        self._basis = _FilterBank(9, min_cutoff, beta)
        self._pose = _FilterBank(63, min_cutoff, beta)
        self._presence = 0.0
        self._energy = 0.0
        self._last: SmoothedPose = SmoothedPose(present=0.0)

    def set_tuning(self, min_cutoff: float, beta: float) -> None:
        """Retune every filter at once, for live adjustment while running."""
        self.min_cutoff = max(0.05, min_cutoff)
        self.beta = max(0.0, beta)
        for bank in (self._wrist, self._palm, self._curl, self._screen,
                     self._pose, self._basis):
            for f in bank._filters:
                f.min_cutoff, f.beta = self.min_cutoff, self.beta
        for f in (self._pinch, self._spread):
            f.min_cutoff, f.beta = self.min_cutoff, self.beta

    def update(
        self,
        features: HandFeatures | None,
        present: bool,
        dt: float,
    ) -> SmoothedPose:
        target = 1.0 if present else 0.0
        step = dt / PRESENCE_FADE if PRESENCE_FADE > 0 else 1.0
        if self._presence < target:
            self._presence = min(target, self._presence + step)
        else:
            self._presence = max(target, self._presence - step)

        if features is None:
            self._last = SmoothedPose(
                present=self._presence,
                wrist=self._last.wrist,
                palm=self._last.palm,
                curl=self._last.curl,
                pinch=self._last.pinch,
                spread=self._last.spread,
                screen=self._last.screen,
                pose=self._last.pose,
                basis=self._last.basis,
                motion=self._last.motion,
            )
            return self._last

        wrist = self._wrist(features.wrist, dt)
        palm = self._palm(features.palm, dt)
        curl = self._curl(features.curl, dt)
        pinch = self._pinch(features.pinch, dt)
        spread = self._spread(features.spread, dt)
        screen = self._screen(features.screen, dt)
        pose = self._pose(features.pose, dt)

        # Filtering the nine components independently leaves something that is
        # nearly but not exactly a rotation, which would shear the mesh, so
        # re-orthonormalize before anyone renders with it.
        raw = self._basis(features.basis, dt)
        basis = orthonormalize((raw[0:3], raw[3:6], raw[6:9]))
        basis_flat = tuple(v for row in basis for v in row)

        # Derivatives come from the filters themselves, which already track rate
        # of change -- differencing the smoothed output again would only add lag.
        vx, vy, _ = self._wrist.speeds()
        wrist_speed = math.hypot(vx, vy)
        turn_speed = math.sqrt(sum(s * s for s in self._palm.speeds()))

        # Weighted rather than summed: either kind of movement alone should be
        # able to reach full energy, but neither should saturate on its own
        # noise floor.
        raw_energy = min(
            1.0,
            0.65 * (wrist_speed / _SPEED_FULL) + 0.5 * (turn_speed / _TURN_FULL),
        )
        # Energy drives glow and lean, where a snappy attack and slow decay
        # looks far better than a symmetric filter.
        rate = 12.0 if raw_energy > self._energy else 3.0
        self._energy += (raw_energy - self._energy) * min(1.0, rate * dt)

        length = math.hypot(vx, vy)
        direction = (vx / length, vy / length) if length > _EPSILON else (0.0, 0.0)

        self._last = SmoothedPose(
            present=self._presence,
            wrist=wrist,
            palm=palm,
            curl=curl,
            pinch=pinch,
            spread=spread,
            screen=screen,
            pose=pose,
            basis=basis_flat,
            motion=MotionState(
                wrist_speed=wrist_speed,
                turn_speed=turn_speed,
                energy=self._energy,
                direction=direction,
            ),
        )
        return self._last
