"""Real-time hand-driven puppet, rendered in-process with raylib.

Runs the render loop on the main thread at display rate while ``HandTracker``
produces poses on its own thread, so the puppet stays at 60 fps even though
MediaPipe only manages about 30.

The hand skeleton stays available on a key toggle: when the puppet looks wrong
it answers whether the problem is in tracking or in the mapping.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyray as rl

from gesture_ai.landmarks import HAND_CONNECTIONS
from gesture_ai.model import ensure_model
from gesture_ai.motion import PoseSmoother
from gesture_ai.puppet import PuppetRenderer, ground_shadow
from gesture_ai.rig import RigMapping, solve
from gesture_ai.tracker import HandTracker

DEFAULT_MODEL_PATH = Path("models") / "hand_landmarker.task"
WINDOW_TITLE = "gesture-ai puppet"

# Camera warm-up and shader compilation dominate the first seconds and say
# nothing about steady-state performance.
BENCH_WARMUP = 3.0

# Hand overlay, matching the original OpenCV preview in render.py: thin lines,
# small filled joints. Orange is the exact joint colour from there, BGR
# (0, 128, 255). The lines there were green; white reads better over a dimmed
# camera backdrop, and is what was asked for.
_HAND_LINE = rl.Color(236, 240, 248, 235)
_HAND_DOT = rl.Color(255, 128, 0, 255)
_HAND_LINE_WIDTH = 1.6
_HAND_DOT_RADIUS = 3.0

_BG = rl.Color(16, 17, 23, 255)
_GRID = rl.Color(40, 42, 54, 255)
_TEXT = rl.Color(225, 228, 240, 255)
_DIM = rl.Color(130, 135, 155, 255)
_GOOD = rl.Color(120, 220, 140, 255)
_BAD = rl.Color(230, 110, 110, 255)
_SHADOW = rl.Color(0, 0, 0, 90)


def cover_fit(src_w: int, src_h: int, dst_w: int, dst_h: int):
    """Scale and offset that fill a window with a source image, cropping to fit.

    Returned as a function mapping normalized 0..1 source coordinates to screen
    pixels. The backdrop and the landmark overlay must share this, or the dots
    drift off the fingers wherever the aspect ratios differ.
    """
    scale = max(dst_w / src_w, dst_h / src_h)
    off_x = (src_w - dst_w / scale) * 0.5
    off_y = (src_h - dst_h / scale) * 0.5

    def to_screen(nx: float, ny: float) -> tuple[float, float]:
        return (nx * src_w - off_x) * scale, (ny * src_h - off_y) * scale

    return to_screen


def draw_hand_overlay(screen, to_screen) -> None:
    """Draw the tracked hand the way the OpenCV preview did.

    Screen space rather than the 3D scene: these are normalized image
    coordinates, so drawing them flat puts them exactly on the hand in the
    backdrop, and 2D primitives give clean thin strokes where ``draw_line_3d``
    and low-segment spheres looked ragged.
    """
    if len(screen) != 42:
        return
    points = [to_screen(screen[i * 2], screen[i * 2 + 1]) for i in range(21)]

    for start, end in HAND_CONNECTIONS:
        a, b = points[start], points[end]
        rl.draw_line_ex(
            rl.Vector2(a[0], a[1]), rl.Vector2(b[0], b[1]),
            _HAND_LINE_WIDTH, _HAND_LINE,
        )
    for x, y in points:
        rl.draw_circle_v(rl.Vector2(x, y), _HAND_DOT_RADIUS, _HAND_DOT)


class Backdrop:
    """The camera feed, drawn behind the puppet.

    Everything runs in one process now, so the frame the tracker already
    captured goes straight to the GPU -- no second camera handle, and no
    inter-process texture sharing to arrange.
    """

    def __init__(self, width: int, height: int) -> None:
        blank = rl.gen_image_color(width, height, rl.Color(0, 0, 0, 255))
        self._texture = rl.load_texture_from_image(blank)
        rl.unload_image(blank)
        self._width = width
        self._height = height
        # Reused every frame; allocating a new RGBA buffer per frame would
        # churn several megabytes a second through the allocator.
        self._buffer = np.zeros((height, width, 4), dtype=np.uint8)

    def update(self, frame) -> None:
        """Upload one BGR frame."""
        if frame is None:
            return
        if frame.shape[0] != self._height or frame.shape[1] != self._width:
            frame = cv2.resize(frame, (self._width, self._height))
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA, dst=self._buffer)
        # Cast explicitly: the C parameter is a const void*, which the binding
        # cannot infer a type for on its own.
        rl.update_texture(
            self._texture, rl.ffi.cast("void *", rl.ffi.from_buffer(self._buffer))
        )

    def draw(self) -> None:
        screen_w = rl.get_screen_width()
        screen_h = rl.get_screen_height()
        # Cover the window while preserving aspect ratio: crop, never stretch.
        # draw_hand_overlay uses the same fit via cover_fit, so the landmarks
        # stay registered to the video underneath them.
        scale = max(screen_w / self._width, screen_h / self._height)
        crop_w = screen_w / scale
        crop_h = screen_h / scale
        source = rl.Rectangle(
            (self._width - crop_w) * 0.5, (self._height - crop_h) * 0.5,
            crop_w, crop_h,
        )
        dest = rl.Rectangle(0, 0, float(screen_w), float(screen_h))
        # Dimmed, so the puppet stays the subject instead of competing with a
        # brightly lit room.
        rl.draw_texture_pro(
            self._texture, source, dest, rl.Vector2(0, 0), 0.0,
            rl.Color(110, 110, 125, 255),
        )
        rl.draw_rectangle_gradient_v(
            0, 0, screen_w, screen_h,
            rl.Color(10, 10, 18, 70), rl.Color(6, 6, 12, 190),
        )

    def unload(self) -> None:
        rl.unload_texture(self._texture)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gesture-ai-puppet",
        description="Hand-driven puppet rendered with raylib.",
    )
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--width", type=int, default=1280, help="Window width")
    parser.add_argument("--height", type=int, default=720, help="Window height")
    parser.add_argument(
        "--capture-width", type=int, default=960, help="Camera capture width"
    )
    parser.add_argument(
        "--capture-height", type=int, default=540, help="Camera capture height"
    )
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL_PATH, help="Landmarker model"
    )
    parser.add_argument(
        "--fps", type=int, default=60, help="Render frame-rate cap; 0 uncaps"
    )
    parser.add_argument(
        "--mirror",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mirror the camera horizontally",
    )
    parser.add_argument(
        "--min-cutoff",
        type=float,
        default=1.2,
        help="One Euro minimum cutoff; lower is smoother but laggier",
    )
    parser.add_argument(
        "--benchmark",
        type=float,
        default=0.0,
        help="Run for N seconds, print timing statistics, and exit",
    )
    parser.add_argument(
        "--show-hand",
        action="store_true",
        help="Start with the tracked hand overlay visible (toggled with h)",
    )
    parser.add_argument(
        "--no-backdrop",
        action="store_true",
        help="Skip the camera backdrop; saves a texture upload per frame",
    )
    parser.add_argument(
        "--screenshot",
        type=str,
        default="",
        help="Write a PNG of the last benchmark frame to this path",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.02,
        help="One Euro speed coefficient; higher tracks fast motion harder",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        model_path = ensure_model(args.model)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1

    tracker = HandTracker(
        model_path=model_path,
        camera=args.camera,
        width=args.capture_width,
        height=args.capture_height,
        mirror=args.mirror,
        keep_frame=not args.no_backdrop,
    )
    try:
        tracker.start()
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1

    smoother = PoseSmoother(args.min_cutoff, args.beta)

    rl.set_trace_log_level(rl.TraceLogLevel.LOG_WARNING)
    rl.set_config_flags(rl.ConfigFlags.FLAG_MSAA_4X_HINT)
    rl.init_window(args.width, args.height, WINDOW_TITLE)
    if args.fps:
        rl.set_target_fps(args.fps)

    camera = rl.Camera3D(
        rl.Vector3(0.0, 3.8, 16.0),
        rl.Vector3(0.0, 2.9, 0.0),
        rl.Vector3(0.0, 1.0, 0.0),
        48.0,
        rl.CameraProjection.CAMERA_PERSPECTIVE,
    )
    puppet = PuppetRenderer()
    mapping = RigMapping()
    backdrop = None if args.no_backdrop else Backdrop(
        args.capture_width, args.capture_height
    )
    show_hand = args.show_hand
    show_grid = False
    show_backdrop = backdrop is not None

    last_seq = -1
    drawn_seq = -1
    stale_frames = 0
    previous = time.perf_counter()

    bench_start = previous
    seen_after_warmup: set[int] = set()
    frame_times: list[float] = []
    pose_ages: list[int] = []
    seen_seqs: set[int] = set()

    try:
        while not rl.window_should_close():
            if args.benchmark and time.perf_counter() - bench_start > args.benchmark + BENCH_WARMUP:
                break
            now = time.perf_counter()
            dt = min(now - previous, 0.1)
            previous = now

            latest = tracker.latest()
            if latest is not None:
                if latest.seq != last_seq:
                    last_seq = latest.seq
                    stale_frames = 0
                else:
                    stale_frames += 1
                pose = smoother.update(latest.features, latest.present, dt)
                seen_seqs.add(latest.seq)
            else:
                pose = smoother.update(None, False, dt)

            if args.benchmark and now - bench_start > BENCH_WARMUP:
                frame_times.append(dt)
                pose_ages.append(stale_frames)
                seen_after_warmup.add(latest.seq if latest else -1)

            joints = solve(pose, now - bench_start, mapping)

            if backdrop is not None and latest is not None and latest.seq != drawn_seq:
                backdrop.update(latest.frame)
                drawn_seq = latest.seq

            rl.begin_drawing()
            rl.clear_background(_BG)
            if backdrop is not None and show_backdrop:
                backdrop.draw()

            rl.begin_mode_3d(camera)
            if show_grid:
                rl.draw_grid(24, 1.0)
            centre, radius = ground_shadow(joints)
            rl.draw_cylinder_ex(
                centre,
                rl.Vector3(centre.x, centre.y + 0.01, centre.z),
                radius, radius, 20, _SHADOW,
            )
            puppet.draw(joints, camera)

            rl.end_mode_3d()

            if show_hand and pose.present > 0.01:
                draw_hand_overlay(
                    pose.screen,
                    cover_fit(
                        args.capture_width, args.capture_height,
                        rl.get_screen_width(), rl.get_screen_height(),
                    ),
                )

            _draw_hud(pose, tracker, stale_frames, smoother)
            rl.end_drawing()

            if args.screenshot and args.benchmark:
                remaining = args.benchmark + BENCH_WARMUP - (now - bench_start)
                if remaining < 0.05:
                    rl.take_screenshot(args.screenshot)

            if rl.is_key_pressed(rl.KeyboardKey.KEY_H):
                show_hand = not show_hand
            if rl.is_key_pressed(rl.KeyboardKey.KEY_B) and backdrop is not None:
                show_backdrop = not show_backdrop
            if rl.is_key_pressed(rl.KeyboardKey.KEY_G):
                show_grid = not show_grid
            if rl.is_key_pressed(rl.KeyboardKey.KEY_M):
                tracker.mirror = not tracker.mirror
            # Live filter tuning: the feel of the smoothing cannot be judged
            # from numbers, only by moving a hand while it changes.
            if rl.is_key_down(rl.KeyboardKey.KEY_LEFT_BRACKET):
                smoother.set_tuning(smoother.min_cutoff - 1.5 * dt, smoother.beta)
            if rl.is_key_down(rl.KeyboardKey.KEY_RIGHT_BRACKET):
                smoother.set_tuning(smoother.min_cutoff + 1.5 * dt, smoother.beta)
            if rl.is_key_down(rl.KeyboardKey.KEY_SEMICOLON):
                smoother.set_tuning(smoother.min_cutoff, smoother.beta - 0.05 * dt)
            if rl.is_key_down(rl.KeyboardKey.KEY_APOSTROPHE):
                smoother.set_tuning(smoother.min_cutoff, smoother.beta + 0.05 * dt)
    finally:
        if backdrop is not None:
            backdrop.unload()
        puppet.unload()
        rl.close_window()
        tracker.stop()

    if args.benchmark and frame_times:
        _report(frame_times, pose_ages, seen_after_warmup, tracker)

    return 0


def _report(frame_times, pose_ages, seen_seqs, tracker: HandTracker) -> None:
    import statistics

    total = sum(frame_times)
    fps = [1.0 / t for t in frame_times if t > 0]
    ms = sorted(t * 1000.0 for t in frame_times)
    p95 = ms[int(len(ms) * 0.95)]
    p99 = ms[int(len(ms) * 0.99)]

    print()
    print("== render ==")
    print(f"  frames            {len(frame_times)} in {total:.1f}s")
    print(f"  mean fps          {len(frame_times) / total:.1f}")
    print(f"  median frame      {statistics.median(ms):.2f} ms")
    print(f"  p95 / p99 frame   {p95:.2f} / {p99:.2f} ms")
    print(f"  worst frame       {ms[-1]:.2f} ms")
    print(f"  min fps observed  {min(fps):.1f}")
    print()
    print("== tracker ==")
    print(f"  measured fps      {tracker.fps:.1f}")
    print(f"  inference         {tracker.inference_ms:.1f} ms")
    print(f"  unique poses      {len(seen_seqs)}")
    print(f"  poses/sec         {len(seen_seqs) / total:.1f}")
    print()
    print("== handoff ==")
    print(f"  mean repeats/pose {statistics.mean(pose_ages):.2f}")
    print(f"  max repeats       {max(pose_ages)}")
    print(f"  (repeats = render frames showing an already-seen pose;")
    print(f"   ~1 expected at 60fps render against 30fps tracking)")


def _wrap(angle: float) -> float:
    """Fold an accumulated angle back into -180..180 for display."""
    return (angle + 180.0) % 360.0 - 180.0


def _draw_hud(pose, tracker: HandTracker, stale_frames: int, smoother) -> None:
    render_fps = rl.get_fps()
    tracked = pose.present > 0.5

    rl.draw_text(f"render {render_fps:3d} fps", 12, 12, 20, _TEXT)
    rl.draw_text(
        f"tracker {tracker.fps:5.1f} fps  ({tracker.inference_ms:4.1f} ms infer)",
        12, 36, 20, _DIM,
    )
    rl.draw_text(
        "TRACKING" if tracked else "NO HAND",
        12, 60, 20, _GOOD if tracked else _BAD,
    )
    rl.draw_text(f"repeat frames {stale_frames:3d}", 12, 84, 20, _DIM)

    motion = pose.motion
    rl.draw_text(
        f"speed {motion.wrist_speed:5.2f}/s   turn {motion.turn_speed:6.1f} deg/s"
        f"   energy {motion.energy:4.2f}",
        12, 112, 20, _TEXT,
    )
    rl.draw_text(
        # Wrapped for display: the unwrapper accumulates past 360 to keep the
        # signal continuous, which is right for filtering but unreadable here.
        f"palm  p{_wrap(pose.palm[0]):+7.1f}  y{_wrap(pose.palm[1]):+7.1f}"
        f"  r{_wrap(pose.palm[2]):+7.1f}",
        12, 136, 20, _DIM,
    )
    rl.draw_text(
        "curl  " + "  ".join(f"{c:.2f}" for c in pose.curl)
        + f"    pinch {pose.pinch:.2f}  spread {pose.spread:.2f}",
        12, 160, 20, _DIM,
    )
    rl.draw_text(
        f"filter  min_cutoff {smoother.min_cutoff:5.2f}   beta {smoother.beta:5.3f}",
        12, 184, 20, _DIM,
    )
    rl.draw_text(
        "h hand   b backdrop   g grid   m mirror   [ ] cutoff   ; ' beta   esc quit",
        12, rl.get_screen_height() - 28, 18, _DIM,
    )


if __name__ == "__main__":
    raise SystemExit(main())
