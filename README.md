# gesture-ai

A lean OpenCV desktop demo that runs Google's pretrained
[MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
on a live webcam, draws the 21 hand landmarks and their skeleton over the video,
shows one selectable joint's normalized coordinate, predicts and displays how
many fingers are raised (`0..5`), and can record every detected hand to a local
JSON Lines file.

It also ships a second application, `gesture-ai-puppet`, which renders a 3D
robot that mimics your hand in real time -- entirely in-process, with no other
software required. See [The puppet](#the-puppet).

For driving an external renderer instead, it can stream hand parameters over OSC
and share the camera over Spout. See
[Streaming to TouchDesigner](#streaming-to-touchdesigner).

Scope is deliberately small: one camera, one hand, no video-file input, no
gesture classification, no GUI framework, no training.

## Setup

```bash
uv sync
```

The pretrained model is downloaded automatically on first run into
`models/hand_landmarker.task`. To fetch it ahead of time (or to verify network
access) without opening the camera:

```bash
uv run gesture-ai --download-model
```

Then start the app:

```bash
uv run gesture-ai
```

## Controls

| Key | Action |
| --- | --- |
| `r` | Toggle JSON Lines recording |
| `o` | Toggle OSC streaming |
| `m` | Toggle horizontal mirroring of the preview |
| `[` | Select the previous landmark (wraps at 0) |
| `]` | Select the next landmark (wraps at 20) |
| `q` or `Esc` | Quit |

The status line at the top of the window shows the measured FPS, `REC` while
recording or `READY` otherwise, and the control hints.

## Command-line options

```bash
uv run gesture-ai --camera 1 --width 960 --height 540 --fps 24 --joint 4
```

| Flag | Default | Notes |
| --- | --- | --- |
| `--camera` | `0` | OpenCV camera index |
| `--width`, `--height` | `1280`, `720` | Requested capture size (the driver may pick another) |
| `--fps` | `30.0` | Requested camera FPS |
| `--model` | `models/hand_landmarker.task` | Model file override |
| `--output` | `captures/hand_landmarks.jsonl` | JSONL output override |
| `--joint` | `8` | Landmark shown on screen; `0..20`, default is the index fingertip |
| `--mirror` / `--no-mirror` | mirrored | Initial mirror state |
| `--detection-confidence` | `0.5` | Minimum detection confidence, `[0, 1]` |
| `--presence-confidence` | `0.5` | Minimum presence confidence, `[0, 1]` |
| `--tracking-confidence` | `0.5` | Minimum tracking confidence, `[0, 1]` |
| `--osc` / `--no-osc` | off | Stream hand parameters over OSC |
| `--osc-host` | `127.0.0.1` | OSC destination host |
| `--osc-port` | `7000` | OSC destination port |
| `--spout` / `--no-spout` | off | Share the camera frame as a Spout sender (Windows) |
| `--spout-name` | `gesture-ai` | Spout sender name |
| `--download-model` | off | Ensure the model exists, print its path, exit |

Landmark indices follow the canonical MediaPipe order: `0 WRIST`, `1..4` thumb
(`CMC`, `MCP`, `IP`, `TIP`), `5..8` index finger, `9..12` middle finger,
`13..16` ring finger, `17..20` pinky.

## Finger counting

While a hand is detected, the overlay shows `Fingers up: N`. The index, middle,
ring, and pinky fingers count as raised when their tip is farther from the
wrist than their PIP joint. The thumb folds across the palm rather than toward
the wrist, so it is counted as raised when its tip is farther from the pinky
MCP (the opposite side of the hand) than its IP joint is. Both checks are
orientation independent, so they work however the hand is turned. The result
ranges from `0` (fist) to `5` (open palm).

## Recording format

Recording writes one JSON object per line to `captures/hand_landmarks.jsonl`
(appending across sessions). The file is opened on the first written record and
flushed after every line, so it can be tailed live.

```json
{
  "schema_version": 1,
  "timestamp_ms": 1234,
  "image_size": {"width": 1280, "height": 720},
  "hands": [
    {
      "handedness": {"label": "Right", "score": 0.95},
      "normalized_landmarks": [{"x": 0.1, "y": 0.2, "z": -0.3}],
      "world_landmarks": [{"x": 0.01, "y": 0.02, "z": -0.03}]
    }
  ]
}
```

Semantics:

- `timestamp_ms` is a monotonic, strictly increasing offset from application
  start, not a wall-clock time.
- `image_size` describes the frame the landmarks were computed on, after
  mirroring.
- Frames with no detected hand are intentionally omitted, so line count is not
  frame count and timestamps have gaps.
- Full float precision is preserved; the three-decimal values on screen are for
  display only.
- The on-screen overlay shows one joint at a time because 21 coordinate
  triplets are unreadable at webcam resolution. The JSONL file is the complete
  data source.

### Normalized vs. world landmarks

- `normalized_landmarks` are image-space coordinates: `x` and `y` in roughly
  `[0, 1]` relative to frame width and height, and `z` an approximate depth
  relative to the wrist, in the same scale as `x`. Values can fall slightly
  outside `[0, 1]` when the model extrapolates past the frame edge.
- `world_landmarks` are an estimated 3D hand pose in metres, with the origin at
  the hand's approximate geometric centre. They are a model estimate, not a
  camera-calibrated measurement: absolute position, scale, and orientation
  relative to the real camera are not recoverable from them.

## The puppet

```bash
uv run gesture-ai-puppet
```

A humanoid mecha, built from primitives, driven live by your hand over a dimmed
camera backdrop. Renders at 60 fps on integrated graphics.

| Key | Action |
| --- | --- |
| `h` | Overlay the tracked hand skeleton (also `--show-hand`) |
| `b` | Toggle the camera backdrop |
| `g` | Toggle the ground grid |
| `m` | Toggle mirroring |
| `[` `]` | Filter cutoff -- lower is smoother, higher is more responsive |
| `;` `'` | Filter speed coefficient |
| `Esc` | Quit |

### How your hand drives it

| Robot | Hand |
| --- | --- |
| Position on screen | Wrist position |
| Distance from camera | Apparent hand size |
| Torso orientation | Palm orientation |
| Right arm | Index finger -- extend to raise |
| Left arm | Pinky finger |
| Elbows | Middle finger |
| Legs | Ring finger |
| Head tilt | Thumb |
| Claws | Thumb-to-index pinch |
| Stance width | Finger splay |
| Lean and glow | Speed of movement |

A hand landmarker reports no elbow and no shoulder, so this is a *puppet*
mapping rather than an anatomical one -- which finger drives which limb is a
choice, not a measurement. Every constant lives in `RigMapping`
(`src/gesture_ai/rig.py`) so it can be retuned in one place.

### Why it stays smooth

MediaPipe manages about 30 poses per second on CPU; the display wants 60. So
tracking runs on its own thread and publishes into a **single slot holding only
the newest pose** (`src/gesture_ai/tracker.py`). A queue would grow a backlog
under load and the puppet would drift behind your hand; dropping stale poses
keeps latency flat. This works because `VideoCapture.read()` and MediaPipe
inference both release the GIL.

A [One Euro filter](https://gery.casiez.net/1euro/) then does double duty
(`src/gesture_ai/motion.py`): it smooths hard when the hand is nearly still and
gets out of the way when it moves fast, and being time-aware, it interpolates
the 30 Hz pose stream up to 60 Hz.

Measured on an i9-11900H with Intel UHD integrated graphics: **60.0 fps render,
worst frame 17.7 ms, ~29 poses/sec, 8-13 ms inference.**

### Rotation without gimbal lock

The torso is driven by the palm's **rotation matrix**, not by euler angles.
Euler decomposition puts yaw on the middle axis, so it gimbal-locks whenever the
palm turns edge-on -- which happens constantly -- and the puppet flips violently
for no real movement. `HandFeatures.basis` carries the raw frame for this
reason; `HandFeatures.palm` keeps the euler angles for display and OSC, where
the degeneracy does not matter.

Verify the whole pipeline without a camera or a window:

```bash
uv run python scripts/selfcheck.py
```

## Streaming to TouchDesigner

Only needed if you want an external renderer; `gesture-ai-puppet` needs none of
this.

```bash
uv sync --extra spout
uv run gesture-ai --osc --spout
```

Two one-way channels on localhost: hand parameters over OSC/UDP, and the camera
image over Spout. Keeping MediaPipe in its own process matters -- inference costs
15-25 ms per frame, so running it inside TouchDesigner would block the cook
thread and drag a 60 fps render down to the tracker's rate.

Check the stream without opening TouchDesigner:

```bash
uv run python scripts/osc_probe.py
```

### OSC addresses

One bundle per frame, about 840 bytes, so each frame is a single unfragmented
UDP packet. A TouchDesigner `OSC In CHOP` on port 7000 turns these into named
channels with no parsing script.

| Address | Payload | Meaning |
| --- | --- | --- |
| `/hand/present` | `int` | `1` while a hand is tracked, `0` otherwise |
| `/hand/frame` | `int` | Milliseconds since the app started |
| `/hand/handedness` | `float` | Signed confidence: negative left, positive right |
| `/hand/wrist` | 3 floats | Normalized screen x, y, and apparent hand scale |
| `/hand/palm` | 3 floats | Palm pitch, yaw, roll in degrees |
| `/hand/curl` | 5 floats | Per-finger curl, thumb first, `0` straight to `1` fisted |
| `/hand/pinch` | `float` | Thumb-to-index gap, `0` touching to `1` wide open |
| `/hand/spread` | `float` | Finger splay, `0` together to `1` fanned |
| `/hand/pose` | 63 floats | World landmarks, flattened xyz |
| `/hand/screen` | 42 floats | Normalized landmarks, flattened xy |

Every frame is sent, including frames with no hand: those repeat the last known
pose with `/hand/present 0`. If they sent nothing instead, the stream would fall
silent and leave the receiver holding a stale pose with no way to know tracking
had stopped.

Prefer `/hand/palm`, `/hand/curl`, `/hand/pinch` and `/hand/spread` over the raw
`/hand/pose` array for driving a rig. They are already scale- and
distance-invariant, and there are a dozen of them rather than 63.

### Notes for the receiving end

- MediaPipe world landmarks are **Y-down**; TouchDesigner is **Y-up**, and the Z
  sign differs. Negate Y and Z in a `Math CHOP`. This is the usual cause of a
  puppet that moves upside down or inverted in depth.
- Smooth on the receiving side (`Lag CHOP` or `Filter CHOP`) rather than here, so
  it can be tuned live. This also interpolates the ~30 fps stream up to 60 fps.
- Gate on `/hand/present` and cross-fade to an idle animation, or the puppet
  freezes mid-pose whenever the hand leaves frame.
- The frame is mirrored *before* inference, so landmarks already sit in mirrored
  space and align with the Spout texture. It also means the handedness label
  refers to the mirrored image, not the anatomical hand.
- Do not also open the webcam with a `Video Device In TOP`. Windows generally
  refuses the second opener, which is why the frame is shared over Spout.

## Getting good results

- Use even, front-facing lighting; strong backlight silhouettes the hand and
  detection confidence drops.
- Keep all fingers visible and unoccluded — a hand edge-on to the camera hides
  joints the model then has to guess.
- Move slowly enough to avoid motion blur; blur hurts tracking more than low
  resolution does.
- Sit at a practical distance so the hand fills a useful part of the frame
  without leaving it, roughly 0.3–1 m for a typical laptop webcam.
- If the preview stutters, request a lower resolution and FPS
  (`--width 640 --height 480 --fps 15`); the model runs on CPU.

## Privacy

- The webcam feed, the model file, and every capture stay on this machine. The
  only network access is the one-time model download from Google's storage
  host.
- Hand-landmark recordings are body-measurement data and can be biometric-like:
  hand geometry is fairly distinctive, and captures may also carry context from
  when and how long you recorded. Treat `captures/` accordingly before sharing.
- `models/` and `captures/` are Git ignored, so downloaded assets and
  recordings are not committed by accident.
