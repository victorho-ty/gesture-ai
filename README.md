# gesture-ai

A lean OpenCV desktop demo that runs Google's pretrained
[MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
on a live webcam, draws the 21 hand landmarks and their skeleton over the video,
shows one selectable joint's normalized coordinate, predicts and displays how
many fingers are raised (`0..5`), and can record every detected hand to a local
JSON Lines file.

Scope is deliberately small: one camera, one hand, no video-file input, no
gesture classification, no GUI framework, no web streaming, no training.

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
