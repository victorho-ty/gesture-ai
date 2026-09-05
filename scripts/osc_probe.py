"""Stand-in for TouchDesigner: print what gesture-ai is sending over OSC.

Proves out the data layer before any TouchDesigner network exists, so that a
later misalignment in TD is unambiguously a TD problem.

    uv run python scripts/osc_probe.py

Then, in another terminal:

    uv run gesture-ai --osc
"""

from __future__ import annotations

import argparse
import time

from pythonosc import dispatcher as osc_dispatcher
from pythonosc import osc_server

_INTERESTING = ("/hand/present", "/hand/wrist", "/hand/palm", "/hand/curl",
                "/hand/pinch", "/hand/spread")


class Probe:
    def __init__(self) -> None:
        self.values: dict[str, tuple] = {}
        self.bundles = 0
        self.present = 0
        self.started = time.perf_counter()
        self.last_print = 0.0

    def record(self, address: str, *args) -> None:
        self.values[address] = args
        if address == "/hand/present":
            self.bundles += 1
            self.present += int(args[0])
            self._maybe_print()

    def _maybe_print(self) -> None:
        now = time.perf_counter()
        if now - self.last_print < 0.25:
            return
        self.last_print = now

        rate = self.bundles / max(now - self.started, 1e-6)
        present = self.values.get("/hand/present", (0,))[0]
        wrist = self.values.get("/hand/wrist", (0.0, 0.0, 0.0))
        palm = self.values.get("/hand/palm", (0.0, 0.0, 0.0))
        curl = self.values.get("/hand/curl", (0.0,) * 5)
        pinch = self.values.get("/hand/pinch", (0.0,))[0]
        spread = self.values.get("/hand/spread", (0.0,))[0]
        pose = len(self.values.get("/hand/pose", ()))
        screen = len(self.values.get("/hand/screen", ()))

        print(
            f"\r{rate:5.1f}/s  present={present}  "
            f"wrist=({wrist[0]:+.2f},{wrist[1]:+.2f},{wrist[2]:.2f})  "
            f"palm=({palm[0]:+7.1f},{palm[1]:+7.1f},{palm[2]:+7.1f})  "
            f"curl=[{' '.join(f'{c:.2f}' for c in curl)}]  "
            f"pinch={pinch:.2f} spread={spread:.2f}  "
            f"pose={pose} screen={screen}   ",
            end="",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7000)
    args = parser.parse_args()

    probe = Probe()
    dispatcher = osc_dispatcher.Dispatcher()
    dispatcher.set_default_handler(probe.record)

    server = osc_server.BlockingOSCUDPServer((args.host, args.port), dispatcher)
    print(f"Listening for gesture-ai OSC on {args.host}:{args.port} (ctrl-c to stop)")
    print("Expect ~30 bundles/sec, pose=63, screen=42, and present flipping to 0")
    print("when the hand leaves frame while the pose values hold their last value.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
