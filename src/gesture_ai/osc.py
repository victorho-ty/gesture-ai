"""OSC emission of hand parameters, for TouchDesigner and similar receivers.

One UDP bundle per frame, addressed so that a TouchDesigner ``OSC In CHOP``
turns it into named channels with no parsing script. The whole bundle is about
900 bytes, comfortably inside the 1472-byte UDP payload limit, so each frame is
a single unfragmented packet.

The emitter deliberately mirrors ``JsonlRecorder`` -- ``write`` then ``close`` --
so it drops into the same call site and the same teardown.
"""

from __future__ import annotations

from pythonosc import osc_bundle_builder, osc_message_builder, udp_client

from gesture_ai.features import EulerUnwrapper, HandFeatures, hand_features

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7000

_ARG_INT = osc_message_builder.OscMessageBuilder.ARG_TYPE_INT
_ARG_FLOAT = osc_message_builder.OscMessageBuilder.ARG_TYPE_FLOAT


def _message(address: str, values, arg_type: str):
    builder = osc_message_builder.OscMessageBuilder(address=address)
    for value in values:
        builder.add_arg(value, arg_type)
    return builder.build()


class OscEmitter:
    """Sends one bundle of hand parameters per frame over UDP."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
        self._host = host
        self._port = port
        self._client = udp_client.SimpleUDPClient(host, port)
        self._unwrapper = EulerUnwrapper()
        self._last: HandFeatures | None = None

    @property
    def target(self) -> str:
        return f"{self._host}:{self._port}"

    def write(self, result, timestamp_ms: int) -> bool:
        """Emit one frame. Returns whether a hand was present.

        Called on *every* frame, including frames with no hand. That matters:
        ``hand_features`` returns ``None`` when nothing is detected, and if those
        frames emitted nothing the stream would simply fall silent and leave the
        receiver holding a stale pose forever. Instead the last known pose is
        re-sent with ``/hand/present 0``, which gives the receiver an explicit
        edge to blend away from.
        """
        features = hand_features(result, self._unwrapper)
        present = features is not None

        if present:
            self._last = features
        else:
            # A hand that reappears elsewhere should not inherit the old
            # rotation winding, which would smear a spin across the gap.
            self._unwrapper.reset()
            features = self._last

        bundle = osc_bundle_builder.OscBundleBuilder(
            osc_bundle_builder.IMMEDIATELY
        )
        bundle.add_content(
            _message("/hand/present", (1 if present else 0,), _ARG_INT)
        )
        bundle.add_content(
            _message("/hand/frame", (int(timestamp_ms),), _ARG_INT)
        )

        if features is not None:
            bundle.add_content(
                _message("/hand/handedness", (features.handedness,), _ARG_FLOAT)
            )
            bundle.add_content(_message("/hand/wrist", features.wrist, _ARG_FLOAT))
            bundle.add_content(_message("/hand/palm", features.palm, _ARG_FLOAT))
            bundle.add_content(_message("/hand/curl", features.curl, _ARG_FLOAT))
            bundle.add_content(
                _message("/hand/pinch", (features.pinch,), _ARG_FLOAT)
            )
            bundle.add_content(
                _message("/hand/spread", (features.spread,), _ARG_FLOAT)
            )
            bundle.add_content(_message("/hand/pose", features.pose, _ARG_FLOAT))
            bundle.add_content(
                _message("/hand/screen", features.screen, _ARG_FLOAT)
            )

        self._client.send(bundle.build())
        return present

    def close(self) -> None:
        self._last = None
        self._unwrapper.reset()
