"""Spout texture sharing, for compositing the camera feed inside TouchDesigner.

Spout hands a GPU texture to another Windows application with no encode step, so
TouchDesigner can composite a robot over the live camera without either process
opening the webcam twice -- Windows generally refuses the second opener.

The awkward part: Spout is an OpenGL mechanism and needs a current GL context,
which a plain OpenCV application does not have. So this module creates a hidden
GLFW window whose only job is to own that context.
"""

from __future__ import annotations

# GL_RGBA, spelled out rather than imported so this module needs no GL binding.
_GL_RGBA = 0x1908


class SpoutUnavailable(RuntimeError):
    """Raised when the optional Spout dependencies are missing or unusable."""


class SpoutSender:
    """Publishes BGR frames as a named Spout sender."""

    def __init__(self, name: str, width: int, height: int) -> None:
        try:
            import glfw
            import SpoutGL
        except ImportError as error:
            raise SpoutUnavailable(
                "Spout output needs the 'SpoutGL' and 'glfw' packages; "
                "install them or run without --spout."
            ) from error

        try:
            import cv2
        except ImportError as error:  # pragma: no cover - cv2 is a core dep
            raise SpoutUnavailable("Spout output needs OpenCV.") from error

        self._cv2 = cv2
        self._glfw = glfw
        self._name = name
        self._window = None
        self._sender = None

        if not glfw.init():
            raise SpoutUnavailable("Could not initialise GLFW for the GL context.")

        # Never shown; it exists purely so Spout has a current GL context.
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self._window = glfw.create_window(width, height, f"{name} (spout)", None, None)
        if not self._window:
            glfw.terminate()
            raise SpoutUnavailable("Could not create a hidden GLFW window.")
        glfw.make_context_current(self._window)

        self._sender = SpoutGL.SpoutSender()
        self._sender.setSenderName(name)

    @property
    def name(self) -> str:
        return self._name

    def send(self, frame) -> None:
        """Publish one BGR frame.

        Send the *clean* camera frame, before ``draw_overlay`` runs -- the
        receiver wants video to composite against, not the debug skeleton.
        """
        if self._sender is None:
            return
        height, width = frame.shape[:2]
        rgba = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGBA)
        # OpenCV rows run top-down and OpenGL expects bottom-up, so invert.
        self._sender.sendImage(
            rgba.tobytes(), width, height, _GL_RGBA, True, 0
        )

    def close(self) -> None:
        if self._sender is not None:
            try:
                self._sender.releaseSender()
            except Exception:
                pass
            self._sender = None
        if self._window is not None:
            self._glfw.destroy_window(self._window)
            self._window = None
            self._glfw.terminate()
