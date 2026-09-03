"""Live MediaPipe hand-landmark webcam demo."""


def main() -> None:
    """Console entry point; imports the application lazily."""
    from gesture_ai.app import main as app_main

    raise SystemExit(app_main())
