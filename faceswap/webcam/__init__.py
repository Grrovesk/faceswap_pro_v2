"""v2 webcam tab -- live face-swap MJPEG stream.

Self-contained, no imports from ui.app or ui.stream_server. All
heavy ML routes through core.pipeline which v2 already uses for the
Face Swap tab.
"""
from . import state  # noqa: F401
