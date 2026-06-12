"""Rotoscoping tab (Phase 1 MVP).

End-to-end flow: video upload -> frame extraction cache -> SAM2 daemon
click capture -> propagate -> mask overlay preview -> send to Lipsync tab.

Modules
-------
ui : ``build_tab()`` -- the Gradio tab definition.  Called from
     ``faceswap/ui.py`` next to the other tab builders.
"""
from .ui import build_tab

__all__ = ["build_tab"]
