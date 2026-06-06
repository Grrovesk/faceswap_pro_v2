"""Unit tests for presets module."""
import sys, os, tempfile, shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from faceswap import presets as _p


def setup_function(_):
    # Use a temp PRESETS_DIR per test
    _p.PRESETS_DIR = Path(tempfile.mkdtemp()) / "presets"


def test_save_and_list():
    out = _p.save_preset(
        "my-test", isolate=True, quick=False, enhance=True,
        extend_single=False, ls_steps=25, ls_guidance=1.7,
        ls_deepcache=True, ls_seed=42,
        voice_model="", voice_transpose=0,
    )
    assert "my-test" in out
    assert "my-test" in _p.list_presets()


def test_load_round_trip():
    _p.save_preset(
        "alpha", isolate=False, quick=True, enhance=False,
        extend_single=True, ls_steps=30, ls_guidance=2.0,
        ls_deepcache=False, ls_seed=123,
        voice_model="some_model.pth", voice_transpose=-3,
    )
    loaded = _p.load_preset("alpha")
    assert loaded is not None
    assert loaded["isolate_vocals"] is False
    assert loaded["quick_test"] is True
    assert loaded["enhance_faces"] is False
    assert loaded["extend_single"] is True
    assert loaded["latentsync"]["inference_steps"] == 30
    assert loaded["latentsync"]["guidance_scale"] == 2.0
    assert loaded["latentsync"]["enable_deepcache"] is False
    assert loaded["latentsync"]["seed"] == 123
    assert loaded["voice_swap"]["model_basename"] == "some_model.pth"
    assert loaded["voice_swap"]["transpose_semitones"] == -3


def test_safe_name_strips_bad_chars():
    n = _p._safe_name("Hello/World:???")
    assert "/" not in n
    assert ":" not in n
    assert "?" not in n


def test_delete():
    _p.save_preset(
        "to-delete", isolate=True, quick=False, enhance=True,
        extend_single=False, ls_steps=20, ls_guidance=1.5,
        ls_deepcache=True, ls_seed=-1,
        voice_model="", voice_transpose=0,
    )
    assert "to-delete" in _p.list_presets()
    assert _p.delete_preset("to-delete") is True
    assert "to-delete" not in _p.list_presets()


def test_load_nonexistent_returns_none():
    assert _p.load_preset("nope") is None
