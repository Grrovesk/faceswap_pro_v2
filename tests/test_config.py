"""Tests for the LipsyncJob config dataclass."""
import sys, os, tempfile, pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from faceswap.config import LatentSyncKnobs, LipsyncJob, VoiceSwap


def test_knobs_defaults():
    k = LatentSyncKnobs()
    assert k.inference_steps == 20
    assert k.guidance_scale == 1.5
    assert k.enable_deepcache is True
    assert k.seed == -1


def test_shape_predicates():
    p1 = Path(tempfile.NamedTemporaryFile(suffix=".mp4").name)
    p2 = Path(tempfile.NamedTemporaryFile(suffix=".mp4").name)
    a = Path(tempfile.NamedTemporaryFile(suffix=".wav").name)
    j1 = LipsyncJob(face_paths=[p1], audio_path=a)
    j2 = LipsyncJob(face_paths=[p1, p2], audio_path=a)
    assert j1.is_single_clip and not j1.is_multi_clip
    assert j2.is_multi_clip and not j2.is_single_clip


def test_validate_missing_face():
    a = Path(tempfile.NamedTemporaryFile(suffix=".wav").name)
    j = LipsyncJob(face_paths=[Path("/no/such/file.mp4")], audio_path=a)
    with pytest.raises(FileNotFoundError):
        j.validate()


def test_validate_empty_faces():
    a = Path(tempfile.NamedTemporaryFile(suffix=".wav").name)
    j = LipsyncJob(face_paths=[], audio_path=a)
    with pytest.raises(ValueError):
        j.validate()
