"""RestoreFormer++ backend (stub for T2-2 v1).

NOT YET INSTALLED in v2.  To enable:
  1. Clone https://github.com/wzhouxiff/RestoreFormerPlusPlus into
     v2/external_repos/RestoreFormerPlusPlus/
  2. Download RestoreFormer++ weights from the repo's releases
     into v2/models/face_restoration/restoreformer_plus.pth
  3. pip install the repo's requirements.txt into the active venv
  4. Re-run -- this backend will detect the install and switch from
     this stub to the full inference path.

Until then, calling this backend raises a clear runtime error so
the UI dropdown still surfaces the option as a documented future
feature without silently falling back to GFPGAN.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

_THIS = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_EXPECTED_REPO = PROJECT_ROOT / "external_repos" / "RestoreFormerPlusPlus"
_EXPECTED_WEIGHT = (PROJECT_ROOT / "models" / "face_restoration"
                     / "restoreformer_plus.pth")


def _is_installed() -> bool:
    return _EXPECTED_REPO.is_dir() and _EXPECTED_WEIGHT.is_file()


def enhance(video_path: Path,
             log: Callable[[str], None] = print,
             **kwargs) -> Path:
    if not _is_installed():
        msg = (
            "[restoreformer] NOT INSTALLED.  To enable RestoreFormer++:\n"
            f"  1. git clone https://github.com/wzhouxiff/"
            f"RestoreFormerPlusPlus.git "
            f"{_EXPECTED_REPO}\n"
            "  2. Download RestoreFormer++ weights into "
            f"{_EXPECTED_WEIGHT}\n"
            "  3. pip install -r requirements.txt from the cloned repo\n"
            "  4. Restart and re-select the RestoreFormer backend.\n"
            "Returning input video unchanged."
        )
        log(msg)
        return Path(video_path)

    raise NotImplementedError(
        "[restoreformer] install detected but the inference adapter "
        "isn't shipped in T2-2 v1.  Open a follow-up task to wire "
        "the model load + paste-back pattern, mirroring "
        "codeformer_backend.py."
    )
