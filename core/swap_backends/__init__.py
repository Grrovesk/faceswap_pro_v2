"""Swap-backend registry + factory.

Phase 1 of the multi-backend roadmap (see CHANGELOG entry
``Tier 1 #1`` and ``docs/swap_backend_comparison.md``).  Phase 1
ships ONLY the ``inswapper_128`` backend behind the abstraction --
default behavior for end users is unchanged.

Usage
-----

    from core.swap_backends import get_backend, list_backends

    BackendCls = get_backend("inswapper_128")
    engine = BackendCls(model_path=".../inswapper_128.onnx", device_id=0)
    engine.initialize()
    engine.set_source_embedding(emb)
    out = engine.swap_paste_back(frame, kps, emb)

    for info in list_backends():
        print(info.name, info.license_label)

Phase 2 will add ``ghost_a`` (Apache-2.0, 256 native).  Phase 3 will
add ``simswap_512`` (CC-BY-NC-4.0, 512 native).  Both register via
the same :func:`register_backend` hook below; nothing else in the
codebase needs to change to surface them in the UI.
"""

from __future__ import annotations

from typing import Dict, List, Type

from .base import BackendInfo, SwapBackend

# Internal registry.  Backends are lazy-loaded on first request so
# that importing ``core.swap_backends`` does not pull in onnxruntime
# / torch / etc. for backends the user is not using.
_REGISTRY: Dict[str, Type[SwapBackend]] = {}

# Stable name aliases.  Keys are user-facing strings; values are the
# canonical ``BackendInfo.name`` registered in ``_REGISTRY``.  The
# resolver in :func:`get_backend` applies these transparently.
_ALIASES: Dict[str, str] = {
    "inswapper": "inswapper_128",
}


def register_backend(cls: Type[SwapBackend]) -> None:
    """Register a concrete backend class.

    The class must expose an ``info`` attribute of type
    :class:`BackendInfo`.  The registry key is ``cls.info.name``.
    """
    info = getattr(cls, "info", None)
    if not isinstance(info, BackendInfo):
        raise TypeError(
            f"{cls!r} is missing a BackendInfo 'info' class attribute "
            f"(got {info!r}).  Did you forget to set it?"
        )
    _REGISTRY[info.name] = cls


def _ensure_loaded(name: str) -> None:
    """Lazy-import the backend module that owns ``name``."""
    if name in _REGISTRY:
        return
    if name in ("inswapper_128", "inswapper"):
        from .inswapper import InswapperBackend
        register_backend(InswapperBackend)
    # Future backends register here.  Phase 1 ships only inswapper_128;
    # the SimSwap-512 backend was prototyped, integrated, A/B-tested
    # against the reference, and removed because it did not improve
    # quality over inswapper_128 + GFPGAN on real footage.  The
    # abstraction layer remains so a future backend (PuLID, InstantID,
    # or anything else) can drop in without touching anything else.


def get_backend(name: str) -> Type[SwapBackend]:
    """Look up a backend class by name.

    Accepts the canonical ``BackendInfo.name`` and any alias from
    :data:`_ALIASES` (e.g. ``"inswapper"`` -> ``"inswapper_128"``).

    Raises
    ------
    KeyError
        If ``name`` is not a known backend.  The message includes
        the list of names that ARE known so the user can pick one.
    """
    canonical = _ALIASES.get(name, name)
    _ensure_loaded(canonical)
    if canonical not in _REGISTRY:
        _ensure_loaded("inswapper_128")
        available = sorted(set(_REGISTRY.keys()) | set(_ALIASES.keys()))
        raise KeyError(
            f"Unknown swap backend {name!r}.  "
            f"Available: {available}"
        )
    return _REGISTRY[canonical]


def list_backends() -> List[BackendInfo]:
    """Return :class:`BackendInfo` for every registered backend.

    Currently force-loads ``inswapper_128`` so the default always
    appears.  Once Phases 2 and 3 land, extend the lazy imports in
    :func:`_ensure_loaded` AND add an ``_ensure_loaded(...)`` call
    here for each new backend so the UI sees the full menu.
    """
    _ensure_loaded("inswapper_128")
    return [cls.info for cls in _REGISTRY.values() if cls.info is not None]


__all__ = [
    "BackendInfo",
    "SwapBackend",
    "get_backend",
    "list_backends",
    "register_backend",
]
