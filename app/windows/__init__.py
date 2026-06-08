"""Top-level application windows."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .studio_window import ReSTOLOStudioApp, ReSTOLOStudioWindow

__all__ = ["ReSTOLOStudioApp", "ReSTOLOStudioWindow"]


def __getattr__(name: str):
    if name in {"ReSTOLOStudioApp", "ReSTOLOStudioWindow"}:
        from .studio_window import ReSTOLOStudioApp, ReSTOLOStudioWindow

        return {"ReSTOLOStudioApp": ReSTOLOStudioApp, "ReSTOLOStudioWindow": ReSTOLOStudioWindow}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
