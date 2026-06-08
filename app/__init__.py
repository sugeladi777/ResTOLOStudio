"""ReSTOLO Studio application package."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .runtime import AppRuntime
    from .windows import ReSTOLOStudioApp, ReSTOLOStudioWindow

__all__ = ["AppRuntime", "ReSTOLOStudioApp", "ReSTOLOStudioWindow"]


def __getattr__(name: str):
    if name == "AppRuntime":
        from .runtime import AppRuntime

        return AppRuntime
    if name in {"ReSTOLOStudioApp", "ReSTOLOStudioWindow"}:
        from .windows import ReSTOLOStudioApp, ReSTOLOStudioWindow

        return {"ReSTOLOStudioApp": ReSTOLOStudioApp, "ReSTOLOStudioWindow": ReSTOLOStudioWindow}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
