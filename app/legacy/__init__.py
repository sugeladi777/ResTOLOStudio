"""Legacy shell components kept behind stable application APIs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .workbench_impl import ReSTOLOApp

__all__ = ["ReSTOLOApp", "ReSTOLOLegacyWorkbench"]


def __getattr__(name: str):
    if name in {"ReSTOLOApp", "ReSTOLOLegacyWorkbench"}:
        from .workbench_impl import ReSTOLOApp

        return ReSTOLOApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
