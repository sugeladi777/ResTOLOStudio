"""UI widgets for ReSTOLO Studio."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .annotation_tool import AnnotationTool
    from .loss_curve_dialog import LossCurveDialog

__all__ = ["AnnotationTool", "LossCurveDialog"]


def __getattr__(name: str):
    if name == "AnnotationTool":
        from .annotation_tool import AnnotationTool

        return AnnotationTool
    if name == "LossCurveDialog":
        from .loss_curve_dialog import LossCurveDialog

        return LossCurveDialog
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
