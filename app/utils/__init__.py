"""Utility helpers for ReSTOLO Studio."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .error_matcher import ErrorMatcher, ErrorRule
    from .inference_manager import InferenceManager
    from .model_manager import ModelManager
    from .training_manager import TrainingManager
    from .sxm_parser import load_sxm_as_image

__all__ = [
    "ErrorMatcher",
    "ErrorRule",
    "InferenceManager",
    "ModelManager",
    "TrainingManager",
    "load_sxm_as_image",
]


def __getattr__(name: str):
    if name in {"ErrorMatcher", "ErrorRule"}:
        from .error_matcher import ErrorMatcher, ErrorRule

        return {"ErrorMatcher": ErrorMatcher, "ErrorRule": ErrorRule}[name]
    if name == "InferenceManager":
        from .inference_manager import InferenceManager

        return InferenceManager
    if name == "ModelManager":
        from .model_manager import ModelManager

        return ModelManager
    if name == "TrainingManager":
        from .training_manager import TrainingManager

        return TrainingManager
    if name == "load_sxm_as_image":
        from .sxm_parser import load_sxm_as_image

        return load_sxm_as_image
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
