"""Application services for ReSTOLO Studio."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config_service import ConfigService
    from .inference_service import InferenceService
    from .result_store_service import ResultStoreService

__all__ = ["ConfigService", "InferenceService", "ResultStoreService"]


def __getattr__(name: str):
    if name == "ConfigService":
        from .config_service import ConfigService

        return ConfigService
    if name == "InferenceService":
        from .inference_service import InferenceService

        return InferenceService
    if name == "ResultStoreService":
        from .result_store_service import ResultStoreService

        return ResultStoreService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
