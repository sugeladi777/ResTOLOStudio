"""Application services for ReSTOLO Studio."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .acquisition_workflow_service import AcquisitionWorkflowService
    from .annotation_service import AnnotationService
    from .config_service import ConfigService
    from .dataset_service import DatasetService
    from .image_workflow_service import ImageWorkflowService
    from .inference_service import InferenceService
    from .inference_workflow_service import InferenceWorkflowService
    from .result_store_service import ResultStoreService
    from .resource_loader_service import ResourceLoaderService
    from .session_workflow_service import SessionWorkflowService
    from .sxm_service import SxmService
    from .training_job_service import TrainingJobService
    from .training_runner_service import TrainingRunnerService
    from .training_workflow_service import TrainingWorkflowService

__all__ = [
    "AnnotationService",
    "AcquisitionWorkflowService",
    "ConfigService",
    "DatasetService",
    "ImageWorkflowService",
    "InferenceService",
    "InferenceWorkflowService",
    "ResultStoreService",
    "ResourceLoaderService",
    "SessionWorkflowService",
    "SxmService",
    "TrainingJobService",
    "TrainingRunnerService",
    "TrainingWorkflowService",
]


def __getattr__(name: str):
    if name == "AnnotationService":
        from .annotation_service import AnnotationService

        return AnnotationService
    if name == "AcquisitionWorkflowService":
        from .acquisition_workflow_service import AcquisitionWorkflowService

        return AcquisitionWorkflowService
    if name == "ConfigService":
        from .config_service import ConfigService

        return ConfigService
    if name == "InferenceService":
        from .inference_service import InferenceService

        return InferenceService
    if name == "InferenceWorkflowService":
        from .inference_workflow_service import InferenceWorkflowService

        return InferenceWorkflowService
    if name == "ImageWorkflowService":
        from .image_workflow_service import ImageWorkflowService

        return ImageWorkflowService
    if name == "DatasetService":
        from .dataset_service import DatasetService

        return DatasetService
    if name == "ResultStoreService":
        from .result_store_service import ResultStoreService

        return ResultStoreService
    if name == "ResourceLoaderService":
        from .resource_loader_service import ResourceLoaderService

        return ResourceLoaderService
    if name == "SessionWorkflowService":
        from .session_workflow_service import SessionWorkflowService

        return SessionWorkflowService
    if name == "SxmService":
        from .sxm_service import SxmService

        return SxmService
    if name == "TrainingJobService":
        from .training_job_service import TrainingJobService

        return TrainingJobService
    if name == "TrainingRunnerService":
        from .training_runner_service import TrainingRunnerService

        return TrainingRunnerService
    if name == "TrainingWorkflowService":
        from .training_workflow_service import TrainingWorkflowService

        return TrainingWorkflowService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
