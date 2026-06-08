from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core import AppPaths
from app.services.acquisition_workflow_service import AcquisitionWorkflowService
from app.services.annotation_service import AnnotationService
from app.services.config_service import ConfigService
from app.services.dataset_service import DatasetService
from app.services.image_workflow_service import ImageWorkflowService
from app.services.inference_service import InferenceService
from app.services.inference_workflow_service import InferenceWorkflowService
from app.services.result_store_service import ResultStoreService
from app.services.resource_loader_service import ResourceLoaderService
from app.services.session_workflow_service import SessionWorkflowService
from app.services.sxm_service import SxmService
from app.services.training_job_service import TrainingJobService
from app.services.training_runner_service import TrainingRunnerService
from app.services.training_workflow_service import TrainingWorkflowService
from app.utils.inference_manager import InferenceManager
from nanonis.services import NanonisSessionService, ScanWorkflowService


@dataclass
class AppRuntime:
    paths: AppPaths
    annotation_service: AnnotationService
    acquisition_workflow_service: AcquisitionWorkflowService
    config_service: ConfigService
    dataset_service: DatasetService
    image_workflow_service: ImageWorkflowService
    inference_workflow_service: InferenceWorkflowService
    result_store: ResultStoreService
    resource_loader_service: ResourceLoaderService
    session_workflow_service: SessionWorkflowService
    sxm_service: SxmService
    training_job_service: TrainingJobService
    training_runner_service: TrainingRunnerService
    training_workflow_service: TrainingWorkflowService
    nanonis_service: NanonisSessionService
    workflow_service: ScanWorkflowService

    @classmethod
    def create(cls, project_root: Path) -> "AppRuntime":
        paths = AppPaths.from_project_root(project_root)
        annotation_service = AnnotationService()
        config_service = ConfigService(paths.config_path)
        dataset_service = DatasetService()
        image_workflow_service = ImageWorkflowService(annotation_service, sxm_service := SxmService())
        result_store = ResultStoreService(paths.sessions_root)
        session_workflow_service = SessionWorkflowService(result_store)
        acquisition_workflow_service = AcquisitionWorkflowService(session_workflow_service)
        inference_workflow_service = InferenceWorkflowService(session_workflow_service)
        resource_loader_service = ResourceLoaderService()
        training_job_service = TrainingJobService()
        training_runner_service = TrainingRunnerService()
        training_workflow_service = TrainingWorkflowService(dataset_service, training_job_service)
        nanonis_service = NanonisSessionService(paths.sessions_root / "scratch")
        workflow_service = ScanWorkflowService(nanonis_service)
        config_service.load()
        return cls(
            paths=paths,
            annotation_service=annotation_service,
            acquisition_workflow_service=acquisition_workflow_service,
            config_service=config_service,
            dataset_service=dataset_service,
            image_workflow_service=image_workflow_service,
            inference_workflow_service=inference_workflow_service,
            result_store=result_store,
            resource_loader_service=resource_loader_service,
            session_workflow_service=session_workflow_service,
            sxm_service=sxm_service,
            training_job_service=training_job_service,
            training_runner_service=training_runner_service,
            training_workflow_service=training_workflow_service,
            nanonis_service=nanonis_service,
            workflow_service=workflow_service,
        )

    def create_inference_service(self, inference_manager: InferenceManager) -> InferenceService:
        return InferenceService(inference_manager)

    def create_startup_session(self, label: str = "startup") -> dict:
        return self.session_workflow_service.create_session(label)
