from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core import AppPaths
from app.services.config_service import ConfigService
from app.services.inference_service import InferenceService
from app.services.result_store_service import ResultStoreService
from app.utils.inference_manager import InferenceManager
from nanonis.services import NanonisSessionService, ScanWorkflowService


@dataclass
class AppRuntime:
    paths: AppPaths
    config_service: ConfigService
    result_store: ResultStoreService
    nanonis_service: NanonisSessionService
    workflow_service: ScanWorkflowService

    @classmethod
    def create(cls, project_root: Path) -> "AppRuntime":
        paths = AppPaths.from_project_root(project_root)
        config_service = ConfigService(paths.config_path)
        result_store = ResultStoreService(paths.sessions_root)
        nanonis_service = NanonisSessionService(paths.sessions_root / "scratch")
        workflow_service = ScanWorkflowService(nanonis_service)
        config_service.load()
        return cls(
            paths=paths,
            config_service=config_service,
            result_store=result_store,
            nanonis_service=nanonis_service,
            workflow_service=workflow_service,
        )

    def create_inference_service(self, inference_manager: InferenceManager) -> InferenceService:
        return InferenceService(inference_manager)

    def create_startup_session(self, label: str = "startup") -> dict:
        return self.result_store.create_session(label)
