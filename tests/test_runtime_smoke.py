from __future__ import annotations

from pathlib import Path

from app.runtime import AppRuntime


def test_runtime_create_exposes_core_services(tmp_path: Path):
    runtime = AppRuntime.create(tmp_path)

    assert runtime.paths.project_root == tmp_path
    assert runtime.annotation_service is not None
    assert runtime.config_service is not None
    assert runtime.dataset_service is not None
    assert runtime.image_workflow_service is not None
    assert runtime.inference_workflow_service is not None
    assert runtime.result_store is not None
    assert runtime.resource_loader_service is not None
    assert runtime.session_workflow_service is not None
    assert runtime.sxm_service is not None
    assert runtime.training_job_service is not None
    assert runtime.training_runner_service is not None
    assert runtime.training_workflow_service is not None
    assert runtime.nanonis_service is not None
    assert runtime.workflow_service is not None

    assert runtime.paths.config_path.parent.exists()
    assert runtime.paths.sessions_root.exists()
