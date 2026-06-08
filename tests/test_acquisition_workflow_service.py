from __future__ import annotations

from pathlib import Path

from app.core import ScanResultRecord
from app.services.acquisition_workflow_service import AcquisitionWorkflowService
from app.services.result_store_service import ResultStoreService
from app.services.session_workflow_service import SessionWorkflowService


class DummyNanonisService:
    def __init__(self) -> None:
        self.output_root = None


def test_acquisition_workflow_service_builds_config_and_geometry(tmp_path: Path):
    workflow = AcquisitionWorkflowService(SessionWorkflowService(ResultStoreService(tmp_path / "sessions")))

    config = workflow.build_connection_config(" 127.0.0.1 ", "6501", "10380")
    geometry = workflow.build_scan_geometry("12.5", "8.0", "256", "Z, Current")

    assert config.ip == "127.0.0.1"
    assert config.port == 6501
    assert config.version == 10380
    assert geometry.width_nm == 12.5
    assert geometry.height_nm == 8.0
    assert geometry.pixels == 256
    assert geometry.channels == ["Z", "Current"]


def test_acquisition_workflow_service_starts_session_and_persists_scan_results(tmp_path: Path):
    session_workflow = SessionWorkflowService(ResultStoreService(tmp_path / "sessions"))
    workflow = AcquisitionWorkflowService(session_workflow)
    nanonis_service = DummyNanonisService()

    session = workflow.start_scan_session(None, "scan-demo", nanonis_service)
    record = workflow.append_scan_result(session.id, {"label": "scan_a", "saved": []})
    pre, post = workflow.append_scan_workflow_results(
        session.id,
        {
            "pre_scan": {"label": "pre", "saved": []},
            "post_scan": {"label": "post", "saved": []},
        },
    )

    assert session.label == "scan-demo"
    assert nanonis_service.output_root == session_workflow.scan_dir(session.id)
    assert isinstance(record, ScanResultRecord)
    loaded = session_workflow.list_sessions()[0]
    assert [item.label for item in loaded.scan_results] == ["scan_a", "pre", "post"]
    assert pre.label == "pre"
    assert post.label == "post"
