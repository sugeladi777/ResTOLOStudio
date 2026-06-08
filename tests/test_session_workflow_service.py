from __future__ import annotations

from pathlib import Path

from app.core import InferenceResultRecord, ScanResultRecord, TrainingResultRecord
from app.services.result_store_service import ResultStoreService
from app.services.session_workflow_service import SessionWorkflowService


def test_session_workflow_service_manages_selection_and_labels(tmp_path: Path):
    workflow = SessionWorkflowService(ResultStoreService(tmp_path / "sessions"))

    session = workflow.create_session("demo")
    workflow.append_scan_results(
        session.id,
        ScanResultRecord(label="scan_a"),
        ScanResultRecord(label="scan_b"),
    )
    workflow.append_inference_result(
        session.id,
        InferenceResultRecord(output_dir="infer_out", images=["a.png"]),
    )
    workflow.append_training_result(
        session.id,
        TrainingResultRecord(mode="yolo", project_dir="proj", output_dir="out", status="completed"),
    )

    sessions = workflow.list_sessions()
    assert len(sessions) == 1
    assert workflow.selected_session(0).id == session.id
    assert workflow.selected_session(9) is None
    assert workflow.result_labels(sessions[0]) == ["scan_a", "scan_b"]
    assert workflow.session_list_labels() == [f"{session.id} | scan=2 | infer=1"]
    assert "scan_b" in workflow.result_detail(sessions[0], 1)
    assert workflow.result_detail(sessions[0], 9) == ""


def test_session_workflow_service_ensures_session(tmp_path: Path):
    workflow = SessionWorkflowService(ResultStoreService(tmp_path / "sessions"))

    created = workflow.ensure_session(None, "training")
    reused = workflow.ensure_session(created, "ignored")

    assert created.label == "training"
    assert reused is created
    assert workflow.scan_dir(created.id).exists()
    assert workflow.inference_dir(created.id).exists()
