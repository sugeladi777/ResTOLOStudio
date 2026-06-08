from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.core import InferenceResultRecord, ScanResultRecord
from app.services.inference_service import InferenceService
from app.services.inference_workflow_service import InferenceWorkflowService
from app.services.result_store_service import ResultStoreService
from app.services.session_workflow_service import SessionWorkflowService


class DummyInferenceManager:
    def load_images(self, images):
        self.images = images

    def run_inference(self, yolo_model, resnet_model, output_dir, classes_yaml=""):
        self.last_call = (yolo_model, resnet_model, output_dir, classes_yaml)


def test_inference_workflow_service_prepares_start_context_and_persists_results(tmp_path: Path):
    session_workflow = SessionWorkflowService(ResultStoreService(tmp_path / "sessions"))
    workflow = InferenceWorkflowService(session_workflow)

    context = workflow.prepare_inference_start(None, "infer-demo", " classes.yaml ")
    record = workflow.persist_inference_result(
        context.session.id,
        context.output_dir,
        ["result.png"],
        "actual-output",
    )

    assert context.session.label == "infer-demo"
    assert context.output_dir == session_workflow.inference_dir(context.session.id)
    assert context.classes_yaml == "classes.yaml"
    assert isinstance(record, InferenceResultRecord)
    loaded = session_workflow.list_sessions()[0]
    assert [item.output_dir for item in loaded.inference_results] == [str(context.output_dir)]


def test_inference_workflow_service_selects_scan_result_files(tmp_path: Path):
    session_workflow = SessionWorkflowService(ResultStoreService(tmp_path / "sessions"))
    workflow = InferenceWorkflowService(session_workflow)
    manager = DummyInferenceManager()
    inference_service = InferenceService(manager)
    session = session_workflow.create_session("demo")
    scan_result = ScanResultRecord(label="scan", saved=[{"png": str(tmp_path / "scan.png")}])

    selection = workflow.select_scan_result_for_inference(session, scan_result, inference_service)
    empty_selection = workflow.select_scan_result_for_inference(session, ScanResultRecord(label="empty"), inference_service)

    assert selection is not None
    assert selection.session.id == session.id
    assert selection.files == [str(tmp_path / "scan.png")]
    assert empty_selection is None
