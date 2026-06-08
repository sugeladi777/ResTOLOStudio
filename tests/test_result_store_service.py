from __future__ import annotations

from pathlib import Path

from app.core import InferenceResultRecord, ScanResultRecord, TrainingResultRecord
from app.services.result_store_service import ResultStoreService


def test_result_store_creates_session_tree(tmp_path: Path):
    store = ResultStoreService(tmp_path / "sessions")

    session = store.create_session("demo")
    session_dir = store.session_dir(session.id)

    assert session_dir.exists()
    assert (session_dir / "scan").exists()
    assert (session_dir / "inference").exists()
    assert (session_dir / "training").exists()
    assert (session_dir / "logs").exists()
    assert store.manifest_path(session.id).exists()


def test_result_store_persists_result_records(tmp_path: Path):
    store = ResultStoreService(tmp_path / "sessions")
    session = store.create_session("demo")

    store.append_scan_result(session.id, ScanResultRecord(label="scan_a"))
    store.append_inference_result(session.id, InferenceResultRecord(output_dir="infer_dir", images=["a.png"]))
    store.append_training_result(
        session.id,
        TrainingResultRecord(mode="yolo", project_dir="proj", output_dir="out", status="completed"),
    )

    loaded = store.load_manifest(session.id)
    assert [item.label for item in loaded.scan_results] == ["scan_a"]
    assert [item.output_dir for item in loaded.inference_results] == ["infer_dir"]
    assert [item.mode for item in loaded.training_results] == ["yolo"]
    assert [item.status for item in loaded.training_results] == ["completed"]
