from __future__ import annotations

from pathlib import Path

from PIL import Image

from app.core import AnnotationBox, AnnotationState
from app.runtime import AppRuntime


class DummyInferenceManager:
    def __init__(self) -> None:
        self.loaded_images: list[str] = []
        self.run_calls: list[tuple[str, str, str, str]] = []

    def load_images(self, images: list[str]) -> None:
        self.loaded_images = list(images)

    def run_inference(self, yolo_model: str, resnet_model: str, output_dir: str, classes_yaml: str = "") -> None:
        self.run_calls.append((yolo_model, resnet_model, output_dir, classes_yaml))


class DummyTrainingManager:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int, list[str], str]] = []

    def generate_data_yaml(
        self,
        train_images_dir: str,
        val_images_dir: str,
        class_count: int,
        class_names: list[str],
        data_yaml_path: str,
    ) -> None:
        self.calls.append((train_images_dir, val_images_dir, class_count, list(class_names), data_yaml_path))
        Path(data_yaml_path).write_text("nc: 1\nnames: ['atom']\n", encoding="utf-8")


class DummyNanonisService:
    def __init__(self) -> None:
        self.output_root = None


def _make_image(path: Path) -> str:
    Image.new("RGB", (96, 96), "white").save(path)
    return str(path)


def test_runtime_workflows_support_stm_style_project_flow(tmp_path: Path):
    runtime = AppRuntime.create(tmp_path)
    image_path = _make_image(tmp_path / "scan.png")
    state = AnnotationState(
        images=[image_path],
        annotations={
            image_path: [
                AnnotationBox(cls=0, x=0.5, y=0.5, w=0.4, h=0.4),
            ]
        },
        class_names=["atom"],
        current_index=0,
    )

    nanonis_service = DummyNanonisService()
    scan_session = runtime.acquisition_workflow_service.start_scan_session(None, "stm-scan", nanonis_service)
    scan_record = runtime.acquisition_workflow_service.append_scan_result(
        scan_session.id,
        {
            "label": "scan_a",
            "saved": [{"png": image_path}],
        },
    )

    image_load = runtime.image_workflow_service.convert_for_inference([image_path])
    inference_manager = DummyInferenceManager()
    inference_service = runtime.create_inference_service(inference_manager)
    inference_start = runtime.inference_workflow_service.prepare_inference_start(
        scan_session,
        "infer-run",
        "classes.yaml",
    )
    inference_selection = runtime.inference_workflow_service.select_scan_result_for_inference(
        scan_session,
        scan_record,
        inference_service,
    )
    inference_service.infer_files(
        files=image_load.gray_files or [image_path],
        yolo_model="yolo.pt",
        resnet_model="resnet.pth",
        output_dir=inference_start.output_dir,
        classes_yaml=inference_start.classes_yaml,
    )
    runtime.inference_workflow_service.persist_inference_result(
        scan_session.id,
        inference_start.output_dir,
        ["result.png"],
        "actual-output",
    )

    training_manager = DummyTrainingManager()
    training_session, training_context = runtime.training_workflow_service.ensure_training_context(
        scan_session,
        runtime.session_workflow_service,
        "yolo",
        str(tmp_path / "train-project"),
    )
    yolo_plan = runtime.training_workflow_service.prepare_yolo_training(
        state,
        lambda path: path,
        training_manager,
        lambda message: None,
    )

    loaded_sessions = runtime.session_workflow_service.list_sessions()

    assert nanonis_service.output_root == runtime.session_workflow_service.scan_dir(scan_session.id)
    assert scan_record.label == "scan_a"
    assert image_load.files == [image_path]
    assert image_load.gray_files == [image_path]
    assert inference_selection is not None
    assert inference_selection.files == [image_path]
    assert inference_manager.loaded_images == [image_path]
    assert inference_manager.run_calls == [
        ("yolo.pt", "resnet.pth", str(inference_start.output_dir), "classes.yaml")
    ]
    assert training_session.id == scan_session.id
    assert training_context == {
        "session_id": scan_session.id,
        "mode": "yolo",
        "project_dir": str(tmp_path / "train-project"),
    }
    assert yolo_plan is not None
    assert Path(yolo_plan.data_yaml_path).exists()
    assert training_manager.calls
    assert loaded_sessions[0].scan_results[0].label == "scan_a"
    assert loaded_sessions[0].inference_results[0].images == ["result.png"]
