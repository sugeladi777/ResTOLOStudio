from __future__ import annotations

from pathlib import Path

from PIL import Image
from PyQt5.QtWidgets import QApplication

from app.core import AnnotationBox, AnnotationState
from app.bootstrap import create_application
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
    assert training_context["session_id"] == scan_session.id
    assert training_context["mode"] == "yolo"
    assert training_context["project_dir"] == str(tmp_path / "train-project")
    assert Path(training_context["expected_output_dir"]).name == "yolo_train"
    assert yolo_plan is not None
    assert Path(yolo_plan.data_yaml_path).exists()
    assert training_manager.calls
    assert loaded_sessions[0].scan_results[0].label == "scan_a"
    assert loaded_sessions[0].inference_results[0].images == ["result.png"]


def test_window_automation_flow_restores_sidecars_and_prepares_inference(tmp_path: Path):
    app = QApplication.instance() or QApplication([])
    assets_models = tmp_path / "assets" / "models"
    assets_models.mkdir(parents=True, exist_ok=True)
    (assets_models / "best.pt").write_text("yolo", encoding="utf-8")
    (assets_models / "demo.saving").write_text("resnet", encoding="utf-8")
    (assets_models / "classes.yaml").write_text("names:\n  - atom\n", encoding="utf-8")

    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    image_path = _make_image(input_dir / "sample.png")
    (input_dir / "sample.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    (input_dir / "classes.yaml").write_text("names:\n  - atom\n", encoding="utf-8")

    qt_app, window = create_application(tmp_path)
    assert qt_app is not None

    logs: list[str] = []
    original_log = window.log
    window.log = lambda message: (logs.append(str(message)), original_log(message))[0]

    window.studio._choose_files = lambda _title, _filter: [image_path]
    window.load_images()
    loaded_state = window.annotation_tool.export_state()

    assert loaded_state.images == [image_path]
    assert len(loaded_state.annotations[image_path]) == 1
    assert window.infer_classes_path.text() == str(assets_models / "classes.yaml")

    window.save_annotations()
    session_id = window.current_session.id
    annotation_dir = window.session_workflow_service.annotation_dir(session_id)
    assert (annotation_dir / "sample.png").exists()
    assert (annotation_dir / "sample.txt").exists()
    assert (annotation_dir / "classes.yaml").exists()

    window.acquisition_workflow_service.append_scan_result(
        session_id,
        {
            "label": "scan_a",
            "saved": [{"png": image_path}],
        },
    )
    window.current_session = next(
        session for session in window.session_workflow_service.list_sessions() if session.id == session_id
    )
    window.reload_sessions()
    window.session_list.setCurrentRow(0)
    window._session_selected(0)
    window.result_list.setCurrentRow(0)
    window._result_selected(0)

    inference_calls: list[dict] = []
    window.inference_service = type(
        "DummyInferenceService",
        (),
        {
            "infer_files": lambda self, **kwargs: inference_calls.append(kwargs),
            "scan_result_files": lambda self, scan_result: [image_path],
        },
    )()
    window.inference_manager.images = []
    window.use_selected_result_for_inference()
    window.start_inference()

    assert window.inference_manager.images == [image_path]
    assert inference_calls
    assert inference_calls[0]["files"] == [image_path]
    assert inference_calls[0]["yolo_model"] == str(assets_models / "best.pt")
    assert inference_calls[0]["resnet_model"] == str(assets_models / "demo.saving")
    assert inference_calls[0]["classes_yaml"] == str(annotation_dir / "classes.yaml")
    assert window.pending_inference_session_id == session_id
    assert window.pending_inference_output_dir == window.session_workflow_service.inference_dir(session_id)
    assert any("已自动加载 1 个匹配标注文件" in item for item in logs)

    window.close()
    app.processEvents()
