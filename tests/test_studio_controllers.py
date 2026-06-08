from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.windows.runtime_controller import StudioRuntimeController
from app.windows.studio_controller import StudioController


class DummyButton:
    def __init__(self) -> None:
        self.enabled = None

    def setEnabled(self, value: bool) -> None:
        self.enabled = value


class DummyLineEdit:
    def __init__(self, value: str = "") -> None:
        self._value = value

    def text(self) -> str:
        return self._value

    def setText(self, value: str) -> None:
        self._value = value


class DummyAnnotationTool:
    def __init__(self, state) -> None:
        self._state = state
        self.mode_changes: list[bool] = []
        self.loaded_states: list[object] = []

    def export_state(self):
        return self._state

    def load_state(self, state) -> None:
        self._state = state
        self.loaded_states.append(state)

    def set_annotation_mode(self, enabled: bool) -> None:
        self.mode_changes.append(enabled)


class DummyAnnotationService:
    def has_images(self, state) -> bool:
        return bool(state.images)

    def has_annotations(self, state) -> bool:
        return bool(state.annotations)


class DummyInferenceManager:
    def __init__(self) -> None:
        self.images: list[str] = []
        self.loaded_batches: list[list[str]] = []

    def load_images(self, images: list[str]) -> None:
        self.images = list(images)
        self.loaded_batches.append(list(images))


class DummyToggle:
    def __init__(self) -> None:
        self.visible = None

    def setVisible(self, value: bool) -> None:
        self.visible = value


class DummySxmService:
    def __init__(self, conversion) -> None:
        self.conversion = conversion
        self.calls: list[list[str]] = []

    def convert_files(self, files: list[str]):
        self.calls.append(list(files))
        return self.conversion


class DummyImageWorkflowService:
    def __init__(self, conversion_result) -> None:
        self.conversion_result = conversion_result
        self.annotation_calls: list[list[str]] = []
        self.inference_calls: list[list[str]] = []
        self.gray_path_calls: list[tuple[str, dict[str, dict]]] = []

    def convert_for_annotation(self, files: list[str]):
        self.annotation_calls.append(list(files))
        return self.conversion_result

    def convert_for_inference(self, files: list[str]):
        self.inference_calls.append(list(files))
        return self.conversion_result

    def remap_sxm_display(self, state, sxm_metadata, use_color: bool):
        return state

    def gray_path_for(self, image_path: str, sxm_metadata: dict[str, dict]) -> str:
        self.gray_path_calls.append((image_path, dict(sxm_metadata)))
        for gray_path, meta in sxm_metadata.items():
            if image_path == meta.get("color_path"):
                return gray_path
        return image_path


class DummyAcquisitionWorkflowService:
    def __init__(self) -> None:
        self.config_calls: list[tuple[str, str, str]] = []
        self.geometry_calls: list[tuple[str, str, str, str]] = []
        self.started_sessions: list[tuple[object, str, object]] = []
        self.append_scan_calls: list[tuple[str, object]] = []
        self.append_workflow_calls: list[tuple[str, dict]] = []

    def build_connection_config(self, ip: str, port: str, version: str):
        self.config_calls.append((ip, port, version))
        return SimpleNamespace(ip=ip.strip(), port=int(port.strip()), version=int(version.strip()))

    def build_scan_geometry(self, width_nm: str, height_nm: str, pixels: str, channels: str):
        self.geometry_calls.append((width_nm, height_nm, pixels, channels))
        return SimpleNamespace(
            to_dict=lambda: {
                "width_nm": float(width_nm),
                "height_nm": float(height_nm),
                "center_x_nm": 0.0,
                "center_y_nm": 0.0,
                "angle_deg": 0.0,
                "pixels": int(pixels),
                "channels": [part.strip() for part in channels.split(",") if part.strip()],
            }
        )

    def start_scan_session(self, current_session, label: str, nanonis_service):
        self.started_sessions.append((current_session, label, nanonis_service))
        return SimpleNamespace(id="session-1", label=label.strip() or "session")

    def append_scan_result(self, session_id: str, result):
        self.append_scan_calls.append((session_id, result))
        return SimpleNamespace(label=result["label"])

    def append_scan_workflow_results(self, session_id: str, result: dict):
        self.append_workflow_calls.append((session_id, result))
        return SimpleNamespace(label="pre"), SimpleNamespace(label="post")


class DummyInferenceWorkflowService:
    def __init__(self) -> None:
        self.start_calls: list[tuple[object, str, str]] = []
        self.selection_calls: list[tuple[object, object, object]] = []
        self.persist_calls: list[tuple[str, object, list[str], str]] = []

    def prepare_inference_start(self, current_session, label: str, classes_yaml: str):
        self.start_calls.append((current_session, label, classes_yaml))
        return SimpleNamespace(
            session=SimpleNamespace(id="infer-session", label=label.strip() or "session"),
            output_dir=Path("infer-output"),
            classes_yaml=classes_yaml.strip(),
        )

    def select_scan_result_for_inference(self, session, scan_result, inference_service):
        self.selection_calls.append((session, scan_result, inference_service))
        if session is None or scan_result is None:
            return None
        return SimpleNamespace(session=session, files=["scan_a.png", "scan_b.png"])

    def persist_inference_result(self, session_id: str, pending_output_dir, result_images: list[str], actual_output_dir: str):
        self.persist_calls.append((session_id, pending_output_dir, list(result_images), actual_output_dir))
        return SimpleNamespace(output_dir=str(pending_output_dir), images=result_images)


class DummyListWidget:
    def __init__(self) -> None:
        self.items: list[str] = []
        self.current_row = None

    def clear(self) -> None:
        self.items.clear()

    def addItem(self, value: str) -> None:
        self.items.append(value)

    def setCurrentRow(self, index: int) -> None:
        self.current_row = index


class DummySessionWorkflowForController:
    def session_list_labels(self) -> list[str]:
        return ["session-1 | scan=1 | infer=0"]

    def list_sessions(self):
        return [SimpleNamespace(id="session-1", scan_results=[1], inference_results=[])]


def test_studio_controller_convert_sxm_files_updates_window_state(tmp_path: Path):
    color_file = tmp_path / "scan_color.png"
    color_file.write_text("color", encoding="utf-8")
    gray_file = tmp_path / "scan_gray.png"
    annotation_state = SimpleNamespace(images=[str(gray_file)], annotations={str(gray_file): []})

    conversion = SimpleNamespace(
        files=[str(gray_file)],
        gray_files=[],
        annotation_state=annotation_state,
        sxm_metadata={str(gray_file): {"color_path": str(color_file)}},
        sxm_original_paths={str(gray_file): "scan.sxm"},
        sxm_color_paths={str(gray_file): str(color_file)},
        has_sxm_files=True,
        logs=["converted scan.sxm"],
    )
    logs: list[str] = []
    annotation_tool = DummyAnnotationTool(SimpleNamespace(images=[], annotations={}))
    window = SimpleNamespace(
        image_workflow_service=DummyImageWorkflowService(conversion),
        sxm_color_toggle=DummyToggle(),
        annotation_tool=annotation_tool,
        log=logs.append,
    )

    controller = StudioController(window)
    files = controller.convert_sxm_files(["scan.sxm"])

    assert files == [str(gray_file)]
    assert window.image_workflow_service.annotation_calls == [["scan.sxm"]]
    assert window.sxm_metadata == conversion.sxm_metadata
    assert window.sxm_original_paths == conversion.sxm_original_paths
    assert window.sxm_color_paths == conversion.sxm_color_paths
    assert window._has_sxm_files is True
    assert window.sxm_color_toggle.visible is True
    assert annotation_tool.loaded_states[-1] is annotation_state
    assert logs == ["converted scan.sxm"]
    assert controller.get_gray_path(str(color_file)) == str(gray_file)
    assert controller.get_gray_path(str(gray_file)) == str(gray_file)


def test_studio_controller_uses_acquisition_workflow_for_scan_state():
    acquisition = DummyAcquisitionWorkflowService()
    logs: list[str] = []
    window = SimpleNamespace(
        acquisition_workflow_service=acquisition,
        session_workflow_service=DummySessionWorkflowForController(),
        nano_ip_edit=DummyLineEdit("127.0.0.1"),
        nano_port_edit=DummyLineEdit("6501"),
        nano_version_edit=DummyLineEdit("10380"),
        scan_width_edit=DummyLineEdit("10"),
        scan_height_edit=DummyLineEdit("20"),
        scan_pixels_edit=DummyLineEdit("128"),
        scan_channels_edit=DummyLineEdit("Z, Current"),
        scan_label_edit=DummyLineEdit("scan-demo"),
        nanonis_service=SimpleNamespace(),
        session_list=DummyListWidget(),
        pending_scan_results=[],
        log=logs.append,
        current_session=None,
    )

    controller = StudioController(window)
    config = controller._nanonis_config()
    geometry = controller._scan_geometry()
    session = controller._new_session()
    controller.handle_service_result("scan_and_save", {"label": "scan_a"})
    controller.handle_service_result("scan_pulse_scan", {"pre_scan": {"label": "pre"}, "post_scan": {"label": "post"}})

    assert config.port == 6501
    assert geometry["pixels"] == 128
    assert geometry["channels"] == ["Z", "Current"]
    assert session.id == "session-1"
    assert acquisition.config_calls == [("127.0.0.1", "6501", "10380")]
    assert acquisition.geometry_calls[0] == ("10", "20", "128", "Z, Current")
    assert acquisition.started_sessions[0][1] == "scan-demo"
    assert acquisition.append_scan_calls == [("session-1", {"label": "scan_a"})]
    assert acquisition.append_workflow_calls == [("session-1", {"pre_scan": {"label": "pre"}, "post_scan": {"label": "post"}})]
    assert logs[-2:] == ["Scan completed: scan_a", "Workflow completed: pre-scan -> pulse -> post-scan"]


def test_runtime_controller_update_button_states_and_tab_sync(tmp_path: Path):
    resnet_dir = tmp_path / "resnet-data"
    resnet_dir.mkdir()
    state = SimpleNamespace(images=["color.png"], annotations={"color.png": ["box"]})
    annotation_tool = DummyAnnotationTool(state)
    inference_manager = DummyInferenceManager()
    logs: list[str] = []

    window = SimpleNamespace(
        annotation_tool=annotation_tool,
        annotation_service=DummyAnnotationService(),
        inference_manager=inference_manager,
        model_manager=SimpleNamespace(yolo_model=object(), resnet_model=object()),
        train_resnet_data_path=DummyLineEdit(str(resnet_dir)),
        load_annotations_btn=DummyButton(),
        save_annotations_btn=DummyButton(),
        crop_resnet_dataset_btn=DummyButton(),
        train_load_images_btn=DummyButton(),
        train_load_annotations_btn=DummyButton(),
        train_load_yolo_model_btn=DummyButton(),
        train_load_resnet_model_btn=DummyButton(),
        train_yolo_btn=DummyButton(),
        train_resnet_btn=DummyButton(),
        infer_load_images_btn=DummyButton(),
        infer_load_yolo_model_btn=DummyButton(),
        infer_load_resnet_model_btn=DummyButton(),
        start_inference_btn=DummyButton(),
        log=logs.append,
        detect_and_display_classes=lambda: None,
        _get_gray_path=lambda path: f"gray::{path}",
    )
    inference_manager.images = ["gray.png"]

    controller = StudioRuntimeController(window)

    controller.update_button_states()
    controller.on_tab_changed(2)

    assert window.load_annotations_btn.enabled is True
    assert window.save_annotations_btn.enabled is True
    assert window.crop_resnet_dataset_btn.enabled is True
    assert window.train_yolo_btn.enabled is True
    assert window.train_resnet_btn.enabled is True
    assert window.start_inference_btn.enabled is True
    assert annotation_tool.mode_changes[-1] is False
    assert inference_manager.loaded_batches[-1] == ["gray::color.png"]
    assert logs[-1] == "Switched to inference mode"


def test_studio_controller_uses_inference_workflow_for_start_and_finish():
    inference_workflow = DummyInferenceWorkflowService()
    inference_service_calls: list[dict] = []
    logs: list[str] = []
    window = SimpleNamespace(
        inference_workflow_service=inference_workflow,
        session_workflow_service=DummySessionWorkflowForController(),
        model_manager=SimpleNamespace(yolo_model="yolo.pt", resnet_model="resnet.pth"),
        inference_manager=SimpleNamespace(
            images=["img_a.png"],
            actual_output_dir="actual-dir",
            get_inference_results=lambda output_dir: ["result_a.png"],
        ),
        infer_classes_path=DummyLineEdit(" classes.yaml "),
        scan_label_edit=DummyLineEdit("infer-demo"),
        inference_service=SimpleNamespace(
            infer_files=lambda **kwargs: inference_service_calls.append(kwargs),
        ),
        progress_bar=SimpleNamespace(show=lambda: None, setValue=lambda value: None),
        disable_controls=lambda: None,
        current_session=None,
        pending_inference_session_id=None,
        pending_inference_output_dir=None,
        session_list=DummyListWidget(),
        log=logs.append,
    )

    controller = StudioController(window)
    controller.start_inference()
    controller.on_inference_finished()

    assert inference_workflow.start_calls == [(None, "infer-demo", " classes.yaml ")]
    assert window.current_session.id == "infer-session"
    assert window.pending_inference_session_id is None
    assert window.pending_inference_output_dir is None
    assert inference_service_calls == [
        {
            "files": ["img_a.png"],
            "yolo_model": "yolo.pt",
            "resnet_model": "resnet.pth",
            "output_dir": Path("infer-output"),
            "classes_yaml": "classes.yaml",
        }
    ]
    assert inference_workflow.persist_calls == [
        ("infer-session", Path("infer-output"), ["result_a.png"], "actual-dir")
    ]
    assert logs[0] == "Starting inference for session: infer-session"


def test_runtime_controller_disables_actions_when_inputs_missing():
    state = SimpleNamespace(images=[], annotations={})
    window = SimpleNamespace(
        annotation_tool=DummyAnnotationTool(state),
        annotation_service=DummyAnnotationService(),
        inference_manager=DummyInferenceManager(),
        model_manager=SimpleNamespace(yolo_model=None, resnet_model=None),
        train_resnet_data_path=DummyLineEdit(""),
        load_annotations_btn=DummyButton(),
        save_annotations_btn=DummyButton(),
        crop_resnet_dataset_btn=DummyButton(),
        train_load_images_btn=DummyButton(),
        train_load_annotations_btn=DummyButton(),
        train_load_yolo_model_btn=DummyButton(),
        train_load_resnet_model_btn=DummyButton(),
        train_yolo_btn=DummyButton(),
        train_resnet_btn=DummyButton(),
        infer_load_images_btn=DummyButton(),
        infer_load_yolo_model_btn=DummyButton(),
        infer_load_resnet_model_btn=DummyButton(),
        start_inference_btn=DummyButton(),
        log=lambda message: None,
    )

    controller = StudioRuntimeController(window)
    controller.update_button_states()

    assert window.load_annotations_btn.enabled is False
    assert window.save_annotations_btn.enabled is False
    assert window.crop_resnet_dataset_btn.enabled is False
    assert window.train_load_images_btn.enabled is True
    assert window.train_load_annotations_btn.enabled is False
    assert window.train_yolo_btn.enabled is False
    assert window.train_resnet_btn.enabled is False
    assert window.start_inference_btn.enabled is False
