from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PyQt5.QtWidgets import QApplication

from app.core import AnnotationState
from app.ui.annotation_tool import AnnotationTool
from app.windows.studio_window import ReSTOLOStudioApp
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


class DummyTextBlock:
    def __init__(self, value: str = "") -> None:
        self._value = value

    def setPlainText(self, value: str) -> None:
        self._value = value

    def toPlainText(self) -> str:
        return self._value


class DummyComboBox:
    def __init__(self, items: list[str] | None = None, index: int = 0) -> None:
        self.items = list(items or [])
        self._index = index

    def currentText(self) -> str:
        if 0 <= self._index < len(self.items):
            return self.items[self._index]
        return ""

    def currentIndex(self) -> int:
        return self._index

    def addItem(self, text: str) -> None:
        self.items.append(text)

    def clear(self) -> None:
        self.items.clear()
        self._index = 0

    def blockSignals(self, _value: bool) -> None:
        return None


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

    def currentRow(self) -> int:
        return self.current_row if self.current_row is not None else -1


class DummySessionWorkflowForController:
    def list_sessions(self):
        return [SimpleNamespace(id="session-1", scan_results=[1], inference_results=[])]

    def session_list_labels(self, sessions=None) -> list[str]:
        source = sessions if sessions is not None else self.list_sessions()
        return [
            (
                f"{session.id} | 待训练\n"
                f"扫描 {len(getattr(session, 'scan_results', []) or [])} 条 "
                f"训练 {len(getattr(session, 'training_results', []) or [])} 次 "
                f"推理 {len(getattr(session, 'inference_results', []) or [])} 次"
            )
            for session in source
        ]

    def selected_session(self, index: int):
        if index != 0:
            return None
        return SimpleNamespace(
            id="session-1",
            label="demo-session",
            path="",
            scan_results=[],
            inference_results=[],
            training_results=[],
        )

    def selected_scan_result(self, session_index: int, result_index: int):
        return None, None


def test_annotation_tool_load_state_renders_pixmap_for_png(tmp_path: Path):
    app = QApplication.instance() or QApplication([])

    image_path = tmp_path / "sample.png"
    image_path.write_bytes(
        bytes.fromhex(
            "89504E470D0A1A0A0000000D4948445200000001000000010802000000907753DE"
            "0000000C49444154789C63F8FFFF3F0005FE02FEA7E5C9A00000000049454E44AE426082"
        )
    )

    tool = AnnotationTool()
    tool.resize(900, 700)
    tool.load_state(AnnotationState(images=[str(image_path)]))
    app.processEvents()

    pixmap = tool.image_label.pixmap()
    assert pixmap is not None
    assert not pixmap.isNull()
    assert "sample.png" in tool.status_label.text()


def test_studio_window_persists_and_restores_ui_sizes(tmp_path: Path):
    app = QApplication.instance() or QApplication([])
    runtime = __import__("app.runtime", fromlist=["AppRuntime"]).AppRuntime.create(tmp_path)

    window = ReSTOLOStudioApp(runtime)
    window.resize(1500, 900)
    window._persist_window_geometry()
    window._persist_splitter_state(610, 0)

    assert runtime.config_service.data["ui_window_width"] == 1500
    assert runtime.config_service.data["ui_window_height"] == 900
    assert runtime.config_service.data["ui_left_panel_width"] == 610

    other_runtime = __import__("app.runtime", fromlist=["AppRuntime"]).AppRuntime.create(tmp_path)
    other_window = ReSTOLOStudioApp(other_runtime)
    app.processEvents()

    assert other_window.width() >= 1500
    assert other_window.height() >= 900


def test_studio_window_uses_wider_default_left_panel_width(tmp_path: Path):
    app = QApplication.instance() or QApplication([])
    runtime = __import__("app.runtime", fromlist=["AppRuntime"]).AppRuntime.create(tmp_path)

    window = ReSTOLOStudioApp(runtime)
    window.show()
    app.processEvents()

    sizes = window.main_splitter.sizes()
    assert sizes[0] >= 560


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
        result_list=DummyListWidget(),
        session_search_edit=DummyLineEdit(""),
        session_sort_combo=DummyComboBox(["最近活跃优先"]),
        session_browser_context_text=DummyTextBlock(),
        result_compare_combo=DummyComboBox(["不进行对比"]),
        result_compare_summary_text=DummyTextBlock(),
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
    assert logs[-2:] == ["扫描完成：scan_a", "流程完成：预扫 -> 脉冲 -> 后扫"]


def test_studio_controller_reload_sessions_uses_richer_labels():
    session_list = DummyListWidget()
    window = SimpleNamespace(
        session_workflow_service=DummySessionWorkflowForController(),
        session_list=session_list,
        result_list=DummyListWidget(),
        session_search_edit=DummyLineEdit(""),
        session_sort_combo=DummyComboBox(["最近活跃优先"]),
        session_browser_context_text=DummyTextBlock(),
        result_compare_combo=DummyComboBox(["不进行对比"]),
        result_compare_summary_text=DummyTextBlock(),
        result_detail_text=DummyTextBlock(),
        result_preview_label=SimpleNamespace(clear=lambda: None, setText=lambda _value: None),
        result_preview_caption=DummyLineEdit(),
        result_compare_preview_label=SimpleNamespace(clear=lambda: None, setText=lambda _value: None),
        result_compare_preview_caption=DummyLineEdit(),
    )

    controller = StudioController(window)
    controller.reload_sessions()

    assert len(session_list.items) == 1
    assert "待训练" in session_list.items[0]
    assert "扫描 1 条" in session_list.items[0]


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
        workspace_mode_detail=DummyLineEdit(),
        workflow_focus_detail=DummyLineEdit(),
        operator_hint_detail=DummyLineEdit(),
        workspace_images_value=DummyLineEdit(),
        workspace_images_detail=DummyLineEdit(),
        workspace_annotations_value=DummyLineEdit(),
        workspace_annotations_detail=DummyLineEdit(),
        workspace_inference_queue_value=DummyLineEdit(),
        workspace_inference_queue_detail=DummyLineEdit(),
        workspace_session_value=DummyLineEdit(),
        workspace_session_detail=DummyLineEdit(),
        acquisition_session_name_value=DummyLineEdit(),
        acquisition_session_name_detail=DummyLineEdit(),
        acquisition_scan_count_value=DummyLineEdit(),
        acquisition_scan_count_detail=DummyLineEdit(),
        acquisition_training_state_value=DummyLineEdit(),
        acquisition_training_state_detail=DummyLineEdit(),
        acquisition_inference_state_value=DummyLineEdit(),
        acquisition_inference_state_detail=DummyLineEdit(),
        nano_endpoint_value=DummyLineEdit(),
        nano_endpoint_detail=DummyLineEdit(),
        nano_link_value=DummyLineEdit(),
        nano_link_detail=DummyLineEdit(),
        nano_protocol_value=DummyLineEdit(),
        nano_protocol_detail=DummyLineEdit(),
        nano_last_action_value=DummyLineEdit(),
        nano_last_action_detail=DummyLineEdit(),
        log_recent_value=DummyLineEdit(),
        log_recent_detail=DummyLineEdit(),
        log_level_value=DummyLineEdit(),
        log_level_detail=DummyLineEdit(),
        result_session_scan_count_value=DummyLineEdit(),
        result_session_scan_count_detail=DummyLineEdit(),
        result_stage_value=DummyLineEdit(),
        result_stage_detail=DummyLineEdit(),
        result_training_state_value=DummyLineEdit(),
        result_training_state_detail=DummyLineEdit(),
        result_difference_value=DummyLineEdit(),
        result_difference_detail=DummyLineEdit(),
        overview_session_value=DummyLineEdit(),
        overview_session_detail=DummyLineEdit(),
        overview_training_value=DummyLineEdit(),
        overview_training_detail=DummyLineEdit(),
        overview_inference_value=DummyLineEdit(),
        overview_inference_detail=DummyLineEdit(),
        overview_next_action_value=DummyLineEdit(),
        overview_next_action_detail=DummyLineEdit(),
        acquisition_session_context_text=DummyTextBlock(),
        latest_scan_context_text=DummyTextBlock(),
        session_stage_context_text=DummyTextBlock(),
        latest_activity_context_text=DummyTextBlock(),
        result_difference_context_text=DummyTextBlock(),
        session_timeline_text=DummyTextBlock(),
        session_sort_combo=DummyComboBox(["最近活跃优先"]),
        result_compare_combo=DummyComboBox(["不进行对比"]),
        result_compare_summary_text=DummyTextBlock(),
        tab_widget=SimpleNamespace(currentIndex=lambda: 0),
        log=logs.append,
        detect_and_display_classes=lambda: None,
        _get_gray_path=lambda path: f"gray::{path}",
        current_session=SimpleNamespace(label="demo", id="session-1", scan_results=[], training_results=[], inference_results=[]),
    )
    inference_manager.images = ["gray.png"]

    controller = StudioRuntimeController(window)

    controller.update_button_states()
    controller.on_tab_changed(3)

    assert window.load_annotations_btn.enabled is True
    assert window.save_annotations_btn.enabled is True
    assert window.crop_resnet_dataset_btn.enabled is True
    assert window.train_yolo_btn.enabled is True
    assert window.train_resnet_btn.enabled is True
    assert window.start_inference_btn.enabled is True
    assert annotation_tool.mode_changes[-1] is False
    assert inference_manager.loaded_batches[-1] == ["gray::color.png"]
    assert logs[-1] == "已切换到推理模式。"


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
        result_list=DummyListWidget(),
        session_search_edit=DummyLineEdit(""),
        session_sort_combo=DummyComboBox(["最近活跃优先"]),
        session_browser_context_text=DummyTextBlock(),
        result_compare_combo=DummyComboBox(["不进行对比"]),
        result_compare_summary_text=DummyTextBlock(),
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
    assert logs[0] == "开始执行推理，会话：infer-session"


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
        workspace_mode_detail=DummyLineEdit(),
        workflow_focus_detail=DummyLineEdit(),
        operator_hint_detail=DummyLineEdit(),
        workspace_images_value=DummyLineEdit(),
        workspace_images_detail=DummyLineEdit(),
        workspace_annotations_value=DummyLineEdit(),
        workspace_annotations_detail=DummyLineEdit(),
        workspace_inference_queue_value=DummyLineEdit(),
        workspace_inference_queue_detail=DummyLineEdit(),
        workspace_session_value=DummyLineEdit(),
        workspace_session_detail=DummyLineEdit(),
        acquisition_session_name_value=DummyLineEdit(),
        acquisition_session_name_detail=DummyLineEdit(),
        acquisition_scan_count_value=DummyLineEdit(),
        acquisition_scan_count_detail=DummyLineEdit(),
        acquisition_training_state_value=DummyLineEdit(),
        acquisition_training_state_detail=DummyLineEdit(),
        acquisition_inference_state_value=DummyLineEdit(),
        acquisition_inference_state_detail=DummyLineEdit(),
        nano_endpoint_value=DummyLineEdit(),
        nano_endpoint_detail=DummyLineEdit(),
        nano_link_value=DummyLineEdit(),
        nano_link_detail=DummyLineEdit(),
        nano_protocol_value=DummyLineEdit(),
        nano_protocol_detail=DummyLineEdit(),
        nano_last_action_value=DummyLineEdit(),
        nano_last_action_detail=DummyLineEdit(),
        log_recent_value=DummyLineEdit(),
        log_recent_detail=DummyLineEdit(),
        log_level_value=DummyLineEdit(),
        log_level_detail=DummyLineEdit(),
        result_session_scan_count_value=DummyLineEdit(),
        result_session_scan_count_detail=DummyLineEdit(),
        result_stage_value=DummyLineEdit(),
        result_stage_detail=DummyLineEdit(),
        result_training_state_value=DummyLineEdit(),
        result_training_state_detail=DummyLineEdit(),
        result_difference_value=DummyLineEdit(),
        result_difference_detail=DummyLineEdit(),
        overview_session_value=DummyLineEdit(),
        overview_session_detail=DummyLineEdit(),
        overview_training_value=DummyLineEdit(),
        overview_training_detail=DummyLineEdit(),
        overview_inference_value=DummyLineEdit(),
        overview_inference_detail=DummyLineEdit(),
        overview_next_action_value=DummyLineEdit(),
        overview_next_action_detail=DummyLineEdit(),
        acquisition_session_context_text=DummyTextBlock(),
        latest_scan_context_text=DummyTextBlock(),
        session_stage_context_text=DummyTextBlock(),
        latest_activity_context_text=DummyTextBlock(),
        result_difference_context_text=DummyTextBlock(),
        session_timeline_text=DummyTextBlock(),
        session_sort_combo=DummyComboBox(["最近活跃优先"]),
        result_compare_combo=DummyComboBox(["不进行对比"]),
        result_compare_summary_text=DummyTextBlock(),
        tab_widget=SimpleNamespace(currentIndex=lambda: 0),
        log=lambda message: None,
        current_session=SimpleNamespace(label="demo", id="session-1", scan_results=[], training_results=[], inference_results=[]),
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


def test_log_slot_updates_log_snapshot_labels():
    entries: list[str] = []

    def log_slot(target, message):
        if hasattr(target, "log_text"):
            target.log_text.append(message)
        lowered = message.lower()
        level = "信息"
        detail = "普通工作流事件"
        if "error" in lowered or "failed" in lowered or "失败" in message:
            level = "错误"
            detail = "需要优先检查当前步骤"
        elif "warning" in lowered or "未找到" in message or "请先" in message:
            level = "提示"
            detail = "当前流程需要补充前置条件"
        elif "complete" in lowered or "completed" in lowered or "完成" in message:
            level = "完成"
            detail = "可以进入下一阶段或复查结果"
        elif "connecting" in lowered or "starting" in lowered or "进行中" in message:
            level = "进行中"
            detail = "当前任务正在执行"
        if hasattr(target, "log_recent_value"):
            target.log_recent_value.setText(message[:48] + ("..." if len(message) > 48 else ""))
        if hasattr(target, "log_recent_detail"):
            target.log_recent_detail.setText(detail)
        if hasattr(target, "log_level_value"):
            target.log_level_value.setText(level)
        if hasattr(target, "log_level_detail"):
            target.log_level_detail.setText(message if len(message) <= 80 else message[:80] + "...")

    dummy = SimpleNamespace(
        log_text=SimpleNamespace(append=entries.append),
        log_recent_value=DummyLineEdit(),
        log_recent_detail=DummyLineEdit(),
        log_level_value=DummyLineEdit(),
        log_level_detail=DummyLineEdit(),
    )
    log_slot(dummy, "Training complete")

    assert entries == ["Training complete"]
    assert dummy.log_level_value.text() == "完成"
    assert "下一阶段" in dummy.log_recent_detail.text()


def test_studio_controller_opens_selected_directories(tmp_path: Path, monkeypatch):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    training_dir = tmp_path / "training"
    training_dir.mkdir()
    inference_dir = tmp_path / "inference"
    inference_dir.mkdir()
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    result_png = result_dir / "scan.png"
    result_png.write_text("png", encoding="utf-8")

    session = SimpleNamespace(
        id="session-1",
        label="demo-session",
        path=str(session_dir),
        scan_results=[],
        inference_results=[SimpleNamespace(output_dir="", actual_output_dir=str(inference_dir), images=["scan.png"])],
        training_results=[SimpleNamespace(output_dir=str(training_dir))],
    )
    scan_result = SimpleNamespace(saved=[{"png": str(result_png)}], raw={})
    session.scan_results = [scan_result]

    opened_paths: list[str] = []
    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr("app.windows.studio_controller.os.startfile", lambda path: opened_paths.append(path), raising=False)
    monkeypatch.setattr(
        "app.windows.studio_controller.QMessageBox.warning",
        lambda *args: warnings.append((args[1], args[2])),
    )

    session_list = DummyListWidget()
    session_list.setCurrentRow(0)
    result_list = DummyListWidget()
    result_list.setCurrentRow(0)

    session_workflow = DummySessionWorkflowForController()
    session_workflow.list_sessions = lambda: [session]
    session_workflow.selected_session = lambda index: session if index == 0 else None
    session_workflow.selected_scan_result = lambda session_index, result_index: (session, scan_result)

    logs: list[str] = []
    window = SimpleNamespace(
        session_workflow_service=session_workflow,
        session_list=session_list,
        result_list=result_list,
        session_search_edit=DummyLineEdit(""),
        session_sort_combo=DummyComboBox(["最近活跃优先"]),
        result_compare_combo=DummyComboBox(["不进行对比"]),
        result_compare_summary_text=DummyTextBlock(),
        log=logs.append,
    )

    controller = StudioController(window)
    controller.open_selected_session_directory()
    controller.open_selected_training_output()
    controller.open_selected_inference_output()
    controller.open_selected_result_directory()

    assert warnings == []
    assert opened_paths == [
        str(session_dir),
        str(training_dir),
        str(inference_dir),
        str(result_dir),
    ]
    assert logs[-4:] == [
        f"已打开目录：{session_dir}",
        f"已打开目录：{training_dir}",
        f"已打开目录：{inference_dir}",
        f"已打开目录：{result_dir}",
    ]


def test_studio_controller_reports_missing_directories(monkeypatch):
    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "app.windows.studio_controller.QMessageBox.warning",
        lambda *args: warnings.append((args[1], args[2])),
    )

    session_list = DummyListWidget()
    session_list.setCurrentRow(0)
    result_list = DummyListWidget()
    result_list.setCurrentRow(0)

    session_workflow = DummySessionWorkflowForController()
    session_workflow.list_sessions = lambda: [
        SimpleNamespace(
            id="session-1",
            label="demo-session",
            path="",
            scan_results=[],
            inference_results=[],
            training_results=[],
        )
    ]
    session_workflow.selected_session = lambda index: SimpleNamespace(
        id="session-1",
        label="demo-session",
        path="",
        scan_results=[],
        inference_results=[],
        training_results=[],
    )
    session_workflow.selected_scan_result = lambda session_index, result_index: (None, None)

    logs: list[str] = []
    window = SimpleNamespace(
        session_workflow_service=session_workflow,
        session_list=session_list,
        result_list=result_list,
        session_search_edit=DummyLineEdit(""),
        session_sort_combo=DummyComboBox(["最近活跃优先"]),
        result_compare_combo=DummyComboBox(["不进行对比"]),
        result_compare_summary_text=DummyTextBlock(),
        log=logs.append,
    )

    controller = StudioController(window)
    controller.open_selected_session_directory()
    controller.open_selected_training_output()
    controller.open_selected_inference_output()
    controller.open_selected_result_directory()

    assert [message for _, message in warnings] == [
        "当前会话目录暂不可用。",
        "当前会话还没有训练输出目录。",
        "当前会话还没有推理输出目录。",
        "当前结果还没有导出目录。",
    ]
    assert logs == [message for _, message in warnings]
