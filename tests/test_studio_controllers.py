from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PyQt5.QtWidgets import QApplication, QWidget

from app.core import AnnotationState
from app.ui.annotation_tool import AnnotationTool
from app.windows.studio_window import ReSTOLOStudioApp
from app.windows.runtime_controller import StudioRuntimeController
from app.windows.studio_controller import StudioController


class DummyButton:
    def __init__(self) -> None:
        self.enabled = None
        self.tooltip = ""

    def setEnabled(self, value: bool) -> None:
        self.enabled = value

    def setToolTip(self, value: str) -> None:
        self.tooltip = value


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

    def clear(self) -> None:
        self._value = ""


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


class DummyResourceLoaderService:
    def __init__(self) -> None:
        self.yolo_calls: list[str] = []
        self.resnet_calls: list[str] = []
        self.classes_calls: list[str] = []

    def load_yolo_model(self, file_path: str, model_manager, log, *targets):
        self.yolo_calls.append(file_path)
        for target in targets:
            if target is not None:
                target.setText(file_path)
        return True

    def load_resnet_model(self, file_path: str, model_manager, log, *targets):
        self.resnet_calls.append(file_path)
        for target in targets:
            if target is not None:
                target.setText(file_path)
        return True

    def load_classes_file(self, file_path: str, log, *targets):
        self.classes_calls.append(file_path)
        for target in targets:
            if target is not None:
                target.setText(file_path)
        return True


class DummyAnnotationIoService:
    def __init__(self) -> None:
        self.save_calls: list[tuple[object, str]] = []

    def save_annotations(self, state, directory: str):
        self.save_calls.append((state, directory))
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        classes_path = path / "classes.yaml"
        classes_path.write_text("names: ['atom']\n", encoding="utf-8")
        return path, ["atom"]


class DummyDatasetExportService(DummyAnnotationService):
    def __init__(self) -> None:
        self.crop_calls: list[tuple[str, dict[int, str]]] = []

    def annotation_class_summary(self, _annotation_tool):
        return SimpleNamespace(used_classes={0}, class_names=["atom"])

    def crop_resnet_dataset(self, _annotation_tool, _gray_resolver, directory: str, selected_class_map: dict[int, str]):
        self.crop_calls.append((directory, dict(selected_class_map)))
        Path(directory).mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(crop_count=1, actual_classes={"atom"})


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


def test_studio_window_prefers_latest_session_training_models_on_startup(tmp_path: Path):
    app = QApplication.instance() or QApplication([])
    runtime_module = __import__("app.runtime", fromlist=["AppRuntime"])
    runtime = runtime_module.AppRuntime.create(tmp_path)
    session = runtime.session_workflow_service.create_session("trained")
    runtime.session_workflow_service.append_training_result(
        session.id,
        {
            "mode": "yolo",
            "project_dir": str(tmp_path / "project"),
            "output_dir": str(tmp_path / "project" / "yolo_train"),
            "status": "completed",
            "yolo_model_path": "trained-best.pt",
            "resnet_model_path": "trained-best.saving",
            "classes_yaml_path": "trained-classes.yaml",
        },
    )

    other_runtime = runtime_module.AppRuntime.create(tmp_path)
    window = ReSTOLOStudioApp(other_runtime)
    app.processEvents()

    assert window.current_session.id == session.id
    assert window.train_yolo_model_path.text() == "trained-best.pt"
    assert window.infer_yolo_model_path.text() == "trained-best.pt"
    assert window.train_resnet_model_path.text() == "trained-best.saving"
    assert window.infer_resnet_model_path.text() == "trained-best.saving"
    assert window.train_classes_path.text() == "trained-classes.yaml"
    assert window.infer_classes_path.text() == "trained-classes.yaml"


def test_studio_window_prefers_session_annotation_classes_when_no_training_record(tmp_path: Path):
    app = QApplication.instance() or QApplication([])
    runtime_module = __import__("app.runtime", fromlist=["AppRuntime"])
    runtime = runtime_module.AppRuntime.create(tmp_path)
    session = runtime.session_workflow_service.create_session("annotated")
    annotation_dir = runtime.session_workflow_service.annotation_dir(session.id)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    classes_path = annotation_dir / "classes.yaml"
    classes_path.write_text("names:\n  - atom\n  - vacancy\n", encoding="utf-8")

    other_runtime = runtime_module.AppRuntime.create(tmp_path)
    window = ReSTOLOStudioApp(other_runtime)
    app.processEvents()

    assert window.current_session.id == session.id
    assert window.train_classes_path.text() == str(classes_path)
    assert window.infer_classes_path.text() == str(classes_path)


def test_studio_window_restores_saved_annotation_state_on_startup(tmp_path: Path):
    app = QApplication.instance() or QApplication([])
    runtime_module = __import__("app.runtime", fromlist=["AppRuntime"])
    runtime = runtime_module.AppRuntime.create(tmp_path)
    session = runtime.session_workflow_service.create_session("annotated-state")
    annotation_dir = runtime.session_workflow_service.annotation_dir(session.id)
    image_path = annotation_dir / "sample.png"
    image_path.write_bytes(
        bytes.fromhex(
            "89504E470D0A1A0A0000000D4948445200000001000000010802000000907753DE"
            "0000000C49444154789C63F8FFFF3F0005FE02FEA7E5C9A00000000049454E44AE426082"
        )
    )
    (annotation_dir / "sample.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    (annotation_dir / "classes.yaml").write_text("names:\n  - atom\n", encoding="utf-8")

    other_runtime = runtime_module.AppRuntime.create(tmp_path)
    window = ReSTOLOStudioApp(other_runtime)
    app.processEvents()

    state = window.annotation_tool.export_state()
    assert window.current_session.id == session.id
    assert len(state.images) == 1
    assert Path(state.images[0]).name == "sample.png"
    assert len(state.annotations[state.images[0]]) == 1


def test_studio_window_restores_saved_external_resource_paths_from_config(tmp_path: Path):
    app = QApplication.instance() or QApplication([])
    runtime_module = __import__("app.runtime", fromlist=["AppRuntime"])
    runtime = runtime_module.AppRuntime.create(tmp_path)

    yolo_path = tmp_path / "custom-yolo.pt"
    resnet_path = tmp_path / "custom-resnet.saving"
    classes_path = tmp_path / "custom-classes.yaml"
    resnet_data_dir = tmp_path / "resnet-data"
    yolo_path.write_text("yolo", encoding="utf-8")
    resnet_path.write_text("resnet", encoding="utf-8")
    classes_path.write_text("names:\n  - atom\n", encoding="utf-8")
    (resnet_data_dir / "atom").mkdir(parents=True)

    payload = dict(runtime.config_service.data)
    payload["recent_yolo_model_path"] = str(yolo_path)
    payload["recent_resnet_model_path"] = str(resnet_path)
    payload["recent_classes_yaml_path"] = str(classes_path)
    payload["recent_resnet_data_path"] = str(resnet_data_dir)
    runtime.config_service.save(payload)

    other_runtime = runtime_module.AppRuntime.create(tmp_path)
    window = ReSTOLOStudioApp(other_runtime)
    app.processEvents()

    assert window.train_yolo_model_path.text() == str(yolo_path)
    assert window.infer_yolo_model_path.text() == str(yolo_path)
    assert window.train_resnet_model_path.text() == str(resnet_path)
    assert window.infer_resnet_model_path.text() == str(resnet_path)
    assert window.train_classes_path.text() == str(classes_path)
    assert window.infer_classes_path.text() == str(classes_path)
    assert window.train_resnet_data_path.text() == ""


def test_studio_window_exposes_visible_status_sections(tmp_path: Path):
    app = QApplication.instance() or QApplication([])
    runtime_module = __import__("app.runtime", fromlist=["AppRuntime"])
    runtime = runtime_module.AppRuntime.create(tmp_path)

    window = ReSTOLOStudioApp(runtime)
    app.processEvents()

    assert window.annotation_status_detail.text() == "尚未加载图像。"
    assert window.training_dataset_status_detail.text() == "训练数据尚未准备好；请先加载图像，再加载或绘制标注。"
    assert window.inference_input_status_detail.text() == "尚未加载推理图像。"
    assert window.inference_result_status_detail.text() == "当前没有推理结果。"
    assert not window.findChild(QWidget, "annotation_status_group").isHidden()
    assert not window.findChild(QWidget, "training_status_group").isHidden()
    assert not window.findChild(QWidget, "inference_status_group").isHidden()


def test_studio_window_keeps_session_training_models_over_saved_recent_paths(tmp_path: Path):
    app = QApplication.instance() or QApplication([])
    runtime_module = __import__("app.runtime", fromlist=["AppRuntime"])
    runtime = runtime_module.AppRuntime.create(tmp_path)

    session = runtime.session_workflow_service.create_session("trained")
    runtime.session_workflow_service.append_training_result(
        session.id,
        {
            "mode": "yolo",
            "project_dir": str(tmp_path / "project"),
            "output_dir": str(tmp_path / "project" / "yolo_train"),
            "status": "completed",
            "yolo_model_path": "session-best.pt",
            "resnet_model_path": "session-best.saving",
            "classes_yaml_path": "session-classes.yaml",
        },
    )

    recent_yolo = tmp_path / "recent-yolo.pt"
    recent_resnet = tmp_path / "recent-resnet.saving"
    recent_classes = tmp_path / "recent-classes.yaml"
    recent_yolo.write_text("recent", encoding="utf-8")
    recent_resnet.write_text("recent", encoding="utf-8")
    recent_classes.write_text("names:\n  - atom\n", encoding="utf-8")

    payload = dict(runtime.config_service.data)
    payload["recent_yolo_model_path"] = str(recent_yolo)
    payload["recent_resnet_model_path"] = str(recent_resnet)
    payload["recent_classes_yaml_path"] = str(recent_classes)
    runtime.config_service.save(payload)

    other_runtime = runtime_module.AppRuntime.create(tmp_path)
    window = ReSTOLOStudioApp(other_runtime)
    app.processEvents()

    assert window.current_session.id == session.id
    assert window.train_yolo_model_path.text() == "session-best.pt"
    assert window.infer_yolo_model_path.text() == "session-best.pt"
    assert window.train_resnet_model_path.text() == "session-best.saving"
    assert window.infer_resnet_model_path.text() == "session-best.saving"
    assert window.train_classes_path.text() == "session-classes.yaml"
    assert window.infer_classes_path.text() == "session-classes.yaml"


def test_studio_window_restores_saved_acquisition_settings_from_config(tmp_path: Path):
    app = QApplication.instance() or QApplication([])
    runtime_module = __import__("app.runtime", fromlist=["AppRuntime"])
    runtime = runtime_module.AppRuntime.create(tmp_path)

    payload = dict(runtime.config_service.data)
    payload.update(
        {
            "recent_nanonis_ip": "192.168.0.10",
            "recent_nanonis_port": "6600",
            "recent_nanonis_version": "20400",
            "recent_scan_label": "demo-scan",
            "recent_scan_bias": "1.8",
            "recent_scan_setpoint": "7.0e-11",
            "recent_scan_width": "12",
            "recent_scan_height": "15",
            "recent_scan_pixels": "512",
            "recent_scan_channels": "Z, Current",
            "recent_pulse_bias": "3.1",
            "recent_pulse_width": "0.2",
        }
    )
    runtime.config_service.save(payload)

    other_runtime = runtime_module.AppRuntime.create(tmp_path)
    window = ReSTOLOStudioApp(other_runtime)
    app.processEvents()

    assert window.nano_ip_edit.text() == "192.168.0.10"
    assert window.nano_port_edit.text() == "6600"
    assert window.nano_version_edit.text() == "20400"
    assert window.scan_label_edit.text() == "demo-scan"
    assert window.scan_bias_edit.text() == "1.8"
    assert window.scan_setpoint_edit.text() == "7.0e-11"
    assert window.scan_width_edit.text() == "12"
    assert window.scan_height_edit.text() == "15"
    assert window.scan_pixels_edit.text() == "512"
    assert window.scan_channels_edit.text() == "Z, Current"
    assert window.pulse_bias_edit.text() == "3.1"
    assert window.pulse_width_edit.text() == "0.2"


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


def test_studio_controller_load_images_auto_restores_matching_annotations_and_classes(tmp_path: Path):
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(
        bytes.fromhex(
            "89504E470D0A1A0A0000000D4948445200000001000000010802000000907753DE"
            "0000000C49444154789C63F8FFFF3F0005FE02FEA7E5C9A00000000049454E44AE426082"
        )
    )
    (tmp_path / "sample.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    (tmp_path / "classes.yaml").write_text("names:\n  - atom\n", encoding="utf-8")

    annotation_state = AnnotationState(images=[str(image_path)], annotations={str(image_path): []}, class_names=["0"])
    conversion = SimpleNamespace(
        files=[str(image_path)],
        gray_files=[],
        annotation_state=annotation_state,
        sxm_metadata={},
        sxm_original_paths={},
        sxm_color_paths={},
        has_sxm_files=False,
        logs=[],
    )
    annotation_tool = DummyAnnotationTool(annotation_state)
    resource_loader = DummyResourceLoaderService()
    logs: list[str] = []
    window = SimpleNamespace(
        image_workflow_service=DummyImageWorkflowService(conversion),
        annotation_service=__import__("app.services.annotation_service", fromlist=["AnnotationService"]).AnnotationService(),
        annotation_tool=annotation_tool,
        resource_loader_service=resource_loader,
        train_classes_path=DummyLineEdit(),
        infer_classes_path=DummyLineEdit(),
        sxm_color_toggle=DummyToggle(),
        log=logs.append,
        update_button_states=lambda: None,
    )

    controller = StudioController(window)
    controller._choose_files = lambda _title, _filter: [str(image_path)]
    detect_calls: list[str] = []
    controller._detect_and_display_classes = lambda: detect_calls.append("detected")
    controller.load_images()

    restored = annotation_tool.export_state()
    assert len(restored.annotations[str(image_path)]) == 1
    assert resource_loader.classes_calls == [str(tmp_path / "classes.yaml")]
    assert detect_calls == ["detected"]
    assert any("已自动加载 1 个匹配标注文件" in message for message in logs)
    assert any("已自动加载类别文件：classes.yaml" in message for message in logs)


def test_studio_controller_load_images_does_not_override_manual_classes_path(tmp_path: Path):
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(
        bytes.fromhex(
            "89504E470D0A1A0A0000000D4948445200000001000000010802000000907753DE"
            "0000000C49444154789C63F8FFFF3F0005FE02FEA7E5C9A00000000049454E44AE426082"
        )
    )
    (tmp_path / "sample.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    (tmp_path / "classes.yaml").write_text("names:\n  - atom\n", encoding="utf-8")

    annotation_state = AnnotationState(images=[str(image_path)], annotations={str(image_path): []}, class_names=["0"])
    conversion = SimpleNamespace(
        files=[str(image_path)],
        gray_files=[],
        annotation_state=annotation_state,
        sxm_metadata={},
        sxm_original_paths={},
        sxm_color_paths={},
        has_sxm_files=False,
        logs=[],
    )
    annotation_tool = DummyAnnotationTool(annotation_state)
    resource_loader = DummyResourceLoaderService()
    logs: list[str] = []
    window = SimpleNamespace(
        image_workflow_service=DummyImageWorkflowService(conversion),
        annotation_service=__import__("app.services.annotation_service", fromlist=["AnnotationService"]).AnnotationService(),
        annotation_tool=annotation_tool,
        resource_loader_service=resource_loader,
        train_classes_path=DummyLineEdit("manual-train.yaml"),
        infer_classes_path=DummyLineEdit("manual-infer.yaml"),
        sxm_color_toggle=DummyToggle(),
        log=logs.append,
        update_button_states=lambda: None,
    )

    controller = StudioController(window)
    controller._choose_files = lambda _title, _filter: [str(image_path)]
    detect_calls: list[str] = []
    controller._detect_and_display_classes = lambda: detect_calls.append("detected")
    controller.load_images()

    restored = annotation_tool.export_state()
    assert len(restored.annotations[str(image_path)]) == 1
    assert resource_loader.classes_calls == []
    assert window.train_classes_path.text() == "manual-train.yaml"
    assert window.infer_classes_path.text() == "manual-infer.yaml"
    assert detect_calls == ["detected"]
    assert not any("classes.yaml" in message for message in logs)


def test_studio_controller_load_images_auto_restores_sxm_sidecar_files_from_original_directory(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    converted_dir = tmp_path / "converted"
    raw_dir.mkdir()
    converted_dir.mkdir()

    sxm_path = raw_dir / "scan.sxm"
    sxm_path.write_text("sxm", encoding="utf-8")
    gray_path = converted_dir / "scan.png"
    gray_path.write_bytes(
        bytes.fromhex(
            "89504E470D0A1A0A0000000D4948445200000001000000010802000000907753DE"
            "0000000C49444154789C63F8FFFF3F0005FE02FEA7E5C9A00000000049454E44AE426082"
        )
    )
    (raw_dir / "scan.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    (raw_dir / "classes.yaml").write_text("names:\n  - atom\n", encoding="utf-8")

    annotation_state = AnnotationState(images=[str(gray_path)], annotations={str(gray_path): []}, class_names=["0"])
    conversion = SimpleNamespace(
        files=[str(gray_path)],
        gray_files=[],
        annotation_state=annotation_state,
        sxm_metadata={str(gray_path): {"color_path": ""}},
        sxm_original_paths={str(gray_path): str(sxm_path)},
        sxm_color_paths={},
        has_sxm_files=True,
        logs=["converted scan.sxm"],
    )
    annotation_tool = DummyAnnotationTool(annotation_state)
    resource_loader = DummyResourceLoaderService()
    logs: list[str] = []
    window = SimpleNamespace(
        image_workflow_service=DummyImageWorkflowService(conversion),
        annotation_service=__import__("app.services.annotation_service", fromlist=["AnnotationService"]).AnnotationService(),
        annotation_tool=annotation_tool,
        resource_loader_service=resource_loader,
        train_classes_path=DummyLineEdit(),
        infer_classes_path=DummyLineEdit(),
        sxm_color_toggle=DummyToggle(),
        log=logs.append,
        update_button_states=lambda: None,
    )

    controller = StudioController(window)
    controller._choose_files = lambda _title, _filter: [str(sxm_path)]
    controller._detect_and_display_classes = lambda: None
    controller.load_images()

    restored = annotation_tool.export_state()
    assert len(restored.annotations[str(gray_path)]) == 1
    assert resource_loader.classes_calls == [str(raw_dir / "classes.yaml")]
    assert any("已自动加载 1 个匹配标注文件" in message for message in logs)


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
        update_button_states=lambda: None,
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


def test_studio_controller_persists_acquisition_settings_to_config():
    config_saves: list[dict] = []
    window = SimpleNamespace(
        config_service=SimpleNamespace(
            data={},
            save=lambda payload: config_saves.append(dict(payload)),
        ),
        nano_ip_edit=DummyLineEdit("192.168.0.10"),
        nano_port_edit=DummyLineEdit("6600"),
        nano_version_edit=DummyLineEdit("20400"),
        scan_label_edit=DummyLineEdit("demo-scan"),
        scan_bias_edit=DummyLineEdit("1.8"),
        scan_setpoint_edit=DummyLineEdit("7.0e-11"),
        scan_width_edit=DummyLineEdit("12"),
        scan_height_edit=DummyLineEdit("15"),
        scan_pixels_edit=DummyLineEdit("512"),
        scan_channels_edit=DummyLineEdit("Z, Current"),
        pulse_bias_edit=DummyLineEdit("3.1"),
        pulse_width_edit=DummyLineEdit("0.2"),
    )

    controller = StudioController(window)
    controller.persist_current_acquisition_settings()

    latest = config_saves[-1]
    assert latest["recent_nanonis_ip"] == "192.168.0.10"
    assert latest["recent_nanonis_port"] == "6600"
    assert latest["recent_nanonis_version"] == "20400"
    assert latest["recent_scan_label"] == "demo-scan"
    assert latest["recent_scan_bias"] == "1.8"
    assert latest["recent_scan_setpoint"] == "7.0e-11"
    assert latest["recent_scan_width"] == "12"
    assert latest["recent_scan_height"] == "15"
    assert latest["recent_scan_pixels"] == "512"
    assert latest["recent_scan_channels"] == "Z, Current"
    assert latest["recent_pulse_bias"] == "3.1"
    assert latest["recent_pulse_width"] == "0.2"


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
        use_selected_result_btn=DummyButton(),
        open_result_dir_btn=DummyButton(),
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
    assert logs[-1] == "已切换到应用模式。"


def test_runtime_controller_on_training_finished_persists_and_applies_latest_artifacts():
    logs: list[str] = []
    appended_records: list[tuple[str, object]] = []
    enable_calls: list[str] = []
    closed_dialogs: list[str] = []
    resource_loader = DummyResourceLoaderService()
    record = SimpleNamespace(
        yolo_model_path="trained-best.pt",
        resnet_model_path="trained-best.saving",
        classes_yaml_path="trained-classes.yaml",
    )
    window = SimpleNamespace(
        log=logs.append,
        pending_training_context={"session_id": "session-1", "mode": "yolo"},
        current_session=SimpleNamespace(id="session-1"),
        training_job_service=SimpleNamespace(complete_record=lambda context, status: record),
        session_workflow_service=SimpleNamespace(
            append_training_result=lambda session_id, saved_record: appended_records.append((session_id, saved_record))
        ),
        resource_loader_service=resource_loader,
        model_manager=SimpleNamespace(),
        train_yolo_model_path=DummyLineEdit(),
        infer_yolo_model_path=DummyLineEdit(),
        train_resnet_model_path=DummyLineEdit(),
        infer_resnet_model_path=DummyLineEdit(),
        train_classes_path=DummyLineEdit(),
        infer_classes_path=DummyLineEdit(),
        training_run_status_detail=DummyLineEdit(),
        progress_bar=SimpleNamespace(hide=lambda: enable_calls.append("hide_progress")),
        loss_curve_dialog=SimpleNamespace(force_close=lambda: closed_dialogs.append("closed")),
    )

    controller = StudioRuntimeController(window)
    controller._refresh_status_labels = lambda: None
    controller.enable_controls = lambda: enable_calls.append("enable_controls")
    controller.on_training_finished()

    assert logs[0] == "训练完成。"
    assert appended_records == [("session-1", record)]
    assert resource_loader.yolo_calls == ["trained-best.pt"]
    assert resource_loader.resnet_calls == ["trained-best.saving"]
    assert resource_loader.classes_calls == ["trained-classes.yaml"]
    assert window.train_yolo_model_path.text() == "trained-best.pt"
    assert window.infer_yolo_model_path.text() == "trained-best.pt"
    assert window.train_resnet_model_path.text() == "trained-best.saving"
    assert window.infer_resnet_model_path.text() == "trained-best.saving"
    assert window.train_classes_path.text() == "trained-classes.yaml"
    assert window.infer_classes_path.text() == "trained-classes.yaml"
    assert window.pending_training_context is None
    assert enable_calls == ["hide_progress", "enable_controls"]
    assert closed_dialogs == ["closed"]


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
        use_selected_result_btn=DummyButton(),
        open_result_dir_btn=DummyButton(),
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
    assert window.use_selected_result_btn.enabled is False
    assert window.open_result_dir_btn.enabled is False
    assert "请先加载图像" in window.load_annotations_btn.tooltip
    assert "检测模型权重" in window.train_yolo_btn.tooltip
    assert "推理图像" in window.start_inference_btn.tooltip
    assert "选择一条扫描结果" in window.use_selected_result_btn.tooltip


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


def test_studio_controller_preview_path_skips_empty_scan_exports(tmp_path: Path):
    empty_png = tmp_path / "empty.png"
    valid_png = tmp_path / "valid.png"
    empty_npy = tmp_path / "empty.npy"
    valid_npy = tmp_path / "valid.npy"

    empty_png.write_bytes(
        bytes.fromhex(
            "89504E470D0A1A0A0000000D4948445200000001000000010802000000907753DE"
            "0000000C49444154789C63F8FFFF3F0005FE02FEA7E5C9A00000000049454E44AE426082"
        )
    )
    valid_png.write_bytes(
        bytes.fromhex(
            "89504E470D0A1A0A0000000D4948445200000001000000010802000000907753DE"
            "0000000C49444154789C63F8FFFF3F0005FE02FEA7E5C9A00000000049454E44AE426082"
        )
    )

    import numpy as np

    np.save(empty_npy, np.zeros((0, 0)))
    np.save(valid_npy, np.ones((4, 4)))

    controller = StudioController(SimpleNamespace())
    result = SimpleNamespace(
        saved=[
            {"png": str(empty_png), "npy": str(empty_npy)},
            {"png": str(valid_png), "npy": str(valid_npy)},
        ],
        raw={},
    )

    assert controller._preview_path_from_result(result) == str(valid_png)


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


def test_studio_controller_applies_latest_training_models_for_selected_result():
    inference_workflow = DummyInferenceWorkflowService()
    resource_loader = DummyResourceLoaderService()
    session = SimpleNamespace(
        id="session-1",
        label="demo-session",
        path="",
        scan_results=[SimpleNamespace(label="scan-a", saved=[{"sxm": "scan.sxm"}], raw={})],
        inference_results=[],
        training_results=[
            SimpleNamespace(
                yolo_model_path="trained-best.pt",
                resnet_model_path="trained-best.saving",
                classes_yaml_path="trained-classes.yaml",
                raw={},
            )
        ],
    )
    session_list = DummyListWidget()
    session_list.setCurrentRow(0)
    result_list = DummyListWidget()
    result_list.setCurrentRow(0)
    window = SimpleNamespace(
        inference_workflow_service=inference_workflow,
        inference_service=SimpleNamespace(),
        session_workflow_service=SimpleNamespace(result_labels=lambda current_session: []),
        resource_loader_service=resource_loader,
        model_manager=SimpleNamespace(),
        image_workflow_service=DummyImageWorkflowService(
            SimpleNamespace(
                files=["scan_gray.png"],
                gray_files=["scan_gray.png"],
                annotation_state=None,
                sxm_metadata={},
                sxm_original_paths={},
                sxm_color_paths={},
                has_sxm_files=False,
                logs=[],
            )
        ),
        inference_manager=DummyInferenceManager(),
        train_yolo_model_path=DummyLineEdit(),
        infer_yolo_model_path=DummyLineEdit(),
        train_resnet_model_path=DummyLineEdit(),
        infer_resnet_model_path=DummyLineEdit(),
        train_classes_path=DummyLineEdit(),
        infer_classes_path=DummyLineEdit(),
        tab_widget=SimpleNamespace(setCurrentIndex=lambda index: None),
        update_button_states=lambda: None,
        log=lambda message: None,
        session_list=session_list,
        result_list=result_list,
        session_search_edit=DummyLineEdit(""),
        session_sort_combo=DummyComboBox(["latest"]),
        session_browser_context_text=DummyTextBlock(),
        result_compare_combo=DummyComboBox(["none"]),
        result_compare_summary_text=DummyTextBlock(),
    )

    controller = StudioController(window)
    controller._visible_sessions = lambda: [session]
    controller.use_selected_result_for_inference()

    assert resource_loader.yolo_calls == ["trained-best.pt"]
    assert resource_loader.resnet_calls == ["trained-best.saving"]
    assert resource_loader.classes_calls == ["trained-classes.yaml"]


def test_studio_controller_on_session_selected_applies_session_annotation_classes(tmp_path: Path):
    resource_loader = DummyResourceLoaderService()
    classes_path = tmp_path / "session-1" / "annotation" / "classes.yaml"
    classes_path.parent.mkdir(parents=True, exist_ok=True)
    classes_path.write_text("names:\n  - atom\n", encoding="utf-8")
    session = SimpleNamespace(
        id="session-1",
        label="demo-session",
        path=str(tmp_path / "session-1"),
        scan_results=[],
        inference_results=[],
        training_results=[],
    )
    session_list = DummyListWidget()
    window = SimpleNamespace(
        session_workflow_service=SimpleNamespace(
            result_labels=lambda current_session: [],
            annotation_dir=lambda session_id: classes_path.parent,
            training_dir=lambda session_id: tmp_path / "session-1" / "training",
        ),
        resource_loader_service=resource_loader,
        model_manager=SimpleNamespace(),
        train_classes_path=DummyLineEdit(),
        infer_classes_path=DummyLineEdit(),
        result_list=DummyListWidget(),
        result_detail_text=DummyTextBlock(),
        result_preview_label=SimpleNamespace(clear=lambda: None, setText=lambda _value: None),
        result_preview_caption=DummyLineEdit(),
        result_compare_combo=DummyComboBox(["不进行对比"]),
        result_compare_summary_text=DummyTextBlock(),
        result_compare_preview_label=SimpleNamespace(clear=lambda: None, setText=lambda _value: None),
        result_compare_preview_caption=DummyLineEdit(),
        session_list=session_list,
        session_search_edit=DummyLineEdit(""),
        session_sort_combo=DummyComboBox(["latest"]),
        session_browser_context_text=DummyTextBlock(),
        log=lambda message: None,
    )

    controller = StudioController(window)
    controller._visible_sessions = lambda: [session]
    controller.on_session_selected(0)

    assert resource_loader.classes_calls == [str(classes_path)]


def test_studio_controller_on_session_selected_restores_saved_annotation_state(tmp_path: Path):
    image_path = tmp_path / "session-1" / "annotation" / "sample.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(
        bytes.fromhex(
            "89504E470D0A1A0A0000000D4948445200000001000000010802000000907753DE"
            "0000000C49444154789C63F8FFFF3F0005FE02FEA7E5C9A00000000049454E44AE426082"
        )
    )
    (image_path.parent / "sample.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    (image_path.parent / "classes.yaml").write_text("names:\n  - atom\n", encoding="utf-8")
    session = SimpleNamespace(
        id="session-1",
        label="demo-session",
        path=str(tmp_path / "session-1"),
        scan_results=[],
        inference_results=[],
        training_results=[],
    )
    annotation_tool = DummyAnnotationTool(SimpleNamespace(images=[], annotations={}, class_names=["0"]))
    window = SimpleNamespace(
        current_session=None,
        session_workflow_service=SimpleNamespace(
            result_labels=lambda current_session: [],
            annotation_dir=lambda session_id: image_path.parent,
            training_dir=lambda session_id: tmp_path / "session-1" / "training",
        ),
        annotation_service=__import__("app.services.annotation_service", fromlist=["AnnotationService"]).AnnotationService(),
        annotation_tool=annotation_tool,
        resource_loader_service=DummyResourceLoaderService(),
        model_manager=SimpleNamespace(),
        train_classes_path=DummyLineEdit(),
        infer_classes_path=DummyLineEdit(),
        class_checkboxes=[],
        class_indices=[],
        classes_layout=SimpleNamespace(addWidget=lambda widget: None),
        classes_hint_label=None,
        result_list=DummyListWidget(),
        result_detail_text=DummyTextBlock(),
        result_preview_label=SimpleNamespace(clear=lambda: None, setText=lambda _value: None),
        result_preview_caption=DummyLineEdit(),
        result_compare_combo=DummyComboBox(["不进行对比"]),
        result_compare_summary_text=DummyTextBlock(),
        result_compare_preview_label=SimpleNamespace(clear=lambda: None, setText=lambda _value: None),
        result_compare_preview_caption=DummyLineEdit(),
        session_list=DummyListWidget(),
        session_search_edit=DummyLineEdit(""),
        session_sort_combo=DummyComboBox(["latest"]),
        session_browser_context_text=DummyTextBlock(),
        log=lambda message: None,
    )

    controller = StudioController(window)
    controller._visible_sessions = lambda: [session]
    controller._detect_and_display_classes = lambda: None
    controller.on_session_selected(0)

    restored = annotation_tool.export_state()
    assert window.current_session is session
    assert len(restored.images) == 1
    assert Path(restored.images[0]).name == "sample.png"
    assert len(restored.annotations[restored.images[0]]) == 1


def test_studio_controller_on_session_selected_restores_latest_scan_when_no_saved_annotation(tmp_path: Path):
    scan_image = tmp_path / "scan-result.png"
    scan_image.write_bytes(
        bytes.fromhex(
            "89504E470D0A1A0A0000000D4948445200000001000000010802000000907753DE"
            "0000000C49444154789C63F8FFFF3F0005FE02FEA7E5C9A00000000049454E44AE426082"
        )
    )
    session = SimpleNamespace(
        id="session-1",
        label="demo-session",
        path=str(tmp_path / "session-1"),
        scan_results=[SimpleNamespace(label="scan-a", saved=[{"png": str(scan_image)}], raw={})],
        inference_results=[],
        training_results=[],
    )
    annotation_tool = DummyAnnotationTool(SimpleNamespace(images=[], annotations={}, class_names=["0"]))
    window = SimpleNamespace(
        current_session=None,
        session_workflow_service=SimpleNamespace(
            result_labels=lambda current_session: [],
            result_detail=lambda current_session, index: "",
            annotation_dir=lambda session_id: tmp_path / "session-1" / "annotation",
            training_dir=lambda session_id: tmp_path / "session-1" / "training",
        ),
        annotation_service=__import__("app.services.annotation_service", fromlist=["AnnotationService"]).AnnotationService(),
        annotation_tool=annotation_tool,
        resource_loader_service=DummyResourceLoaderService(),
        model_manager=SimpleNamespace(),
        train_classes_path=DummyLineEdit(),
        infer_classes_path=DummyLineEdit(),
        class_checkboxes=[],
        class_indices=[],
        classes_layout=SimpleNamespace(addWidget=lambda widget: None),
        classes_hint_label=None,
        update_button_states=lambda: None,
        result_list=DummyListWidget(),
        result_detail_text=DummyTextBlock(),
        result_preview_label=SimpleNamespace(clear=lambda: None, setText=lambda _value: None),
        result_preview_caption=DummyLineEdit(),
        result_compare_combo=DummyComboBox(["不进行对比"]),
        result_compare_summary_text=DummyTextBlock(),
        result_compare_preview_label=SimpleNamespace(clear=lambda: None, setText=lambda _value: None),
        result_compare_preview_caption=DummyLineEdit(),
        session_list=DummyListWidget(),
        session_search_edit=DummyLineEdit(""),
        session_sort_combo=DummyComboBox(["latest"]),
        session_browser_context_text=DummyTextBlock(),
        log=lambda message: None,
    )

    controller = StudioController(window)
    controller._visible_sessions = lambda: [session]
    controller._detect_and_display_classes = lambda: None
    controller._update_result_preview = lambda result: None
    controller._update_compare_preview = lambda: None
    controller.on_session_selected(0)

    restored = annotation_tool.export_state()
    assert window.current_session is session
    assert restored.images == [str(scan_image)]


def test_studio_controller_on_session_selected_selects_latest_result_and_updates_detail(tmp_path: Path):
    first_image = tmp_path / "scan-a.png"
    second_image = tmp_path / "scan-b.png"
    for path in (first_image, second_image):
        path.write_bytes(
            bytes.fromhex(
                "89504E470D0A1A0A0000000D4948445200000001000000010802000000907753DE"
                "0000000C49444154789C63F8FFFF3F0005FE02FEA7E5C9A00000000049454E44AE426082"
            )
        )

    session = SimpleNamespace(
        id="session-1",
        label="demo-session",
        path=str(tmp_path / "session-1"),
        scan_results=[
            SimpleNamespace(label="scan-a", saved=[{"png": str(first_image)}], raw={}),
            SimpleNamespace(label="scan-b", saved=[{"png": str(second_image)}], raw={}),
        ],
        inference_results=[],
        training_results=[],
    )
    result_list = DummyListWidget()
    preview_text: list[str] = []
    window = SimpleNamespace(
        current_session=None,
        session_workflow_service=SimpleNamespace(
            result_labels=lambda current_session: ["1. scan-a", "2. scan-b"],
            result_detail=lambda current_session, index: f"detail-{index}",
            annotation_dir=lambda session_id: tmp_path / "session-1" / "annotation",
            training_dir=lambda session_id: tmp_path / "session-1" / "training",
        ),
        annotation_service=DummyAnnotationService(),
        annotation_tool=DummyAnnotationTool(SimpleNamespace(images=[], annotations={}, class_names=["0"])),
        resource_loader_service=DummyResourceLoaderService(),
        model_manager=SimpleNamespace(),
        train_classes_path=DummyLineEdit(),
        infer_classes_path=DummyLineEdit(),
        class_checkboxes=[],
        class_indices=[],
        classes_layout=SimpleNamespace(addWidget=lambda widget: None),
        classes_hint_label=None,
        update_button_states=lambda: None,
        result_list=result_list,
        result_detail_text=DummyTextBlock(),
        result_preview_label=SimpleNamespace(clear=lambda: None, setText=lambda value: preview_text.append(value), setPixmap=lambda pixmap: None),
        result_preview_caption=DummyLineEdit(),
        result_compare_combo=DummyComboBox(["不进行对比"]),
        result_compare_summary_text=DummyTextBlock(),
        result_compare_preview_label=SimpleNamespace(clear=lambda: None, setText=lambda _value: None),
        result_compare_preview_caption=DummyLineEdit(),
        session_list=DummyListWidget(),
        session_search_edit=DummyLineEdit(""),
        session_sort_combo=DummyComboBox(["latest"]),
        session_browser_context_text=DummyTextBlock(),
        log=lambda message: None,
    )

    controller = StudioController(window)
    controller._visible_sessions = lambda: [session]
    controller._detect_and_display_classes = lambda: None
    controller._restore_session_workspace = lambda _session: None
    controller._update_compare_preview = lambda: None
    controller.on_session_selected(0)

    assert result_list.currentRow() == 1
    assert window.result_detail_text.toPlainText() == "detail-1"


def test_studio_controller_start_inference_auto_loads_default_models_when_missing(tmp_path: Path):
    inference_workflow = DummyInferenceWorkflowService()
    inference_service_calls: list[dict] = []
    logs: list[str] = []
    default_yolo = tmp_path / "assets" / "models" / "best.pt"
    default_resnet = tmp_path / "assets" / "models" / "demo.saving"
    default_classes = tmp_path / "assets" / "models" / "classes.yaml"
    default_yolo.parent.mkdir(parents=True, exist_ok=True)
    default_yolo.write_text("yolo", encoding="utf-8")
    default_resnet.write_text("resnet", encoding="utf-8")
    default_classes.write_text("names: ['atom']\n", encoding="utf-8")

    resource_loader = __import__(
        "app.services.resource_loader_service",
        fromlist=["ResourceLoaderService"],
    ).ResourceLoaderService()
    model_manager = __import__("app.utils.model_manager", fromlist=["ModelManager"]).ModelManager()
    window = SimpleNamespace(
        inference_workflow_service=inference_workflow,
        session_workflow_service=DummySessionWorkflowForController(),
        resource_loader_service=resource_loader,
        runtime=SimpleNamespace(
            paths=SimpleNamespace(
                default_yolo_model_path=default_yolo,
                default_resnet_model_path=default_resnet,
                default_classes_path=default_classes,
            )
        ),
        model_manager=model_manager,
        inference_manager=SimpleNamespace(
            images=["img_a.png"],
            actual_output_dir="actual-dir",
            get_inference_results=lambda output_dir: ["result_a.png"],
        ),
        infer_classes_path=DummyLineEdit(""),
        train_classes_path=DummyLineEdit(""),
        train_yolo_model_path=DummyLineEdit(""),
        infer_yolo_model_path=DummyLineEdit(""),
        train_resnet_model_path=DummyLineEdit(""),
        infer_resnet_model_path=DummyLineEdit(""),
        scan_label_edit=DummyLineEdit("infer-demo"),
        inference_service=SimpleNamespace(
            infer_files=lambda **kwargs: inference_service_calls.append(kwargs),
        ),
        progress_bar=SimpleNamespace(show=lambda: None, setValue=lambda value: None),
        disable_controls=lambda: None,
        current_session=SimpleNamespace(label="demo", id="session-1", scan_results=[], training_results=[], inference_results=[]),
        pending_inference_session_id=None,
        pending_inference_output_dir=None,
        session_list=DummyListWidget(),
        result_list=DummyListWidget(),
        session_search_edit=DummyLineEdit(""),
        session_sort_combo=DummyComboBox(["latest"]),
        session_browser_context_text=DummyTextBlock(),
        result_compare_combo=DummyComboBox(["none"]),
        result_compare_summary_text=DummyTextBlock(),
        log=logs.append,
    )

    controller = StudioController(window)
    controller.start_inference()

    assert window.model_manager.yolo_model == str(default_yolo)
    assert window.model_manager.resnet_model == str(default_resnet)
    assert window.infer_classes_path.text() == str(default_classes)
    assert inference_service_calls[0]["yolo_model"] == str(default_yolo)
    assert inference_service_calls[0]["resnet_model"] == str(default_resnet)


def test_studio_controller_start_inference_requires_explicit_images_when_queue_empty():
    inference_workflow = DummyInferenceWorkflowService()
    inference_service_calls: list[dict] = []
    session = SimpleNamespace(
        id="session-1",
        label="demo-session",
        scan_results=[SimpleNamespace(label="scan-a", saved=[{"png": "scan_a.png"}], raw={})],
        training_results=[],
        inference_results=[],
    )
    window = SimpleNamespace(
        inference_workflow_service=inference_workflow,
        session_workflow_service=DummySessionWorkflowForController(),
        resource_loader_service=DummyResourceLoaderService(),
        model_manager=SimpleNamespace(yolo_model="yolo.pt", resnet_model="resnet.pth"),
        inference_manager=DummyInferenceManager(),
        infer_classes_path=DummyLineEdit("classes.yaml"),
        train_classes_path=DummyLineEdit(),
        train_yolo_model_path=DummyLineEdit(),
        infer_yolo_model_path=DummyLineEdit(),
        train_resnet_model_path=DummyLineEdit(),
        infer_resnet_model_path=DummyLineEdit(),
        scan_label_edit=DummyLineEdit("infer-demo"),
        inference_service=SimpleNamespace(
            infer_files=lambda **kwargs: inference_service_calls.append(kwargs),
        ),
        progress_bar=SimpleNamespace(show=lambda: None, setValue=lambda value: None),
        disable_controls=lambda: None,
        current_session=session,
        pending_inference_session_id=None,
        pending_inference_output_dir=None,
        session_list=DummyListWidget(),
        result_list=DummyListWidget(),
        session_search_edit=DummyLineEdit(""),
        session_sort_combo=DummyComboBox(["latest"]),
        session_browser_context_text=DummyTextBlock(),
        result_compare_combo=DummyComboBox(["none"]),
        result_compare_summary_text=DummyTextBlock(),
        log=lambda message: None,
        update_button_states=lambda: None,
    )

    controller = StudioController(window)
    controller.start_inference()

    assert inference_workflow.start_calls == []
    assert inference_service_calls == []
    assert window.inference_manager.images == []


def test_studio_controller_save_annotations_uses_session_annotation_dir(tmp_path: Path):
    state = SimpleNamespace(images=["img.png"], annotations={"img.png": ["box"]}, class_names=["atom"])
    annotation_tool = DummyAnnotationTool(state)
    annotation_service = DummyAnnotationIoService()
    resource_loader = DummyResourceLoaderService()
    logs: list[str] = []

    annotation_dir = tmp_path / "session-1" / "annotation"
    session_workflow = SimpleNamespace(
        ensure_session=lambda current_session, label: current_session,
        annotation_dir=lambda session_id: annotation_dir,
    )
    window = SimpleNamespace(
        current_session=SimpleNamespace(id="session-1", label="demo"),
        session_workflow_service=session_workflow,
        annotation_tool=annotation_tool,
        annotation_service=annotation_service,
        resource_loader_service=resource_loader,
        train_classes_path=DummyLineEdit(),
        infer_classes_path=DummyLineEdit(),
        log=logs.append,
        update_button_states=lambda: None,
    )

    controller = StudioController(window)
    controller.save_annotations()

    assert annotation_service.save_calls == [(state, str(annotation_dir))]
    assert resource_loader.classes_calls == [str(annotation_dir / "classes.yaml")]
    assert any("标注已保存到" in message for message in logs)


def test_studio_controller_crop_resnet_dataset_uses_session_training_dir(tmp_path: Path):
    state = SimpleNamespace(images=["img.png"], annotations={"img.png": ["box"]}, class_names=["atom"])
    annotation_tool = DummyAnnotationTool(state)
    dataset_service = DummyDatasetExportService()
    resource_loader = DummyResourceLoaderService()
    logs: list[str] = []

    training_dir = tmp_path / "session-1" / "training"
    session_workflow = SimpleNamespace(
        ensure_session=lambda current_session, label: current_session,
        training_dir=lambda session_id: training_dir,
    )
    window = SimpleNamespace(
        current_session=SimpleNamespace(id="session-1", label="demo"),
        session_workflow_service=session_workflow,
        annotation_tool=annotation_tool,
        annotation_service=DummyAnnotationService(),
        dataset_service=dataset_service,
        resource_loader_service=resource_loader,
        train_classes_path=DummyLineEdit(),
        infer_classes_path=DummyLineEdit(),
        log=logs.append,
        _get_gray_path=lambda path: path,
    )

    controller = StudioController(window)
    controller.crop_resnet_dataset()

    expected_dir = training_dir / "resnet_crop_export"
    assert dataset_service.crop_calls == [(str(expected_dir), {0: "atom"})]
    assert resource_loader.classes_calls == [str(expected_dir / "classes.yaml")]
    assert (expected_dir / "classes.yaml").exists()
    assert any("分类裁剪完成" in message for message in logs)
