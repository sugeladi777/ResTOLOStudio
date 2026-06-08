from __future__ import annotations

import os
import threading

from PyQt5.QtWidgets import QCheckBox, QFileDialog, QMessageBox

from app.core import ScanResultRecord, SessionRecord
from app.windows.studio_ui import BASE_COLOR, BORDER_COLOR, PANEL_BG, TEXT_COLOR


class StudioController:
    """Coordinates acquisition, sessions, and inference entrypoints."""

    def __init__(self, window):
        self.window = window

    def _nanonis_config(self):
        return self.window.acquisition_workflow_service.build_connection_config(
            self.window.nano_ip_edit.text(),
            self.window.nano_port_edit.text(),
            self.window.nano_version_edit.text(),
        )

    def _run_service(self, key: str, fn) -> None:
        def worker():
            try:
                result = fn()
                self.window.service_result_signal.emit(key, result)
            except Exception as exc:  # noqa: BLE001
                self.window.service_error_signal.emit(key, str(exc))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def _scan_geometry(self) -> dict:
        return self.window.acquisition_workflow_service.build_scan_geometry(
            self.window.scan_width_edit.text(),
            self.window.scan_height_edit.text(),
            self.window.scan_pixels_edit.text(),
            self.window.scan_channels_edit.text(),
        ).to_dict()

    def _all_sessions(self) -> list[SessionRecord]:
        return self.window.session_workflow_service.list_sessions()

    def _selected_session(self) -> SessionRecord | None:
        return self.window.session_workflow_service.selected_session(self.window.session_list.currentRow())

    def _selected_scan_result(self) -> ScanResultRecord | None:
        _, result = self.window.session_workflow_service.selected_scan_result(
            self.window.session_list.currentRow(),
            self.window.result_list.currentRow(),
        )
        return result

    def _refresh_result_list(self, session: SessionRecord | None) -> None:
        self.window.result_list.clear()
        self.window.result_detail_text.clear()
        for label in self.window.session_workflow_service.result_labels(session):
            self.window.result_list.addItem(label)

    def _set_pending_scan_results(self, *results: ScanResultRecord) -> None:
        self.window.pending_scan_results = list(results)
        self.reload_sessions()

    def _confirm(self, title: str, message: str) -> bool:
        return QMessageBox.question(self.window, title, message) == QMessageBox.Yes

    def _new_session(self) -> SessionRecord:
        self.window.current_session = self.window.acquisition_workflow_service.start_scan_session(
            getattr(self.window, "current_session", None),
            self.window.scan_label_edit.text(),
            self.window.nanonis_service,
        )
        return self.window.current_session

    def _apply_image_load_result(self, load_result, load_inference_images: bool = False) -> list[str]:
        self.window.sxm_metadata = load_result.sxm_metadata
        self.window.sxm_original_paths = load_result.sxm_original_paths
        self.window.sxm_color_paths = load_result.sxm_color_paths
        self.window._has_sxm_files = load_result.has_sxm_files

        for message in load_result.logs:
            self.window.log(message)

        if hasattr(self.window, "sxm_color_toggle"):
            self.window.sxm_color_toggle.setVisible(load_result.has_sxm_files)

        if load_result.annotation_state is not None and hasattr(self.window, "annotation_tool") and self.window.annotation_tool:
            self.window.annotation_tool.load_state(load_result.annotation_state)

        if load_inference_images:
            self.window.inference_manager.load_images(load_result.gray_files)

        return load_result.files

    def convert_sxm_files(self, files: list[str]) -> list[str]:
        load_result = self.window.image_workflow_service.convert_for_annotation(files)
        return self._apply_image_load_result(load_result)

    def set_sxm_color_mode(self, state) -> None:
        use_color = bool(state)
        if not getattr(self.window, "sxm_metadata", None):
            return

        if hasattr(self.window, "annotation_tool") and hasattr(self.window.annotation_tool, "export_state"):
            current_state = self.window.annotation_tool.export_state()
            updated_state = self.window.image_workflow_service.remap_sxm_display(
                current_state,
                self.window.sxm_metadata,
                use_color=use_color,
            )
            if updated_state.images != current_state.images or updated_state.annotations != current_state.annotations:
                self.window.annotation_tool.load_state(updated_state)

        self.window.log(f"SXM display mode: {'color' if use_color else 'grayscale'}")

    def get_gray_path(self, image_path: str) -> str:
        return self.window.image_workflow_service.gray_path_for(
            image_path,
            getattr(self.window, "sxm_metadata", {}),
        )

    def _detect_and_display_classes(self) -> None:
        summary = self.window.dataset_service.annotation_class_summary(self.window.annotation_tool)
        if not summary.used_classes:
            return

        for checkbox in self.window.class_checkboxes:
            checkbox.deleteLater()
        self.window.class_checkboxes.clear()
        self.window.class_indices = []

        if getattr(self.window, "classes_hint_label", None):
            self.window.classes_hint_label.deleteLater()
            self.window.classes_hint_label = None

        for index, class_name in enumerate(summary.class_names):
            if index not in summary.used_classes:
                continue
            checkbox = QCheckBox(class_name)
            checkbox.setChecked(True)
            checkbox.setStyleSheet(
                f"""
                    QCheckBox {{
                        color: {TEXT_COLOR};
                        spacing: 8px;
                        background-color: transparent;
                    }}
                    QCheckBox::indicator {{
                        width: 16px;
                        height: 16px;
                        border-radius: 3px;
                        border: 2px solid {BORDER_COLOR};
                        background-color: {PANEL_BG};
                    }}
                    QCheckBox::indicator:checked {{
                        background-color: {BASE_COLOR};
                        border-color: {BASE_COLOR};
                    }}
                """
            )
            self.window.classes_layout.addWidget(checkbox)
            self.window.class_checkboxes.append(checkbox)
            self.window.class_indices.append(index)

        self.window.log(f"Detected {len(self.window.class_checkboxes)} annotated classes")

    def apply_default_model_paths(self) -> None:
        self.window.resource_loader_service.apply_default_model_paths(
            self.window.runtime.paths,
            self.window.model_manager,
            self.window.train_yolo_model_path,
            self.window.infer_yolo_model_path,
            self.window.train_resnet_model_path,
            self.window.infer_resnet_model_path,
            self.window.infer_classes_path,
        )

    def load_infer_images(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self.window,
            "Select inference images",
            "",
            "Image files (*.jpg *.jpeg *.png *.bmp *.sxm)",
        )
        if not files:
            return

        load_result = self.window.image_workflow_service.convert_for_inference(files)
        files = self._apply_image_load_result(load_result, load_inference_images=True)
        self.window.log(f"Loaded {len(files)} inference images")
        self.window.update_button_states()

    def load_images(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self.window,
            "Select images",
            "",
            "Image files (*.jpg *.jpeg *.png *.bmp *.sxm)",
        )
        if not files:
            return
        files = self.convert_sxm_files(files)
        self.window.log(f"Loaded {len(files)} images")
        self.window.update_button_states()

    def load_annotations(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self.window,
            "Select annotations",
            "",
            "Text files (*.txt)",
        )
        if not files:
            return
        self.window.log(f"Loaded {len(files)} annotation files")
        state = self.window.annotation_tool.export_state()
        updated_state = self.window.annotation_service.load_annotation_files(state, files)
        self.window.annotation_tool.load_state(updated_state)
        self.window.update_button_states()
        self._detect_and_display_classes()

    def detect_and_display_classes(self) -> None:
        self._detect_and_display_classes()

    def save_annotations(self) -> None:
        directory = QFileDialog.getExistingDirectory(self.window, "Select output directory")
        if not directory:
            return
        state = self.window.annotation_tool.export_state()
        self.window.annotation_service.save_annotations(state, directory)
        self.window.log(f"Saved annotations to {directory}")
        self.window.update_button_states()

    def crop_resnet_dataset(self) -> None:
        directory = QFileDialog.getExistingDirectory(self.window, "Select output directory")
        if not directory:
            return

        self.window.log("Preparing cropped ResNet dataset...")
        state = self.window.annotation_tool.export_state()
        if not self.window.annotation_service.has_images(state):
            self.window.log("Error: load training images first")
            return
        if not self.window.annotation_service.has_annotations(state):
            self.window.log("Error: load or create annotations first")
            return

        summary = self.window.dataset_service.annotation_class_summary(self.window.annotation_tool)
        if summary.used_classes:
            class_names = [str(index) for index in range(max(summary.used_classes) + 1)]
            self.window.log(f"Using adaptive classes: {class_names}")
        else:
            class_names = state.class_names
            self.window.log(f"No annotations found, using default classes: {class_names}")

        selected_class_map = {idx: str(class_names[idx]) for idx in summary.used_classes if idx < len(class_names)}
        crop_summary = self.window.dataset_service.crop_resnet_dataset(
            self.window.annotation_tool,
            self.window._get_gray_path,
            directory,
            selected_class_map,
        )
        self.window.log(f"Annotated classes present: {list(crop_summary.actual_classes)}")
        self.window.log(f"ResNet dataset crop complete, total crops: {crop_summary.crop_count}")

    def load_yolo_model(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self.window,
            "Select YOLO model",
            "",
            "Model files (*.pt *.pth)",
        )
        loaded = self.window.resource_loader_service.load_yolo_model(
            file_path,
            self.window.model_manager,
            self.window.log,
            getattr(self.window, "train_yolo_model_path", None),
            getattr(self.window, "infer_yolo_model_path", None),
        )
        if loaded:
            self.window.update_button_states()

    def load_resnet_data(self) -> None:
        directory = QFileDialog.getExistingDirectory(self.window, "Select ResNet dataset directory")
        loaded, _ = self.window.resource_loader_service.load_resnet_dataset(
            directory,
            self.window.log,
            getattr(self.window, "train_resnet_data_path", None),
        )
        if loaded:
            self.window.update_button_states()

    def load_classes_yaml(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self.window,
            "Select classes YAML",
            "",
            "YAML files (*.yaml)",
        )
        loaded = self.window.resource_loader_service.load_classes_file(
            file_path,
            self.window.log,
            getattr(self.window, "train_classes_path", None),
            getattr(self.window, "infer_classes_path", None),
        )
        if loaded:
            self.window.update_button_states()

    def load_resnet_model(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self.window,
            "Select ResNet model",
            "",
            "Model files (*.saving *.pt *.pth)",
        )
        loaded = self.window.resource_loader_service.load_resnet_model(
            file_path,
            self.window.model_manager,
            self.window.log,
            getattr(self.window, "train_resnet_model_path", None),
            getattr(self.window, "infer_resnet_model_path", None),
        )
        if loaded:
            self.window.update_button_states()

    def load_classes_file(self) -> None:
        self.load_classes_yaml()

    def connect_nanonis(self) -> None:
        self.window.log("Connecting to Nanonis...")
        self._run_service("connect_nanonis", lambda: self.window.nanonis_service.connect(self._nanonis_config()))

    def disconnect_nanonis(self) -> None:
        self.window.nanonis_service.disconnect()
        self.window.nano_status_text.setPlainText("Disconnected")
        self.window.log("Disconnected from Nanonis")

    def refresh_nanonis_status(self) -> None:
        self._run_service("refresh_nanonis", self.window.nanonis_service.status)

    def set_nanonis_bias(self) -> None:
        bias_v = float(self.window.scan_bias_edit.text().strip())
        self._run_service("set_bias", lambda: self.window.nanonis_service.set_bias(bias_v))

    def set_nanonis_setpoint(self) -> None:
        setpoint_a = float(self.window.scan_setpoint_edit.text().strip())
        self._run_service("set_setpoint", lambda: self.window.nanonis_service.set_setpoint(setpoint_a))

    def set_nanonis_feedback(self, enabled: bool) -> None:
        if not enabled and not self._confirm("Confirm", "Turning off feedback may be risky. Continue?"):
            return
        self._run_service("set_feedback", lambda: self.window.nanonis_service.set_feedback(enabled))

    def apply_nanonis_scan(self) -> None:
        self._run_service(
            "apply_scan",
            lambda: self.window.nanonis_service.apply_scan(**self._scan_geometry()),
        )

    def scan_and_save_from_nanonis(self) -> None:
        session = self._new_session()
        self.window.log(f"Starting scan for session: {session.id}")
        self._run_service(
            "scan_and_save",
            lambda: self.window.nanonis_service.scan_and_save(
                label=session.id,
                **self._scan_geometry(),
            ),
        )

    def run_scan_pulse_scan_workflow(self) -> None:
        if not self._confirm("Confirm", "Run pre-scan, bias pulse, and post-scan workflow?"):
            return

        session = self._new_session()
        self._run_service(
            "scan_pulse_scan",
            lambda: self.window.workflow_service.scan_pulse_scan(
                label=session.id,
                **self._scan_geometry(),
                pulse_bias_v=float(self.window.pulse_bias_edit.text().strip()),
                pulse_width_s=float(self.window.pulse_width_edit.text().strip()),
            ),
        )

    def start_inference(self) -> None:
        if not self.window.model_manager.yolo_model or not self.window.model_manager.resnet_model:
            self.window.log("Error: please load YOLO and ResNet models first")
            return
        if not self.window.inference_manager.images:
            self.window.log("Error: please load inference images first")
            return

        context = self.window.inference_workflow_service.prepare_inference_start(
            self.window.current_session,
            self.window.scan_label_edit.text(),
            self.window.infer_classes_path.text(),
        )
        self.window.current_session = context.session
        self.window.pending_inference_session_id = context.session.id
        self.window.pending_inference_output_dir = context.output_dir

        self.window.log(f"Starting inference for session: {context.session.id}")
        self.window.inference_service.infer_files(
            files=self.window.inference_manager.images,
            yolo_model=self.window.model_manager.yolo_model,
            resnet_model=self.window.model_manager.resnet_model,
            output_dir=context.output_dir,
            classes_yaml=context.classes_yaml,
        )
        self.window.progress_bar.show()
        self.window.progress_bar.setValue(0)
        self.window.disable_controls()

    def use_selected_result_for_inference(self) -> None:
        session = self._selected_session()
        selected_result = self._selected_scan_result()
        selection = self.window.inference_workflow_service.select_scan_result_for_inference(
            session,
            selected_result,
            self.window.inference_service,
        )
        if selection is None:
            return

        self.window.current_session = selection.session
        load_result = self.window.image_workflow_service.convert_for_inference(selection.files)
        self._apply_image_load_result(load_result, load_inference_images=True)
        self.window.tab_widget.setCurrentIndex(2)
        self.window.log(f"Loaded {len(selection.files)} scan result images for inference")
        self.window.update_button_states()

    def on_inference_finished(self) -> None:
        if self.window.pending_inference_session_id and self.window.pending_inference_output_dir:
            results = self.window.inference_manager.get_inference_results(
                str(self.window.pending_inference_output_dir)
            )
            self.window.inference_workflow_service.persist_inference_result(
                self.window.pending_inference_session_id,
                self.window.pending_inference_output_dir,
                results,
                self.window.inference_manager.actual_output_dir or "",
            )
            self.reload_sessions()
            self.window.pending_inference_session_id = None
            self.window.pending_inference_output_dir = None

    def reload_sessions(self) -> None:
        self.window.session_list.clear()
        sessions = self._all_sessions()
        for label in self.window.session_workflow_service.session_list_labels():
            self.window.session_list.addItem(label)
        if sessions:
            self.window.session_list.setCurrentRow(0)

    def on_session_selected(self, index: int) -> None:
        self._refresh_result_list(self.window.session_workflow_service.selected_session(index))

    def on_result_selected(self, index: int) -> None:
        session = self._selected_session()
        detail = self.window.session_workflow_service.result_detail(session, index)
        if not detail:
            self.window.result_detail_text.clear()
            return
        self.window.result_detail_text.setPlainText(detail)

    def handle_service_result(self, key: str, result: object) -> None:
        if key in {"connect_nanonis", "refresh_nanonis", "set_bias", "set_setpoint", "set_feedback", "apply_scan"}:
            self.window.nano_status_text.setPlainText(str(result))
            self.window.log(f"{key} completed")
            return

        if key == "scan_and_save":
            record = self.window.acquisition_workflow_service.append_scan_result(
                self.window.current_session.id,
                result,
            )
            self._set_pending_scan_results(record)
            self.window.log(f"Scan completed: {record.label}")
            return

        if key == "scan_pulse_scan":
            pre_scan, post_scan = self.window.acquisition_workflow_service.append_scan_workflow_results(
                self.window.current_session.id,
                result,
            )
            self._set_pending_scan_results(pre_scan, post_scan)
            self.window.log("Workflow completed: pre-scan -> pulse -> post-scan")

    def handle_service_error(self, key: str, message: str) -> None:
        self.window.log(f"{key} failed: {message}")
        QMessageBox.warning(self.window, "Operation Failed", f"{key} failed:\n{message}")
