import os
import threading

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QCheckBox, QFileDialog, QMessageBox

from app.core import ScanResultRecord, SessionRecord
from app.windows.studio_ui import BASE_COLOR, BORDER_COLOR, PANEL_BG, TEXT_COLOR


class StudioController:
    """Coordinates file loading, acquisition, sessions, and inference actions."""

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

        threading.Thread(target=worker, daemon=True).start()

    def _scan_geometry(self) -> dict:
        return self.window.acquisition_workflow_service.build_scan_geometry(
            self.window.scan_width_edit.text(),
            self.window.scan_height_edit.text(),
            self.window.scan_pixels_edit.text(),
            self.window.scan_channels_edit.text(),
        ).to_dict()

    def _all_sessions(self) -> list[SessionRecord]:
        return self.window.session_workflow_service.list_sessions()

    def _session_filter_text(self) -> str:
        search_edit = getattr(self.window, "session_search_edit", None)
        if search_edit is None:
            return ""
        return search_edit.text().strip().lower()

    def _session_matches_filter(self, session: SessionRecord, filter_text: str) -> bool:
        if not filter_text:
            return True
        searchable = [
            getattr(session, "id", ""),
            getattr(session, "label", ""),
            f"scan={len(getattr(session, 'scan_results', []) or [])}",
            f"infer={len(getattr(session, 'inference_results', []) or [])}",
            f"train={len(getattr(session, 'training_results', []) or [])}",
        ]
        if getattr(session, "training_results", []) or []:
            searchable.append("训练")
        if getattr(session, "inference_results", []) or []:
            searchable.append("推理")
        if getattr(session, "scan_results", []) or []:
            searchable.append("扫描")
        haystack = " ".join(str(item).lower() for item in searchable)
        return filter_text in haystack

    def _visible_sessions(self) -> list[SessionRecord]:
        sessions = [session for session in self._all_sessions() if self._session_matches_filter(session, self._session_filter_text())]
        sort_text = getattr(getattr(self.window, "session_sort_combo", None), "currentText", lambda: "最近活跃优先")()
        if sort_text == "扫描结果最多":
            sessions.sort(key=lambda item: len(getattr(item, "scan_results", []) or []), reverse=True)
        elif sort_text == "训练结果最多":
            sessions.sort(key=lambda item: len(getattr(item, "training_results", []) or []), reverse=True)
        elif sort_text == "推理结果最多":
            sessions.sort(key=lambda item: len(getattr(item, "inference_results", []) or []), reverse=True)
        else:
            sessions.sort(
                key=lambda item: (
                    len(getattr(item, "inference_results", []) or []),
                    len(getattr(item, "training_results", []) or []),
                    len(getattr(item, "scan_results", []) or []),
                ),
                reverse=True,
            )
        return sessions

    def _selected_session(self) -> SessionRecord | None:
        sessions = self._visible_sessions()
        index = self.window.session_list.currentRow()
        if index < 0 or index >= len(sessions):
            return None
        return sessions[index]

    def _selected_scan_result(self) -> ScanResultRecord | None:
        session = self._selected_session()
        if session is None:
            return None
        result_index = self.window.result_list.currentRow()
        if result_index < 0 or result_index >= len(getattr(session, "scan_results", []) or []):
            return None
        return session.scan_results[result_index]

    def _refresh_result_list(self, session: SessionRecord | None) -> None:
        self.window.result_list.clear()
        self.window.result_detail_text.clear()

        compare_combo = getattr(self.window, "result_compare_combo", None)
        if compare_combo is not None:
            compare_combo.blockSignals(True)
            compare_combo.clear()
            compare_combo.addItem("不进行对比")
            summary_text = getattr(self.window, "result_compare_summary_text", None)
            if summary_text is not None:
                summary_text.clear()

        labels = self.window.session_workflow_service.result_labels(session)
        for index, label in enumerate(labels):
            self.window.result_list.addItem(label)
            if compare_combo is not None:
                compare_combo.addItem(f"{index + 1}. {label}")

        if compare_combo is not None:
            compare_combo.blockSignals(False)

        preview_label = getattr(self.window, "result_preview_label", None)
        if preview_label is not None:
            preview_label.clear()
            preview_label.setText("尚未选择结果")

        preview_caption = getattr(self.window, "result_preview_caption", None)
        if preview_caption is not None:
            preview_caption.setText("选择扫描结果后，这里会显示预览。")

        compare_preview_label = getattr(self.window, "result_compare_preview_label", None)
        if compare_preview_label is not None:
            compare_preview_label.clear()
            compare_preview_label.setText("未选择对比结果")

        compare_preview_caption = getattr(self.window, "result_compare_preview_caption", None)
        if compare_preview_caption is not None:
            compare_preview_caption.setText("选择对比结果后，这里会显示另一条结果的预览。")

    def _build_result_comparison_summary(self) -> str:
        session = self._selected_session()
        current_result = self._selected_scan_result()
        compare_combo = getattr(self.window, "result_compare_combo", None)
        if session is None or current_result is None or compare_combo is None:
            return "先选择会话和扫描结果，再查看对比摘要。"

        compare_index = compare_combo.currentIndex() - 1
        results = getattr(session, "scan_results", []) or []
        current_index = self.window.result_list.currentRow()
        if compare_index < 0 or compare_index >= len(results):
            return "当前显示的是已选结果。若同一会话里还有其他结果，可在上方选择对比。"
        if compare_index == current_index:
            return "当前对比对象与已选结果相同，请选择另一条扫描结果。"

        other = results[compare_index]
        current_saved = len(getattr(current_result, "saved", []) or [])
        other_saved = len(getattr(other, "saved", []) or [])
        current_channels = ", ".join(
            str(item)
            for item in (
                (getattr(current_result, "raw", {}) or {}).get("channels")
                or (getattr(current_result, "raw", {}) or {}).get("scan_channels")
                or []
            )
        ) or "未记录"
        other_channels = ", ".join(
            str(item)
            for item in (
                (getattr(other, "raw", {}) or {}).get("channels")
                or (getattr(other, "raw", {}) or {}).get("scan_channels")
                or []
            )
        ) or "未记录"
        return (
            f"当前结果：{getattr(current_result, 'label', '未命名')}\n"
            f"对比结果：{getattr(other, 'label', '未命名')}\n"
            f"导出文件：{current_saved} vs {other_saved}\n"
            f"采集通道：{current_channels} vs {other_channels}"
        )

    def _preview_path_from_result(self, result: ScanResultRecord | None) -> str | None:
        if result is None:
            return None
        for item in getattr(result, "saved", []) or []:
            for key in ("png", "jpg", "jpeg", "bmp"):
                path = item.get(key)
                if path and os.path.exists(path):
                    return path
        raw = getattr(result, "raw", {}) or {}
        for key in ("png", "jpg", "image", "image_path"):
            path = raw.get(key)
            if path and os.path.exists(path):
                return path
        return None

    def _set_result_preview_placeholder(self, message: str, caption: str | None = None) -> None:
        preview_label = getattr(self.window, "result_preview_label", None)
        if preview_label is None:
            return
        preview_label.clear()
        preview_label.setText(message)
        preview_caption = getattr(self.window, "result_preview_caption", None)
        if preview_caption is not None and caption is not None:
            preview_caption.setText(caption)

    def _set_compare_preview_placeholder(self, message: str, caption: str | None = None) -> None:
        compare_label = getattr(self.window, "result_compare_preview_label", None)
        if compare_label is None:
            return
        compare_label.clear()
        compare_label.setText(message)
        compare_caption = getattr(self.window, "result_compare_preview_caption", None)
        if compare_caption is not None and caption is not None:
            compare_caption.setText(caption)

    def _update_compare_preview(self) -> None:
        compare_label = getattr(self.window, "result_compare_preview_label", None)
        compare_caption = getattr(self.window, "result_compare_preview_caption", None)
        compare_combo = getattr(self.window, "result_compare_combo", None)
        session = self._selected_session()
        if compare_label is None or compare_combo is None or session is None:
            return

        compare_index = compare_combo.currentIndex() - 1
        results = getattr(session, "scan_results", []) or []
        current_index = self.window.result_list.currentRow()
        if compare_index < 0 or compare_index >= len(results) or compare_index == current_index:
            self._set_compare_preview_placeholder("未选择对比结果", "选择对比结果后，这里会显示另一条结果的预览。")
            return

        other = results[compare_index]
        preview_path = self._preview_path_from_result(other)
        if not preview_path:
            self._set_compare_preview_placeholder(
                "对比结果没有可预览图像",
                f"对比结果：{getattr(other, 'label', '未命名')}",
            )
            return

        pixmap = QPixmap(preview_path)
        if pixmap.isNull():
            self._set_compare_preview_placeholder("对比预览加载失败", os.path.basename(preview_path))
            return

        compare_label.setPixmap(pixmap.scaled(420, 260, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        if compare_caption is not None:
            compare_caption.setText(f"对比预览：{getattr(other, 'label', '未命名')} / {os.path.basename(preview_path)}")

    def _set_pending_scan_results(self, *results: ScanResultRecord) -> None:
        self.window.pending_scan_results = list(results)
        self.reload_sessions()

    def _session_browser_context(self, sessions: list[SessionRecord]) -> str:
        filter_text = self._session_filter_text()
        if not sessions:
            if filter_text:
                return f"当前筛选词：{filter_text}\n没有匹配的实验会话。"
            return "当前还没有实验会话。完成第一次扫描后，这里会显示历史批次。"

        latest = sessions[0]
        stage = "采集阶段"
        if getattr(latest, "inference_results", []) or []:
            stage = "推理复查阶段"
        elif getattr(latest, "training_results", []) or []:
            stage = "训练完成阶段"
        elif getattr(latest, "scan_results", []) or []:
            stage = "已有扫描结果"
        prefix = f"当前筛选词：{filter_text}\n" if filter_text else ""
        return (
            f"{prefix}匹配会话数：{len(sessions)}\n"
            f"最近活跃批次：{getattr(latest, 'label', '未命名')} ({getattr(latest, 'id', '未记录')})\n"
            f"当前阶段：{stage}"
        )

    def _confirm(self, title: str, message: str) -> bool:
        return QMessageBox.question(self.window, title, message) == QMessageBox.Yes

    def _choose_file(self, title: str, file_filter: str) -> str:
        file_path, _ = QFileDialog.getOpenFileName(self.window, title, "", file_filter)
        return file_path

    def _choose_files(self, title: str, file_filter: str) -> list[str]:
        files, _ = QFileDialog.getOpenFileNames(self.window, title, "", file_filter)
        return files

    def _choose_directory(self, title: str) -> str:
        return QFileDialog.getExistingDirectory(self.window, title)

    def _refresh_buttons_if(self, loaded: bool) -> None:
        if loaded:
            self.window.update_button_states()

    def _new_session(self) -> SessionRecord:
        self.window.current_session = self.window.acquisition_workflow_service.start_scan_session(
            getattr(self.window, "current_session", None),
            self.window.scan_label_edit.text(),
            self.window.nanonis_service,
        )
        return self.window.current_session

    def _update_result_preview(self, result: ScanResultRecord | None) -> None:
        preview_label = getattr(self.window, "result_preview_label", None)
        if preview_label is None:
            return

        preview_path = self._preview_path_from_result(result)
        if not preview_path:
            self._set_result_preview_placeholder("当前结果没有可预览图像", "如结果包含导出图像，这里会自动显示第一张。")
            return

        pixmap = QPixmap(preview_path)
        if pixmap.isNull():
            self._set_result_preview_placeholder("预览加载失败", os.path.basename(preview_path))
            return

        preview_label.setPixmap(pixmap.scaled(420, 260, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        preview_caption = getattr(self.window, "result_preview_caption", None)
        if preview_caption is not None:
            preview_caption.setText(f"当前预览：{os.path.basename(preview_path)}")

    def _open_directory(self, path: str | None, missing_message: str) -> None:
        if not path or not os.path.isdir(path):
            self.window.log(missing_message)
            QMessageBox.warning(self.window, "目录不可用", missing_message)
            return

        startfile = getattr(os, "startfile", None)
        if startfile is not None:
            startfile(path)
        else:
            QMessageBox.information(self.window, "目录路径", path)
        self.window.log(f"已打开目录：{path}")

    def _selected_training_output_dir(self) -> str | None:
        session = self._selected_session()
        if session is None:
            return None
        training_results = getattr(session, "training_results", []) or []
        if not training_results:
            return None
        return getattr(training_results[-1], "output_dir", None)

    def _selected_inference_output_dir(self) -> str | None:
        session = self._selected_session()
        if session is None:
            return None
        inference_results = getattr(session, "inference_results", []) or []
        if not inference_results:
            return None
        record = inference_results[-1]
        return getattr(record, "actual_output_dir", None) or getattr(record, "output_dir", None)

    def _selected_result_directory(self) -> str | None:
        result = self._selected_scan_result()
        preview_path = self._preview_path_from_result(result)
        if preview_path:
            return os.path.dirname(preview_path)
        return None

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
        return self._apply_image_load_result(self.window.image_workflow_service.convert_for_annotation(files))

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

        self.window.log(f"SXM 显示模式：{'彩色' if use_color else '灰度'}")

    def get_gray_path(self, image_path: str) -> str:
        return self.window.image_workflow_service.gray_path_for(image_path, getattr(self.window, "sxm_metadata", {}))

    def _detect_and_display_classes(self) -> None:
        summary = self.window.dataset_service.annotation_class_summary(self.window.annotation_tool)
        if not summary.used_classes:
            return

        for checkbox in self.window.class_checkboxes:
            checkbox.deleteLater()
        self.window.class_checkboxes.clear()
        self.window.class_indices = []

        hint = getattr(self.window, "classes_hint_label", None)
        if hint is not None:
            hint.deleteLater()
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

        self.window.log(f"已识别 {len(self.window.class_checkboxes)} 个标注类别")

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
        files = self._choose_files("选择推理图像", "图像文件 (*.jpg *.jpeg *.png *.bmp *.sxm)")
        if not files:
            return
        converted = self._apply_image_load_result(
            self.window.image_workflow_service.convert_for_inference(files),
            load_inference_images=True,
        )
        self.window.log(f"已加载 {len(converted)} 张推理图像")
        self.window.update_button_states()

    def load_images(self) -> None:
        files = self._choose_files("选择图像", "图像文件 (*.jpg *.jpeg *.png *.bmp *.sxm)")
        if not files:
            return
        converted = self.convert_sxm_files(files)
        self.window.log(f"已加载 {len(converted)} 张图像")
        self.window.update_button_states()

    def load_annotations(self) -> None:
        files = self._choose_files("选择标注", "文本文件 (*.txt)")
        if not files:
            return
        self.window.log(f"已加载 {len(files)} 个标注文件")
        state = self.window.annotation_tool.export_state()
        updated_state = self.window.annotation_service.load_annotation_files(state, files)
        self.window.annotation_tool.load_state(updated_state)
        self.window.update_button_states()
        self._detect_and_display_classes()

    def detect_and_display_classes(self) -> None:
        self._detect_and_display_classes()

    def save_annotations(self) -> None:
        directory = self._choose_directory("选择输出目录")
        if not directory:
            return
        state = self.window.annotation_tool.export_state()
        self.window.annotation_service.save_annotations(state, directory)
        self.window.log(f"标注已保存到：{directory}")
        self.window.update_button_states()

    def crop_resnet_dataset(self) -> None:
        directory = self._choose_directory("选择输出目录")
        if not directory:
            return

        self.window.log("正在准备分类裁剪数据...")
        state = self.window.annotation_tool.export_state()
        if not self.window.annotation_service.has_images(state):
            self.window.log("错误：请先加载训练图像")
            return
        if not self.window.annotation_service.has_annotations(state):
            self.window.log("错误：请先加载或创建标注")
            return

        summary = self.window.dataset_service.annotation_class_summary(self.window.annotation_tool)
        if summary.used_classes:
            class_names = [str(index) for index in range(max(summary.used_classes) + 1)]
            self.window.log(f"使用自适应类别：{class_names}")
        else:
            class_names = state.class_names
            self.window.log(f"未发现标注，使用默认类别：{class_names}")

        selected_class_map = {idx: str(class_names[idx]) for idx in summary.used_classes if idx < len(class_names)}
        crop_summary = self.window.dataset_service.crop_resnet_dataset(
            self.window.annotation_tool,
            self.window._get_gray_path,
            directory,
            selected_class_map,
        )
        self.window.log(f"已标注类别：{list(crop_summary.actual_classes)}")
        self.window.log(f"分类裁剪完成，共生成 {crop_summary.crop_count} 张")

    def load_yolo_model(self) -> None:
        file_path = self._choose_file("选择检测模型", "模型文件 (*.pt *.pth)")
        loaded = self.window.resource_loader_service.load_yolo_model(
            file_path,
            self.window.model_manager,
            self.window.log,
            getattr(self.window, "train_yolo_model_path", None),
            getattr(self.window, "infer_yolo_model_path", None),
        )
        self._refresh_buttons_if(loaded)

    def load_resnet_data(self) -> None:
        directory = self._choose_directory("选择分类数据目录")
        loaded, _ = self.window.resource_loader_service.load_resnet_dataset(
            directory,
            self.window.log,
            getattr(self.window, "train_resnet_data_path", None),
        )
        self._refresh_buttons_if(loaded)

    def load_classes_yaml(self) -> None:
        file_path = self._choose_file("选择类别文件", "YAML 文件 (*.yaml)")
        loaded = self.window.resource_loader_service.load_classes_file(
            file_path,
            self.window.log,
            getattr(self.window, "train_classes_path", None),
            getattr(self.window, "infer_classes_path", None),
        )
        self._refresh_buttons_if(loaded)

    def load_resnet_model(self) -> None:
        file_path = self._choose_file("选择分类模型", "模型文件 (*.saving *.pt *.pth)")
        loaded = self.window.resource_loader_service.load_resnet_model(
            file_path,
            self.window.model_manager,
            self.window.log,
            getattr(self.window, "train_resnet_model_path", None),
            getattr(self.window, "infer_resnet_model_path", None),
        )
        self._refresh_buttons_if(loaded)

    def load_classes_file(self) -> None:
        self.load_classes_yaml()

    def connect_nanonis(self) -> None:
        self.window.log("正在连接设备...")
        self._run_service("connect_nanonis", lambda: self.window.nanonis_service.connect(self._nanonis_config()))

    def disconnect_nanonis(self) -> None:
        self.window.nanonis_service.disconnect()
        self.window.nano_status_text.setPlainText("已断开")
        self.window.log("设备已断开")

    def refresh_nanonis_status(self) -> None:
        self._run_service("refresh_nanonis", self.window.nanonis_service.status)

    def set_nanonis_bias(self) -> None:
        bias_v = float(self.window.scan_bias_edit.text().strip())
        self._run_service("set_bias", lambda: self.window.nanonis_service.set_bias(bias_v))

    def set_nanonis_setpoint(self) -> None:
        setpoint_a = float(self.window.scan_setpoint_edit.text().strip())
        self._run_service("set_setpoint", lambda: self.window.nanonis_service.set_setpoint(setpoint_a))

    def set_nanonis_feedback(self, enabled: bool) -> None:
        if not enabled and not self._confirm("确认", "关闭 Feedback 可能带来风险，是否继续？"):
            return
        self._run_service("set_feedback", lambda: self.window.nanonis_service.set_feedback(enabled))

    def apply_nanonis_scan(self) -> None:
        self._run_service("apply_scan", lambda: self.window.nanonis_service.apply_scan(**self._scan_geometry()))

    def scan_and_save_from_nanonis(self) -> None:
        session = self._new_session()
        self.window.log(f"开始扫描，会话：{session.id}")
        self._run_service("scan_and_save", lambda: self.window.nanonis_service.scan_and_save(label=session.id, **self._scan_geometry()))

    def run_scan_pulse_scan_workflow(self) -> None:
        if not self._confirm("确认", "是否执行预扫、脉冲和后扫的自动流程？"):
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
            self.window.log("错误：请先加载检测模型和分类模型")
            return
        if not self.window.inference_manager.images:
            self.window.log("错误：请先加载推理图像")
            return

        context = self.window.inference_workflow_service.prepare_inference_start(
            self.window.current_session,
            self.window.scan_label_edit.text(),
            self.window.infer_classes_path.text(),
        )
        self.window.current_session = context.session
        self.window.pending_inference_session_id = context.session.id
        self.window.pending_inference_output_dir = context.output_dir

        self.window.log(f"开始执行推理，会话：{context.session.id}")
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
        self.window.log(f"已加载 {len(selection.files)} 张结果图像用于推理")
        self.window.update_button_states()

    def open_selected_session_directory(self) -> None:
        session = self._selected_session()
        path = getattr(session, "path", None) if session is not None else None
        self._open_directory(path, "当前会话目录暂不可用。")

    def open_selected_training_output(self) -> None:
        self._open_directory(self._selected_training_output_dir(), "当前会话还没有训练输出目录。")

    def open_selected_inference_output(self) -> None:
        self._open_directory(self._selected_inference_output_dir(), "当前会话还没有推理输出目录。")

    def open_selected_result_directory(self) -> None:
        self._open_directory(self._selected_result_directory(), "当前结果还没有导出目录。")

    def on_result_comparison_changed(self, index: int) -> None:
        if hasattr(self.window, "result_compare_summary_text"):
            self.window.result_compare_summary_text.setPlainText(self._build_result_comparison_summary())
        self._update_compare_preview()

    def on_inference_finished(self) -> None:
        if self.window.pending_inference_session_id and self.window.pending_inference_output_dir:
            results = self.window.inference_manager.get_inference_results(str(self.window.pending_inference_output_dir))
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
        sessions = self._visible_sessions()
        browser_context = getattr(self.window, "session_browser_context_text", None)
        if browser_context is not None:
            browser_context.setPlainText(self._session_browser_context(sessions))
        for label in self.window.session_workflow_service.session_list_labels(sessions):
            self.window.session_list.addItem(label)
        if sessions:
            self.window.session_list.setCurrentRow(0)

    def on_session_selected(self, index: int) -> None:
        sessions = self._visible_sessions()
        self._refresh_result_list(sessions[index] if 0 <= index < len(sessions) else None)

    def on_result_selected(self, index: int) -> None:
        session = self._selected_session()
        result = self._selected_scan_result()
        detail = self.window.session_workflow_service.result_detail(session, index)
        if not detail or result is None:
            self.window.result_detail_text.clear()
            self._update_result_preview(None)
            if hasattr(self.window, "result_compare_summary_text"):
                self.window.result_compare_summary_text.setPlainText(self._build_result_comparison_summary())
            self._update_compare_preview()
            return
        self.window.result_detail_text.setPlainText(detail)
        self._update_result_preview(result)
        if hasattr(self.window, "result_compare_summary_text"):
            self.window.result_compare_summary_text.setPlainText(self._build_result_comparison_summary())
        self._update_compare_preview()

    def handle_service_result(self, key: str, result: object) -> None:
        if key in {"connect_nanonis", "refresh_nanonis", "set_bias", "set_setpoint", "set_feedback", "apply_scan"}:
            self.window.nano_status_text.setPlainText(str(result))
            self.window.log(f"{key} 已完成")
            return

        if key == "scan_and_save":
            record = self.window.acquisition_workflow_service.append_scan_result(self.window.current_session.id, result)
            self._set_pending_scan_results(record)
            self.window.log(f"扫描完成：{record.label}")
            return

        if key == "scan_pulse_scan":
            pre_scan, post_scan = self.window.acquisition_workflow_service.append_scan_workflow_results(
                self.window.current_session.id,
                result,
            )
            self._set_pending_scan_results(pre_scan, post_scan)
            self.window.log("流程完成：预扫 -> 脉冲 -> 后扫")

    def handle_service_error(self, key: str, message: str) -> None:
        self.window.log(f"{key} 失败：{message}")
        QMessageBox.warning(self.window, "操作失败", f"{key} 失败：\n{message}")
