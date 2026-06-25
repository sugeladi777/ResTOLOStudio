from __future__ import annotations

import os
import threading

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPainter, QPixmap
from PyQt5.QtWidgets import QApplication, QCheckBox, QFileDialog, QInputDialog, QMessageBox

from app.core import ScanResultRecord, SessionRecord
from app.windows.studio_ui import BASE_COLOR, BORDER_COLOR, PANEL_BG, TEXT_COLOR


class StudioController:
    """Coordinates file loading, acquisition, sessions, and inference actions."""

    def __init__(self, window):
        self.window = window

    def _set_async_busy(self, busy: bool) -> None:
        if busy:
            if hasattr(self.window, "progress_bar"):
                self.window.progress_bar.show()
                self.window.progress_bar.setRange(0, 0)
            if hasattr(self.window, "disable_controls"):
                self.window.disable_controls()
            return

        if hasattr(self.window, "progress_bar"):
            self.window.progress_bar.setRange(0, 100)
            self.window.progress_bar.setValue(0)
            self.window.progress_bar.hide()
        if hasattr(self.window, "enable_controls"):
            self.window.enable_controls()

    def _nanonis_config(self):
        return self.window.acquisition_workflow_service.build_connection_config(
            self.window.nano_ip_edit.text(),
            self.window.nano_port_edit.text(),
            self.window.nano_version_edit.text(),
        )

    def _run_service(self, key: str, fn) -> None:
        self._set_async_busy(True)

        def worker():
            try:
                result = fn()
                self.window.service_result_signal.emit(key, result)
            except Exception as exc:  # noqa: BLE001
                self.window.service_error_signal.emit(key, str(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _model_targets(self, model_type: str):
        return (
            getattr(self.window, f"train_{model_type}_model_path", None),
            getattr(self.window, f"infer_{model_type}_model_path", None),
        )

    def _classes_targets(self):
        return (
            getattr(self.window, "train_classes_path", None),
            getattr(self.window, "infer_classes_path", None),
        )

    def _targets_are_empty(self, *targets) -> bool:
        for target in targets:
            if getattr(target, "text", lambda: "")().strip():
                return False
        return True

    def _load_yolo_model_path(self, file_path: str):
        resource_loader = getattr(self.window, "resource_loader_service", None)
        model_manager = getattr(self.window, "model_manager", None)
        log = getattr(self.window, "log", None)
        if resource_loader is None or model_manager is None or log is None:
            return False
        return resource_loader.load_yolo_model(file_path, model_manager, log, *self._model_targets("yolo"))

    def _load_resnet_model_path(self, file_path: str):
        resource_loader = getattr(self.window, "resource_loader_service", None)
        model_manager = getattr(self.window, "model_manager", None)
        log = getattr(self.window, "log", None)
        if resource_loader is None or model_manager is None or log is None:
            return False
        return resource_loader.load_resnet_model(file_path, model_manager, log, *self._model_targets("resnet"))

    def _load_classes_path(self, file_path: str):
        resource_loader = getattr(self.window, "resource_loader_service", None)
        log = getattr(self.window, "log", None)
        if resource_loader is None or log is None:
            return False
        return resource_loader.load_classes_file(file_path, log, *self._classes_targets())

    def _ensure_inference_models_ready(self) -> None:
        model_manager = getattr(self.window, "model_manager", None)
        if model_manager is None:
            return
        if getattr(model_manager, "yolo_model", None) and getattr(model_manager, "resnet_model", None):
            return

        self._apply_training_record_models(self._latest_training_record(getattr(self.window, "current_session", None)))
        if getattr(model_manager, "yolo_model", None) and getattr(model_manager, "resnet_model", None):
            return

        runtime_paths = getattr(getattr(self.window, "runtime", None), "paths", None)
        resource_loader = getattr(self.window, "resource_loader_service", None)
        log = getattr(self.window, "log", None)
        if runtime_paths is None or resource_loader is None or log is None:
            return

        if not getattr(model_manager, "yolo_model", None):
            self._load_yolo_model_path(str(runtime_paths.default_yolo_model_path))
        if not getattr(model_manager, "resnet_model", None):
            self._load_resnet_model_path(str(runtime_paths.default_resnet_model_path))
        infer_classes_path = getattr(self.window, "infer_classes_path", None)
        if infer_classes_path is not None and not infer_classes_path.text().strip():
            self._load_classes_path(str(runtime_paths.default_classes_path))

    def _scan_geometry(self) -> dict:
        center_x_edit = getattr(self.window, "scan_center_x_edit", None)
        center_y_edit = getattr(self.window, "scan_center_y_edit", None)
        try:
            geometry = self.window.acquisition_workflow_service.build_scan_geometry(
                self.window.scan_width_edit.text(),
                self.window.scan_height_edit.text(),
                center_x_edit.text() if center_x_edit is not None else "0",
                center_y_edit.text() if center_y_edit is not None else "0",
                self.window.scan_pixels_edit.text(),
                self.window.scan_channels_edit.text(),
            )
        except TypeError:
            geometry = self.window.acquisition_workflow_service.build_scan_geometry(
                self.window.scan_width_edit.text(),
                self.window.scan_height_edit.text(),
                self.window.scan_pixels_edit.text(),
                self.window.scan_channels_edit.text(),
            )
        return geometry.to_dict()

    def _pending_scan_context(self, workflow: str, include_pulse: bool = False) -> dict:
        context = {
            "workflow": workflow,
            "session_label": self.window.scan_label_edit.text().strip(),
            "bias_v": float(self.window.scan_bias_edit.text().strip()),
            "setpoint_a": float(self.window.scan_setpoint_edit.text().strip()),
        }
        if include_pulse:
            context["pulse_bias_v"] = float(self.window.pulse_bias_edit.text().strip())
            context["pulse_width_s"] = float(self.window.pulse_width_edit.text().strip())
        return context

    def _scan_electrical_settings(self) -> dict:
        return {
            "bias_v": float(self.window.scan_bias_edit.text().strip()),
            "setpoint_a": float(self.window.scan_setpoint_edit.text().strip()),
        }

    def _configured_scan_request(self, *, label: str) -> dict:
        request = self._scan_geometry()
        request.update(self._scan_electrical_settings())
        request["label"] = label
        return request

    def _take_pending_scan_context(self) -> dict:
        context = dict(getattr(self.window, "pending_scan_context", {}) or {})
        self.window.pending_scan_context = None
        return context

    def _enrich_scan_result_payload(self, payload: dict, context: dict) -> dict:
        enriched = dict(payload)
        raw_context = dict(enriched.get("scan_context", {}) or {})
        raw_context.update(context)
        if raw_context:
            enriched["scan_context"] = raw_context
        return enriched

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
        sessions.sort(
            key=lambda item: (
                str(getattr(item, "created_at", "") or ""),
                str(getattr(item, "id", "") or ""),
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

    def _selected_session_index(self) -> int:
        session_list = getattr(self.window, "session_list", None)
        if session_list is None or not hasattr(session_list, "currentRow"):
            return -1
        return int(session_list.currentRow())

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

        labels = self.window.session_workflow_service.result_labels(session)
        for label in labels:
            self.window.result_list.addItem(label)

        preview_label = getattr(self.window, "result_preview_label", None)
        if preview_label is not None:
            preview_label.clear()
            preview_label.setText("尚未选择结果")

        preview_caption = getattr(self.window, "result_preview_caption", None)
        if preview_caption is not None:
            preview_caption.setText("选择扫描结果后，这里会显示预览。")

        compare_combo = getattr(self.window, "result_compare_combo", None)
        if compare_combo is not None:
            if hasattr(compare_combo, "blockSignals"):
                compare_combo.blockSignals(True)
            if hasattr(compare_combo, "clear"):
                compare_combo.clear()
            if hasattr(compare_combo, "addItem"):
                compare_combo.addItem("不比较")
                for result in list(getattr(session, "scan_results", []) or []):
                    compare_combo.addItem(getattr(result, "label", "result"))
            if hasattr(compare_combo, "setCurrentIndex"):
                compare_combo.setCurrentIndex(0)
            if hasattr(compare_combo, "blockSignals"):
                compare_combo.blockSignals(False)

        if hasattr(self.window, "result_compare_summary_text"):
            self.window.result_compare_summary_text.setPlainText(self._build_result_comparison_summary())
        self._set_compare_preview_placeholder("尚未选择对比项", "选择对比项后，这里会显示对比预览。")

    def _clear_workspace_view(self) -> None:
        annotation_tool = getattr(self.window, "annotation_tool", None)
        annotation_service = getattr(self.window, "annotation_service", None)
        if annotation_tool is not None and annotation_service is not None:
            annotation_tool.load_state(annotation_service.create_state([]))
        update_button_states = getattr(self.window, "update_button_states", None)
        if callable(update_button_states):
            update_button_states()

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
        candidates = []
        for item in getattr(result, "saved", []) or []:
            for key in ("png", "jpg", "jpeg", "bmp", "sxm"):
                path = item.get(key)
                if path and os.path.exists(path):
                    candidates.append((item, path))
        for item, path in candidates:
            npy_path = item.get("npy")
            if npy_path and os.path.exists(npy_path):
                try:
                    import numpy as np

                    data = np.load(npy_path, mmap_mode="r")
                    if getattr(data, "size", 0) > 0:
                        return path
                except Exception:  # noqa: BLE001
                    pass
        for _, path in candidates:
            return path
        raw = getattr(result, "raw", {}) or {}
        for key in ("png", "jpg", "image", "image_path", "nanonis_file_path"):
            path = raw.get(key)
            if path and os.path.exists(path):
                return path
        return None

    def _preview_overlay_lines(self, result: ScanResultRecord | None) -> list[str]:
        session_service = getattr(self.window, "session_workflow_service", None)
        if session_service is None or result is None:
            return []
        return list(session_service.result_overlay_lines(result))

    def _annotated_preview_pixmap(self, preview_path: str, result: ScanResultRecord | None) -> QPixmap:
        pixmap = QPixmap(preview_path)
        if pixmap.isNull():
            return pixmap

        scaled = pixmap.scaled(420, 260, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        lines = self._preview_overlay_lines(result)
        if not lines:
            return scaled

        annotated = QPixmap(scaled)
        painter = QPainter(annotated)
        painter.setRenderHint(QPainter.Antialiasing)

        font = painter.font()
        font.setPointSize(max(font.pointSize(), 9))
        painter.setFont(font)
        metrics = painter.fontMetrics()

        line_height = metrics.lineSpacing()
        text_width = max(metrics.horizontalAdvance(line) for line in lines)
        box_width = min(annotated.width() - 20, text_width + 20)
        box_height = min(annotated.height() - 20, line_height * len(lines) + 16)
        box_y = annotated.height() - box_height - 10

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(16, 16, 16, 190))
        painter.drawRoundedRect(10, box_y, box_width, box_height, 8, 8)

        painter.setPen(QColor(245, 247, 248))
        text_y = box_y + 12 + metrics.ascent()
        for line in lines:
            painter.drawText(20, text_y, line)
            text_y += line_height

        painter.end()
        return annotated

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
        if QApplication.instance() is None:
            self._set_compare_preview_placeholder("预览不可用", "当前上下文中没有活动的 Qt 应用。")
            return

        compare_index = compare_combo.currentIndex() - 1
        results = getattr(session, "scan_results", []) or []
        current_index = self.window.result_list.currentRow()
        if compare_index < 0 or compare_index >= len(results) or compare_index == current_index:
            self._set_compare_preview_placeholder("尚未选择对比结果", "选择对比结果后，这里会显示另一条结果的预览。")
            return

        other = results[compare_index]
        preview_path = self._preview_path_from_result(other)
        if not preview_path:
            self._set_compare_preview_placeholder(
                "对比结果没有可预览图像",
                f"对比结果：{getattr(other, 'label', '未命名')}",
            )
            return

        pixmap = self._annotated_preview_pixmap(preview_path, other)
        if pixmap.isNull():
            self._set_compare_preview_placeholder("对比预览加载失败", os.path.basename(preview_path))
            return

        compare_label.setPixmap(pixmap)
        if compare_caption is not None:
            overlay_lines = self._preview_overlay_lines(other)
            compare_caption.setText(
                f"对比预览：{getattr(other, 'label', '未命名')} / {os.path.basename(preview_path)}"
                + (f" | {overlay_lines[1]}" if len(overlay_lines) > 1 else "")
            )

    def _set_pending_scan_results(self, *results: ScanResultRecord) -> None:
        self.window.pending_scan_results = list(results)
        self.reload_sessions()
        session_list = getattr(self.window, "session_list", None)
        result_list = getattr(self.window, "result_list", None)
        result_detail_text = getattr(self.window, "result_detail_text", None)
        current_session = getattr(self.window, "current_session", None)
        if session_list is not None and result_list is not None and result_detail_text is not None and current_session is not None:
            sessions = self._visible_sessions()
            for index, session in enumerate(sessions):
                if getattr(session, "id", None) == getattr(current_session, "id", None):
                    session_list.setCurrentRow(index)
                    self.on_session_selected(index)
                    break
        if result_list is not None and result_detail_text is not None and results:
            result_list.setCurrentRow(len(getattr(current_session, "scan_results", []) or []) - 1)

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

    def _config_payload(self) -> dict:
        config_service = getattr(self.window, "config_service", None)
        return dict(getattr(config_service, "data", {}) or {}) if config_service is not None else {}

    def _save_config_payload(self, payload: dict) -> None:
        config_service = getattr(self.window, "config_service", None)
        if config_service is not None:
            config_service.data = dict(payload)
            config_service.save(payload)

    def _persist_resource_path(self, key: str, value: str) -> None:
        if not value:
            return
        payload = self._config_payload()
        if payload.get(key) == value:
            return
        payload[key] = value
        self._save_config_payload(payload)

    def _persist_text_value(self, key: str, value: str) -> None:
        payload = self._config_payload()
        normalized = str(value)
        if payload.get(key) == normalized:
            return
        payload[key] = normalized
        self._save_config_payload(payload)

    def _acquisition_config_fields(self) -> list[tuple[str, str]]:
        return [
            ("recent_nanonis_ip", "nano_ip_edit"),
            ("recent_nanonis_port", "nano_port_edit"),
            ("recent_nanonis_version", "nano_version_edit"),
            ("recent_scan_label", "scan_label_edit"),
            ("recent_scan_bias", "scan_bias_edit"),
            ("recent_scan_setpoint", "scan_setpoint_edit"),
            ("recent_scan_width", "scan_width_edit"),
            ("recent_scan_height", "scan_height_edit"),
            ("recent_scan_center_x", "scan_center_x_edit"),
            ("recent_scan_center_y", "scan_center_y_edit"),
            ("recent_scan_pixels", "scan_pixels_edit"),
            ("recent_scan_channels", "scan_channels_edit"),
            ("recent_pulse_bias", "pulse_bias_edit"),
            ("recent_pulse_width", "pulse_width_edit"),
        ]

    def persist_current_acquisition_settings(self) -> None:
        for key, attr_name in self._acquisition_config_fields():
            widget = getattr(self.window, attr_name, None)
            if widget is None or not hasattr(widget, "text"):
                continue
            self._persist_text_value(key, widget.text())

    def apply_saved_acquisition_settings(self) -> None:
        payload = self._config_payload()
        for key, attr_name in self._acquisition_config_fields():
            value = payload.get(key)
            if value is None:
                continue
            widget = getattr(self.window, attr_name, None)
            if widget is None or not hasattr(widget, "setText"):
                continue
            widget.setText(str(value))

    def apply_saved_resource_paths(self) -> None:
        payload = self._config_payload()
        resource_loader = getattr(self.window, "resource_loader_service", None)
        model_manager = getattr(self.window, "model_manager", None)
        log = getattr(self.window, "log", None)
        if resource_loader is None or log is None:
            return

        yolo_path = str(payload.get("recent_yolo_model_path", "") or "")
        if yolo_path and os.path.exists(yolo_path) and model_manager is not None and self._targets_are_empty(*self._model_targets("yolo")):
            self._load_yolo_model_path(yolo_path)

        resnet_path = str(payload.get("recent_resnet_model_path", "") or "")
        if resnet_path and os.path.exists(resnet_path) and model_manager is not None and self._targets_are_empty(*self._model_targets("resnet")):
            self._load_resnet_model_path(resnet_path)

        classes_path = str(payload.get("recent_classes_yaml_path", "") or "")
        if classes_path and os.path.exists(classes_path) and self._targets_are_empty(*self._classes_targets()):
            self._load_classes_path(classes_path)

    def _new_session(self) -> SessionRecord:
        self.window.current_session = self.window.acquisition_workflow_service.start_scan_session(
            getattr(self.window, "current_session", None),
            self.window.scan_label_edit.text(),
            self.window.nanonis_service,
        )
        return self.window.current_session

    def _ensure_current_session(self, label: str = "workspace") -> SessionRecord | None:
        session_service = getattr(self.window, "session_workflow_service", None)
        if session_service is None:
            return getattr(self.window, "current_session", None)
        session = session_service.ensure_session(getattr(self.window, "current_session", None), label)
        self.window.current_session = session
        return session

    def _default_annotation_output_dir(self) -> str | None:
        session = self._ensure_current_session("annotation")
        session_service = getattr(self.window, "session_workflow_service", None)
        if session is None or session_service is None:
            return None
        directory = session_service.annotation_dir(session.id)
        directory.mkdir(parents=True, exist_ok=True)
        return str(directory)

    def _default_resnet_crop_output_dir(self) -> str | None:
        session = self._ensure_current_session("training")
        session_service = getattr(self.window, "session_workflow_service", None)
        if session is None or session_service is None:
            return None
        directory = session_service.training_dir(session.id) / "resnet_crop_export"
        directory.mkdir(parents=True, exist_ok=True)
        return str(directory)

    def _restore_session_annotation_state(self, session: SessionRecord | None) -> bool:
        if session is None:
            return False
        session_service = getattr(self.window, "session_workflow_service", None)
        annotation_service = getattr(self.window, "annotation_service", None)
        annotation_tool = getattr(self.window, "annotation_tool", None)
        if session_service is None or annotation_service is None or annotation_tool is None:
            return False

        try:
            annotation_dir = session_service.annotation_dir(session.id)
        except Exception:  # noqa: BLE001
            return False

        state = annotation_service.load_saved_annotation_state(str(annotation_dir))
        if state is None or not getattr(state, "images", None):
            return False
        annotation_tool.load_state(state)
        self._detect_and_display_classes()
        return True

    def _restore_session_workspace(self, session: SessionRecord | None) -> None:
        if self._restore_session_annotation_state(session):
            return
        if session is None:
            self._clear_workspace_view()
            return
        results = list(getattr(session, "scan_results", []) or [])
        if not results:
            self._clear_workspace_view()
            return
        latest_result = results[-1]
        self._show_scan_result_in_workspace(latest_result)

    def _sync_latest_result_selection(self, session: SessionRecord | None) -> None:
        result_list = getattr(self.window, "result_list", None)
        if result_list is None or session is None:
            return
        results = list(getattr(session, "scan_results", []) or [])
        if not results:
            return
        latest_index = len(results) - 1
        result_list.setCurrentRow(latest_index)
        self.on_result_selected(latest_index)

    def _update_result_preview(self, result: ScanResultRecord | None) -> None:
        preview_label = getattr(self.window, "result_preview_label", None)
        if preview_label is None:
            return
        if QApplication.instance() is None:
            self._set_result_preview_placeholder("预览不可用", "当前上下文中没有活动的 Qt 应用。")
            return

        preview_path = self._preview_path_from_result(result)
        if not preview_path:
            self._set_result_preview_placeholder("当前结果没有可预览图像", "如结果包含导出图像，这里会自动显示第一张。")
            return

        pixmap = self._annotated_preview_pixmap(preview_path, result)
        if pixmap.isNull():
            self._set_result_preview_placeholder("预览加载失败", os.path.basename(preview_path))
            return

        preview_label.setPixmap(pixmap)
        preview_caption = getattr(self.window, "result_preview_caption", None)
        if preview_caption is not None:
            overlay_lines = self._preview_overlay_lines(result)
            preview_caption.setText(
                f"当前预览：{os.path.basename(preview_path)}"
                + (f" | {overlay_lines[1]}" if len(overlay_lines) > 1 else "")
            )

    def _show_scan_result_in_workspace(self, result: ScanResultRecord | None) -> None:
        if result is None:
            return
        annotation_tool = getattr(self.window, "annotation_tool", None)
        annotation_service = getattr(self.window, "annotation_service", None)
        session = getattr(self.window, "current_session", None)
        if annotation_tool is None or annotation_service is None or session is None:
            return
        if not hasattr(annotation_service, "create_state"):
            return
        preview_paths: list[str] = []
        selected_preview_path = self._preview_path_from_result(result)
        for item in getattr(session, "scan_results", []) or []:
            preview_path = self._preview_path_from_result(item)
            if preview_path and preview_path not in preview_paths:
                preview_paths.append(preview_path)
        if not preview_paths:
            return
        state = annotation_service.create_state(preview_paths)
        if selected_preview_path in preview_paths:
            state.current_index = preview_paths.index(selected_preview_path)
        annotation_tool.load_state(state)
        self.window.update_button_states()

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

    def _latest_training_record(self, session: SessionRecord | None):
        if session is None:
            return None
        training_results = getattr(session, "training_results", []) or []
        if not training_results:
            return None
        return training_results[-1]

    def _session_classes_yaml_path(self, session: SessionRecord | None) -> str:
        if session is None:
            return ""
        session_service = getattr(self.window, "session_workflow_service", None)
        if session_service is None:
            return ""

        candidates = []
        try:
            candidates.append(session_service.annotation_dir(session.id) / "classes.yaml")
        except Exception:  # noqa: BLE001
            pass
        try:
            training_dir = session_service.training_dir(session.id)
            candidates.append(training_dir / "classes.yaml")
            candidates.append(training_dir / "resnet_crop_export" / "classes.yaml")
        except Exception:  # noqa: BLE001
            pass

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return ""

    def _apply_session_resource_paths(self, session: SessionRecord | None) -> None:
        self._apply_training_record_models(self._latest_training_record(session))
        classes_yaml_path = self._session_classes_yaml_path(session)
        if classes_yaml_path:
            self._load_classes_path(classes_yaml_path)

    def _apply_training_record_models(self, record) -> None:
        if record is None:
            return

        raw = getattr(record, "raw", {}) or {}
        yolo_model_path = getattr(record, "yolo_model_path", "") or raw.get("yolo_model_path", "") or ""
        resnet_model_path = getattr(record, "resnet_model_path", "") or raw.get("resnet_model_path", "") or ""
        classes_yaml_path = getattr(record, "classes_yaml_path", "") or raw.get("classes_yaml_path", "") or ""

        if yolo_model_path:
            self._load_yolo_model_path(yolo_model_path)
        if resnet_model_path:
            self._load_resnet_model_path(resnet_model_path)
        if classes_yaml_path:
            self._load_classes_path(classes_yaml_path)

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

    def _annotation_candidate_directories(self, source_files: list[str], display_files: list[str]) -> list[str]:
        candidates: list[str] = []
        seen: set[str] = set()
        sxm_original_paths = getattr(self.window, "sxm_original_paths", {}) or {}

        def add_candidate(path: str) -> None:
            if not path:
                return
            directory = os.path.dirname(path)
            if not directory:
                return
            normalized = os.path.normcase(os.path.abspath(directory))
            if normalized in seen:
                return
            seen.add(normalized)
            candidates.append(directory)

        for path in source_files:
            add_candidate(path)
        for path in display_files:
            add_candidate(path)
            add_candidate(sxm_original_paths.get(path, ""))
        return candidates

    def _auto_restore_annotations_for_loaded_images(self, source_files: list[str], display_files: list[str]) -> None:
        annotation_service = getattr(self.window, "annotation_service", None)
        annotation_tool = getattr(self.window, "annotation_tool", None)
        resource_loader = getattr(self.window, "resource_loader_service", None)
        if annotation_service is None or annotation_tool is None or resource_loader is None or not display_files:
            return

        candidate_directories = self._annotation_candidate_directories(source_files, display_files)
        annotation_paths = annotation_service.discover_annotation_files(display_files, candidate_directories)
        if annotation_paths:
            state = annotation_tool.export_state()
            updated_state = annotation_service.load_annotation_files(state, annotation_paths)
            annotation_tool.load_state(updated_state)
            self.window.log(f"已自动加载 {len(annotation_paths)} 个匹配标注文件")

        classes_path = annotation_service.discover_classes_yaml(candidate_directories)
        train_classes_path = getattr(self.window, "train_classes_path", None)
        infer_classes_path = getattr(self.window, "infer_classes_path", None)
        should_restore_classes = bool(classes_path) and not (
            (train_classes_path is not None and train_classes_path.text().strip())
            or (infer_classes_path is not None and infer_classes_path.text().strip())
        )
        if should_restore_classes:
            resource_loader.load_classes_file(
                classes_path,
                self.window.log,
                train_classes_path,
                infer_classes_path,
            )
            self.window.log(f"已自动加载类别文件：{os.path.basename(classes_path)}")

        if annotation_paths:
            self._detect_and_display_classes()

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
            getattr(self.window, "train_classes_path", None),
        )

    def apply_current_session_model_paths(self) -> None:
        session = getattr(self.window, "current_session", None)
        self._apply_session_resource_paths(session)
        self._restore_session_workspace(session)

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
        self._auto_restore_annotations_for_loaded_images(files, converted)
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
        directory = self._default_annotation_output_dir()
        if not directory:
            return
        state = self.window.annotation_tool.export_state()
        output_dir, _used_class_names = self.window.annotation_service.save_annotations(state, directory)
        classes_path = output_dir / "classes.yaml" if output_dir is not None else None
        if classes_path is not None and classes_path.exists():
            self.window.resource_loader_service.load_classes_file(
                str(classes_path),
                self.window.log,
                getattr(self.window, "train_classes_path", None),
                getattr(self.window, "infer_classes_path", None),
            )
        self.window.log(f"标注已保存到：{directory}")
        self.window.update_button_states()

    def crop_resnet_dataset(self) -> None:
        directory = self._default_resnet_crop_output_dir()
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
            class_names = list(summary.class_names)
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
        classes_path = os.path.join(directory, "classes.yaml")
        if selected_class_map:
            try:
                import yaml

                payload = {"names": list(selected_class_map.values()), "nc": len(selected_class_map)}
                with open(classes_path, "w", encoding="utf-8") as handle:
                    yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)
                self.window.resource_loader_service.load_classes_file(
                    classes_path,
                    self.window.log,
                    getattr(self.window, "train_classes_path", None),
                    getattr(self.window, "infer_classes_path", None),
                )
            except Exception as exc:  # noqa: BLE001
                self.window.log(f"写入类别文件失败：{exc}")
        self.window.log(f"已标注类别：{list(crop_summary.actual_classes)}")
        if hasattr(crop_summary, "class_counts"):
            self.window.log(f"分类裁剪类别统计：{crop_summary.class_counts}")
        self.window.log(f"分类裁剪完成，共生成 {crop_summary.crop_count} 张")

    def load_yolo_model(self) -> None:
        file_path = self._choose_file("选择检测模型", "模型文件 (*.pt *.pth)")
        loaded = self._load_yolo_model_path(file_path)
        if loaded:
            self._persist_resource_path("recent_yolo_model_path", file_path)
        self._refresh_buttons_if(loaded)

    def load_resnet_data(self) -> None:
        directory = self._choose_directory("选择分类数据目录")
        loaded, _ = self.window.resource_loader_service.load_resnet_dataset(
            directory,
            self.window.log,
            getattr(self.window, "train_resnet_data_path", None),
        )
        if loaded:
            self._persist_resource_path("recent_resnet_data_path", directory)
        self._refresh_buttons_if(loaded)

    def load_classes_yaml(self) -> None:
        file_path = self._choose_file("选择类别文件", "YAML 文件 (*.yaml)")
        loaded = self._load_classes_path(file_path)
        if loaded:
            self._persist_resource_path("recent_classes_yaml_path", file_path)
        self._refresh_buttons_if(loaded)

    def load_resnet_model(self) -> None:
        file_path = self._choose_file("选择分类模型", "模型文件 (*.saving *.pt *.pth)")
        loaded = self._load_resnet_model_path(file_path)
        if loaded:
            self._persist_resource_path("recent_resnet_model_path", file_path)
        self._refresh_buttons_if(loaded)

    def load_classes_file(self) -> None:
        self.load_classes_yaml()

    def connect_nanonis(self) -> None:
        self.persist_current_acquisition_settings()
        self.window.log("正在连接设备...")
        self._run_service("connect_nanonis", lambda: self.window.nanonis_service.connect(self._nanonis_config()))

    def disconnect_nanonis(self) -> None:
        self.window.log("正在断开设备...")
        self._run_service("disconnect_nanonis", self.window.nanonis_service.disconnect)

    def refresh_nanonis_status(self) -> None:
        self._run_service("refresh_nanonis", self.window.nanonis_service.status)

    def set_nanonis_bias(self) -> None:
        self.persist_current_acquisition_settings()
        bias_v = float(self.window.scan_bias_edit.text().strip())
        self._run_service("set_bias", lambda: self.window.nanonis_service.set_bias(bias_v))

    def set_nanonis_setpoint(self) -> None:
        self.persist_current_acquisition_settings()
        setpoint_a = float(self.window.scan_setpoint_edit.text().strip())
        self._run_service("set_setpoint", lambda: self.window.nanonis_service.set_setpoint(setpoint_a))

    def set_nanonis_feedback(self, enabled: bool) -> None:
        if not enabled and not self._confirm("确认", "关闭 Feedback 可能带来风险，是否继续？"):
            return
        self._run_service("set_feedback", lambda: self.window.nanonis_service.set_feedback(enabled))

    def apply_nanonis_scan(self) -> None:
        self.persist_current_acquisition_settings()
        self._run_service("apply_scan", lambda: self.window.nanonis_service.apply_scan(**self._scan_geometry()))

    def scan_and_save_from_nanonis(self) -> None:
        self.persist_current_acquisition_settings()
        session = self._new_session()
        self.window.pending_scan_context = self._pending_scan_context("single_scan")
        self.window.log(f"开始扫描，会话：{session.id}")
        self._run_service(
            "scan_and_save",
            lambda: self.window.nanonis_service.configure_and_scan_and_save(**self._configured_scan_request(label=session.id)),
        )

    def run_scan_pulse_scan_workflow(self) -> None:
        if not self._confirm("确认", "是否执行预扫、脉冲和后扫的自动流程？"):
            return

        self.persist_current_acquisition_settings()
        session = self._new_session()
        self.window.pending_scan_context = self._pending_scan_context("scan_pulse_scan", include_pulse=True)
        self._run_service(
            "scan_pulse_scan",
            lambda: self.window.workflow_service.scan_pulse_scan(
                label=session.id,
                **self._scan_electrical_settings(),
                **self._scan_geometry(),
                pulse_bias_v=float(self.window.pulse_bias_edit.text().strip()),
                pulse_width_s=float(self.window.pulse_width_edit.text().strip()),
            ),
        )

    def start_inference(self) -> None:
        self._ensure_inference_models_ready()
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
        self._apply_training_record_models(self._latest_training_record(selection.session))
        load_result = self.window.image_workflow_service.convert_for_inference(selection.files)
        self._apply_image_load_result(load_result, load_inference_images=True)
        self.window.tab_widget.setCurrentIndex(3)
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

    def create_session(self) -> None:
        session_service = getattr(self.window, "session_workflow_service", None)
        if session_service is None:
            return
        label, ok = QInputDialog.getText(
            self.window,
            "新建会话",
            "请输入会话名称：",
            text="session",
        )
        if not ok:
            return
        session_name = label.strip() or "session"
        session = session_service.create_session(session_name)
        self.window.current_session = session
        self.window.log(f"已创建新会话：{session.id}")
        self.reload_sessions()
        sessions = self._visible_sessions()
        for index, item in enumerate(sessions):
            if getattr(item, "id", None) == session.id:
                self.on_session_selected(index)
                break

    def activate_selected_session(self) -> None:
        index = self._selected_session_index()
        session = self._selected_session()
        if session is None or index < 0:
            self.window.log("请先在会话列表中选择一个会话。")
            return
        self.on_session_selected(index)
        self.window.log(f"当前会话已切换为：{session.id}")

    def rename_selected_session(self) -> None:
        session_service = getattr(self.window, "session_workflow_service", None)
        session = self._selected_session()
        if session_service is None or session is None:
            self.window.log("请先在会话列表中选择一个会话。")
            return
        label, ok = QInputDialog.getText(
            self.window,
            "重命名会话",
            "请输入新的会话名称：",
            text=getattr(session, "label", "") or getattr(session, "id", ""),
        )
        if not ok:
            return
        new_label = label.strip()
        if not new_label:
            self.window.log("会话名称不能为空。")
            return
        updated = session_service.rename_session(session.id, new_label)
        self.window.current_session = updated
        self.window.log(f"会话已重命名为：{new_label}")
        self.reload_sessions()
        sessions = self._visible_sessions()
        for index, item in enumerate(sessions):
            if getattr(item, "id", None) == updated.id:
                self.on_session_selected(index)
                break

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
        current_session = getattr(self.window, "current_session", None)
        for label in self.window.session_workflow_service.session_list_labels(sessions):
            self.window.session_list.addItem(label)
        session_browser_context_text = getattr(self.window, "session_browser_context_text", None)
        if session_browser_context_text is not None and hasattr(session_browser_context_text, "setPlainText"):
            session_browser_context_text.setPlainText(self._session_browser_context(sessions))
        if sessions:
            selected_index = 0
            if current_session is not None:
                for index, session in enumerate(sessions):
                    if getattr(session, "id", None) == getattr(current_session, "id", None):
                        selected_index = index
                        break
            self.window.session_list.setCurrentRow(selected_index)

    def on_session_selected(self, index: int) -> None:
        sessions = self._visible_sessions()
        session = sessions[index] if 0 <= index < len(sessions) else None
        session_list = getattr(self.window, "session_list", None)
        if session_list is not None and hasattr(session_list, "setCurrentRow") and 0 <= index < len(sessions):
            session_list.setCurrentRow(index)
        self.window.current_session = session
        self._refresh_result_list(session)
        self._sync_latest_result_selection(session)
        self._apply_session_resource_paths(session)
        self._restore_session_workspace(session)

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
        self._show_scan_result_in_workspace(result)
        if hasattr(self.window, "result_compare_summary_text"):
            self.window.result_compare_summary_text.setPlainText(self._build_result_comparison_summary())
        self._update_compare_preview()

    def handle_service_result(self, key: str, result: object) -> None:
        self._set_async_busy(False)

        if key in {"connect_nanonis", "refresh_nanonis", "set_bias", "set_setpoint", "set_feedback", "apply_scan"}:
            self.window.nano_status_text.setPlainText(str(result))
            self.window.log(f"{key} 已完成")
            return

        if key == "disconnect_nanonis":
            self.window.nano_status_text.setPlainText("已断开")
            self.window.log("disconnect_nanonis 已完成")
            return

        if key == "scan_and_save":
            context = self._take_pending_scan_context()
            payload = self._enrich_scan_result_payload(result, context) if isinstance(result, dict) else result
            record = self.window.acquisition_workflow_service.append_scan_result(self.window.current_session.id, payload)
            self._set_pending_scan_results(record)
            self._show_scan_result_in_workspace(record)
            self.window.log(f"扫描完成：{record.label}")
            return

        if key == "scan_pulse_scan":
            context = self._take_pending_scan_context()
            if isinstance(result, dict):
                workflow_payload = dict(result)
                if isinstance(workflow_payload.get("pre_scan"), dict):
                    workflow_payload["pre_scan"] = self._enrich_scan_result_payload(workflow_payload["pre_scan"], context)
                if isinstance(workflow_payload.get("post_scan"), dict):
                    workflow_payload["post_scan"] = self._enrich_scan_result_payload(workflow_payload["post_scan"], context)
                result = workflow_payload
            pre_scan, post_scan = self.window.acquisition_workflow_service.append_scan_workflow_results(
                self.window.current_session.id,
                result,
            )
            self._set_pending_scan_results(pre_scan, post_scan)
            self._show_scan_result_in_workspace(post_scan)
            self.window.log("流程完成：预扫 -> 脉冲 -> 后扫")

    def handle_service_error(self, key: str, message: str) -> None:
        self._set_async_busy(False)
        if key in {"scan_and_save", "scan_pulse_scan"}:
            self.window.pending_scan_context = None
        self.window.log(f"{key} 失败：{message}")
        QMessageBox.warning(self.window, "操作失败", f"{key} 失败：\n{message}")
