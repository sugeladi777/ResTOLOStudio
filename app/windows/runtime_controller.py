from __future__ import annotations

import os

from PyQt5.QtWidgets import QLineEdit, QMessageBox, QPushButton


class StudioRuntimeController:
    """Keeps the studio UI in sync with runtime state."""

    def __init__(self, window):
        self.window = window

    def disable_controls(self) -> None:
        tab_widget = getattr(self.window, "tab_widget", None)
        if tab_widget is not None:
            tab_widget.setEnabled(False)
        if hasattr(self.window, "findChildren"):
            for widget in self.window.findChildren(QPushButton):
                widget.setEnabled(False)
            for widget in self.window.findChildren(QLineEdit):
                widget.setEnabled(False)

    def enable_controls(self) -> None:
        tab_widget = getattr(self.window, "tab_widget", None)
        if tab_widget is not None:
            tab_widget.setEnabled(True)
        if hasattr(self.window, "findChildren"):
            for widget in self.window.findChildren(QPushButton):
                widget.setEnabled(True)
            for widget in self.window.findChildren(QLineEdit):
                widget.setEnabled(True)
        self.update_button_states()

    def _set_label_text(self, attr_name: str, text: str) -> None:
        target = getattr(self.window, attr_name, None)
        if target is None:
            return
        if hasattr(target, "setText"):
            target.setText(text)

    def _set_label_state(self, attr_name: str, state: str) -> None:
        target = getattr(self.window, attr_name, None)
        if target is None or not hasattr(target, "setProperty"):
            return
        target.setProperty("uiState", state)
        style = target.style()
        if style is not None:
            style.unpolish(target)
            style.polish(target)
        target.update()

    def _annotation_state(self):
        tool = getattr(self.window, "annotation_tool", None)
        if tool is None or not hasattr(tool, "export_state"):
            return None
        return tool.export_state()

    def _has_yolo_model(self) -> bool:
        return bool(getattr(getattr(self.window, "model_manager", None), "yolo_model", None))

    def _has_resnet_model(self) -> bool:
        return bool(getattr(getattr(self.window, "model_manager", None), "resnet_model", None))

    def _inference_images(self) -> list[str]:
        return list(getattr(getattr(self.window, "inference_manager", None), "images", []) or [])

    def _selected_scan_result(self):
        selector = getattr(getattr(self.window, "studio", None), "_selected_scan_result", None)
        if callable(selector):
            return selector()

        result_list = getattr(self.window, "result_list", None)
        session = getattr(self.window, "current_session", None)
        current_row = result_list.currentRow() if result_list is not None and hasattr(result_list, "currentRow") else -1
        results = list(getattr(session, "scan_results", []) or [])
        if 0 <= current_row < len(results):
            return results[current_row]
        return None

    def _selected_session(self):
        selector = getattr(getattr(self.window, "studio", None), "_selected_session", None)
        if callable(selector):
            return selector()
        return getattr(self.window, "current_session", None)

    def _has_selected_result(self) -> bool:
        return self._selected_scan_result() is not None

    def _has_selected_result_directory(self) -> bool:
        result = self._selected_scan_result()
        if result is None:
            return False
        directory = getattr(result, "directory", None) or getattr(result, "output_dir", None)
        return bool(directory and os.path.isdir(directory))

    def _set_button_state(self, attr_name: str, enabled: bool, tooltip: str = "") -> None:
        button = getattr(self.window, attr_name, None)
        if button is None:
            return
        if hasattr(button, "setEnabled"):
            button.setEnabled(enabled)
        if hasattr(button, "setToolTip"):
            button.setToolTip("" if enabled else tooltip)

    def _used_annotation_classes(self, annotations) -> list[str]:
        used_classes: set[str] = set()
        for boxes in annotations.values():
            for box in boxes:
                if hasattr(box, "cls"):
                    used_classes.add(str(box.cls))
                elif isinstance(box, (list, tuple)) and box:
                    used_classes.add(str(box[0]))
        return sorted(used_classes)

    def _model_status_text(self) -> str:
        parts: list[str] = []
        if self._has_yolo_model():
            parts.append("检测模型已就绪")
        if self._has_resnet_model():
            parts.append("分类模型已就绪")
        try:
            import torch

            device_text = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU（CUDA 当前不可用）"
            parts.append(f"训练设备：{device_text}")
        except Exception:  # noqa: BLE001
            pass
        return "，".join(parts) if parts else "模型尚未加载；如已配置默认权重，启动后会自动填入路径。"

    def _training_dataset_status_text(self, image_count: int, annotation_count: int, annotations=None) -> str:
        train_data_path = getattr(self.window, "train_resnet_data_path", None)
        has_resnet_data = bool(train_data_path and train_data_path.text().strip())
        if image_count and annotation_count:
            counts: dict[int, int] = {}
            for boxes in (annotations or {}).values():
                for box in boxes:
                    try:
                        class_index = int(box.cls if hasattr(box, "cls") else box[0])
                    except (TypeError, ValueError, IndexError):
                        continue
                    counts[class_index] = counts.get(class_index, 0) + 1
            warning = ""
            scarce = [str(index) for index, count in counts.items() if count < 2]
            if scarce:
                warning = f"；类别 {', '.join(scarce)} 仅有单样本，不能独立验证"
            return f"训练数据已就绪：{image_count} 张图像，{annotation_count} 个标注，类别统计 {dict(sorted(counts.items()))}{warning}。"
        if has_resnet_data:
            return "已加载外部分类型数据。"
        if image_count:
            return "已加载图像；请继续加载或绘制标注。"
        return "训练数据尚未准备好；请先加载图像，再加载或绘制标注。"

    def _training_run_status_text(self) -> str:
        return "训练任务进行中。" if getattr(self.window, "pending_training_context", None) is not None else "当前没有训练任务。"

    def _inference_result_status_text(self) -> str:
        pending = getattr(self.window, "pending_inference_session_id", None)
        return f"推理任务进行中：{pending}" if pending else "当前没有推理结果。"

    def _refresh_status_labels(self) -> None:
        state = self._annotation_state()
        images = list(getattr(state, "images", []) or []) if state is not None else []
        annotations = getattr(state, "annotations", {}) or {} if state is not None else {}
        annotation_count = sum(len(items) for items in annotations.values())

        if images:
            self._set_label_text("annotation_status_detail", f"已加载 {len(images)} 张图像，共 {annotation_count} 个标注。")
            self._set_label_state("annotation_status_detail", "ready")
        else:
            self._set_label_text("annotation_status_detail", "尚未加载图像。")
            self._set_label_state("annotation_status_detail", "normal")

        used_classes = self._used_annotation_classes(annotations)
        if used_classes:
            self._set_label_text("annotation_classes_detail", f"已使用类别：{', '.join(used_classes)}")
            self._set_label_state("annotation_classes_detail", "ready")
        else:
            self._set_label_text("annotation_classes_detail", "尚未识别到已使用类别。")
            self._set_label_state("annotation_classes_detail", "normal")

        infer_images = self._inference_images()
        if infer_images:
            self._set_label_text("inference_input_status_detail", f"推理队列中有 {len(infer_images)} 张图像。")
            self._set_label_text("inference_batch_status_detail", f"当前批次：{len(infer_images)} 张图像。")
            self._set_label_state("inference_input_status_detail", "ready")
            self._set_label_state("inference_batch_status_detail", "ready")
        else:
            self._set_label_text("inference_input_status_detail", "尚未加载推理图像。")
            self._set_label_text("inference_batch_status_detail", "当前没有推理批次。")
            self._set_label_state("inference_input_status_detail", "normal")
            self._set_label_state("inference_batch_status_detail", "normal")

        model_text = self._model_status_text()
        self._set_label_text("training_model_status_detail", model_text)
        self._set_label_text("inference_model_status_detail", model_text)
        self._set_label_text(
            "training_dataset_status_detail",
            self._training_dataset_status_text(len(images), annotation_count, annotations),
        )
        self._set_label_text("training_run_status_detail", self._training_run_status_text())
        self._set_label_text("inference_result_status_detail", self._inference_result_status_text())
        model_state = "ready" if self._has_yolo_model() and self._has_resnet_model() else "warning"
        dataset_state = "ready" if images and annotation_count else "warning" if images else "normal"
        self._set_label_state("training_model_status_detail", model_state)
        self._set_label_state("inference_model_status_detail", model_state)
        self._set_label_state("training_dataset_status_detail", dataset_state)
        self._set_label_state(
            "training_run_status_detail",
            "running" if getattr(self.window, "pending_training_context", None) is not None else "normal",
        )
        self._set_label_state(
            "inference_result_status_detail",
            "running" if getattr(self.window, "pending_inference_session_id", None) else "normal",
        )

        current_index = getattr(getattr(self.window, "tab_widget", None), "currentIndex", lambda: 0)()
        mode_text = {
            0: "采集",
            1: "标注",
            2: "训练",
            3: "应用",
            4: "结果",
        }.get(current_index, "工作台")
        self._set_label_text("workspace_mode_detail", mode_text)

    def on_inference_error(self, error_msg: str) -> None:
        error_display = f"推理失败：{error_msg}"
        matcher = getattr(self.window, "error_matcher", None)
        if matcher is not None:
            match_result = matcher.format_error(error_msg)
            if match_result:
                message, category, suggestion = match_result
                error_display = f"[{category}] {message}"
                if suggestion:
                    error_display += f"\n\n建议：{suggestion}"
        self.window.log(error_display)
        if hasattr(self.window, "set_log_expanded"):
            self.window.set_log_expanded(True)
        QMessageBox.warning(self.window, "推理失败", error_display)
        self._set_label_text("inference_result_status_detail", error_display)
        self._set_label_state("inference_result_status_detail", "error")
        self.enable_controls()

    def on_inference_finished(self) -> None:
        self.window.log("推理完成。")
        output_dir = getattr(self.window, "output_dir", None)
        manager = getattr(self.window, "inference_manager", None)
        actual_output_dir = getattr(manager, "actual_output_dir", None) if manager is not None else None
        if output_dir and actual_output_dir and os.path.exists(actual_output_dir):
            result_images = manager.get_inference_results(output_dir)
            if result_images:
                self.window.log(f"找到 {len(result_images)} 张推理结果图像。")
                self._set_label_text("inference_result_status_detail", f"推理完成，生成 {len(result_images)} 张结果图像。")
                self._set_label_state("inference_result_status_detail", "success")
                annotation_tool = getattr(self.window, "annotation_tool", None)
                if annotation_tool is not None and hasattr(self.window, "annotation_service"):
                    state = self.window.annotation_service.create_state(result_images)
                    annotation_tool.load_state(state)
                    self.window.log("已在标注区显示推理结果。")
            else:
                self.window.log("推理完成，但未找到结果图像。")
                self._set_label_text("inference_result_status_detail", "推理完成，但没有结果图像。")
        self._refresh_status_labels()
        self.enable_controls()

    def on_progress_updated(self, current: int, total: int) -> None:
        if total > 0 and hasattr(self.window, "progress_bar"):
            self.window.progress_bar.setValue(int((current / total) * 100))

    def _persist_training_result(self, status: str) -> None:
        context = getattr(self.window, "pending_training_context", None)
        session = getattr(self.window, "current_session", None)
        if context is None or session is None or not hasattr(self.window, "training_job_service"):
            return
        record = self.window.training_job_service.complete_record(context, status=status)
        self.window.session_workflow_service.append_training_result(session.id, record)
        self._apply_training_artifacts(record)
        self.window.pending_training_context = None

    def _apply_training_artifacts(self, record) -> None:
        loader = getattr(self.window, "resource_loader_service", None)
        model_manager = getattr(self.window, "model_manager", None)
        log = getattr(self.window, "log", None)
        if loader is None or model_manager is None or log is None:
            return

        yolo_targets = (
            getattr(self.window, "train_yolo_model_path", None),
            getattr(self.window, "infer_yolo_model_path", None),
        )
        resnet_targets = (
            getattr(self.window, "train_resnet_model_path", None),
            getattr(self.window, "infer_resnet_model_path", None),
        )
        classes_targets = (
            getattr(self.window, "train_classes_path", None),
            getattr(self.window, "infer_classes_path", None),
        )

        if getattr(record, "yolo_model_path", ""):
            loader.load_yolo_model(record.yolo_model_path, model_manager, log, *yolo_targets)
        if getattr(record, "resnet_model_path", ""):
            loader.load_resnet_model(record.resnet_model_path, model_manager, log, *resnet_targets)
        if getattr(record, "classes_yaml_path", ""):
            loader.load_classes_file(record.classes_yaml_path, log, *classes_targets)

    def on_training_finished(self) -> None:
        self.window.log("训练完成。")
        self._set_label_text("training_run_status_detail", "训练已完成，结果已写入当前会话。")
        self._set_label_state("training_run_status_detail", "success")
        if hasattr(self.window, "progress_bar"):
            self.window.progress_bar.hide()
        self._persist_training_result("completed")
        dialog = getattr(self.window, "loss_curve_dialog", None)
        if dialog is not None and hasattr(dialog, "force_close"):
            dialog.force_close()
        self._refresh_status_labels()
        self.enable_controls()

    def on_training_error(self, error_msg: str) -> None:
        error_display = f"训练失败：{error_msg}"
        matcher = getattr(self.window, "error_matcher", None)
        if matcher is not None:
            match_result = matcher.format_error(error_msg)
            if match_result:
                message, category, suggestion = match_result
                error_display = f"[{category}] {message}"
                if suggestion:
                    error_display += f"\n\n建议：{suggestion}"
        self.window.log(error_display)
        if hasattr(self.window, "set_log_expanded"):
            self.window.set_log_expanded(True)
        self._set_label_text("training_run_status_detail", error_display)
        self._set_label_state("training_run_status_detail", "error")
        if hasattr(self.window, "progress_bar"):
            self.window.progress_bar.hide()
        self._persist_training_result("failed")
        QMessageBox.warning(self.window, "训练失败", error_display)
        self.enable_controls()

    def on_training_progress_updated(self, current: int, total: int) -> None:
        try:
            if total > 0 and hasattr(self.window, "progress_bar"):
                progress = min(100, max(0, int((current / total) * 100)))
                self.window.progress_bar.setValue(progress)
                self.window.log(f"训练进度：第 {current}/{total} 轮")
                self._set_label_text("training_run_status_detail", f"训练进行中：第 {current}/{total} 轮，进度 {progress}%。")
                self._set_label_state("training_run_status_detail", "running")
        except Exception as exc:  # noqa: BLE001
            self.window.log(f"更新训练进度失败：{exc}")

    def on_train_loss_updated(self, epoch: int, box: float, obj: float, cls: float, total: float) -> None:
        dialog = getattr(self.window, "loss_curve_dialog", None)
        if dialog is not None and hasattr(dialog, "update_train_loss"):
            dialog.update_train_loss(epoch, box, obj, cls, total)

    def on_val_metrics_updated(self, epoch: int, precision: float, recall: float, map50: float, map50_95: float) -> None:
        dialog = getattr(self.window, "loss_curve_dialog", None)
        if dialog is not None and hasattr(dialog, "update_val_metrics"):
            dialog.update_val_metrics(epoch, precision, recall, map50, map50_95)

    def on_resnet_loss_updated(self, epoch: int, train_loss: float, pred_error: float) -> None:
        dialog = getattr(self.window, "loss_curve_dialog", None)
        if dialog is not None and hasattr(dialog, "update_resnet_loss"):
            dialog.update_resnet_loss(epoch, train_loss, pred_error)

    def on_annotation_updated(self) -> None:
        self.window.log("标注内容已更新。")
        self._refresh_status_labels()
        self.update_button_states()
        self.window.detect_and_display_classes()

    def on_tab_changed(self, index: int) -> None:
        annotation_tool = getattr(self.window, "annotation_tool", None)
        if annotation_tool is not None:
            state = annotation_tool.export_state()
            is_annotation = index == 1
            if hasattr(annotation_tool, "set_annotation_mode"):
                annotation_tool.set_annotation_mode(is_annotation)
            mode_logs = {
                0: "已切换到采集模式。",
                1: "已切换到标注模式。",
                2: "已切换到训练模式。",
                3: "已切换到应用模式。",
                4: "已切换到结果页面。",
            }
            if index in mode_logs:
                self.window.log(mode_logs[index])
            if index == 3 and getattr(state, "images", None):
                self.window.inference_manager.load_images([self.window._get_gray_path(path) for path in state.images])
        self._refresh_status_labels()
        self.update_button_states()

    def update_button_states(self) -> None:
        try:
            state = self._annotation_state()
            has_images = bool(state and self.window.annotation_service.has_images(state))
            has_annotations = bool(state and self.window.annotation_service.has_annotations(state))
            has_yolo_model = self._has_yolo_model()
            has_resnet_model = self._has_resnet_model()
            has_infer_images = len(self._inference_images()) > 0

            resnet_data_path = getattr(self.window, "train_resnet_data_path", None)
            has_resnet_data = bool(resnet_data_path and resnet_data_path.text().strip() and os.path.exists(resnet_data_path.text().strip()))

            has_selected_result = self._has_selected_result()
            has_selected_result_directory = self._has_selected_result_directory()

            button_states = {
                "load_annotations_btn": (has_images, "请先加载图像，标注文件需要和图像顺序/名称对应。"),
                "save_annotations_btn": (has_annotations, "请先绘制或加载至少一个标注框。"),
                "crop_resnet_dataset_btn": (has_annotations, "请先完成标注，才能导出分类裁剪数据。"),
                "train_load_images_btn": (True, ""),
                "train_load_annotations_btn": (has_images, "请先加载训练图像，再加载对应标注。"),
                "train_load_yolo_model_btn": (True, ""),
                "train_load_resnet_model_btn": (True, ""),
                "train_yolo_btn": (
                    has_images and has_annotations and has_yolo_model,
                    "训练检测模型需要：图像、标注、检测模型权重。",
                ),
                "train_resnet_btn": (
                    has_resnet_model and (has_resnet_data or (has_images and has_annotations)),
                    "训练分类模型需要：分类模型权重，以及外部 ResNet 数据或当前图像标注。",
                ),
                "infer_load_images_btn": (True, ""),
                "infer_load_yolo_model_btn": (True, ""),
                "infer_load_resnet_model_btn": (True, ""),
                "start_inference_btn": (
                    has_infer_images and has_yolo_model and has_resnet_model,
                    "开始推理需要：推理图像、检测模型、分类模型。",
                ),
                "use_selected_result_btn": (has_selected_result, "请先在结果列表中选择一条扫描结果。"),
                "open_result_dir_btn": (has_selected_result_directory, "请先选择带有导出目录的扫描结果。"),
            }
            for attr_name, (enabled, tooltip) in button_states.items():
                self._set_button_state(attr_name, enabled, tooltip)

            self._refresh_status_labels()
        except Exception as exc:  # noqa: BLE001
            if hasattr(self.window, "log"):
                self.window.log(f"刷新按钮状态失败：{exc}")
