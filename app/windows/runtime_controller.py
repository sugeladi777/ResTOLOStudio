from __future__ import annotations

import os

from PyQt5.QtWidgets import QLineEdit, QMessageBox, QPushButton

from app.windows.studio_ui import DARK_BG


class StudioRuntimeController:
    """Owns runtime callbacks and UI state updates for the studio window."""

    def __init__(self, window):
        self.window = window

    def disable_controls(self) -> None:
        self.window.tab_widget.setEnabled(False)
        for widget in self.window.findChildren(QPushButton):
            widget.setEnabled(False)
        for widget in self.window.findChildren(QLineEdit):
            widget.setEnabled(False)

    def enable_controls(self) -> None:
        self.window.tab_widget.setEnabled(True)
        for widget in self.window.findChildren(QPushButton):
            widget.setEnabled(True)
        for widget in self.window.findChildren(QLineEdit):
            widget.setEnabled(True)
        self.update_button_states()

    def on_inference_error(self, error_msg: str) -> None:
        match_result = self.window.error_matcher.format_error(error_msg)
        if match_result:
            message, category, suggestion = match_result
            error_display = f"[{category}] {message}"
            if suggestion:
                error_display += f"\n\nSuggestion: {suggestion}"
        else:
            error_display = "Inference failed, check logs for details"
        self.window.log(error_display)
        QMessageBox.warning(self.window, "Inference Failed", error_display)
        self.enable_controls()

    def on_inference_finished(self) -> None:
        self.window.log("Inference complete")
        if hasattr(self.window, "output_dir"):
            actual_output_dir = getattr(self.window.inference_manager, "actual_output_dir", None)
            if actual_output_dir and os.path.exists(actual_output_dir):
                result_images = self.window.inference_manager.get_inference_results(self.window.output_dir)
                if result_images:
                    self.window.log(f"Found {len(result_images)} inference result images")
                    if hasattr(self.window, "annotation_tool") and self.window.annotation_tool:
                        state = self.window.annotation_service.create_state(result_images)
                        self.window.annotation_tool.load_state(state)
                        self.window.log("Displayed inference results in annotation view")
                else:
                    self.window.log("No inference result images found")
            else:
                self.window.log("Inference output directory was not found")
        self.enable_controls()

    def on_progress_updated(self, current: int, total: int) -> None:
        if total > 0:
            self.window.progress_bar.setValue(int((current / total) * 100))

    def _persist_training_result(self, status: str) -> None:
        if getattr(self.window, "pending_training_context", None) and getattr(self.window, "current_session", None):
            if hasattr(self.window, "training_job_service"):
                record = self.window.training_job_service.complete_record(
                    self.window.pending_training_context,
                    output_dir=self.window.training_job_service.output_dir_for_window(self.window),
                    status=status,
                )
                self.window.session_workflow_service.append_training_result(self.window.current_session.id, record)
            self.window.pending_training_context = None

    def on_training_finished(self) -> None:
        self.window.log("Training complete")
        self.window.progress_bar.hide()
        self.enable_controls()
        self._persist_training_result("completed")
        if hasattr(self.window, "loss_curve_dialog") and self.window.loss_curve_dialog:
            save_dir = None
            if getattr(self.window, "_yolo_project_dir", None):
                save_dir = os.path.join(self.window._yolo_project_dir, "yolo_train")
            elif getattr(self.window, "_resnet_project_dir", None):
                save_dir = os.path.join(self.window._resnet_project_dir, "resnet_train")
            if save_dir and os.path.exists(save_dir):
                try:
                    save_path = os.path.join(save_dir, "loss_curves.png")
                    self.window.loss_curve_dialog.figure.savefig(save_path, dpi=150, facecolor=DARK_BG)
                    self.window.log(f"Saved loss curves to: {save_path}")
                except Exception as exc:  # noqa: BLE001
                    self.window.log(f"Failed to save loss curves: {exc}")
            self.window.loss_curve_dialog.force_close()

    def on_training_error(self, error_msg: str) -> None:
        match_result = self.window.error_matcher.format_error(error_msg)
        if match_result:
            message, category, suggestion = match_result
            error_display = f"[{category}] {message}"
            if suggestion:
                error_display += f"\n\nSuggestion: {suggestion}"
        else:
            error_display = f"Training failed: {error_msg}"
        self.window.log(error_display)
        self.window.progress_bar.hide()
        self._persist_training_result("failed")
        QMessageBox.warning(self.window, "Training Failed", error_display)
        self.enable_controls()

    def on_training_progress_updated(self, current: int, total: int) -> None:
        try:
            if total > 0 and hasattr(self.window, "progress_bar"):
                progress = min(100, max(0, int((current / total) * 100)))
                self.window.progress_bar.setValue(progress)
                self.window.log(f"Training progress: Epoch {current}/{total}")
        except Exception as exc:  # noqa: BLE001
            self.window.log(f"Failed to update training progress: {exc}")

    def on_train_loss_updated(self, epoch: int, box: float, obj: float, cls: float, total: float) -> None:
        if hasattr(self.window, "loss_curve_dialog") and self.window.loss_curve_dialog:
            self.window.loss_curve_dialog.update_train_loss(epoch, box, obj, cls, total)

    def on_val_metrics_updated(
        self,
        epoch: int,
        precision: float,
        recall: float,
        map50: float,
        map50_95: float,
    ) -> None:
        if hasattr(self.window, "loss_curve_dialog") and self.window.loss_curve_dialog:
            self.window.loss_curve_dialog.update_val_metrics(epoch, precision, recall, map50, map50_95)

    def on_resnet_loss_updated(self, epoch: int, train_loss: float, pred_error: float) -> None:
        if hasattr(self.window, "loss_curve_dialog") and self.window.loss_curve_dialog:
            self.window.loss_curve_dialog.update_resnet_loss(epoch, train_loss, pred_error)

    def on_annotation_updated(self) -> None:
        self.window.log("Annotations updated")
        self.update_button_states()
        self.window.detect_and_display_classes()

    def on_tab_changed(self, index: int) -> None:
        if hasattr(self.window, "annotation_tool") and self.window.annotation_tool:
            state = self.window.annotation_tool.export_state()
            if index == 0:
                self.window.annotation_tool.set_annotation_mode(True)
                self.window.log("Switched to annotation mode")
            else:
                self.window.annotation_tool.set_annotation_mode(False)
                if index == 1:
                    self.window.log("Switched to training mode")
                else:
                    self.window.log("Switched to inference mode")
                    if state.images:
                        self.window.inference_manager.load_images(
                            [self.window._get_gray_path(path) for path in state.images]
                        )
        self.update_button_states()

    def update_button_states(self) -> None:
        try:
            has_images = False
            has_annotations = False
            if hasattr(self.window, "annotation_tool") and self.window.annotation_tool:
                state = self.window.annotation_tool.export_state()
                has_images = self.window.annotation_service.has_images(state)
                has_annotations = self.window.annotation_service.has_annotations(state)

            has_yolo_model = hasattr(self.window.model_manager, "yolo_model") and self.window.model_manager.yolo_model is not None
            has_resnet_model = hasattr(self.window.model_manager, "resnet_model") and self.window.model_manager.resnet_model is not None
            has_infer_images = hasattr(self.window.inference_manager, "images") and len(self.window.inference_manager.images) > 0

            if hasattr(self.window, "load_annotations_btn"):
                self.window.load_annotations_btn.setEnabled(has_images)
            if hasattr(self.window, "save_annotations_btn"):
                self.window.save_annotations_btn.setEnabled(has_annotations)
            if hasattr(self.window, "crop_resnet_dataset_btn"):
                self.window.crop_resnet_dataset_btn.setEnabled(has_annotations)

            if hasattr(self.window, "train_load_images_btn"):
                self.window.train_load_images_btn.setEnabled(True)
            if hasattr(self.window, "train_load_annotations_btn"):
                self.window.train_load_annotations_btn.setEnabled(has_images)
            if hasattr(self.window, "train_load_yolo_model_btn"):
                self.window.train_load_yolo_model_btn.setEnabled(True)
            if hasattr(self.window, "train_load_resnet_model_btn"):
                self.window.train_load_resnet_model_btn.setEnabled(True)
            if hasattr(self.window, "train_yolo_btn"):
                self.window.train_yolo_btn.setEnabled(has_images and has_annotations and has_yolo_model)

            has_resnet_data = False
            if hasattr(self.window, "train_resnet_data_path"):
                resnet_data_path = self.window.train_resnet_data_path.text()
                has_resnet_data = bool(resnet_data_path and os.path.exists(resnet_data_path))

            if hasattr(self.window, "train_resnet_btn"):
                self.window.train_resnet_btn.setEnabled(
                    has_resnet_model and (has_resnet_data or (has_images and has_annotations))
                )

            if hasattr(self.window, "infer_load_images_btn"):
                self.window.infer_load_images_btn.setEnabled(True)
            if hasattr(self.window, "infer_load_yolo_model_btn"):
                self.window.infer_load_yolo_model_btn.setEnabled(True)
            if hasattr(self.window, "infer_load_resnet_model_btn"):
                self.window.infer_load_resnet_model_btn.setEnabled(True)
            if hasattr(self.window, "start_inference_btn"):
                self.window.start_inference_btn.setEnabled(has_infer_images and has_yolo_model and has_resnet_model)
        except Exception as exc:  # noqa: BLE001
            if hasattr(self.window, "log"):
                self.window.log(f"Failed to update button states: {exc}")
