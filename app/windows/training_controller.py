from __future__ import annotations

import os
import threading
from pathlib import Path

from PyQt5.QtWidgets import QFileDialog, QMessageBox

from app.core import AppPaths
from app.services.training_runner_service import ResnetTrainingCallbacks
from app.ui.loss_curve_dialog import LossCurveDialog


class StudioTrainingController:
    """Owns training workflows for the studio window."""

    def __init__(self, window):
        self.window = window

    def _app_paths(self) -> AppPaths:
        runtime = getattr(self.window, "runtime", None)
        if runtime is not None and hasattr(runtime, "paths"):
            return runtime.paths
        return AppPaths.from_project_root(Path(__file__).resolve().parents[2])

    def _choose_training_project_dir(self):
        project_dir = QFileDialog.getExistingDirectory(self.window, "Select training output directory")
        if not project_dir:
            self.window.log("Training cancelled")
            return None
        return project_dir

    def _annotation_state(self):
        return self.window.annotation_tool.export_state()

    def _has_annotation_training_data(self) -> bool:
        state = self._annotation_state()
        return self.window.training_workflow_service.validate_annotation_training_data(
            state,
            self.window.annotation_service,
            self.window.log,
        )

    def _has_resnet_source(self) -> bool:
        data_path = self._resnet_data_path()
        if data_path:
            return True
        return self._has_annotation_training_data()

    def _prepare_training_ui(self) -> None:
        self.window.progress_bar.show()
        self.window.progress_bar.setValue(0)
        self.window.disable_controls()

    def _start_background_training(self, target) -> None:
        train_thread = threading.Thread(target=target, daemon=True)
        train_thread.start()

    def _training_parameters(self) -> tuple[int, int]:
        return self.window.epochs_spin.value(), self.window.batch_spin.value()

    def _selected_class_indices(self) -> list[int]:
        return self.window.dataset_service.selected_class_indices(
            self.window.annotation_tool,
            getattr(self.window, "class_checkboxes", []),
            getattr(self.window, "class_indices", []),
        )

    def _class_names_for_indices(self, class_indices: list[int]) -> list[str]:
        return self.window.dataset_service.class_names_for_indices(self.window.annotation_tool, class_indices)

    def _ensure_yolo_loss_dialog(self) -> None:
        if not hasattr(self.window, "loss_curve_dialog") or self.window.loss_curve_dialog is None:
            self.window.loss_curve_dialog = LossCurveDialog(self.window)
        else:
            self.window.loss_curve_dialog.epochs.clear()
            self.window.loss_curve_dialog.box_loss.clear()
            self.window.loss_curve_dialog.obj_loss.clear()
            self.window.loss_curve_dialog.cls_loss.clear()
            self.window.loss_curve_dialog.total_loss.clear()
            self.window.loss_curve_dialog.precision.clear()
            self.window.loss_curve_dialog.recall.clear()
            self.window.loss_curve_dialog.map50.clear()
            self.window.loss_curve_dialog.map50_95.clear()
            self.window.loss_curve_dialog.val_epochs.clear()
            self.window.loss_curve_dialog.ax_train.clear()
            self.window.loss_curve_dialog.ax_val.clear()
            self.window.loss_curve_dialog._setup_yolo_axes()
            self.window.loss_curve_dialog.canvas.draw_idle()
        self.window.loss_curve_dialog.show()
        self.window.loss_curve_dialog.raise_()

    def _ensure_resnet_loss_dialog(self) -> None:
        self.window.loss_curve_dialog = LossCurveDialog(self.window, mode="resnet")
        self.window.loss_curve_dialog.show()
        self.window.loss_curve_dialog.raise_()

    def _set_training_project_dirs(self, project_dir: str) -> None:
        self.window._yolo_project_dir = project_dir
        self.window._resnet_project_dir = project_dir

    def _prepare_training_context(self, mode: str, project_dir: str) -> None:
        if getattr(self.window, "session_workflow_service", None) is None:
            return
        session, context = self.window.training_workflow_service.ensure_training_context(
            getattr(self.window, "current_session", None),
            self.window.session_workflow_service,
            mode,
            project_dir,
        )
        self.window.current_session = session
        self.window.pending_training_context = context

    def _run_yolo_training_job(
        self,
        yolo_model_path: str,
        data_yaml_path: str,
        epochs: int,
        batch_size: int,
        img_size: int,
        device: str,
        project_dir: str,
    ) -> None:
        try:
            self.window.training_runner_service.run_yolo(
                training_manager=self.window.training_manager,
                app_paths=self._app_paths(),
                yolo_model_path=yolo_model_path,
                data_yaml_path=data_yaml_path,
                epochs=epochs,
                batch_size=batch_size,
                img_size=img_size,
                device=device,
                project_dir=project_dir,
            )
        except Exception as exc:  # noqa: BLE001
            error_msg = f"Training failed: {exc}"
            self.window.log(error_msg)
            QMessageBox.warning(self.window, "Training Failed", error_msg)

    def train_yolo(self) -> None:
        self.window.log("Starting YOLO training...")
        if not self._has_annotation_training_data():
            return

        epochs, batch_size = self._training_parameters()
        device = "0"
        state = self._annotation_state()

        project_dir = self._choose_training_project_dir()
        if not project_dir:
            return

        self._set_training_project_dirs(project_dir)
        self._prepare_training_context("yolo", project_dir)
        plan = self.window.training_workflow_service.prepare_yolo_training(
            state,
            self.window._get_gray_path,
            self.window.training_manager,
            self.window.log,
        )
        if plan is None:
            return

        img_size = plan.img_size
        self.window.log(f"Auto-detected img_size: {img_size}")
        self.window.log(
            f"Training params: epochs={epochs}, batch_size={batch_size}, img_size={img_size}, device={device}"
        )

        self._prepare_training_ui()
        yolo_model_path = self.window.train_yolo_model_path.text() or str(self._app_paths().default_yolo_model_path)
        self.window.log("YOLO training started, check terminal output for details")
        self._ensure_yolo_loss_dialog()
        self._start_background_training(
            lambda: self._run_yolo_training_job(
                yolo_model_path,
                plan.data_yaml_path,
                epochs,
                batch_size,
                img_size,
                device,
                project_dir,
            )
        )

    def _resnet_data_path(self) -> str | None:
        path_widget = getattr(self.window, "train_resnet_data_path", None)
        if not path_widget:
            return None
        data_path = path_widget.text()
        return data_path if data_path and os.path.exists(data_path) else None

    def _run_resnet_training_job(
        self,
        training_path: str,
        testing_path: str,
        saving_path: str,
        epochs: int,
        batch_size: int,
    ) -> None:
        try:
            self.window.training_runner_service.run_resnet(
                pretrained_model_path=self.window.model_manager.resnet_model,
                training_path=training_path,
                testing_path=testing_path,
                saving_path=saving_path,
                epochs=epochs,
                batch_size=batch_size,
                enable_imbalance=self.window.imbalance_checkbox.isChecked(),
                callbacks=ResnetTrainingCallbacks(
                    log=self.window.log,
                    on_progress=lambda current, total: self.window.training_progress_signal.emit(current, total),
                    on_resnet_loss=lambda epoch, train_loss, pred_error: self.window.resnet_loss_signal.emit(
                        epoch,
                        train_loss,
                        pred_error,
                    ),
                ),
            )
            self.window.log("ResNet training complete")
            self.window.training_finished_signal.emit()
        except Exception as exc:  # noqa: BLE001
            import traceback

            error_msg = f"Training failed: {exc}\n{traceback.format_exc()}"
            self.window.log(error_msg)
            self.window.training_error_signal.emit(str(exc))
            QMessageBox.warning(self.window, "Training Failed", error_msg)
        finally:
            self.window.enable_controls()

    def train_resnet(self) -> None:
        self.window.log("Starting ResNet training...")
        if not self._has_resnet_source():
            return

        project_dir = self._choose_training_project_dir()
        if not project_dir:
            return
        self.window._resnet_project_dir = project_dir
        self._prepare_training_context("resnet", project_dir)
        selected_class_indices = self._selected_class_indices()
        class_names = self._class_names_for_indices(selected_class_indices)
        plan = self.window.training_workflow_service.prepare_resnet_training(
            self.window.annotation_tool,
            self.window._get_gray_path,
            project_dir,
            self._resnet_data_path(),
            selected_class_indices,
            class_names,
            self.window.log,
        )
        if plan is None:
            return

        self._prepare_training_ui()
        epochs, batch_size = self._training_parameters()
        self._ensure_resnet_loss_dialog()
        self._start_background_training(
            lambda: self._run_resnet_training_job(
                plan.training_path,
                plan.testing_path,
                plan.saving_path,
                epochs,
                batch_size,
            )
        )
        self.window.log("ResNet training started, check terminal output for details")
