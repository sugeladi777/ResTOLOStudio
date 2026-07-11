from __future__ import annotations

import os
import threading
from pathlib import Path

from PyQt5.QtWidgets import QFileDialog

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
        session_service = getattr(self.window, "session_workflow_service", None)
        current_session = getattr(self.window, "current_session", None)
        if session_service is not None:
            session = session_service.ensure_session(current_session, "training")
            self.window.current_session = session
            project_dir = session_service.training_dir(session.id)
            project_dir.mkdir(parents=True, exist_ok=True)
            self.window.log(f"使用当前会话训练目录：{project_dir}")
            return str(project_dir)

        project_dir = QFileDialog.getExistingDirectory(self.window, "选择训练输出目录")
        if not project_dir:
            self.window.log("已取消训练")
            return None
        return project_dir

    def _annotation_state(self):
        return self.window.annotation_tool.export_state()

    def _saved_annotation_state(self):
        session_service = getattr(self.window, "session_workflow_service", None)
        current_session = getattr(self.window, "current_session", None)
        annotation_service = getattr(self.window, "annotation_service", None)
        if session_service is None or current_session is None or annotation_service is None:
            return None
        annotation_dir = session_service.annotation_dir(current_session.id)
        if not annotation_dir.exists():
            return None
        return annotation_service.load_saved_annotation_state(str(annotation_dir))

    def _annotation_training_source(self):
        saved_state = self._saved_annotation_state()
        if saved_state is not None and self.window.annotation_service.has_images(saved_state) and self.window.annotation_service.has_annotations(saved_state):
            self.window.log("分类训练将优先使用当前会话中已保存的标注数据")
            return saved_state, (lambda path: path), True

        state = self._annotation_state()
        if self.window.annotation_service.has_images(state) and self.window.annotation_service.has_annotations(state):
            return state, self.window._get_gray_path, False

        return state, self.window._get_gray_path, False

    def _has_annotation_training_data(self) -> bool:
        state, _, _ = self._annotation_training_source()
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
        threading.Thread(target=target, daemon=True).start()

    def _yolo_training_parameters(self) -> tuple[int, int]:
        return self.window.yolo_epochs_spin.value(), self.window.yolo_batch_spin.value()

    def _resnet_training_parameters(self) -> tuple[int, int]:
        return self.window.resnet_epochs_spin.value(), self.window.resnet_batch_spin.value()

    def _selected_class_indices(self) -> list[int]:
        return self.window.dataset_service.selected_class_indices(
            self.window.annotation_tool,
            getattr(self.window, "class_checkboxes", []),
            getattr(self.window, "class_indices", []),
        )

    def _class_names_for_indices(self, annotation_source, class_indices: list[int]) -> list[str]:
        return self.window.dataset_service.class_names_for_indices(annotation_source, class_indices)

    def _ensure_yolo_loss_dialog(self) -> None:
        if not hasattr(self.window, "loss_curve_dialog") or self.window.loss_curve_dialog is None:
            self.window.loss_curve_dialog = LossCurveDialog(self.window)
        else:
            self.window.loss_curve_dialog.epochs.clear()
            self.window.loss_curve_dialog.box_loss.clear()
            self.window.loss_curve_dialog.obj_loss.clear()
            self.window.loss_curve_dialog.cls_loss.clear()
            self.window.loss_curve_dialog.total_loss.clear()
            self.window.loss_curve_dialog.ax_train.clear()
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
            error_msg = f"训练失败：{exc}"
            self.window.log(error_msg)
            self.window.training_error_signal.emit(str(exc))

    def train_yolo(self) -> None:
        self.window.log("开始训练检测模型...")
        if not self._has_annotation_training_data():
            return

        epochs, batch_size = self._yolo_training_parameters()
        try:
            import torch

            device = "0" if torch.cuda.is_available() else "cpu"
            device_name = torch.cuda.get_device_name(0) if device == "0" else "CPU"
        except Exception:  # noqa: BLE001
            device = "cpu"
            device_name = "CPU"
        self.window.log(f"训练设备：{device_name}")
        state, gray_path_resolver, _ = self._annotation_training_source()

        project_dir = self._choose_training_project_dir()
        if not project_dir:
            return

        self._set_training_project_dirs(project_dir)
        self._prepare_training_context("yolo", project_dir)
        plan = self.window.training_workflow_service.prepare_yolo_training(
            state,
            gray_path_resolver,
            self.window.training_manager,
            self.window.log,
        )
        if plan is None:
            return

        img_size = plan.img_size
        self.window.log(f"自动识别图像尺寸：{img_size}")
        self.window.log(f"训练参数：轮数={epochs}，批大小={batch_size}，图像尺寸={img_size}，设备={device}")

        self._prepare_training_ui()
        yolo_model_path = self.window.train_yolo_model_path.text() or str(self._app_paths().default_yolo_model_path)
        self.window.log("检测模型训练已启动，请查看终端输出。")
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

    def _classes_yaml_path(self) -> str:
        for attr_name in ("train_classes_path", "infer_classes_path"):
            widget = getattr(self.window, attr_name, None)
            if widget is None or not hasattr(widget, "text"):
                continue
            value = widget.text().strip()
            if value:
                return value
        return ""

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
                enable_augment=getattr(self.window, "augment_checkbox", None).isChecked()
                if getattr(self.window, "augment_checkbox", None) is not None
                else True,
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
            self.window.log("分类模型训练完成")
            self.window.training_finished_signal.emit()
        except Exception as exc:  # noqa: BLE001
            import traceback

            error_msg = f"训练失败：{exc}\n{traceback.format_exc()}"
            self.window.log(error_msg)
            self.window.training_error_signal.emit(str(exc))

    def train_resnet(self) -> None:
        self.window.log("开始训练分类模型...")
        if not self._has_resnet_source():
            return

        project_dir = self._choose_training_project_dir()
        if not project_dir:
            return

        self.window._resnet_project_dir = project_dir
        self._prepare_training_context("resnet", project_dir)
        annotation_source, gray_path_resolver, using_saved_state = self._annotation_training_source()
        selected_class_indices = self.window.dataset_service.selected_class_indices(
            annotation_source,
            [] if using_saved_state else getattr(self.window, "class_checkboxes", []),
            [] if using_saved_state else getattr(self.window, "class_indices", []),
        )
        class_names = self._class_names_for_indices(annotation_source, selected_class_indices) if selected_class_indices else []
        plan = self.window.training_workflow_service.prepare_resnet_training(
            annotation_source,
            gray_path_resolver,
            project_dir,
            self._resnet_data_path(),
            selected_class_indices,
            class_names,
            self._classes_yaml_path(),
            self.window.log,
        )
        if plan is None:
            return

        self._prepare_training_ui()
        epochs, batch_size = self._resnet_training_parameters()
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
        self.window.log("分类模型训练已启动，请查看终端输出。")
