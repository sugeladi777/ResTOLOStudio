from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QFileDialog

from app.legacy import ReSTOLOApp
from app.runtime import AppRuntime
from app.windows.studio_actions import StudioActionsMixin
from app.windows.studio_panels import StudioPanelsMixin


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ReSTOLOStudioApp(StudioActionsMixin, StudioPanelsMixin, ReSTOLOApp):
    service_result_signal = pyqtSignal(str, object)
    service_error_signal = pyqtSignal(str, str)

    def __init__(self, runtime: AppRuntime | None = None):
        super().__init__()
        self.runtime = runtime or AppRuntime.create(PROJECT_ROOT)
        self.setWindowTitle("ReSTOLO Studio")
        self.config_service = self.runtime.config_service
        self.result_store = self.runtime.result_store
        self.nanonis_service = self.runtime.nanonis_service
        self.workflow_service = self.runtime.workflow_service
        self.inference_service = self.runtime.create_inference_service(self.inference_manager)
        self.current_session = self.runtime.create_startup_session("startup")
        self.pending_inference_session_id = None
        self.pending_inference_output_dir = None
        self.pending_scan_results = []

        self.service_result_signal.connect(self._handle_service_result)
        self.service_error_signal.connect(self._handle_service_error)

        self._apply_default_model_paths()
        self._build_studio_tabs()
        self.reload_sessions()

    def _apply_default_model_paths(self):
        yolo_model = self.runtime.paths.default_yolo_model_path
        resnet_model = self.runtime.paths.default_resnet_model_path
        classes_file = self.runtime.paths.default_classes_path
        if yolo_model.exists():
            self.model_manager.load_yolo_model(str(yolo_model))
            self.train_yolo_model_path.setText(str(yolo_model))
            self.infer_yolo_model_path.setText(str(yolo_model))
        if resnet_model.exists():
            self.model_manager.load_resnet_model(str(resnet_model))
            self.train_resnet_model_path.setText(str(resnet_model))
            self.infer_resnet_model_path.setText(str(resnet_model))
        if classes_file.exists():
            self.infer_classes_path.setText(str(classes_file))

    def load_infer_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择推理图片",
            "",
            "Image files (*.jpg *.jpeg *.png *.bmp *.sxm)",
        )
        if files:
            files = self._convert_sxm_files(files)
            self.log(f"加载了 {len(files)} 张推理图片")
            self.inference_manager.load_images([self._get_gray_path(f) for f in files])
            if hasattr(self, "annotation_tool") and self.annotation_tool:
                self.annotation_tool.load_images(files)
            self.update_button_states()


ReSTOLOStudioWindow = ReSTOLOStudioApp

__all__ = ["ReSTOLOStudioApp", "ReSTOLOStudioWindow"]
