from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QMainWindow

from app.runtime import AppRuntime
from app.windows.runtime_controller import StudioRuntimeController
from app.windows.studio_shell import StudioShellSignalsMixin, initialize_studio_shell
from app.windows.studio_controller import StudioController
from app.windows.studio_panels import StudioPanelsMixin
from app.windows.studio_ui import StudioUiMixin
from app.windows.training_controller import StudioTrainingController


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ReSTOLOStudioApp(StudioUiMixin, StudioPanelsMixin, StudioShellSignalsMixin, QMainWindow):
    service_result_signal = pyqtSignal(str, object)
    service_error_signal = pyqtSignal(str, str)

    def __init__(self, runtime: AppRuntime | None = None):
        QMainWindow.__init__(self)
        initialize_studio_shell(self)
        self.runtime = runtime or AppRuntime.create(PROJECT_ROOT)
        self.setWindowTitle("ReSTOLO Studio")
        self.annotation_service = self.runtime.annotation_service
        self.acquisition_workflow_service = self.runtime.acquisition_workflow_service
        self.config_service = self.runtime.config_service
        self.dataset_service = self.runtime.dataset_service
        self.image_workflow_service = self.runtime.image_workflow_service
        self.inference_workflow_service = self.runtime.inference_workflow_service
        self.result_store = self.runtime.result_store
        self.resource_loader_service = self.runtime.resource_loader_service
        self.session_workflow_service = self.runtime.session_workflow_service
        self.sxm_service = self.runtime.sxm_service
        self.training_job_service = self.runtime.training_job_service
        self.training_runner_service = self.runtime.training_runner_service
        self.training_workflow_service = self.runtime.training_workflow_service
        self.nanonis_service = self.runtime.nanonis_service
        self.workflow_service = self.runtime.workflow_service
        self.inference_service = self.runtime.create_inference_service(self.inference_manager)
        self.current_session = self.runtime.create_startup_session("startup")
        self.pending_inference_session_id = None
        self.pending_inference_output_dir = None
        self.pending_training_context = None
        self.pending_scan_results = []
        self.studio = StudioController(self)
        self.training = StudioTrainingController(self)
        self.runtime_controller = StudioRuntimeController(self)

        self.service_result_signal.connect(self.studio.handle_service_result)
        self.service_error_signal.connect(self.studio.handle_service_error)

        self.studio.apply_default_model_paths()
        self._build_studio_tabs()
        self.studio.reload_sessions()

    def load_infer_images(self):
        return self.studio.load_infer_images()

    def load_images(self):
        return self.studio.load_images()

    def load_annotations(self):
        return self.studio.load_annotations()

    def detect_and_display_classes(self):
        return self.studio.detect_and_display_classes()

    def save_annotations(self):
        return self.studio.save_annotations()

    def crop_resnet_dataset(self):
        return self.studio.crop_resnet_dataset()

    def load_yolo_model(self):
        return self.studio.load_yolo_model()

    def load_resnet_data(self):
        return self.studio.load_resnet_data()

    def load_classes_yaml(self):
        return self.studio.load_classes_yaml()

    def load_resnet_model(self):
        return self.studio.load_resnet_model()

    def load_classes_file(self):
        return self.studio.load_classes_file()

    def train_yolo(self):
        return self.training.train_yolo()

    def train_resnet(self):
        return self.training.train_resnet()

    def disable_controls(self):
        return self.runtime_controller.disable_controls()

    def enable_controls(self):
        return self.runtime_controller.enable_controls()

    def on_inference_error(self, error_msg):
        return self.runtime_controller.on_inference_error(error_msg)

    def connect_nanonis(self):
        return self.studio.connect_nanonis()

    def disconnect_nanonis(self):
        return self.studio.disconnect_nanonis()

    def refresh_nanonis_status(self):
        return self.studio.refresh_nanonis_status()

    def set_nanonis_bias(self):
        return self.studio.set_nanonis_bias()

    def set_nanonis_setpoint(self):
        return self.studio.set_nanonis_setpoint()

    def set_nanonis_feedback(self, enabled: bool):
        return self.studio.set_nanonis_feedback(enabled)

    def apply_nanonis_scan(self):
        return self.studio.apply_nanonis_scan()

    def scan_and_save_from_nanonis(self):
        return self.studio.scan_and_save_from_nanonis()

    def run_scan_pulse_scan_workflow(self):
        return self.studio.run_scan_pulse_scan_workflow()

    def start_inference(self):
        return self.studio.start_inference()

    def use_selected_result_for_inference(self):
        return self.studio.use_selected_result_for_inference()

    def on_inference_finished(self):
        self.runtime_controller.on_inference_finished()
        return self.studio.on_inference_finished()

    def on_progress_updated(self, current, total):
        return self.runtime_controller.on_progress_updated(current, total)

    def on_training_finished(self):
        return self.runtime_controller.on_training_finished()

    def on_training_error(self, error_msg):
        return self.runtime_controller.on_training_error(error_msg)

    def on_training_progress_updated(self, current, total):
        return self.runtime_controller.on_training_progress_updated(current, total)

    def on_train_loss_updated(self, epoch, box, obj, cls, total):
        return self.runtime_controller.on_train_loss_updated(epoch, box, obj, cls, total)

    def on_val_metrics_updated(self, epoch, precision, recall, map50, map50_95):
        return self.runtime_controller.on_val_metrics_updated(epoch, precision, recall, map50, map50_95)

    def on_resnet_loss_updated(self, epoch, train_loss, pred_error):
        return self.runtime_controller.on_resnet_loss_updated(epoch, train_loss, pred_error)

    def on_annotation_updated(self):
        return self.runtime_controller.on_annotation_updated()

    def on_tab_changed(self, index):
        return self.runtime_controller.on_tab_changed(index)

    def update_button_states(self):
        return self.runtime_controller.update_button_states()

    def reload_sessions(self):
        return self.studio.reload_sessions()

    def _session_selected(self, index: int):
        return self.studio.on_session_selected(index)

    def _result_selected(self, index: int):
        return self.studio.on_result_selected(index)


ReSTOLOStudioWindow = ReSTOLOStudioApp

__all__ = ["ReSTOLOStudioApp", "ReSTOLOStudioWindow"]
