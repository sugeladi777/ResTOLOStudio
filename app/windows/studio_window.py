from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import QTimer, pyqtSignal
from PyQt5.QtWidgets import QMainWindow

from app.runtime import AppRuntime
from app.windows.runtime_controller import StudioRuntimeController
from app.windows.studio_controller import StudioController
from app.windows.studio_panels import StudioPanelsMixin
from app.windows.studio_shell import StudioShellSignalsMixin, initialize_studio_shell
from app.windows.studio_ui import StudioUiMixin
from app.windows.training_controller import StudioTrainingController


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ReSTOLOStudioApp(StudioUiMixin, StudioPanelsMixin, StudioShellSignalsMixin, QMainWindow):
    service_result_signal = pyqtSignal(str, object)
    service_error_signal = pyqtSignal(str, str)

    def __init__(self, runtime: AppRuntime | None = None):
        QMainWindow.__init__(self)
        self._style_refresh_timer = QTimer(self)
        self._style_refresh_timer.setSingleShot(True)
        self._style_refresh_timer.timeout.connect(self._refresh_responsive_styles)
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
        self._restore_window_geometry()
        self._restore_splitter_state()
        splitter = getattr(self, "main_splitter", None)
        if splitter is not None:
            splitter.splitterMoved.connect(self._persist_splitter_state)
        if hasattr(self, "tab_widget"):
            self.tab_widget.setTabText(0, "标注")
            self.tab_widget.setTabText(1, "训练")
            self.tab_widget.setTabText(2, "推理")
            if self.tab_widget.count() > 3:
                self.tab_widget.setTabText(3, "采集")
            if self.tab_widget.count() > 4:
                self.tab_widget.setTabText(4, "结果")
        self.studio.reload_sessions()
        self.update_button_states()
        if hasattr(self, "tab_widget"):
            self.on_tab_changed(self.tab_widget.currentIndex())
        self._refresh_responsive_styles()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "_style_refresh_timer"):
            self._style_refresh_timer.start(80)
        self._persist_window_geometry()

    def _refresh_responsive_styles(self):
        self.apply_styles()
        annotation_tool = getattr(self, "annotation_tool", None)
        if annotation_tool is not None and hasattr(annotation_tool, "apply_responsive_styles"):
            annotation_tool.apply_responsive_styles()

    def _restore_splitter_state(self):
        splitter = getattr(self, "main_splitter", None)
        left_widget = getattr(self, "left_panel_widget", None)
        config_service = getattr(self, "config_service", None)
        if splitter is None or left_widget is None or config_service is None:
            return
        ui_config = dict(getattr(config_service, "data", {}) or {})
        left_width = ui_config.get("ui_left_panel_width")
        if not isinstance(left_width, int):
            return
        left_width = max(left_widget.minimumWidth(), min(left_width, left_widget.maximumWidth()))
        total_width = max(self.width(), left_width + 400)
        splitter.setSizes([left_width, max(400, total_width - left_width)])

    def _persist_splitter_state(self, pos: int, index: int):
        del index
        left_widget = getattr(self, "left_panel_widget", None)
        config_service = getattr(self, "config_service", None)
        if left_widget is None or config_service is None:
            return
        left_width = max(left_widget.minimumWidth(), min(int(pos), left_widget.maximumWidth()))
        payload = dict(getattr(config_service, "data", {}) or {})
        if payload.get("ui_left_panel_width") == left_width:
            return
        payload["ui_left_panel_width"] = left_width
        config_service.save(payload)

    def _restore_window_geometry(self):
        config_service = getattr(self, "config_service", None)
        if config_service is None:
            return
        payload = dict(getattr(config_service, "data", {}) or {})
        width = payload.get("ui_window_width")
        height = payload.get("ui_window_height")
        if not isinstance(width, int) or not isinstance(height, int):
            return
        self.resize(max(1000, width), max(700, height))

    def _persist_window_geometry(self):
        config_service = getattr(self, "config_service", None)
        if config_service is None:
            return
        payload = dict(getattr(config_service, "data", {}) or {})
        width = int(self.width())
        height = int(self.height())
        if payload.get("ui_window_width") == width and payload.get("ui_window_height") == height:
            return
        payload["ui_window_width"] = width
        payload["ui_window_height"] = height
        config_service.save(payload)

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
        controller = getattr(self, "runtime_controller", None)
        if controller is None:
            return None
        return controller.disable_controls()

    def enable_controls(self):
        controller = getattr(self, "runtime_controller", None)
        if controller is None:
            return None
        return controller.enable_controls()

    def on_inference_error(self, error_msg):
        controller = getattr(self, "runtime_controller", None)
        if controller is None:
            return None
        return controller.on_inference_error(error_msg)

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

    def open_selected_session_directory(self):
        return self.studio.open_selected_session_directory()

    def open_selected_training_output(self):
        return self.studio.open_selected_training_output()

    def open_selected_inference_output(self):
        return self.studio.open_selected_inference_output()

    def open_selected_result_directory(self):
        return self.studio.open_selected_result_directory()

    def _on_sxm_color_toggle(self, state):
        return self.studio.set_sxm_color_mode(state)

    def _get_gray_path(self, image_path: str) -> str:
        return self.studio.get_gray_path(image_path)

    def on_inference_finished(self):
        controller = getattr(self, "runtime_controller", None)
        if controller is not None:
            controller.on_inference_finished()
        return self.studio.on_inference_finished()

    def on_progress_updated(self, current, total):
        controller = getattr(self, "runtime_controller", None)
        if controller is None:
            return None
        return controller.on_progress_updated(current, total)

    def on_training_finished(self):
        controller = getattr(self, "runtime_controller", None)
        if controller is None:
            return None
        return controller.on_training_finished()

    def on_training_error(self, error_msg):
        controller = getattr(self, "runtime_controller", None)
        if controller is None:
            return None
        return controller.on_training_error(error_msg)

    def on_training_progress_updated(self, current, total):
        controller = getattr(self, "runtime_controller", None)
        if controller is None:
            return None
        return controller.on_training_progress_updated(current, total)

    def on_train_loss_updated(self, epoch, box, obj, cls, total):
        controller = getattr(self, "runtime_controller", None)
        if controller is None:
            return None
        return controller.on_train_loss_updated(epoch, box, obj, cls, total)

    def on_val_metrics_updated(self, epoch, precision, recall, map50, map50_95):
        controller = getattr(self, "runtime_controller", None)
        if controller is None:
            return None
        return controller.on_val_metrics_updated(epoch, precision, recall, map50, map50_95)

    def on_resnet_loss_updated(self, epoch, train_loss, pred_error):
        controller = getattr(self, "runtime_controller", None)
        if controller is None:
            return None
        return controller.on_resnet_loss_updated(epoch, train_loss, pred_error)

    def on_annotation_updated(self):
        controller = getattr(self, "runtime_controller", None)
        if controller is None:
            return None
        return controller.on_annotation_updated()

    def on_tab_changed(self, index):
        controller = getattr(self, "runtime_controller", None)
        if controller is None:
            return None
        return controller.on_tab_changed(index)

    def update_button_states(self):
        controller = getattr(self, "runtime_controller", None)
        if controller is None:
            return None
        return controller.update_button_states()

    def reload_sessions(self):
        return self.studio.reload_sessions()

    def _session_selected(self, index: int):
        return self.studio.on_session_selected(index)

    def _result_selected(self, index: int):
        return self.studio.on_result_selected(index)

    def _result_comparison_changed(self, index: int):
        return self.studio.on_result_comparison_changed(index)


ReSTOLOStudioWindow = ReSTOLOStudioApp

__all__ = ["ReSTOLOStudioApp", "ReSTOLOStudioWindow"]
