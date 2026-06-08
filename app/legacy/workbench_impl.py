from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QMainWindow

from app.legacy.workbench_bindings import apply_workbench_bindings
from app.legacy.workbench_layout import initialize_workbench
from app.legacy.workbench_ui import WorkbenchUiMixin


class ReSTOLOApp(WorkbenchUiMixin, QMainWindow):
    training_finished_signal = pyqtSignal()
    training_error_signal = pyqtSignal(str)
    log_signal = pyqtSignal(str)
    training_progress_signal = pyqtSignal(int, int)
    resnet_loss_signal = pyqtSignal(int, float, float)  # epoch, train_loss, pred_error

    def __init__(self):
        super().__init__()
        initialize_workbench(self)

    def log(self, message):
        """Append a log message from worker threads via Qt signals."""
        self.log_signal.emit(message)

    def _log_slot(self, message):
        """Update the on-screen log inside the UI thread."""
        if hasattr(self, "log_text"):
            self.log_text.append(message)


apply_workbench_bindings(ReSTOLOApp)
