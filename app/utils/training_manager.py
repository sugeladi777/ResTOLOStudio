import re
import subprocess
import threading
from pathlib import Path

import yaml
from PyQt5.QtCore import QObject, pyqtSignal

from app.core import AppPaths

APP_PATHS = AppPaths.from_project_root(Path(__file__).resolve().parents[2])
ML_ROOT = APP_PATHS.ml_root


class TrainingManager(QObject):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)
    train_loss_signal = pyqtSignal(int, float, float, float, float)
    val_metrics_signal = pyqtSignal(int, float, float, float, float)

    def __init__(self, python_path="python"):
        super().__init__()
        self.training_process = None
        self.log_callback = None
        self.error_callback = None
        self.python_path = python_path
        self._current_epoch = 0

    def set_log_callback(self, callback):
        self.log_callback = callback
        self.log_signal.connect(callback)

    def set_finished_callback(self, callback):
        self.finished_signal.connect(callback)

    def set_error_callback(self, callback):
        self.error_callback = callback
        self.error_signal.connect(callback)

    def generate_data_yaml(self, train_path, val_path, nc, names, output_path):
        data_config = {"train": train_path, "val": val_path, "nc": nc, "names": names}
        with open(output_path, "w", encoding="utf-8") as handle:
            yaml.dump(data_config, handle, default_flow_style=False)
        return output_path

    def train_yolo(self, weights, cfg, data_yaml, hyp, epochs, batch_size, img_size, device, project, name):
        command = [
            self.python_path,
            "YOLO_train.py",
            "--weights",
            weights,
            "--cfg",
            cfg,
            "--data",
            data_yaml,
            "--hyp",
            hyp,
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--img-size",
            str(img_size),
            str(img_size),
            "--device",
            device,
            "--project",
            project,
            "--name",
            name,
            "--exist-ok",
            "--no-augment",
            "--patience",
            "30",
            "--workers",
            "0",
        ]
        self._run_training(command)

    def _parse_yolo_train_loss(self, line):
        if "100%|" not in line:
            return
        pattern = r"^\s*(\d+)/(\d+)\s+[\d.]+G\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+\s+\d+:"
        match = re.match(pattern, line)
        if not match:
            return
        epoch = int(match.group(1))
        box = float(match.group(3))
        obj = float(match.group(4))
        cls = float(match.group(5))
        total = float(match.group(6))
        self._current_epoch = epoch
        self.train_loss_signal.emit(epoch, box, obj, cls, total)

    def _parse_yolo_val_metrics(self, line):
        if not line.strip().startswith("all"):
            return
        parts = line.strip().split()
        if len(parts) < 7:
            return
        try:
            precision = float(parts[3])
            recall = float(parts[4])
            map50 = float(parts[5])
            map50_95 = float(parts[6])
        except (ValueError, IndexError):
            return
        self.val_metrics_signal.emit(self._current_epoch, precision, recall, map50, map50_95)

    def _emit_log(self, message):
        if self.log_callback:
            self.log_signal.emit(message)
        else:
            print(message)

    def _emit_error(self, message):
        self._emit_log(message)
        if self.error_callback:
            self.error_signal.emit(message)

    def _run_training(self, command):
        def run():
            self._current_epoch = 0
            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=str(ML_ROOT),
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                self.training_process = process

                current_epoch = 0
                total_epochs = 0

                for raw_line in iter(process.stdout.readline, ""):
                    line = raw_line.strip()
                    if not line:
                        continue
                    self._emit_log(line)

                    if "/" in raw_line and "G" in raw_line:
                        slash_pos = raw_line.find("/")
                        try:
                            part_before = raw_line[:slash_pos].strip().split()[-1]
                            part_after = raw_line[slash_pos + 1 :].strip().split()[0]
                            current_epoch = int(part_before)
                            total_epochs = int(part_after)
                            if current_epoch > 0:
                                self.progress_signal.emit(current_epoch, total_epochs)
                        except Exception:
                            pass

                    self._parse_yolo_train_loss(raw_line)
                    self._parse_yolo_val_metrics(raw_line)

                process.wait()
                if process.returncode != 0:
                    self._emit_error(f"训练失败，退出码: {process.returncode}")
                    return
                self.finished_signal.emit()
            except Exception as exc:
                self._emit_error(f"错误: {exc}")
            finally:
                self.training_process = None

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
