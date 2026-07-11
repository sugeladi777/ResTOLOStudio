from __future__ import annotations

import importlib
import re
import sys
from dataclasses import dataclass

from app.core import AppPaths


@dataclass
class ResnetTrainingCallbacks:
    log: callable
    on_progress: callable
    on_resnet_loss: callable


class TrainingRunnerService:
    """Runs YOLO and ResNet training jobs outside the window/controller layer."""

    def run_yolo(
        self,
        training_manager,
        app_paths: AppPaths,
        yolo_model_path: str,
        data_yaml_path: str,
        epochs: int,
        batch_size: int,
        img_size: int,
        device: str,
        project_dir: str,
    ) -> None:
        training_manager.train_yolo(
            weights=yolo_model_path,
            cfg=str(app_paths.yolo_config_path),
            data_yaml=data_yaml_path,
            hyp=str(app_paths.yolo_hyperparameters_path),
            epochs=epochs,
            batch_size=batch_size,
            img_size=img_size,
            device=device,
            project=project_dir,
            name="yolo_train",
        )

    def run_resnet(
        self,
        pretrained_model_path: str,
        training_path: str,
        testing_path: str,
        saving_path: str,
        epochs: int,
        batch_size: int,
        enable_imbalance: bool,
        enable_augment: bool,
        callbacks: ResnetTrainingCallbacks,
    ) -> None:
        callbacks.log("开始训练分类模型")
        resnet_module = importlib.import_module("ml.resnet_finetune")
        parser = _ResnetLogParser(callbacks, epochs)
        custom_stream = _CustomStream(parser.handle_line)
        original_stdout = sys.stdout
        original_argv = sys.argv.copy()
        sys.stdout = custom_stream
        try:
            callbacks.log("类别不平衡处理：开启（加权交叉熵，不复制样本）")
            callbacks.log("Data augmentation: disabled")
            sys.argv = [
                "ResNet_train.py",
                "--training_path",
                training_path,
                "--testing_path",
                testing_path,
                "--saving_path",
                saving_path,
                "--epochs",
                str(epochs),
                "--batch_size",
                str(batch_size),
                "--pretrained_model",
                pretrained_model_path,
                "--patience",
                "30",
            ]
            resnet_module.main()
        finally:
            sys.argv = original_argv
            sys.stdout = original_stdout
            custom_stream.flush_remaining(callbacks.log)
        parser.emit_last()


class _CustomStream:
    def __init__(self, callback):
        self.callback = callback
        self.buffer: list[str] = []
        self.original_stdout = sys.stdout

    def write(self, text):
        if text is not None:
            self.buffer.append(text)
            if "\n" in text:
                lines = "".join(self.buffer).split("\n")
                for line in lines[:-1]:
                    if line.strip():
                        self.callback(line.strip())
                self.buffer = [lines[-1]]
        if self.original_stdout:
            self.original_stdout.write(text)

    def flush(self):
        if self.original_stdout:
            self.original_stdout.flush()

    def flush_remaining(self, log) -> None:
        if self.buffer:
            remaining_text = "".join(self.buffer).strip()
            if remaining_text:
                log(remaining_text)


class _ResnetLogParser:
    def __init__(self, callbacks: ResnetTrainingCallbacks, total_epochs: int):
        self.callbacks = callbacks
        self.total_epochs = total_epochs
        self.last_epoch = 0
        self.last_loss = None
        self.last_error = None

    def handle_line(self, line: str) -> None:
        self.callbacks.log(line)
        if "Epoch" not in line or "Training Loss" not in line:
            return
        try:
            loss_match = re.search(r"Training Loss:\s*([-\d.]+)", line)
            error_match = re.search(r"Prediction Error:\s*([-\d.]+)", line)
            epoch_match = re.search(r"Epoch\s+(\d+)", line)
            if not (loss_match and error_match and epoch_match):
                return
            train_loss = float(loss_match.group(1))
            pred_error = float(error_match.group(1))
            current_epoch = int(epoch_match.group(1))
            if current_epoch != self.last_epoch and self.last_loss is not None:
                self.callbacks.on_resnet_loss(self.last_epoch, self.last_loss, self.last_error)
            self.last_epoch = current_epoch
            self.last_loss = train_loss
            self.last_error = pred_error
            if current_epoch > 0:
                self.callbacks.on_progress(current_epoch, self.total_epochs)
        except Exception:  # noqa: BLE001
            return

    def emit_last(self) -> None:
        if self.last_loss is not None:
            self.callbacks.on_resnet_loss(self.last_epoch, self.last_loss, self.last_error)
