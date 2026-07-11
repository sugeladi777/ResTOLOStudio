import os
import shutil
import sys
import tempfile
import threading
import traceback
from pathlib import Path

from PIL import Image
from PyQt5.QtCore import QObject, pyqtSignal

from app.core import AppPaths

APP_PATHS = AppPaths.from_project_root(Path(__file__).resolve().parents[2])
PROJECT_ROOT = APP_PATHS.project_root
ML_ROOT = APP_PATHS.ml_root


class _StreamRelay:
    def __init__(self, callback):
        self.callback = callback
        self.buffer = []

    def write(self, data):
        self.buffer.append(data)
        if "\n" in data:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        output = "".join(self.buffer)
        self.buffer = []
        self.callback(output)


class InferenceManager(QObject):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.images = []
        self.inference_logs = []
        self.log_callback = None
        self.finished_callback = None
        self.error_callback = None
        self.actual_output_dir = None

    def set_log_callback(self, callback):
        self.log_callback = callback
        self.log_signal.connect(callback)

    def set_finished_callback(self, callback):
        self.finished_callback = callback
        self.finished_signal.connect(callback)

    def set_error_callback(self, callback):
        self.error_callback = callback
        self.error_signal.connect(callback)

    def load_images(self, image_paths):
        self.images = image_paths

    def run_inference(self, yolo_model, resnet_model, output_dir, classes_yaml=""):
        if not self.images:
            return

        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(yolo_model):
            self._emit_error(f"错误: YOLO模型文件不存在: {yolo_model}")
            return
        if not os.path.exists(resnet_model):
            self._emit_error(f"错误: ResNet模型文件不存在: {resnet_model}")
            return

        temp_dir = tempfile.mkdtemp(prefix="restolo_infer_")
        for image_path in self.images:
            shutil.copy(image_path, temp_dir)

        self._run_inference_directly(
            yolo_model=yolo_model,
            source=temp_dir,
            resnet_dir=os.path.dirname(resnet_model) + "\\",
            resnet_name=os.path.basename(resnet_model),
            project=output_dir,
            name="inference",
            classes_yaml=classes_yaml,
            cleanup_source=True,
        )

    def _run_inference_directly(
        self,
        yolo_model,
        source,
        resnet_dir,
        resnet_name,
        project,
        name,
        classes_yaml="",
        cleanup_source=False,
    ):
        def run():
            self.inference_logs = []
            self.actual_output_dir = None
            total_images = len(self.images)
            current_image = 0

            try:
                sys.path.append(str(PROJECT_ROOT))
                sys.path.append(str(ML_ROOT))

                import torch
                import ReSTolo_detect

                original_argv = sys.argv.copy()
                original_stdout = sys.stdout
                original_stderr = sys.stderr

                def handle_output(data):
                    nonlocal current_image
                    for raw_line in data.splitlines():
                        line = raw_line.strip()
                        if not line:
                            continue
                        self.inference_logs.append(line)
                        self._emit_log(line)

                        if "image " in line and "/" in line:
                            try:
                                progress = line.split("image ", 1)[1].split(" ", 1)[0]
                                current, total = map(int, progress.split("/"))
                                current_image = current
                                if total > 0:
                                    self.progress_signal.emit(int(((current - 1) / total) * 100), 100)
                            except Exception:
                                pass
                        elif "/" in line and "type:" in line:
                            try:
                                box_progress = line.split(" ", 1)[0]
                                current_box, total_boxes = map(int, box_progress.split("/"))
                                if total_images > 0 and total_boxes > 0:
                                    image_progress = (current_image - 1) / total_images
                                    box_fraction = current_box / total_boxes / total_images
                                    self.progress_signal.emit(int((image_progress + box_fraction) * 100), 100)
                            except Exception:
                                pass
                        elif "Done." in line and "s)" in line and total_images > 0:
                            self.progress_signal.emit(int((current_image / total_images) * 100), 100)

                        if "Results saved to" in line:
                            parts = line.split("Results saved to ", 1)
                            if len(parts) > 1:
                                self.actual_output_dir = parts[1].strip()

                relay = _StreamRelay(handle_output)
                sys.stdout = relay
                sys.stderr = relay

                try:
                    device = "cuda:0" if torch.cuda.is_available() else "cpu"
                    inference_settings = {}
                    try:
                        checkpoint = torch.load(yolo_model, map_location="cpu", weights_only=False)
                        if isinstance(checkpoint, dict):
                            inference_settings = dict(checkpoint.get("inference_settings", {}) or {})
                    except Exception:  # noqa: BLE001
                        inference_settings = {}
                    sys.argv = [
                        "ReSTolo_detect.py",
                        "--weights",
                        yolo_model,
                        "--source",
                        source,
                        "--resnet_dir",
                        resnet_dir,
                        "--resnet_name",
                        resnet_name,
                        "--project",
                        project,
                        "--name",
                        name,
                        "--save-txt",
                        "--save-conf",
                        "--device",
                        device,
                    ]
                    if "conf_thres" in inference_settings:
                        sys.argv.extend(["--conf-thres", str(inference_settings["conf_thres"])])
                    if "iou_thres" in inference_settings:
                        sys.argv.extend(["--iou-thres", str(inference_settings["iou_thres"])])
                    if classes_yaml and os.path.exists(classes_yaml):
                        sys.argv.extend(["--classes_yaml", classes_yaml])

                    parser = self._build_detect_parser(device)
                    ReSTolo_detect.opt = parser.parse_args()

                    try:
                        ReSTolo_detect.detect()
                    except RuntimeError as exc:
                        if "CUDA error" not in str(exc):
                            raise
                        self._emit_log(f"CUDA错误: {exc}")
                        self._emit_log("尝试使用CPU运行...")
                        sys.argv = [arg for arg in sys.argv if arg != device]
                        sys.argv.extend(["--device", "cpu"])
                        ReSTolo_detect.opt = parser.parse_args()
                        ReSTolo_detect.detect()
                finally:
                    relay.flush()
                    sys.stdout = original_stdout
                    sys.stderr = original_stderr
                    sys.argv = original_argv
                    self.progress_signal.emit(100, 100)
            except Exception as exc:
                self._emit_error(f"错误: {exc}\n{traceback.format_exc()}")
                return
            finally:
                if cleanup_source:
                    shutil.rmtree(source, ignore_errors=True)
                self.finished_signal.emit()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

    def _build_detect_parser(self, device):
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--weights", nargs="+", type=str, default=str(APP_PATHS.default_yolo_model_path))
        parser.add_argument("--source", type=str, default="dataset/low_reso")
        parser.add_argument("--img-size", type=int, default=640)
        parser.add_argument("--conf-thres", type=float, default=0.25)
        parser.add_argument("--iou-thres", type=float, default=0.45)
        parser.add_argument("--class-conf-thres", type=float, default=0.5)
        parser.add_argument("--device", default=device)
        parser.add_argument("--view-img", action="store_true")
        parser.add_argument("--save-txt", action="store_true")
        parser.add_argument("--save-conf", action="store_true")
        parser.add_argument("--nosave", action="store_true", default=False)
        parser.add_argument("--classes", nargs="+", type=int)
        parser.add_argument("--agnostic-nms", action="store_true")
        parser.add_argument("--augment", action="store_true")
        parser.add_argument("--update", action="store_true")
        parser.add_argument("--project", default="runs/detect")
        parser.add_argument("--name", default="test")
        parser.add_argument("--exist-ok", action="store_true")
        parser.add_argument("--mag", default=1, type=float)
        parser.add_argument("--resnet_dir", default=str(APP_PATHS.models_root) + os.sep, type=str)
        parser.add_argument("--resnet_name", default=APP_PATHS.default_resnet_model_path.name, type=str)
        parser.add_argument("--classes_yaml", default="", type=str)
        return parser

    def _emit_log(self, message):
        if self.log_callback:
            self.log_signal.emit(message)
        else:
            print(message)

    def _emit_error(self, message):
        self.inference_logs.append(message)
        if self.log_callback:
            self.log_signal.emit(message)
        if self.error_callback:
            self.error_signal.emit(message)

    def get_inference_results(self, output_dir):
        result_images = []
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}

        def find_images(directory):
            images = []
            if not os.path.exists(directory):
                return images
            for root, _, files in os.walk(directory):
                for file_name in files:
                    if os.path.splitext(file_name)[1] in image_extensions:
                        images.append(os.path.abspath(os.path.join(root, file_name)))
            return images

        candidate_dir = self.actual_output_dir if self.actual_output_dir and os.path.exists(self.actual_output_dir) else output_dir
        result_images.extend(find_images(candidate_dir))

        valid_images = []
        for image_path in result_images:
            if not (os.path.exists(image_path) and os.path.isfile(image_path)):
                continue
            try:
                with Image.open(image_path):
                    valid_images.append(image_path)
            except Exception:
                continue
        return valid_images
