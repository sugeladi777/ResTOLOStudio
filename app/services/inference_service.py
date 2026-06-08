from __future__ import annotations

from pathlib import Path

from app.core import ScanResultRecord
from app.utils.inference_manager import InferenceManager


class InferenceService:
    def __init__(self, manager: InferenceManager):
        self.manager = manager

    def infer_files(self, files: list[str], yolo_model: str, resnet_model: str, output_dir: Path, classes_yaml: str = "") -> None:
        self.manager.load_images(files)
        self.manager.run_inference(yolo_model, resnet_model, str(output_dir), classes_yaml=classes_yaml)

    def scan_result_files(self, scan_result: dict | ScanResultRecord) -> list[str]:
        if isinstance(scan_result, ScanResultRecord):
            payload = scan_result.to_dict()
        else:
            payload = scan_result
        files = []
        for item in payload.get("saved", []):
            png_path = item.get("png")
            if png_path:
                files.append(str(Path(png_path)))
        return files
