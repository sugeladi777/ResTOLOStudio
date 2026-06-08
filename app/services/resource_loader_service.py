from __future__ import annotations

import os
from pathlib import Path


class ResourceLoaderService:
    """Coordinates model/data/classes path loading and simple resource validation."""

    def apply_default_model_paths(
        self,
        runtime_paths,
        model_manager,
        train_yolo_target,
        infer_yolo_target,
        train_resnet_target,
        infer_resnet_target,
        infer_classes_target,
    ) -> None:
        self._set_default_model_path(
            runtime_paths.default_yolo_model_path,
            model_manager.load_yolo_model,
            train_yolo_target,
            infer_yolo_target,
        )
        self._set_default_model_path(
            runtime_paths.default_resnet_model_path,
            model_manager.load_resnet_model,
            train_resnet_target,
            infer_resnet_target,
        )
        if runtime_paths.default_classes_path.exists():
            infer_classes_target.setText(str(runtime_paths.default_classes_path))

    def load_yolo_model(self, file_path: str, model_manager, log, *targets) -> bool:
        if not file_path:
            return False
        model_manager.load_yolo_model(file_path)
        log(f"Loaded YOLO model: {file_path}")
        for target in targets:
            if target is not None:
                target.setText(file_path)
        return True

    def load_resnet_model(self, file_path: str, model_manager, log, *targets) -> bool:
        if not file_path:
            return False
        model_manager.load_resnet_model(file_path)
        log(f"Loaded ResNet model: {file_path}")
        for target in targets:
            if target is not None:
                target.setText(file_path)
        return True

    def load_classes_file(self, file_path: str, log, *targets) -> bool:
        if not file_path:
            return False
        log(f"Loaded classes YAML: {file_path}")
        for target in targets:
            if target is not None:
                target.setText(file_path)
        return True

    def load_resnet_dataset(self, directory: str, log, target=None) -> tuple[bool, int | None]:
        if not directory:
            return False, None
        log(f"Loaded ResNet dataset: {directory}")
        if target is not None:
            target.setText(directory)
        try:
            classes = os.listdir(directory)
            num_classes = len([item for item in classes if os.path.isdir(os.path.join(directory, item))])
            log(f"Detected {num_classes} classes")
            return True, num_classes
        except Exception as exc:  # noqa: BLE001
            log(f"Failed to validate dataset directory: {exc}")
            return True, None

    def _set_default_model_path(self, path: Path, loader, *targets) -> None:
        if not path.exists():
            return
        resolved = str(path)
        loader(resolved)
        for target in targets:
            target.setText(resolved)
