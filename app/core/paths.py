from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppPaths:
    project_root: Path
    assets_root: Path
    models_root: Path
    error_patterns_path: Path
    yolo_config_path: Path
    yolo_hyperparameters_path: Path
    sessions_root: Path
    config_path: Path

    @classmethod
    def from_project_root(cls, project_root: Path) -> "AppPaths":
        root = Path(project_root).resolve()
        assets_root = root / "assets"
        return cls(
            project_root=root,
            assets_root=assets_root,
            models_root=assets_root / "models",
            error_patterns_path=assets_root / "config" / "error_patterns.yaml",
            yolo_config_path=root / "ml" / "models" / "yolov5m_molecule.yaml",
            yolo_hyperparameters_path=root / "ml" / "data" / "hyp.scratch.yaml",
            sessions_root=root / "sessions",
            config_path=root / "restolo_studio.json",
        )

    @property
    def default_yolo_model_path(self) -> Path:
        return self.models_root / "best.pt"

    @property
    def default_resnet_model_path(self) -> Path:
        return self.models_root / "demo.saving"

    @property
    def default_classes_path(self) -> Path:
        return self.models_root / "classes.yaml"

    @property
    def ml_root(self) -> Path:
        return self.project_root / "ml"
