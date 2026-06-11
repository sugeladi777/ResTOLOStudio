from __future__ import annotations

import os
from pathlib import Path

from app.core import TrainingResultRecord


class TrainingJobService:
    """Creates persisted training job records from UI/runtime context."""

    def start_context(self, session_id: str, mode: str, project_dir: str) -> dict:
        expected_output_dir = os.path.join(project_dir, "yolo_train" if mode == "yolo" else "resnet_train")
        return {
            "session_id": session_id,
            "mode": mode,
            "project_dir": project_dir,
            "expected_output_dir": expected_output_dir,
        }

    def complete_record(self, context: dict, status: str) -> TrainingResultRecord:
        output_dir = self.output_dir_for_context(context)
        yolo_model_path = self.find_yolo_model(output_dir)
        resnet_model_path = self.find_resnet_model(output_dir)
        classes_yaml_path = self.find_classes_yaml(context, output_dir)
        payload = {
            "mode": context.get("mode", ""),
            "project_dir": context.get("project_dir", ""),
            "output_dir": output_dir,
            "status": status,
            "yolo_model_path": yolo_model_path,
            "resnet_model_path": resnet_model_path,
            "classes_yaml_path": classes_yaml_path,
        }
        return TrainingResultRecord(
            mode=payload["mode"],
            project_dir=payload["project_dir"],
            output_dir=payload["output_dir"],
            status=payload["status"],
            yolo_model_path=payload["yolo_model_path"],
            resnet_model_path=payload["resnet_model_path"],
            classes_yaml_path=payload["classes_yaml_path"],
            raw=payload,
        )

    def output_dir_for_context(self, context: dict) -> str:
        expected = str(context.get("expected_output_dir", "") or "")
        if expected and os.path.exists(expected):
            return expected
        project_dir = str(context.get("project_dir", "") or "")
        mode = str(context.get("mode", "") or "")
        if project_dir and mode:
            fallback = os.path.join(project_dir, "yolo_train" if mode == "yolo" else "resnet_train")
            if os.path.exists(fallback):
                return fallback
            return fallback
        return expected

    def find_yolo_model(self, output_dir: str) -> str:
        if not output_dir:
            return ""
        weights_dir = Path(output_dir) / "weights"
        for candidate in (weights_dir / "best.pt", weights_dir / "last.pt"):
            if candidate.exists():
                return str(candidate)
        return ""

    def find_resnet_model(self, output_dir: str) -> str:
        if not output_dir or not os.path.isdir(output_dir):
            return ""
        patterns = ("Model_best_*.saving", "Model_*.saving", "*.saving", "*.pt", "*.pth")
        for pattern in patterns:
            matches = sorted(
                Path(output_dir).glob(pattern),
                key=lambda item: item.stat().st_mtime_ns,
                reverse=True,
            )
            if matches:
                return str(matches[0])
        return ""

    def find_classes_yaml(self, context: dict, output_dir: str) -> str:
        candidates: list[Path] = []
        if output_dir:
            candidates.append(Path(output_dir) / "classes.yaml")
            candidates.append(Path(output_dir).parent / "classes.yaml")
        project_dir = str(context.get("project_dir", "") or "")
        if project_dir:
            candidates.append(Path(project_dir) / "classes.yaml")
            candidates.append(Path(project_dir) / "resnet_crop" / "classes.yaml")
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return ""
