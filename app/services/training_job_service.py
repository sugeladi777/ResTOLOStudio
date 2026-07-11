from __future__ import annotations

import os
import json
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
            "training_metadata": self.load_training_metadata(output_dir),
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
        exact_final = Path(output_dir) / "Model_final.saving"
        if exact_final.exists():
            return str(exact_final)
        exact_best = Path(output_dir) / "Model_best.saving"
        if exact_best.exists():
            return str(exact_best)
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

    def load_training_metadata(self, output_dir: str) -> dict:
        if not output_dir:
            return {}
        report_path = Path(output_dir) / "training_report.json"
        if report_path.exists():
            try:
                return json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return {}
        results_path = Path(output_dir) / "results.txt"
        if results_path.exists():
            lines = [line.strip() for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if lines:
                values = lines[-1].split()
                if len(values) >= 15:
                    try:
                        metrics = [float(value) for value in values[-7:]]
                        return {
                            "precision": metrics[0],
                            "recall": metrics[1],
                            "map50": metrics[2],
                            "map50_95": metrics[3],
                        }
                    except ValueError:
                        pass
        return {}

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
