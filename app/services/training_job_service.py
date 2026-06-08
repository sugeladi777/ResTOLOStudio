from __future__ import annotations

import os

from app.core import TrainingResultRecord


class TrainingJobService:
    """Creates persisted training job records from UI/runtime context."""

    def start_context(self, session_id: str, mode: str, project_dir: str) -> dict:
        return {
            "session_id": session_id,
            "mode": mode,
            "project_dir": project_dir,
        }

    def complete_record(self, context: dict, output_dir: str, status: str) -> TrainingResultRecord:
        payload = {
            "mode": context.get("mode", ""),
            "project_dir": context.get("project_dir", ""),
            "output_dir": output_dir,
            "status": status,
        }
        return TrainingResultRecord(
            mode=payload["mode"],
            project_dir=payload["project_dir"],
            output_dir=payload["output_dir"],
            status=payload["status"],
            raw=payload,
        )

    def output_dir_for_window(self, window) -> str:
        if getattr(window, "_yolo_project_dir", None):
            path = os.path.join(window._yolo_project_dir, "yolo_train")
            if os.path.exists(path):
                return path
        if getattr(window, "_resnet_project_dir", None):
            path = os.path.join(window._resnet_project_dir, "resnet_train")
            if os.path.exists(path):
                return path
        return ""
