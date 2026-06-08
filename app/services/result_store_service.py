from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


class ResultStoreService:
    def __init__(self, sessions_root: Path):
        self.sessions_root = Path(sessions_root)
        self.sessions_root.mkdir(parents=True, exist_ok=True)

    def create_session(self, label: str | None = None) -> dict:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_label = (label or "session").replace(" ", "_")
        session_dir = self.sessions_root / f"{stamp}_{safe_label}"
        for name in ("scan", "inference", "training", "logs"):
            (session_dir / name).mkdir(parents=True, exist_ok=True)
        manifest = {
            "id": session_dir.name,
            "label": label or "session",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "scan_results": [],
            "inference_results": [],
            "training_results": [],
        }
        self._write_manifest(session_dir, manifest)
        return manifest

    def session_dir(self, session_id: str) -> Path:
        return self.sessions_root / session_id

    def scan_dir(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "scan"

    def inference_dir(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "inference"

    def manifest_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "session_manifest.json"

    def load_manifest(self, session_id: str) -> dict:
        return json.loads(self.manifest_path(session_id).read_text(encoding="utf-8"))

    def _write_manifest(self, session_dir: Path, payload: dict) -> None:
        (session_dir / "session_manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def append_scan_result(self, session_id: str, result: dict) -> dict:
        manifest = self.load_manifest(session_id)
        manifest["scan_results"].append(result)
        self._write_manifest(self.session_dir(session_id), manifest)
        return manifest

    def append_inference_result(self, session_id: str, result: dict) -> dict:
        manifest = self.load_manifest(session_id)
        manifest["inference_results"].append(result)
        self._write_manifest(self.session_dir(session_id), manifest)
        return manifest

    def list_sessions(self) -> list[dict]:
        results = []
        for manifest_path in sorted(self.sessions_root.glob("*/session_manifest.json"), reverse=True):
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            payload["path"] = str(manifest_path.parent)
            results.append(payload)
        return results
