from __future__ import annotations

from pathlib import Path

from app.core import InferenceResultRecord, ScanResultRecord, SessionRecord, TrainingResultRecord
from app.services.result_store_service import ResultStoreService


class SessionWorkflowService:
    """Coordinates session creation, selection, and result persistence."""

    def __init__(self, result_store: ResultStoreService):
        self.result_store = result_store

    def list_sessions(self) -> list[SessionRecord]:
        return self.result_store.list_sessions()

    def create_session(self, label: str | None = None) -> SessionRecord:
        return self.result_store.create_session(label or "session")

    def ensure_session(self, current_session: SessionRecord | None, label: str | None = None) -> SessionRecord:
        if current_session is not None:
            return current_session
        return self.create_session(label)

    def scan_dir(self, session_id: str) -> Path:
        return self.result_store.scan_dir(session_id)

    def inference_dir(self, session_id: str) -> Path:
        return self.result_store.inference_dir(session_id)

    def selected_session(self, index: int) -> SessionRecord | None:
        sessions = self.list_sessions()
        if index < 0 or index >= len(sessions):
            return None
        return sessions[index]

    def selected_scan_result(
        self,
        session_index: int,
        result_index: int,
    ) -> tuple[SessionRecord | None, ScanResultRecord | None]:
        session = self.selected_session(session_index)
        if session is None:
            return None, None
        if result_index < 0 or result_index >= len(session.scan_results):
            return session, None
        return session, session.scan_results[result_index]

    def session_list_labels(self) -> list[str]:
        return [
            f"{session.id} | scan={len(session.scan_results)} | infer={len(session.inference_results)}"
            for session in self.list_sessions()
        ]

    def result_labels(self, session: SessionRecord | None) -> list[str]:
        if session is None:
            return []
        return [item.label for item in session.scan_results]

    def result_detail(self, session: SessionRecord | None, result_index: int) -> str:
        if session is None or result_index < 0 or result_index >= len(session.scan_results):
            return ""
        return str(session.scan_results[result_index].to_dict())

    def append_scan_results(self, session_id: str, *results: dict | ScanResultRecord) -> SessionRecord:
        manifest = None
        for result in results:
            manifest = self.result_store.append_scan_result(session_id, result)
        return manifest or self.result_store.load_manifest(session_id)

    def append_inference_result(
        self,
        session_id: str,
        result: dict | InferenceResultRecord,
    ) -> SessionRecord:
        return self.result_store.append_inference_result(session_id, result)

    def append_training_result(
        self,
        session_id: str,
        result: dict | TrainingResultRecord,
    ) -> SessionRecord:
        return self.result_store.append_training_result(session_id, result)
