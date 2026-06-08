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

    def _session_stage_text(self, session: SessionRecord) -> str:
        if session.inference_results:
            return "结果复查"
        if session.training_results:
            return "待推理"
        if session.scan_results:
            return "待训练"
        return "采集准备"

    def session_list_labels(self, sessions: list[SessionRecord] | None = None) -> list[str]:
        source = sessions if sessions is not None else self.list_sessions()
        return [
            (
                f"{getattr(session, 'label', '') or session.id} | {self._session_stage_text(session)}\n"
                f"扫描 {len(session.scan_results)} 组  训练 {len(session.training_results)} 次  推理 {len(session.inference_results)} 次"
            )
            for session in source
        ]

    def result_labels(self, session: SessionRecord | None) -> list[str]:
        if session is None:
            return []
        labels: list[str] = []
        for index, item in enumerate(session.scan_results, start=1):
            saved = len(getattr(item, "saved", []) or [])
            raw = getattr(item, "raw", {}) or {}
            channels = raw.get("channels") or raw.get("scan_channels") or []
            channel_text = ", ".join(str(channel) for channel in channels[:2]) if channels else "未记录通道"
            if len(channels) > 2:
                channel_text += "..."
            labels.append(
                f"{index}. {item.label or f'扫描结果 {index}'}\n"
                f"导出 {saved} 个文件 | {channel_text}"
            )
        return labels

    def result_detail(self, session: SessionRecord | None, result_index: int) -> str:
        if session is None or result_index < 0 or result_index >= len(session.scan_results):
            return ""
        result = session.scan_results[result_index]
        raw = getattr(result, "raw", {}) or {}
        saved = getattr(result, "saved", []) or []
        channels = raw.get("channels") or raw.get("scan_channels") or []
        lines = [f"结果：{result.label or f'扫描结果 {result_index + 1}'}"]
        lines.append(f"导出文件：{len(saved)}")
        lines.append(f"通道：{', '.join(str(channel) for channel in channels) if channels else '未记录'}")
        if saved:
            preview_keys = [key for key in ("png", "jpg", "jpeg", "bmp") if saved[0].get(key)]
            if preview_keys:
                lines.append(f"可预览格式：{', '.join(preview_keys)}")
        return "\n".join(lines)

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
