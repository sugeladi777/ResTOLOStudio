from __future__ import annotations

from datetime import datetime
from pathlib import Path

from app.core import InferenceResultRecord, ScanResultRecord, SessionRecord, TrainingResultRecord
from app.services.result_store_service import ResultStoreService


class SessionWorkflowService:
    """Coordinates session creation, selection, result persistence, and scan-result presentation."""

    def __init__(self, result_store: ResultStoreService):
        self.result_store = result_store

    def list_sessions(self) -> list[SessionRecord]:
        return self.result_store.list_sessions()

    def create_session(self, label: str | None = None) -> SessionRecord:
        return self.result_store.create_session(label or "session")

    def rename_session(self, session_id: str, label: str) -> SessionRecord:
        return self.result_store.rename_session(session_id, label)

    def ensure_session(self, current_session: SessionRecord | None, label: str | None = None) -> SessionRecord:
        if current_session is not None:
            return current_session
        return self.create_session(label)

    def scan_dir(self, session_id: str) -> Path:
        return self.result_store.scan_dir(session_id)

    def inference_dir(self, session_id: str) -> Path:
        return self.result_store.inference_dir(session_id)

    def training_dir(self, session_id: str) -> Path:
        return self.result_store.training_dir(session_id)

    def annotation_dir(self, session_id: str) -> Path:
        return self.result_store.annotation_dir(session_id)

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

    def _safe_float(self, value) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _format_nm(self, value: float | None) -> str:
        if value is None:
            return "未记录"
        return f"{value:.2f} nm"

    def _format_value(self, value: float | None, unit: str, precision: int = 3) -> str:
        if value is None:
            return "未记录"
        return f"{value:.{precision}f} {unit}"

    def _format_scientific(self, value: float | None, unit: str) -> str:
        if value is None:
            return "未记录"
        return f"{value:.3e} {unit}"

    def _format_time(self, stamp: str) -> str:
        if not stamp:
            return "未记录"
        try:
            return datetime.fromisoformat(stamp).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            return stamp

    def _result_channels(self, result: ScanResultRecord) -> list[str]:
        raw = getattr(result, "raw", {}) or {}
        request = raw.get("scan_request") or {}
        channels = request.get("channels") or raw.get("channels") or raw.get("scan_channels") or []
        if channels:
            return [str(channel) for channel in channels]
        saved = getattr(result, "saved", []) or []
        names = [str(item.get("channel_name")) for item in saved if item.get("channel_name")]
        return names

    def _scan_info(self, result: ScanResultRecord | None) -> dict:
        if result is None:
            return {}

        raw = getattr(result, "raw", {}) or {}
        request = raw.get("scan_request") or {}
        context = raw.get("scan_context") or {}
        scan_frame = raw.get("scan_frame") or []
        scan_buffer = raw.get("scan_buffer") or []

        width_nm = self._safe_float(request.get("width_nm"))
        height_nm = self._safe_float(request.get("height_nm"))
        center_x_nm = self._safe_float(request.get("center_x_nm"))
        center_y_nm = self._safe_float(request.get("center_y_nm"))
        angle_deg = self._safe_float(request.get("angle_deg"))
        pixels = request.get("pixels")
        lines = request.get("lines")

        if len(scan_frame) >= 5:
            frame_center_x = self._safe_float(scan_frame[0])
            frame_center_y = self._safe_float(scan_frame[1])
            frame_width = self._safe_float(scan_frame[2])
            frame_height = self._safe_float(scan_frame[3])
            center_x_nm = center_x_nm if center_x_nm is not None else (frame_center_x * 1e9 if frame_center_x is not None else None)
            center_y_nm = center_y_nm if center_y_nm is not None else (frame_center_y * 1e9 if frame_center_y is not None else None)
            width_nm = width_nm if width_nm is not None else (frame_width * 1e9 if frame_width is not None else None)
            height_nm = height_nm if height_nm is not None else (frame_height * 1e9 if frame_height is not None else None)
            angle_deg = angle_deg if angle_deg is not None else self._safe_float(scan_frame[4])

        if len(scan_buffer) >= 4:
            if pixels is None:
                pixels = scan_buffer[2]
            if lines is None:
                lines = scan_buffer[3]

        return {
            "label": getattr(result, "label", "") or raw.get("label", ""),
            "captured_at": str(raw.get("captured_at", "") or context.get("captured_at", "")),
            "workflow": str(context.get("workflow", "") or raw.get("workflow", "")),
            "session_label": str(context.get("session_label", "") or raw.get("session_label", "")),
            "width_nm": width_nm,
            "height_nm": height_nm,
            "center_x_nm": center_x_nm,
            "center_y_nm": center_y_nm,
            "angle_deg": angle_deg,
            "pixels": int(pixels) if pixels is not None else None,
            "lines": int(lines) if lines is not None else None,
            "channels": self._result_channels(result),
            "bias_v": self._safe_float(context.get("bias_v")),
            "setpoint_a": self._safe_float(context.get("setpoint_a")),
            "pulse_bias_v": self._safe_float(context.get("pulse_bias_v")),
            "pulse_width_s": self._safe_float(context.get("pulse_width_s")),
            "nanonis_file_path": str(raw.get("nanonis_file_path", "")),
            "saved_count": len(getattr(result, "saved", []) or []),
        }

    def result_overlay_lines(self, result: ScanResultRecord | None) -> list[str]:
        info = self._scan_info(result)
        if not info:
            return []

        lines = [info.get("label") or "扫描结果"]
        width_nm = info.get("width_nm")
        height_nm = info.get("height_nm")
        pixels = info.get("pixels")
        if width_nm is not None and height_nm is not None:
            pixel_text = f"{pixels}px" if pixels is not None else "像素未记录"
            lines.append(f"{width_nm:.2f} × {height_nm:.2f} nm | {pixel_text}")

        bias_v = info.get("bias_v")
        setpoint_a = info.get("setpoint_a")
        if bias_v is not None or setpoint_a is not None:
            bias_text = f"{bias_v:.3f} V" if bias_v is not None else "未记录"
            current_text = f"{setpoint_a:.3e} A" if setpoint_a is not None else "未记录"
            lines.append(f"偏压 {bias_text} | 电流 {current_text}")

        channels = info.get("channels") or []
        if channels:
            lines.append("通道: " + ", ".join(channels[:2]) + ("..." if len(channels) > 2 else ""))

        captured_at = info.get("captured_at", "")
        if captured_at:
            lines.append(self._format_time(captured_at))
        return lines[:5]

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
            info = self._scan_info(item)
            channels = info.get("channels") or []
            channel_text = ", ".join(channels[:2]) if channels else "未记录通道"
            if len(channels) > 2:
                channel_text += "..."
            width_nm = info.get("width_nm")
            height_nm = info.get("height_nm")
            pixels = info.get("pixels")
            size_text = (
                f"{width_nm:.2f}×{height_nm:.2f} nm | {pixels}px"
                if width_nm is not None and height_nm is not None and pixels is not None
                else "尺寸信息待补充"
            )
            time_text = self._format_time(info.get("captured_at", "")).split(" ")[-1] if info.get("captured_at") else "未记录时间"
            labels.append(
                f"{index}. {item.label or f'扫描结果 {index}'}\n"
                f"{size_text} | {time_text} | {channel_text} | 导出 {info.get('saved_count', 0)} 个文件"
            )
        return labels

    def result_detail(self, session: SessionRecord | None, result_index: int) -> str:
        if session is None or result_index < 0 or result_index >= len(session.scan_results):
            return ""

        result = session.scan_results[result_index]
        info = self._scan_info(result)
        saved = getattr(result, "saved", []) or []
        lines = [f"结果：{info.get('label') or f'扫描结果 {result_index + 1}'}"]
        lines.append(f"采集时间：{self._format_time(info.get('captured_at', ''))}")
        lines.append(f"工作流：{info.get('workflow') or '单次扫描'}")
        lines.append(f"尺寸：{self._format_nm(info.get('width_nm'))} × {self._format_nm(info.get('height_nm'))}")
        lines.append(
            "中心："
            f"X={self._format_nm(info.get('center_x_nm'))} | "
            f"Y={self._format_nm(info.get('center_y_nm'))}"
        )
        lines.append(
            f"像素：{info.get('pixels') or '未记录'} × {info.get('lines') or info.get('pixels') or '未记录'}"
        )
        lines.append(f"角度：{self._format_value(info.get('angle_deg'), 'deg', precision=1)}")
        channels = info.get("channels") or []
        lines.append(f"通道：{', '.join(channels) if channels else '未记录'}")
        lines.append(f"偏压：{self._format_value(info.get('bias_v'), 'V')}")
        lines.append(f"设定电流：{self._format_scientific(info.get('setpoint_a'), 'A')}")
        if info.get("pulse_bias_v") is not None or info.get("pulse_width_s") is not None:
            lines.append(
                "脉冲："
                f"{self._format_value(info.get('pulse_bias_v'), 'V')} | "
                f"{self._format_value(info.get('pulse_width_s'), 's')}"
            )
        lines.append(f"导出文件：{len(saved)}")
        if info.get("nanonis_file_path"):
            lines.append(f"SXM 文件：{info['nanonis_file_path']}")
        if saved:
            preview_keys = [key for key in ("png", "jpg", "jpeg", "bmp") if saved[0].get(key)]
            if preview_keys:
                lines.append(f"预览格式：{', '.join(preview_keys)}")
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
