from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from contextlib import nullcontext

from nanonis.ExperimentClient import NanonisExperimentClient


@dataclass
class NanonisConnectionConfig:
    ip: str = "127.0.0.1"
    port: int = 6501
    version: int = 10380
    max_buf_size: int = 1024 * 1024
    connect_timeout: float = 5.0
    response_timeout: float = 10.0


class NanonisSessionService:
    def __init__(self, output_root: Path):
        self.output_root = Path(output_root)
        self.client: NanonisExperimentClient | None = None
        self.config = NanonisConnectionConfig()

    def set_output_root(self, output_root: Path) -> None:
        self.output_root = Path(output_root)
        if self.client is not None:
            self.client.output_dir = self.output_root

    @property
    def connected(self) -> bool:
        return self.client is not None

    def connect(self, config: NanonisConnectionConfig | None = None) -> dict:
        if config is not None:
            self.config = config
        if self.client is not None:
            return self.status()
        self.client = NanonisExperimentClient(
            ip=self.config.ip,
            port=self.config.port,
            version=self.config.version,
            output_dir=str(self.output_root),
            max_buf_size=self.config.max_buf_size,
            connect_timeout=self.config.connect_timeout,
            response_timeout=self.config.response_timeout,
        )
        self.client.connect()
        return self.status()

    def disconnect(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None

    def _require(self) -> NanonisExperimentClient:
        if self.client is None:
            raise RuntimeError("Nanonis is not connected")
        return self.client

    def _connection_summary(self, client: NanonisExperimentClient) -> dict:
        summary_fn = getattr(client, "connection_summary", None)
        if callable(summary_fn):
            return summary_fn()
        return {
            "ip": self.config.ip,
            "port": self.config.port,
            "version": self.config.version,
        }

    def _slow_command_timeout(self, client: NanonisExperimentClient, extra_seconds: float = 15.0):
        timeout_s = max(self.config.response_timeout + extra_seconds, self.config.response_timeout)
        temporary_timeout = getattr(client, "temporary_response_timeout", None)
        if callable(temporary_timeout):
            return temporary_timeout(timeout_s)
        return nullcontext()

    def _safe_status(self, *, fallback_message: str) -> dict:
        client = self._require()
        try:
            return {
                "connection": self._connection_summary(client),
                "status": client.read_status(),
            }
        except TimeoutError:
            return {
                "connection": self._connection_summary(client),
                "status": fallback_message,
            }

    def status(self) -> dict:
        return self._safe_status(fallback_message="状态读取超时，但连接已建立。")

    def signal_names(self) -> list[str]:
        return list(self._require().signal_names())

    def _resolve_scan_channels(self, channels: Iterable[str]) -> tuple[int, ...]:
        client = self._require()
        requested = list(channels)
        try:
            with self._slow_command_timeout(client):
                return client.scan_channel_indexes_for_signal_names(requested)
        except (TimeoutError, ValueError):
            with self._slow_command_timeout(client):
                buffer_info = client.scan.BufferGet()
            current_indexes = tuple(buffer_info[1]) if len(buffer_info) > 1 else ()
            if current_indexes:
                return current_indexes
            raise

    def apply_scan(self, width_nm: float, height_nm: float, center_x_nm: float, center_y_nm: float, angle_deg: float, pixels: int, channels: Iterable[str]) -> dict:
        client = self._require()
        with self._slow_command_timeout(client):
            client.set_scan_frame_nm(width_nm, height_nm, center_x_nm, center_y_nm, angle_deg)
            channel_indexes = self._resolve_scan_channels(channels)
            client.set_scan_buffer(channel_indexes=channel_indexes, pixels=pixels, lines=pixels)
        return self._safe_status(fallback_message="扫描参数已应用，但状态读取超时。")

    def configure_scan(
        self,
        *,
        bias_v: float,
        setpoint_a: float,
        width_nm: float,
        height_nm: float,
        center_x_nm: float,
        center_y_nm: float,
        angle_deg: float,
        pixels: int,
        channels: Iterable[str],
    ) -> dict:
        client = self._require()
        with self._slow_command_timeout(client):
            client.set_bias(bias_v)
            client.set_setpoint(setpoint_a)
            client.set_scan_frame_nm(width_nm, height_nm, center_x_nm, center_y_nm, angle_deg)
            channel_indexes = self._resolve_scan_channels(channels)
            client.set_scan_buffer(channel_indexes=channel_indexes, pixels=pixels, lines=pixels)
        return self._safe_status(fallback_message="扫描前参数已下发，但状态读取超时。")

    def set_bias(self, bias_v: float) -> dict:
        client = self._require()
        client.set_bias(bias_v)
        return self._safe_status(fallback_message="偏压已设置，但状态读取超时。")

    def set_setpoint(self, setpoint_a: float) -> dict:
        client = self._require()
        client.set_setpoint(setpoint_a)
        return self._safe_status(fallback_message="电流设定已应用，但状态读取超时。")

    def set_feedback(self, enabled: bool) -> dict:
        client = self._require()
        client.set_feedback(enabled)
        return self._safe_status(fallback_message="反馈状态已切换，但状态读取超时。")

    def scan_and_save(self, *, label: str, width_nm: float, height_nm: float, center_x_nm: float, center_y_nm: float, angle_deg: float, pixels: int, channels: Iterable[str], timeout_ms: int = 300000, direction: int = 1) -> dict:
        client = self._require()
        channel_indexes = self._resolve_scan_channels(channels)
        result = client.scan_and_save(
            label=label,
            width_nm=width_nm,
            height_nm=height_nm,
            center_x_nm=center_x_nm,
            center_y_nm=center_y_nm,
            angle_deg=angle_deg,
            channel_indexes=channel_indexes,
            pixels=pixels,
            lines=pixels,
            direction=direction,
            timeout_ms=timeout_ms,
        )
        result.setdefault("captured_at", datetime.now().isoformat(timespec="seconds"))
        result.setdefault(
            "scan_request",
            {
                "width_nm": width_nm,
                "height_nm": height_nm,
                "center_x_nm": center_x_nm,
                "center_y_nm": center_y_nm,
                "angle_deg": angle_deg,
                "pixels": pixels,
                "lines": pixels,
                "channels": [str(channel) for channel in channels],
            },
        )
        return result

    def configure_and_scan_and_save(
        self,
        *,
        label: str,
        bias_v: float,
        setpoint_a: float,
        width_nm: float,
        height_nm: float,
        center_x_nm: float,
        center_y_nm: float,
        angle_deg: float,
        pixels: int,
        channels: Iterable[str],
        timeout_ms: int = 300000,
        direction: int = 1,
    ) -> dict:
        self.configure_scan(
            bias_v=bias_v,
            setpoint_a=setpoint_a,
            width_nm=width_nm,
            height_nm=height_nm,
            center_x_nm=center_x_nm,
            center_y_nm=center_y_nm,
            angle_deg=angle_deg,
            pixels=pixels,
            channels=channels,
        )
        return self.scan_and_save(
            label=label,
            width_nm=width_nm,
            height_nm=height_nm,
            center_x_nm=center_x_nm,
            center_y_nm=center_y_nm,
            angle_deg=angle_deg,
            pixels=pixels,
            channels=channels,
            timeout_ms=timeout_ms,
            direction=direction,
        )

    def bias_pulse(self, bias_value: float, bias_pulse_width: float, z_hold: int = 1, rel_abs: int = 2) -> object:
        client = self._require()
        return client.execute_module_method(
            "Bias",
            "Pulse",
            {
                "bias_pulse_width": bias_pulse_width,
                "bias_value": bias_value,
                "z_hold": z_hold,
                "rel_abs": rel_abs,
                "wait_until_done": 1,
            },
        )


class ScanWorkflowService:
    def __init__(self, session_service: NanonisSessionService):
        self.session_service = session_service

    def scan_pulse_scan(
        self,
        *,
        label: str,
        bias_v: float,
        setpoint_a: float,
        width_nm: float,
        height_nm: float,
        center_x_nm: float,
        center_y_nm: float,
        angle_deg: float,
        pixels: int,
        channels: Iterable[str],
        pulse_bias_v: float,
        pulse_width_s: float,
        timeout_ms: int = 300000,
        direction: int = 1,
    ) -> dict:
        self.session_service.configure_scan(
            bias_v=bias_v,
            setpoint_a=setpoint_a,
            width_nm=width_nm,
            height_nm=height_nm,
            center_x_nm=center_x_nm,
            center_y_nm=center_y_nm,
            angle_deg=angle_deg,
            pixels=pixels,
            channels=channels,
        )
        pre = self.session_service.scan_and_save(
            label=f"{label}_pre",
            width_nm=width_nm,
            height_nm=height_nm,
            center_x_nm=center_x_nm,
            center_y_nm=center_y_nm,
            angle_deg=angle_deg,
            pixels=pixels,
            channels=channels,
            timeout_ms=timeout_ms,
            direction=direction,
        )
        pulse = self.session_service.bias_pulse(pulse_bias_v, pulse_width_s)
        post = self.session_service.scan_and_save(
            label=f"{label}_post",
            width_nm=width_nm,
            height_nm=height_nm,
            center_x_nm=center_x_nm,
            center_y_nm=center_y_nm,
            angle_deg=angle_deg,
            pixels=pixels,
            channels=channels,
            timeout_ms=timeout_ms,
            direction=direction,
        )
        return {"pre_scan": pre, "pulse": pulse, "post_scan": post}
