from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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

    def status(self) -> dict:
        client = self._require()
        return {
            "connection": client.connection_summary(),
            "status": client.read_status(),
        }

    def signal_names(self) -> list[str]:
        return list(self._require().signal_names())

    def apply_scan(self, width_nm: float, height_nm: float, center_x_nm: float, center_y_nm: float, angle_deg: float, pixels: int, channels: Iterable[str]) -> dict:
        client = self._require()
        client.set_scan_frame_nm(width_nm, height_nm, center_x_nm, center_y_nm, angle_deg)
        channel_indexes = client.scan_channel_indexes_for_signal_names(list(channels))
        client.set_scan_buffer(channel_indexes=channel_indexes, pixels=pixels, lines=pixels)
        return client.read_status()

    def set_bias(self, bias_v: float) -> dict:
        client = self._require()
        client.set_bias(bias_v)
        return client.read_status()

    def set_setpoint(self, setpoint_a: float) -> dict:
        client = self._require()
        client.set_setpoint(setpoint_a)
        return client.read_status()

    def set_feedback(self, enabled: bool) -> dict:
        client = self._require()
        client.set_feedback(enabled)
        return client.read_status()

    def scan_and_save(self, *, label: str, width_nm: float, height_nm: float, center_x_nm: float, center_y_nm: float, angle_deg: float, pixels: int, channels: Iterable[str], timeout_ms: int = 300000, direction: int = 1) -> dict:
        client = self._require()
        channel_indexes = client.scan_channel_indexes_for_signal_names(list(channels))
        return client.scan_and_save(
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
