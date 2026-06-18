from __future__ import annotations

from dataclasses import dataclass

from app.core import ScanResultRecord, SessionRecord
from app.services.session_workflow_service import SessionWorkflowService
from nanonis.services import NanonisConnectionConfig


@dataclass
class ScanGeometry:
    width_nm: float
    height_nm: float
    center_x_nm: float
    center_y_nm: float
    angle_deg: float | None
    pixels: int
    channels: list[str]

    def to_dict(self) -> dict:
        return {
            "width_nm": self.width_nm,
            "height_nm": self.height_nm,
            "center_x_nm": self.center_x_nm,
            "center_y_nm": self.center_y_nm,
            "angle_deg": self.angle_deg,
            "pixels": self.pixels,
            "channels": list(self.channels),
        }


class AcquisitionWorkflowService:
    """Coordinates Nanonis connection settings, scan parameters, and scan result persistence."""

    def __init__(self, session_workflow_service: SessionWorkflowService):
        self.session_workflow_service = session_workflow_service

    def build_connection_config(self, ip: str, port: str, version: str) -> NanonisConnectionConfig:
        return NanonisConnectionConfig(
            ip=ip.strip(),
            port=int(port.strip()),
            version=int(version.strip()),
        )

    def build_scan_geometry(
        self,
        width_nm: str,
        height_nm: str,
        center_x_nm: str,
        center_y_nm: str,
        pixels: str,
        channels: str,
    ) -> ScanGeometry:
        return ScanGeometry(
            width_nm=float(width_nm.strip()),
            height_nm=float(height_nm.strip()),
            center_x_nm=float(center_x_nm.strip()),
            center_y_nm=float(center_y_nm.strip()),
            angle_deg=None,
            pixels=int(pixels.strip()),
            channels=[part.strip() for part in channels.split(",") if part.strip()],
        )

    def start_scan_session(
        self,
        current_session: SessionRecord | None,
        label: str,
        nanonis_service,
    ) -> SessionRecord:
        requested_label = label.strip() or "session"
        session = current_session or self.session_workflow_service.create_session(requested_label)
        if getattr(session, "label", "") != requested_label:
            session = self.session_workflow_service.rename_session(session.id, requested_label)
        output_root = self.session_workflow_service.scan_dir(session.id)
        if hasattr(nanonis_service, "set_output_root"):
            nanonis_service.set_output_root(output_root)
        else:
            nanonis_service.output_root = output_root
        return session

    def append_scan_result(self, session_id: str, result: dict | ScanResultRecord) -> ScanResultRecord:
        record = result if isinstance(result, ScanResultRecord) else ScanResultRecord.from_dict(result)
        self.session_workflow_service.append_scan_results(session_id, record)
        return record

    def append_scan_workflow_results(self, session_id: str, result: dict) -> tuple[ScanResultRecord, ScanResultRecord]:
        pre_scan = ScanResultRecord.from_dict(result["pre_scan"])
        post_scan = ScanResultRecord.from_dict(result["post_scan"])
        self.session_workflow_service.append_scan_results(session_id, pre_scan, post_scan)
        return pre_scan, post_scan
