from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core import InferenceResultRecord, ScanResultRecord, SessionRecord
from app.services.inference_service import InferenceService
from app.services.session_workflow_service import SessionWorkflowService


@dataclass
class InferenceStartContext:
    session: SessionRecord
    output_dir: Path
    classes_yaml: str


@dataclass
class InferenceSelection:
    session: SessionRecord
    files: list[str]


class InferenceWorkflowService:
    """Coordinates inference session context, result selection, and result persistence."""

    def __init__(self, session_workflow_service: SessionWorkflowService):
        self.session_workflow_service = session_workflow_service

    def prepare_inference_start(
        self,
        current_session: SessionRecord | None,
        label: str,
        classes_yaml: str,
    ) -> InferenceStartContext:
        session = self.session_workflow_service.ensure_session(current_session, label.strip() or "session")
        return InferenceStartContext(
            session=session,
            output_dir=self.session_workflow_service.inference_dir(session.id),
            classes_yaml=classes_yaml.strip(),
        )

    def select_scan_result_for_inference(
        self,
        session: SessionRecord | None,
        scan_result: ScanResultRecord | None,
        inference_service: InferenceService,
    ) -> InferenceSelection | None:
        if session is None or scan_result is None:
            return None
        files = inference_service.scan_result_files(scan_result)
        if not files:
            return None
        return InferenceSelection(session=session, files=files)

    def persist_inference_result(
        self,
        session_id: str,
        pending_output_dir: Path | str,
        result_images: list[str],
        actual_output_dir: str,
    ) -> InferenceResultRecord:
        output_dir = str(pending_output_dir)
        record = InferenceResultRecord(
            output_dir=output_dir,
            images=result_images,
            actual_output_dir=actual_output_dir or "",
            raw={
                "output_dir": output_dir,
                "images": result_images,
                "actual_output_dir": actual_output_dir,
            },
        )
        self.session_workflow_service.append_inference_result(session_id, record)
        return record
