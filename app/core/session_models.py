from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScanResultRecord:
    label: str = "scan_result"
    saved: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ScanResultRecord":
        return cls(
            label=payload.get("label", "scan_result"),
            saved=list(payload.get("saved", [])),
            raw=dict(payload),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.raw)
        payload.setdefault("label", self.label)
        payload.setdefault("saved", self.saved)
        return payload


@dataclass
class InferenceResultRecord:
    output_dir: str = ""
    images: list[str] = field(default_factory=list)
    actual_output_dir: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "InferenceResultRecord":
        return cls(
            output_dir=str(payload.get("output_dir", "")),
            images=list(payload.get("images", [])),
            actual_output_dir=str(payload.get("actual_output_dir", "")),
            raw=dict(payload),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.raw)
        payload.setdefault("output_dir", self.output_dir)
        payload.setdefault("images", self.images)
        payload.setdefault("actual_output_dir", self.actual_output_dir)
        return payload


@dataclass
class TrainingResultRecord:
    mode: str = ""
    project_dir: str = ""
    output_dir: str = ""
    status: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainingResultRecord":
        return cls(
            mode=str(payload.get("mode", "")),
            project_dir=str(payload.get("project_dir", "")),
            output_dir=str(payload.get("output_dir", "")),
            status=str(payload.get("status", "")),
            raw=dict(payload),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.raw)
        payload.setdefault("mode", self.mode)
        payload.setdefault("project_dir", self.project_dir)
        payload.setdefault("output_dir", self.output_dir)
        payload.setdefault("status", self.status)
        return payload


@dataclass
class SessionRecord:
    id: str
    label: str
    created_at: str
    scan_results: list[ScanResultRecord] = field(default_factory=list)
    inference_results: list[InferenceResultRecord] = field(default_factory=list)
    training_results: list[TrainingResultRecord] = field(default_factory=list)
    path: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SessionRecord":
        return cls(
            id=str(payload["id"]),
            label=str(payload.get("label", "session")),
            created_at=str(payload.get("created_at", "")),
            scan_results=[ScanResultRecord.from_dict(item) for item in payload.get("scan_results", [])],
            inference_results=[
                InferenceResultRecord.from_dict(item) for item in payload.get("inference_results", [])
            ],
            training_results=[
                TrainingResultRecord.from_dict(item) for item in payload.get("training_results", [])
            ],
            path=str(payload.get("path", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "created_at": self.created_at,
            "scan_results": [item.to_dict() for item in self.scan_results],
            "inference_results": [item.to_dict() for item in self.inference_results],
            "training_results": [item.to_dict() for item in self.training_results],
            "path": self.path,
        }
