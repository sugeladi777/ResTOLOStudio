from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

from app.core import ScanResultRecord
from app.services.acquisition_workflow_service import AcquisitionWorkflowService
from app.services.result_store_service import ResultStoreService
from app.services.session_workflow_service import SessionWorkflowService


class DummyNanonisService:
    def __init__(self) -> None:
        self.output_root = None
        self.set_output_root_calls = []

    def set_output_root(self, output_root) -> None:
        self.output_root = output_root
        self.set_output_root_calls.append(output_root)


class DummyConnectedNanonisService:
    def __init__(self) -> None:
        self.frame_calls = []
        self.buffer_calls = []
        self.read_status_calls = 0
        self.timeout_values = []
        self.scan = type("DummyScan", (), {"BufferGet": lambda self: [2, [0, 14], 256, 256]})()

    def _require(self):
        return self

    def temporary_response_timeout(self, timeout_s):
        self.timeout_values.append(timeout_s)
        return nullcontext()

    def connection_summary(self):
        return {"ip": "127.0.0.1", "port": 6501, "version": 10380}

    def set_scan_frame_nm(self, *args):
        self.frame_calls.append(args)

    def scan_channel_indexes_for_signal_names(self, channels):
        raise TimeoutError("timeout")

    def set_scan_buffer(self, channel_indexes, pixels, lines):
        self.buffer_calls.append((tuple(channel_indexes), pixels, lines))

    def read_status(self):
        self.read_status_calls += 1
        return {"ok": True}


class DummyConnectedNanonisServiceWithStatusTimeout(DummyConnectedNanonisService):
    def read_status(self):
        raise TimeoutError("timeout")


def test_acquisition_workflow_service_builds_config_and_geometry(tmp_path: Path):
    workflow = AcquisitionWorkflowService(SessionWorkflowService(ResultStoreService(tmp_path / "sessions")))

    config = workflow.build_connection_config(" 127.0.0.1 ", "6501", "10380")
    geometry = workflow.build_scan_geometry("12.5", "8.0", "256", "Z, Current")

    assert config.ip == "127.0.0.1"
    assert config.port == 6501
    assert config.version == 10380
    assert geometry.width_nm == 12.5
    assert geometry.height_nm == 8.0
    assert geometry.pixels == 256
    assert geometry.channels == ["Z", "Current"]


def test_acquisition_workflow_service_starts_session_and_persists_scan_results(tmp_path: Path):
    session_workflow = SessionWorkflowService(ResultStoreService(tmp_path / "sessions"))
    workflow = AcquisitionWorkflowService(session_workflow)
    nanonis_service = DummyNanonisService()

    session = workflow.start_scan_session(None, "scan-demo", nanonis_service)
    record = workflow.append_scan_result(session.id, {"label": "scan_a", "saved": []})
    pre, post = workflow.append_scan_workflow_results(
        session.id,
        {
            "pre_scan": {"label": "pre", "saved": []},
            "post_scan": {"label": "post", "saved": []},
        },
    )

    assert session.label == "scan-demo"
    assert nanonis_service.output_root == session_workflow.scan_dir(session.id)
    assert isinstance(record, ScanResultRecord)
    loaded = session_workflow.list_sessions()[0]
    assert [item.label for item in loaded.scan_results] == ["scan_a", "pre", "post"]
    assert pre.label == "pre"
    assert post.label == "post"
    assert nanonis_service.set_output_root_calls == [session_workflow.scan_dir(session.id)]


def test_nanonis_service_apply_scan_falls_back_to_current_buffer_channels_on_timeout(tmp_path: Path):
    from nanonis.services import NanonisSessionService

    service = DummyConnectedNanonisService()
    session = NanonisSessionService(tmp_path)
    session.client = service

    result = session.apply_scan(
        width_nm=5.0,
        height_nm=5.0,
        center_x_nm=0.0,
        center_y_nm=0.0,
        angle_deg=0.0,
        pixels=256,
        channels=["Z", "Current"],
    )

    assert service.buffer_calls == [((0, 14), 256, 256)]
    assert result["status"] == {"ok": True}


def test_nanonis_service_connect_returns_fallback_status_when_status_read_times_out(tmp_path: Path):
    from nanonis.services import NanonisSessionService

    service = DummyConnectedNanonisServiceWithStatusTimeout()
    session = NanonisSessionService(tmp_path)
    session.client = service

    result = session.status()

    assert result["status"] == "状态读取超时，但连接已建立。"


def test_nanonis_service_apply_scan_returns_fallback_status_when_status_read_times_out(tmp_path: Path):
    from nanonis.services import NanonisSessionService

    service = DummyConnectedNanonisServiceWithStatusTimeout()
    session = NanonisSessionService(tmp_path)
    session.client = service

    result = session.apply_scan(
        width_nm=5.0,
        height_nm=5.0,
        center_x_nm=0.0,
        center_y_nm=0.0,
        angle_deg=0.0,
        pixels=256,
        channels=["Z", "Current"],
    )

    assert service.buffer_calls == [((0, 14), 256, 256)]
    assert result["status"] == "扫描参数已应用，但状态读取超时。"
