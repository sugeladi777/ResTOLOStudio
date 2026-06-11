from __future__ import annotations

from pathlib import Path

import numpy as np

from nanonis.ExperimentClient import NanonisExperimentClient


def test_experiment_client_matches_common_signal_aliases_for_old_versions():
    client = NanonisExperimentClient(version=10380)
    client.input_slots = lambda: [
        {"slot": 0, "signal_index": 14, "name": "Current (A)"},
        {"slot": 1, "signal_index": 30, "name": "Z (m)"},
    ]

    assert client.scan_channel_indexes_for_signal_names(["Z", "Current"]) == (1, 0)


def test_experiment_client_falls_back_to_signal_names_when_old_slot_query_times_out():
    client = NanonisExperimentClient(version=10380)

    def _timeout():
        raise TimeoutError("timeout")

    client.input_slots = _timeout
    client.signal_names = lambda: ["Bias", "Current (A)", "Z (m)"]

    assert client.scan_channel_indexes_for_signal_names(["Z", "Current"]) == (2, 1)


def test_experiment_client_scan_and_save_skips_empty_channel_results(tmp_path: Path):
    client = NanonisExperimentClient(version=10380, output_dir=tmp_path)
    client.temporary_response_timeout = lambda timeout_s: __import__("contextlib").nullcontext()
    client.set_scan_frame_nm = lambda **kwargs: None
    client.set_scan_buffer = lambda **kwargs: None
    client.configure_autosave = lambda **kwargs: None
    client._wait_for_scan_start = lambda **kwargs: None
    client.save_scan_manifest = lambda result: tmp_path / "manifest.json"

    class DummyScan:
        def Start(self):
            return None

        def WaitEndOfScan(self, timeout):
            return False, 0, "demo.sxm"

        def FrameDataGrab(self, channel_index, direction):
            if channel_index == 30:
                return "", np.zeros((0, 0)), "up"
            return "Current (A)", np.ones((4, 4)), "up"

        def FrameGet(self):
            return [0, 0, 1, 1, 0]

        def BufferGet(self):
            return [1, [0], 4, 4]

    client.scan = DummyScan()

    result = client.scan_and_save(label="demo", channel_indexes=(30, 0), pixels=4, width_nm=None)

    assert len(result["saved"]) == 1
    assert result["saved"][0]["channel_index"] == 0
