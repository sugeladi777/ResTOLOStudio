from datetime import datetime
import time
import importlib
import inspect
import json
from pathlib import Path
import struct
from contextlib import contextmanager
import zlib

import numpy as np
from PIL import Image

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = None
    plt = None

from nanonis import nanonisTCP
from nanonis.Bias import Bias
from nanonis.Current import Current
from nanonis.Scan import Scan
from nanonis.Signals import Signals
from nanonis.ZController import ZController


MODULE_CLASS_NAMES = {
    "AutoApproach": ("nanonis.AutoApproach", "AutoApproach"),
    "Bias": ("nanonis.Bias", "Bias"),
    "BiasSpectr": ("nanonis.BiasSpectr", "BiasSpectr"),
    "Current": ("nanonis.Current", "Current"),
    "FolMe": ("nanonis.FolMe", "FolMe"),
    "LockIn": ("nanonis.LockIn", "LockIn"),
    "Marks": ("nanonis.Marks", "Marks"),
    "Motor": ("nanonis.Motor", "Motor"),
    "Osci2T": ("nanonis.Osci2T", "Osci2T"),
    "Pattern": ("nanonis.Pattern", "Pattern"),
    "Piezo": ("nanonis.Piezo", "Piezo"),
    "SafeTip": ("nanonis.SafeTip", "SafeTip"),
    "Scan": ("nanonis.Scan", "Scan"),
    "Signals": ("nanonis.Signals", "Signals"),
    "TipRec": ("nanonis.TipRec", "TipRec"),
    "TipShaper": ("nanonis.TipShaper", "TipShaper"),
    "UserOut": ("nanonis.UserOut", "UserOut"),
    "Util": ("nanonis.Util", "Util"),
    "ZController": ("nanonis.ZController", "ZController"),
}


class NanonisExperimentClient:
    def __init__(
        self,
        ip="127.0.0.1",
        port=6501,
        version=10380,
        output_dir="scan_output",
        max_buf_size=1024 * 1024,
        connect_timeout=5.0,
        response_timeout=10.0,
    ):
        self.ip = ip
        self.port = port
        self.version = version
        self.output_dir = Path(output_dir)
        self.max_buf_size = max_buf_size
        self.connect_timeout = connect_timeout
        self.response_timeout = response_timeout

        self.ntcp = None
        self.bias = None
        self.current = None
        self.scan = None
        self.signals = None
        self.zctrl = None
        self._module_instances = {}

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def connect(self):
        self.ntcp = nanonisTCP(
            self.ip,
            self.port,
            version=self.version,
            max_buf_size=self.max_buf_size,
            connect_timeout=self.connect_timeout,
            response_timeout=self.response_timeout,
        )
        self.bias = Bias(self.ntcp)
        self.current = Current(self.ntcp)
        self.scan = Scan(self.ntcp)
        self.signals = Signals(self.ntcp)
        self.zctrl = ZController(self.ntcp)
        self._module_instances = {
            "Bias": self.bias,
            "Current": self.current,
            "Scan": self.scan,
            "Signals": self.signals,
            "ZController": self.zctrl,
        }
        return self

    def connection_summary(self):
        return {
            "ip": self.ip,
            "port": self.port,
            "version": self.version,
            "max_buf_size": self.max_buf_size,
            "connect_timeout": self.connect_timeout,
            "response_timeout": self.response_timeout,
            "signal_lookup_mode": "signal_names" if self.version >= 11798 else "input_slots",
            "available_modules": list(self.available_modules()),
        }

    def close(self):
        if self.ntcp is not None:
            self.ntcp.close_connection()
        self.ntcp = None
        self.bias = None
        self.current = None
        self.scan = None
        self.signals = None
        self.zctrl = None
        self._module_instances = {}

    def read_status(self):
        return {
            "bias_v": self.bias.Get(),
            "current_a": self.current.Get(),
            "z_status": self.zctrl.StatusGet(),
            "z_feedback": self.zctrl.OnOffGet(),
            "z_setpoint_a": self.zctrl.SetpntGet(),
            "scan_frame": self.scan.FrameGet(),
            "scan_buffer": self.scan.BufferGet(),
        }

    def signal_names(self):
        return self.signals.NamesGet()

    def input_slots(self):
        names, signal_indexes = self.signals.InSlotsGet()
        return [
            {"slot": slot, "signal_index": signal_index, "name": name}
            for slot, (name, signal_index) in enumerate(zip(names, signal_indexes))
        ]

    def _normalize_signal_name(self, signal_name):
        return "".join(ch.lower() for ch in str(signal_name) if ch.isalnum())

    def _signal_aliases(self, signal_name):
        normalized = self._normalize_signal_name(signal_name)
        aliases = {normalized}
        alias_map = {
            "z": {"z", "zm", "zmeter", "zposition", "zpos"},
            "current": {"current", "curr", "currenta", "tunnelcurrent"},
        }
        aliases.update(alias_map.get(normalized, set()))
        return aliases

    def _match_signal_index_from_slots(self, signal_name, slots):
        aliases = self._signal_aliases(signal_name)
        for slot in slots:
            slot_name = self._normalize_signal_name(slot.get("name", ""))
            if slot_name in aliases:
                return slot["slot"]
        for slot in slots:
            slot_name = self._normalize_signal_name(slot.get("name", ""))
            if any(alias and alias in slot_name for alias in aliases):
                return slot["slot"]
        return None

    def _match_signal_index_from_names(self, signal_name, signal_names):
        aliases = self._signal_aliases(signal_name)
        for index, name in enumerate(signal_names):
            normalized = self._normalize_signal_name(name)
            if normalized in aliases:
                return index
        for index, name in enumerate(signal_names):
            normalized = self._normalize_signal_name(name)
            if any(alias and alias in normalized for alias in aliases):
                return index
        return None

    def scan_channel_indexes_for_signal_names(self, signal_names):
        if self.version < 11798:
            try:
                slots = self.input_slots()
            except TimeoutError:
                slots = None

            if slots:
                channel_indexes = []
                for signal_name in signal_names:
                    match = self._match_signal_index_from_slots(signal_name, slots)
                    if match is None:
                        raise ValueError(f"Signal is not assigned to an input slot: {signal_name}")
                    channel_indexes.append(match)
                return tuple(channel_indexes)

        available_names = self.signal_names()
        channel_indexes = []
        for signal_name in signal_names:
            match = self._match_signal_index_from_names(signal_name, available_names)
            if match is None:
                raise ValueError(f"Signal is not available: {signal_name}")
            channel_indexes.append(match)
        return tuple(channel_indexes)

    def available_modules(self):
        return tuple(MODULE_CLASS_NAMES.keys())

    def get_module(self, module_name):
        if module_name not in MODULE_CLASS_NAMES:
            raise KeyError(f"Unknown module: {module_name}")

        if module_name not in self._module_instances:
            module_path, class_name = MODULE_CLASS_NAMES[module_name]
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            self._module_instances[module_name] = cls(self.ntcp)

        return self._module_instances[module_name]

    def module_methods(self, module_name):
        module = self.get_module(module_name)
        methods = []
        for name, member in inspect.getmembers(module, predicate=callable):
            if name.startswith("_"):
                continue
            methods.append(name)
        return tuple(methods)

    def module_doc(self, module_name):
        module = self.get_module(module_name)
        return inspect.getdoc(module) or ""

    def method_signature(self, module_name, method_name):
        module = self.get_module(module_name)
        method = getattr(module, method_name)
        return inspect.signature(method)

    def method_doc(self, module_name, method_name):
        module = self.get_module(module_name)
        method = getattr(module, method_name)
        return inspect.getdoc(method) or ""

    def execute_module_method(self, module_name, method_name, kwargs):
        module = self.get_module(module_name)
        method = getattr(module, method_name)
        timeout_ms = kwargs.get("timeout") if isinstance(kwargs, dict) else None
        if isinstance(timeout_ms, (int, float)) and timeout_ms > 0:
            with self.temporary_response_timeout(timeout_ms / 1000 + 10):
                return method(**kwargs)
        return method(**kwargs)

    @contextmanager
    def temporary_response_timeout(self, timeout_s):
        if self.ntcp is None:
            yield
            return
        previous_timeout = self.ntcp.get_response_timeout()
        self.ntcp.set_response_timeout(timeout_s)
        try:
            yield
        finally:
            self.ntcp.set_response_timeout(previous_timeout)

    def set_feedback(self, enabled):
        self.zctrl.OnOffSet(1 if enabled else 0)

    def set_bias(self, voltage):
        self.bias.Set(voltage)

    def set_setpoint(self, setpoint_a):
        self.zctrl.SetpntSet(setpoint_a)

    def set_scan_frame_nm(self, width_nm, height_nm=None, center_x_nm=0, center_y_nm=0, angle_deg=None):
        if height_nm is None:
            height_nm = width_nm
        if angle_deg is None:
            current_frame = self.scan.FrameGet()
            if len(current_frame) >= 5:
                angle_deg = current_frame[4]
            else:
                angle_deg = 0
        self.scan.FrameSet(
            center_x_nm * 1e-9,
            center_y_nm * 1e-9,
            width_nm * 1e-9,
            height_nm * 1e-9,
            angle_deg,
        )
        return self.scan.FrameGet()

    def set_scan_buffer(self, channel_indexes=(30, 0), pixels=128, lines=None):
        if lines is None:
            lines = pixels
        self.scan.BufferSet(channel_indexes=list(channel_indexes), pixels=pixels, lines=lines)
        return self.scan.BufferGet()

    def configure_autosave(self, series_name="python_scan_%y%m%d_%H-%M-%S", comment=""):
        self.scan.PropsSet(autosave=1, series_name=series_name, comment=comment)

    def scan_and_save(
        self,
        label,
        width_nm=None,
        height_nm=None,
        channel_indexes=(30, 0),
        pixels=128,
        lines=None,
        direction=1,
        timeout_ms=300_000,
        center_x_nm=0,
        center_y_nm=0,
        angle_deg=None,
        start_poll_timeout_ms=8000,
        start_poll_interval_ms=250,
    ):
        if width_nm is not None:
            self.set_scan_frame_nm(
                width_nm=width_nm,
                height_nm=height_nm,
                center_x_nm=center_x_nm,
                center_y_nm=center_y_nm,
                angle_deg=angle_deg,
            )

        self.set_scan_buffer(channel_indexes=channel_indexes, pixels=pixels, lines=lines)
        self.configure_autosave(
            series_name=f"{label}_%y%m%d_%H-%M-%S",
            comment=f"Python scan: {label}",
        )

        self.scan.Start()
        self._wait_for_scan_start(
            timeout_ms=start_poll_timeout_ms,
            poll_interval_ms=start_poll_interval_ms,
        )
        with self.temporary_response_timeout(timeout_ms / 1000 + 10):
            timeout, _, nanonis_file_path = self.scan.WaitEndOfScan(timeout=timeout_ms)
        if timeout:
            self.scan.Stop()
            raise TimeoutError(f"Scan timed out after {timeout_ms} ms")

        saved = []
        for channel_index in channel_indexes:
            channel_name, data, scan_direction = self.scan.FrameDataGrab(channel_index, direction)
            if getattr(data, "size", 0) == 0:
                continue
            base_path = self.save_frame_data(
                label=label,
                channel_name=channel_name,
                data=data,
                scan_direction=scan_direction,
            )
            saved.append(
                {
                    "channel_index": channel_index,
                    "channel_name": channel_name,
                    "base_path": str(base_path),
                    "npy": str(base_path.with_suffix(".npy")),
                    "csv": str(base_path.with_suffix(".csv")),
                    "png": str(base_path.with_name(base_path.name + "_model.png")),
                    "preview_png": str(base_path.with_suffix(".png")),
                    "model_png": str(base_path.with_name(base_path.name + "_model.png")),
                    "preprocessing": {
                        "normalization": "percentile_1_99",
                        "colormap": "inferno",
                        "geometry": "raw_pixels",
                    },
                }
            )

        result = {
            "label": label,
            "nanonis_file_path": nanonis_file_path,
            "scan_frame": self.scan.FrameGet(),
            "scan_buffer": self.scan.BufferGet(),
            "saved": saved,
        }
        result["manifest_path"] = str(self.save_scan_manifest(result))
        return result

    def _wait_for_scan_start(self, timeout_ms=8000, poll_interval_ms=250):
        if self.version < 14000:
            # Older RT Engine versions do not expose Scan.StatusGet reliably.
            # Fall back to a short grace period so Start has time to take effect.
            time.sleep(max(poll_interval_ms, 200) / 1000)
            return

        deadline = time.monotonic() + max(timeout_ms, poll_interval_ms) / 1000
        last_state = None
        while time.monotonic() < deadline:
            try:
                last_state = self.scan.StatusGet()
            except Exception:
                # If status polling is not accepted by the controller, do not
                # block the workflow here; WaitEndOfScan will surface the real error.
                time.sleep(poll_interval_ms / 1000)
                return
            if last_state:
                return
            time.sleep(poll_interval_ms / 1000)

        raise TimeoutError("Scan start was acknowledged, but the scan did not enter running state in time")

    def save_frame_data(self, label, channel_name, data, scan_direction):
        self.output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_label = _safe_name(label)
        safe_channel_name = _safe_name(channel_name)
        base = self.output_dir / f"{timestamp}_{safe_label}_{safe_channel_name}_{scan_direction}"

        np.save(base.with_suffix(".npy"), data)
        np.savetxt(base.with_suffix(".csv"), data, delimiter=",")

        png_path = base.with_suffix(".png")
        if plt is not None:
            _write_matplotlib_png(png_path, data, channel_name)
        else:
            write_grayscale_png(png_path, data)
        write_model_png(base.with_name(base.name + "_model.png"), data)
        return base

    def save_scan_manifest(self, result):
        self.output_dir.mkdir(exist_ok=True)
        if result["saved"]:
            first_base = Path(result["saved"][0]["base_path"])
            manifest_path = first_base.with_name(first_base.name + "_manifest.json")
        else:
            manifest_path = self.output_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{_safe_name(result['label'])}_manifest.json"

        manifest = {
            "label": result["label"],
            "nanonis_file_path": result.get("nanonis_file_path"),
            "scan_frame": result.get("scan_frame"),
            "scan_buffer": result.get("scan_buffer"),
            "saved": result.get("saved", []),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest_path


def _safe_name(value):
    return str(value).replace("/", "_").replace("\\", "_").replace(" ", "_").replace(":", "_")


def _write_matplotlib_png(path, data, channel_name):
    plt.figure(figsize=(6, 5))
    plt.imshow(data, origin="lower", cmap="viridis", aspect="auto")
    plt.colorbar(label=channel_name)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def write_grayscale_png(path, data):
    finite_data = np.asarray(data, dtype=float)
    finite_data = np.nan_to_num(finite_data, nan=0.0, posinf=0.0, neginf=0.0)

    low, high = np.percentile(finite_data, [1, 99])
    if high <= low:
        low = float(finite_data.min())
        high = float(finite_data.max())

    if high > low:
        image = np.clip((finite_data - low) / (high - low), 0, 1)
    else:
        image = np.zeros_like(finite_data)

    image = np.flipud((image * 255).astype(np.uint8))
    height, width = image.shape
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))

    def chunk(chunk_type, payload):
        return (
            struct.pack(">I", len(payload))
            + chunk_type
            + payload
            + struct.pack(">I", zlib.crc32(chunk_type + payload) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def write_model_png(path, data):
    finite_data = np.asarray(data, dtype=float)
    finite_data = np.nan_to_num(finite_data, nan=0.0, posinf=0.0, neginf=0.0)
    low, high = np.percentile(finite_data, [1, 99])
    if high <= low:
        low = float(finite_data.min())
        high = float(finite_data.max())
    if high > low:
        normalized = np.clip((finite_data - low) / (high - low), 0, 1)
    else:
        normalized = np.zeros_like(finite_data)
    normalized = np.flipud(normalized)

    if matplotlib is not None:
        colored = matplotlib.colormaps["inferno"](normalized, bytes=True)[..., :3]
        Image.fromarray(colored, mode="RGB").save(path)
    else:
        Image.fromarray((normalized * 255).astype(np.uint8), mode="L").convert("RGB").save(path)
