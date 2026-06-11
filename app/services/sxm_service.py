from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from hashlib import md5

from app.utils.sxm_parser import load_sxm_as_image


@dataclass
class SxmConversionResult:
    files: list[str]
    metadata: dict[str, dict]
    original_paths: dict[str, str]
    color_paths: dict[str, str]
    has_sxm_files: bool
    logs: list[str]


class SxmService:
    """Converts SXM files into reusable image assets and metadata."""

    def ensure_gray_image(self, file_path: str) -> str:
        gray_path, _, _, _ = self._ensure_image_assets(file_path)
        return gray_path

    def convert_files(self, files: list[str]) -> SxmConversionResult:
        result_files: list[str] = []
        metadata: dict[str, dict] = {}
        original_paths: dict[str, str] = {}
        color_paths: dict[str, str] = {}
        logs: list[str] = []
        has_sxm_files = False

        for file_path in files:
            if not file_path.lower().endswith(".sxm"):
                result_files.append(file_path)
                continue

            has_sxm_files = True
            try:
                gray_path, color_path, sxm, _ = self._ensure_image_assets(file_path)

                result_files.append(gray_path)
                original_paths[file_path] = gray_path
                color_paths[file_path] = color_path

                px_x, px_y = sxm.get_pixel_size_nm()
                metadata[gray_path] = {
                    "original_path": file_path,
                    "scan_range_nm": sxm.get_scan_range_nm(),
                    "pixel_size_nm": sxm.get_pixel_size_nm(),
                    "channels": sxm.get_channel_names(),
                    "summary": sxm.get_metadata_summary(),
                    "gray_path": gray_path,
                    "color_path": color_path,
                }
                logs.append(
                    "已转换 SXM 文件 "
                    f"{os.path.basename(file_path)} "
                    f"(扫描范围：{sxm.get_scan_range_nm()[0]:.2f}x{sxm.get_scan_range_nm()[1]:.2f} nm，"
                    f"像素尺寸：{px_x:.4f}x{px_y:.4f} nm/像素)"
                )
            except Exception as exc:  # noqa: BLE001
                logs.append(f"加载 SXM 文件失败 {os.path.basename(file_path)}：{exc}")

        return SxmConversionResult(
            files=result_files,
            metadata=metadata,
            original_paths=original_paths,
            color_paths=color_paths,
            has_sxm_files=has_sxm_files,
            logs=logs,
        )

    def _ensure_image_assets(self, file_path: str):
        img_gray, sxm = load_sxm_as_image(file_path, use_color=False)
        img_color, _ = load_sxm_as_image(file_path, use_color=True)
        if not img_gray:
            raise ValueError(f"Unable to decode SXM image: {file_path}")

        cache_dir = self._cache_dir_for(file_path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        basename = Path(file_path).stem
        gray_path = cache_dir / f"{basename}_gray.png"
        color_path = cache_dir / f"{basename}_color.png"
        if not gray_path.exists():
            img_gray.save(gray_path)
        if img_color and not color_path.exists():
            img_color.save(color_path)
        return str(gray_path), str(color_path), sxm, cache_dir

    def _cache_dir_for(self, file_path: str) -> Path:
        source = Path(file_path)
        fingerprint = md5(
            f"{source.resolve()}|{source.stat().st_mtime_ns}|{source.stat().st_size}".encode("utf-8")
        ).hexdigest()[:12]
        return Path(tempfile.gettempdir()) / "restolo_sxm_cache" / f"{source.stem}_{fingerprint}"
