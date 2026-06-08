from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

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
                img_gray, sxm = load_sxm_as_image(file_path, use_color=False)
                img_color, _ = load_sxm_as_image(file_path, use_color=True)
                if not img_gray:
                    continue

                tmp_dir = tempfile.mkdtemp(prefix="restolo_sxm_")
                basename = Path(file_path).stem
                gray_path = os.path.join(tmp_dir, f"{basename}_gray.png")
                color_path = os.path.join(tmp_dir, f"{basename}_color.png")
                img_gray.save(gray_path)
                if img_color:
                    img_color.save(color_path)

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
