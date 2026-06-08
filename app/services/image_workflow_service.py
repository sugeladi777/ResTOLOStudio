from __future__ import annotations

import os
from dataclasses import dataclass

from app.core import AnnotationState
from app.services.annotation_service import AnnotationService
from app.services.sxm_service import SxmConversionResult, SxmService


@dataclass
class ImageLoadResult:
    files: list[str]
    gray_files: list[str]
    annotation_state: AnnotationState | None
    sxm_metadata: dict[str, dict]
    sxm_original_paths: dict[str, str]
    sxm_color_paths: dict[str, str]
    has_sxm_files: bool
    logs: list[str]


class ImageWorkflowService:
    """Coordinates image loading across SXM conversion, annotation, and inference views."""

    def __init__(self, annotation_service: AnnotationService, sxm_service: SxmService):
        self.annotation_service = annotation_service
        self.sxm_service = sxm_service

    def convert_for_annotation(self, files: list[str]) -> ImageLoadResult:
        conversion = self.sxm_service.convert_files(files)
        return self._build_load_result(conversion, load_inference_images=False)

    def convert_for_inference(self, files: list[str]) -> ImageLoadResult:
        conversion = self.sxm_service.convert_files(files)
        return self._build_load_result(conversion, load_inference_images=True)

    def remap_sxm_display(self, state: AnnotationState, sxm_metadata: dict[str, dict], use_color: bool) -> AnnotationState:
        path_map: dict[str, str] = {}
        for gray_path, meta in sxm_metadata.items():
            color_path = meta.get("color_path", "")
            if use_color and color_path and os.path.exists(color_path):
                path_map[gray_path] = color_path
                path_map[color_path] = color_path
            else:
                path_map[gray_path] = gray_path
                path_map[color_path] = gray_path
        return self.annotation_service.remap_image_paths(state, path_map)

    def gray_path_for(self, image_path: str, sxm_metadata: dict[str, dict]) -> str:
        for gray_path, meta in sxm_metadata.items():
            if image_path == meta.get("color_path"):
                return gray_path
        return image_path

    def _build_load_result(self, conversion: SxmConversionResult, load_inference_images: bool) -> ImageLoadResult:
        files = list(conversion.files)
        gray_files = [self.gray_path_for(path, conversion.metadata) for path in files] if load_inference_images else []
        annotation_state = self.annotation_service.create_state(files) if files else None
        return ImageLoadResult(
            files=files,
            gray_files=gray_files,
            annotation_state=annotation_state,
            sxm_metadata=dict(conversion.metadata),
            sxm_original_paths=dict(conversion.original_paths),
            sxm_color_paths=dict(conversion.color_paths),
            has_sxm_files=conversion.has_sxm_files,
            logs=list(conversion.logs),
        )
