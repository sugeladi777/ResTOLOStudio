from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.core import AnnotationState
from app.services.image_workflow_service import ImageWorkflowService


class DummyAnnotationService:
    def __init__(self) -> None:
        self.created_states: list[list[str]] = []
        self.remap_calls: list[tuple[AnnotationState, dict[str, str]]] = []

    def create_state(self, files: list[str]) -> AnnotationState:
        self.created_states.append(list(files))
        return AnnotationState(
            images=list(files),
            annotations={path: [] for path in files},
            class_names=["0"],
            current_index=0,
        )

    def remap_image_paths(self, state: AnnotationState, path_map: dict[str, str]) -> AnnotationState:
        self.remap_calls.append((state, dict(path_map)))
        return AnnotationState(
            images=[path_map.get(path, path) for path in state.images],
            annotations={path_map.get(path, path): boxes for path, boxes in state.annotations.items()},
            class_names=list(state.class_names),
            current_index=state.current_index,
        )


class DummySxmService:
    def __init__(self, conversion) -> None:
        self.conversion = conversion
        self.calls: list[list[str]] = []

    def convert_files(self, files: list[str]):
        self.calls.append(list(files))
        return self.conversion


def test_image_workflow_service_builds_annotation_and_inference_load_results():
    annotation_service = DummyAnnotationService()
    conversion = SimpleNamespace(
        files=["scan_gray.png", "plain.png"],
        metadata={"scan_gray.png": {"color_path": "scan_color.png"}},
        original_paths={"scan_gray.png": "scan.sxm"},
        color_paths={"scan_gray.png": "scan_color.png"},
        has_sxm_files=True,
        logs=["converted"],
    )
    workflow = ImageWorkflowService(annotation_service, DummySxmService(conversion))

    annotation_result = workflow.convert_for_annotation(["scan.sxm", "plain.png"])
    inference_result = workflow.convert_for_inference(["scan.sxm", "plain.png"])

    assert annotation_result.files == ["scan_gray.png", "plain.png"]
    assert annotation_result.gray_files == []
    assert annotation_result.annotation_state is not None
    assert inference_result.gray_files == ["scan_gray.png", "plain.png"]
    assert inference_result.annotation_state is not None
    assert annotation_service.created_states == [
        ["scan_gray.png", "plain.png"],
        ["scan_gray.png", "plain.png"],
    ]


def test_image_workflow_service_remaps_sxm_display_paths(tmp_path: Path):
    annotation_service = DummyAnnotationService()
    workflow = ImageWorkflowService(annotation_service, DummySxmService(None))
    color_path = tmp_path / "scan_color.png"
    color_path.write_text("color", encoding="utf-8")
    gray_path = tmp_path / "scan_gray.png"
    state = AnnotationState(
        images=[str(gray_path)],
        annotations={str(gray_path): []},
        class_names=["0"],
        current_index=0,
    )
    metadata = {str(gray_path): {"color_path": str(color_path)}}

    color_state = workflow.remap_sxm_display(state, metadata, use_color=True)
    gray_state = workflow.remap_sxm_display(color_state, metadata, use_color=False)

    assert color_state.images == [str(color_path)]
    assert gray_state.images == [str(gray_path)]
    assert workflow.gray_path_for(str(color_path), metadata) == str(gray_path)
    assert workflow.gray_path_for("plain.png", metadata) == "plain.png"
