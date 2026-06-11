from __future__ import annotations

from pathlib import Path

from PIL import Image

from app.core import AnnotationBox, AnnotationState, ScanResultRecord
from app.services.annotation_service import AnnotationService
from app.services.dataset_service import DatasetService
from app.services.inference_service import InferenceService


class DummyInferenceManager:
    def load_images(self, images):
        self.images = list(images)

    def run_inference(self, yolo_model, resnet_model, output_dir, classes_yaml=""):
        self.last_call = (yolo_model, resnet_model, output_dir, classes_yaml)


def _make_image(path: Path, size: tuple[int, int] = (120, 120)) -> str:
    Image.new("RGB", size, "white").save(path)
    return str(path)


def test_file_pipeline_annotation_round_trip_and_dataset_outputs(tmp_path: Path):
    image_path = _make_image(tmp_path / "sample.png")
    annotation_service = AnnotationService()
    dataset_service = DatasetService()
    state = AnnotationState(
        images=[image_path],
        annotations={
            image_path: [
                AnnotationBox(cls=0, x=0.5, y=0.5, w=0.4, h=0.4),
            ]
        },
        class_names=["atom"],
        current_index=0,
    )

    saved_dir = tmp_path / "saved"
    output_dir, used_class_names = annotation_service.save_annotations(state, str(saved_dir))
    assert output_dir == saved_dir
    assert used_class_names == ["atom"]
    assert (saved_dir / "sample.txt").exists()
    assert (saved_dir / "classes.yaml").exists()

    loaded_state = annotation_service.load_annotation_files(
        annotation_service.create_state([image_path], class_names=["atom"]),
        [str(saved_dir / "sample.txt")],
    )
    assert len(loaded_state.annotations[image_path]) == 1
    assert loaded_state.annotations[image_path][0].to_tuple() == (0, 0.5, 0.5, 0.4, 0.4)

    images_dir = tmp_path / "yolo" / "images" / "train"
    labels_dir = tmp_path / "yolo" / "labels" / "train"
    dataset_service.write_yolo_split(
        loaded_state,
        lambda path: path,
        [image_path],
        str(images_dir),
        str(labels_dir),
    )
    assert (images_dir / "sample.png").exists()
    assert (labels_dir / "sample.txt").read_text(encoding="utf-8").strip() == "0 0.5 0.5 0.4 0.4"

    crop_summary = dataset_service.crop_resnet_dataset(
        loaded_state,
        lambda path: path,
        str(tmp_path / "resnet"),
        {0: "atom"},
    )
    assert crop_summary.crop_count == 1
    assert crop_summary.actual_classes == {"atom"}
    assert crop_summary.class_counts == {"atom": 1}
    crop_files = list((tmp_path / "resnet" / "atom").glob("crop_*.jpg"))
    assert len(crop_files) == 1

    restored_state = annotation_service.load_saved_annotation_state(str(saved_dir))
    assert restored_state is not None
    assert restored_state.class_names == ["atom"]
    assert len(restored_state.images) == 1
    restored_image_path = restored_state.images[0]
    assert Path(restored_image_path).parent == saved_dir
    assert restored_state.annotations[restored_image_path][0].to_tuple() == (0, 0.5, 0.5, 0.4, 0.4)


def test_crop_resnet_dataset_clears_old_output_and_keeps_per_class_counts(tmp_path: Path):
    image_path = _make_image(tmp_path / "sample.png")
    dataset_service = DatasetService()
    state = AnnotationState(
        images=[image_path],
        annotations={
            image_path: [
                AnnotationBox(cls=0, x=0.3, y=0.3, w=0.2, h=0.2),
                AnnotationBox(cls=1, x=0.7, y=0.7, w=0.2, h=0.2),
            ]
        },
        class_names=["0", "1"],
        current_index=0,
    )
    output_dir = tmp_path / "resnet"
    stale_dir = output_dir / "stale"
    stale_dir.mkdir(parents=True, exist_ok=True)
    (stale_dir / "old.jpg").write_text("old", encoding="utf-8")

    crop_summary = dataset_service.crop_resnet_dataset(
        state,
        lambda path: path,
        str(output_dir),
        {},
    )

    assert crop_summary.crop_count == 2
    assert crop_summary.actual_classes == {"0", "1"}
    assert crop_summary.class_counts == {"0": 1, "1": 1}
    assert not stale_dir.exists()
    assert len(list((output_dir / "0").glob("crop_*.jpg"))) == 1
    assert len(list((output_dir / "1").glob("crop_*.jpg"))) == 1


def test_file_pipeline_scan_result_selection_uses_real_png_path(tmp_path: Path):
    image_path = _make_image(tmp_path / "scan_output.png")
    inference_service = InferenceService(DummyInferenceManager())
    scan_result = ScanResultRecord(
        label="scan",
        saved=[{"png": image_path}, {"png": ""}, {}],
    )

    files = inference_service.scan_result_files(scan_result)

    assert files == [image_path]


def test_file_pipeline_scan_result_selection_supports_sxm_path(tmp_path: Path):
    sxm_path = tmp_path / "scan.sxm"
    sxm_path.write_text("stub", encoding="utf-8")
    inference_service = InferenceService(DummyInferenceManager())
    scan_result = ScanResultRecord(
        label="scan",
        raw={"nanonis_file_path": str(sxm_path)},
    )

    files = inference_service.scan_result_files(scan_result)

    assert files == [str(sxm_path)]


def test_write_yolo_split_remaps_sparse_class_ids(tmp_path: Path):
    image_path = _make_image(tmp_path / "sample.png")
    dataset_service = DatasetService()
    state = AnnotationState(
        images=[image_path],
        annotations={
            image_path: [
                AnnotationBox(cls=1, x=0.5, y=0.5, w=0.4, h=0.4),
            ]
        },
        class_names=["background", "atom"],
        current_index=0,
    )

    class_index_map, class_names = dataset_service.remap_class_indices(state)
    images_dir = tmp_path / "yolo" / "images" / "train"
    labels_dir = tmp_path / "yolo" / "labels" / "train"
    dataset_service.write_yolo_split(
        state,
        lambda path: path,
        [image_path],
        str(images_dir),
        str(labels_dir),
        class_index_map=class_index_map,
    )

    assert class_index_map == {1: 0}
    assert class_names == ["atom"]
    assert (labels_dir / "sample.txt").read_text(encoding="utf-8").strip() == "0 0.5 0.5 0.4 0.4"
