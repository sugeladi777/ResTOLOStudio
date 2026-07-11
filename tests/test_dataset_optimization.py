from __future__ import annotations

import io
import zipfile
from collections import Counter
from hashlib import sha256
from pathlib import Path

import numpy as np
from PIL import Image

from app.services.dataset_service import DatasetService
from app.services.training_job_service import TrainingJobService
from app.services.training_workflow_service import TrainingWorkflowService
from ml.molecule_preprocessing import crop_normalized_box, crop_xyxy_array
from ml.resnet_finetune import _reported_class_metrics


def _png_bytes(color: str = "white") -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (40, 30), color).save(stream, format="PNG")
    return stream.getvalue()


def test_annotation_archive_is_read_only_and_uses_ascii_workspace_names(tmp_path: Path):
    archive_path = tmp_path / "测试数据.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("数据/图片1.png", _png_bytes())
        archive.writestr("数据/图片2.png", _png_bytes("black"))
        archive.writestr("数据/标签/图片1.txt", "1 0.5 0.5 0.2 0.2\n")

    before = sha256(archive_path.read_bytes()).hexdigest()
    result = DatasetService().extract_annotation_archive(str(archive_path))

    assert sha256(archive_path.read_bytes()).hexdigest() == before == result.source_hash
    assert len(result.image_paths) == 2
    assert len(result.annotation_paths) == 1
    assert result.unmatched_images == 1
    assert all(Path(path).name.isascii() for path in result.image_paths + result.annotation_paths)


def test_yolo_split_is_deterministic_and_disjoint():
    images = [f"image_{index}.png" for index in range(10)]
    service = DatasetService()

    train, val = service.split_yolo_images(images)
    repeated_train, repeated_val = service.split_yolo_images(images)

    assert (train, val) == (repeated_train, repeated_val)
    assert len(train) == 8
    assert len(val) == 2
    assert set(train).isdisjoint(val)


def test_training_and_inference_use_identical_square_crop():
    array = np.arange(60 * 80 * 3, dtype=np.uint8).reshape(60, 80, 3)
    image = Image.fromarray(array)
    normalized = crop_normalized_box(image, x=0.5, y=0.5, width=0.25, height=0.2)
    xyxy = crop_xyxy_array(array, (30, 24, 50, 36))

    assert normalized.size == (224, 224)
    assert np.array_equal(np.asarray(normalized), np.asarray(xyxy))


def test_resnet_group_split_keeps_singleton_class_out_of_validation(tmp_path: Path):
    crop_dir = tmp_path / "crops"
    for class_name in ("1", "2"):
        (crop_dir / class_name).mkdir(parents=True)
    Image.new("RGB", (16, 16)).save(crop_dir / "1" / "crop_source_0000_crop_00000.jpg")
    for source_index in range(4):
        Image.new("RGB", (16, 16)).save(
            crop_dir / "2" / f"crop_source_{source_index:04d}_crop_{source_index:05d}.jpg"
        )

    workflow = TrainingWorkflowService(DatasetService(), TrainingJobService())
    train_dir, val_dir = workflow._split_resnet_crop_dataset(str(crop_dir), str(tmp_path / "split"), lambda _: None)

    train_singletons = list((Path(train_dir) / "1").glob("*.jpg"))
    val_singletons = list((Path(val_dir) / "1").glob("*.jpg"))
    train_groups = {path.name.split("_crop_", 1)[0] for path in Path(train_dir).rglob("*.jpg")}
    val_groups = {path.name.split("_crop_", 1)[0] for path in Path(val_dir).rglob("*.jpg")}
    assert len(train_singletons) == 1
    assert not val_singletons
    assert train_groups.isdisjoint(val_groups)


def test_resnet_metrics_preserve_original_class_indices_and_validation_status():
    metrics = _reported_class_metrics(
        {"0": {"support": 3, "recall": 1.0, "f1": 0.8}},
        ["molecule-a", "molecule-b"],
        [1, 8],
        Counter({0: 14, 1: 1}),
    )

    assert metrics["1"]["class_name"] == "molecule-a"
    assert metrics["1"]["validation_status"] == "verifiable"
    assert metrics["8"] == {
        "class_name": "molecule-b",
        "sample_count": 1,
        "validation_support": 0,
        "validation_status": "insufficient",
        "recall": None,
        "f1": None,
    }
