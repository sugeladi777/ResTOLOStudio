from __future__ import annotations

import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath

from app.core import AnnotationState
from PIL import Image

from ml.molecule_preprocessing import crop_normalized_box


@dataclass
class AnnotationClassSummary:
    used_classes: set[int]
    class_names: list[str]


@dataclass
class ResnetCropSummary:
    crop_count: int
    actual_classes: set[str]
    class_counts: dict[str, int]
    skipped_missing_class: int = 0
    skipped_invalid_box: int = 0
    skipped_image_error: int = 0


@dataclass
class ArchiveDatasetResult:
    image_paths: list[str]
    annotation_paths: list[str]
    workspace: str
    source_hash: str
    unmatched_images: int


class DatasetService:
    """Reusable dataset preparation helpers for annotation and training flows."""

    def _state(self, annotation_source) -> AnnotationState:
        if isinstance(annotation_source, AnnotationState):
            return annotation_source.normalized()
        if hasattr(annotation_source, "export_state"):
            return annotation_source.export_state().normalized()
        return AnnotationState(
            images=list(getattr(annotation_source, "images", [])),
            annotations={
                path: list(getattr(annotation_source, "annotations", {}).get(path, []))
                for path in getattr(annotation_source, "images", [])
            },
            class_names=list(getattr(annotation_source, "class_names", [])),
            current_index=int(getattr(annotation_source, "current_index", 0)),
        ).normalized()

    def extract_annotation_archive(self, archive_path: str) -> ArchiveDatasetResult:
        source = Path(archive_path)
        source_hash = sha256(source.read_bytes()).hexdigest()
        workspace = Path(tempfile.mkdtemp(prefix="restolo_dataset_"))
        images_dir = workspace / "images"
        labels_dir = workspace / "labels"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)

        with zipfile.ZipFile(source) as archive:
            members = [name for name in archive.namelist() if not name.endswith("/")]
            label_members = {
                PurePosixPath(name).stem: name
                for name in members
                if PurePosixPath(name).suffix.lower() == ".txt"
            }
            image_members = [
                name
                for name in members
                if PurePosixPath(name).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            ]

            def image_sort_key(name: str):
                stem = PurePosixPath(name).stem
                numbers = tuple(int(value) for value in re.findall(r"\d+", stem))
                return (0 if stem in label_members else 1, numbers, stem)

            image_members.sort(key=image_sort_key)

            image_paths: list[str] = []
            annotation_paths: list[str] = []
            for index, member in enumerate(image_members):
                extension = PurePosixPath(member).suffix.lower()
                ascii_stem = f"image_{index:04d}"
                image_path = images_dir / f"{ascii_stem}{extension}"
                image_path.write_bytes(archive.read(member))
                image_paths.append(str(image_path))

                label_member = label_members.get(PurePosixPath(member).stem)
                if label_member:
                    label_path = labels_dir / f"{ascii_stem}.txt"
                    label_path.write_bytes(archive.read(label_member))
                    annotation_paths.append(str(label_path))

        if source_hash != sha256(source.read_bytes()).hexdigest():
            raise RuntimeError("数据压缩包在读取期间发生变化")
        return ArchiveDatasetResult(
            image_paths=image_paths,
            annotation_paths=annotation_paths,
            workspace=str(workspace),
            source_hash=source_hash,
            unmatched_images=len(image_paths) - len(annotation_paths),
        )

    def annotation_class_summary(self, annotation_source) -> AnnotationClassSummary:
        state = self._state(annotation_source)
        used_classes: set[int] = set()
        for boxes in state.annotations.values():
            for box in boxes:
                used_classes.add(box.cls if hasattr(box, "cls") else box[0])

        if not used_classes:
            return AnnotationClassSummary(used_classes=set(), class_names=[])

        max_class = max(used_classes)
        class_names = []
        for index in range(max_class + 1):
            if index < len(state.class_names):
                class_names.append(state.class_names[index])
            else:
                class_names.append(str(index))
        return AnnotationClassSummary(used_classes=used_classes, class_names=class_names)

    def annotated_images(self, annotation_source) -> list[str]:
        state = self._state(annotation_source)
        images = []
        for image_path in state.images:
            if image_path in state.annotations and state.annotations[image_path]:
                images.append(image_path)
        return images

    def selected_class_indices(self, annotation_source, class_checkboxes, class_indices) -> list[int]:
        if class_checkboxes and class_indices:
            selected = [
                class_idx
                for checkbox, class_idx in zip(class_checkboxes, class_indices)
                if checkbox.isChecked()
            ]
            if selected:
                return selected

        used_classes = self.annotation_class_summary(annotation_source).used_classes
        return sorted(used_classes)

    def class_names_for_indices(self, annotation_source, class_indices: list[int]) -> list[str]:
        state = self._state(annotation_source)
        class_names = []
        for idx in class_indices:
            if idx < len(state.class_names):
                class_names.append(state.class_names[idx])
            else:
                class_names.append(str(idx))
        return class_names

    def remap_class_indices(
        self,
        annotation_source,
        class_indices: list[int] | None = None,
    ) -> tuple[dict[int, int], list[str]]:
        state = self._state(annotation_source)
        if class_indices is None:
            class_indices = sorted(self.annotation_class_summary(state).used_classes)

        normalized_indices = sorted(dict.fromkeys(int(idx) for idx in class_indices))
        class_map = {old_idx: new_idx for new_idx, old_idx in enumerate(normalized_indices)}
        class_names = self.class_names_for_indices(state, normalized_indices)
        return class_map, class_names

    def optimal_image_size(self, images: list[str]) -> int:
        sizes = []
        for image_path in images:
            try:
                with Image.open(image_path) as img:
                    sizes.append(img.size)
            except Exception:  # noqa: BLE001
                continue
        if not sizes:
            return 640

        avg_width = sum(width for width, _ in sizes) / len(sizes)
        avg_height = sum(height for _, height in sizes) / len(sizes)

        def round_to_multiple(value, base=32):
            return base * round(value / base)

        optimal_size = max(round_to_multiple(avg_width), round_to_multiple(avg_height))
        return min(max(int(optimal_size), 320), 1280)

    def split_yolo_images(self, annotated_images: list[str]) -> tuple[list[str], list[str]]:
        images = sorted(dict.fromkeys(annotated_images))
        if len(images) <= 1:
            return images, images

        import random

        random.Random(20260711).shuffle(images)
        val_count = min(len(images) - 1, max(1, round(len(images) * 0.2)))
        return images[val_count:], images[:val_count]

    def write_yolo_split(
        self,
        annotation_source,
        gray_path_resolver,
        image_paths: list[str],
        images_dir: str,
        labels_dir: str,
        class_index_map: dict[int, int] | None = None,
    ) -> None:
        state = self._state(annotation_source)
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        for index, image_path in enumerate(image_paths):
            source_path = gray_path_resolver(image_path)
            extension = os.path.splitext(source_path)[1] or ".png"
            base_name = f"img_{index:04d}"
            target_image_path = os.path.join(images_dir, f"{base_name}{extension}")
            shutil.copy(source_path, target_image_path)
            label_path = os.path.join(labels_dir, f"{base_name}.txt")
            with open(label_path, "w", encoding="utf-8") as handle:
                for box in state.annotations[image_path]:
                    cls, x, y, w, h = box.to_tuple() if hasattr(box, "to_tuple") else box
                    if class_index_map is not None:
                        if cls not in class_index_map:
                            continue
                        cls = class_index_map[cls]
                    handle.write(f"{cls} {x} {y} {w} {h}\n")

    def crop_resnet_dataset(
        self,
        annotation_source,
        gray_path_resolver,
        output_dir: str,
        selected_class_map: dict[int, str],
    ) -> ResnetCropSummary:
        state = self._state(annotation_source)
        if not selected_class_map:
            summary = self.annotation_class_summary(state)
            selected_class_map = {
                idx: name
                for idx, name in enumerate(summary.class_names)
                if idx in summary.used_classes
            }
        selected_class_map = dict(sorted(selected_class_map.items(), key=lambda item: int(item[0])))
        os.makedirs(output_dir, exist_ok=True)
        self._clear_resnet_crop_output(output_dir)
        for class_name in selected_class_map.values():
            os.makedirs(os.path.join(output_dir, str(class_name)), exist_ok=True)

        crop_count = 0
        actual_classes: set[str] = set()
        class_counts: dict[str, int] = {}
        skipped_missing_class = 0
        skipped_invalid_box = 0
        skipped_image_error = 0
        for image_index, image_path in enumerate(state.images):
            annotations = state.annotations.get(image_path)
            if not annotations:
                continue
            try:
                gray_path = gray_path_resolver(image_path)
                with Image.open(gray_path) as img:
                    width, height = img.size
                    for box in annotations:
                        cls, x, y, w, h = box.to_tuple() if hasattr(box, "to_tuple") else box
                        if cls not in selected_class_map:
                            skipped_missing_class += 1
                            continue
                        if w <= 0 or h <= 0:
                            skipped_invalid_box += 1
                            continue
                        crop = crop_normalized_box(img, x, y, w, h)
                        class_name = str(selected_class_map[cls])
                        actual_classes.add(class_name)
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
                        class_crop_index = class_counts[class_name] - 1
                        crop_path = os.path.join(
                            output_dir,
                            class_name,
                            f"crop_source_{image_index:04d}_crop_{class_crop_index:05d}.jpg",
                        )
                        crop.save(crop_path)
                        crop_count += 1
            except Exception:  # noqa: BLE001
                skipped_image_error += 1
                continue

        return ResnetCropSummary(
            crop_count=crop_count,
            actual_classes=actual_classes,
            class_counts=class_counts,
            skipped_missing_class=skipped_missing_class,
            skipped_invalid_box=skipped_invalid_box,
            skipped_image_error=skipped_image_error,
        )

    def _clear_resnet_crop_output(self, output_dir: str) -> None:
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
