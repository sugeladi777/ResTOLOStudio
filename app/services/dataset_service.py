from __future__ import annotations

import os
import random
import shutil
from dataclasses import dataclass

from app.core import AnnotationState
from PIL import Image


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
        shuffled_images = annotated_images.copy()
        random.shuffle(shuffled_images)
        return shuffled_images, shuffled_images

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
        for image_path in state.images:
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
                        x1 = max(0, int((x - w / 2) * width))
                        y1 = max(0, int((y - h / 2) * height))
                        x2 = min(width, int((x + w / 2) * width))
                        y2 = min(height, int((y + h / 2) * height))
                        if x2 <= x1 or y2 <= y1:
                            skipped_invalid_box += 1
                            continue
                        crop = img.crop((x1, y1, x2, y2))
                        if crop.mode not in ("RGB", "L"):
                            crop = crop.convert("RGB")
                        class_name = str(selected_class_map[cls])
                        actual_classes.add(class_name)
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
                        class_crop_index = class_counts[class_name] - 1
                        crop_path = os.path.join(output_dir, class_name, f"crop_{class_crop_index}.jpg")
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
