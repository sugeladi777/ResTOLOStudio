from __future__ import annotations

import os
import shutil
from pathlib import Path

import yaml

from app.core import AnnotationBox, AnnotationState


class AnnotationService:
    """Owns annotation domain state and file IO outside the widget layer."""

    def create_state(self, image_paths: list[str], class_names: list[str] | None = None) -> AnnotationState:
        names = list(class_names) if class_names else [str(i) for i in range(10)]
        return AnnotationState(
            images=list(image_paths),
            annotations={path: [] for path in image_paths},
            class_names=names,
            current_index=0,
        ).normalized()

    def has_images(self, state: AnnotationState) -> bool:
        return bool(state.images)

    def has_annotations(self, state: AnnotationState) -> bool:
        return any(state.annotations.get(path) for path in state.images)

    def used_classes(self, state: AnnotationState) -> set[int]:
        used: set[int] = set()
        for boxes in state.annotations.values():
            for box in boxes:
                used.add(box.cls)
        return used

    def annotated_images(self, state: AnnotationState) -> list[str]:
        return [image_path for image_path in state.images if state.annotations.get(image_path)]

    def remap_image_paths(self, state: AnnotationState, path_map: dict[str, str]) -> AnnotationState:
        remapped_images = [path_map.get(path, path) for path in state.images]
        remapped_annotations: dict[str, list[AnnotationBox]] = {}
        for image_path, boxes in state.annotations.items():
            remapped_annotations[path_map.get(image_path, image_path)] = list(boxes)
        return AnnotationState(
            images=remapped_images,
            annotations=remapped_annotations,
            class_names=list(state.class_names),
            current_index=state.current_index,
        ).normalized()

    def load_annotation_files(self, state: AnnotationState, annotation_paths: list[str]) -> AnnotationState:
        updated = state.normalized()
        annotations = {path: list(boxes) for path, boxes in updated.annotations.items()}

        for ann_path in annotation_paths:
            image_path = self._match_annotation_image(updated.images, ann_path)
            if image_path is None:
                continue
            annotations[image_path] = self._parse_annotation_file(ann_path)

        return AnnotationState(
            images=list(updated.images),
            annotations=annotations,
            class_names=list(updated.class_names),
            current_index=updated.current_index,
        ).normalized()

    def save_annotations(self, state: AnnotationState, directory: str) -> tuple[Path | None, list[str]]:
        output_dir = self._ensure_output_directory(directory)
        if output_dir is None:
            return None, []

        normalized = state.normalized()
        self._copy_annotation_images(output_dir, normalized)
        for image_path, boxes in normalized.annotations.items():
            self._write_annotation_file(output_dir, image_path, boxes)

        _, used_class_names = self._write_classes_yaml(output_dir, normalized)
        return output_dir, used_class_names

    def load_saved_annotation_state(self, directory: str) -> AnnotationState | None:
        base_dir = self._ensure_output_directory(directory)
        if base_dir is None:
            return None

        image_paths = self._saved_image_paths(base_dir)
        if not image_paths:
            return None

        class_names = self._read_classes_yaml(base_dir / "classes.yaml")
        state = self.create_state(image_paths, class_names=class_names)
        annotation_paths = [str(path) for path in sorted(base_dir.glob("*.txt"))]
        if not annotation_paths:
            return state
        return self.load_annotation_files(state, annotation_paths)

    def discover_annotation_files(self, image_paths: list[str], candidate_directories: list[str]) -> list[str]:
        if not image_paths:
            return []

        images = [str(Path(path)) for path in image_paths]
        discovered: list[str] = []
        seen: set[str] = set()
        for directory in candidate_directories:
            if not directory:
                continue
            base_dir = Path(directory)
            if not base_dir.exists() or not base_dir.is_dir():
                continue
            for ann_path in sorted(base_dir.glob("*.txt")):
                resolved = str(ann_path)
                if resolved in seen:
                    continue
                if self._match_annotation_image(images, resolved) is None:
                    continue
                seen.add(resolved)
                discovered.append(resolved)
        return discovered

    def discover_classes_yaml(self, candidate_directories: list[str]) -> str:
        seen: set[str] = set()
        for directory in candidate_directories:
            if not directory:
                continue
            base_dir = Path(directory)
            if not base_dir.exists() or not base_dir.is_dir():
                continue
            for candidate in (base_dir / "classes.yaml", base_dir / "classes.yml"):
                resolved = str(candidate)
                if resolved in seen:
                    continue
                seen.add(resolved)
                if candidate.exists():
                    return resolved
        return ""

    def _parse_annotation_file(self, ann_path: str) -> list[AnnotationBox]:
        boxes: list[AnnotationBox] = []
        with open(ann_path, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if not parts or (len(parts) >= 2 and parts[0] == "["):
                    continue
                if len(parts) < 5:
                    continue
                boxes.append(
                    AnnotationBox(
                        cls=int(parts[0]),
                        x=float(parts[1]),
                        y=float(parts[2]),
                        w=float(parts[3]),
                        h=float(parts[4]),
                    )
                )
        return boxes

    def _match_annotation_image(self, images: list[str], ann_path: str) -> str | None:
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            image_path = os.path.splitext(ann_path)[0] + ext
            if image_path in images:
                return image_path

        prefix = Path(ann_path).stem
        for image_path in images:
            if Path(image_path).stem.startswith(prefix):
                return image_path
        return None

    def _ensure_output_directory(self, directory: str) -> Path | None:
        if not directory:
            return None

        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _write_annotation_file(self, output_dir: Path, image_path: str, boxes: list[AnnotationBox]) -> Path:
        ann_path = output_dir / f"{Path(image_path).stem}.txt"
        with ann_path.open("w", encoding="utf-8") as handle:
            for box in boxes:
                handle.write(f"{box.cls} {box.x} {box.y} {box.w} {box.h}\n")
        return ann_path

    def _copy_annotation_images(self, output_dir: Path, state: AnnotationState) -> None:
        for image_path in state.images:
            source = Path(image_path)
            if not source.exists():
                continue
            target = output_dir / source.name
            if source.resolve() == target.resolve():
                continue
            shutil.copy2(source, target)

    def _saved_image_paths(self, directory: Path) -> list[str]:
        image_paths: list[str] = []
        for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            image_paths.extend(str(path) for path in sorted(directory.glob(pattern)))
        return image_paths

    def _read_classes_yaml(self, path: Path) -> list[str] | None:
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle) or {}
            names = payload.get("names", [])
            if isinstance(names, dict):
                names = [name for _, name in sorted(names.items(), key=lambda item: int(item[0]))]
            if not isinstance(names, list):
                return None
            class_names = [str(name) for name in names]
            indices = payload.get("indices")
            if not isinstance(indices, list):
                return class_names

            indexed_names: list[str] = []
            for raw_index, name in zip(indices, class_names):
                try:
                    index = int(raw_index)
                except (TypeError, ValueError):
                    return class_names
                while len(indexed_names) <= index:
                    indexed_names.append(str(len(indexed_names)))
                indexed_names[index] = name
            return indexed_names
        except Exception:  # noqa: BLE001
            return None

    def _write_classes_yaml(self, output_dir: Path, state: AnnotationState) -> tuple[Path, list[str]]:
        used_classes = sorted(self.used_classes(state))
        used_class_names = [
            state.class_names[cls] if cls < len(state.class_names) else str(cls)
            for cls in used_classes
        ]
        classes_path = output_dir / "classes.yaml"
        payload = {"names": used_class_names, "indices": used_classes, "nc": len(used_class_names)}
        with classes_path.open("w", encoding="utf-8") as handle:
            yaml.dump(payload, handle, default_flow_style=False, allow_unicode=True)
        return classes_path, used_class_names
