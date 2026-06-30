from __future__ import annotations

import os
import random
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml

from app.core import AnnotationState, SessionRecord
from app.services.dataset_service import DatasetService
from app.services.training_job_service import TrainingJobService


@dataclass
class YoloTrainingPlan:
    data_yaml_path: str
    annotated_images: list[str]
    img_size: int
    class_names: list[str]


@dataclass
class ResnetTrainingPlan:
    training_path: str
    testing_path: str
    saving_path: str
    project_dir: str
    class_names: list[str]


class TrainingWorkflowService:
    """Prepares training session context and datasets outside the window layer."""

    def __init__(self, dataset_service: DatasetService, training_job_service: TrainingJobService):
        self.dataset_service = dataset_service
        self.training_job_service = training_job_service

    def ensure_training_context(
        self,
        current_session: SessionRecord | None,
        session_workflow_service,
        mode: str,
        project_dir: str,
    ) -> tuple[SessionRecord | None, dict | None]:
        if session_workflow_service is None:
            return current_session, None
        session = session_workflow_service.ensure_session(current_session, "training")
        context = self.training_job_service.start_context(session.id, mode, project_dir)
        return session, context

    def validate_annotation_training_data(
        self,
        state: AnnotationState,
        annotation_service,
        log,
    ) -> bool:
        if not annotation_service.has_images(state):
            log("错误：请先加载训练图像")
            return False
        if not annotation_service.has_annotations(state):
            log("错误：请先加载或创建标注")
            return False
        return True

    def prepare_yolo_training(
        self,
        state: AnnotationState,
        gray_path_resolver,
        training_manager,
        log,
    ) -> YoloTrainingPlan | None:
        annotated_images = self.dataset_service.annotated_images(state)
        total_images = len(state.images)
        removed_count = total_images - len(annotated_images)
        if removed_count > 0:
            log(f"已移除 {removed_count} 张未标注图像，保留 {len(annotated_images)} 张已标注图像")
        else:
            log(f"全部 {total_images} 张图像都包含标注")
        if not annotated_images:
            log("错误：未找到已标注图像")
            return None

        img_size = self.dataset_service.optimal_image_size(annotated_images)
        temp_dir = tempfile.mkdtemp(prefix="restolo_yolo_")
        data_yaml_path = os.path.join(temp_dir, "data.yaml")
        train_images, val_images = self.dataset_service.split_yolo_images(annotated_images)
        total_annotated = len(annotated_images)
        if total_annotated == 1:
            log("仅有一张已标注图像，将同时用于训练和验证")
        else:
            log(f"已标注图像：{total_annotated}，训练 {len(train_images)}，验证 {len(val_images)}")

        train_images_dir = os.path.join(temp_dir, "images", "train")
        val_images_dir = os.path.join(temp_dir, "images", "val")
        train_labels_dir = os.path.join(temp_dir, "labels", "train")
        val_labels_dir = os.path.join(temp_dir, "labels", "val")
        used_classes = self.dataset_service.annotation_class_summary(state).used_classes
        class_index_map = {int(cls): 0 for cls in used_classes}
        class_names = ["object"]
        self.dataset_service.write_yolo_split(
            state,
            gray_path_resolver,
            train_images,
            train_images_dir,
            train_labels_dir,
            class_index_map=class_index_map,
        )
        self.dataset_service.write_yolo_split(
            state,
            gray_path_resolver,
            val_images,
            val_images_dir,
            val_labels_dir,
            class_index_map=class_index_map,
        )

        training_manager.generate_data_yaml(
            train_images_dir,
            val_images_dir,
            1,
            class_names,
            data_yaml_path,
        )
        log(f"已生成数据配置：{data_yaml_path}")
        log(f"类别数量：{len(class_names)}")
        return YoloTrainingPlan(
            data_yaml_path=data_yaml_path,
            annotated_images=annotated_images,
            img_size=img_size,
            class_names=class_names,
        )

    def prepare_resnet_training(
        self,
        annotation_source,
        gray_path_resolver,
        project_dir: str,
        resnet_data_path: str | None,
        selected_class_indices: list[int],
        class_names: list[str],
        classes_yaml_path: str | None,
        log,
    ) -> ResnetTrainingPlan | None:
        selected_class_names = self._resolve_selected_class_names(
            annotation_source,
            selected_class_indices,
            class_names,
        )
        resolved_class_names = self._resolve_training_class_names(
            annotation_source,
            selected_class_names,
            classes_yaml_path,
            log,
        )
        training_path, testing_path = self._prepare_resnet_paths(
            annotation_source,
            gray_path_resolver,
            project_dir,
            resnet_data_path,
            selected_class_indices,
            selected_class_names,
            log,
        )
        if not training_path or not testing_path:
            return None
        saving_path = os.path.join(project_dir, "resnet_train")
        os.makedirs(saving_path, exist_ok=True)
        plan_class_names = resolved_class_names if resnet_data_path else selected_class_names
        class_indices = None if resnet_data_path else selected_class_indices
        self._write_classes_yaml(saving_path, plan_class_names, class_indices)
        return ResnetTrainingPlan(
            training_path=training_path,
            testing_path=testing_path,
            saving_path=saving_path,
            project_dir=project_dir,
            class_names=plan_class_names,
        )

    def _write_classes_yaml(
        self,
        output_dir: str,
        class_names: list[str],
        class_indices: list[int] | None = None,
    ) -> None:
        if not output_dir or not class_names:
            return
        payload = {"names": list(class_names), "nc": len(class_names)}
        if class_indices and len(class_indices) == len(class_names):
            payload["indices"] = [int(index) for index in class_indices]
        classes_path = Path(output_dir) / "classes.yaml"
        with classes_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)

    def _resolve_selected_class_names(
        self,
        annotation_source,
        selected_class_indices: list[int],
        class_names: list[str],
    ) -> list[str]:
        if class_names:
            return list(class_names)
        if selected_class_indices:
            return self.dataset_service.class_names_for_indices(annotation_source, selected_class_indices)
        summary = self.dataset_service.annotation_class_summary(annotation_source)
        return list(summary.class_names)

    def _resolve_training_class_names(
        self,
        annotation_source,
        class_names: list[str],
        classes_yaml_path: str | None,
        log,
    ) -> list[str]:
        if class_names:
            return list(class_names)

        state = self.dataset_service._state(annotation_source)
        summary = self.dataset_service.annotation_class_summary(state)
        if summary.class_names:
            return list(summary.class_names)
        if getattr(state, "class_names", None):
            return list(state.class_names)

        yaml_names = self._load_class_names_from_yaml(classes_yaml_path, log)
        if yaml_names:
            log(f"训练类别来自类别文件：{yaml_names}")
            return yaml_names
        return []

    def _load_class_names_from_yaml(self, classes_yaml_path: str | None, log) -> list[str]:
        if not classes_yaml_path:
            return []
        path = Path(classes_yaml_path)
        if not path.exists():
            log(f"类别文件不存在，忽略：{classes_yaml_path}")
            return []
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle) or {}
            names = payload.get("names", [])
            if isinstance(names, dict):
                names = [name for _, name in sorted(names.items(), key=lambda item: int(item[0]))]
            if not isinstance(names, list):
                return []
            return [str(name) for name in names]
        except Exception as exc:  # noqa: BLE001
            log(f"读取类别文件失败，忽略：{exc}")
            return []

    def _prepare_resnet_paths(
        self,
        annotation_source,
        gray_path_resolver,
        project_dir: str,
        resnet_data_path: str | None,
        selected_class_indices: list[int],
        selected_class_names: list[str],
        log,
    ) -> tuple[str | None, str | None]:
        if resnet_data_path:
            log(f"使用分类数据集：{resnet_data_path}")
            dataset_classes = self._list_resnet_classes(resnet_data_path)
            log(f"数据集类别：{dataset_classes}")
            dataset_path = resnet_data_path + os.sep
            return dataset_path, dataset_path

        crop_dir = os.path.join(project_dir, "resnet_crop")
        os.makedirs(crop_dir, exist_ok=True)
        log(f"选中训练类别索引：{selected_class_indices}")
        log(f"训练类别：{selected_class_names}")
        selected_class_map = {idx: name for idx, name in zip(selected_class_indices, selected_class_names)}
        if not selected_class_map and selected_class_indices:
            fallback_names = self.dataset_service.class_names_for_indices(annotation_source, selected_class_indices)
            selected_class_map = {idx: name for idx, name in zip(selected_class_indices, fallback_names)}

        annotation_summary = self.dataset_service.annotation_class_summary(annotation_source)
        log(f"当前已标注类别索引：{sorted(annotation_summary.used_classes)}")
        log(f"训练类别映射：{selected_class_map}")

        summary = self.dataset_service.crop_resnet_dataset(
            annotation_source,
            gray_path_resolver,
            crop_dir,
            selected_class_map,
        )
        if summary.crop_count == 0:
            log(
                f"分类裁剪失败统计：类别过滤={summary.skipped_missing_class}，"
                f"无效框={summary.skipped_invalid_box}，图像读取失败={summary.skipped_image_error}"
            )
            log("错误：未能裁剪出任何已标注区域")
            return None, None

        log(f"已裁剪 {summary.crop_count} 个标注区域")
        log(f"分类裁剪类别统计：{summary.class_counts}")
        split_root = os.path.join(project_dir, "resnet_train_split")
        train_dir, val_dir = self._split_resnet_crop_dataset(crop_dir, split_root, log)
        return train_dir + os.sep, val_dir + os.sep

    def _split_resnet_crop_dataset(self, crop_dir: str, split_root: str, log) -> tuple[str, str]:
        if os.path.exists(split_root):
            shutil.rmtree(split_root)
        train_dir = os.path.join(split_root, "train")
        val_dir = os.path.join(split_root, "val")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        rng = random.Random(20260630)
        for class_name in sorted(os.listdir(crop_dir)):
            class_dir = os.path.join(crop_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            images = [
                name
                for name in sorted(os.listdir(class_dir))
                if os.path.isfile(os.path.join(class_dir, name))
            ]
            rng.shuffle(images)
            duplicate_singleton = len(images) == 1
            if len(images) <= 1:
                val_names = images
            else:
                val_count = max(1, int(round(len(images) * 0.2)))
                val_count = min(val_count, len(images) - 1)
                val_names = images[:val_count]
            val_set = set(val_names)
            for split_dir in (train_dir, val_dir):
                os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)
            for image_name in images:
                source = os.path.join(class_dir, image_name)
                if duplicate_singleton:
                    shutil.copy2(source, os.path.join(train_dir, class_name, image_name))
                    shutil.copy2(source, os.path.join(val_dir, class_name, image_name))
                    continue
                target_root = val_dir if image_name in val_set else train_dir
                shutil.copy2(source, os.path.join(target_root, class_name, image_name))
            if not images:
                log(f"类别 {class_name} 没有裁剪样本，已保留空目录")

        log(f"分类训练集目录：{train_dir}")
        log(f"分类验证集目录：{val_dir}")
        return train_dir, val_dir

    def _list_resnet_classes(self, resnet_data_path: str) -> list[str]:
        class_names = []
        for item in os.listdir(resnet_data_path):
            item_path = os.path.join(resnet_data_path, item)
            if os.path.isdir(item_path):
                class_names.append(item)
        return class_names
