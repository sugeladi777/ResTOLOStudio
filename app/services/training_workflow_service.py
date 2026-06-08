from __future__ import annotations

import os
import random
import shutil
import tempfile
import time
from dataclasses import dataclass

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
        self.dataset_service.write_yolo_split(state, gray_path_resolver, train_images, train_images_dir, train_labels_dir)
        self.dataset_service.write_yolo_split(state, gray_path_resolver, val_images, val_images_dir, val_labels_dir)

        class_names = list(state.class_names)
        training_manager.generate_data_yaml(
            train_images_dir,
            val_images_dir,
            len(class_names),
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
        log,
    ) -> ResnetTrainingPlan | None:
        training_path, testing_path = self._prepare_resnet_paths(
            annotation_source,
            gray_path_resolver,
            project_dir,
            resnet_data_path,
            selected_class_indices,
            class_names,
            log,
        )
        if not training_path or not testing_path:
            return None
        saving_path = os.path.join(project_dir, "resnet_train")
        os.makedirs(saving_path, exist_ok=True)
        return ResnetTrainingPlan(
            training_path=training_path,
            testing_path=testing_path,
            saving_path=saving_path,
            project_dir=project_dir,
        )

    def _prepare_resnet_paths(
        self,
        annotation_source,
        gray_path_resolver,
        project_dir: str,
        resnet_data_path: str | None,
        selected_class_indices: list[int],
        class_names: list[str],
        log,
    ) -> tuple[str | None, str | None]:
        if resnet_data_path:
            log(f"使用分类数据集：{resnet_data_path}")
            dataset_classes = self._list_resnet_classes(resnet_data_path)
            log(f"数据集类别：{dataset_classes}")
            try:
                return self._run_kfold_split(resnet_data_path, log)
            except Exception as exc:  # noqa: BLE001
                log(f"数据划分失败，改用回退方案：{exc}")
                return self._fallback_project_split(resnet_data_path, project_dir, dataset_classes)

        crop_dir = os.path.join(project_dir, "resnet_crop")
        os.makedirs(crop_dir, exist_ok=True)
        log(f"选中训练类别索引：{selected_class_indices}")
        log(f"训练类别：{class_names}")
        selected_class_map = {idx: name for idx, name in zip(selected_class_indices, class_names)}
        summary = self.dataset_service.crop_resnet_dataset(
            annotation_source,
            gray_path_resolver,
            crop_dir,
            selected_class_map,
        )
        if summary.crop_count == 0:
            log("错误：未能裁剪出任何已标注区域")
            return None, None

        log(f"已裁剪 {summary.crop_count} 个标注区域")
        try:
            training_path, testing_path = self._run_kfold_split(crop_dir, log)
            log("数据划分完成")
            return training_path, testing_path
        except Exception as exc:  # noqa: BLE001
            log(f"数据划分失败，改用回退方案：{exc}")
            return self._fallback_project_split(crop_dir, project_dir, sorted(summary.actual_classes))

    def _run_kfold_split(self, source_dir: str, log) -> tuple[str, str]:
        from ml.data_divider import kfold

        log(f"正在划分数据：{source_dir}，k=10")
        kfold(source_dir, k=10)
        jour = f"{time.localtime().tm_mon}{time.localtime().tm_mday}"
        return source_dir + f"_train_{jour}_0/", source_dir + f"_test_{jour}_0/"

    def _list_resnet_classes(self, resnet_data_path: str) -> list[str]:
        class_names = []
        for item in os.listdir(resnet_data_path):
            item_path = os.path.join(resnet_data_path, item)
            if os.path.isdir(item_path):
                class_names.append(item)
        return class_names

    def _fallback_project_split(self, source_root: str, project_dir: str, class_names: list[str]) -> tuple[str, str]:
        train_dir = os.path.join(project_dir, "train")
        val_dir = os.path.join(project_dir, "val")
        self._fallback_split_classification_dataset(source_root, train_dir, val_dir, class_names)
        return train_dir + "/", val_dir + "/"

    def _fallback_split_classification_dataset(
        self,
        source_root: str,
        train_dir: str,
        val_dir: str,
        class_names: list[str],
    ) -> None:
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        for class_name in class_names:
            os.makedirs(os.path.join(train_dir, str(class_name)), exist_ok=True)
            os.makedirs(os.path.join(val_dir, str(class_name)), exist_ok=True)

        for class_name in class_names:
            class_dir = os.path.join(source_root, str(class_name))
            if not os.path.exists(class_dir):
                continue
            images = os.listdir(class_dir)
            random.shuffle(images)
            split_idx = max(1, int(len(images) * 0.9))
            train_images = images[:split_idx]
            val_images = images[split_idx:]
            for image_name in train_images:
                shutil.copy(os.path.join(class_dir, image_name), os.path.join(train_dir, str(class_name), image_name))
            for image_name in val_images:
                shutil.copy(os.path.join(class_dir, image_name), os.path.join(val_dir, str(class_name), image_name))
