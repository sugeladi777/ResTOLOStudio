from __future__ import annotations

from pathlib import Path

from app.core import AppPaths
from app.services.resource_loader_service import ResourceLoaderService


class DummyModelManager:
    def __init__(self) -> None:
        self.yolo_loaded: list[str] = []
        self.resnet_loaded: list[str] = []

    def load_yolo_model(self, path: str) -> None:
        self.yolo_loaded.append(path)

    def load_resnet_model(self, path: str) -> None:
        self.resnet_loaded.append(path)


class DummyTarget:
    def __init__(self) -> None:
        self.value = ""

    def setText(self, value: str) -> None:
        self.value = value


def test_resource_loader_service_applies_default_paths(tmp_path: Path):
    paths = AppPaths.from_project_root(tmp_path)
    paths.default_yolo_model_path.parent.mkdir(parents=True, exist_ok=True)
    paths.default_resnet_model_path.parent.mkdir(parents=True, exist_ok=True)
    paths.default_classes_path.parent.mkdir(parents=True, exist_ok=True)
    paths.default_yolo_model_path.write_text("yolo", encoding="utf-8")
    paths.default_resnet_model_path.write_text("resnet", encoding="utf-8")
    paths.default_classes_path.write_text("names: ['atom']\n", encoding="utf-8")

    service = ResourceLoaderService()
    manager = DummyModelManager()
    train_yolo = DummyTarget()
    infer_yolo = DummyTarget()
    train_resnet = DummyTarget()
    infer_resnet = DummyTarget()
    infer_classes = DummyTarget()

    service.apply_default_model_paths(
        paths,
        manager,
        train_yolo,
        infer_yolo,
        train_resnet,
        infer_resnet,
        infer_classes,
    )

    assert manager.yolo_loaded == [str(paths.default_yolo_model_path)]
    assert manager.resnet_loaded == [str(paths.default_resnet_model_path)]
    assert train_yolo.value == str(paths.default_yolo_model_path)
    assert infer_yolo.value == str(paths.default_yolo_model_path)
    assert train_resnet.value == str(paths.default_resnet_model_path)
    assert infer_resnet.value == str(paths.default_resnet_model_path)
    assert infer_classes.value == str(paths.default_classes_path)


def test_resource_loader_service_loads_resources_and_validates_dataset(tmp_path: Path):
    service = ResourceLoaderService()
    manager = DummyModelManager()
    logs: list[str] = []
    yolo_target = DummyTarget()
    resnet_target = DummyTarget()
    classes_target = DummyTarget()
    dataset_target = DummyTarget()

    dataset_dir = tmp_path / "dataset"
    (dataset_dir / "atom").mkdir(parents=True)
    (dataset_dir / "void").mkdir(parents=True)

    assert service.load_yolo_model("model.pt", manager, logs.append, yolo_target) is True
    assert service.load_resnet_model("model.pth", manager, logs.append, resnet_target) is True
    assert service.load_classes_file("classes.yaml", logs.append, classes_target) is True
    loaded_dataset, class_count = service.load_resnet_dataset(str(dataset_dir), logs.append, dataset_target)

    assert loaded_dataset is True
    assert class_count == 2
    assert manager.yolo_loaded[-1] == "model.pt"
    assert manager.resnet_loaded[-1] == "model.pth"
    assert yolo_target.value == "model.pt"
    assert resnet_target.value == "model.pth"
    assert classes_target.value == "classes.yaml"
    assert dataset_target.value == str(dataset_dir)
    assert "已加载检测模型：model.pt" in logs
    assert "已加载分类模型：model.pth" in logs
    assert "已加载类别文件：classes.yaml" in logs
    assert f"已加载分类数据：{dataset_dir}" in logs
