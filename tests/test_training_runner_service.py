from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

from app.core import AppPaths
from app.services.training_runner_service import ResnetTrainingCallbacks, TrainingRunnerService


class DummyTrainingManager:
    def __init__(self) -> None:
        self.kwargs = None

    def train_yolo(self, **kwargs) -> None:
        self.kwargs = kwargs


def test_training_runner_service_runs_yolo_with_expected_arguments(tmp_path: Path):
    service = TrainingRunnerService()
    manager = DummyTrainingManager()
    paths = AppPaths.from_project_root(tmp_path)

    service.run_yolo(
        training_manager=manager,
        app_paths=paths,
        yolo_model_path="weights.pt",
        data_yaml_path="data.yaml",
        epochs=10,
        batch_size=4,
        img_size=640,
        device="0",
        project_dir=str(tmp_path),
    )

    assert manager.kwargs is not None
    assert manager.kwargs["weights"] == "weights.pt"
    assert manager.kwargs["data_yaml"] == "data.yaml"
    assert manager.kwargs["epochs"] == 10
    assert manager.kwargs["project"] == str(tmp_path)
    assert manager.kwargs["name"] == "yolo_train"


def test_training_runner_service_runs_resnet_and_emits_callbacks(monkeypatch):
    service = TrainingRunnerService()
    logs: list[str] = []
    progress: list[tuple[int, int]] = []
    losses: list[tuple[int, float, float]] = []

    fake_module = ModuleType("ml.ResNet_train")

    def fake_main():
        print("Epoch 1 Training Loss: 0.5 Prediction Error: 0.2")
        print("Epoch 2 Training Loss: 0.3 Prediction Error: 0.1")

    fake_module.main = fake_main
    monkeypatch.setattr("importlib.import_module", lambda name: fake_module)

    callbacks = ResnetTrainingCallbacks(
        log=logs.append,
        on_progress=lambda current, total: progress.append((current, total)),
        on_resnet_loss=lambda epoch, train_loss, pred_error: losses.append((epoch, train_loss, pred_error)),
    )
    original_argv = sys.argv.copy()
    try:
        service.run_resnet(
            pretrained_model_path="resnet.pth",
            training_path="train/",
            testing_path="test/",
            saving_path="save/",
            epochs=5,
            batch_size=2,
            enable_imbalance=True,
            callbacks=callbacks,
        )
    finally:
        sys.argv = original_argv

    assert any("Starting ResNet training" in line for line in logs)
    assert any("Class imbalance handling: enabled" in line for line in logs)
    assert progress == [(1, 5), (2, 5)]
    assert losses[-1] == (2, 0.3, 0.1)
