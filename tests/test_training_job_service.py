from __future__ import annotations

from app.services.training_job_service import TrainingJobService


def test_training_job_service_finds_yolo_artifacts(tmp_path):
    project_dir = tmp_path / "project"
    output_dir = project_dir / "yolo_train"
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True)
    best_pt = weights_dir / "best.pt"
    best_pt.write_text("yolo", encoding="utf-8")
    classes_yaml = project_dir / "classes.yaml"
    classes_yaml.write_text("names: ['atom']\n", encoding="utf-8")

    service = TrainingJobService()
    context = service.start_context("session-1", "yolo", str(project_dir))
    record = service.complete_record(context, status="completed")

    assert record.output_dir == str(output_dir)
    assert record.yolo_model_path == str(best_pt)
    assert record.resnet_model_path == ""
    assert record.classes_yaml_path == str(classes_yaml)


def test_training_job_service_finds_resnet_artifacts(tmp_path):
    project_dir = tmp_path / "project"
    output_dir = project_dir / "resnet_train"
    output_dir.mkdir(parents=True)
    best_model = output_dir / "Model_best_newer.saving"
    best_model.write_text("best", encoding="utf-8")

    service = TrainingJobService()
    context = service.start_context("session-1", "resnet", str(project_dir))
    record = service.complete_record(context, status="completed")

    assert record.output_dir == str(output_dir)
    assert record.yolo_model_path == ""
    assert record.resnet_model_path == str(best_model)
