from __future__ import annotations

from pathlib import Path

from PIL import Image

from app.core import AnnotationBox, AnnotationState, SessionRecord
from app.services.dataset_service import DatasetService
from app.services.training_job_service import TrainingJobService
from app.services.training_workflow_service import TrainingWorkflowService


class DummyAnnotationService:
    def has_images(self, state) -> bool:
        return bool(state.images)

    def has_annotations(self, state) -> bool:
        return any(state.annotations.get(path) for path in state.images)


class DummyTrainingManager:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int, list[str], str]] = []

    def generate_data_yaml(
        self,
        train_images_dir: str,
        val_images_dir: str,
        class_count: int,
        class_names: list[str],
        data_yaml_path: str,
    ) -> None:
        self.calls.append((train_images_dir, val_images_dir, class_count, list(class_names), data_yaml_path))
        Path(data_yaml_path).write_text("nc: 1\nnames: ['atom']\n", encoding="utf-8")


class DummySessionWorkflowService:
    def __init__(self, session: SessionRecord) -> None:
        self.session = session
        self.calls: list[tuple[SessionRecord | None, str]] = []

    def ensure_session(self, current_session: SessionRecord | None, label: str):
        self.calls.append((current_session, label))
        return current_session or self.session


def _make_image(path: Path) -> str:
    Image.new("RGB", (64, 64), "white").save(path)
    return str(path)


def _annotation_state(image_path: str) -> AnnotationState:
    return AnnotationState(
        images=[image_path],
        annotations={
            image_path: [
                AnnotationBox(cls=0, x=0.5, y=0.5, w=0.5, h=0.5),
            ]
        },
        class_names=["atom"],
        current_index=0,
    )


def test_training_workflow_service_prepares_yolo_plan_and_context(tmp_path: Path):
    image_path = _make_image(tmp_path / "sample.png")
    state = _annotation_state(image_path)
    logs: list[str] = []
    training_manager = DummyTrainingManager()
    workflow = TrainingWorkflowService(DatasetService(), TrainingJobService())
    session = SessionRecord(id="s1", label="training", created_at="now")
    session_service = DummySessionWorkflowService(session)

    ensured_session, context = workflow.ensure_training_context(None, session_service, "yolo", str(tmp_path))
    plan = workflow.prepare_yolo_training(state, lambda path: path, training_manager, logs.append)

    assert ensured_session is session
    assert context["session_id"] == "s1"
    assert context["mode"] == "yolo"
    assert context["project_dir"] == str(tmp_path)
    assert Path(context["expected_output_dir"]).name == "yolo_train"
    assert plan is not None
    assert Path(plan.data_yaml_path).exists()
    assert plan.img_size >= 320
    assert plan.class_names == ["object"]
    assert training_manager.calls
    assert any("已生成数据配置" in message for message in logs)


def test_training_workflow_service_prepares_resnet_plan_from_annotations(tmp_path: Path):
    image_path = _make_image(tmp_path / "sample.png")
    state = _annotation_state(image_path)
    logs: list[str] = []
    workflow = TrainingWorkflowService(DatasetService(), TrainingJobService())
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    plan = workflow.prepare_resnet_training(
        state,
        lambda path: path,
        str(project_dir),
        None,
        [0],
        ["atom"],
        "",
        logs.append,
    )

    assert plan is not None
    assert plan.class_names == ["atom"]
    assert plan.saving_path.endswith("resnet_train")
    assert Path(plan.saving_path).exists()
    assert any("已裁剪" in message for message in logs)
    assert plan.training_path == plan.testing_path
    assert plan.training_path.endswith("resnet_crop\\")


def test_training_workflow_service_resnet_ignores_classes_yaml_when_annotation_selection_exists(tmp_path: Path):
    image_path = _make_image(tmp_path / "sample.png")
    state = _annotation_state(image_path)
    logs: list[str] = []
    workflow = TrainingWorkflowService(DatasetService(), TrainingJobService())
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    classes_yaml = tmp_path / "classes.yaml"
    classes_yaml.write_text("names:\n  - atom\n  - vacancy\n", encoding="utf-8")

    plan = workflow.prepare_resnet_training(
        state,
        lambda path: path,
        str(project_dir),
        None,
        [0],
        ["atom"],
        str(classes_yaml),
        logs.append,
    )

    assert plan is not None
    assert plan.class_names == ["atom"]
    assert not any("训练类别来自类别文件" in message for message in logs)


def test_training_workflow_service_validates_annotation_training_data():
    workflow = TrainingWorkflowService(DatasetService(), TrainingJobService())
    annotation_service = DummyAnnotationService()
    logs: list[str] = []

    empty_state = AnnotationState(images=[], annotations={}, class_names=["0"], current_index=0)
    assert workflow.validate_annotation_training_data(empty_state, annotation_service, logs.append) is False
    assert logs[-1] == "错误：请先加载训练图像"

    logs.clear()
    image_state = AnnotationState(images=["a.png"], annotations={"a.png": []}, class_names=["0"], current_index=0)
    assert workflow.validate_annotation_training_data(image_state, annotation_service, logs.append) is False
    assert logs[-1] == "错误：请先加载或创建标注"


def test_training_workflow_service_remaps_sparse_yolo_classes(tmp_path: Path):
    image_path = _make_image(tmp_path / "sample.png")
    state = AnnotationState(
        images=[image_path],
        annotations={
            image_path: [
                AnnotationBox(cls=1, x=0.5, y=0.5, w=0.5, h=0.5),
            ]
        },
        class_names=["0", "1"],
        current_index=0,
    )
    logs: list[str] = []
    training_manager = DummyTrainingManager()
    workflow = TrainingWorkflowService(DatasetService(), TrainingJobService())

    plan = workflow.prepare_yolo_training(state, lambda path: path, training_manager, logs.append)

    assert plan is not None
    assert plan.class_names == ["object"]
    assert training_manager.calls[0][2] == 1
    assert training_manager.calls[0][3] == ["object"]
    train_labels_dir = Path(training_manager.calls[0][0]).parents[1] / "labels" / "train"
    assert next(train_labels_dir.glob("*.txt")).read_text(encoding="utf-8").strip() == "0 0.5 0.5 0.5 0.5"


def test_training_workflow_service_resnet_uses_selected_annotation_classes(tmp_path: Path):
    image_path = _make_image(tmp_path / "sample.png")
    state = AnnotationState(
        images=[image_path],
        annotations={
            image_path: [
                AnnotationBox(cls=0, x=0.5, y=0.5, w=0.5, h=0.5),
                AnnotationBox(cls=1, x=0.25, y=0.25, w=0.25, h=0.25),
            ]
        },
        class_names=["atom", "vacancy"],
        current_index=0,
    )
    logs: list[str] = []
    workflow = TrainingWorkflowService(DatasetService(), TrainingJobService())
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    plan = workflow.prepare_resnet_training(
        state,
        lambda path: path,
        str(project_dir),
        None,
        [1],
        ["vacancy"],
        "",
        logs.append,
    )

    assert plan is not None
    assert plan.class_names == ["vacancy"]
    crop_dir = project_dir / "resnet_crop"
    assert (crop_dir / "vacancy").exists()
    assert not (crop_dir / "atom").exists()
