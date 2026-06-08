from __future__ import annotations

import importlib
import os
import random
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

from PIL import Image
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from app.core import AppPaths
from app.ui.loss_curve_dialog import LossCurveDialog


DEFAULT_APP_PATHS = AppPaths.from_project_root(Path(__file__).resolve().parents[2])


def _app_paths(window):
    runtime = getattr(window, "runtime", None)
    if runtime is not None and hasattr(runtime, "paths"):
        return runtime.paths
    return DEFAULT_APP_PATHS


def _choose_training_project_dir(window):
    project_dir = QFileDialog.getExistingDirectory(window, "选择训练结果保存目录")
    if not project_dir:
        window.log("训练取消")
        return None
    return project_dir


def _has_annotation_training_data(window):
    if not getattr(window.annotation_tool, "images", None):
        window.log("错误: 请先加载训练图片")
        return False
    if not getattr(window.annotation_tool, "annotations", None):
        window.log("错误: 请先加载或创建标注")
        return False
    return True


def _prepare_training_ui(window):
    window.progress_bar.show()
    window.progress_bar.setValue(0)
    window.disable_controls()


def _start_background_training(target):
    train_thread = threading.Thread(target=target, daemon=True)
    train_thread.start()


def _training_parameters(window):
    return window.epochs_spin.value(), window.batch_spin.value()


def _split_dirs(root_dir):
    return (
        os.path.join(root_dir, "train"),
        os.path.join(root_dir, "val"),
    )


def _run_kfold_split(window, source_dir):
    from ml.data_divider import kfold

    window.log(f"执行数据划分: {source_dir}, k=10")
    kfold(source_dir, k=10)
    jour = f"{time.localtime().tm_mon}{time.localtime().tm_mday}"
    return (
        source_dir + f"_train_{jour}_0/",
        source_dir + f"_test_{jour}_0/",
    )


def _annotated_images(window):
    images = []
    for image_path in getattr(window.annotation_tool, "images", []):
        if image_path in window.annotation_tool.annotations and window.annotation_tool.annotations[image_path]:
            images.append(image_path)
    return images


def _selected_class_indices(window):
    if getattr(window, "class_checkboxes", None) and hasattr(window, "class_indices"):
        selected = [
            class_idx
            for checkbox, class_idx in zip(window.class_checkboxes, window.class_indices)
            if checkbox.isChecked()
        ]
        if selected:
            return selected

    used_classes = set()
    for annotations in window.annotation_tool.annotations.values():
        for cls, *_ in annotations:
            used_classes.add(cls)
    return sorted(used_classes)


def _class_names_for_indices(window, class_indices):
    class_names = []
    for idx in class_indices:
        if idx < len(window.annotation_tool.class_names):
            class_names.append(window.annotation_tool.class_names[idx])
        else:
            class_names.append(str(idx))
    return class_names


def _optimal_img_size(images):
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


def _ensure_yolo_loss_dialog(window):
    if not hasattr(window, "loss_curve_dialog") or window.loss_curve_dialog is None:
        window.loss_curve_dialog = LossCurveDialog(window)
    else:
        window.loss_curve_dialog.epochs.clear()
        window.loss_curve_dialog.box_loss.clear()
        window.loss_curve_dialog.obj_loss.clear()
        window.loss_curve_dialog.cls_loss.clear()
        window.loss_curve_dialog.total_loss.clear()
        window.loss_curve_dialog.precision.clear()
        window.loss_curve_dialog.recall.clear()
        window.loss_curve_dialog.map50.clear()
        window.loss_curve_dialog.map50_95.clear()
        window.loss_curve_dialog.val_epochs.clear()
        window.loss_curve_dialog.ax_train.clear()
        window.loss_curve_dialog.ax_val.clear()
        window.loss_curve_dialog._setup_yolo_axes()
        window.loss_curve_dialog.canvas.draw_idle()
    window.loss_curve_dialog.show()
    window.loss_curve_dialog.raise_()


def _ensure_resnet_loss_dialog(window):
    window.loss_curve_dialog = LossCurveDialog(window, mode="resnet")
    window.loss_curve_dialog.show()
    window.loss_curve_dialog.raise_()


def _set_training_project_dirs(window, project_dir):
    window._yolo_project_dir = project_dir
    window._resnet_project_dir = project_dir


def _validate_yolo_annotations(window, annotated_images):
    total_images = len(window.annotation_tool.images)
    removed_count = total_images - len(annotated_images)
    if removed_count > 0:
        window.log(f"剔除了 {removed_count} 张无标注图片，保留了 {len(annotated_images)} 张有标注图片")
    else:
        window.log(f"全部 {total_images} 张图片都有标注")
    if not annotated_images:
        window.log("错误: 没有找到有标注的图片，请先补充标注")
        return False
    return True


def _prepare_yolo_data(window, annotated_images):
    temp_dir = tempfile.mkdtemp(prefix="restolo_yolo_")
    data_yaml_path = os.path.join(temp_dir, "data.yaml")
    train_images_dir, val_images_dir = _split_yolo_dataset(window, temp_dir, annotated_images)
    class_names = window.annotation_tool.class_names
    window.training_manager.generate_data_yaml(
        train_images_dir,
        val_images_dir,
        len(class_names),
        class_names,
        data_yaml_path,
    )
    window.log(f"数据配置文件已生成: {data_yaml_path}")
    window.log(f"类别数量: {len(class_names)}")
    return data_yaml_path


def _run_yolo_training_job(window, app_paths, yolo_model_path, data_yaml_path, epochs, batch_size, img_size, device, project_dir):
    try:
        window.training_manager.train_yolo(
            weights=yolo_model_path,
            cfg=str(app_paths.yolo_config_path),
            data_yaml=data_yaml_path,
            hyp=str(app_paths.yolo_hyperparameters_path),
            epochs=epochs,
            batch_size=batch_size,
            img_size=img_size,
            device=device,
            project=project_dir,
            name="yolo_train",
        )
    except Exception as exc:  # noqa: BLE001
        error_msg = f"训练过程中出错: {exc}"
        window.log(error_msg)
        QMessageBox.warning(window, "训练失败", error_msg)


def _split_yolo_dataset(window, temp_dir, annotated_images):
    shuffled_images = annotated_images.copy()
    random.shuffle(shuffled_images)
    total_annotated = len(shuffled_images)
    if total_annotated == 1:
        train_images = shuffled_images
        val_images = shuffled_images
        window.log("只有 1 张有标注的图片，将同时用于训练和验证")
    elif total_annotated <= 4:
        train_images = shuffled_images[1:]
        val_images = shuffled_images[:1]
        window.log(f"共有 {total_annotated} 张有标注的图片，{len(train_images)} 张训练，{len(val_images)} 张验证")
    else:
        val_size = max(1, int(total_annotated * 0.1))
        train_images = shuffled_images[val_size:]
        val_images = shuffled_images[:val_size]
        window.log(f"共有 {total_annotated} 张有标注的图片，{len(train_images)} 张训练，{len(val_images)} 张验证")

    train_images_dir = os.path.join(temp_dir, "images", "train")
    val_images_dir = os.path.join(temp_dir, "images", "val")
    train_labels_dir = os.path.join(temp_dir, "labels", "train")
    val_labels_dir = os.path.join(temp_dir, "labels", "val")
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    for image_path in train_images:
        shutil.copy(window._get_gray_path(image_path), train_images_dir)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(train_labels_dir, f"{base_name}.txt")
        with open(label_path, "w", encoding="utf-8") as handle:
            for cls, x, y, w, h in window.annotation_tool.annotations[image_path]:
                handle.write(f"{cls} {x} {y} {w} {h}\n")

    for image_path in val_images:
        shutil.copy(window._get_gray_path(image_path), val_images_dir)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(val_labels_dir, f"{base_name}.txt")
        with open(label_path, "w", encoding="utf-8") as handle:
            for cls, x, y, w, h in window.annotation_tool.annotations[image_path]:
                handle.write(f"{cls} {x} {y} {w} {h}\n")

    return train_images_dir, val_images_dir


def train_yolo(window):
    window.log("开始训练 YOLO 模型...")
    if not _has_annotation_training_data(window):
        return

    epochs, batch_size = _training_parameters(window)
    device = "0"

    annotated_images = _annotated_images(window)
    img_size = _optimal_img_size(annotated_images)
    window.log(f"自动识别的 img_size: {img_size}")

    project_dir = _choose_training_project_dir(window)
    if not project_dir:
        return

    _set_training_project_dirs(window, project_dir)

    if not _validate_yolo_annotations(window, annotated_images):
        return

    data_yaml_path = _prepare_yolo_data(window, annotated_images)
    window.log(f"训练参数: epochs={epochs}, batch_size={batch_size}, img_size={img_size}, device={device}")

    _prepare_training_ui(window)

    app_paths = _app_paths(window)
    yolo_model_path = window.train_yolo_model_path.text() or str(app_paths.default_yolo_model_path)

    window.log("YOLO 训练已开始，请查看终端输出")
    _ensure_yolo_loss_dialog(window)

    _start_background_training(
        lambda: _run_yolo_training_job(
            window,
            app_paths,
            yolo_model_path,
            data_yaml_path,
            epochs,
            batch_size,
            img_size,
            device,
            project_dir,
        )
    )


def _list_resnet_classes(resnet_data_path):
    class_names = []
    for item in os.listdir(resnet_data_path):
        item_path = os.path.join(resnet_data_path, item)
        if os.path.isdir(item_path):
            class_names.append(item)
    return class_names


def _fallback_split_classification_dataset(source_root, train_dir, val_dir, class_names):
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


def _fallback_project_split(window, source_root, project_dir, class_names):
    train_dir, val_dir = _split_dirs(project_dir)
    _fallback_split_classification_dataset(source_root, train_dir, val_dir, class_names)
    return train_dir + "/", val_dir + "/"


def _resnet_data_path(window):
    path_widget = getattr(window, "train_resnet_data_path", None)
    if not path_widget:
        return None
    data_path = path_widget.text()
    return data_path if data_path and os.path.exists(data_path) else None


def _has_resnet_source(window):
    if _resnet_data_path(window):
        return True
    return _has_annotation_training_data(window)


def _prepare_resnet_crop_dir(project_dir):
    crop_dir = os.path.join(project_dir, "resnet_crop")
    os.makedirs(crop_dir, exist_ok=True)
    return crop_dir


def _prepare_resnet_saving_dir(project_dir):
    saving_path = os.path.join(project_dir, "resnet_train")
    os.makedirs(saving_path, exist_ok=True)
    return saving_path


def _crop_annotation_boxes(window, crop_dir, selected_class_map):
    for class_name in selected_class_map.values():
        os.makedirs(os.path.join(crop_dir, str(class_name)), exist_ok=True)

    crop_count = 0
    actual_classes = set()
    for image_path in window.annotation_tool.images:
        annotations = window.annotation_tool.annotations.get(image_path)
        if not annotations:
            continue
        try:
            gray_path = window._get_gray_path(image_path)
            with Image.open(gray_path) as img:
                width, height = img.size
                for cls, x, y, w, h in annotations:
                    if cls not in selected_class_map:
                        continue
                    x1 = max(0, int((x - w / 2) * width))
                    y1 = max(0, int((y - h / 2) * height))
                    x2 = min(width, int((x + w / 2) * width))
                    y2 = min(height, int((y + h / 2) * height))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    crop = img.crop((x1, y1, x2, y2))
                    class_name = str(selected_class_map[cls])
                    actual_classes.add(class_name)
                    crop_path = os.path.join(crop_dir, class_name, f"crop_{crop_count}.jpg")
                    crop.save(crop_path)
                    crop_count += 1
        except Exception as exc:  # noqa: BLE001
            window.log(f"裁剪图片时出错: {exc}")
    return crop_count, actual_classes


def _prepare_resnet_from_annotations(window, project_dir):
    crop_dir = _prepare_resnet_crop_dir(project_dir)
    selected_class_indices = _selected_class_indices(window)

    window.log(f"用户选择的训练类别索引: {selected_class_indices}")

    class_names = _class_names_for_indices(window, selected_class_indices)
    window.log(f"当前训练类别: {class_names}")

    selected_class_map = {idx: name for idx, name in zip(selected_class_indices, class_names)}
    crop_count, actual_classes = _crop_annotation_boxes(window, crop_dir, selected_class_map)

    if crop_count == 0:
        window.log("错误: 没有成功裁剪任何标注区域")
        return None, None

    window.log(f"共裁剪了 {crop_count} 个标注区域")
    try:
        training_path, testing_path = _run_kfold_split(window, crop_dir)
        window.log("数据划分完成")
        return training_path, testing_path
    except Exception as exc:  # noqa: BLE001
        window.log(f"数据划分失败，使用默认划分方式: {exc}")
        return _fallback_project_split(window, crop_dir, project_dir, sorted(actual_classes))


def _prepare_resnet_paths(window, project_dir):
    resnet_data_path = _resnet_data_path(window)
    if resnet_data_path:
        window.log(f"使用 ResNet 格式数据: {resnet_data_path}")
        class_names = _list_resnet_classes(resnet_data_path)
        window.log(f"从数据目录检测到类别: {class_names}")
        try:
            return _run_kfold_split(window, resnet_data_path)
        except Exception as exc:  # noqa: BLE001
            window.log(f"数据划分失败，使用默认划分方式: {exc}")
            return _fallback_project_split(window, resnet_data_path, project_dir, class_names)

    return _prepare_resnet_from_annotations(window, project_dir)


def _run_resnet_training(window, training_path, testing_path, saving_path, epochs, batch_size):
    resnet_model_path = window.model_manager.resnet_model
    window.log("开始执行 ResNet 训练")
    resnet_module = importlib.import_module("ml.ResNet_train")

    class CustomStream:
        def __init__(self, callback):
            self.callback = callback
            self.buffer = []
            self.original_stdout = sys.stdout

        def write(self, text):
            if text is not None:
                self.buffer.append(text)
                if "\n" in text:
                    lines = "".join(self.buffer).split("\n")
                    for line in lines[:-1]:
                        if line.strip():
                            self.callback(line.strip())
                    self.buffer = [lines[-1]]
            if self.original_stdout:
                self.original_stdout.write(text)

        def flush(self):
            if self.original_stdout:
                self.original_stdout.flush()

        def getvalue(self):
            return "".join(self.buffer)

        def isatty(self):
            return self.original_stdout.isatty() if self.original_stdout else False

        def fileno(self):
            return self.original_stdout.fileno() if self.original_stdout else -1

    resnet_last_epoch = 0
    resnet_last_loss = None
    resnet_last_error = None

    def handle_resnet_log(line):
        nonlocal resnet_last_epoch, resnet_last_loss, resnet_last_error
        window.log(line)
        if "Epoch" not in line or "Training Loss" not in line:
            return
        try:
            import re

            loss_match = re.search(r"Training Loss:\s*([-\d.]+)", line)
            error_match = re.search(r"Prediction Error:\s*([-\d.]+)", line)
            epoch_match = re.search(r"Epoch\s+(\d+)", line)
            if not (loss_match and error_match and epoch_match):
                return
            train_loss = float(loss_match.group(1))
            pred_error = float(error_match.group(1))
            current_epoch = int(epoch_match.group(1))
            if current_epoch != resnet_last_epoch and resnet_last_loss is not None:
                window.resnet_loss_signal.emit(resnet_last_epoch, resnet_last_loss, resnet_last_error)
            resnet_last_epoch = current_epoch
            resnet_last_loss = train_loss
            resnet_last_error = pred_error
            if current_epoch > 0:
                window.training_progress_signal.emit(current_epoch, epochs)
        except Exception:  # noqa: BLE001
            return

    custom_stream = CustomStream(handle_resnet_log)
    original_stdout = sys.stdout
    original_argv = sys.argv.copy()
    sys.stdout = custom_stream
    try:
        enable_imbalance = window.imbalance_checkbox.isChecked()
        window.log(f"类别不平衡处理: {'启用' if enable_imbalance else '禁用'}")
        sys.argv = [
            "ResNet_train.py",
            "--training_path",
            training_path,
            "--testing_path",
            testing_path,
            "--saving_path",
            saving_path,
            "--epochs",
            str(epochs),
            "--batch_size",
            str(batch_size),
            "--pretrained_model",
            resnet_model_path,
            "--imbalance",
            str(enable_imbalance),
        ]
        resnet_module.main()
    finally:
        sys.argv = original_argv
        sys.stdout = original_stdout
        if custom_stream.buffer:
            remaining_text = "".join(custom_stream.buffer).strip()
            if remaining_text:
                window.log(remaining_text)

    if resnet_last_loss is not None:
        window.resnet_loss_signal.emit(resnet_last_epoch, resnet_last_loss, resnet_last_error)


def _run_resnet_training_job(window, training_path, testing_path, saving_path, epochs, batch_size):
    try:
        _run_resnet_training(window, training_path, testing_path, saving_path, epochs, batch_size)
        window.log("ResNet 模型训练完成")
        window.training_finished_signal.emit()
    except Exception as exc:  # noqa: BLE001
        import traceback

        error_msg = f"训练过程中出错: {exc}\n{traceback.format_exc()}"
        window.log(error_msg)
        window.training_error_signal.emit(str(exc))
        QMessageBox.warning(window, "训练失败", error_msg)
    finally:
        window.enable_controls()


def train_resnet(window):
    window.log("开始训练 ResNet 模型...")
    if not _has_resnet_source(window):
        return

    project_dir = _choose_training_project_dir(window)
    if not project_dir:
        return
    window._resnet_project_dir = project_dir

    training_path, testing_path = _prepare_resnet_paths(window, project_dir)
    if not training_path or not testing_path:
        return

    _prepare_training_ui(window)
    saving_path = _prepare_resnet_saving_dir(project_dir)
    epochs, batch_size = _training_parameters(window)
    _ensure_resnet_loss_dialog(window)
    _start_background_training(
        lambda: _run_resnet_training_job(
            window,
            training_path,
            testing_path,
            saving_path,
            epochs,
            batch_size,
        )
    )
    window.log("ResNet 训练已开始，请查看终端输出")
