from __future__ import annotations

import os
import tempfile
from pathlib import Path

from PIL import Image
from PyQt5.QtWidgets import QCheckBox, QFileDialog

from app.legacy.workbench_ui import BASE_COLOR, BORDER_COLOR, PANEL_BG, TEXT_COLOR
from app.utils.sxm_parser import load_sxm_as_image


def convert_sxm_files(window, files):
    result_files = []
    window.sxm_metadata = {}
    window.sxm_original_paths = {}
    window.sxm_color_paths = {}
    window._has_sxm_files = False

    for file_path in files:
        if not file_path.lower().endswith(".sxm"):
            result_files.append(file_path)
            continue

        window._has_sxm_files = True
        try:
            img_gray, sxm = load_sxm_as_image(file_path, use_color=False)
            img_color, _ = load_sxm_as_image(file_path, use_color=True)
            if not img_gray:
                continue

            tmp_dir = tempfile.mkdtemp(prefix="restolo_sxm_")
            basename = Path(file_path).stem
            gray_path = os.path.join(tmp_dir, f"{basename}_gray.png")
            color_path = os.path.join(tmp_dir, f"{basename}_color.png")
            img_gray.save(gray_path)
            if img_color:
                img_color.save(color_path)

            result_files.append(gray_path)
            window.sxm_original_paths[file_path] = gray_path
            window.sxm_color_paths[file_path] = color_path
            px_x, px_y = sxm.get_pixel_size_nm()
            window.sxm_metadata[gray_path] = {
                "original_path": file_path,
                "scan_range_nm": sxm.get_scan_range_nm(),
                "pixel_size_nm": sxm.get_pixel_size_nm(),
                "channels": sxm.get_channel_names(),
                "summary": sxm.get_metadata_summary(),
                "gray_path": gray_path,
                "color_path": color_path,
            }
            window.log(
                f"SXM文件已转换: {os.path.basename(file_path)} "
                f"(扫描范围: {sxm.get_scan_range_nm()[0]:.2f}x{sxm.get_scan_range_nm()[1]:.2f} nm, "
                f"像素尺寸: {px_x:.4f}x{px_y:.4f} nm/pixel)"
            )
        except Exception as exc:  # noqa: BLE001
            window.log(f"SXM文件加载失败 {os.path.basename(file_path)}: {exc}")

    if hasattr(window, "sxm_color_toggle"):
        window.sxm_color_toggle.setVisible(window._has_sxm_files)

    return result_files


def set_sxm_color_mode(window, state):
    use_color = bool(state)
    if not getattr(window, "sxm_metadata", None):
        return

    path_map = {}
    for gray_path, meta in window.sxm_metadata.items():
        color_path = meta.get("color_path", "")
        if use_color and color_path and os.path.exists(color_path):
            path_map[gray_path] = color_path
            path_map[color_path] = color_path
        else:
            path_map[gray_path] = gray_path
            path_map[color_path] = gray_path

    if hasattr(window, "annotation_tool") and hasattr(window.annotation_tool, "images"):
        updated = False
        for index, img_path in enumerate(window.annotation_tool.images):
            if img_path in path_map:
                new_path = path_map[img_path]
                if new_path != img_path:
                    window.annotation_tool.images[index] = new_path
                    updated = True

        if hasattr(window.annotation_tool, "annotations"):
            old_keys = [key for key in window.annotation_tool.annotations if key in path_map]
            for old_key in old_keys:
                new_key = path_map[old_key]
                if new_key != old_key:
                    window.annotation_tool.annotations[new_key] = window.annotation_tool.annotations.pop(old_key)

        if updated and hasattr(window.annotation_tool, "current_index"):
            window.annotation_tool.show_current_image()

    window.log(f"SXM显示模式: {'彩色' if use_color else '灰度'}")


def get_gray_path(window, image_path):
    for gray_path, meta in getattr(window, "sxm_metadata", {}).items():
        if image_path == meta.get("color_path"):
            return gray_path
    return image_path


def load_images(window):
    files, _ = QFileDialog.getOpenFileNames(window, "选择图片", "", "Image files (*.jpg *.jpeg *.png *.bmp *.sxm)")
    if not files:
        return
    files = convert_sxm_files(window, files)
    window.log(f"加载了 {len(files)} 张图片")
    window.annotation_tool.load_images(files)
    window.update_button_states()


def load_annotations(window):
    files, _ = QFileDialog.getOpenFileNames(window, "选择标注文件", "", "Text files (*.txt)")
    if not files:
        return
    window.log(f"加载了 {len(files)} 个标注文件")
    window.annotation_tool.load_annotations(files)
    window.update_button_states()
    detect_and_display_classes(window)


def detect_and_display_classes(window):
    annotations = getattr(window.annotation_tool, "annotations", None)
    if not annotations:
        return

    used_classes = set()
    for boxes in annotations.values():
        for box in boxes:
            used_classes.add(box[0])
    if not used_classes:
        return

    max_class = max(used_classes)
    class_names = []
    for index in range(max_class + 1):
        if hasattr(window.annotation_tool, "class_names") and index < len(window.annotation_tool.class_names):
            class_names.append(window.annotation_tool.class_names[index])
        else:
            class_names.append(str(index))

    for checkbox in window.class_checkboxes:
        checkbox.deleteLater()
    window.class_checkboxes.clear()
    window.class_indices = []

    if getattr(window, "classes_hint_label", None):
        window.classes_hint_label.deleteLater()
        window.classes_hint_label = None

    for index, class_name in enumerate(class_names):
        if index not in used_classes:
            continue
        checkbox = QCheckBox(class_name)
        checkbox.setChecked(True)
        checkbox.setStyleSheet(
            f"""
                QCheckBox {{
                    color: {TEXT_COLOR};
                    spacing: 8px;
                    background-color: transparent;
                }}
                QCheckBox::indicator {{
                    width: 16px;
                    height: 16px;
                    border-radius: 3px;
                    border: 2px solid {BORDER_COLOR};
                    background-color: {PANEL_BG};
                }}
                QCheckBox::indicator:checked {{
                    background-color: {BASE_COLOR};
                    border-color: {BASE_COLOR};
                }}
            """
        )
        window.classes_layout.addWidget(checkbox)
        window.class_checkboxes.append(checkbox)
        window.class_indices.append(index)

    window.log(f"自动识别到 {len(window.class_checkboxes)} 个有标注的类别")


def save_annotations(window):
    directory = QFileDialog.getExistingDirectory(window, "选择保存目录")
    if not directory:
        return
    window.annotation_tool.save_annotations(directory)
    window.log(f"标注已保存到 {directory}")
    window.update_button_states()


def crop_resnet_dataset(window):
    directory = QFileDialog.getExistingDirectory(window, "选择保存目录")
    if not directory:
        return

    window.log("开始裁剪 ResNet 训练集...")
    if not getattr(window.annotation_tool, "images", None):
        window.log("错误: 请先加载训练图片")
        return
    if not getattr(window.annotation_tool, "annotations", None):
        window.log("错误: 请先加载或创建标注")
        return

    used_classes = set()
    for annotations in window.annotation_tool.annotations.values():
        for box in annotations:
            used_classes.add(box[0])

    if used_classes:
        max_class = max(used_classes)
        class_names = [str(index) for index in range(max_class + 1)]
        window.log(f"使用自适应类别: {class_names}")
    else:
        class_names = window.annotation_tool.class_names
        window.log(f"没有标注数据，使用默认类别: {class_names}")

    actual_classes = set()
    for image_path in window.annotation_tool.images:
        if image_path not in window.annotation_tool.annotations:
            continue
        for box in window.annotation_tool.annotations[image_path]:
            cls = box[0]
            if cls < len(class_names):
                actual_classes.add(str(class_names[cls]))

    window.log(f"实际有标注的类别: {list(actual_classes)}")
    for class_name in actual_classes:
        os.makedirs(os.path.join(directory, class_name), exist_ok=True)

    crop_count = 0
    for image_path in window.annotation_tool.images:
        if image_path not in window.annotation_tool.annotations:
            continue
        try:
            img = Image.open(image_path)
            width, height = img.size
            for box in window.annotation_tool.annotations[image_path]:
                cls, x, y, w, h = box
                x1 = max(0, int((x - w / 2) * width))
                y1 = max(0, int((y - h / 2) * height))
                x2 = min(width, int((x + w / 2) * width))
                y2 = min(height, int((y + h / 2) * height))
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = img.crop((x1, y1, x2, y2))
                class_name = str(class_names[cls])
                crop_path = os.path.join(directory, class_name, f"crop_{crop_count}.jpg")
                crop.save(crop_path)
                crop_count += 1
        except Exception as exc:  # noqa: BLE001
            window.log(f"裁剪图片时出错: {exc}")

    window.log(f"ResNet训练集裁剪完成，共裁剪了 {crop_count} 个标注区域")


def load_yolo_model(window):
    file_path, _ = QFileDialog.getOpenFileName(window, "选择检测模型", "", "Model files (*.pt *.pth)")
    if not file_path:
        return
    window.model_manager.load_yolo_model(file_path)
    window.log(f"加载YOLO模型: {file_path}")
    if hasattr(window, "train_yolo_model_path"):
        window.train_yolo_model_path.setText(file_path)
    if hasattr(window, "infer_yolo_model_path"):
        window.infer_yolo_model_path.setText(file_path)
    window.update_button_states()


def load_resnet_data(window):
    directory = QFileDialog.getExistingDirectory(window, "选择ResNet格式数据目录")
    if not directory:
        return
    window.log(f"加载ResNet格式数据: {directory}")
    if hasattr(window, "train_resnet_data_path"):
        window.train_resnet_data_path.setText(directory)
    try:
        classes = os.listdir(directory)
        num_classes = len([item for item in classes if os.path.isdir(os.path.join(directory, item))])
        window.log(f"检测到 {num_classes} 个类别")
    except Exception as exc:  # noqa: BLE001
        window.log(f"验证数据目录失败: {exc}")
    window.update_button_states()


def load_classes_yaml(window):
    file_path, _ = QFileDialog.getOpenFileName(window, "选择类别YAML文件", "", "YAML files (*.yaml)")
    if not file_path:
        return
    window.log(f"加载类别YAML文件: {file_path}")
    if hasattr(window, "train_classes_path"):
        window.train_classes_path.setText(file_path)
    if hasattr(window, "infer_classes_path"):
        window.infer_classes_path.setText(file_path)
    window.update_button_states()


def load_resnet_model(window):
    file_path, _ = QFileDialog.getOpenFileName(window, "选择分类模型", "", "Model files (*.saving *.pt *.pth)")
    if not file_path:
        return
    window.model_manager.load_resnet_model(file_path)
    window.log(f"加载ResNet模型: {file_path}")
    if hasattr(window, "train_resnet_model_path"):
        window.train_resnet_model_path.setText(file_path)
    if hasattr(window, "infer_resnet_model_path"):
        window.infer_resnet_model_path.setText(file_path)
    window.update_button_states()


def load_classes_file(window):
    load_classes_yaml(window)
