from __future__ import annotations

import os


def on_tab_changed(window, index):
    if hasattr(window, "annotation_tool") and window.annotation_tool:
        if index == 0:
            window.annotation_tool.set_annotation_mode(True)
            window.log("切换到标注模式")
        else:
            window.annotation_tool.set_annotation_mode(False)
            if index == 1:
                window.log("切换到训练模式")
            else:
                window.log("切换到推理模式")
                if hasattr(window.annotation_tool, "images") and window.annotation_tool.images:
                    window.inference_manager.load_images([window._get_gray_path(path) for path in window.annotation_tool.images])
    window.update_button_states()


def update_button_states(window):
    try:
        has_images = False
        has_annotations = False
        if hasattr(window, "annotation_tool") and window.annotation_tool:
            has_images = len(window.annotation_tool.images) > 0
            if has_images:
                for boxes in window.annotation_tool.annotations.values():
                    if len(boxes) > 0:
                        has_annotations = True
                        break

        has_yolo_model = hasattr(window.model_manager, "yolo_model") and window.model_manager.yolo_model is not None
        has_resnet_model = hasattr(window.model_manager, "resnet_model") and window.model_manager.resnet_model is not None
        has_infer_images = hasattr(window.inference_manager, "images") and len(window.inference_manager.images) > 0

        if hasattr(window, "load_annotations_btn"):
            window.load_annotations_btn.setEnabled(has_images)
        if hasattr(window, "save_annotations_btn"):
            window.save_annotations_btn.setEnabled(has_annotations)
        if hasattr(window, "crop_resnet_dataset_btn"):
            window.crop_resnet_dataset_btn.setEnabled(has_annotations)

        if hasattr(window, "train_load_images_btn"):
            window.train_load_images_btn.setEnabled(True)
        if hasattr(window, "train_load_annotations_btn"):
            window.train_load_annotations_btn.setEnabled(has_images)
        if hasattr(window, "train_load_yolo_model_btn"):
            window.train_load_yolo_model_btn.setEnabled(True)
        if hasattr(window, "train_load_resnet_model_btn"):
            window.train_load_resnet_model_btn.setEnabled(True)
        if hasattr(window, "train_yolo_btn"):
            window.train_yolo_btn.setEnabled(has_images and has_annotations and has_yolo_model)

        has_resnet_data = False
        if hasattr(window, "train_resnet_data_path"):
            resnet_data_path = window.train_resnet_data_path.text()
            has_resnet_data = bool(resnet_data_path and os.path.exists(resnet_data_path))

        if hasattr(window, "train_resnet_btn"):
            window.train_resnet_btn.setEnabled(has_resnet_model and (has_resnet_data or (has_images and has_annotations)))

        if hasattr(window, "infer_load_images_btn"):
            window.infer_load_images_btn.setEnabled(True)
        if hasattr(window, "infer_load_yolo_model_btn"):
            window.infer_load_yolo_model_btn.setEnabled(True)
        if hasattr(window, "infer_load_resnet_model_btn"):
            window.infer_load_resnet_model_btn.setEnabled(True)
        if hasattr(window, "start_inference_btn"):
            window.start_inference_btn.setEnabled(has_infer_images and has_yolo_model and has_resnet_model)
    except Exception as exc:  # noqa: BLE001
        if hasattr(window, "log"):
            window.log(f"更新按钮状态时出错: {exc}")
