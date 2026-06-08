"""Legacy method bindings kept outside the base shell class."""

from app.legacy import workbench_data, workbench_runtime, workbench_state, workbench_training


def apply_workbench_bindings(workbench_cls):
    workbench_cls._convert_sxm_files = workbench_data.convert_sxm_files
    workbench_cls._on_sxm_color_toggle = workbench_data.set_sxm_color_mode
    workbench_cls._get_gray_path = workbench_data.get_gray_path
    workbench_cls.load_images = workbench_data.load_images
    workbench_cls.load_annotations = workbench_data.load_annotations
    workbench_cls.detect_and_display_classes = workbench_data.detect_and_display_classes
    workbench_cls.save_annotations = workbench_data.save_annotations
    workbench_cls.crop_resnet_dataset = workbench_data.crop_resnet_dataset
    workbench_cls.load_yolo_model = workbench_data.load_yolo_model
    workbench_cls.load_resnet_data = workbench_data.load_resnet_data
    workbench_cls.load_classes_yaml = workbench_data.load_classes_yaml
    workbench_cls.load_resnet_model = workbench_data.load_resnet_model
    workbench_cls.load_classes_file = workbench_data.load_classes_file
    workbench_cls.train_yolo = workbench_training.train_yolo
    workbench_cls.train_resnet = workbench_training.train_resnet
    workbench_cls.disable_controls = workbench_runtime.disable_controls
    workbench_cls.enable_controls = workbench_runtime.enable_controls
    workbench_cls.on_inference_error = workbench_runtime.on_inference_error
    workbench_cls.on_inference_finished = workbench_runtime.on_inference_finished
    workbench_cls.on_progress_updated = workbench_runtime.on_progress_updated
    workbench_cls.on_training_finished = workbench_runtime.on_training_finished
    workbench_cls.on_training_error = workbench_runtime.on_training_error
    workbench_cls.on_training_progress_updated = workbench_runtime.on_training_progress_updated
    workbench_cls.on_train_loss_updated = workbench_runtime.on_train_loss_updated
    workbench_cls.on_val_metrics_updated = workbench_runtime.on_val_metrics_updated
    workbench_cls.on_resnet_loss_updated = workbench_runtime.on_resnet_loss_updated
    workbench_cls.on_annotation_updated = workbench_runtime.on_annotation_updated
    workbench_cls.on_tab_changed = workbench_state.on_tab_changed
    workbench_cls.update_button_states = workbench_state.update_button_states
