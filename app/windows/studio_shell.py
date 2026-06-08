from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QScrollArea,
    QSplitter,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.core import AppPaths
from app.ui.annotation_tool import AnnotationTool
from app.utils.error_matcher import ErrorMatcher
from app.utils.inference_manager import InferenceManager
from app.utils.model_manager import ModelManager
from app.utils.training_manager import TrainingManager
from app.windows.studio_ui import BASE_COLOR, BORDER_COLOR, DARK_BG, MUTED_COLOR, PANEL_BG, TEXT_COLOR


def _build_annotation_tab(window):
    tab = QWidget()
    layout = QVBoxLayout(tab)
    layout.setContentsMargins(15, 15, 15, 15)
    layout.setSpacing(12)

    data_group = window.create_group("数据管理")
    data_layout = QVBoxLayout()
    data_layout.setSpacing(10)
    window.load_images_btn = window.create_button("加载图片", window.load_images)
    window.load_annotations_btn = window.create_button("加载标注", window.load_annotations)
    window.save_annotations_btn = window.create_button("保存标注", window.save_annotations)
    data_layout.addWidget(window.load_images_btn)
    data_layout.addWidget(window.load_annotations_btn)
    data_layout.addWidget(window.save_annotations_btn)
    data_group.layout().addLayout(data_layout)
    layout.addWidget(data_group)
    layout.addStretch()
    return tab


def _build_training_classes_group(window):
    group = window.create_group("选择训练类别")
    layout = QVBoxLayout()

    window.classes_scroll_area = QScrollArea()
    window.classes_scroll_area.setWidgetResizable(True)
    window.classes_scroll_area.setMaximumHeight(80)
    window.classes_scroll_area.setStyleSheet(
        f"""
            QScrollArea {{
                border: 1px solid {BORDER_COLOR};
                border-radius: 6px;
                background-color: {DARK_BG};
            }}
        """
    )

    window.classes_container = QWidget()
    window.classes_container.setStyleSheet(f"background-color: {DARK_BG};")
    window.classes_layout = QHBoxLayout(window.classes_container)
    window.classes_layout.setSpacing(12)
    window.classes_layout.setContentsMargins(5, 5, 5, 5)
    window.classes_hint_label = QLabel("请先加载标注文件以自动识别类别")
    window.classes_hint_label.setStyleSheet(f"color: {MUTED_COLOR};")
    window.classes_layout.addWidget(window.classes_hint_label)
    window.classes_scroll_area.setWidget(window.classes_container)

    layout.addWidget(window.classes_scroll_area)
    group.layout().addLayout(layout)
    window.class_checkboxes = []
    return group


def _build_training_tab(window):
    tab = QWidget()
    layout = QVBoxLayout(tab)
    layout.setContentsMargins(15, 15, 15, 15)
    layout.setSpacing(12)

    data_group = window.create_group("训练数据")
    data_layout = QVBoxLayout()
    data_layout.setSpacing(10)
    window.train_load_images_btn = window.create_button("加载图片", window.load_images)
    window.train_load_annotations_btn = window.create_button("加载标注", window.load_annotations)
    window.train_load_resnet_data_btn = window.create_button("加载ResNet数据", window.load_resnet_data)
    window.train_resnet_data_path = window.create_line_edit("ResNet数据路径")
    data_layout.addWidget(window.train_load_images_btn)
    data_layout.addWidget(window.train_load_annotations_btn)
    data_layout.addWidget(window.train_load_resnet_data_btn)
    data_layout.addWidget(window.train_resnet_data_path)
    data_group.layout().addLayout(data_layout)
    layout.addWidget(data_group)

    layout.addWidget(_build_training_classes_group(window))

    model_group = window.create_group("模型管理")
    model_layout = QVBoxLayout()
    model_layout.setSpacing(10)
    window.train_load_yolo_model_btn = window.create_button("加载检测模型(YOLO)", window.load_yolo_model)
    window.train_yolo_model_path = window.create_line_edit("检测模型路径")
    window.train_load_resnet_model_btn = window.create_button("加载分类模型(ResNet)", window.load_resnet_model)
    window.train_resnet_model_path = window.create_line_edit("分类模型路径")
    model_layout.addWidget(window.train_load_yolo_model_btn)
    model_layout.addWidget(window.train_yolo_model_path)
    model_layout.addWidget(window.train_load_resnet_model_btn)
    model_layout.addWidget(window.train_resnet_model_path)
    model_group.layout().addLayout(model_layout)
    layout.addWidget(model_group)

    params_group = window.create_group("训练参数")
    params_layout = QVBoxLayout()
    params_layout.setSpacing(10)

    epochs_layout = QHBoxLayout()
    epochs_label = window.create_label("Epochs:")
    epochs_layout.addWidget(epochs_label)
    window.epochs_spin = QSpinBox()
    window.epochs_spin.setRange(1, 1000)
    window.epochs_spin.setValue(300)
    window.epochs_spin.setButtonSymbols(QSpinBox.NoButtons)
    epochs_layout.addWidget(window.epochs_spin)
    params_layout.addLayout(epochs_layout)

    batch_layout = QHBoxLayout()
    batch_label = window.create_label("Batch Size:")
    batch_layout.addWidget(batch_label)
    window.batch_spin = QSpinBox()
    window.batch_spin.setRange(1, 128)
    window.batch_spin.setValue(16)
    window.batch_spin.setButtonSymbols(QSpinBox.NoButtons)
    batch_layout.addWidget(window.batch_spin)
    params_layout.addLayout(batch_layout)

    imbalance_layout = QHBoxLayout()
    imbalance_label = window.create_label("处理类别不平衡:")
    imbalance_layout.addWidget(imbalance_label)
    window.imbalance_checkbox = QCheckBox("")
    window.imbalance_checkbox.setChecked(True)
    imbalance_layout.addWidget(window.imbalance_checkbox)
    params_layout.addLayout(imbalance_layout)

    params_group.layout().addLayout(params_layout)
    layout.addWidget(params_group)

    control_group = window.create_group("训练控制")
    control_layout = QVBoxLayout()
    control_layout.setSpacing(10)
    window.train_yolo_btn = window.create_button("训练YOLO", window.train_yolo)
    window.train_resnet_btn = window.create_button("训练ResNet", window.train_resnet)
    control_layout.addWidget(window.train_yolo_btn)
    control_layout.addWidget(window.train_resnet_btn)
    control_group.layout().addLayout(control_layout)
    layout.addWidget(control_group)
    layout.addStretch()
    return tab


def _build_inference_tab(window):
    tab = QWidget()
    layout = QVBoxLayout(tab)
    layout.setContentsMargins(15, 15, 15, 15)
    layout.setSpacing(12)

    data_group = window.create_group("推理数据")
    data_layout = QVBoxLayout()
    data_layout.setSpacing(10)
    window.infer_load_images_btn = window.create_button("加载图片", window.load_infer_images, required=True)
    data_layout.addWidget(window.infer_load_images_btn)
    data_group.layout().addLayout(data_layout)
    layout.addWidget(data_group)

    model_group = window.create_group("模型管理")
    model_layout = QVBoxLayout()
    model_layout.setSpacing(10)
    window.infer_load_yolo_model_btn = window.create_button("加载检测模型(YOLO)", window.load_yolo_model, required=True)
    window.infer_yolo_model_path = window.create_line_edit("YOLO模型路径")
    window.infer_load_resnet_model_btn = window.create_button("加载分类模型(ResNet)", window.load_resnet_model, required=True)
    window.infer_resnet_model_path = window.create_line_edit("ResNet模型路径")
    window.infer_load_classes_btn = window.create_button("加载类别文件", window.load_classes_file)
    window.infer_classes_path = window.create_line_edit("classes.yaml 文件路径")
    model_layout.addWidget(window.infer_load_yolo_model_btn)
    model_layout.addWidget(window.infer_yolo_model_path)
    model_layout.addWidget(window.infer_load_resnet_model_btn)
    model_layout.addWidget(window.infer_resnet_model_path)
    model_layout.addWidget(window.infer_load_classes_btn)
    model_layout.addWidget(window.infer_classes_path)
    model_group.layout().addLayout(model_layout)
    layout.addWidget(model_group)

    control_group = window.create_group("推理控制")
    control_layout = QVBoxLayout()
    control_layout.setSpacing(10)
    window.start_inference_btn = window.create_button("开始推理", window.start_inference)
    control_layout.addWidget(window.start_inference_btn)
    control_group.layout().addLayout(control_layout)
    layout.addWidget(control_group)
    layout.addStretch()
    return tab


def _build_right_panel(window):
    layout = QVBoxLayout()
    layout.setContentsMargins(15, 15, 15, 15)
    layout.setSpacing(12)

    window.annotation_tool = AnnotationTool()
    layout.addWidget(window.annotation_tool, 1)

    sxm_toggle_layout = QHBoxLayout()
    sxm_toggle_layout.addStretch()
    window.sxm_color_toggle = QCheckBox("SXM彩色显示")
    window.sxm_color_toggle.setChecked(False)
    window.sxm_color_toggle.setVisible(False)
    window.sxm_color_toggle.setStyleSheet(
        f"""
            QCheckBox {{
                color: {TEXT_COLOR};
                spacing: 6px;
                background-color: transparent;
            }}
            QCheckBox::indicator {{
                width: 14px;
                height: 14px;
                border-radius: 3px;
                border: 1px solid {BORDER_COLOR};
                background-color: {PANEL_BG};
            }}
            QCheckBox::indicator:checked {{
                background-color: {BASE_COLOR};
                border-color: {BASE_COLOR};
            }}
        """
    )
    window.sxm_color_toggle.stateChanged.connect(window._on_sxm_color_toggle)
    sxm_toggle_layout.addWidget(window.sxm_color_toggle)
    layout.addLayout(sxm_toggle_layout)

    log_layout = QVBoxLayout()
    log_layout.setSpacing(8)
    log_title = window.create_label("运行日志")
    log_title.setStyleSheet("font-weight: bold;")
    log_layout.addWidget(log_title)
    window.log_text = QTextEdit()
    window.log_text.setReadOnly(True)
    window.log_text.setPlaceholderText("日志输出区域...")
    log_layout.addWidget(window.log_text)
    layout.addLayout(log_layout)

    window.progress_bar = QProgressBar()
    window.progress_bar.setRange(0, 100)
    window.progress_bar.setValue(0)
    window.progress_bar.setTextVisible(True)
    window.progress_bar.setFormat("%p%")
    layout.addWidget(window.progress_bar)
    window.progress_bar.hide()
    return layout


def initialize_studio_shell(window):
    window.setWindowTitle("ReSTOLO Studio")
    window.setGeometry(100, 100, 1200, 800)

    window.apply_styles()
    window.model_manager = ModelManager()

    paths = AppPaths.from_project_root(Path(__file__).resolve().parents[2])
    window.error_matcher = ErrorMatcher(str(paths.error_patterns_path))
    window.inference_manager = InferenceManager()
    window.training_manager = TrainingManager()

    window.inference_manager.set_log_callback(window.log)
    window.training_manager.set_log_callback(window.log)
    window.inference_manager.set_finished_callback(window.on_inference_finished)
    window.training_manager.set_finished_callback(window.on_training_finished)
    window.inference_manager.set_error_callback(window.on_inference_error)
    window.training_manager.set_error_callback(window.on_training_error)
    window.inference_manager.progress_signal.connect(window.on_progress_updated)
    window.training_manager.progress_signal.connect(window.on_training_progress_updated)
    window.log_signal.connect(window._log_slot)
    window.training_progress_signal.connect(window.on_training_progress_updated)
    window.training_manager.train_loss_signal.connect(window.on_train_loss_updated)
    window.training_manager.val_metrics_signal.connect(window.on_val_metrics_updated)
    window.resnet_loss_signal.connect(window.on_resnet_loss_updated)
    window.inference_success = False

    main_widget = QWidget()
    window.setCentralWidget(main_widget)

    window.tab_widget = QTabWidget()
    window.tab_widget.tabBar().setChangeCurrentOnDrag(False)
    window.tab_widget.setUsesScrollButtons(False)

    window.tab_widget.addTab(_build_annotation_tab(window), "标注模式")
    window.tab_widget.addTab(_build_training_tab(window), "训练模式")
    window.tab_widget.addTab(_build_inference_tab(window), "推理模式")
    window.tab_widget.currentChanged.connect(window.on_tab_changed)
    window.update_button_states()

    left_layout = QVBoxLayout()
    left_layout.addWidget(window.tab_widget)
    left_layout.addStretch()

    right_layout = _build_right_panel(window)

    splitter = QSplitter(Qt.Horizontal)
    left_widget = QWidget()
    left_widget.setLayout(left_layout)
    right_widget = QWidget()
    right_widget.setLayout(right_layout)
    splitter.addWidget(left_widget)
    splitter.addWidget(right_widget)
    splitter.setSizes([300, 900])

    main_layout = QHBoxLayout(main_widget)
    main_layout.addWidget(splitter)

    window.annotation_tool.annotation_updated.connect(window.on_annotation_updated)
    window.on_tab_changed(0)


class StudioShellSignalsMixin:
    training_finished_signal = pyqtSignal()
    training_error_signal = pyqtSignal(str)
    log_signal = pyqtSignal(str)
    training_progress_signal = pyqtSignal(int, int)
    resnet_loss_signal = pyqtSignal(int, float, float)

    def log(self, message):
        self.log_signal.emit(message)

    def _log_slot(self, message):
        if hasattr(self, "log_text"):
            self.log_text.append(message)
