from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QScrollArea,
    QSplitter,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)

from app.core import AppPaths
from app.ui.annotation_tool import AnnotationTool
from app.utils.error_matcher import ErrorMatcher
from app.utils.inference_manager import InferenceManager
from app.utils.model_manager import ModelManager
from app.utils.training_manager import TrainingManager
from app.windows.studio_ui import BASE_COLOR, BORDER_COLOR, DARK_BG, MUTED_COLOR, PANEL_BG, TEXT_COLOR


def _create_collapsible_section(
    title: str,
    content: QWidget,
    expanded: bool = False,
    object_name: str = "",
) -> QWidget:
    container = QWidget()
    if object_name:
        container.setObjectName(object_name)
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)

    toggle = QToolButton()
    toggle.setText(title)
    toggle.setProperty("sectionToggle", "true")
    toggle.setCheckable(True)
    toggle.setChecked(expanded)
    toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
    toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
    content.setVisible(expanded)

    def _toggle(checked: bool) -> None:
        toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        content.setVisible(checked)

    toggle.toggled.connect(_toggle)
    container.toggle_button = toggle
    container.content_widget = content
    container.set_expanded = toggle.setChecked
    container.is_expanded = toggle.isChecked
    layout.addWidget(toggle)
    layout.addWidget(content)
    return container


def _create_status_group(window, object_name: str, *labels: QLabel) -> QWidget:
    group = window.create_group("当前状态")
    group.setObjectName(object_name)
    layout = QVBoxLayout()
    layout.setSpacing(4)
    for label in labels:
        label.setWordWrap(True)
        label.setProperty("uiState", "normal")
        layout.addWidget(label)
    group.layout().addLayout(layout)
    return group


def _add_widgets(layout, *widgets: QWidget) -> None:
    for widget in widgets:
        layout.addWidget(widget)


def _resource_row(window, label_text: str, path_widget: QWidget, button: QWidget) -> QWidget:
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)
    label = window.create_label(label_text)
    label.setMinimumWidth(78)
    label.setProperty("muted", "true")
    button.setText("更换")
    button.setMaximumWidth(88)
    layout.addWidget(label)
    layout.addWidget(path_widget, 1)
    layout.addWidget(button)
    return row


def _style_reference_group(group: QWidget) -> QWidget:
    group.setMinimumHeight(0)
    return group


def _hide_runtime_widget(widget: QWidget) -> None:
    widget.setVisible(False)
    widget.setMaximumHeight(0)


def _build_hidden_runtime_panel(window) -> QWidget:
    panel = QWidget()
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    for attr_name in ("workspace_mode_detail",):
        setattr(window, attr_name, QLabel(""))
    panel.hide()
    return panel


def _wrap_tab_with_scroll(content: QWidget) -> QScrollArea:
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setStyleSheet(
        f"""
        QScrollArea {{
            background-color: {PANEL_BG};
            border: none;
        }}
        QScrollArea > QWidget > QWidget {{
            background-color: {PANEL_BG};
        }}
        """
    )
    content.setStyleSheet("background-color: transparent;")
    scroll.setWidget(content)
    return scroll


def _build_annotation_tab(window) -> QWidget:
    tab = QWidget()
    layout = QVBoxLayout(tab)
    layout.setContentsMargins(12, 12, 12, 12)
    layout.setSpacing(10)

    data_group = _style_reference_group(window.create_group("数据管理"))
    data_layout = QGridLayout()
    data_layout.setHorizontalSpacing(6)
    data_layout.setVerticalSpacing(6)
    window.load_images_btn = window.create_button("加载图像", window.load_images)
    window.load_annotations_btn = window.create_button("加载标注", window.load_annotations)
    window.save_annotations_btn = window.create_button("保存标注", window.save_annotations)
    window.crop_resnet_dataset_btn = window.create_button("导出分类裁剪", window.crop_resnet_dataset)
    window.crop_resnet_dataset_btn.setToolTip("按当前标注导出 ResNet 分类训练裁剪图。")
    _hide_runtime_widget(window.crop_resnet_dataset_btn)
    data_layout.addWidget(window.load_images_btn, 0, 0)
    data_layout.addWidget(window.load_annotations_btn, 0, 1)
    data_layout.addWidget(window.save_annotations_btn, 1, 0, 1, 2)
    data_layout.addWidget(window.crop_resnet_dataset_btn, 2, 0, 1, 2)
    data_group.layout().addLayout(data_layout)
    layout.addWidget(data_group)

    window.launch_summary_text = QTextEdit()
    window.launch_summary_text.setReadOnly(True)
    window.launch_summary_text.setMinimumHeight(0)
    window.launch_summary_text.setMaximumHeight(0)
    window.launch_summary_text.setPlaceholderText("")
    window.style_readonly_summary(window.launch_summary_text)
    window.launch_summary_text.hide()
    layout.addWidget(window.launch_summary_text)

    window.annotation_status_detail = QLabel("")
    window.annotation_classes_detail = QLabel("")
    window.inference_input_status_detail = QLabel("")
    status_group = _create_status_group(window, "annotation_status_group", window.annotation_status_detail, window.annotation_classes_detail)
    layout.addWidget(status_group)
    layout.addStretch()
    return tab


def _build_training_classes_group(window) -> QWidget:
    group = window.create_group("选择训练类别")
    layout = QVBoxLayout()
    window.classes_scroll_area = QScrollArea()
    window.classes_scroll_area.setWidgetResizable(True)
    window.classes_scroll_area.setMinimumHeight(92)
    window.classes_scroll_area.setMaximumHeight(150)
    window.classes_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    window.classes_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
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
    window.classes_layout = QVBoxLayout(window.classes_container)
    window.classes_layout.setSpacing(6)
    window.classes_layout.setContentsMargins(8, 8, 8, 8)
    window.classes_hint_label = QLabel("请先加载标注文件以自动识别类别")
    window.classes_hint_label.setStyleSheet(f"color: {MUTED_COLOR};")
    window.classes_layout.addWidget(window.classes_hint_label)
    window.classes_scroll_area.setWidget(window.classes_container)

    layout.addWidget(window.classes_scroll_area)
    group.layout().addLayout(layout)
    window.class_checkboxes = []
    return group


def _build_training_tab(window) -> QWidget:
    tab = QWidget()
    layout = QVBoxLayout(tab)
    layout.setContentsMargins(12, 12, 12, 12)
    layout.setSpacing(10)

    data_group = _style_reference_group(window.create_group("训练数据"))
    data_layout = QGridLayout()
    data_layout.setHorizontalSpacing(6)
    data_layout.setVerticalSpacing(6)
    window.train_data_layout = data_layout
    window.train_load_images_btn = window.create_button("加载图像", window.load_images)
    window.train_load_annotations_btn = window.create_button("加载标注", window.load_annotations)
    window.train_load_resnet_data_btn = window.create_button("加载ResNet数据", window.load_resnet_data)
    window.train_resnet_data_path = window.create_line_edit("ResNet 数据路径")
    data_layout.addWidget(window.train_load_images_btn, 0, 0)
    data_layout.addWidget(window.train_load_annotations_btn, 0, 1)
    data_layout.addWidget(
        _resource_row(window, "外部分类集", window.train_resnet_data_path, window.train_load_resnet_data_btn),
        1,
        0,
        1,
        2,
    )
    data_group.layout().addLayout(data_layout)
    layout.addWidget(data_group)

    layout.addWidget(_build_training_classes_group(window))

    model_group = _style_reference_group(window.create_group("模型管理"))
    model_layout = QVBoxLayout()
    model_layout.setSpacing(8)
    window.train_model_layout = model_layout
    window.train_load_yolo_model_btn = window.create_button("加载检测模型(YOLO)", window.load_yolo_model)
    window.train_yolo_model_path = window.create_line_edit("检测模型路径")
    window.train_load_resnet_model_btn = window.create_button("加载分类模型(ResNet)", window.load_resnet_model)
    window.train_resnet_model_path = window.create_line_edit("分类模型路径")
    window.train_classes_path = window.create_line_edit("类别文件路径")
    _hide_runtime_widget(window.train_classes_path)
    model_layout.addWidget(
        _resource_row(window, "检测模型", window.train_yolo_model_path, window.train_load_yolo_model_btn)
    )
    model_layout.addWidget(
        _resource_row(window, "分类模型", window.train_resnet_model_path, window.train_load_resnet_model_btn)
    )
    model_layout.addWidget(window.train_classes_path)
    model_group.layout().addLayout(model_layout)

    params_group = window.create_group("训练参数")
    params_layout = QVBoxLayout()
    params_layout.setSpacing(8)
    yolo_epochs_layout = QHBoxLayout()
    yolo_epochs_layout.addWidget(window.create_label("Epochs:"))
    window.yolo_epochs_spin = QSpinBox()
    window.yolo_epochs_spin.setRange(1, 1000)
    window.yolo_epochs_spin.setValue(200)
    window.yolo_epochs_spin.setButtonSymbols(QSpinBox.NoButtons)
    yolo_epochs_layout.addWidget(window.yolo_epochs_spin)
    params_layout.addLayout(yolo_epochs_layout)
    yolo_batch_layout = QHBoxLayout()
    yolo_batch_layout.addWidget(window.create_label("Batch Size:"))
    window.yolo_batch_spin = QSpinBox()
    window.yolo_batch_spin.setRange(1, 128)
    window.yolo_batch_spin.setValue(4)
    window.yolo_batch_spin.setButtonSymbols(QSpinBox.NoButtons)
    yolo_batch_layout.addWidget(window.yolo_batch_spin)
    params_layout.addLayout(yolo_batch_layout)
    resnet_epochs_layout = QHBoxLayout()
    resnet_epochs_layout.addWidget(window.create_label("分类轮数"))
    window.resnet_epochs_spin = QSpinBox()
    window.resnet_epochs_spin.setRange(1, 1000)
    window.resnet_epochs_spin.setValue(300)
    window.resnet_epochs_spin.setButtonSymbols(QSpinBox.NoButtons)
    resnet_epochs_layout.addWidget(window.resnet_epochs_spin)
    params_layout.addLayout(resnet_epochs_layout)
    resnet_batch_layout = QHBoxLayout()
    resnet_batch_layout.addWidget(window.create_label("分类批大小"))
    window.resnet_batch_spin = QSpinBox()
    window.resnet_batch_spin.setRange(1, 128)
    window.resnet_batch_spin.setValue(16)
    window.resnet_batch_spin.setButtonSymbols(QSpinBox.NoButtons)
    resnet_batch_layout.addWidget(window.resnet_batch_spin)
    params_layout.addLayout(resnet_batch_layout)
    window.yolo_epochs_spin.valueChanged.connect(window.resnet_epochs_spin.setValue)
    window.yolo_batch_spin.valueChanged.connect(window.resnet_batch_spin.setValue)
    for index in range(params_layout.count() - 2, params_layout.count()):
        item = params_layout.itemAt(index)
        nested = item.layout() if item is not None else None
        if nested is None:
            continue
        for child_index in range(nested.count()):
            widget = nested.itemAt(child_index).widget()
            if widget is not None:
                _hide_runtime_widget(widget)
    imbalance_layout = QHBoxLayout()
    imbalance_layout.addWidget(window.create_label("处理类别不平衡"))
    window.imbalance_checkbox = QCheckBox("")
    window.imbalance_checkbox.setChecked(True)
    window.imbalance_checkbox.setToolTip("固定使用类别加权损失，不复制稀有类别样本。")
    window.imbalance_checkbox.setEnabled(False)
    imbalance_layout.addWidget(window.imbalance_checkbox)
    params_layout.addLayout(imbalance_layout)
    augment_layout = QHBoxLayout()
    augment_layout.addWidget(window.create_label("数据增强"))
    window.augment_checkbox = QCheckBox("")
    window.augment_checkbox.setChecked(False)
    augment_layout.addWidget(window.augment_checkbox)
    params_layout.addLayout(augment_layout)
    augment_label = augment_layout.itemAt(0).widget()
    if augment_label is not None:
        _hide_runtime_widget(augment_label)
    _hide_runtime_widget(window.augment_checkbox)
    params_group.layout().addLayout(params_layout)

    window.training_dataset_status_detail = QLabel("")
    window.training_model_status_detail = QLabel("")
    window.training_run_status_detail = QLabel("")
    status_group = _create_status_group(
        window,
        "training_status_group",
        window.training_dataset_status_detail,
        window.training_model_status_detail,
        window.training_run_status_detail,
    )
    layout.addWidget(status_group)

    control_group = window.create_group("开始训练")
    control_layout = QGridLayout()
    control_layout.setHorizontalSpacing(6)
    control_layout.setVerticalSpacing(6)
    window.train_yolo_btn = window.create_button("训练检测模型", window.train_yolo, accent=True)
    window.train_resnet_btn = window.create_button("训练分类模型", window.train_resnet)
    control_layout.addWidget(window.train_yolo_btn, 0, 0)
    control_layout.addWidget(window.train_resnet_btn, 0, 1)
    control_group.layout().addLayout(control_layout)
    layout.addWidget(control_group)

    advanced_content = QWidget()
    advanced_layout = QVBoxLayout(advanced_content)
    advanced_layout.setContentsMargins(0, 0, 0, 0)
    advanced_layout.setSpacing(8)
    advanced_layout.addWidget(model_group)
    advanced_layout.addWidget(params_group)
    window.training_advanced_section = _create_collapsible_section(
        "高级训练设置",
        advanced_content,
        expanded=False,
        object_name="training_advanced_section",
    )
    layout.addWidget(window.training_advanced_section)
    layout.addStretch()
    return tab


def _build_inference_tab(window) -> QWidget:
    tab = QWidget()
    layout = QVBoxLayout(tab)
    layout.setContentsMargins(12, 12, 12, 12)
    layout.setSpacing(10)

    data_group = _style_reference_group(window.create_group("推理数据"))
    data_layout = QVBoxLayout()
    data_layout.setSpacing(8)
    window.infer_load_images_btn = window.create_button("加载图像", window.load_infer_images, required=True)
    data_layout.addWidget(window.infer_load_images_btn)
    data_group.layout().addLayout(data_layout)
    layout.addWidget(data_group)

    model_group = _style_reference_group(window.create_group("模型管理"))
    model_layout = QVBoxLayout()
    model_layout.setSpacing(8)
    window.infer_model_layout = model_layout
    window.infer_load_yolo_model_btn = window.create_button("加载检测模型(YOLO)", window.load_yolo_model, required=True)
    window.infer_yolo_model_path = window.create_line_edit("YOLO 模型路径")
    window.infer_load_resnet_model_btn = window.create_button("加载分类模型(ResNet)", window.load_resnet_model, required=True)
    window.infer_resnet_model_path = window.create_line_edit("ResNet 模型路径")
    window.infer_load_classes_btn = window.create_button("加载类别文件", window.load_classes_file)
    window.infer_classes_path = window.create_line_edit("类别文件路径")
    model_layout.addWidget(
        _resource_row(window, "检测模型", window.infer_yolo_model_path, window.infer_load_yolo_model_btn)
    )
    model_layout.addWidget(
        _resource_row(window, "分类模型", window.infer_resnet_model_path, window.infer_load_resnet_model_btn)
    )
    model_layout.addWidget(
        _resource_row(window, "类别文件", window.infer_classes_path, window.infer_load_classes_btn)
    )
    model_group.layout().addLayout(model_layout)

    window.inference_batch_status_detail = QLabel("")
    window.inference_model_status_detail = QLabel("")
    window.inference_result_status_detail = QLabel("")
    status_group = _create_status_group(
        window,
        "inference_status_group",
        window.inference_input_status_detail,
        window.inference_batch_status_detail,
        window.inference_model_status_detail,
        window.inference_result_status_detail,
    )
    layout.addWidget(status_group)

    control_group = _style_reference_group(window.create_group("执行任务"))
    control_layout = QVBoxLayout()
    control_layout.setSpacing(8)
    window.start_inference_btn = window.create_button("开始推理", window.start_inference, accent=True)
    control_layout.addWidget(window.start_inference_btn)
    control_group.layout().addLayout(control_layout)
    layout.addWidget(control_group)

    window.inference_advanced_section = _create_collapsible_section(
        "高级模型设置",
        model_group,
        expanded=False,
        object_name="inference_advanced_section",
    )
    layout.addWidget(window.inference_advanced_section)
    layout.addStretch()
    return tab


def _build_right_panel(window) -> QVBoxLayout:
    layout = QVBoxLayout()
    layout.setContentsMargins(12, 12, 12, 12)
    layout.setSpacing(10)
    window.annotation_tool = AnnotationTool()
    window.annotation_tool.setMinimumHeight(520)
    layout.addWidget(window.annotation_tool, 1)

    toggle_row = QHBoxLayout()
    toggle_row.setContentsMargins(0, 0, 0, 0)
    toggle_row.addStretch()
    window.sxm_color_toggle = QCheckBox("SXM 彩色显示")
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
    toggle_row.addWidget(window.sxm_color_toggle)
    layout.addLayout(toggle_row)

    window.log_text = QTextEdit()
    window.log_text.setReadOnly(True)
    window.log_text.setMinimumHeight(90)
    window.log_text.setMaximumHeight(240)
    window.log_text.setPlaceholderText("日志输出区域...")
    window.style_readonly_summary(window.log_text)
    log_panel = QWidget()
    log_layout = QVBoxLayout(log_panel)
    log_layout.setContentsMargins(0, 0, 0, 0)
    log_layout.setSpacing(4)
    log_layout.addWidget(window.log_text)
    window.log_section = _create_collapsible_section(
        "运行日志 · 就绪",
        log_panel,
        expanded=False,
        object_name="log_section",
    )
    window.log_toggle = window.log_section.toggle_button
    window.log_toggle.toggled.connect(window._on_log_section_toggled)
    layout.addWidget(window.log_section)

    window.progress_bar = QProgressBar()
    window.progress_bar.setRange(0, 100)
    window.progress_bar.setValue(0)
    window.progress_bar.setTextVisible(True)
    window.progress_bar.setFormat("%p%")
    window.progress_bar.hide()
    layout.addWidget(window.progress_bar)

    layout.addWidget(_build_hidden_runtime_panel(window))
    return layout


def initialize_studio_shell(window) -> None:
    window.setWindowTitle("ReSTOLO - 分子检测与分类系统")
    window.setGeometry(120, 120, 1280, 800)
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
    window.tab_widget.setUsesScrollButtons(True)
    window.tab_widget.setElideMode(Qt.ElideNone)
    window.tab_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    window.tab_widget.addTab(_wrap_tab_with_scroll(_build_annotation_tab(window)), "标注")
    window.tab_widget.addTab(_wrap_tab_with_scroll(_build_training_tab(window)), "训练")
    window.tab_widget.addTab(_wrap_tab_with_scroll(_build_inference_tab(window)), "应用")
    window.tab_widget.currentChanged.connect(window.on_tab_changed)

    left_layout = QVBoxLayout()
    left_layout.setContentsMargins(0, 0, 0, 0)
    left_layout.setSpacing(0)
    left_layout.addWidget(window.tab_widget)
    left_layout.addStretch()

    right_layout = _build_right_panel(window)
    splitter = QSplitter(Qt.Horizontal)
    splitter.setChildrenCollapsible(False)
    splitter.setHandleWidth(12)
    left_widget = QWidget()
    left_widget.setLayout(left_layout)
    left_widget.setMinimumWidth(460)
    left_widget.setMaximumWidth(620)
    right_widget = QWidget()
    right_widget.setLayout(right_layout)
    splitter.addWidget(left_widget)
    splitter.addWidget(right_widget)
    splitter.setSizes([500, 900])
    window.main_splitter = splitter
    window.left_panel_widget = left_widget
    window.right_panel_widget = right_widget

    main_layout = QHBoxLayout(main_widget)
    main_layout.addWidget(splitter)

    window.annotation_tool.annotation_updated.connect(window.on_annotation_updated)
    window.update_button_states()
    window.on_tab_changed(0)


class StudioShellSignalsMixin:
    training_finished_signal = pyqtSignal()
    training_error_signal = pyqtSignal(str)
    log_signal = pyqtSignal(str)
    training_progress_signal = pyqtSignal(int, int)
    resnet_loss_signal = pyqtSignal(int, float, float)

    def log(self, message):
        self.log_signal.emit(message)

    def _save_ui_preference(self, key: str, value) -> None:
        config_service = getattr(self, "config_service", None)
        if config_service is None:
            return
        payload = dict(getattr(config_service, "data", {}) or {})
        if payload.get(key) == value:
            return
        payload[key] = value
        config_service.save(payload)

    def _on_log_section_toggled(self, expanded: bool) -> None:
        self._save_ui_preference("ui_log_expanded", bool(expanded))

    def set_log_expanded(self, expanded: bool) -> None:
        section = getattr(self, "log_section", None)
        if section is not None and hasattr(section, "set_expanded"):
            section.set_expanded(bool(expanded))

    def _log_slot(self, message):
        if hasattr(self, "log_text"):
            self.log_text.append(message)
        text = str(message).strip().splitlines()[0] if str(message).strip() else "就绪"
        if hasattr(self, "log_toggle"):
            summary = text if len(text) <= 28 else text[:27] + "…"
            self.log_toggle.setText(f"运行日志 · {summary}")
        auto_expand_tokens = ("失败", "错误", "开始训练", "开始推理", "训练设备", "Training Loss")
        if any(token in text for token in auto_expand_tokens):
            self.set_log_expanded(True)
