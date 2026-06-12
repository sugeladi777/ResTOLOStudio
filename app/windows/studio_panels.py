from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.windows.studio_shell import _create_collapsible_section, _wrap_tab_with_scroll
from app.windows.studio_ui import BORDER_COLOR, DARK_BG, MUTED_COLOR, PANEL_BG, TEXT_COLOR


class StudioPanelsMixin:
    def _build_studio_tabs(self):
        self.tab_widget.insertTab(0, _wrap_tab_with_scroll(self._build_acquisition_tab()), "采集")
        self.tab_widget.addTab(_wrap_tab_with_scroll(self._build_results_tab()), "结果")

    def _styled_readonly_text(self, *, min_height: int = 0, max_height: int | None = None, placeholder: str = "") -> QTextEdit:
        text = QTextEdit()
        text.setReadOnly(True)
        text.setPlaceholderText(placeholder)
        if min_height:
            text.setMinimumHeight(min_height)
        if max_height is not None:
            text.setMaximumHeight(max_height)
        self.style_readonly_summary(text)
        return text

    def _preview_label(self, empty_text: str, minimum_height: int = 180) -> QLabel:
        label = QLabel(empty_text)
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumHeight(minimum_height)
        label.setWordWrap(True)
        label.setStyleSheet(
            f"""
            QLabel {{
                border: 1px solid {BORDER_COLOR};
                border-radius: 2px;
                background-color: {DARK_BG};
                color: {MUTED_COLOR};
                padding: 12px;
            }}
            """
        )
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return label

    def _build_info_banner(self, title: str, description: str) -> QWidget:
        panel = QFrame()
        panel.setStyleSheet(
            f"""
            QFrame {{
                border: 1px solid {BORDER_COLOR};
                border-radius: 2px;
                background-color: {PANEL_BG};
            }}
            """
        )
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(4)

        title_label = QLabel(title)
        title_label.setStyleSheet(f"font-weight: 600; color: {TEXT_COLOR};")
        description_label = QLabel(description)
        description_label.setWordWrap(True)
        description_label.setStyleSheet(f"color: {MUTED_COLOR};")

        layout.addWidget(title_label)
        layout.addWidget(description_label)
        return panel

    def _build_acquisition_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        layout.addWidget(
            self._build_info_banner(
                "采集工作台",
                "先连接设备并确认状态，再设置扫描参数。单次扫描会写入当前会话，预扫-脉冲-后扫会按同一批次连续保存。",
            )
        )
        layout.addWidget(self._build_connection_group())
        layout.addWidget(self._build_scan_group())
        layout.addWidget(self._build_workflow_group())
        layout.addStretch()
        return tab

    def _build_results_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.setSpacing(6)

        self.session_search_edit = QLineEdit()
        self.session_search_edit.setPlaceholderText("搜索会话标签、阶段或结果数量")
        self.session_sort_combo = QComboBox()
        self.session_sort_combo.addItems(
            [
                "最近活跃优先",
                "扫描结果最多",
                "训练结果最多",
                "推理结果最多",
            ]
        )
        toolbar_layout.addWidget(self.session_search_edit, 3)
        toolbar_layout.addWidget(self.session_sort_combo, 2)
        layout.addWidget(toolbar)

        self.session_browser_context_text = self._styled_readonly_text(
            min_height=64,
            max_height=88,
            placeholder="这里会显示当前会话概览。",
        )
        layout.addWidget(self.session_browser_context_text)

        self.session_list = QListWidget()
        self.session_list.setMinimumHeight(150)

        self.new_session_btn = self.create_button("新建会话", self.create_session)
        self.activate_session_btn = self.create_button("设为当前", self.activate_selected_session)
        self.rename_session_btn = self.create_button("重命名", self.rename_selected_session)
        self.open_session_dir_btn = self.create_button("打开会话目录", self.open_selected_session_directory)

        session_group = self.create_group("会话")
        session_layout = session_group.layout()
        session_layout.addWidget(self.session_list)
        session_actions = QGridLayout()
        session_actions.setHorizontalSpacing(6)
        session_actions.setVerticalSpacing(6)
        session_actions.addWidget(self.new_session_btn, 0, 0)
        session_actions.addWidget(self.activate_session_btn, 0, 1)
        session_actions.addWidget(self.rename_session_btn, 1, 0)
        session_actions.addWidget(self.open_session_dir_btn, 1, 1)
        session_layout.addLayout(session_actions)
        layout.addWidget(session_group)

        self.result_list = QListWidget()
        self.result_list.setMinimumHeight(180)
        self.result_detail_text = self._styled_readonly_text(
            min_height=120,
            max_height=180,
            placeholder="选择扫描结果后，这里会显示扫描参数、导出文件和来源信息。",
        )

        self.use_selected_result_btn = self.create_button("用于推理", self.use_selected_result_for_inference, accent=True)
        self.open_result_dir_btn = self.create_button("打开结果目录", self.open_selected_result_directory)
        self.refresh_sessions_btn = self.create_button("刷新会话", self.reload_sessions)

        result_group = self.create_group("扫描结果")
        result_layout = result_group.layout()
        result_layout.addWidget(self.result_list)
        result_actions = QHBoxLayout()
        result_actions.setSpacing(6)
        result_actions.addWidget(self.use_selected_result_btn, 2)
        result_actions.addWidget(self.open_result_dir_btn, 2)
        result_actions.addWidget(self.refresh_sessions_btn, 1)
        result_layout.addLayout(result_actions)
        result_layout.addWidget(self.result_detail_text)
        layout.addWidget(result_group)

        self.result_preview_label = self._preview_label("选择扫描结果后，这里会显示带扫描信息的预览。", minimum_height=220)
        self.result_preview_caption = QLabel("当前预览：未选择结果")
        self.result_preview_caption.setStyleSheet(f"color: {MUTED_COLOR};")

        preview_group = self.create_group("当前预览")
        preview_layout = preview_group.layout()
        preview_layout.addWidget(self.result_preview_label)
        preview_layout.addWidget(self.result_preview_caption)
        layout.addWidget(preview_group)

        compare_panel = QWidget()
        compare_layout = QVBoxLayout(compare_panel)
        compare_layout.setContentsMargins(0, 0, 0, 0)
        compare_layout.setSpacing(6)

        compare_toolbar = QHBoxLayout()
        compare_toolbar.setContentsMargins(0, 0, 0, 0)
        compare_toolbar.setSpacing(6)
        compare_toolbar.addWidget(self.create_label("对比对象"))
        self.result_compare_combo = QComboBox()
        self.result_compare_combo.addItem("不进行对比")
        compare_toolbar.addWidget(self.result_compare_combo, 1)
        compare_layout.addLayout(compare_toolbar)

        self.result_compare_summary_text = self._styled_readonly_text(
            min_height=72,
            max_height=96,
            placeholder="选择对比对象后，这里会显示摘要对比。",
        )
        compare_layout.addWidget(self.result_compare_summary_text)

        self.result_compare_preview_label = self._preview_label("未选择对比结果", minimum_height=180)
        self.result_compare_preview_caption = QLabel("对比预览：未选择结果")
        self.result_compare_preview_caption.setStyleSheet(f"color: {MUTED_COLOR};")
        compare_layout.addWidget(self.result_compare_preview_label)
        compare_layout.addWidget(self.result_compare_preview_caption)

        layout.addWidget(_create_collapsible_section("结果对比", compare_panel, expanded=False))
        layout.addStretch()

        self.session_list.currentRowChanged.connect(self._session_selected)
        self.result_list.currentRowChanged.connect(self._result_selected)
        self.result_compare_combo.currentIndexChanged.connect(self._result_comparison_changed)
        self.session_search_edit.textChanged.connect(lambda _text: self.reload_sessions())
        self.session_sort_combo.currentIndexChanged.connect(lambda _index: self.reload_sessions())
        return tab

    def _build_connection_group(self):
        group = self.create_group("设备连接")
        layout = group.layout()

        form = QGridLayout()
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(6)

        self.nano_ip_edit = self.create_line_edit("127.0.0.1")
        self.nano_ip_edit.setText("127.0.0.1")
        self.nano_port_edit = self.create_line_edit("6501")
        self.nano_port_edit.setText("6501")
        self.nano_version_edit = self.create_line_edit("10380")
        self.nano_version_edit.setText("10380")

        form.addWidget(self.create_label("IP"), 0, 0)
        form.addWidget(self.nano_ip_edit, 0, 1)
        form.addWidget(self.create_label("端口"), 0, 2)
        form.addWidget(self.nano_port_edit, 0, 3)
        form.addWidget(self.create_label("版本"), 1, 0)
        form.addWidget(self.nano_version_edit, 1, 1)

        button_row = QHBoxLayout()
        button_row.setSpacing(6)
        button_row.addWidget(self.create_button("连接", self.connect_nanonis))
        button_row.addWidget(self.create_button("断开", self.disconnect_nanonis))
        button_row.addWidget(self.create_button("刷新状态", self.refresh_nanonis_status))
        button_row.addStretch()

        self.nano_status_text = self._styled_readonly_text(
            min_height=78,
            max_height=96,
            placeholder="连接后，这里会显示设备状态与最新反馈。",
        )

        layout.addLayout(form)
        layout.addLayout(button_row)
        layout.addWidget(self.nano_status_text)
        return group

    def _build_scan_group(self):
        group = self.create_group("扫描参数")
        layout = group.layout()

        self.scan_label_edit = self.create_line_edit("实验标签")
        self.scan_bias_edit = self.create_line_edit("2.0")
        self.scan_bias_edit.setText("2.0")
        self.scan_setpoint_edit = self.create_line_edit("5.0e-11")
        self.scan_setpoint_edit.setText("5.0e-11")
        self.scan_width_edit = self.create_line_edit("5")
        self.scan_width_edit.setText("5")
        self.scan_height_edit = self.create_line_edit("5")
        self.scan_height_edit.setText("5")
        self.scan_center_x_edit = self.create_line_edit("0")
        self.scan_center_x_edit.setText("0")
        self.scan_center_y_edit = self.create_line_edit("0")
        self.scan_center_y_edit.setText("0")
        self.scan_pixels_edit = self.create_line_edit("256")
        self.scan_pixels_edit.setText("256")
        self.scan_channels_edit = self.create_line_edit("Z, 电流")

        primary_form = QFormLayout()
        primary_form.setHorizontalSpacing(10)
        primary_form.setVerticalSpacing(6)
        primary_form.addRow("会话标签", self.scan_label_edit)
        primary_form.addRow("偏压 (V)", self.scan_bias_edit)
        primary_form.addRow("设定电流 (A)", self.scan_setpoint_edit)
        primary_form.addRow("通道", self.scan_channels_edit)

        geometry_grid = QGridLayout()
        geometry_grid.setHorizontalSpacing(8)
        geometry_grid.setVerticalSpacing(6)
        geometry_grid.addWidget(self.create_label("宽度 (nm)"), 0, 0)
        geometry_grid.addWidget(self.scan_width_edit, 0, 1)
        geometry_grid.addWidget(self.create_label("高度 (nm)"), 0, 2)
        geometry_grid.addWidget(self.scan_height_edit, 0, 3)
        geometry_grid.addWidget(self.create_label("中心 X (nm)"), 1, 0)
        geometry_grid.addWidget(self.scan_center_x_edit, 1, 1)
        geometry_grid.addWidget(self.create_label("中心 Y (nm)"), 1, 2)
        geometry_grid.addWidget(self.scan_center_y_edit, 1, 3)
        geometry_grid.addWidget(self.create_label("像素"), 2, 0)
        geometry_grid.addWidget(self.scan_pixels_edit, 2, 1)

        controls_grid = QGridLayout()
        controls_grid.setHorizontalSpacing(6)
        controls_grid.setVerticalSpacing(6)
        controls_grid.addWidget(self.create_button("设置偏压", self.set_nanonis_bias), 0, 0)
        controls_grid.addWidget(self.create_button("设置电流", self.set_nanonis_setpoint), 0, 1)
        controls_grid.addWidget(self.create_button("反馈开启", lambda: self.set_nanonis_feedback(True)), 1, 0)
        controls_grid.addWidget(self.create_button("反馈关闭", lambda: self.set_nanonis_feedback(False)), 1, 1)

        action_row = QHBoxLayout()
        action_row.setSpacing(6)
        action_row.addWidget(self.create_button("应用参数", self.apply_nanonis_scan))
        action_row.addWidget(self.create_button("扫描并保存", self.scan_and_save_from_nanonis, accent=True))

        layout.addLayout(primary_form)
        layout.addWidget(_create_collapsible_section("几何参数", self._panel_from_layout(geometry_grid), expanded=True))
        layout.addWidget(_create_collapsible_section("设备控制", self._panel_from_layout(controls_grid), expanded=False))
        layout.addLayout(action_row)
        return group

    def _build_workflow_group(self):
        group = self.create_group("实验流程")
        layout = group.layout()

        self.pulse_bias_edit = self.create_line_edit("2.5")
        self.pulse_bias_edit.setText("2.5")
        self.pulse_width_edit = self.create_line_edit("0.1")
        self.pulse_width_edit.setText("0.1")

        form = QFormLayout()
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(6)
        form.addRow("脉冲偏压 (V)", self.pulse_bias_edit)
        form.addRow("脉冲宽度 (s)", self.pulse_width_edit)

        layout.addLayout(form)
        layout.addWidget(
            self._build_info_banner(
                "半自动流程",
                "执行预扫、脉冲、后扫的连续流程，适合快速对比脉冲前后的表面变化。",
            )
        )
        layout.addWidget(self.create_button("执行预扫-脉冲-后扫", self.run_scan_pulse_scan_workflow, accent=True))
        return group

    def _panel_from_layout(self, child_layout) -> QWidget:
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(0)
        panel_layout.addLayout(child_layout)
        return panel
