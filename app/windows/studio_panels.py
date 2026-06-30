from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
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

    def _preview_caption(self, text: str = "") -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet(f"color: {MUTED_COLOR};")
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

    def _field_row(self, label_text: str, field: QWidget) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        label = self.create_label(label_text)
        label.setMinimumWidth(92)
        layout.addWidget(label)
        layout.addWidget(field, 1)
        return row

    def _two_field_row(self, left_label: str, left_field: QWidget, right_label: str, right_field: QWidget) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self._field_row(left_label, left_field), 1)
        layout.addWidget(self._field_row(right_label, right_field), 1)
        return row

    def _build_acquisition_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        layout.addWidget(
            self._build_info_banner(
                "采集工作台",
                "保留最常用的扫描设置：几何参数、偏压和设定电流。修改后直接点击“扫描并保存”，程序会自动下发参数并执行扫描。",
            )
        )
        layout.addWidget(self._build_connection_group())
        layout.addWidget(self._build_scan_group())
        layout.addWidget(_create_collapsible_section("高级流程", self._build_workflow_group(), expanded=False))
        layout.addStretch()
        return tab

    def _build_results_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        toolbar = QWidget()
        toolbar_layout = QVBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.setSpacing(6)
        self.session_search_edit = QLineEdit()
        self.session_search_edit.setPlaceholderText("搜索会话标签、阶段或结果数量")
        toolbar_layout.addWidget(self.session_search_edit)
        layout.addWidget(toolbar)

        self.session_list = QListWidget()
        self.session_list.setMinimumHeight(150)
        self.session_list.setWordWrap(True)
        self.session_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.session_list.setTextElideMode(Qt.ElideNone)
        self.session_list.setStyleSheet("font-size: 17px;")

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
        self.session_browser_context_text = self._styled_readonly_text(min_height=72, max_height=96, placeholder="这里会显示会话摘要。")
        session_layout.addWidget(self.session_browser_context_text)
        layout.addWidget(session_group)

        self.result_list = QListWidget()
        self.result_list.setMinimumHeight(180)
        self.result_list.setWordWrap(True)
        self.result_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.result_list.setTextElideMode(Qt.ElideNone)
        self.result_list.setStyleSheet("font-size: 17px;")
        self.result_detail_text = self._styled_readonly_text(min_height=144, max_height=220, placeholder="选择扫描结果后，这里会显示扫描参数、导出文件和来源信息。")
        self.result_detail_text.setStyleSheet(self.result_detail_text.styleSheet() + "QTextEdit { font-size: 16px; line-height: 1.5; }")
        self.result_preview_label = self._preview_label("尚未选择结果", minimum_height=220)
        self.result_preview_caption = self._preview_caption("选择扫描结果后，这里会显示预览图。")
        self.result_compare_combo = QComboBox()
        self.result_compare_combo.addItem("不比较")
        self.result_compare_summary_text = self._styled_readonly_text(min_height=96, max_height=120, placeholder="这里会显示对比摘要。")
        self.result_compare_preview_label = self._preview_label("尚未选择对比项", minimum_height=200)
        self.result_compare_preview_caption = self._preview_caption("选择对比项后，这里会显示对比预览。")

        self.use_selected_result_btn = self.create_button("用于推理", self.use_selected_result_for_inference)
        self.open_result_dir_btn = self.create_button("打开结果目录", self.open_selected_result_directory)
        self.refresh_sessions_btn = self.create_button("刷新会话", self.reload_sessions)

        result_group = self.create_group("扫描结果")
        result_layout = result_group.layout()
        result_layout.addWidget(self.result_list)
        result_actions = QGridLayout()
        result_actions.setHorizontalSpacing(6)
        result_actions.setVerticalSpacing(6)
        result_actions.addWidget(self.use_selected_result_btn, 0, 0)
        result_actions.addWidget(self.open_result_dir_btn, 0, 1)
        result_actions.addWidget(self.refresh_sessions_btn, 1, 0, 1, 2)
        result_layout.addLayout(result_actions)
        result_layout.addWidget(self.result_detail_text)
        result_layout.addWidget(self.result_preview_label)
        result_layout.addWidget(self.result_preview_caption)
        result_layout.addWidget(self.create_label("结果对比"))
        result_layout.addWidget(self.result_compare_combo)
        result_layout.addWidget(self.result_compare_summary_text)
        result_layout.addWidget(self.result_compare_preview_label)
        result_layout.addWidget(self.result_compare_preview_caption)
        layout.addWidget(result_group)
        layout.addStretch()

        self.session_list.currentRowChanged.connect(self._session_selected)
        self.result_list.currentRowChanged.connect(self._result_selected)
        self.result_compare_combo.currentIndexChanged.connect(self._result_comparison_changed)
        self.session_search_edit.textChanged.connect(lambda _text: self.reload_sessions())
        return tab

    def _build_connection_group(self):
        group = self.create_group("设备连接")
        layout = group.layout()

        self.nano_ip_edit = self.create_line_edit("127.0.0.1")
        self.nano_ip_edit.setText("127.0.0.1")
        self.nano_port_edit = self.create_line_edit("6501")
        self.nano_port_edit.setText("6501")
        self.nano_version_edit = self.create_line_edit("10380")
        self.nano_version_edit.setText("10380")

        button_grid = QGridLayout()
        button_grid.setHorizontalSpacing(6)
        button_grid.setVerticalSpacing(6)
        button_grid.addWidget(self.create_button("连接", self.connect_nanonis), 0, 0)
        button_grid.addWidget(self.create_button("断开", self.disconnect_nanonis), 0, 1)
        button_grid.addWidget(self.create_button("刷新状态", self.refresh_nanonis_status), 0, 2)

        self.nano_status_text = self._styled_readonly_text(min_height=60, max_height=80, placeholder="连接后，这里会显示设备状态与最新反馈。")
        layout.addWidget(self._two_field_row("IP", self.nano_ip_edit, "端口", self.nano_port_edit))
        layout.addWidget(self._field_row("版本", self.nano_version_edit))
        layout.addLayout(button_grid)
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

        summary_banner = self._build_info_banner(
            "一键扫描",
            "点击“扫描并保存”后，会自动设置偏压、设定电流和扫描范围，然后开始扫描并把结果写入当前会话。",
        )

        primary_panel = QWidget()
        primary_layout = QVBoxLayout(primary_panel)
        primary_layout.setContentsMargins(0, 0, 0, 0)
        primary_layout.setSpacing(6)
        primary_layout.addWidget(self._field_row("会话标签", self.scan_label_edit))
        primary_layout.addWidget(self._two_field_row("偏压 (V)", self.scan_bias_edit, "电流 (A)", self.scan_setpoint_edit))
        primary_layout.addWidget(self._field_row("通道", self.scan_channels_edit))

        geometry_panel = QWidget()
        geometry_layout = QVBoxLayout(geometry_panel)
        geometry_layout.setContentsMargins(0, 0, 0, 0)
        geometry_layout.setSpacing(6)
        geometry_layout.addWidget(self._two_field_row("宽度 (nm)", self.scan_width_edit, "高度 (nm)", self.scan_height_edit))
        geometry_layout.addWidget(self._two_field_row("中心 X", self.scan_center_x_edit, "中心 Y", self.scan_center_y_edit))
        geometry_layout.addWidget(self._field_row("像素", self.scan_pixels_edit))
        self.scan_angle_hint = QLabel("自动沿用当前设备角度")
        self.scan_angle_hint.setWordWrap(True)
        self.scan_angle_hint.setStyleSheet(f"color: {MUTED_COLOR};")
        geometry_layout.addWidget(self.scan_angle_hint)

        action_row = QGridLayout()
        action_row.setHorizontalSpacing(6)
        action_row.setVerticalSpacing(6)
        action_row.addWidget(self.create_button("扫描并保存", self.scan_and_save_from_nanonis, accent=True), 0, 0)
        action_row.addWidget(self.create_button("执行预扫-脉冲-后扫", self.run_scan_pulse_scan_workflow), 0, 1)

        layout.addWidget(summary_banner)
        layout.addWidget(primary_panel)
        layout.addWidget(_create_collapsible_section("几何参数", geometry_panel, expanded=True))
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
        return group

    def _panel_from_layout(self, child_layout) -> QWidget:
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(0)
        panel_layout.addLayout(child_layout)
        return panel
