from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHBoxLayout, QListWidget, QTextEdit, QToolButton, QVBoxLayout, QWidget


def _create_collapsible_section(title: str, content: QWidget, expanded: bool = False) -> QWidget:
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)

    toggle = QToolButton()
    toggle.setText(title)
    toggle.setCheckable(True)
    toggle.setChecked(expanded)
    toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
    toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
    content.setVisible(expanded)

    def _toggle(checked: bool) -> None:
        toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        content.setVisible(checked)

    toggle.toggled.connect(_toggle)
    layout.addWidget(toggle)
    layout.addWidget(content)
    return container


class StudioPanelsMixin:
    def _build_studio_tabs(self):
        self.tab_widget.addTab(self._build_acquisition_tab(), "采集")
        self.tab_widget.addTab(self._build_results_tab(), "结果")

    def _build_acquisition_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addWidget(self._build_connection_group())
        layout.addWidget(self._build_scan_group())
        layout.addWidget(self._build_workflow_group())
        layout.addStretch()
        return tab

    def _build_results_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.session_list = QListWidget()
        self.session_list.setMinimumHeight(140)
        self.result_list = QListWidget()
        self.result_list.setMinimumHeight(140)
        self.result_detail_text = QTextEdit()
        self.result_detail_text.setReadOnly(True)
        self.result_detail_text.setMaximumHeight(110)

        self.use_selected_result_btn = self.create_button("用于推理", self.use_selected_result_for_inference, accent=True)
        self.refresh_sessions_btn = self.create_button("刷新会话", self.reload_sessions)

        layout.addWidget(self.create_label("会话"))
        layout.addWidget(self.session_list)
        layout.addWidget(self.create_label("结果"))
        layout.addWidget(self.result_list)

        actions = QHBoxLayout()
        actions.setSpacing(4)
        actions.addWidget(self.use_selected_result_btn, 1)
        actions.addWidget(self.refresh_sessions_btn, 1)
        layout.addLayout(actions)

        detail_panel = QWidget()
        detail_layout = QVBoxLayout(detail_panel)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.setSpacing(0)
        detail_layout.addWidget(self.result_detail_text)
        layout.addWidget(_create_collapsible_section("详情", detail_panel, expanded=False))
        layout.addStretch()

        self.session_list.currentRowChanged.connect(self._session_selected)
        self.result_list.currentRowChanged.connect(self._result_selected)
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
        self.nano_status_text = QTextEdit()
        self.nano_status_text.setReadOnly(True)
        self.nano_status_text.setPlaceholderText("设备状态")
        self.nano_status_text.setMaximumHeight(72)

        layout.addWidget(self.create_label("IP"))
        layout.addWidget(self.nano_ip_edit)
        layout.addWidget(self.create_label("端口"))
        layout.addWidget(self.nano_port_edit)
        layout.addWidget(self.create_label("版本"))
        layout.addWidget(self.nano_version_edit)
        layout.addWidget(self.create_button("连接", self.connect_nanonis, accent=True))
        layout.addWidget(self.create_button("断开", self.disconnect_nanonis))
        layout.addWidget(self.create_button("刷新状态", self.refresh_nanonis_status))
        status_panel = QWidget()
        status_layout = QVBoxLayout(status_panel)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(0)
        status_layout.addWidget(self.nano_status_text)
        layout.addWidget(_create_collapsible_section("状态", status_panel, expanded=False))
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
        self.scan_pixels_edit = self.create_line_edit("256")
        self.scan_pixels_edit.setText("256")
        self.scan_channels_edit = self.create_line_edit("Z, 电流")

        layout.addWidget(self.create_label("会话标签"))
        layout.addWidget(self.scan_label_edit)
        layout.addWidget(self.create_label("偏压 (V)"))
        layout.addWidget(self.scan_bias_edit)
        layout.addWidget(self.create_button("设置偏压", self.set_nanonis_bias))
        layout.addWidget(self.create_label("电流 (A)"))
        layout.addWidget(self.scan_setpoint_edit)
        layout.addWidget(self.create_button("设置电流", self.set_nanonis_setpoint))
        layout.addWidget(self.create_button("反馈开", lambda: self.set_nanonis_feedback(True)))
        layout.addWidget(self.create_button("反馈关", lambda: self.set_nanonis_feedback(False)))
        compact_panel = QWidget()
        compact_layout = QVBoxLayout(compact_panel)
        compact_layout.setContentsMargins(0, 0, 0, 0)
        compact_layout.setSpacing(4)
        compact_layout.addWidget(self.create_label("宽度 (nm)"))
        compact_layout.addWidget(self.scan_width_edit)
        compact_layout.addWidget(self.create_label("高度 (nm)"))
        compact_layout.addWidget(self.scan_height_edit)
        compact_layout.addWidget(self.create_label("像素"))
        compact_layout.addWidget(self.scan_pixels_edit)
        compact_layout.addWidget(self.create_label("通道"))
        compact_layout.addWidget(self.scan_channels_edit)
        layout.addWidget(self.create_button("应用参数", self.apply_nanonis_scan))
        layout.addWidget(self.create_button("扫描并保存", self.scan_and_save_from_nanonis, accent=True))
        layout.addWidget(_create_collapsible_section("更多", compact_panel, expanded=False))
        return group

    def _build_workflow_group(self):
        group = self.create_group("实验流程")
        layout = group.layout()

        self.pulse_bias_edit = self.create_line_edit("2.5")
        self.pulse_bias_edit.setText("2.5")
        self.pulse_width_edit = self.create_line_edit("0.1")
        self.pulse_width_edit.setText("0.1")

        layout.addWidget(self.create_label("脉冲偏压 (V)"))
        layout.addWidget(self.pulse_bias_edit)
        layout.addWidget(self.create_label("脉冲宽度 (s)"))
        layout.addWidget(self.pulse_width_edit)
        layout.addWidget(self.create_button("执行预扫-脉冲-后扫", self.run_scan_pulse_scan_workflow))
        return group
