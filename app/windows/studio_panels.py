from __future__ import annotations

from PyQt5.QtWidgets import QListWidget, QTextEdit, QVBoxLayout, QWidget


class StudioPanelsMixin:
    def _build_studio_tabs(self):
        self.tab_widget.addTab(self._build_acquisition_tab(), "采集控制")
        self.tab_widget.addTab(self._build_results_tab(), "结果中心")

    def _build_acquisition_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(self._build_connection_group())
        layout.addWidget(self._build_scan_group())
        layout.addWidget(self._build_workflow_group())
        layout.addStretch()
        return tab

    def _build_results_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.session_list = QListWidget()
        self.result_list = QListWidget()
        self.result_detail_text = QTextEdit()
        self.result_detail_text.setReadOnly(True)

        open_scan_btn = self.create_button("使用选中结果做推理", self.use_selected_result_for_inference)
        reload_btn = self.create_button("刷新会话", self.reload_sessions)

        layout.addWidget(self.create_label("实验会话"))
        layout.addWidget(self.session_list)
        layout.addWidget(self.create_label("扫描结果"))
        layout.addWidget(self.result_list)
        layout.addWidget(open_scan_btn)
        layout.addWidget(reload_btn)
        layout.addWidget(self.create_label("结果详情"))
        layout.addWidget(self.result_detail_text)

        self.session_list.currentRowChanged.connect(self._session_selected)
        self.result_list.currentRowChanged.connect(self._result_selected)
        return tab

    def _build_connection_group(self):
        group = self.create_group("Nanonis 连接")
        layout = group.layout()

        self.nano_ip_edit = self.create_line_edit("127.0.0.1")
        self.nano_ip_edit.setText("127.0.0.1")
        self.nano_port_edit = self.create_line_edit("6501")
        self.nano_port_edit.setText("6501")
        self.nano_version_edit = self.create_line_edit("10380")
        self.nano_version_edit.setText("10380")
        self.nano_status_text = QTextEdit()
        self.nano_status_text.setReadOnly(True)

        layout.addWidget(self.create_label("IP"))
        layout.addWidget(self.nano_ip_edit)
        layout.addWidget(self.create_label("端口"))
        layout.addWidget(self.nano_port_edit)
        layout.addWidget(self.create_label("版本"))
        layout.addWidget(self.nano_version_edit)
        layout.addWidget(self.create_button("连接", self.connect_nanonis))
        layout.addWidget(self.create_button("断开", self.disconnect_nanonis))
        layout.addWidget(self.create_button("刷新状态", self.refresh_nanonis_status))
        layout.addWidget(self.nano_status_text)
        return group

    def _build_scan_group(self):
        group = self.create_group("扫描采集")
        layout = group.layout()

        self.scan_label_edit = self.create_line_edit("session_label")
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
        self.scan_channels_edit = self.create_line_edit("Z (m), Current (A)")

        layout.addWidget(self.create_label("会话标签"))
        layout.addWidget(self.scan_label_edit)
        layout.addWidget(self.create_label("Bias (V)"))
        layout.addWidget(self.scan_bias_edit)
        layout.addWidget(self.create_button("设置 Bias", self.set_nanonis_bias))
        layout.addWidget(self.create_label("Setpoint (A)"))
        layout.addWidget(self.scan_setpoint_edit)
        layout.addWidget(self.create_button("设置 Setpoint", self.set_nanonis_setpoint))
        layout.addWidget(self.create_button("Feedback 开", lambda: self.set_nanonis_feedback(True)))
        layout.addWidget(self.create_button("Feedback 关", lambda: self.set_nanonis_feedback(False)))
        layout.addWidget(self.create_label("宽度 (nm)"))
        layout.addWidget(self.scan_width_edit)
        layout.addWidget(self.create_label("高度 (nm)"))
        layout.addWidget(self.scan_height_edit)
        layout.addWidget(self.create_label("像素"))
        layout.addWidget(self.scan_pixels_edit)
        layout.addWidget(self.create_label("通道"))
        layout.addWidget(self.scan_channels_edit)
        layout.addWidget(self.create_button("应用扫描参数", self.apply_nanonis_scan))
        layout.addWidget(self.create_button("扫描并保存", self.scan_and_save_from_nanonis, accent=True))
        return group

    def _build_workflow_group(self):
        group = self.create_group("半自动实验流")
        layout = group.layout()

        self.pulse_bias_edit = self.create_line_edit("2.5")
        self.pulse_bias_edit.setText("2.5")
        self.pulse_width_edit = self.create_line_edit("0.1")
        self.pulse_width_edit.setText("0.1")

        layout.addWidget(self.create_label("Pulse Bias (V)"))
        layout.addWidget(self.pulse_bias_edit)
        layout.addWidget(self.create_label("Pulse 宽度 (s)"))
        layout.addWidget(self.pulse_width_edit)
        layout.addWidget(self.create_button("执行预扫描-Pulse-后扫描", self.run_scan_pulse_scan_workflow))
        return group
