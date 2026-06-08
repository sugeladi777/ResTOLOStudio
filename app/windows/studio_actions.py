from __future__ import annotations

import threading

from PyQt5.QtWidgets import QMessageBox

from nanonis.services import NanonisConnectionConfig


class StudioActionsMixin:
    def _nanonis_config(self) -> NanonisConnectionConfig:
        return NanonisConnectionConfig(
            ip=self.nano_ip_edit.text().strip(),
            port=int(self.nano_port_edit.text().strip()),
            version=int(self.nano_version_edit.text().strip()),
        )

    def _run_service(self, key: str, fn):
        def worker():
            try:
                result = fn()
                self.service_result_signal.emit(key, result)
            except Exception as exc:  # noqa: BLE001
                self.service_error_signal.emit(key, str(exc))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def connect_nanonis(self):
        self.log("连接 Nanonis...")
        self._run_service("connect_nanonis", lambda: self.nanonis_service.connect(self._nanonis_config()))

    def disconnect_nanonis(self):
        self.nanonis_service.disconnect()
        self.nano_status_text.setPlainText("未连接")
        self.log("已断开 Nanonis")

    def refresh_nanonis_status(self):
        self._run_service("refresh_nanonis", self.nanonis_service.status)

    def set_nanonis_bias(self):
        bias_v = float(self.scan_bias_edit.text().strip())
        self._run_service("set_bias", lambda: self.nanonis_service.set_bias(bias_v))

    def set_nanonis_setpoint(self):
        setpoint_a = float(self.scan_setpoint_edit.text().strip())
        self._run_service("set_setpoint", lambda: self.nanonis_service.set_setpoint(setpoint_a))

    def set_nanonis_feedback(self, enabled: bool):
        if not enabled and QMessageBox.question(self, "确认", "关闭 feedback 有风险，确认继续？") != QMessageBox.Yes:
            return
        self._run_service("set_feedback", lambda: self.nanonis_service.set_feedback(enabled))

    def _scan_channels(self) -> list[str]:
        return [part.strip() for part in self.scan_channels_edit.text().split(",") if part.strip()]

    def _scan_geometry(self) -> dict:
        return {
            "width_nm": float(self.scan_width_edit.text().strip()),
            "height_nm": float(self.scan_height_edit.text().strip()),
            "center_x_nm": 0.0,
            "center_y_nm": 0.0,
            "angle_deg": 0.0,
            "pixels": int(self.scan_pixels_edit.text().strip()),
            "channels": self._scan_channels(),
        }

    def _all_sessions(self) -> list[dict]:
        return self.result_store.list_sessions()

    def _selected_session(self) -> dict | None:
        sessions = self._all_sessions()
        session_index = self.session_list.currentRow()
        if session_index < 0 or session_index >= len(sessions):
            return None
        return sessions[session_index]

    def apply_nanonis_scan(self):
        self._run_service(
            "apply_scan",
            lambda: self.nanonis_service.apply_scan(**self._scan_geometry()),
        )

    def _new_session(self) -> dict:
        self.current_session = self.result_store.create_session(self.scan_label_edit.text().strip() or "session")
        self.nanonis_service.output_root = self.result_store.scan_dir(self.current_session["id"])
        return self.current_session

    def scan_and_save_from_nanonis(self):
        session = self._new_session()
        self.log(f"开始扫描，会话: {session['id']}")
        self._run_service(
            "scan_and_save",
            lambda: self.nanonis_service.scan_and_save(
                label=session["id"],
                **self._scan_geometry(),
            ),
        )

    def run_scan_pulse_scan_workflow(self):
        if QMessageBox.question(self, "确认", "即将执行预扫描、Bias Pulse 和后扫描，确认继续？") != QMessageBox.Yes:
            return
        session = self._new_session()
        self._run_service(
            "scan_pulse_scan",
            lambda: self.workflow_service.scan_pulse_scan(
                label=session["id"],
                **self._scan_geometry(),
                pulse_bias_v=float(self.pulse_bias_edit.text().strip()),
                pulse_width_s=float(self.pulse_width_edit.text().strip()),
            ),
        )

    def start_inference(self):
        if not self.model_manager.yolo_model or not self.model_manager.resnet_model:
            self.log("错误: 请先加载 YOLO 和 ResNet 模型")
            return
        if not self.inference_manager.images:
            self.log("错误: 请先加载推理图片")
            return
        if self.current_session is None:
            self._new_session()
        session_id = self.current_session["id"]
        output_dir = self.result_store.inference_dir(session_id)
        self.pending_inference_session_id = session_id
        self.pending_inference_output_dir = output_dir
        classes_yaml = self.infer_classes_path.text().strip()
        self.log(f"开始推理，会话: {session_id}")
        self.inference_service.infer_files(
            files=self.inference_manager.images,
            yolo_model=self.model_manager.yolo_model,
            resnet_model=self.model_manager.resnet_model,
            output_dir=output_dir,
            classes_yaml=classes_yaml,
        )
        self.progress_bar.show()
        self.progress_bar.setValue(0)
        self.disable_controls()

    def use_selected_result_for_inference(self):
        result_index = self.result_list.currentRow()
        session = self._selected_session()
        if session is None:
            return
        results = session.get("scan_results", [])
        if result_index < 0 or result_index >= len(results):
            return
        files = self.inference_service.scan_result_files(results[result_index])
        if not files:
            QMessageBox.warning(self, "无结果", "选中的扫描结果没有可用 PNG 文件。")
            return
        self.current_session = session
        self.inference_manager.load_images(files)
        self.annotation_tool.load_images(files)
        self.tab_widget.setCurrentIndex(2)
        self.log(f"已将 {len(files)} 张扫描结果导入推理")
        self.update_button_states()

    def on_inference_finished(self):
        super().on_inference_finished()
        if self.pending_inference_session_id and self.pending_inference_output_dir:
            results = self.inference_manager.get_inference_results(str(self.pending_inference_output_dir))
            self.result_store.append_inference_result(
                self.pending_inference_session_id,
                {
                    "output_dir": str(self.pending_inference_output_dir),
                    "images": results,
                    "actual_output_dir": self.inference_manager.actual_output_dir,
                },
            )
            self.reload_sessions()
            self.pending_inference_session_id = None
            self.pending_inference_output_dir = None

    def reload_sessions(self):
        sessions = self._all_sessions()
        self.session_list.clear()
        for session in sessions:
            self.session_list.addItem(
                f"{session['id']} | scan={len(session.get('scan_results', []))} | infer={len(session.get('inference_results', []))}"
            )
        if sessions:
            self.session_list.setCurrentRow(0)

    def _session_selected(self, index: int):
        self.result_list.clear()
        self.result_detail_text.clear()
        sessions = self._all_sessions()
        if index < 0 or index >= len(sessions):
            return
        session = sessions[index]
        for item in session.get("scan_results", []):
            self.result_list.addItem(item.get("label", "scan_result"))

    def _result_selected(self, index: int):
        session = self._selected_session()
        if session is None:
            return
        results = session.get("scan_results", [])
        if index < 0 or index >= len(results):
            return
        self.result_detail_text.setPlainText(str(results[index]))

    def _handle_service_result(self, key: str, result: object):
        if key in {"connect_nanonis", "refresh_nanonis", "set_bias", "set_setpoint", "set_feedback", "apply_scan"}:
            self.nano_status_text.setPlainText(str(result))
            self.log(f"{key} 完成")
            return
        if key == "scan_and_save":
            self.result_store.append_scan_result(self.current_session["id"], result)
            self.pending_scan_results = [result]
            self.reload_sessions()
            self.log(f"扫描完成: {result.get('label')}")
            return
        if key == "scan_pulse_scan":
            self.result_store.append_scan_result(self.current_session["id"], result["pre_scan"])
            self.result_store.append_scan_result(self.current_session["id"], result["post_scan"])
            self.pending_scan_results = [result["pre_scan"], result["post_scan"]]
            self.reload_sessions()
            self.log("工作流完成: 预扫描 -> Pulse -> 后扫描")

    def _handle_service_error(self, key: str, message: str):
        self.log(f"{key} 失败: {message}")
        QMessageBox.warning(self, "操作失败", f"{key} 失败:\n{message}")
