from __future__ import annotations

import os

from PyQt5.QtWidgets import QLineEdit, QMessageBox, QPushButton

from app.legacy.workbench_ui import DARK_BG


def disable_controls(window):
    window.tab_widget.setEnabled(False)
    for widget in window.findChildren(QPushButton):
        widget.setEnabled(False)
    for widget in window.findChildren(QLineEdit):
        widget.setEnabled(False)


def enable_controls(window):
    window.tab_widget.setEnabled(True)
    for widget in window.findChildren(QPushButton):
        widget.setEnabled(True)
    for widget in window.findChildren(QLineEdit):
        widget.setEnabled(True)
    window.update_button_states()


def on_inference_error(window, error_msg):
    match_result = window.error_matcher.format_error(error_msg)
    if match_result:
        message, category, suggestion = match_result
        error_display = f"[{category}] {message}"
        if suggestion:
            error_display += f"\n\n建议: {suggestion}"
    else:
        error_display = "推理失败，请查看日志输出"
    window.log(error_display)
    QMessageBox.warning(window, "推理失败", error_display)
    enable_controls(window)


def on_inference_finished(window):
    window.log("推理完成")
    if hasattr(window, "output_dir"):
        actual_output_dir = getattr(window.inference_manager, "actual_output_dir", None)
        if actual_output_dir and os.path.exists(actual_output_dir):
            result_images = window.inference_manager.get_inference_results(window.output_dir)
            if result_images:
                window.log(f"找到 {len(result_images)} 张推理结果图片")
                if hasattr(window, "annotation_tool") and window.annotation_tool:
                    window.annotation_tool.load_images(result_images)
                    window.log("推理结果已显示在图片框中")
            else:
                window.log("未找到推理结果图片")
        else:
            window.log("未找到推理结果目录，推理可能未成功完成")
    enable_controls(window)


def on_progress_updated(window, current, total):
    if total > 0:
        window.progress_bar.setValue(int((current / total) * 100))


def on_training_finished(window):
    window.log("训练完成")
    window.progress_bar.hide()
    enable_controls(window)
    if hasattr(window, "loss_curve_dialog") and window.loss_curve_dialog:
        save_dir = None
        if getattr(window, "_yolo_project_dir", None):
            save_dir = os.path.join(window._yolo_project_dir, "yolo_train")
        elif getattr(window, "_resnet_project_dir", None):
            save_dir = os.path.join(window._resnet_project_dir, "resnet_train")
        if save_dir and os.path.exists(save_dir):
            try:
                save_path = os.path.join(save_dir, "loss_curves.png")
                window.loss_curve_dialog.figure.savefig(save_path, dpi=150, facecolor=DARK_BG)
                window.log(f"Loss 曲线已保存到: {save_path}")
            except Exception as exc:  # noqa: BLE001
                window.log(f"保存 Loss 曲线失败: {exc}")
        window.loss_curve_dialog.force_close()


def on_training_error(window, error_msg):
    match_result = window.error_matcher.format_error(error_msg)
    if match_result:
        message, category, suggestion = match_result
        error_display = f"[{category}] {message}"
        if suggestion:
            error_display += f"\n\n建议: {suggestion}"
    else:
        error_display = f"训练失败: {error_msg}"
    window.log(error_display)
    window.progress_bar.hide()
    QMessageBox.warning(window, "训练失败", error_display)
    enable_controls(window)


def on_training_progress_updated(window, current, total):
    try:
        if total > 0 and hasattr(window, "progress_bar"):
            progress = min(100, max(0, int((current / total) * 100)))
            window.progress_bar.setValue(progress)
            window.log(f"训练进度: Epoch {current}/{total}")
    except Exception as exc:  # noqa: BLE001
        window.log(f"更新进度时出错: {exc}")


def on_train_loss_updated(window, epoch, box, obj, cls, total):
    if hasattr(window, "loss_curve_dialog") and window.loss_curve_dialog:
        window.loss_curve_dialog.update_train_loss(epoch, box, obj, cls, total)


def on_val_metrics_updated(window, epoch, precision, recall, map50, map50_95):
    if hasattr(window, "loss_curve_dialog") and window.loss_curve_dialog:
        window.loss_curve_dialog.update_val_metrics(epoch, precision, recall, map50, map50_95)


def on_resnet_loss_updated(window, epoch, train_loss, pred_error):
    if hasattr(window, "loss_curve_dialog") and window.loss_curve_dialog:
        window.loss_curve_dialog.update_resnet_loss(epoch, train_loss, pred_error)


def on_annotation_updated(window):
    window.log("标注已更新")
    window.update_button_states()
    window.detect_and_display_classes()
