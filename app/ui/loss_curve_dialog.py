from PyQt5.QtWidgets import QDialog, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

DARK_BG = "#1A1A1A"
PANEL_BG = "#252525"
TEXT_COLOR = "#E8E8E8"
BORDER_COLOR = "#404040"
BASE_COLOR = "#5AE54A"


class LossCurveDialog(QDialog):
    """实时训练曲线弹窗，支持 YOLO 和 ResNet 两种模式。"""

    def __init__(self, parent=None, mode="yolo"):
        super().__init__(parent)
        self.mode = mode
        self.setWindowTitle("训练曲线")
        self.setMinimumSize(700 if mode == "yolo" else 500, 500)
        self.setStyleSheet(
            f"""
            QDialog {{
                background-color: {DARK_BG};
            }}
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        self.figure = Figure(figsize=(8, 5), dpi=100, facecolor=DARK_BG)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        if mode == "yolo":
            self.ax_train = self.figure.add_subplot(211)
            self.ax_metrics = self.figure.add_subplot(212)
            self._setup_yolo_axes()
            self.epochs = []
            self.box_loss = []
            self.obj_loss = []
            self.cls_loss = []
            self.total_loss = []
            self.metric_epochs = []
            self.precision_data = []
            self.recall_data = []
            self.map50_data = []
            self.map50_95_data = []
        else:
            self.ax_resnet = self.figure.add_subplot(111)
            self._setup_resnet_axes()
            self.resnet_epochs = []
            self.train_loss_data = []
            self.pred_error_data = []

        self.figure.tight_layout(pad=2.0)

    def _style_ax(self, ax):
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_COLOR, labelsize=10)
        ax.grid(True, color="#505050", alpha=0.28, linewidth=0.7)
        for spine in ax.spines.values():
            spine.set_color(BORDER_COLOR)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)

    def _setup_yolo_axes(self):
        self._style_ax(self.ax_train)
        self.ax_train.set_title("训练损失", fontsize=13)
        self.ax_train.set_xlabel("轮次", fontsize=11)
        self.ax_train.set_ylabel("损失", fontsize=11)

        self._style_ax(self.ax_metrics)
        self.ax_metrics.set_title("验证指标", fontsize=13)
        self.ax_metrics.set_xlabel("轮次", fontsize=11)
        self.ax_metrics.set_ylabel("分数", fontsize=11)
        self.ax_metrics.set_ylim(0.0, 1.05)

    def _setup_resnet_axes(self):
        self._style_ax(self.ax_resnet)
        self.ax_resnet.set_title("ResNet 训练曲线", fontsize=13)
        self.ax_resnet.set_xlabel("轮次", fontsize=11)
        self.ax_resnet.set_ylabel("损失", fontsize=11)

    def update_train_loss(self, epoch, box, obj, cls, total):
        self.epochs.append(epoch)
        self.box_loss.append(box)
        self.obj_loss.append(obj)
        self.cls_loss.append(cls)
        self.total_loss.append(total)
        self._redraw_yolo_train()

    def _redraw_yolo_train(self):
        self.ax_train.clear()
        self._style_ax(self.ax_train)
        self.ax_train.set_title("训练损失", fontsize=13)
        self.ax_train.set_xlabel("轮次", fontsize=11)
        self.ax_train.set_ylabel("损失", fontsize=11)

        self.ax_train.plot(self.epochs, self.box_loss, label="框损失", color="#FF6B6B", linewidth=1.2)
        self.ax_train.plot(self.epochs, self.obj_loss, label="目标损失", color="#4ECDC4", linewidth=1.2)
        self.ax_train.plot(self.epochs, self.cls_loss, label="分类损失", color="#FFE66D", linewidth=1.2)
        self.ax_train.plot(self.epochs, self.total_loss, label="总损失", color=BASE_COLOR, linewidth=1.8)

        legend = self.ax_train.legend(loc="upper right", fontsize=10, facecolor=PANEL_BG, edgecolor=BORDER_COLOR)
        for text in legend.get_texts():
            text.set_color(TEXT_COLOR)

        self.figure.tight_layout(pad=2.0)
        self.canvas.draw_idle()

    def update_val_metrics(self, epoch, precision, recall, map50, map50_95):
        self.metric_epochs.append(epoch)
        self.precision_data.append(precision)
        self.recall_data.append(recall)
        self.map50_data.append(map50)
        self.map50_95_data.append(map50_95)
        self._redraw_yolo_metrics()

    def _redraw_yolo_metrics(self):
        self.ax_metrics.clear()
        self._style_ax(self.ax_metrics)
        self.ax_metrics.set_title("验证指标", fontsize=13)
        self.ax_metrics.set_xlabel("轮次", fontsize=11)
        self.ax_metrics.set_ylabel("分数", fontsize=11)
        self.ax_metrics.set_ylim(0.0, 1.05)

        self.ax_metrics.plot(self.metric_epochs, self.precision_data, label="Precision", color="#5BC0EB", linewidth=1.2)
        self.ax_metrics.plot(self.metric_epochs, self.recall_data, label="Recall", color="#9BC53D", linewidth=1.2)
        self.ax_metrics.plot(self.metric_epochs, self.map50_data, label="mAP@0.5", color="#FDE74C", linewidth=1.2)
        self.ax_metrics.plot(self.metric_epochs, self.map50_95_data, label="mAP@0.5:0.95", color=BASE_COLOR, linewidth=1.8)
        if self.map50_95_data:
            best_index = max(range(len(self.map50_95_data)), key=self.map50_95_data.__getitem__)
            best_epoch = self.metric_epochs[best_index]
            best_score = self.map50_95_data[best_index]
            self.ax_metrics.scatter([best_epoch], [best_score], color="#FFFFFF", edgecolor=BASE_COLOR, zorder=5)
            self.ax_metrics.annotate(
                f"最佳 {best_score:.3f}",
                (best_epoch, best_score),
                xytext=(8, 8),
                textcoords="offset points",
                color=TEXT_COLOR,
                fontsize=10,
            )

        legend = self.ax_metrics.legend(loc="lower right", fontsize=10, facecolor=PANEL_BG, edgecolor=BORDER_COLOR)
        for text in legend.get_texts():
            text.set_color(TEXT_COLOR)

        self.figure.tight_layout(pad=2.0)
        self.canvas.draw_idle()

    def update_resnet_loss(self, epoch, train_loss, pred_error):
        self.resnet_epochs.append(epoch)
        self.train_loss_data.append(train_loss)
        self.pred_error_data.append(pred_error)
        self._redraw_resnet()

    def _redraw_resnet(self):
        self.ax_resnet.clear()
        self._style_ax(self.ax_resnet)
        self.ax_resnet.set_title("ResNet 训练曲线", fontsize=13)
        self.ax_resnet.set_xlabel("轮次", fontsize=11)
        self.ax_resnet.set_ylabel("损失", fontsize=11)

        self.ax_resnet.plot(self.resnet_epochs, self.train_loss_data, label="训练损失", color="#FF6B6B", linewidth=1.2)
        self.ax_resnet.plot(self.resnet_epochs, self.pred_error_data, label="预测误差", color=BASE_COLOR, linewidth=1.8)
        if self.pred_error_data:
            best_index = min(range(len(self.pred_error_data)), key=self.pred_error_data.__getitem__)
            best_epoch = self.resnet_epochs[best_index]
            best_error = self.pred_error_data[best_index]
            self.ax_resnet.scatter([best_epoch], [best_error], color="#FFFFFF", edgecolor=BASE_COLOR, zorder=5)
            self.ax_resnet.annotate(
                f"最佳 {best_error:.3f}",
                (best_epoch, best_error),
                xytext=(8, 8),
                textcoords="offset points",
                color=TEXT_COLOR,
                fontsize=10,
            )

        legend = self.ax_resnet.legend(loc="upper right", fontsize=10, facecolor=PANEL_BG, edgecolor=BORDER_COLOR)
        for text in legend.get_texts():
            text.set_color(TEXT_COLOR)

        self.figure.tight_layout(pad=2.0)
        self.canvas.draw_idle()

    def force_close(self):
        self._force_close = True
        self.close()

    def closeEvent(self, event):
        if getattr(self, "_force_close", False):
            event.accept()
        else:
            event.ignore()
            self.hide()
