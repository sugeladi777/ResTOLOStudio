from PyQt5.QtWidgets import QDialog, QVBoxLayout
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

DARK_BG = "#1A1A1A"
PANEL_BG = "#252525"
TEXT_COLOR = "#E8E8E8"
BORDER_COLOR = "#404040"
BASE_COLOR = "#5AE54A"


class LossCurveDialog(QDialog):
    """训练曲线实时展示弹窗，支持YOLO和ResNet两种模式"""

    def __init__(self, parent=None, mode='yolo'):
        """
        mode: 'yolo' - 双子图(训练Loss + 验证指标)
              'resnet' - 单子图(Training Loss + Prediction Error)
        """
        super().__init__(parent)
        self.mode = mode
        self.setWindowTitle("训练曲线")
        self.setMinimumSize(700 if mode == 'yolo' else 500, 500)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {DARK_BG};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # 创建matplotlib画布
        self.figure = Figure(figsize=(8, 5), dpi=100, facecolor=DARK_BG)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        if mode == 'yolo':
            self.ax_train = self.figure.add_subplot(121)
            self.ax_val = self.figure.add_subplot(122)
            self._setup_yolo_axes()
            # YOLO数据存储
            self.epochs = []
            self.box_loss = []
            self.obj_loss = []
            self.cls_loss = []
            self.total_loss = []
            self.precision = []
            self.recall = []
            self.map50 = []
            self.map50_95 = []
            self.val_epochs = []
        else:
            self.ax_resnet = self.figure.add_subplot(111)
            self._setup_resnet_axes()
            # ResNet数据存储
            self.resnet_epochs = []
            self.train_loss_data = []
            self.pred_error_data = []

        self.figure.tight_layout(pad=2.0)

    def _style_ax(self, ax):
        """统一设置坐标轴样式"""
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_COLOR, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(BORDER_COLOR)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)

    def _setup_yolo_axes(self):
        self._style_ax(self.ax_train)
        self._style_ax(self.ax_val)
        self.ax_train.set_title("训练Loss", fontsize=11)
        self.ax_train.set_xlabel("Epoch", fontsize=9)
        self.ax_train.set_ylabel("Loss", fontsize=9)
        self.ax_val.set_title("验证指标", fontsize=11)
        self.ax_val.set_xlabel("Epoch", fontsize=9)
        self.ax_val.set_ylabel("Value", fontsize=9)

    def _setup_resnet_axes(self):
        self._style_ax(self.ax_resnet)
        self.ax_resnet.set_title("ResNet训练曲线", fontsize=11)
        self.ax_resnet.set_xlabel("Epoch", fontsize=9)
        self.ax_resnet.set_ylabel("Loss", fontsize=9)

    # ---- YOLO模式方法 ----

    def update_train_loss(self, epoch, box, obj, cls, total):
        """YOLO: 更新训练loss数据并重绘"""
        self.epochs.append(epoch)
        self.box_loss.append(box)
        self.obj_loss.append(obj)
        self.cls_loss.append(cls)
        self.total_loss.append(total)
        self._redraw_yolo_train()

    def update_val_metrics(self, epoch, precision, recall, map50, map50_95):
        """YOLO: 更新验证指标数据并重绘"""
        self.val_epochs.append(epoch)
        self.precision.append(precision)
        self.recall.append(recall)
        self.map50.append(map50)
        self.map50_95.append(map50_95)
        self._redraw_yolo_val()

    def _redraw_yolo_train(self):
        self.ax_train.clear()
        self._style_ax(self.ax_train)
        self.ax_train.set_title("训练Loss", fontsize=11)
        self.ax_train.set_xlabel("Epoch", fontsize=9)
        self.ax_train.set_ylabel("Loss", fontsize=9)

        self.ax_train.plot(self.epochs, self.box_loss, label='box', color='#FF6B6B', linewidth=1.2)
        self.ax_train.plot(self.epochs, self.obj_loss, label='obj', color='#4ECDC4', linewidth=1.2)
        self.ax_train.plot(self.epochs, self.cls_loss, label='cls', color='#FFE66D', linewidth=1.2)
        self.ax_train.plot(self.epochs, self.total_loss, label='total', color=BASE_COLOR, linewidth=1.8)

        legend = self.ax_train.legend(loc='upper right', fontsize=8,
                                       facecolor=PANEL_BG, edgecolor=BORDER_COLOR)
        for text in legend.get_texts():
            text.set_color(TEXT_COLOR)

        self.figure.tight_layout(pad=2.0)
        self.canvas.draw_idle()

    def _redraw_yolo_val(self):
        self.ax_val.clear()
        self._style_ax(self.ax_val)
        self.ax_val.set_title("验证指标", fontsize=11)
        self.ax_val.set_xlabel("Epoch", fontsize=9)
        self.ax_val.set_ylabel("Value", fontsize=9)

        if self.val_epochs:
            self.ax_val.plot(self.val_epochs, self.precision, label='P', color='#FF6B6B', linewidth=1.2)
            self.ax_val.plot(self.val_epochs, self.recall, label='R', color='#4ECDC4', linewidth=1.2)
            self.ax_val.plot(self.val_epochs, self.map50, label='mAP@.5', color='#FFE66D', linewidth=1.2)
            self.ax_val.plot(self.val_epochs, self.map50_95, label='mAP@.5:.95', color=BASE_COLOR, linewidth=1.8)

        legend = self.ax_val.legend(loc='lower right', fontsize=8,
                                     facecolor=PANEL_BG, edgecolor=BORDER_COLOR)
        for text in legend.get_texts():
            text.set_color(TEXT_COLOR)

        self.figure.tight_layout(pad=2.0)
        self.canvas.draw_idle()

    # ---- ResNet模式方法 ----

    def update_resnet_loss(self, epoch, train_loss, pred_error):
        """ResNet: 更新训练loss和预测误差并重绘"""
        self.resnet_epochs.append(epoch)
        self.train_loss_data.append(train_loss)
        self.pred_error_data.append(pred_error)
        self._redraw_resnet()

    def _redraw_resnet(self):
        self.ax_resnet.clear()
        self._style_ax(self.ax_resnet)
        self.ax_resnet.set_title("ResNet训练曲线", fontsize=11)
        self.ax_resnet.set_xlabel("Epoch", fontsize=9)
        self.ax_resnet.set_ylabel("Loss", fontsize=9)

        self.ax_resnet.plot(self.resnet_epochs, self.train_loss_data, label='Training Loss', color='#FF6B6B', linewidth=1.2)
        self.ax_resnet.plot(self.resnet_epochs, self.pred_error_data, label='Prediction Error', color=BASE_COLOR, linewidth=1.8)

        legend = self.ax_resnet.legend(loc='upper right', fontsize=8,
                                        facecolor=PANEL_BG, edgecolor=BORDER_COLOR)
        for text in legend.get_texts():
            text.set_color(TEXT_COLOR)

        self.figure.tight_layout(pad=2.0)
        self.canvas.draw_idle()

    # ---- 通用方法 ----

    def force_close(self):
        """强制关闭窗口（训练完成时调用）"""
        self._force_close = True
        self.close()

    def closeEvent(self, event):
        """关闭时只隐藏，不销毁（除非是强制关闭）"""
        if getattr(self, '_force_close', False):
            event.accept()
        else:
            event.ignore()
            self.hide()
