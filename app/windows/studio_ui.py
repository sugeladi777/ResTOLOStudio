from __future__ import annotations

from PyQt5.QtCore import QEvent, QSize, Qt
from PyQt5.QtGui import QColor, QCursor, QPainter
from PyQt5.QtWidgets import QApplication, QGroupBox, QLabel, QLineEdit, QPushButton, QToolTip, QVBoxLayout


BASE_COLOR = "#5AE54A"
HIGHLIGHT_COLOR = "#D3FF9E"
MUTED_COLOR = "#B0FC97"
ACCENT_COLOR = "#9CF96D"
DEEP_SHADE_COLOR = "#008400"
DARK_BG = "#1A1A1A"
PANEL_BG = "#252525"
TEXT_COLOR = "#E8E8E8"
BORDER_COLOR = "#404040"


class RequiredButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.PointingHandCursor)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setPen(QColor(Qt.red))
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        x = 5
        y = self.height() // 2 + self.fontMetrics().ascent() // 2
        painter.drawText(x, y, "*")

    def sizeHint(self):
        hint = super().sizeHint()
        return QSize(hint.width() + 15, hint.height())


class StudioUiMixin:
    def apply_styles(self):
        self.setStyleSheet(
            f"""
            QMainWindow {{
                background-color: {DARK_BG};
            }}
            QTabWidget::pane {{
                border: 1px solid {BORDER_COLOR};
                background-color: {PANEL_BG};
                border-radius: 8px;
            }}
            QTabBar::tab {{
                background-color: {PANEL_BG};
                color: {TEXT_COLOR};
                padding: 8px 20px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                font-weight: bold;
                font-size: 12px;
                min-width: 60px;
            }}
            QTabBar::tab:first {{
                border-top-left-radius: 8px;
            }}
            QTabBar::tab:last {{
                border-top-right-radius: 8px;
            }}
            QTabBar::tab:selected {{
                background-color: {BASE_COLOR};
                color: {DARK_BG};
            }}
            QTabBar::tab:hover {{
                background-color: {MUTED_COLOR};
                color: {DARK_BG};
            }}
            QTabBar QToolButton {{
                max-width: 0px;
                max-height: 0px;
                visibility: hidden;
            }}
            QGroupBox {{
                border: 2px solid {BORDER_COLOR};
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 10px;
                font-weight: bold;
                color: {HIGHLIGHT_COLOR};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 5px;
                color: {HIGHLIGHT_COLOR};
            }}
            QPushButton {{
                background-color: {PANEL_BG};
                color: {TEXT_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: 6px;
                padding: 6px 16px;
                font-weight: bold;
                font-size: 14px;
                min-height: 24px;
            }}
            QPushButton:hover {{
                background-color: {MUTED_COLOR};
                color: {DARK_BG};
                border-color: {BASE_COLOR};
            }}
            QPushButton:pressed {{
                background-color: {DEEP_SHADE_COLOR};
                color: {TEXT_COLOR};
            }}
            QPushButton:disabled {{
                background-color: {PANEL_BG};
                color: #666666;
                border-color: #333333;
            }}
            QPushButton[accent="true"] {{
                background-color: {BASE_COLOR};
                color: {DARK_BG};
                border: 2px solid {ACCENT_COLOR};
            }}
            QPushButton[accent="true"]:hover {{
                background-color: {HIGHLIGHT_COLOR};
                border-color: {HIGHLIGHT_COLOR};
            }}
            QLineEdit, QSpinBox, QTextEdit {{
                background-color: {DARK_BG};
                color: {TEXT_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: 5px;
                padding: 4px 10px;
            }}
            QLineEdit:focus, QSpinBox:focus {{
                border: 2px solid {BASE_COLOR};
            }}
            QLabel {{
                color: {TEXT_COLOR};
            }}
            QToolTip {{
                background-color: {PANEL_BG};
                color: {TEXT_COLOR};
                border: 1px solid {BORDER_COLOR};
                padding: 5px;
            }}
            QProgressBar {{
                border: 1px solid {BORDER_COLOR};
                border-radius: 5px;
                text-align: center;
                background-color: {DARK_BG};
                color: #FFFFFF;
                font-weight: bold;
            }}
            QProgressBar::chunk {{
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {BASE_COLOR}, stop:1 {ACCENT_COLOR});
                border-radius: 4px;
            }}
            QSplitter::handle {{
                background-color: {BORDER_COLOR};
            }}
            QScrollArea {{
                border: 1px solid {BORDER_COLOR};
                border-radius: 6px;
                background-color: {DARK_BG};
            }}
            QMessageBox {{
                background-color: {PANEL_BG};
            }}
            QMessageBox QLabel {{
                color: {TEXT_COLOR};
                padding: 5px;
            }}
            """
        )
        app = QApplication.instance()
        if app:
            app.setStyleSheet(
                f"""
                QToolTip {{
                    background-color: {PANEL_BG};
                    color: {TEXT_COLOR};
                    border: 1px solid {BORDER_COLOR};
                    padding: 4px 8px;
                }}
                """
            )

    def create_group(self, title):
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        layout.setSpacing(8)
        return group

    def create_button(self, text, callback, accent=False, required=False):
        btn = RequiredButton(text) if required else QPushButton(text)
        btn.clicked.connect(callback)
        btn.setCursor(Qt.PointingHandCursor)
        if accent:
            btn.setProperty("accent", "true")
        return btn

    def create_label(self, text, tooltip=""):
        label = QLabel(text)
        label.setToolTip(tooltip)
        label.setMouseTracking(True)
        label.installEventFilter(self)
        return label

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Enter and hasattr(obj, "toolTip"):
            tooltip = obj.toolTip()
            if tooltip:
                QToolTip.showText(QCursor.pos(), tooltip.strip())
        elif event.type() == QEvent.Leave:
            QToolTip.hideText()
        return super().eventFilter(obj, event)

    def create_line_edit(self, placeholder=""):
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        return edit
