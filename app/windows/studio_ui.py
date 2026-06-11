from __future__ import annotations

from PyQt5.QtCore import QEvent, QSize, Qt
from PyQt5.QtGui import QColor, QCursor, QFont, QPainter
from PyQt5.QtWidgets import QApplication, QGroupBox, QLabel, QLineEdit, QPushButton, QToolTip, QVBoxLayout


BASE_COLOR = "#8A939B"
HIGHLIGHT_COLOR = "#D3D8DC"
MUTED_COLOR = "#A7AFB6"
ACCENT_COLOR = "#A4ADB4"
DEEP_SHADE_COLOR = "#444A50"
DARK_BG = "#181818"
PANEL_BG = "#232323"
TEXT_COLOR = "#E4E4E4"
BORDER_COLOR = "#3A3A3A"
SELECTED_BG = "#5E768A"
SELECTED_BORDER = "#7E98AE"


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
        app = QApplication.instance()
        scale = self._ui_scale_factor()
        if app:
            app.setFont(QFont("Microsoft YaHei UI", self._scaled_int(12, scale)))
        self.setStyleSheet(self._build_stylesheet(scale))
        if app:
            app.setStyleSheet(
                f"""
                QToolTip {{
                    background-color: {PANEL_BG};
                    color: {TEXT_COLOR};
                    border: 1px solid {BORDER_COLOR};
                    padding: {self._scaled_int(4, scale)}px {self._scaled_int(8, scale)}px;
                    font-size: {self._scaled_int(13, scale)}px;
                }}
                """
            )

    def _scaled_int(self, value: int, scale: float) -> int:
        return max(1, int(round(value * scale)))

    def _ui_scale_factor(self) -> float:
        width = max(900, self.width() or 1000)
        height = max(620, self.height() or 660)
        scale = min(width / 1000.0, height / 660.0)
        return max(1.0, min(scale, 1.6))

    def _build_stylesheet(self, scale: float) -> str:
        return f"""
            QMainWindow {{
                background-color: {DARK_BG};
            }}
            QTabWidget::pane {{
                border: none;
                background-color: {PANEL_BG};
                border-radius: 0px;
            }}
            QTabBar::tab {{
                background-color: {PANEL_BG};
                color: {TEXT_COLOR};
                padding: {self._scaled_int(8, scale)}px {self._scaled_int(12, scale)}px;
                border: 1px solid {BORDER_COLOR};
                border-bottom: none;
                border-top-left-radius: 2px;
                border-top-right-radius: 2px;
                font-weight: normal;
                font-size: {self._scaled_int(14, scale)}px;
                min-width: {self._scaled_int(64, scale)}px;
            }}
            QTabBar::tab:first {{
                border-top-left-radius: 2px;
            }}
            QTabBar::tab:last {{
                border-top-right-radius: 2px;
            }}
            QTabBar::tab:selected {{
                background-color: {SELECTED_BG};
                color: #F5F7F8;
                border-color: {SELECTED_BORDER};
            }}
            QTabBar::tab:hover {{
                background-color: #2B2B2B;
                color: {TEXT_COLOR};
            }}
            QTabBar QToolButton {{
                max-width: 0px;
                max-height: 0px;
                width: 0px;
                height: 0px;
                border: none;
                padding: 0px;
                margin: 0px;
            }}
            QGroupBox {{
                border: 1px solid #2E2E2E;
                border-radius: 2px;
                margin-top: {self._scaled_int(10, scale)}px;
                padding-top: {self._scaled_int(5, scale)}px;
                font-weight: normal;
                font-size: {self._scaled_int(14, scale)}px;
                color: {TEXT_COLOR};
                background-color: transparent;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: {self._scaled_int(6, scale)}px;
                padding: 0 {self._scaled_int(3, scale)}px;
                color: {TEXT_COLOR};
            }}
            QPushButton {{
                background-color: {PANEL_BG};
                color: {TEXT_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: 2px;
                padding: {self._scaled_int(7, scale)}px {self._scaled_int(12, scale)}px;
                font-weight: normal;
                font-size: {self._scaled_int(14, scale)}px;
                min-height: {self._scaled_int(34, scale)}px;
            }}
            QPushButton:hover {{
                background-color: #2F2F2F;
                color: {TEXT_COLOR};
                border-color: {BASE_COLOR};
            }}
            QPushButton:focus {{
                background-color: {PANEL_BG};
                color: {TEXT_COLOR};
                border: 1px solid {BASE_COLOR};
                outline: none;
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
                background-color: {SELECTED_BG};
                color: #F5F7F8;
                border: 1px solid {SELECTED_BORDER};
            }}
            QPushButton[accent="true"]:hover {{
                background-color: #70879A;
                border-color: #8EA6BA;
            }}
            QPushButton[accent="true"]:focus {{
                background-color: {SELECTED_BG};
                color: #F5F7F8;
                border: 1px solid {SELECTED_BORDER};
                outline: none;
            }}
            QLineEdit, QSpinBox, QTextEdit {{
                background-color: {DARK_BG};
                color: {TEXT_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: 2px;
                padding: {self._scaled_int(6, scale)}px {self._scaled_int(10, scale)}px;
                font-size: {self._scaled_int(14, scale)}px;
                min-height: {self._scaled_int(30, scale)}px;
            }}
            QLineEdit:focus, QSpinBox:focus {{
                border: 2px solid {BASE_COLOR};
            }}
            QLabel {{
                color: {TEXT_COLOR};
                font-size: {self._scaled_int(14, scale)}px;
            }}
            QToolTip {{
                background-color: {PANEL_BG};
                color: {TEXT_COLOR};
                border: 1px solid {BORDER_COLOR};
                padding: {self._scaled_int(5, scale)}px;
            }}
            QListWidget {{
                background-color: {DARK_BG};
                color: {TEXT_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: 2px;
                padding: 2px;
                font-size: {self._scaled_int(14, scale)}px;
            }}
            QListWidget::item {{
                padding: {self._scaled_int(7, scale)}px {self._scaled_int(8, scale)}px;
                border-radius: 2px;
                margin-bottom: 1px;
                min-height: {self._scaled_int(30, scale)}px;
            }}
            QListWidget::item:selected {{
                background-color: rgba(94, 118, 138, 0.92);
                color: #F5F7F8;
            }}
            QProgressBar {{
                border: 1px solid {BORDER_COLOR};
                border-radius: 2px;
                text-align: center;
                background-color: {DARK_BG};
                color: #FFFFFF;
                font-weight: normal;
                min-height: {self._scaled_int(22, scale)}px;
                font-size: {self._scaled_int(13, scale)}px;
            }}
            QProgressBar::chunk {{
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {BASE_COLOR}, stop:1 {ACCENT_COLOR});
                border-radius: 2px;
            }}
            QSplitter::handle {{
                background-color: {BORDER_COLOR};
                width: {self._scaled_int(10, scale)}px;
            }}
            QSplitter::handle:hover {{
                background-color: {BASE_COLOR};
            }}
            QToolButton {{
                background-color: transparent;
                color: {MUTED_COLOR};
                border: none;
                padding: {self._scaled_int(5, scale)}px {self._scaled_int(6, scale)}px;
                font-size: {self._scaled_int(13, scale)}px;
                text-align: left;
            }}
            QToolButton:hover {{
                color: {TEXT_COLOR};
            }}
            QMenu {{
                background-color: {PANEL_BG};
                color: {TEXT_COLOR};
                border: 1px solid {BORDER_COLOR};
                font-size: {self._scaled_int(13, scale)}px;
                padding: {self._scaled_int(4, scale)}px;
            }}
            QMenu::item {{
                padding: {self._scaled_int(6, scale)}px {self._scaled_int(14, scale)}px;
                border-radius: 2px;
            }}
            QMenu::item:selected {{
                background-color: #2F2F2F;
            }}
            QScrollArea {{
                border: 1px solid {BORDER_COLOR};
                border-radius: 2px;
                background-color: {DARK_BG};
            }}
            QMessageBox {{
                background-color: {PANEL_BG};
            }}
            QMessageBox QLabel {{
                color: {TEXT_COLOR};
                padding: {self._scaled_int(5, scale)}px;
            }}
            """

    def create_group(self, title):
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        scale = self._ui_scale_factor()
        layout.setContentsMargins(
            self._scaled_int(5, scale),
            self._scaled_int(5, scale),
            self._scaled_int(5, scale),
            self._scaled_int(5, scale),
        )
        layout.setSpacing(self._scaled_int(3, scale))
        return group

    def create_button(self, text, callback, accent=False, required=False):
        btn = RequiredButton(text) if required else QPushButton(text)
        btn.clicked.connect(callback)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setAutoDefault(False)
        btn.setDefault(False)
        btn.setFocusPolicy(Qt.NoFocus)
        if accent:
            btn.setProperty("accent", "true")
        return btn

    def create_label(self, text, tooltip=""):
        label = QLabel(text)
        label.setToolTip(tooltip)
        label.setMouseTracking(True)
        label.installEventFilter(self)
        return label

    def style_readonly_summary(self, widget):
        scale = self._ui_scale_factor()
        widget.setStyleSheet(
            f"""
            QTextEdit {{
                background-color: {DARK_BG};
                color: {TEXT_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: 2px;
                padding: {self._scaled_int(4, scale)}px {self._scaled_int(6, scale)}px;
                font-size: {self._scaled_int(13, scale)}px;
            }}
            """
        )
        return widget

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
        edit.setMinimumHeight(self._scaled_int(34, self._ui_scale_factor()))
        return edit
