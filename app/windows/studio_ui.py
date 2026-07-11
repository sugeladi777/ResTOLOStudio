from __future__ import annotations

from PyQt5.QtCore import QEvent, QSize, Qt
from PyQt5.QtGui import QColor, QCursor, QFont, QPainter
from PyQt5.QtWidgets import QApplication, QGroupBox, QLabel, QLineEdit, QPushButton, QToolTip, QVBoxLayout


BASE_COLOR = "#5AE54A"
HIGHLIGHT_COLOR = "#D3FF9E"
MUTED_COLOR = "#B0FC97"
ACCENT_COLOR = "#9CF96D"
DEEP_SHADE_COLOR = "#008400"
SELECTED_TAB_BG = "#18351A"
HOVER_BG = "#30452F"
WARNING_COLOR = "#F2C94C"
ERROR_COLOR = "#FF6B6B"
DARK_BG = "#1A1A1A"
PANEL_BG = "#252525"
TEXT_COLOR = "#E8E8E8"
BORDER_COLOR = "#404040"
SELECTED_BG = SELECTED_TAB_BG
SELECTED_BORDER = BASE_COLOR


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
            app.setFont(QFont("Microsoft YaHei UI", self._scaled_int(14, scale)))
        self.setStyleSheet(self._build_stylesheet(scale))
        if app:
            app.setStyleSheet(
                f"""
                QToolTip {{
                    background-color: {PANEL_BG};
                    color: {TEXT_COLOR};
                    border: 1px solid {BORDER_COLOR};
                    padding: {self._scaled_int(4, scale)}px {self._scaled_int(8, scale)}px;
                    font-size: {self._scaled_int(16, scale)}px;
                }}
                """
            )

    def _scaled_int(self, value: int, scale: float) -> int:
        return max(1, int(round(value * scale)))

    def _ui_scale_factor(self) -> float:
        width = max(1100, self.width() or 1280)
        height = max(720, self.height() or 800)
        scale = min(width / 1280.0, height / 800.0)
        return max(1.0, min(scale, 1.25))

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
                color: #C7C7C7;
                padding: {self._scaled_int(8, scale)}px {self._scaled_int(10, scale)}px;
                border: 1px solid {BORDER_COLOR};
                border-bottom: 3px solid transparent;
                border-top-left-radius: {self._scaled_int(5, scale)}px;
                border-top-right-radius: {self._scaled_int(5, scale)}px;
                font-weight: bold;
                font-size: {self._scaled_int(15, scale)}px;
                min-width: {self._scaled_int(54, scale)}px;
            }}
            QTabBar::tab:first {{
                border-top-left-radius: {self._scaled_int(8, scale)}px;
            }}
            QTabBar::tab:last {{
                border-top-right-radius: {self._scaled_int(8, scale)}px;
            }}
            QTabBar::tab:selected {{
                background-color: {SELECTED_BG};
                color: {HIGHLIGHT_COLOR};
                border-color: {BORDER_COLOR};
                border-bottom-color: {SELECTED_BORDER};
            }}
            QTabBar::tab:hover {{
                background-color: {HOVER_BG};
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
                border: 1px solid {BORDER_COLOR};
                border-radius: {self._scaled_int(8, scale)}px;
                margin-top: {self._scaled_int(12, scale)}px;
                padding-top: {self._scaled_int(10, scale)}px;
                font-weight: bold;
                font-size: {self._scaled_int(17, scale)}px;
                color: {HIGHLIGHT_COLOR};
                background-color: transparent;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: {self._scaled_int(10, scale)}px;
                padding: 0 {self._scaled_int(5, scale)}px;
                color: {HIGHLIGHT_COLOR};
            }}
            QPushButton {{
                background-color: {PANEL_BG};
                color: {TEXT_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: {self._scaled_int(6, scale)}px;
                padding: {self._scaled_int(6, scale)}px {self._scaled_int(16, scale)}px;
                font-weight: 500;
                font-size: {self._scaled_int(16, scale)}px;
                min-height: {self._scaled_int(38, scale)}px;
            }}
            QPushButton:hover {{
                background-color: {HOVER_BG};
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
                background-color: {BASE_COLOR};
                color: {DARK_BG};
                border: 1px solid {BASE_COLOR};
            }}
            QPushButton[danger="true"] {{
                color: {ERROR_COLOR};
                border-color: #704040;
            }}
            QPushButton[danger="true"]:hover {{
                background-color: #4A2525;
                color: #FFFFFF;
                border-color: {ERROR_COLOR};
            }}
            QPushButton[accent="true"]:hover {{
                background-color: {ACCENT_COLOR};
                border-color: {ACCENT_COLOR};
            }}
            QPushButton[accent="true"]:focus {{
                background-color: {BASE_COLOR};
                color: {DARK_BG};
                border: 1px solid {BASE_COLOR};
                outline: none;
            }}
            QPushButton[accent="true"]:disabled {{
                background-color: #242424;
                color: #606060;
                border-color: #353535;
            }}
            QLineEdit, QSpinBox, QTextEdit {{
                background-color: {DARK_BG};
                color: {TEXT_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: {self._scaled_int(6, scale)}px;
                padding: {self._scaled_int(6, scale)}px {self._scaled_int(10, scale)}px;
                font-size: {self._scaled_int(16, scale)}px;
                min-height: {self._scaled_int(38, scale)}px;
            }}
            QLineEdit:focus, QSpinBox:focus {{
                border: 2px solid {BASE_COLOR};
            }}
            QComboBox {{
                background-color: {DARK_BG};
                color: {TEXT_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: {self._scaled_int(6, scale)}px;
                padding: {self._scaled_int(6, scale)}px {self._scaled_int(10, scale)}px;
                font-size: {self._scaled_int(16, scale)}px;
                min-height: {self._scaled_int(38, scale)}px;
            }}
            QComboBox:focus {{
                border: 2px solid {BASE_COLOR};
            }}
            QComboBox QAbstractItemView {{
                background-color: {PANEL_BG};
                color: {TEXT_COLOR};
                border: 1px solid {BORDER_COLOR};
                selection-background-color: {BASE_COLOR};
                selection-color: {DARK_BG};
            }}
            QLabel {{
                color: {TEXT_COLOR};
                font-size: {self._scaled_int(16, scale)}px;
            }}
            QLabel[muted="true"] {{
                color: #A8A8A8;
                font-size: {self._scaled_int(14, scale)}px;
            }}
            QLabel[uiState="normal"] {{
                color: #B8B8B8;
            }}
            QLabel[uiState="ready"], QLabel[uiState="success"] {{
                color: {MUTED_COLOR};
            }}
            QLabel[uiState="running"] {{
                color: #79CFFF;
            }}
            QLabel[uiState="warning"] {{
                color: {WARNING_COLOR};
            }}
            QLabel[uiState="error"] {{
                color: {ERROR_COLOR};
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
                font-size: {self._scaled_int(16, scale)}px;
            }}
            QListWidget::item {{
                padding: {self._scaled_int(7, scale)}px {self._scaled_int(8, scale)}px;
                border-radius: {self._scaled_int(6, scale)}px;
                margin-bottom: 1px;
                min-height: {self._scaled_int(40, scale)}px;
            }}
            QListWidget::item:selected {{
                background-color: #234A26;
                color: {HIGHLIGHT_COLOR};
                border-left: 3px solid {BASE_COLOR};
            }}
            QProgressBar {{
                border: 1px solid {BORDER_COLOR};
                border-radius: 2px;
                text-align: center;
                background-color: {DARK_BG};
                color: #FFFFFF;
                font-weight: normal;
                min-height: {self._scaled_int(28, scale)}px;
                font-size: {self._scaled_int(16, scale)}px;
            }}
            QProgressBar::chunk {{
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {BASE_COLOR}, stop:1 {ACCENT_COLOR});
                border-radius: {self._scaled_int(6, scale)}px;
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
                font-size: {self._scaled_int(15, scale)}px;
                text-align: left;
            }}
            QToolButton:hover {{
                color: {TEXT_COLOR};
            }}
            QToolButton[sectionToggle="true"] {{
                background-color: #202020;
                border: 1px solid {BORDER_COLOR};
                border-radius: {self._scaled_int(6, scale)}px;
                padding: {self._scaled_int(7, scale)}px {self._scaled_int(9, scale)}px;
                color: #CFCFCF;
                font-weight: 500;
            }}
            QToolButton[sectionToggle="true"]:checked {{
                color: {HIGHLIGHT_COLOR};
                border-color: #4F7048;
            }}
            QMenu {{
                background-color: {PANEL_BG};
                color: {TEXT_COLOR};
                border: 1px solid {BORDER_COLOR};
                font-size: {self._scaled_int(16, scale)}px;
                padding: {self._scaled_int(4, scale)}px;
            }}
            QMenu::item {{
                padding: {self._scaled_int(6, scale)}px {self._scaled_int(14, scale)}px;
                border-radius: 2px;
            }}
            QMenu::item:selected {{
                background-color: {BASE_COLOR};
                color: {DARK_BG};
            }}
            QScrollArea {{
                border: 1px solid {BORDER_COLOR};
                border-radius: {self._scaled_int(6, scale)}px;
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
            self._scaled_int(10, scale),
            self._scaled_int(8, scale),
            self._scaled_int(10, scale),
            self._scaled_int(10, scale),
        )
        layout.setSpacing(self._scaled_int(6, scale))
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
                font-size: {self._scaled_int(15, scale)}px;
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
        edit.setMinimumHeight(self._scaled_int(40, self._ui_scale_factor()))
        return edit
