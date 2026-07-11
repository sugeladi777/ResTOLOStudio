from __future__ import annotations

import os

import numpy as np
from PyQt5.QtCore import QPoint, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPalette, QPixmap
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from PIL import Image, ImageDraw

from app.core import AnnotationBox, AnnotationState

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency in lightweight test envs
    cv2 = None


BASE_COLOR = "#5AE54A"
HIGHLIGHT_COLOR = "#D3FF9E"
MUTED_COLOR = "#B0FC97"
ACCENT_COLOR = "#9CF96D"
DEEP_SHADE_COLOR = "#008400"
DARK_BG = "#181818"
PANEL_BG = "#232323"
TEXT_COLOR = "#E8E8E8"
BORDER_COLOR = "#3A3A3A"
SELECTED_BG = PANEL_BG
SELECTED_BORDER = BASE_COLOR


def _debug(message: str) -> None:
    if os.environ.get("RESTOLO_DEBUG_UI") == "1":
        print(message)


BOX_COLORS = [
    (0, 255, 0),
    (0, 0, 255),
    (255, 0, 0),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (128, 128, 0),
]


class AnnotationTool(QWidget):
    annotation_updated = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 560)
        self._pending_refresh_timer = QTimer(self)
        self._pending_refresh_timer.setSingleShot(True)
        self._pending_refresh_timer.timeout.connect(self._refresh_current_image)
        self.apply_responsive_styles()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        self._build_control_bar(layout)
        self._build_class_selector(layout)
        self._build_image_panel(layout)
        self._build_status_bar(layout)
        self._initialize_state()
        self._connect_signals()
        self._install_event_filters()

    def _scaled_int(self, value: int, scale: float) -> int:
        return max(1, int(round(value * scale)))

    def _ui_scale_factor(self) -> float:
        width = max(840, self.width() or 840)
        height = max(560, self.height() or 560)
        scale = min(width / 840.0, height / 560.0)
        return max(1.0, min(scale, 1.6))

    def apply_responsive_styles(self) -> None:
        scale = self._ui_scale_factor()
        self.setStyleSheet(
            f"""
            AnnotationTool {{
                background-color: {DARK_BG};
            }}
            QPushButton {{
                background-color: {PANEL_BG};
                color: {TEXT_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: 3px;
                padding: {self._scaled_int(8, scale)}px {self._scaled_int(14, scale)}px;
                font-weight: normal;
                font-size: {self._scaled_int(16, scale)}px;
                min-height: {self._scaled_int(44, scale)}px;
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
            QPushButton:checked {{
                background-color: {SELECTED_BG};
                color: {HIGHLIGHT_COLOR};
                border: 2px solid {SELECTED_BORDER};
            }}
            QPushButton[danger="true"] {{
                color: #FF8A8A;
                border-color: #704040;
            }}
            QPushButton[danger="true"]:hover {{
                background-color: #4A2525;
                color: #FFFFFF;
                border-color: #FF6B6B;
            }}
            QLabel {{
                color: {TEXT_COLOR};
                background-color: transparent;
                font-size: {self._scaled_int(17, scale)}px;
            }}
            QScrollArea {{
                border: 1px solid {BORDER_COLOR};
                border-radius: 3px;
                background-color: {DARK_BG};
            }}
            """
        )

    def _build_control_bar(self, layout: QVBoxLayout) -> None:
        self.control_layout = QHBoxLayout()
        self.control_layout.setSpacing(6)

        self.prev_btn = QPushButton("← 上一张")
        self.next_btn = QPushButton("下一张 →")
        self.clear_btn = QPushButton("清空标注")
        self.delete_btn = QPushButton("删除选中")
        self.reset_view_btn = QPushButton("重置视图")
        self.more_btn = QToolButton()
        self.more_btn.setText("更多")
        self.more_btn.setPopupMode(QToolButton.InstantPopup)
        self.more_btn.setCursor(Qt.PointingHandCursor)
        self.clear_btn.setProperty("danger", "true")
        self.delete_btn.setProperty("danger", "true")

        more_menu = QMenu(self)
        self.clear_action = more_menu.addAction("清空标注")
        self.delete_action = more_menu.addAction("删除选中")
        self.reset_view_action = more_menu.addAction("重置视图")
        self.more_btn.setMenu(more_menu)

        for button in (self.prev_btn, self.next_btn, self.clear_btn, self.delete_btn, self.reset_view_btn):
            button.setCursor(Qt.PointingHandCursor)
            button.setAutoDefault(False)
            button.setDefault(False)
            button.setFocusPolicy(Qt.NoFocus)
            self.control_layout.addWidget(button)

        self.more_btn.setFocusPolicy(Qt.NoFocus)
        self.control_layout.addWidget(self.more_btn)
        self.more_btn.hide()
        self.control_layout.addStretch()
        layout.addLayout(self.control_layout)

    def _build_class_selector(self, layout: QVBoxLayout) -> None:
        self.class_scroll_area = QScrollArea()
        self.class_scroll_area.setWidgetResizable(True)
        self.class_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.class_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scale = self._ui_scale_factor()
        self.class_scroll_area.setMinimumHeight(self._scaled_int(82, scale))
        self.class_scroll_area.setStyleSheet(
            f"""
            QScrollArea {{
                border: 1px solid {BORDER_COLOR};
                border-radius: 3px;
                background-color: {DARK_BG};
            }}
            QScrollBar:horizontal {{
                background-color: {DARK_BG};
                height: {self._scaled_int(8, scale)}px;
                border-radius: 4px;
                margin: 0px;
            }}
            QScrollBar::handle:horizontal {{
                background-color: {BORDER_COLOR};
                border-radius: 4px;
                min-width: {self._scaled_int(24, scale)}px;
            }}
            QScrollArea QPushButton {{
                background-color: {PANEL_BG};
                color: {TEXT_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: 3px;
                padding: {self._scaled_int(8, scale)}px {self._scaled_int(14, scale)}px;
                font-weight: normal;
                font-size: {self._scaled_int(16, scale)}px;
                min-height: {self._scaled_int(44, scale)}px;
            }}
            QScrollArea QPushButton:hover {{
                background-color: #2F2F2F;
                color: {TEXT_COLOR};
                border-color: {BASE_COLOR};
            }}
            QScrollArea QPushButton:focus {{
                background-color: {PANEL_BG};
                color: {TEXT_COLOR};
                border: 1px solid {BASE_COLOR};
                outline: none;
            }}
            QScrollArea QPushButton:pressed {{
                background-color: {DEEP_SHADE_COLOR};
                color: {TEXT_COLOR};
            }}
            QScrollArea QPushButton:checked {{
                background-color: {SELECTED_BG};
                color: {TEXT_COLOR};
                border: 2px solid {SELECTED_BORDER};
            }}
            """
        )

        class_container = QWidget()
        class_container.setStyleSheet(f"background-color: {DARK_BG};")
        self.class_layout = QHBoxLayout(class_container)
        self.class_layout.setSpacing(6)
        self.class_layout.setContentsMargins(6, 6, 6, 6)

        class_label = QLabel("选择类别:")
        class_label.setStyleSheet(f"color: {HIGHLIGHT_COLOR}; font-weight: bold;")
        self.class_layout.addWidget(class_label)

        self.class_buttons: list[QPushButton] = []
        for i in range(10):
            btn = QPushButton(str(i))
            btn.setCheckable(True)
            btn.setChecked(i == 0)
            btn.clicked.connect(lambda checked, cls=i: self.select_class(cls))
            btn.mouseDoubleClickEvent = lambda event, cls=i: self.edit_class_name(cls)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setAutoDefault(False)
            btn.setDefault(False)
            btn.setFocusPolicy(Qt.NoFocus)
            self.class_layout.addWidget(btn)
            self.class_buttons.append(btn)

        self.add_btn = QPushButton("+")
        self.add_btn.setCursor(Qt.PointingHandCursor)
        self.add_btn.setAutoDefault(False)
        self.add_btn.setDefault(False)
        self.add_btn.setFocusPolicy(Qt.NoFocus)
        self.add_btn.clicked.connect(self.add_new_class)
        self.class_layout.addWidget(self.add_btn)

        self.remove_btn = QPushButton("-")
        self.remove_btn.setCursor(Qt.PointingHandCursor)
        self.remove_btn.setAutoDefault(False)
        self.remove_btn.setDefault(False)
        self.remove_btn.setFocusPolicy(Qt.NoFocus)
        self.remove_btn.clicked.connect(self.remove_last_class)
        self.class_layout.addWidget(self.remove_btn)

        self.class_scroll_area.setWidget(class_container)
        layout.addWidget(self.class_scroll_area)

    def _build_image_panel(self, layout: QVBoxLayout) -> None:
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setAlignment(Qt.AlignCenter)
        self.scroll_area.setBackgroundRole(QPalette.Dark)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(False)
        self.image_label.setMinimumSize(640, 420)
        self.image_label.setWordWrap(True)
        self.image_label.setMouseTracking(True)
        self._show_empty_placeholder("未加载图片\n\n请在左侧选择采集结果、加载图像，或切换到标注/应用页面开始处理。")

        self.scroll_area.setWidget(self.image_label)
        layout.addWidget(self.scroll_area, 1)

    def _show_empty_placeholder(self, message: str) -> None:
        self.image_label.clear()
        self.image_label.setStyleSheet(
            f"""
            QLabel {{
                background-color: #151915;
                border: 1px dashed #426043;
                border-radius: 6px;
                color: #9CF96D;
                padding: 28px;
                font-size: 20px;
            }}
            """
        )
        self.image_label.setText(message)

    def _apply_image_canvas_style(self) -> None:
        self.image_label.setStyleSheet(f"background-color: {DARK_BG}; border: none; padding: 0px;")

    def _build_status_bar(self, layout: QVBoxLayout) -> None:
        self.status_label = QLabel("就绪")
        scale = self._ui_scale_factor()
        self.status_label.setStyleSheet(
            f"""
            QLabel {{
                color: {MUTED_COLOR};
                padding: {self._scaled_int(8, scale)}px {self._scaled_int(10, scale)}px;
                background-color: {PANEL_BG};
                border-radius: 3px;
                font-size: {self._scaled_int(17, scale)}px;
            }}
            """
        )
        layout.addWidget(self.status_label)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.apply_responsive_styles()
        self._update_toolbar_density()
        if hasattr(self, "class_scroll_area"):
            scale = self._ui_scale_factor()
            self.class_scroll_area.setMinimumHeight(self._scaled_int(82, scale))

    def _update_toolbar_density(self) -> None:
        if not hasattr(self, "more_btn"):
            return
        compact = self.width() < 900
        show_editing_tools = bool(self.annotation_mode and not compact)
        self.clear_btn.setVisible(show_editing_tools)
        self.delete_btn.setVisible(show_editing_tools)
        self.reset_view_btn.setVisible(not compact)
        self.more_btn.setVisible(bool(self.annotation_mode and compact))

    def _initialize_state(self) -> None:
        self.images: list[str] = []
        self.current_index = 0
        self.annotations: dict[str, list[tuple[int, float, float, float, float]]] = {}
        self.is_drawing = False
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.current_image = None
        self.class_names = [str(i) for i in range(10)]
        self.selected_box_index = -1
        self.current_class = 0

        self.zoom_level = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 10.0
        self.is_panning = False
        self.last_pan_point = QPoint()
        self.annotation_mode = True

    def load_state(self, state: AnnotationState) -> None:
        normalized = state.normalized()
        self.images = list(normalized.images)
        self.current_index = normalized.current_index
        self.annotations = {
            image_path: [box.to_tuple() for box in normalized.annotations.get(image_path, [])]
            for image_path in normalized.images
        }
        for image_path, boxes in normalized.annotations.items():
            if image_path not in self.annotations:
                self.annotations[image_path] = [box.to_tuple() for box in boxes]

        self.class_names = list(normalized.class_names)
        while len(self.class_buttons) < len(self.class_names):
            self.add_new_class()
        while len(self.class_buttons) > len(self.class_names):
            self.remove_last_class()

        for index, button in enumerate(self.class_buttons):
            button.setText(self.class_names[index] if index < len(self.class_names) else str(index))

        if self.current_class >= len(self.class_buttons):
            self.current_class = max(0, len(self.class_buttons) - 1)
        if self.class_buttons:
            self.select_class(self.current_class)

        self.selected_box_index = -1
        self.reset_view()
        if self.images:
            self.show_current_image()
        else:
            self.current_image = None
            self._show_empty_placeholder("未加载图片\n\n请在左侧加载图像或选择已有扫描结果。")
            self.status_label.setText("未加载图片")

    def export_state(self) -> AnnotationState:
        return AnnotationState(
            images=list(self.images),
            annotations={
                image_path: [AnnotationBox.from_tuple(box) for box in boxes]
                for image_path, boxes in self.annotations.items()
            },
            class_names=list(self.class_names),
            current_index=self.current_index,
        ).normalized()

    def _connect_signals(self) -> None:
        self.prev_btn.clicked.connect(self.prev_image)
        self.next_btn.clicked.connect(self.next_image)
        self.clear_btn.clicked.connect(self.clear_annotations)
        self.delete_btn.clicked.connect(self.delete_selected_box)
        self.reset_view_btn.clicked.connect(self.reset_view)
        self.clear_action.triggered.connect(self.clear_annotations)
        self.delete_action.triggered.connect(self.delete_selected_box)
        self.reset_view_action.triggered.connect(self.reset_view)

    def _install_event_filters(self) -> None:
        self.scroll_area.installEventFilter(self)
        self.image_label.installEventFilter(self)

    def _viewport_size(self) -> tuple[int, int]:
        max_width = self.scroll_area.width()
        max_height = self.scroll_area.height()
        if max_width <= 0:
            max_width = 800
        if max_height <= 0:
            max_height = 600
        return max_width, max_height

    def _label_image_metrics(self):
        pixmap = self.image_label.pixmap()
        if not pixmap:
            return None

        label_width = self.image_label.width()
        label_height = self.image_label.height()
        if label_width == 0 or label_height == 0:
            return None

        pixmap_width = pixmap.width()
        pixmap_height = pixmap.height()
        offset_x = (label_width - pixmap_width) // 2
        offset_y = (label_height - pixmap_height) // 2

        return {
            "pixmap": pixmap,
            "pixmap_width": pixmap_width,
            "pixmap_height": pixmap_height,
            "offset_x": offset_x,
            "offset_y": offset_y,
        }

    def _aspect_pixmap_size(self, image_width: int, image_height: int) -> tuple[int, int]:
        max_width, max_height = self._viewport_size()
        aspect_ratio = image_width / image_height
        if max_width / max_height > aspect_ratio:
            base_height = max_height
            base_width = int(base_height * aspect_ratio)
        else:
            base_width = max_width
            base_height = int(base_width / aspect_ratio)
        return base_width, base_height

    def _image_coords_from_label_points(self, start_point: QPoint, end_point: QPoint):
        if self.current_image is None:
            return None

        metrics = self._label_image_metrics()
        if not metrics:
            return None

        start_x = start_point.x() - metrics["offset_x"]
        start_y = start_point.y() - metrics["offset_y"]
        end_x = end_point.x() - metrics["offset_x"]
        end_y = end_point.y() - metrics["offset_y"]

        if (
            start_x < 0
            or start_y < 0
            or end_x > metrics["pixmap_width"]
            or end_y > metrics["pixmap_height"]
        ):
            return None

        h, w, _ = self.current_image.shape
        base_width, base_height = self._aspect_pixmap_size(w, h)
        final_width = int(base_width * self.zoom_level)
        final_height = int(base_height * self.zoom_level)

        scale_x = final_width / w
        scale_y = final_height / h

        x1 = int(min(start_x, end_x) / scale_x)
        y1 = int(min(start_y, end_y) / scale_y)
        x2 = int(max(start_x, end_x) / scale_x)
        y2 = int(max(start_y, end_y) / scale_y)

        return max(0, x1), max(0, y1), min(w, x2), min(h, y2)

    def _image_coords_from_label_pos(self, pos: QPoint):
        if self.current_image is None:
            return None

        metrics = self._label_image_metrics()
        if not metrics:
            return None

        click_x = pos.x() - metrics["offset_x"]
        click_y = pos.y() - metrics["offset_y"]
        if (
            click_x < 0
            or click_y < 0
            or click_x > metrics["pixmap_width"]
            or click_y > metrics["pixmap_height"]
        ):
            return None

        h, w, _ = self.current_image.shape
        base_width, base_height = self._aspect_pixmap_size(w, h)
        zoom_scale_x = (w / base_width) / self.zoom_level
        zoom_scale_y = (h / base_height) / self.zoom_level
        return click_x * zoom_scale_x, click_y * zoom_scale_y

    def _box_color(self, cls: int):
        return BOX_COLORS[cls % len(BOX_COLORS)]

    def _box_style(self, image_width: int, image_height: int) -> tuple[int, float, int]:
        line_width = max(1, min(int(image_width * 0.005), int(image_height * 0.005)))
        font_scale = max(0.4, min(image_width, image_height) * 0.001)
        font_thickness = max(1, int(min(image_width, image_height) * 0.002))
        return line_width, font_scale, font_thickness

    def _box_pixel_coords(self, box, image_width: int, image_height: int):
        cls, x, y, width, height = box
        x1 = max(0, int((x - width / 2) * image_width))
        y1 = max(0, int((y - height / 2) * image_height))
        x2 = min(image_width, int((x + width / 2) * image_width))
        y2 = min(image_height, int((y + height / 2) * image_height))
        return cls, x1, y1, x2, y2

    def _draw_box(self, image, box, image_width: int, image_height: int, selected: bool = False) -> None:
        cls, x1, y1, x2, y2 = self._box_pixel_coords(box, image_width, image_height)
        color = self._box_color(cls)
        line_width, font_scale, font_thickness = self._box_style(image_width, image_height)

        class_name = self.class_names[cls] if cls < len(self.class_names) else str(cls)
        if cv2 is not None:
            if selected:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), line_width + max(1, line_width // 2))
            else:
                cv2.rectangle(image, (x1, y1), (x2, y2), color, line_width)
            cv2.putText(
                image,
                class_name,
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                font_thickness,
            )
            return

        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)
        outline = (0, 255, 255) if selected else color
        draw.rectangle((x1, y1, x2, y2), outline=outline, width=line_width + (max(1, line_width // 2) if selected else 0))
        draw.text((x1, max(30, y1 - 10)), class_name, fill=color)
        image[:] = np.array(pil_image)

    def _load_image_array(self, image_path: str):
        from app.utils.sxm_parser import load_sxm_as_image

        if image_path.lower().endswith(".sxm"):
            try:
                pil_image, _ = load_sxm_as_image(image_path, use_color=False)
                if pil_image is not None:
                    if pil_image.mode != "RGB":
                        pil_image = pil_image.convert("RGB")
                    return np.array(pil_image)
            except Exception as exc:  # noqa: BLE001
                _debug(f"SXM 读取失败: {exc}")

        try:
            pil_image = Image.open(image_path)
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            return np.array(pil_image)
        except Exception as exc:  # noqa: BLE001
            _debug(f"PIL 读取图片失败: {exc}")

        if cv2 is not None:
            try:
                image = cv2.imread(image_path)
                if image is not None:
                    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except Exception as exc:  # noqa: BLE001
                _debug(f"cv2 读取图片失败: {exc}")

        return None

    def load_images(self, image_paths: list[str]) -> None:
        self.load_state(AnnotationState(images=list(image_paths)))

    def _refresh_current_image(self) -> None:
        if self.current_image is not None:
            self.draw_annotations()

    def reset_view(self) -> None:
        self.zoom_level = 1.0
        self.is_panning = False
        if self.current_image is not None:
            self.draw_annotations()

    def show_current_image(self) -> None:
        if not self.images:
            self.current_image = None
            self._show_empty_placeholder("未加载图片\n\n请在左侧加载图像或选择已有扫描结果。")
            self.status_label.setText("未加载图片")
            return

        image_path = self.images[self.current_index]
        if not os.path.exists(image_path):
            self.current_image = None
            self._show_empty_placeholder(f"图片不存在\n\n{os.path.basename(image_path)}")
            self.status_label.setText(f"图片不存在: {os.path.basename(image_path)}")
            return

        self.current_image = self._load_image_array(image_path)
        if self.current_image is None:
            self._show_empty_placeholder(f"无法读取图片\n\n{os.path.basename(image_path)}")
            self.status_label.setText(f"无法读取图片: {os.path.basename(image_path)}")
            return

        self.draw_annotations()
        self._pending_refresh_timer.start(30)

    def draw_annotations(self) -> None:
        if self.current_image is None:
            return

        image_copy = self.current_image.copy()
        h, w, _ = image_copy.shape

        if self.images and self.current_index < len(self.images):
            image_path = self.images[self.current_index]
            boxes = self.annotations.get(image_path, [])
            _debug(f"绘制标注: {len(boxes)}")
            for i, box in enumerate(boxes):
                self._draw_box(image_copy, box, w, h, selected=(i == self.selected_box_index))

        bytes_per_line = 3 * w
        q_image = QImage(image_copy.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        max_width = self.scroll_area.width() or 800
        max_height = self.scroll_area.height() or 600
        aspect_ratio = w / h
        if max_width / max_height > aspect_ratio:
            base_height = max_height
            base_width = int(base_height * aspect_ratio)
        else:
            base_width = max_width
            base_height = int(base_width / aspect_ratio)

        if self.annotation_mode:
            final_width = int(base_width * self.zoom_level)
            final_height = int(base_height * self.zoom_level)
        else:
            final_width = base_width
            final_height = base_height

        final_pixmap = pixmap.scaled(final_width, final_height, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self._apply_image_canvas_style()
        self.image_label.setPixmap(final_pixmap)
        self.image_label.resize(final_width, final_height)

        self.update_status_bar()

    def update_status_bar(self) -> None:
        if not self.images:
            self.status_label.setText("未加载图片")
            return

        image_path = self.images[self.current_index]
        basename = os.path.basename(image_path)
        box_count = len(self.annotations.get(image_path, []))
        zoom_text = f"{int(self.zoom_level * 100)}%" if self.annotation_mode else "适应窗口"
        self.status_label.setText(
            f"第 {self.current_index + 1}/{len(self.images)} 张 | {basename} | 标注框 {box_count} 个 | 缩放 {zoom_text}"
        )

    def eventFilter(self, obj, event):
        if obj == self.scroll_area and event.type() == event.Resize:
            if self.current_image is not None:
                self.draw_annotations()
            return True

        if obj == self.image_label:
            if event.type() == event.Wheel:
                self.wheelEvent(event)
                return True
            if event.type() == event.MouseButtonPress:
                self.handle_mouse_press(event)
                return True
            if event.type() == event.MouseButtonRelease:
                self.handle_mouse_release(event)
                return True
            if event.type() == event.MouseMove:
                self.handle_mouse_move(event)
                return True

        return super().eventFilter(obj, event)

    def wheelEvent(self, event) -> None:
        if self.current_image is None:
            return
        if not self.annotation_mode:
            return

        delta = event.angleDelta().y()
        zoom_factor = 1.15 if delta > 0 else 0.85
        new_zoom = self.zoom_level * zoom_factor
        self.zoom_level = max(self.min_zoom, min(self.max_zoom, new_zoom))
        self.draw_annotations()

    def handle_mouse_press(self, event) -> None:
        if self.current_image is None:
            return
        if not self.annotation_mode:
            return

        if event.button() == Qt.RightButton:
            if self.zoom_level > 1.0:
                self.is_panning = True
                self.last_pan_point = event.pos()
                self.image_label.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.LeftButton:
            if self.select_box(event.pos()):
                return
            self.is_drawing = True
            self.start_point = event.pos()

    def handle_mouse_move(self, event) -> None:
        if not self.annotation_mode:
            return
        if self.is_panning:
            delta = event.pos() - self.last_pan_point
            self.last_pan_point = event.pos()
            self.scroll_area.horizontalScrollBar().setValue(self.scroll_area.horizontalScrollBar().value() - delta.x())
            self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().value() - delta.y())
        elif self.is_drawing and self.current_image is not None:
            self.end_point = event.pos()
            self.draw_temp_rectangle()

    def handle_mouse_release(self, event) -> None:
        if not self.annotation_mode:
            return
        if event.button() == Qt.RightButton and self.is_panning:
            self.is_panning = False
            self.image_label.setCursor(Qt.PointingHandCursor)
            return

        if event.button() == Qt.LeftButton and self.is_drawing and self.current_image is not None:
            self.is_drawing = False
            self.end_point = event.pos()
            self.add_annotation()

    def draw_temp_rectangle(self) -> None:
        if self.current_image is None:
            return

        image_copy = self.current_image.copy()
        h, w, _ = image_copy.shape

        if self.images and self.current_index < len(self.images):
            image_path = self.images[self.current_index]
            for i, box in enumerate(self.annotations.get(image_path, [])):
                self._draw_box(image_copy, box, w, h, selected=(i == self.selected_box_index))

        if not self.start_point.isNull() and not self.end_point.isNull():
            temp_coords = self._image_coords_from_label_points(self.start_point, self.end_point)
            if temp_coords:
                temp_x1, temp_y1, temp_x2, temp_y2 = temp_coords
                line_width, _, _ = self._box_style(w, h)
                if cv2 is not None:
                    cv2.rectangle(image_copy, (temp_x1, temp_y1), (temp_x2, temp_y2), (0, 0, 255), line_width)
                else:
                    pil_image = Image.fromarray(image_copy)
                    draw = ImageDraw.Draw(pil_image)
                    draw.rectangle((temp_x1, temp_y1, temp_x2, temp_y2), outline=(255, 0, 0), width=line_width)
                    image_copy = np.array(pil_image)

        bytes_per_line = 3 * w
        q_image = QImage(image_copy.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        max_width, max_height = self._viewport_size()
        aspect_ratio = w / h
        if max_width / max_height > aspect_ratio:
            base_height = max_height
            base_width = int(base_height * aspect_ratio)
        else:
            base_width = max_width
            base_height = int(base_width / aspect_ratio)

        final_width = int(base_width * self.zoom_level)
        final_height = int(base_height * self.zoom_level)
        final_pixmap = pixmap.scaled(final_width, final_height, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(final_pixmap)
        self.image_label.resize(final_width, final_height)

    def select_class(self, cls: int) -> None:
        self.current_class = cls
        for i, btn in enumerate(self.class_buttons):
            btn.setChecked(i == cls)
        _debug(f"选择类别: {cls}")

    def edit_class_name(self, cls: int) -> None:
        current_name = self.class_names[cls] if cls < len(self.class_names) else str(cls)
        name, ok = QInputDialog.getText(self, "编辑类别名称", f"请输入类别 {cls} 的名称", text=current_name)
        if ok and name:
            if cls >= len(self.class_names):
                while len(self.class_names) <= cls:
                    self.class_names.append(self._next_default_class_name())
            self.class_names[cls] = name
            if cls < len(self.class_buttons):
                self.class_buttons[cls].setText(name)
            self.draw_annotations()
            self.annotation_updated.emit()

    def _next_default_class_name(self) -> str:
        existing_names = {str(name) for name in self.class_names}
        numeric_names = [int(name) for name in existing_names if name.isdigit()]
        candidate = max(numeric_names) + 1 if numeric_names else len(existing_names)
        while str(candidate) in existing_names:
            candidate += 1
        return str(candidate)

    def add_new_class(self) -> None:
        new_cls = len(self.class_buttons)
        default_name = self._next_default_class_name()
        btn = QPushButton(default_name)
        btn.setCheckable(True)
        btn.clicked.connect(lambda checked, cls=new_cls: self.select_class(cls))
        btn.mouseDoubleClickEvent = lambda event, cls=new_cls: self.edit_class_name(cls)
        btn.setCursor(Qt.PointingHandCursor)
        self.class_layout.insertWidget(len(self.class_buttons) + 1, btn)
        self.class_buttons.append(btn)

        while len(self.class_names) <= new_cls:
            self.class_names.append(self._next_default_class_name())

        self.annotation_updated.emit()

    def remove_last_class(self) -> None:
        if not self.class_buttons:
            return

        last_cls = len(self.class_buttons) - 1
        has_annotations = any(box[0] == last_cls for boxes in self.annotations.values() for box in boxes)

        if has_annotations:
            class_name = self.class_names[last_cls] if last_cls < len(self.class_names) else str(last_cls)
            msg_box = QMessageBox(
                QMessageBox.Warning,
                "确认删除",
                f'类别 "{class_name}" 仍有标注框，删除后相关标注也会被移除。\n确定要继续吗？',
                QMessageBox.NoButton,
                self,
            )
            yes_btn = msg_box.addButton("删除", QMessageBox.YesRole)
            no_btn = msg_box.addButton("取消", QMessageBox.NoRole)
            msg_box.setDefaultButton(no_btn)
            msg_box.exec_()
            if msg_box.clickedButton() != yes_btn:
                return

            for image_path in list(self.annotations.keys()):
                self.annotations[image_path] = [box for box in self.annotations[image_path] if box[0] != last_cls]
            self.draw_annotations()

        btn = self.class_buttons.pop()
        btn.hide()
        btn.deleteLater()

        if self.current_class >= len(self.class_buttons):
            self.current_class = max(0, len(self.class_buttons) - 1)
            if self.class_buttons:
                self.class_buttons[self.current_class].setChecked(True)

        self.annotation_updated.emit()

    def add_annotation(self) -> None:
        if self.current_image is None:
            return

        h, w, _ = self.current_image.shape
        image_coords = self._image_coords_from_label_points(self.start_point, self.end_point)
        if image_coords is None:
            self.draw_annotations()
            return

        x1, y1, x2, y2 = image_coords
        center_x = (x1 + x2) / (2 * w)
        center_y = (y1 + y2) / (2 * h)
        width = (x2 - x1) / w
        height = (y2 - y1) / h

        if width > 0 and height > 0:
            image_path = self.images[self.current_index]
            self.annotations[image_path].append((self.current_class, center_x, center_y, width, height))
            self.draw_annotations()
            self.annotation_updated.emit()

    def clear_annotations(self) -> None:
        if self.images:
            image_path = self.images[self.current_index]
            self.annotations[image_path] = []
            self.draw_annotations()
            self.annotation_updated.emit()

    def prev_image(self) -> None:
        if self.images and self.current_index > 0:
            self.current_index -= 1
            self.reset_view()
            self.show_current_image()

    def next_image(self) -> None:
        if self.images and self.current_index < len(self.images) - 1:
            self.current_index += 1
            self.reset_view()
            self.show_current_image()

    def select_box(self, pos: QPoint) -> bool:
        if not self.images or self.current_image is not None:
            pass
        if not self.images or self.current_image is None:
            return False

        image_path = self.images[self.current_index]
        boxes = self.annotations.get(image_path, [])
        if not boxes:
            return False

        image_coords = self._image_coords_from_label_pos(pos)
        if image_coords is None:
            self.selected_box_index = -1
            self.draw_annotations()
            return False

        orig_x, orig_y = image_coords
        h, w, _ = self.current_image.shape

        for i, box in enumerate(boxes):
            cls, x, y, width, height = box
            x1 = (x - width / 2) * w
            y1 = (y - height / 2) * h
            x2 = (x + width / 2) * w
            y2 = (y + height / 2) * h
            if x1 <= orig_x <= x2 and y1 <= orig_y <= y2:
                self.selected_box_index = i
                self.draw_annotations()
                return True

        self.selected_box_index = -1
        self.draw_annotations()
        return False

    def delete_selected_box(self) -> None:
        if not self.images or self.current_image is None or self.selected_box_index == -1:
            return

        image_path = self.images[self.current_index]
        boxes = self.annotations.get(image_path, [])
        if 0 <= self.selected_box_index < len(boxes):
            del boxes[self.selected_box_index]
            self.annotations[image_path] = boxes
            self.selected_box_index = -1
            self.draw_annotations()
            self.annotation_updated.emit()

    def on_key_press(self, event) -> None:
        if event.key() in {Qt.Key_Delete, Qt.Key_Backspace}:
            if self.selected_box_index != -1:
                self.delete_selected_box()
        elif Qt.Key_0 <= event.key() <= Qt.Key_9:
            self.select_class(event.key() - Qt.Key_0)

    def update_class_name(self, cls: int, name: str) -> None:
        try:
            if cls >= 0:
                while len(self.class_names) <= cls:
                    self.class_names.append(str(len(self.class_names)))
                self.class_names[cls] = name
        except Exception as exc:  # noqa: BLE001
            _debug(f"更新类别名称失败: {exc}")

    def set_annotation_mode(self, enabled: bool) -> None:
        self.annotation_mode = enabled
        self.class_scroll_area.setVisible(enabled)
        self._update_toolbar_density()
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded if enabled else Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded if enabled else Qt.ScrollBarAlwaysOff)
        if not enabled:
            self.zoom_level = 1.0
            self.is_panning = False
        if self.current_image is not None:
            self.draw_annotations()
