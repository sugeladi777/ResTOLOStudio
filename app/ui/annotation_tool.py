from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QScrollArea, QPushButton, QInputDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QImage, QPalette
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
import cv2
import numpy as np
import os

from app.core import AnnotationBox, AnnotationState

BASE_COLOR = "#5AE54A"
HIGHLIGHT_COLOR = "#D3FF9E"
MUTED_COLOR = "#B0FC97"
ACCENT_COLOR = "#9CF96D"
DEEP_SHADE_COLOR = "#008400"
DARK_BG = "#1A1A1A"
PANEL_BG = "#252525"
TEXT_COLOR = "#E8E8E8"
BORDER_COLOR = "#404040"


def _debug(message):
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
        self.setMinimumSize(800, 500)
        self._apply_styles()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        self._build_control_bar(layout)
        self._build_class_selector(layout)
        self._build_image_panel(layout)
        self._build_status_bar(layout)
        self._initialize_state()
        self._connect_signals()
        self._install_event_filters()

    def _apply_styles(self):
        self.setStyleSheet(f"""
            AnnotationTool {{
                background-color: {DARK_BG};
            }}
            QPushButton {{
                background-color: {PANEL_BG};
                color: {TEXT_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                min-height: 28px;
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
            QPushButton:checked {{
                background-color: {BASE_COLOR};
                color: {DARK_BG};
                border: 2px solid {ACCENT_COLOR};
            }}
            QLabel {{
                color: {TEXT_COLOR};
                background-color: transparent;
            }}
            QScrollArea {{
                border: 1px solid {BORDER_COLOR};
                border-radius: 6px;
                background-color: {DARK_BG};
            }}
        """)

    def _build_control_bar(self, layout):
        self.control_layout = QHBoxLayout()
        self.control_layout.setSpacing(10)

        self.prev_btn = QPushButton("← 上一张")
        self.next_btn = QPushButton("下一张 →")
        self.clear_btn = QPushButton("清空标注")
        self.delete_btn = QPushButton("删除选中")
        self.reset_view_btn = QPushButton("重置视图")

        for button in (
            self.prev_btn,
            self.next_btn,
            self.clear_btn,
            self.delete_btn,
            self.reset_view_btn,
        ):
            button.setCursor(Qt.PointingHandCursor)
            self.control_layout.addWidget(button)

        layout.addLayout(self.control_layout)

    def _build_class_selector(self, layout):
        self.class_scroll_area = QScrollArea()
        self.class_scroll_area.setWidgetResizable(True)
        self.class_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.class_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.class_scroll_area.setMinimumHeight(50)
        self.class_scroll_area.setStyleSheet(f"""
            QScrollArea {{
                border: 1px solid {BORDER_COLOR};
                border-radius: 6px;
                background-color: {DARK_BG};
            }}
            QScrollBar:horizontal {{
                background-color: {DARK_BG};
                height: 10px;
                border-radius: 5px;
                margin: 0px;
            }}
            QScrollBar::handle:horizontal {{
                background-color: {BORDER_COLOR};
                border-radius: 5px;
                min-width: 30px;
            }}
            QScrollArea QPushButton {{
                background-color: {PANEL_BG};
                color: {TEXT_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                min-height: 28px;
            }}
            QScrollArea QPushButton:hover {{
                background-color: {MUTED_COLOR};
                color: {TEXT_COLOR};
                border-color: {BASE_COLOR};
            }}
            QScrollArea QPushButton:pressed {{
                background-color: {DEEP_SHADE_COLOR};
                color: {TEXT_COLOR};
            }}
            QScrollArea QPushButton:checked {{
                background-color: {BASE_COLOR};
                color: {TEXT_COLOR};
                border: 2px solid {ACCENT_COLOR};
            }}
        """)

        class_container = QWidget()
        class_container.setStyleSheet(f"background-color: {DARK_BG};")
        self.class_layout = QHBoxLayout(class_container)
        self.class_layout.setSpacing(8)
        self.class_layout.setContentsMargins(5, 5, 5, 5)

        class_label = QLabel("选择类别:")
        class_label.setStyleSheet(f"color: {HIGHLIGHT_COLOR}; font-weight: bold;")
        self.class_layout.addWidget(class_label)

        self.class_buttons = []
        for i in range(10):
            btn = QPushButton(str(i))
            btn.setCheckable(True)
            btn.setChecked(i == 0)
            btn.clicked.connect(lambda checked, cls=i: self.select_class(cls))
            btn.mouseDoubleClickEvent = lambda event, cls=i: self.edit_class_name(cls)
            btn.setCursor(Qt.PointingHandCursor)
            self.class_layout.addWidget(btn)
            self.class_buttons.append(btn)

        self.add_btn = QPushButton("+")
        self.add_btn.setCursor(Qt.PointingHandCursor)
        self.add_btn.clicked.connect(self.add_new_class)
        self.class_layout.addWidget(self.add_btn)

        self.remove_btn = QPushButton("-")
        self.remove_btn.setCursor(Qt.PointingHandCursor)
        self.remove_btn.clicked.connect(self.remove_last_class)
        self.class_layout.addWidget(self.remove_btn)

        self.class_scroll_area.setWidget(class_container)
        layout.addWidget(self.class_scroll_area)

    def _build_image_panel(self, layout):
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setAlignment(Qt.AlignCenter)
        self.scroll_area.setBackgroundRole(QPalette.Dark)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(False)
        self.image_label.setStyleSheet(f"background-color: {DARK_BG}; border: none;")
        self.image_label.setMouseTracking(True)

        self.scroll_area.setWidget(self.image_label)
        layout.addWidget(self.scroll_area, 1)

    def _build_status_bar(self, layout):
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet(f"""
            QLabel {{
                color: {MUTED_COLOR};
                padding: 5px 10px;
                background-color: {PANEL_BG};
                border-radius: 4px;
            }}
        """)
        layout.addWidget(self.status_label)

    def _initialize_state(self):
        self.images = []
        self.current_index = 0
        self.annotations = {}
        self.current_boxes = []
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
        self.pan_offset_x = 0
        self.pan_offset_y = 0

    def load_state(self, state: AnnotationState):
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
            self.image_label.clear()

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

    def _connect_signals(self):
        self.prev_btn.clicked.connect(self.prev_image)
        self.next_btn.clicked.connect(self.next_image)
        self.clear_btn.clicked.connect(self.clear_annotations)
        self.delete_btn.clicked.connect(self.delete_selected_box)
        self.reset_view_btn.clicked.connect(self.reset_view)

    def _install_event_filters(self):
        self.scroll_area.installEventFilter(self)
        self.image_label.installEventFilter(self)

    def _viewport_size(self):
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

    def _aspect_pixmap_size(self, image_width, image_height):
        max_width, max_height = self._viewport_size()
        aspect_ratio = image_width / image_height
        if max_width / max_height > aspect_ratio:
            base_height = max_height
            base_width = int(base_height * aspect_ratio)
        else:
            base_width = max_width
            base_height = int(base_width / aspect_ratio)
        return base_width, base_height

    def _image_coords_from_label_points(self, start_point, end_point):
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

        return (
            max(0, x1),
            max(0, y1),
            min(w, x2),
            min(h, y2),
        )

    def _image_coords_from_label_pos(self, pos):
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

    def _box_color(self, cls):
        return BOX_COLORS[cls % len(BOX_COLORS)]

    def _box_style(self, image_width, image_height):
        line_width = max(1, min(int(image_width * 0.005), int(image_height * 0.005)))
        font_scale = max(0.4, min(image_width, image_height) * 0.001)
        font_thickness = max(1, int(min(image_width, image_height) * 0.002))
        return line_width, font_scale, font_thickness

    def _box_pixel_coords(self, box, image_width, image_height):
        cls, x, y, width, height = box
        x1 = max(0, int((x - width / 2) * image_width))
        y1 = max(0, int((y - height / 2) * image_height))
        x2 = min(image_width, int((x + width / 2) * image_width))
        y2 = min(image_height, int((y + height / 2) * image_height))
        return cls, x1, y1, x2, y2

    def _draw_box(self, image, box, image_width, image_height, selected=False):
        cls, x1, y1, x2, y2 = self._box_pixel_coords(box, image_width, image_height)
        color = self._box_color(cls)
        line_width, font_scale, font_thickness = self._box_style(image_width, image_height)

        if selected:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), line_width + max(1, line_width // 2))
        else:
            cv2.rectangle(image, (x1, y1), (x2, y2), color, line_width)

        class_name = self.class_names[cls] if cls < len(self.class_names) else str(cls)
        cv2.putText(
            image,
            class_name,
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            font_thickness,
        )

    def _load_image_array(self, image_path):
        from PIL import Image

        try:
            pil_image = Image.open(image_path)
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            return np.array(pil_image)
        except Exception as exc:
            _debug(f"使用PIL读取图片失败: {exc}")

        try:
            image = cv2.imread(image_path)
            if image is not None:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as exc:
            _debug(f"使用cv2读取图片失败: {exc}")

        return None

    def load_images(self, image_paths):
        """加载图片"""
        self.load_state(AnnotationState(images=list(image_paths)))

    def reset_view(self):
        """重置视图到默认状态"""
        self.zoom_level = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self.is_panning = False
        if self.current_image is not None:
            self.draw_annotations()

    def show_current_image(self):
        """显示当前图片"""
        if not self.images:
            return

        image_path = self.images[self.current_index]

        if not os.path.exists(image_path):
            return

        self.current_image = self._load_image_array(image_path)
        if self.current_image is None:
            return

        self.draw_annotations()

    def draw_annotations(self):
        """绘制标注"""
        if self.current_image is None:
            return

        image_copy = self.current_image.copy()
        h, w, _ = image_copy.shape

        if self.images and self.current_index < len(self.images):
            image_path = self.images[self.current_index]
            boxes = self.annotations.get(image_path, [])
            _debug(f"绘制标注: {len(boxes)} 个标注")
            for i, box in enumerate(boxes):
                _, x1, y1, x2, y2 = self._box_pixel_coords(box, w, h)
                _debug(f"标注坐标: {x1},{y1} - {x2},{y2}")
                self._draw_box(image_copy, box, w, h, selected=(i == self.selected_box_index))

        h, w, ch = image_copy.shape
        bytes_per_line = ch * w
        q_image = QImage(image_copy.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        max_width = self.scroll_area.width()
        max_height = self.scroll_area.height()

        if max_width <= 0:
            max_width = 800
        if max_height <= 0:
            max_height = 600

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

        self.update_status_bar()

    def update_status_bar(self):
        """更新状态栏"""
        if not self.images:
            return
        basename = os.path.basename(self.images[self.current_index])
        zoom_percent = int(self.zoom_level * 100)
        self.status_label.setText(f"📷 {self.current_index+1}/{len(self.images)} | {basename} | 缩放: {zoom_percent}%")

    def eventFilter(self, obj, event):
        """事件过滤器"""
        if obj == self.scroll_area and event.type() == event.Resize:
            if self.current_image is not None:
                self.draw_annotations()
            return True
        elif obj == self.image_label:
            if event.type() == event.Wheel:
                self.wheelEvent(event)
                return True
            elif event.type() == event.MouseButtonPress:
                self.handle_mouse_press(event)
                return True
            elif event.type() == event.MouseButtonRelease:
                self.handle_mouse_release(event)
                return True
            elif event.type() == event.MouseMove:
                self.handle_mouse_move(event)
                return True

        return super().eventFilter(obj, event)

    def wheelEvent(self, event):
        """鼠标滚轮缩放"""
        if self.current_image is None:
            return

        delta = event.angleDelta().y()
        if delta > 0:
            zoom_factor = 1.15
        else:
            zoom_factor = 0.85

        new_zoom = self.zoom_level * zoom_factor
        new_zoom = max(self.min_zoom, min(self.max_zoom, new_zoom))

        self.zoom_level = new_zoom
        self.draw_annotations()

    def handle_mouse_press(self, event):
        """处理鼠标按下事件"""
        if self.current_image is None:
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

    def handle_mouse_move(self, event):
        """处理鼠标移动事件"""
        if self.is_panning:
            delta = event.pos() - self.last_pan_point
            self.last_pan_point = event.pos()

            h_bar = self.scroll_area.horizontalScrollBar()
            v_bar = self.scroll_area.verticalScrollBar()

            h_bar.setValue(h_bar.value() - delta.x())
            v_bar.setValue(v_bar.value() - delta.y())
        elif self.is_drawing and self.current_image is not None:
            self.end_point = event.pos()
            self.draw_temp_rectangle()

    def handle_mouse_release(self, event):
        """处理鼠标释放事件"""
        if event.button() == Qt.RightButton:
            if self.is_panning:
                self.is_panning = False
                self.image_label.setCursor(Qt.PointingHandCursor)
        elif event.button() == Qt.LeftButton:
            if self.is_drawing and self.current_image is not None:
                self.is_drawing = False
                self.end_point = event.pos()
                self.add_annotation()

    def draw_temp_rectangle(self):
        """绘制临时矩形"""
        if self.current_image is None:
            return

        image_copy = self.current_image.copy()
        h, w, _ = image_copy.shape

        if self.images and self.current_index < len(self.images):
            image_path = self.images[self.current_index]
            boxes = self.annotations.get(image_path, [])
            for i, box in enumerate(boxes):
                self._draw_box(image_copy, box, w, h, selected=(i == self.selected_box_index))

        if not self.start_point.isNull() and not self.end_point.isNull():
            temp_coords = self._image_coords_from_label_points(self.start_point, self.end_point)
            if temp_coords:
                temp_x1, temp_y1, temp_x2, temp_y2 = temp_coords
                line_width, _, _ = self._box_style(w, h)
                cv2.rectangle(image_copy, (temp_x1, temp_y1), (temp_x2, temp_y2), (0, 0, 255), line_width)

        h, w, ch = image_copy.shape
        bytes_per_line = ch * w
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

    def select_class(self, cls):
        """选择类别"""
        self.current_class = cls
        for i, btn in enumerate(self.class_buttons):
            btn.setChecked(i == cls)
        _debug(f"选择类别: {cls}")

    def edit_class_name(self, cls):
        """编辑类别名称"""
        current_name = self.class_names[cls] if cls < len(self.class_names) else str(cls)
        name, ok = QInputDialog.getText(self, "编辑类别名称", f"请输入类别 {cls} 的名称:", text=current_name)
        if ok and name:
            if cls >= len(self.class_names):
                while len(self.class_names) <= cls:
                    self.class_names.append(str(len(self.class_names)))
            self.class_names[cls] = name
            if cls < len(self.class_buttons):
                self.class_buttons[cls].setText(name)
            _debug(f"更新类别 {cls} 名称为: {name}")
            self.draw_annotations()
            self.annotation_updated.emit()

    def add_new_class(self):
        """添加新类别"""
        new_cls = len(self.class_buttons)
        btn = QPushButton(str(new_cls))
        btn.setCheckable(True)
        btn.clicked.connect(lambda checked, cls=new_cls: self.select_class(cls))
        btn.mouseDoubleClickEvent = lambda event, cls=new_cls: self.edit_class_name(cls)
        btn.setCursor(Qt.PointingHandCursor)
        # 插入到"+"按钮之前（索引为：1个label + 当前按钮数量）
        self.class_layout.insertWidget(len(self.class_buttons) + 1, btn)
        self.class_buttons.append(btn)
        
        # 更新类别名称列表
        while len(self.class_names) <= new_cls:
            self.class_names.append(str(len(self.class_names)))
        
        _debug(f"添加了新类别: {new_cls}")
        self.annotation_updated.emit()

    def remove_last_class(self):
        """删除最后一个类别"""
        if not self.class_buttons:
            return
        
        last_cls = len(self.class_buttons) - 1
        
        # 检查该类别是否有标注框
        has_annotations = False
        if hasattr(self, 'annotations'):
            for image_path, boxes in self.annotations.items():
                for box in boxes:
                    if box[0] == last_cls:
                        has_annotations = True
                        break
                if has_annotations:
                    break
        
        if has_annotations:
            class_name = self.class_names[last_cls] if last_cls < len(self.class_names) else str(last_cls)
            msg_box = QMessageBox(QMessageBox.Warning, "确认删除",
                f"类别 \"{class_name}\" 仍有标注框，删除后相关标注也会被移除。\n确定要删除吗？",
                QMessageBox.NoButton, self)
            yes_btn = msg_box.addButton("是", QMessageBox.YesRole)
            no_btn = msg_box.addButton("否", QMessageBox.NoRole)
            msg_box.setDefaultButton(no_btn)
            msg_box.exec_()
            if msg_box.clickedButton() != yes_btn:
                return
            
            # 删除该类别的所有标注
            for image_path in list(self.annotations.keys()):
                self.annotations[image_path] = [box for box in self.annotations[image_path] if box[0] != last_cls]
            self.draw_annotations()
        
        # 从布局中移除按钮
        btn = self.class_buttons.pop()
        btn.deleteLater()
        
        # 如果当前选中的是被删除的类别，切换到前一个
        if self.current_class >= len(self.class_buttons):
            self.current_class = max(0, len(self.class_buttons) - 1)
            if self.class_buttons:
                self.class_buttons[self.current_class].setChecked(True)
        
        _debug(f"删除了类别: {last_cls}")
        self.annotation_updated.emit()

    def add_annotation(self):
        """添加标注"""
        if self.current_image is None:
            return

        h, w, _ = self.current_image.shape

        image_coords = self._image_coords_from_label_points(self.start_point, self.end_point)
        if image_coords is None:
            _debug("点击位置不在图片区域内")
            self.draw_annotations()
            return

        x1, y1, x2, y2 = image_coords

        _debug(f"原始坐标: {x1},{y1} - {x2},{y2}")

        center_x = (x1 + x2) / (2 * w)
        center_y = (y1 + y2) / (2 * h)
        width = (x2 - x1) / w
        height = (y2 - y1) / h

        _debug(f"归一化坐标: {center_x},{center_y}, {width},{height}")

        if width > 0 and height > 0:
            cls = self.current_class
            image_path = self.images[self.current_index]
            self.annotations[image_path].append((cls, center_x, center_y, width, height))
            self.draw_annotations()
            self.annotation_updated.emit()

    def clear_annotations(self):
        """清空当前图片的标注"""
        if self.images:
            image_path = self.images[self.current_index]
            self.annotations[image_path] = []
            self.draw_annotations()
            self.annotation_updated.emit()

    def prev_image(self):
        """上一张图片"""
        if self.images and self.current_index > 0:
            self.current_index -= 1
            self.reset_view()
            self.show_current_image()

    def next_image(self):
        """下一张图片"""
        if self.images and self.current_index < len(self.images) - 1:
            self.current_index += 1
            self.reset_view()
            self.show_current_image()

    def select_box(self, pos):
        """选择标注框"""
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
            x1 = (x - width/2) * w
            y1 = (y - height/2) * h
            x2 = (x + width/2) * w
            y2 = (y + height/2) * h

            if x1 <= orig_x <= x2 and y1 <= orig_y <= y2:
                self.selected_box_index = i
                self.draw_annotations()
                return True

        self.selected_box_index = -1
        self.draw_annotations()
        return False

    def delete_selected_box(self):
        """删除选中的标注框"""
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

    def on_key_press(self, event):
        """键盘事件处理"""
        if event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
            if self.selected_box_index != -1:
                self.delete_selected_box()
                _debug("通过键盘删除选中的标注框")
        elif Qt.Key_0 <= event.key() <= Qt.Key_9:
            cls = event.key() - Qt.Key_0
            self.select_class(cls)
            _debug(f"通过键盘选择类别: {cls}")

    def update_class_name(self, cls, name):
        """更新类别名称"""
        try:
            if cls >= 0:
                while len(self.class_names) <= cls:
                    self.class_names.append(str(len(self.class_names)))
                self.class_names[cls] = name
                _debug(f"更新类别 {cls} 名称为: {name}")
        except Exception as e:
            _debug(f"更新类别名称时出错: {e}")

    def set_annotation_mode(self, enabled):
        """设置标注模式是否启用"""
        for i in range(self.control_layout.count()):
            widget = self.control_layout.itemAt(i).widget()
            if widget:
                if widget.text() in ["清空标注", "删除选中"]:
                    widget.setVisible(enabled)
                else:
                    widget.setVisible(True)
        # 非标注模式下隐藏整个类别选择栏
        self.class_scroll_area.setVisible(enabled)
