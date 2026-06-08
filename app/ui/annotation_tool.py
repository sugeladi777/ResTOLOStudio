from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QScrollArea, QPushButton, QInputDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QPalette
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
import cv2
import numpy as np
import os

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

    def _connect_signals(self):
        self.prev_btn.clicked.connect(self.prev_image)
        self.next_btn.clicked.connect(self.next_image)
        self.clear_btn.clicked.connect(self.clear_annotations)
        self.delete_btn.clicked.connect(self.delete_selected_box)
        self.reset_view_btn.clicked.connect(self.reset_view)

    def _install_event_filters(self):
        self.scroll_area.installEventFilter(self)
        self.image_label.installEventFilter(self)

    def load_images(self, image_paths):
        """加载图片"""
        self.images = image_paths
        self.current_index = 0
        self.annotations = {path: [] for path in image_paths}
        self.reset_view()
        self.show_current_image()

    def reset_view(self):
        """重置视图到默认状态"""
        self.zoom_level = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self.is_panning = False
        if self.current_image is not None:
            self.draw_annotations()

    def load_annotations(self, annotation_paths):
        """加载标注文件"""
        for ann_path in annotation_paths:
            image_exts = [".jpg", ".jpeg", ".png", ".bmp"]
            found = False

            for ext in image_exts:
                image_path = os.path.splitext(ann_path)[0] + ext
                if image_path in self.images:
                    _debug(f"找到匹配的图片: {image_path}")
                    with open(ann_path, 'r') as f:
                        lines = f.readlines()
                        boxes = []
                        for line in lines:
                            parts = line.strip().split()
                            if not parts or (len(parts) >= 2 and parts[0] == '['):
                                continue
                            if len(parts) >= 5:
                                cls = int(parts[0])
                                x = float(parts[1])
                                y = float(parts[2])
                                w = float(parts[3])
                                h = float(parts[4])
                                boxes.append((cls, x, y, w, h))
                        self.annotations[image_path] = boxes
                        _debug(f"加载了 {len(boxes)} 个标注")
                    found = True
                    break

            if not found:
                _debug(f"未找到匹配的图片: {ann_path}")
                base_name = os.path.basename(ann_path)
                for image_path in self.images:
                    if os.path.basename(image_path).startswith(os.path.splitext(base_name)[0]):
                        _debug(f"通过文件名匹配找到图片: {image_path}")
                        with open(ann_path, 'r') as f:
                            lines = f.readlines()
                            boxes = []
                            for line in lines:
                                parts = line.strip().split()
                                if not parts or (len(parts) >= 2 and parts[0] == '['):
                                    continue
                                if len(parts) >= 5:
                                    cls = int(parts[0])
                                    x = float(parts[1])
                                    y = float(parts[2])
                                    w = float(parts[3])
                                    h = float(parts[4])
                                    boxes.append((cls, x, y, w, h))
                            self.annotations[image_path] = boxes
                            _debug(f"加载了 {len(boxes)} 个标注")
                        found = True
                        break
        self.show_current_image()

    def save_annotations(self, directory):
        """保存标注"""
        try:
            _debug(f"接收到保存标注请求，目录: {directory}")

            if not directory:
                _debug("保存目录为空")
                return

            if not os.path.exists(directory):
                try:
                    os.makedirs(directory)
                    _debug(f"已创建目录: {directory}")
                except Exception as e:
                    _debug(f"创建目录失败: {e}")
                    return

            _debug(f"开始保存标注到目录: {directory}")
            _debug(f"annotations字典内容: {self.annotations}")
            _debug(f"共有 {len(self.annotations)} 个图片的标注需要保存")

            _debug(f"images列表长度: {len(self.images)}")
            if self.images:
                _debug(f"当前图片: {self.images[self.current_index]}")

            saved_count = 0
            for image_path, boxes in self.annotations.items():
                try:
                    _debug(f"处理图片: {image_path}")
                    _debug(f"图片路径是否存在: {os.path.exists(image_path)}")

                    base_name = os.path.basename(image_path)
                    _debug(f"图片基本名称: {base_name}")

                    ann_path = os.path.join(directory, os.path.splitext(base_name)[0] + ".txt")
                    _debug(f"生成的标注文件路径: {ann_path}")

                    _debug(f"该图片有 {len(boxes)} 个标注")
                    if boxes:
                        _debug(f"第一个标注: {boxes[0]}")

                    try:
                        with open(ann_path, 'w') as f:
                            for box in boxes:
                                if len(box) >= 5:
                                    cls, x, y, w, h = box
                                    line = f"{cls} {x} {y} {w} {h}\n"
                                    _debug(f"写入行: {line.strip()}")
                                    f.write(line)
                                else:
                                    _debug(f"标注格式错误: {box}")
                        _debug(f"成功写入文件: {ann_path}")
                        if os.path.exists(ann_path):
                            _debug(f"文件已创建，大小: {os.path.getsize(ann_path)} 字节")
                        else:
                            _debug("文件创建失败")
                        saved_count += 1
                    except Exception as e:
                        _debug(f"写入文件时出错: {e}")
                except Exception as e:
                    _debug(f"处理图片时出错: {e}")

            try:
                import yaml
                used_classes = set()
                for boxes in self.annotations.values():
                    for box in boxes:
                        if len(box) >= 1:
                            cls = box[0]
                            used_classes.add(cls)

                used_class_names = [self.class_names[cls] for cls in sorted(used_classes)]
                classes_yaml = os.path.join(directory, 'classes.yaml')
                classes_info = {
                    'names': used_class_names,
                    'nc': len(used_class_names)
                }
                with open(classes_yaml, 'w', encoding='utf-8') as f:
                    yaml.dump(classes_info, f, default_flow_style=False, allow_unicode=True)
                _debug(f"成功生成类别信息文件: {classes_yaml}")
                _debug(f"实际使用的类别: {used_class_names}")
                _debug(f"类别数: {len(used_class_names)}")
            except Exception as e:
                _debug(f"生成类别信息文件失败: {e}")

            _debug(f"保存完成，共保存了 {saved_count} 个标注文件")
        except Exception as e:
            _debug(f"保存标注过程中发生错误: {e}")

    def show_current_image(self):
        """显示当前图片"""
        if not self.images:
            return

        image_path = self.images[self.current_index]

        if not os.path.exists(image_path):
            return

        # 使用PIL读取图片，更好地支持中文路径
        try:
            from PIL import Image
            pil_image = Image.open(image_path)
            # 转换为RGB模式
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            # 转换为numpy数组
            import numpy as np
            self.current_image = np.array(pil_image)
        except Exception as e:
            _debug(f"使用PIL读取图片失败: {e}")
            # 尝试使用cv2作为备选
            try:
                self.current_image = cv2.imread(image_path)
                if self.current_image is not None:
                    self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            except Exception as e2:
                _debug(f"使用cv2读取图片失败: {e2}")
                return

        if self.current_image is None:
            return

        self.draw_annotations()

    def draw_annotations(self):
        """绘制标注"""
        if self.current_image is None:
            return

        image_copy = self.current_image.copy()
        h, w, _ = image_copy.shape

        colors = [
            (0, 255, 0),
            (0, 0, 255),
            (255, 0, 0),
            (255, 255, 0),
            (0, 255, 255),
            (255, 0, 255),
            (128, 0, 0),
            (0, 128, 0),
            (0, 0, 128),
            (128, 128, 0)
        ]

        if self.images and self.current_index < len(self.images):
            image_path = self.images[self.current_index]
            boxes = self.annotations.get(image_path, [])
            _debug(f"绘制标注: {len(boxes)} 个标注")
            for i, box in enumerate(boxes):
                cls, x, y, width, height = box
                x1 = int((x - width/2) * w)
                y1 = int((y - height/2) * h)
                x2 = int((x + width/2) * w)
                y2 = int((y + height/2) * h)

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                _debug(f"标注坐标: {x1},{y1} - {x2},{y2}")

                color = colors[cls % len(colors)]

                line_width = max(1, min(int(w * 0.005), int(h * 0.005)))

                font_scale = max(0.4, min(w, h) * 0.001)
                font_thickness = max(1, int(min(w, h) * 0.002))

                if i == self.selected_box_index:
                    cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 255), line_width + max(1, line_width // 2))
                else:
                    cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, line_width)

                class_name = self.class_names[cls] if cls < len(self.class_names) else str(cls)
                cv2.putText(image_copy, class_name, (x1, max(30, y1-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)

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

        colors = [
            (0, 255, 0),
            (0, 0, 255),
            (255, 0, 0),
            (255, 255, 0),
            (0, 255, 255),
            (255, 0, 255),
            (128, 0, 0),
            (0, 128, 0),
            (0, 0, 128),
            (128, 128, 0)
        ]

        if self.images and self.current_index < len(self.images):
            image_path = self.images[self.current_index]
            boxes = self.annotations.get(image_path, [])
            for i, box in enumerate(boxes):
                cls, x, y, width, height = box
                x1 = int((x - width/2) * w)
                y1 = int((y - height/2) * h)
                x2 = int((x + width/2) * w)
                y2 = int((y + height/2) * h)

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                color = colors[cls % len(colors)]

                line_width = max(1, min(int(w * 0.005), int(h * 0.005)))

                if i == self.selected_box_index:
                    cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 255), line_width + max(1, line_width // 2))
                else:
                    cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, line_width)

                font_scale = max(0.4, min(w, h) * 0.001)
                font_thickness = max(1, int(min(w, h) * 0.002))
                class_name = self.class_names[cls] if cls < len(self.class_names) else str(cls)
                cv2.putText(image_copy, class_name, (x1, max(30, y1-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)

        if not self.start_point.isNull() and not self.end_point.isNull():
            label_width = self.image_label.width()
            label_height = self.image_label.height()
            pixmap = self.image_label.pixmap()

            if pixmap:
                pixmap_width = pixmap.width()
                pixmap_height = pixmap.height()

                offset_x = (label_width - pixmap_width) // 2
                offset_y = (label_height - pixmap_height) // 2

                start_x = self.start_point.x() - offset_x
                start_y = self.start_point.y() - offset_y
                end_x = self.end_point.x() - offset_x
                end_y = self.end_point.y() - offset_y

                if start_x >= 0 and start_y >= 0 and end_x <= pixmap_width and end_y <= pixmap_height:
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

                    scale_x = final_width / w
                    scale_y = final_height / h

                    temp_x1 = int(min(start_x, end_x) / scale_x)
                    temp_y1 = int(min(start_y, end_y) / scale_y)
                    temp_x2 = int(max(start_x, end_x) / scale_x)
                    temp_y2 = int(max(start_y, end_y) / scale_y)

                    temp_x1 = max(0, temp_x1)
                    temp_y1 = max(0, temp_y1)
                    temp_x2 = min(w, temp_x2)
                    temp_y2 = min(h, temp_y2)

                    line_width = max(1, min(int(w * 0.005), int(h * 0.005)))
                    cv2.rectangle(image_copy, (temp_x1, temp_y1), (temp_x2, temp_y2), (0, 0, 255), line_width)

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

        label_width = self.image_label.width()
        label_height = self.image_label.height()

        if label_width == 0 or label_height == 0:
            return

        pixmap = self.image_label.pixmap()
        if not pixmap:
            return

        pixmap_width = pixmap.width()
        pixmap_height = pixmap.height()

        offset_x = (label_width - pixmap_width) // 2
        offset_y = (label_height - pixmap_height) // 2

        start_x = self.start_point.x() - offset_x
        start_y = self.start_point.y() - offset_y
        end_x = self.end_point.x() - offset_x
        end_y = self.end_point.y() - offset_y

        if start_x < 0 or start_y < 0 or end_x > pixmap_width or end_y > pixmap_height:
            _debug("点击位置不在图片区域内")
            self.draw_annotations()
            return

        max_width = self.scroll_area.width()
        max_height = self.scroll_area.height()
        if max_width <= 0:
            max_width = 800
        if max_height <= 0:
            max_height = 600

        aspect_pixmap = self.image_label.pixmap().scaled(max_width, max_height, Qt.KeepAspectRatio)

        scale_x = w / aspect_pixmap.width()
        scale_y = h / aspect_pixmap.height()

        zoom_scale_x = scale_x / self.zoom_level
        zoom_scale_y = scale_y / self.zoom_level

        x1 = int(min(start_x, end_x) * zoom_scale_x)
        y1 = int(min(start_y, end_y) * zoom_scale_y)
        x2 = int(max(start_x, end_x) * zoom_scale_x)
        y2 = int(max(start_y, end_y) * zoom_scale_y)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

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

        label_width = self.image_label.width()
        label_height = self.image_label.height()

        pixmap = self.image_label.pixmap()
        if not pixmap:
            return False

        pixmap_width = pixmap.width()
        pixmap_height = pixmap.height()

        offset_x = (label_width - pixmap_width) // 2
        offset_y = (label_height - pixmap_height) // 2

        click_x = pos.x() - offset_x
        click_y = pos.y() - offset_y

        if click_x < 0 or click_y < 0 or click_x > pixmap_width or click_y > pixmap_height:
            self.selected_box_index = -1
            self.draw_annotations()
            return False

        h, w, _ = self.current_image.shape

        max_width = self.scroll_area.width()
        max_height = self.scroll_area.height()
        if max_width <= 0:
            max_width = 800
        if max_height <= 0:
            max_height = 600

        aspect_pixmap = pixmap.scaled(max_width, max_height, Qt.KeepAspectRatio)

        scale_x = w / aspect_pixmap.width()
        scale_y = h / aspect_pixmap.height()

        zoom_scale_x = scale_x / self.zoom_level
        zoom_scale_y = scale_y / self.zoom_level

        orig_x = click_x * zoom_scale_x
        orig_y = click_y * zoom_scale_y

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
