"""
SXM文件解析器 - 用于读取Nanonis扫描探针显微镜数据文件

SXM文件格式：
- 头部：文本格式，以 :KEY: 开头的键值对
- 数据：在 :SCANIT_END: 标记之后，以二进制格式存储（默认FLOAT MSBFIRST）
- 多通道数据按顺序排列

关键元数据：
- SCAN_PIXELS: 扫描像素数 (width, height)
- SCAN_RANGE: 扫描范围 (x, y)，单位：米
- SCAN_OFFSET: 扫描偏移 (x, y)，单位：米
- DATA_INFO: 通道信息（通道号、名称、单位、方向、校准值、偏移）
"""
import numpy as np
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap


def _create_nanonis_colormap():
    """创建Nanonis (Bruker Nanoscope) 风格的色图"""
    cdict = {
        "red": (
            (0.0, 0.0, 0.0),
            (0.124464, 0.0, 0.0),
            (0.236052, 0.0670103, 0.0670103),
            (0.371245, 0.253338, 0.253338),
            (0.472103, 0.392344, 0.392344),
            (0.611588, 0.584587, 0.584587),
            (0.708155, 0.717678, 0.717678),
            (0.714052, 0.725806, 0.725806),
            (0.890558, 0.969072, 0.969072),
            (0.933476, 0.987464, 0.987464),
            (0.944709, 0.992278, 0.992278),
            (0.965682, 0.995207, 0.995207),
            (0.971401, 0.996006, 0.996006),
            (1, 1, 1),
        ),
        "green": (
            (0.0, 0.0, 0.0),
            (0.124464, 0.0, 0.0),
            (0.236052, 0.0, 0.0),
            (0.371245, 0.0, 0.0),
            (0.472103, 0.0721649, 0.0721649),
            (0.611588, 0.334114, 0.334114),
            (0.708155, 0.515464, 0.515464),
            (0.714052, 0.527471, 0.527471),
            (0.890558, 0.886843, 0.886843),
            (0.933476, 0.974227, 0.974227),
            (0.944709, 0.980523, 0.980523),
            (0.965682, 0.992278, 0.992278),
            (0.971401, 0.993565, 0.993565),
            (1, 1, 1),
        ),
        "blue": (
            (0.0, 0.0, 0.0),
            (0.124464, 0.0, 0.0),
            (0.236052, 0.0, 0.0),
            (0.371245, 0.0, 0.0),
            (0.472103, 0.0, 0.0),
            (0.611588, 0.0, 0.0),
            (0.708155, 0.252575, 0.252575),
            (0.714052, 0.268, 0.268),
            (0.890558, 0.76343, 0.76343),
            (0.933476, 0.883897, 0.883897),
            (0.944709, 0.915426, 0.915426),
            (0.965682, 0.974293, 0.974293),
            (0.971401, 0.990347, 0.990347),
            (1, 1, 1),
        ),
    }
    return LinearSegmentedColormap("nanonis", cdict)


# 全局Nanonis色图实例
_NANONIS_CMAP = _create_nanonis_colormap()


class SXMFile:
    """Nanonis SXM文件解析器"""

    def __init__(self, filepath):
        self.filepath = filepath
        self.header = {}
        self.channels = []  # 通道信息列表
        self.data = {}  # 通道名 -> numpy数组
        self._parse()

    def _parse(self):
        """解析SXM文件"""
        with open(self.filepath, 'rb') as f:
            # 解析头部
            self._parse_header(f)
            # 解析通道信息
            self._parse_data_info()
            # 解析二进制数据
            self._parse_data(f)

    def _parse_header(self, f):
        """解析头部键值对"""
        current_key = None
        current_value = []

        while True:
            line = f.readline()
            if not line:
                break

            # 尝试解码为文本
            try:
                line_str = line.decode('ascii', errors='replace').rstrip('\r\n')
            except Exception:
                break

            # 检查是否到达数据区
            if line_str.strip() == ':SCANIT_END:':
                # 保存最后一个键值对
                if current_key:
                    self.header[current_key] = '\n'.join(current_value)
                break

            # 检查是否是新键
            if line_str.startswith(':') and line_str.endswith(':'):
                # 保存上一个键值对
                if current_key:
                    self.header[current_key] = '\n'.join(current_value)
                current_key = line_str.strip(':')
                current_value = []
            else:
                if current_key:
                    current_value.append(line_str.strip())

    def _parse_data_info(self):
        """解析DATA_INFO字段，提取通道信息"""
        if 'DATA_INFO' not in self.header:
            return

        lines = self.header['DATA_INFO'].strip().split('\n')
        # 第一行是表头，跳过
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) >= 6:
                channel_info = {
                    'channel': int(parts[0]),
                    'name': parts[1],
                    'unit': parts[2],
                    'direction': parts[3],  # both, forward, backward
                    'calibration': float(parts[4]),
                    'offset': float(parts[5])
                }
                self.channels.append(channel_info)

    def _parse_data(self, f):
        """解析二进制数据区"""
        # 获取扫描像素数
        pixels = self.header.get('SCAN_PIXELS', '256 256').strip().split()
        width = int(pixels[0])
        height = int(pixels[1])

        # 获取数据格式
        scanit_type = self.header.get('SCANIT_TYPE', 'FLOAT MSBFIRST').strip()
        if 'FLOAT' in scanit_type:
            dtype = '>f4' if 'MSBFIRST' in scanit_type else '<f4'
        else:
            dtype = '>f4'  # 默认大端浮点

        # 每个通道有forward和backward两个方向的数据
        # direction为"both"的通道有两份数据，其他只有一份数据
        num_arrays = 0
        for ch in self.channels:
            if ch['direction'] == 'both':
                num_arrays += 2  # forward + backward
            else:
                num_arrays += 1

        total_floats = width * height * num_arrays
        raw_data = f.read(total_floats * 4)  # 4 bytes per float

        if len(raw_data) < total_floats * 4:
            # 数据不完整，尝试只读取可用的数据
            total_floats = len(raw_data) // 4

        data_array = np.frombuffer(raw_data, dtype=dtype, count=total_floats)

        # 按通道拆分数据
        idx = 0
        for ch in self.channels:
            n_pixels = width * height
            if ch['direction'] == 'both':
                # Forward数据
                if idx + n_pixels <= len(data_array):
                    forward = data_array[idx:idx + n_pixels].reshape((height, width)).astype(np.float64)
                    # 应用校准
                    forward = forward * ch['calibration'] + ch['offset']
                    # 翻转Y轴，匹配Nanonis显示方向
                    forward = np.flipud(forward)
                    self.data[f"{ch['name']}_forward"] = forward
                idx += n_pixels
                # Backward数据
                if idx + n_pixels <= len(data_array):
                    backward = data_array[idx:idx + n_pixels].reshape((height, width)).astype(np.float64)
                    backward = backward * ch['calibration'] + ch['offset']
                    # 翻转Y轴，匹配Nanonis显示方向
                    backward = np.flipud(backward)
                    self.data[f"{ch['name']}_backward"] = backward
                idx += n_pixels
            else:
                if idx + n_pixels <= len(data_array):
                    arr = data_array[idx:idx + n_pixels].reshape((height, width)).astype(np.float64)
                    arr = arr * ch['calibration'] + ch['offset']
                    # 翻转Y轴，匹配Nanonis显示方向
                    arr = np.flipud(arr)
                    self.data[ch['name']] = arr
                idx += n_pixels

    def get_channel_names(self):
        """获取所有可用的通道名称"""
        return list(self.data.keys())

    def get_channel_image(self, channel_name=None):
        """
        获取指定通道的PIL Image对象（8位灰度图）
        如果未指定通道，返回第一个通道
        """
        if channel_name is None:
            if not self.data:
                return None
            channel_name = list(self.data.keys())[0]

        if channel_name not in self.data:
            return None

        arr = self.data[channel_name]
        return self._array_to_image(arr)

    def get_channel_image_rgb(self, channel_name=None):
        """获取指定通道的PIL Image对象（RGB图，应用伪彩色）"""
        if channel_name is None:
            if not self.data:
                return None
            channel_name = list(self.data.keys())[0]

        if channel_name not in self.data:
            return None

        arr = self.data[channel_name]
        return self._array_to_rgb_image(arr)

    def _array_to_image(self, arr):
        """将numpy数组转为8位灰度PIL Image"""
        # 处理NaN和Inf
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        # 归一化到0-255
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val > min_val:
            normalized = ((arr - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(arr, dtype=np.uint8)
        return Image.fromarray(normalized, mode='L')

    def _array_to_rgb_image(self, arr):
        """将numpy数组转为RGB伪彩色PIL Image"""
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val > min_val:
            normalized = (arr - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(arr)

        # 应用inferno色图
        import matplotlib.cm as cm
        colored = cm.inferno(normalized)[:, :, :3]  # 去掉alpha通道
        rgb_array = (colored * 255).astype(np.uint8)
        return Image.fromarray(rgb_array, mode='RGB')

    def get_scan_range_nm(self):
        """获取扫描范围，单位nm"""
        range_str = self.header.get('SCAN_RANGE', '0 0').strip().split()
        x_range = float(range_str[0]) * 1e9  # m -> nm
        y_range = float(range_str[1]) * 1e9
        return x_range, y_range

    def get_pixel_size_nm(self):
        """获取每个像素的物理尺寸，单位nm"""
        x_range, y_range = self.get_scan_range_nm()
        pixels = self.header.get('SCAN_PIXELS', '256 256').strip().split()
        width = int(pixels[0])
        height = int(pixels[1])
        return x_range / width, y_range / height

    def get_metadata_summary(self):
        """获取元数据摘要"""
        x_range, y_range = self.get_scan_range_nm()
        px_x, px_y = self.get_pixel_size_nm()
        bias = self.header.get('BIAS', 'N/A').strip()

        summary = f"扫描范围: {x_range:.2f} x {y_range:.2f} nm\n"
        summary += f"像素尺寸: {px_x:.4f} x {px_y:.4f} nm/pixel\n"
        summary += f"偏压: {bias} V\n"
        summary += f"通道: {', '.join(self.get_channel_names())}"
        return summary


def load_sxm_as_image(filepath, channel=None, use_color=True):
    """
    加载SXM文件为PIL Image对象
    
    Args:
        filepath: SXM文件路径
        channel: 通道名称（默认第一个通道）
        use_color: 是否使用伪彩色
    
    Returns:
        (PIL.Image, SXMFile) 元组，图片和解析器对象
    """
    sxm = SXMFile(filepath)
    if use_color:
        img = sxm.get_channel_image_rgb(channel)
    else:
        img = sxm.get_channel_image(channel)
    return img, sxm
