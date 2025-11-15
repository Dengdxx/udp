#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
udp_gui.py - UDP 上位机 GUI（使用 ttkbootstrap 美观主题）

功能概览：
- 运行/停止 UDP 监听（调用 udp_image_logger.py run）
- PNG 保存目录、日志 CSV、帧索引 CSV 配置
- 一键合成视频（video）
- 一键对齐（align）
- 示波器（scope）：按日志包字节索引选择，支持可选 bit

依赖要求：
- 必须安装 ttkbootstrap：pip install ttkbootstrap
"""

import os
import sys
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import socket
import struct
import time
from datetime import datetime
from typing import Optional, Tuple
import csv

# 导入 ttkbootstrap（必需）
try:
    import ttkbootstrap as tb
    from ttkbootstrap import ttk
    from ttkbootstrap.scrolled import ScrolledText as TBScrolledText
except ImportError:
    print("=" * 60)
    print("错误：未安装 ttkbootstrap")
    print("请运行以下命令安装：")
    print("    pip install ttkbootstrap")
    print("=" * 60)
    sys.exit(1)

# 导入图像处理库
try:
    import cv2
    import numpy as np
    from PIL import Image, ImageTk
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("[WARN] OpenCV or PIL not available, video display disabled")

# ---------------------- 辅助函数 ----------------------

def sanitize_csv_text(text: str) -> str:
    """清理CSV文本,移除会导致读取问题的特殊字符
    
    Args:
        text: 原始文本
    
    Returns:
        清理后的文本,移除NULL、EOF等特殊字符
    """
    if not text:
        return text
    
    # 移除NULL字符(0x00)和EOF字符(0x1A)
    # 这些字符会导致C++的std::getline提前终止
    text = text.replace('\x00', '')  # NULL
    text = text.replace('\x1A', '')  # EOF/SUB (Ctrl+Z)
    
    # 可选:也移除其他控制字符(保留换行、制表等常用字符)
    # 移除 0x01-0x08, 0x0B-0x0C, 0x0E-0x1F (保留 \t=0x09, \n=0x0A, \r=0x0D)
    cleaned = ''.join(c for c in text if ord(c) >= 0x20 or c in '\t\n\r')
    
    return cleaned

# ---------------------- 辅助函数(继续) ----------------------

# 导入matplotlib用于示波器
try:
    import matplotlib
    matplotlib.use('TkAgg')
    # 设置中文字体支持
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    from collections import deque
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARN] Matplotlib not available, oscilloscope disabled")


# 获取本机所有 IP 地址
def get_local_ips():
    """获取本机所有可用的 IP 地址"""
    ips = ['0.0.0.0', '127.0.0.1']  # 默认选项
    try:
        hostname = socket.gethostname()
        # 获取所有 IP 地址
        for info in socket.getaddrinfo(hostname, None):
            ip = info[4][0]
            if ':' not in ip:  # 只保留 IPv4
                if ip not in ips:
                    ips.append(ip)
    except Exception as e:
        print(f"[WARN] Failed to get local IPs: {e}")
    return ips

# Switch 控件
Switch = getattr(tb, 'Switch', None)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_SCRIPT = os.path.join(SCRIPT_DIR, 'udp_image_logger.py')


# ---------------------- 帧解析工具类 ----------------------
class FrameType:
    IMAGE = 0x01
    LOG = 0x02
    BINARY_IMAGE = 0x03


def parse_image_frame(data: bytes) -> Tuple[int, int, np.ndarray]:
    """解析图像帧 [0x01][H][W][pixels...]"""
    if len(data) < 3:
        raise ValueError("image frame too short")
    h = data[1]
    w = data[2]
    expected = 1 + 1 + 1 + (h * w)
    if len(data) != expected:
        raise ValueError(f"image frame size mismatch: got {len(data)}, expect {expected}")
    pixels = np.frombuffer(data[3:], dtype=np.uint8) if HAS_CV2 else None
    if pixels is not None:
        img = pixels.reshape((h, w))
        return h, w, img
    return h, w, None


def parse_binary_image_frame(data: bytes) -> Tuple[int, int, np.ndarray]:
    """解析二值图像帧 [0x03][H][W][compressed_pixels...]"""
    if len(data) < 3:
        raise ValueError("binary image frame too short")
    h = data[1]
    w = data[2]
    pixel_count = h * w
    expected_bytes = 1 + 1 + 1 + ((pixel_count + 7) // 8)
    if len(data) < expected_bytes:
        raise ValueError(f"binary image frame size mismatch")
    
    if not HAS_CV2:
        return h, w, None
    
    compressed = data[3:]
    img = np.zeros(pixel_count, dtype=np.uint8)
    
    for i in range(pixel_count):
        byte_idx = i // 8
        bit_idx = i % 8
        if byte_idx < len(compressed):
            bit_val = (compressed[byte_idx] >> (7 - bit_idx)) & 0x01
            img[i] = 255 if bit_val else 0
    
    img = img.reshape((h, w))
    return h, w, img


def parse_compressed_image_frame(data: bytes, fixed_h: int = 0, fixed_w: int = 0) -> Tuple[int, int, np.ndarray]:
    """解析STM32压缩图像帧 [FH-4bytes][compressed_pixels][FE-4bytes]
    
    压缩格式: 8个像素压缩成1字节, MSB优先
    - bit7对应第1个像素, bit0对应第8个像素
    - bit=1表示白色(255), bit=0表示黑色(0)
    
    Args:
        data: 原始数据包(包含帧头帧尾)
        fixed_h: 固定高度(从配置读取)
        fixed_w: 固定宽度(从配置读取)
    
    Returns:
        (height, width, image_array)
    """
    if not HAS_CV2:
        return fixed_h, fixed_w, None
    
    # 最小长度检查: 帧头(4) + 帧尾(4) = 8字节
    if len(data) < 8:
        raise ValueError(f"compressed image frame too short: {len(data)} bytes")
    
    # 提取压缩的像素数据(去除帧头4字节和帧尾4字节)
    compressed = data[4:-4]
    
    # 如果提供了固定尺寸,使用固定尺寸
    if fixed_h > 0 and fixed_w > 0:
        h = fixed_h
        w = fixed_w
    else:
        # 动态计算尺寸(需要额外信息,这里暂不支持)
        raise ValueError("compressed image requires fixed H/W in config")
    
    pixel_count = h * w
    expected_compressed_bytes = (pixel_count + 7) // 8
    
    # 验证压缩数据长度
    if len(compressed) < expected_compressed_bytes:
        raise ValueError(
            f"compressed data too short: got {len(compressed)} bytes, "
            f"expected {expected_compressed_bytes} for {h}x{w} image"
        )
    
    # 解压缩: 1字节 -> 8像素
    img = np.zeros(pixel_count, dtype=np.uint8)
    
    for idx in range(pixel_count):
        byte_idx = idx // 8
        bit_offset = idx % 8
        
        # MSB优先: bit7对应第0个像素, bit0对应第7个像素
        bit_val = (compressed[byte_idx] >> (7 - bit_offset)) & 0x01
        img[idx] = 255 if bit_val else 0
    
    img = img.reshape((h, w))
    return h, w, img


def decode_image_by_format(pixel_data: bytes, h: int, w: int, format_type: str) -> np.ndarray:
    """根据指定格式解码图像数据
    
    Args:
        pixel_data: 原始像素数据(不含帧头帧尾、H/W字段)
        h: 图像高度
        w: 图像宽度
        format_type: 图像格式类型
    
    Returns:
        解码后的图像数组 (H, W, C) 或 (H, W)
    """
    if not HAS_CV2:
        return None
    
    pixel_count = h * w
    
    if format_type == '灰度图(8位)':
        # 每个像素1字节,直接读取
        if len(pixel_data) < pixel_count:
            raise ValueError(f"gray image data too short: need {pixel_count}, got {len(pixel_data)}")
        img = np.frombuffer(pixel_data[:pixel_count], dtype=np.uint8)
        return img.reshape((h, w))
    
    elif format_type == '二值图(8位)':
        # 每个像素1字节, 0或255
        if len(pixel_data) < pixel_count:
            raise ValueError(f"binary image data too short: need {pixel_count}, got {len(pixel_data)}")
        pixels = np.frombuffer(pixel_data[:pixel_count], dtype=np.uint8)
        # 转换为二值(非0即255)
        img = np.where(pixels > 127, 255, 0).astype(np.uint8)
        return img.reshape((h, w))
    
    elif format_type == '压缩二值(1位)':
        # 8个像素压缩成1字节, MSB优先
        expected_bytes = (pixel_count + 7) // 8
        if len(pixel_data) < expected_bytes:
            raise ValueError(f"compressed binary data too short: need {expected_bytes}, got {len(pixel_data)}")
        
        img = np.zeros(pixel_count, dtype=np.uint8)
        for idx in range(pixel_count):
            byte_idx = idx // 8
            bit_offset = idx % 8
            bit_val = (pixel_data[byte_idx] >> (7 - bit_offset)) & 0x01
            img[idx] = 255 if bit_val else 0
        return img.reshape((h, w))
    
    elif format_type == 'RGB565':
        # 每个像素2字节, RGB565格式
        expected_bytes = pixel_count * 2
        if len(pixel_data) < expected_bytes:
            raise ValueError(f"RGB565 data too short: need {expected_bytes}, got {len(pixel_data)}")
        
        # 解析RGB565 (小端序: [G2G1G0B4B3B2B1B0][R4R3R2R1R0G5G4G3])
        rgb565 = np.frombuffer(pixel_data[:expected_bytes], dtype=np.uint16)
        
        # 提取RGB分量
        r = ((rgb565 & 0xF800) >> 11).astype(np.uint8)  # 5位R
        g = ((rgb565 & 0x07E0) >> 5).astype(np.uint8)   # 6位G
        b = (rgb565 & 0x001F).astype(np.uint8)          # 5位B
        
        # 扩展到8位 (5位->8位: x8+x3, 6位->8位: x4+x2)
        r = (r << 3) | (r >> 2)
        g = (g << 2) | (g >> 4)
        b = (b << 3) | (b >> 2)
        
        # 组合成RGB图像
        img = np.stack([r, g, b], axis=-1)
        return img.reshape((h, w, 3))
    
    elif format_type == 'RGB888':
        # 每个像素3字节, RGB888格式
        expected_bytes = pixel_count * 3
        if len(pixel_data) < expected_bytes:
            raise ValueError(f"RGB888 data too short: need {expected_bytes}, got {len(pixel_data)}")
        
        img = np.frombuffer(pixel_data[:expected_bytes], dtype=np.uint8)
        return img.reshape((h, w, 3))
    
    elif format_type == 'BGR888':
        # 每个像素3字节, BGR888格式
        expected_bytes = pixel_count * 3
        if len(pixel_data) < expected_bytes:
            raise ValueError(f"BGR888 data too short: need {expected_bytes}, got {len(pixel_data)}")
        
        img = np.frombuffer(pixel_data[:expected_bytes], dtype=np.uint8)
        img = img.reshape((h, w, 3))
        # BGR转RGB
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    elif format_type == 'RGBA8888':
        # 每个像素4字节, RGBA8888格式
        expected_bytes = pixel_count * 4
        if len(pixel_data) < expected_bytes:
            raise ValueError(f"RGBA8888 data too short: need {expected_bytes}, got {len(pixel_data)}")
        
        img = np.frombuffer(pixel_data[:expected_bytes], dtype=np.uint8)
        img = img.reshape((h, w, 4))
        # 只取RGB通道
        return img[:, :, :3]
    
    else:
        raise ValueError(f"Unknown image format: {format_type}")


def parse_log_frame(data: bytes) -> bytes:
    """解析日志帧 [0x02][LEN][payload...]"""
    if len(data) < 1 + 1:
        raise ValueError("log frame too short")
    length = data[1]
    if len(data) != 1 + 1 + length:
        raise ValueError("log frame size mismatch")
    payload = data[2:2 + length]
    return payload


# ---------------------- UDP 接收线程 ----------------------
class UdpVideoReceiver:
    """UDP 视频接收器，在后台线程接收并更新图像"""
    
    def __init__(self, ip: str, port: int, save_png: bool = False, 
                 png_dir: str = 'frames_png',
                 log_csv: str = 'logs.csv',
                 frame_index_csv: str = 'frames_index.csv',
                 enable_custom_image_frame: bool = False,
                 image_frame_header: str = '',
                 image_frame_footer: str = '',
                 image_h_bytes: int = 1,
                 image_w_bytes: int = 1,
                 image_h_order: str = '小端',
                 image_w_order: str = '小端',
                 image_fixed_h: int = 0,  # 固定高度 (0表示从数据解析)
                 image_fixed_w: int = 0,  # 固定宽度 (0表示从数据解析)
                 image_format: str = '灰度图(8位)',  # 图像数据编码格式
                 enable_custom_log_frame: bool = False,
                 log_frame_header: str = '',
                 log_frame_footer: str = '',
                 log_frame_format: str = '标准格式',
                 log_callback = None,  # 添加日志回调
                 log_variables = None):  # 日志变量配置列表
        self.ip = ip
        self.port = port
        self.save_png = save_png
        self.png_dir = png_dir
        self.log_csv = log_csv
        self.frame_index_csv = frame_index_csv
        
        # 自定义图像帧格式
        self.enable_custom_image_frame = enable_custom_image_frame
        self.image_frame_header_bytes = bytes.fromhex(image_frame_header.replace(' ', '')) if image_frame_header else b''
        self.image_frame_footer_bytes = bytes.fromhex(image_frame_footer.replace(' ', '')) if image_frame_footer else b''
        self.image_h_bytes = image_h_bytes
        self.image_w_bytes = image_w_bytes
        self.image_h_order = image_h_order
        self.image_w_order = image_w_order
        self.image_fixed_h = image_fixed_h
        self.image_fixed_w = image_fixed_w
        self.image_format = image_format
        
        # 自定义日志帧格式
        self.enable_custom_log_frame = enable_custom_log_frame
        self.log_frame_header_bytes = bytes.fromhex(log_frame_header.replace(' ', '')) if log_frame_header else b''
        self.log_frame_footer_bytes = bytes.fromhex(log_frame_footer.replace(' ', '')) if log_frame_footer else b''
        self.log_frame_format = log_frame_format
        
        # 日志回调函数
        self.log_callback = log_callback
        
        # 日志变量配置 [(name, byte_pos, data_type, display_format), ...]
        self.log_variables = log_variables if log_variables else []
        
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        
        # 当前帧数据
        self.current_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()
        
        # 统计信息
        self.frame_counter = 0
        self.total_packets = 0
        self.error_packets = 0
        self.fps = 0.0
        self._fps_timer = time.time()
        self._fps_frame_count = 0
        
        # 原始数据缓存（最近的数据包）
        self.recent_data = []  # 存储最近的数据包 [(timestamp, type, data_hex, parsed_info), ...]
        self.max_recent_data = 100  # 最多保存100条
        self.data_lock = threading.Lock()
        
        # 示波器数据缓存
        self.scope_data = {}  # {byte_index: deque([(timestamp, value), ...], maxlen=1000)}
        self.scope_lock = threading.Lock()
        self.scope_start_time = time.time()  # 示波器开始时间
        
        # CSV 文件
        self._log_csv_fp: Optional[object] = None
        self._log_writer: Optional[object] = None
        self._frame_index_fp: Optional[object] = None
        self._frame_index_writer: Optional[object] = None
    
    def start(self):
        """启动 UDP 接收线程"""
        if self._running:
            return False
        
        try:
            # 创建 socket
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.bind((self.ip, self.port))
            self._sock.settimeout(1.0)
            
            # 初始化 CSV
            if self.save_png:
                os.makedirs(self.png_dir, exist_ok=True)
            
            log_csv_exists = os.path.exists(self.log_csv) and os.path.getsize(self.log_csv) > 0
            self._log_csv_fp = open(self.log_csv, 'a', newline='', encoding='utf-8')
            self._log_writer = csv.writer(self._log_csv_fp)
            if not log_csv_exists:
                # 构建动态CSV表头
                csv_header = ["host_recv_iso", "log_text_hex", "log_text_utf8"]
                # 添加配置的日志变量列
                for var_name, _, _, _ in self.log_variables:
                    csv_header.append(var_name)
                self._log_writer.writerow(csv_header)
            
            frame_csv_exists = os.path.exists(self.frame_index_csv) and os.path.getsize(self.frame_index_csv) > 0
            self._frame_index_fp = open(self.frame_index_csv, 'a', newline='', encoding='utf-8')
            self._frame_index_writer = csv.writer(self._frame_index_fp)
            if not frame_csv_exists:
                self._frame_index_writer.writerow(["frame_id", "host_recv_iso", "png_path", "h", "w"])
            
            # 启动线程
            self._stop_event.clear()
            self._running = True
            self._thread = threading.Thread(target=self._receive_loop, daemon=True)
            self._thread.start()
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to start UDP receiver: {e}")
            self._cleanup()
            return False
    
    def stop(self):
        """停止 UDP 接收"""
        if not self._running:
            return
        
        self._stop_event.set()
        self._running = False
        
        if self._thread:
            self._thread.join(timeout=3.0)
        
        self._cleanup()
    
    def _cleanup(self):
        """清理资源"""
        if self._sock:
            try:
                self._sock.close()
            except:
                pass
            self._sock = None
        
        if self._log_csv_fp:
            try:
                self._log_csv_fp.close()
            except:
                pass
            self._log_csv_fp = None
        
        if self._frame_index_fp:
            try:
                self._frame_index_fp.close()
            except:
                pass
            self._frame_index_fp = None
    
    def _update_fps(self):
        """更新 FPS 统计"""
        self._fps_frame_count += 1
        elapsed = time.time() - self._fps_timer
        if elapsed >= 1.0:
            self.fps = self._fps_frame_count / elapsed
            self._fps_frame_count = 0
            self._fps_timer = time.time()
    
    def _parse_custom_frame(self, data: bytes) -> Optional[bytes]:
        """解析自定义帧格式，返回帧数据（不含帧头帧尾）"""
        if not self.enable_custom_image_frame or not self.image_frame_header_bytes:
            return None
        
        # 查找帧头
        header_pos = data.find(self.image_frame_header_bytes)
        if header_pos == -1:
            return None
        
        # 提取数据起始位置
        data_start = header_pos + len(self.image_frame_header_bytes)
        
        # 如果有帧尾，查找帧尾
        if self.image_frame_footer_bytes:
            footer_pos = data.find(self.image_frame_footer_bytes, data_start)
            if footer_pos == -1:
                return None
            frame_data = data[data_start:footer_pos]
        else:
            # 没有帧尾，提取到末尾
            frame_data = data[data_start:]
        
        # 验证最小长度
        if len(frame_data) < 3:
            return None
        
        return frame_data
    
    def _parse_custom_log_frame(self, data: bytes) -> Optional[bytes]:
        """解析自定义日志帧格式，返回 payload"""
        if not self.enable_custom_log_frame or not self.log_frame_header_bytes:
            return None
        
        # 查找帧头
        header_pos = data.find(self.log_frame_header_bytes)
        if header_pos == -1:
            return None
        
        # 提取数据起始位置
        data_start = header_pos + len(self.log_frame_header_bytes)
        
        # 如果有帧尾，查找帧尾
        if self.log_frame_footer_bytes:
            footer_pos = data.find(self.log_frame_footer_bytes, data_start)
            if footer_pos == -1:
                return None
            frame_data = data[data_start:footer_pos]
        else:
            # 没有帧尾，提取到末尾
            frame_data = data[data_start:]
        
        if len(frame_data) < 1:
            return None
        
        # 根据格式解析
        if self.log_frame_format == '标准格式':
            # 标准格式: [0x02][LEN][payload]
            if len(frame_data) < 2:  # 至少需要 1+1
                return None
            try:
                return parse_log_frame(frame_data)
            except:
                return None
        else:  # 纯文本
            # 纯文本格式，直接返回内容
            return frame_data
    
    def _parse_log_variables(self, payload: bytes) -> dict:
        """从日志payload中解析所有配置的变量
        
        Args:
            payload: 日志数据payload
            
        Returns:
            字典 {变量名: 值}
        """
        result = {}
        for var_name, byte_pos, data_type, _ in self.log_variables:
            try:
                value = self._parse_single_log_value(payload, byte_pos, data_type)
                result[var_name] = value if value is not None else ''
            except Exception:
                result[var_name] = ''
        return result
    
    def _parse_single_log_value(self, data: bytes, byte_pos: int, data_type: str):
        """从日志数据中解析指定位置的值
        
        Args:
            data: 日志数据
            byte_pos: 字节位置
            data_type: 数据类型
            
        Returns:
            解析后的值，如果失败返回 None
        """
        if byte_pos >= len(data):
            return None
        
        try:
            if data_type == 'uint8':
                return data[byte_pos]
            elif data_type == 'int8':
                return struct.unpack_from('b', data, byte_pos)[0]
            elif data_type == 'uint16_le':
                if byte_pos + 1 >= len(data):
                    return None
                return struct.unpack_from('<H', data, byte_pos)[0]
            elif data_type == 'uint16_be':
                if byte_pos + 1 >= len(data):
                    return None
                return struct.unpack_from('>H', data, byte_pos)[0]
            elif data_type == 'int16_le':
                if byte_pos + 1 >= len(data):
                    return None
                return struct.unpack_from('<h', data, byte_pos)[0]
            elif data_type == 'int16_be':
                if byte_pos + 1 >= len(data):
                    return None
                return struct.unpack_from('>h', data, byte_pos)[0]
            elif data_type == 'uint32_le':
                if byte_pos + 3 >= len(data):
                    return None
                return struct.unpack_from('<I', data, byte_pos)[0]
            elif data_type == 'uint32_be':
                if byte_pos + 3 >= len(data):
                    return None
                return struct.unpack_from('>I', data, byte_pos)[0]
            elif data_type == 'int32_le':
                if byte_pos + 3 >= len(data):
                    return None
                return struct.unpack_from('<i', data, byte_pos)[0]
            elif data_type == 'int32_be':
                if byte_pos + 3 >= len(data):
                    return None
                return struct.unpack_from('>i', data, byte_pos)[0]
            elif data_type == 'float_le':
                if byte_pos + 3 >= len(data):
                    return None
                return struct.unpack_from('<f', data, byte_pos)[0]
            elif data_type == 'float_be':
                if byte_pos + 3 >= len(data):
                    return None
                return struct.unpack_from('>f', data, byte_pos)[0]
            else:
                return None
        except Exception:
            return None
    
    def _parse_custom_log_frame_OLD(self, data: bytes) -> Optional[bytes]:
        """解析自定义日志帧格式，返回 payload"""
        if not self.enable_custom_log_frame or not self.log_frame_header_bytes:
            return None
        
        # 查找帧头
        header_pos = data.find(self.log_frame_header_bytes)
        if header_pos == -1:
            return None
        
        # 提取数据起始位置
        data_start = header_pos + len(self.log_frame_header_bytes)
        
        # 如果有帧尾，查找帧尾
        if self.log_frame_footer_bytes:
            footer_pos = data.find(self.log_frame_footer_bytes, data_start)
            if footer_pos == -1:
                return None
            frame_data = data[data_start:footer_pos]
        else:
            # 没有帧尾，提取到末尾
            frame_data = data[data_start:]
        
        if len(frame_data) < 1:
            return None
        
        # 根据格式解析
        if self.log_frame_format == '标准格式':
            # 标准格式: [0x02][LEN][payload]
            if len(frame_data) < 2:  # 至少需要 1+1
                return None
            try:
                return parse_log_frame(frame_data)
            except:
                return None
        else:  # 纯文本
            # 纯文本格式，直接返回内容
            return frame_data
    
    def _parse_custom_image_data(self, data: bytes) -> Optional[Tuple[int, int, np.ndarray]]:
        """根据自定义格式解析图像数据
        
        支持两种模式:
        1. 固定尺寸模式: 图像数据只包含像素，H/W通过配置指定
           格式: [像素数据]
        
        2. 动态尺寸模式: 图像数据包含H/W字段
           格式: [H-N字节][W-M字节][像素数据]
        
        注意: data参数已经去除了帧头和帧尾，只包含图像数据部分
        """
        if not HAS_CV2:
            return None
        
        try:
            # 模式1: 固定尺寸 (STM32模式)
            if self.image_fixed_h > 0 and self.image_fixed_w > 0:
                h = self.image_fixed_h
                w = self.image_fixed_w
                
                # 使用格式解码器解码像素数据
                img = decode_image_by_format(data, h, w, self.image_format)
                
                if img is None:
                    return None
                
                return h, w, img
            
            # 模式2: 动态尺寸 (包含H/W字段)
            else:
                idx = 0
                
                # 解析高度 (H字段)
                if len(data) < self.image_h_bytes + self.image_w_bytes:
                    return None
                
                h_bytes = data[idx:idx+self.image_h_bytes]
                if self.image_h_order == '大端':
                    h = int.from_bytes(h_bytes, 'big')
                else:
                    h = int.from_bytes(h_bytes, 'little')
                idx += self.image_h_bytes
                
                # 解析宽度 (W字段)
                w_bytes = data[idx:idx+self.image_w_bytes]
                if self.image_w_order == '大端':
                    w = int.from_bytes(w_bytes, 'big')
                else:
                    w = int.from_bytes(w_bytes, 'little')
                idx += self.image_w_bytes
                
                # 使用格式解码器解码像素数据
                pixel_data = data[idx:]
                img = decode_image_by_format(pixel_data, h, w, self.image_format)
                
                if img is None:
                    return None
                
                return h, w, img
            
        except Exception as e:
            print(f"[ERROR] Custom image parse error: {e}")
            return None
    
    def _receive_loop(self):
        """接收循环（在后台线程运行）"""
        print(f"[INFO] UDP receiver started on {self.ip}:{self.port}")
        if self.enable_custom_image_frame:
            print(f"[INFO] Custom image frame enabled: header={self.image_frame_header_bytes.hex()}, footer={self.image_frame_footer_bytes.hex() if self.image_frame_footer_bytes else 'None'}")
        if self.enable_custom_log_frame:
            print(f"[INFO] Custom log frame enabled: header={self.log_frame_header_bytes.hex()}, footer={self.log_frame_footer_bytes.hex() if self.log_frame_footer_bytes else 'None'}")
        
        while not self._stop_event.is_set():
            try:
                data, addr = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    print(f"[ERROR] Socket error: {e}")
                break
            
            self.total_packets += 1
            host_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            recv_time = time.time()
            
            if not data:
                continue
            
            # 提取示波器数据(从原始UDP数据包中)
            if self.scope_data:
                with self.scope_lock:
                    for key in self.scope_data:
                        byte_idx, bit_idx = key
                        if byte_idx < len(data):
                            byte_val = data[byte_idx]
                            if bit_idx is not None:
                                # 检查是否为切片
                                if isinstance(bit_idx, tuple) and bit_idx[0] == 'slice':
                                    # 位切片：提取多个位并组合成一个值
                                    _, start, end, step = bit_idx
                                    value = 0
                                    bit_positions = list(range(start, end, step))
                                    for i, pos in enumerate(bit_positions):
                                        bit_val = (byte_val >> pos) & 1
                                        value |= (bit_val << i)  # 组合成新的值
                                else:
                                    # 提取单个位
                                    value = (byte_val >> bit_idx) & 1
                            else:
                                # 整个字节值
                                value = byte_val
                            self.scope_data[key].append((recv_time, value))
            
            # 尝试解析自定义图像帧
            if self.enable_custom_image_frame:
                custom_image_data = self._parse_custom_frame(data)
                if custom_image_data is not None:
                    # 使用统一的自定义图像解析器(会根据格式自动选择解码方式)
                    result = self._parse_custom_image_data(custom_image_data)
                    if result is not None:
                        h, w, img = result
                        if img is not None:
                            self.frame_counter += 1
                            self._update_fps()
                            
                            with self.frame_lock:
                                self.current_frame = img.copy()
                            
                            # 根据图像格式生成描述信息
                            format_name = self.image_format
                            if format_name == '压缩二值(1位)':
                                data_size = len(custom_image_data)
                                compression_ratio = (h * w) / data_size if data_size > 0 else 0
                                info = f"{format_name} Frame {self.frame_counter}: {w}x{h}, {data_size} bytes ({compression_ratio:.1f}:1)"
                            else:
                                info = f"{format_name} Frame {self.frame_counter}: {w}x{h}, {len(data)} bytes total"
                            
                            with self.data_lock:
                                self.recent_data.append((
                                    host_iso,
                                    'CUSTOM_IMAGE',
                                    data[:100].hex() + ('...' if len(data) > 100 else ''),
                                    info
                                ))
                                if len(self.recent_data) > self.max_recent_data:
                                    self.recent_data.pop(0)
                            
                            if self.save_png:
                                png_path = os.path.join(self.png_dir, f"frame_{self.frame_counter:06d}.png")
                                # 对于彩色图像,需要转换为BGR保存
                                if len(img.shape) == 3:
                                    cv2.imwrite(png_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                                else:
                                    cv2.imwrite(png_path, img)
                                if self._frame_index_writer:
                                    self._frame_index_writer.writerow([self.frame_counter, host_iso, png_path, h, w])
                                    self._frame_index_fp.flush()
                            
                            continue
            
            # 尝试解析自定义日志帧
            if self.enable_custom_log_frame:
                payload = self._parse_custom_log_frame(data)
                if payload is not None:
                    text_utf8 = ''
                    try:
                        text_utf8 = payload.decode('utf-8', errors='replace')
                        # 清理特殊字符,避免CSV读取问题
                        text_utf8 = sanitize_csv_text(text_utf8)
                    except:
                        pass
                    text_hex = payload.hex()
                    
                    with self.data_lock:
                        display_hex = data.hex()
                        if len(display_hex) > 500:
                            display_hex = display_hex[:500] + '...'
                        
                        self.recent_data.append((
                            host_iso,
                            'CUSTOM_LOG',
                            display_hex,
                            f"Custom LOG: {text_utf8[:50]}" + ('...' if len(text_utf8) > 50 else '')
                        ))
                        if len(self.recent_data) > self.max_recent_data:
                            self.recent_data.pop(0)
                    
                    # 写入CSV（包含解析的变量值）
                    if self._log_writer:
                        # 构建CSV行数据
                        row_data = [host_iso, text_hex, text_utf8]
                        # 解析所有配置的变量并添加到行数据
                        log_vars = self._parse_log_variables(payload)
                        for var_name, _, _, _ in self.log_variables:
                            row_data.append(log_vars.get(var_name, ''))
                        self._log_writer.writerow(row_data)
                        self._log_csv_fp.flush()
                    
                    # 调用日志回调（用于实时显示）
                    if self.log_callback:
                        try:
                            self.log_callback(payload)
                        except Exception as e:
                            print(f"[ERROR] Log callback error: {e}")
                    
                    continue
            
            # 如果启用了自定义帧但无法解析，记录错误
            if self.enable_custom_image_frame or self.enable_custom_log_frame:
                self.error_packets += 1
                with self.data_lock:
                    self.recent_data.append((
                        host_iso,
                        'INVALID_CUSTOM',
                        data[:200].hex() + ('...' if len(data) > 200 else ''),
                        f"Failed to parse custom frame: {len(data)} bytes"
                    ))
                    if len(self.recent_data) > self.max_recent_data:
                        self.recent_data.pop(0)
                continue
            
            # 默认帧格式处理
            ftype = data[0]
            
            try:
                if ftype == FrameType.IMAGE:
                    h, w, img = parse_image_frame(data)
                    if img is not None:
                        self.frame_counter += 1
                        self._update_fps()
                        
                        # 更新当前帧
                        with self.frame_lock:
                            self.current_frame = img.copy()
                        
                        # 记录原始数据
                        with self.data_lock:
                            # 对于图像帧，只保存前 100 字节用于显示
                            self.recent_data.append((
                                host_iso,
                                'IMAGE',
                                data[:100].hex() + ('...' if len(data) > 100 else ''),
                                f"Frame {self.frame_counter}: {w}x{h}, {len(data)} bytes"
                            ))
                            if len(self.recent_data) > self.max_recent_data:
                                self.recent_data.pop(0)
                        
                        # 保存 PNG
                        png_path = ''
                        if self.save_png:
                            png_path = os.path.join(self.png_dir, f"frame_{self.frame_counter:06d}.png")
                            cv2.imwrite(png_path, img)
                        
                        # 记录到 CSV
                        if self._frame_index_writer:
                            self._frame_index_writer.writerow([self.frame_counter, host_iso, png_path, h, w])
                            self._frame_index_fp.flush()
                
                elif ftype == FrameType.BINARY_IMAGE:
                    h, w, img = parse_binary_image_frame(data)
                    if img is not None:
                        self.frame_counter += 1
                        self._update_fps()
                        
                        with self.frame_lock:
                            self.current_frame = img.copy()
                        
                        # 记录原始数据
                        with self.data_lock:
                            # 对于二值图像帧，只保存前 100 字节用于显示
                            self.recent_data.append((
                                host_iso,
                                'BINARY_IMAGE',
                                data[:100].hex() + ('...' if len(data) > 100 else ''),
                                f"Binary Frame {self.frame_counter}: {w}x{h}, {len(data)} bytes"
                            ))
                            if len(self.recent_data) > self.max_recent_data:
                                self.recent_data.pop(0)
                        
                        png_path = ''
                        if self.save_png:
                            png_path = os.path.join(self.png_dir, f"frame_{self.frame_counter:06d}.png")
                            cv2.imwrite(png_path, img)
                        
                        if self._frame_index_writer:
                            self._frame_index_writer.writerow([self.frame_counter, host_iso, png_path, h, w])
                            self._frame_index_fp.flush()
                
                elif ftype == FrameType.LOG:
                    payload = parse_log_frame(data)
                    text_utf8 = ''
                    try:
                        text_utf8 = payload.decode('utf-8', errors='replace')
                        # 清理特殊字符,避免CSV读取问题
                        text_utf8 = sanitize_csv_text(text_utf8)
                    except:
                        pass
                    text_hex = payload.hex()
                    
                    # 记录原始数据（LOG 帧保存完整数据，因为通常不大）
                    with self.data_lock:
                        # 限制显示长度但保存完整 hex
                        display_hex = data.hex()
                        if len(display_hex) > 500:  # 如果太长，截断显示
                            display_hex = display_hex[:500] + '...'
                        
                        self.recent_data.append((
                            host_iso,
                            'LOG',
                            display_hex,
                            f"LOG: {text_utf8[:50]}" + ('...' if len(text_utf8) > 50 else '')
                        ))
                        if len(self.recent_data) > self.max_recent_data:
                            self.recent_data.pop(0)
                    
                    # 写入CSV（包含解析的变量值）
                    if self._log_writer:
                        # 构建CSV行数据
                        row_data = [host_iso, text_hex, text_utf8]
                        # 解析所有配置的变量并添加到行数据
                        log_vars = self._parse_log_variables(payload)
                        for var_name, _, _, _ in self.log_variables:
                            row_data.append(log_vars.get(var_name, ''))
                        self._log_writer.writerow(row_data)
                        self._log_csv_fp.flush()
                    
                    # 调用日志回调（用于实时显示）
                    if self.log_callback:
                        try:
                            self.log_callback(payload)
                        except Exception as e:
                            print(f"[ERROR] Log callback error: {e}")
                else:
                    # 未知类型
                    self.error_packets += 1
                    with self.data_lock:
                        self.recent_data.append((
                            host_iso,
                            f'UNKNOWN(0x{ftype:02X})',
                            data[:100].hex() + ('...' if len(data) > 100 else ''),
                            f"Unknown frame type, {len(data)} bytes"
                        ))
                        if len(self.recent_data) > self.max_recent_data:
                            self.recent_data.pop(0)
            
            except Exception as e:
                self.error_packets += 1
                print(f"[ERROR] Parse error: {e}")
                with self.data_lock:
                    self.recent_data.append((
                        host_iso,
                        'ERROR',
                        data[:100].hex() + ('...' if len(data) > 100 else ''),
                        f"Parse error: {str(e)}"
                    ))
                    if len(self.recent_data) > self.max_recent_data:
                        self.recent_data.pop(0)
        
        print("[INFO] UDP receiver stopped")



class App(tb.Window):
    def __init__(self):
        super().__init__(themename='flatly')  # 可选主题: superhero, cyborg, darkly, litera, flatly, cosmo...
        
        self.title('UDP 上位机 GUI')
        self.geometry('1200x800')
        
        # UDP 视频接收器
        self.video_receiver: Optional[UdpVideoReceiver] = None
        self._video_update_job = None

        # --- 参数 ---
        self.ip = tk.StringVar(value='0.0.0.0')
        self.port = tk.IntVar(value=8080)
        self.show = tk.BooleanVar(value=True)
        self.save_png = tk.BooleanVar(value=False)
        self.png_dir = tk.StringVar(value=os.path.join(os.getcwd(), 'frames_png'))
        self.log_csv = tk.StringVar(value=os.path.join(os.getcwd(), 'logs.csv'))
        self.frame_index_csv = tk.StringVar(value=os.path.join(os.getcwd(), 'frames_index.csv'))

        self.video_png_dir = tk.StringVar(value=os.path.join(os.getcwd(), 'frames_png'))
        self.video_out = tk.StringVar(value=os.path.join(os.getcwd(), 'output.mp4'))
        self.video_fps = tk.IntVar(value=30)

        self.scope_index = tk.IntVar(value=0)
        self.scope_bit = tk.StringVar(value='')  # 允许空，或 0..7
        self.scope_max_points = tk.IntVar(value=2000)
        
        # 自定义帧格式 - 图像帧
        self.enable_custom_image_frame = tk.BooleanVar(value=False)
        self.image_frame_header = tk.StringVar(value='A0FFFFA0')  # 默认使用STM32帧头
        self.image_frame_footer = tk.StringVar(value='B0B00A0D')  # 默认使用STM32帧尾
        self.image_h_bytes = tk.IntVar(value=1)  # H 字段字节数 (动态模式)
        self.image_w_bytes = tk.IntVar(value=1)  # W 字段字节数 (动态模式)
        self.image_h_order = tk.StringVar(value='小端')  # 字节序
        self.image_w_order = tk.StringVar(value='小端')
        self.image_fixed_h = tk.IntVar(value=120)  # 固定高度 (STM32模式)
        self.image_fixed_w = tk.IntVar(value=188)  # 固定宽度 (STM32模式)
        self.image_size_mode = tk.StringVar(value='固定尺寸')  # 固定尺寸 / 动态解析
        self.image_format = tk.StringVar(value='压缩二值(1位)')  # 图像数据编码格式
        
        # 自定义帧格式 - 日志帧
        self.enable_custom_log_frame = tk.BooleanVar(value=False)
        self.log_frame_header = tk.StringVar(value='BB66')
        self.log_frame_footer = tk.StringVar(value='0D0A')
        self.log_frame_format = tk.StringVar(value='标准格式')  # 标准格式 / 纯文本

    # 已移除 C 扩展处理选项（ctypes），保持 GUI 简洁

        self._build_ui()
        
        # 绑定关闭事件
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    # ---------- 小部件工厂（处理 bootstyle 兼容） ----------
    def _btn(self, parent, text, command, bootstyle: str | None = None, width=None):
        """创建带 bootstyle 的按钮"""
        if width is not None:
            return ttk.Button(parent, text=text, command=command, bootstyle=bootstyle, width=width)
        return ttk.Button(parent, text=text, command=command, bootstyle=bootstyle)

    def _nb(self, parent):
        """创建 Notebook"""
        return ttk.Notebook(parent, bootstyle='primary')

    def _scrolled_text(self, parent):
        """创建滚动文本框"""
        return TBScrolledText(parent, autohide=True, height=8, bootstyle='secondary')

    def _add_switch(self, parent, text: str, var, bootstyle: str | None, grid_kwargs: dict):
        """创建开关控件"""
        if Switch is not None:
            w = Switch(parent, text=text, variable=var, bootstyle=bootstyle)
        else:
            w = ttk.Checkbutton(parent, text=text, variable=var)
        w.grid(**grid_kwargs)

    # ---------------- UI 构建 ----------------
    def _build_ui(self):
        # 创建主容器：左侧控制面板 + 右侧视频显示
        main_container = ttk.PanedWindow(self, orient='horizontal')
        main_container.pack(fill='both', expand=True, padx=4, pady=4)
        
        # 左侧控制面板
        left_panel = ttk.Frame(main_container)
        main_container.add(left_panel, weight=1)
        
        # 右侧视频显示区域
        right_panel = ttk.Frame(main_container)
        main_container.add(right_panel, weight=2)
        
        # === 左侧面板内容 ===
        # 顶部工具栏（主题切换）
        toolbar = ttk.Frame(left_panel)
        toolbar.pack(fill='x', padx=8, pady=(8, 4))
        ttk.Label(toolbar, text='主题:').pack(side='left')
        current_theme = 'flatly'
        self.theme_var = tk.StringVar(value=current_theme)
        theme_sel = ttk.Combobox(
            toolbar,
            textvariable=self.theme_var,
            width=16,
            state='readonly',
            values=sorted(self.style.theme_names()),
        )
        theme_sel.pack(side='left', padx=(6, 12))
        self._btn(toolbar, text='应用主题', command=self.apply_theme, bootstyle='secondary-outline').pack(side='left')

        # 标签页
        nb = self._nb(left_panel)
        nb.pack(fill='both', expand=True, padx=8, pady=4)

        nb.add(self._build_tab_run(), text='运行')
        nb.add(self._build_tab_video(), text='视频')
        nb.add(self._build_tab_align(), text='对齐')
        nb.add(self._build_tab_scope(), text='示波')
        nb.add(self._build_tab_custom_frame(), text='自定义帧')

        # 输出区域（带滚动）
        self.out_text = self._scrolled_text(left_panel)
        self.out_text.pack(fill='both', expand=True, padx=8, pady=(4, 8))
        self._log('提示：点击"启动监听"查看实时视频流。')
        
        # === 右侧视频显示区域 ===
        video_frame = ttk.LabelFrame(right_panel, text='实时视频', padding=10)
        video_frame.pack(fill='both', expand=True, padx=8, pady=(8, 4))
        
        # 视频画布
        self.video_canvas = tk.Canvas(video_frame, bg='black', highlightthickness=0)
        self.video_canvas.pack(fill='both', expand=True)
        
        # 统计信息标签
        stats_frame = ttk.Frame(video_frame)
        stats_frame.pack(fill='x', pady=(8, 0))
        
        self.stats_label = ttk.Label(stats_frame, text='等待视频流...', font=('Consolas', 9))
        self.stats_label.pack()
        
        # 实时日志显示区域
        log_display_frame = ttk.LabelFrame(right_panel, text='实时日志显示', padding=10)
        log_display_frame.pack(fill='both', expand=False, padx=8, pady=(4, 4))
        
        # 日志显示画布（用于动态创建标签）
        self.log_canvas = tk.Canvas(log_display_frame, height=100, bg='#1e1e1e', highlightthickness=0)
        self.log_canvas.pack(fill='both', expand=True)
        
        # 日志数据容器
        self.log_labels = {}  # {var_name: label_widget}
        self.log_values = {}  # {var_name: current_value}
        
        # 原始数据显示区域
        data_frame = ttk.LabelFrame(right_panel, text='原始数据监视器', padding=10)
        data_frame.pack(fill='both', expand=False, padx=8, pady=(4, 8))
        
        # 工具栏
        data_toolbar = ttk.Frame(data_frame)
        data_toolbar.pack(fill='x', pady=(0, 4))
        
        self._btn(data_toolbar, text='清空', command=self._clear_data_display, bootstyle='secondary-outline').pack(side='left', padx=2)
        self._btn(data_toolbar, text='刷新', command=self._refresh_data_display, bootstyle='info-outline').pack(side='left', padx=2)
        
        ttk.Label(data_toolbar, text='最大显示:').pack(side='left', padx=(10, 2))
        self.data_display_limit = tk.IntVar(value=20)
        ttk.Spinbox(data_toolbar, from_=10, to=100, width=8, textvariable=self.data_display_limit).pack(side='left')
        
        ttk.Label(data_toolbar, text='编码:').pack(side='left', padx=(10, 2))
        self.data_encoding = tk.StringVar(value='UTF-8')
        encoding_combo = ttk.Combobox(
            data_toolbar,
            textvariable=self.data_encoding,
            width=12,
            state='readonly',
            values=['UTF-8', 'GBK', 'GB2312', 'ASCII', 'Latin-1', 'UTF-16', 'UTF-32', 'Big5']
        )
        encoding_combo.pack(side='left', padx=2)
        encoding_combo.bind('<<ComboboxSelected>>', lambda e: self._refresh_data_display())
        
        ttk.Label(data_toolbar, text='显示格式:').pack(side='left', padx=(10, 2))
        self.data_format = tk.StringVar(value='详细')
        format_combo = ttk.Combobox(
            data_toolbar,
            textvariable=self.data_format,
            width=10,
            state='readonly',
            values=['详细', '简洁', '仅Hex', '仅文本']
        )
        format_combo.pack(side='left', padx=2)
        format_combo.bind('<<ComboboxSelected>>', lambda e: self._refresh_data_display())
        
        # 数据显示文本框
        self.data_text = self._scrolled_text(data_frame)
        container = getattr(self.data_text, '_container', None)
        if container is not None:
            container.pack(fill='both', expand=True)
        else:
            self.data_text.pack(fill='both', expand=True)
        
        # 配置文本框样式（等宽字体）
        try:
            self.data_text.configure(font=('Consolas', 9), height=8)
        except:
            pass
        
        # 显示提示信息
        self._show_video_placeholder()

    def _build_tab_run(self):
        f = ttk.Frame()

        row = 0
        ttk.Label(f, text='绑定 IP:').grid(row=row, column=0, sticky='e', padx=6, pady=6)
        
        # 使用 Combobox 显示可用的 IP 地址
        ip_combo = ttk.Combobox(f, textvariable=self.ip, width=15, values=get_local_ips())
        ip_combo.grid(row=row, column=1, sticky='w')
        
        self._btn(f, text='刷新IP', command=self._refresh_ips, bootstyle='secondary-outline').grid(row=row, column=2, sticky='w', padx=2)

        ttk.Label(f, text='端口:').grid(row=row, column=3, sticky='e')
        ttk.Entry(f, textvariable=self.port, width=8).grid(row=row, column=4, sticky='w')

        row += 1
        # 移除 show 开关（因为现在始终在主窗口显示）
        self._add_switch(
            f,
            text='保存 PNG (--save-png)',
            var=self.save_png,
            bootstyle='info',
            grid_kwargs=dict(row=row, column=0, columnspan=2, sticky='w', padx=6),
        )

        row += 1
        ttk.Label(f, text='PNG 目录:').grid(row=row, column=0, sticky='e', padx=6)
        ttk.Entry(f, textvariable=self.png_dir, width=48).grid(row=row, column=1, columnspan=3, sticky='we')
        self._btn(f, text='选择', command=self._pick_png_dir, bootstyle='secondary-outline').grid(row=row, column=4, sticky='w')

        row += 1
        ttk.Label(f, text='日志 CSV:').grid(row=row, column=0, sticky='e', padx=6)
        ttk.Entry(f, textvariable=self.log_csv, width=48).grid(row=row, column=1, columnspan=3, sticky='we')
        self._btn(f, text='选择', command=self._pick_log_csv, bootstyle='secondary-outline').grid(row=row, column=4, sticky='w')

        row += 1
        ttk.Label(f, text='帧索引 CSV:').grid(row=row, column=0, sticky='e', padx=6)
        ttk.Entry(f, textvariable=self.frame_index_csv, width=48).grid(row=row, column=1, columnspan=3, sticky='we')
        self._btn(f, text='选择', command=self._pick_frame_csv, bootstyle='secondary-outline').grid(row=row, column=4, sticky='w')

        row += 1
        self._btn(f, text='启动监听', command=self.start_run, bootstyle='success').grid(row=row, column=1, columnspan=2, sticky='we', padx=6, pady=10)
        self._btn(f, text='停止监听', command=self.stop_run, bootstyle='danger').grid(row=row, column=3, columnspan=2, sticky='we', padx=6, pady=10)

        for c in range(5):
            f.grid_columnconfigure(c, weight=1)
        return f

    def _build_tab_custom_frame(self):
        """构建自定义帧格式标签页"""
        f = ttk.Frame()
        
        # 创建Notebook用于分页
        custom_nb = self._nb(f)
        custom_nb.pack(fill='both', expand=True, padx=6, pady=6)
        
        # 图像帧配置页
        image_tab = self._build_custom_image_frame_tab()
        custom_nb.add(image_tab, text='图像帧配置')
        
        # 日志帧配置页
        log_tab = self._build_custom_log_frame_tab()
        custom_nb.add(log_tab, text='日志帧配置')
        
        # 日志变量配置页
        log_vars_tab = self._build_log_variables_tab()
        custom_nb.add(log_vars_tab, text='日志变量配置')
        
        return f
    
    def _build_custom_image_frame_tab(self):
        """构建图像帧自定义配置"""
        f = ttk.Frame()
        
        row = 0
        self._add_switch(
            f,
            text='启用图像帧自定义格式',
            var=self.enable_custom_image_frame,
            bootstyle='success',
            grid_kwargs=dict(row=row, column=0, columnspan=4, sticky='w', padx=6, pady=10),
        )
        
        row += 1
        ttk.Separator(f, orient='horizontal').grid(row=row, column=0, columnspan=4, sticky='ew', pady=10)
        
        row += 1
        ttk.Label(f, text='帧头 (Hex):', font=('Arial', 10, 'bold')).grid(row=row, column=0, sticky='e', padx=6, pady=6)
        ttk.Entry(f, textvariable=self.image_frame_header, width=20, font=('Consolas', 10)).grid(row=row, column=1, sticky='w', padx=6)
        ttk.Label(f, text='例: AA55', foreground='gray').grid(row=row, column=2, sticky='w', padx=6)
        
        row += 1
        ttk.Label(f, text='帧尾 (Hex):', font=('Arial', 10, 'bold')).grid(row=row, column=0, sticky='e', padx=6, pady=6)
        ttk.Entry(f, textvariable=self.image_frame_footer, width=20, font=('Consolas', 10)).grid(row=row, column=1, sticky='w', padx=6)
        ttk.Label(f, text='可选', foreground='gray').grid(row=row, column=2, sticky='w', padx=6)
        
        row += 1
        ttk.Separator(f, orient='horizontal').grid(row=row, column=0, columnspan=4, sticky='ew', pady=10)
        
        row += 1
        ttk.Label(f, text='图像格式:', font=('Arial', 10, 'bold')).grid(row=row, column=0, sticky='e', padx=6, pady=6)
        format_combo = ttk.Combobox(f, textvariable=self.image_format, width=18, state='readonly',
                                     values=['灰度图(8位)', '二值图(8位)', '压缩二值(1位)', 
                                             'RGB565', 'RGB888', 'BGR888', 'RGBA8888'])
        format_combo.grid(row=row, column=1, sticky='w', padx=6)
        
        # 格式说明
        format_info = ttk.Label(f, text='← 选择像素数据编码格式', foreground='gray', font=('Arial', 9))
        format_info.grid(row=row, column=2, columnspan=2, sticky='w', padx=6)
        
        row += 1
        ttk.Separator(f, orient='horizontal').grid(row=row, column=0, columnspan=4, sticky='ew', pady=10)
        
        row += 1
        ttk.Label(f, text='尺寸解析模式:', font=('Arial', 10, 'bold')).grid(row=row, column=0, sticky='e', padx=6, pady=6)
        mode_combo = ttk.Combobox(f, textvariable=self.image_size_mode, width=18, state='readonly',
                                   values=['固定尺寸', '动态解析'])
        mode_combo.grid(row=row, column=1, sticky='w', padx=6)
        mode_combo.bind('<<ComboboxSelected>>', self._on_image_size_mode_changed)
        
        # 固定尺寸配置 (STM32模式) - row_fixed
        self.row_fixed = row + 1
        self.fixed_size_label = ttk.Label(f, text='图像尺寸:', font=('Arial', 10))
        
        self.fixed_frame = ttk.Frame(f)
        ttk.Label(self.fixed_frame, text='H:').pack(side='left')
        ttk.Spinbox(self.fixed_frame, textvariable=self.image_fixed_h, from_=1, to=1024, width=6).pack(side='left', padx=2)
        ttk.Label(self.fixed_frame, text='W:').pack(side='left', padx=(10,0))
        ttk.Spinbox(self.fixed_frame, textvariable=self.image_fixed_w, from_=1, to=1024, width=6).pack(side='left', padx=2)
        
        # 动态解析配置 - row_dynamic_h, row_dynamic_w
        self.row_dynamic_h = row + 1
        self.dynamic_h_label = ttk.Label(f, text='H 字段:', font=('Arial', 10))
        self.dynamic_h_spinbox = ttk.Spinbox(f, textvariable=self.image_h_bytes, from_=1, to=4, width=8)
        self.dynamic_h_byte_label = ttk.Label(f, text='字节')
        self.dynamic_h_combo = ttk.Combobox(f, textvariable=self.image_h_order, width=8, state='readonly', 
                                             values=['小端', '大端'])
        
        self.row_dynamic_w = row + 2
        self.dynamic_w_label = ttk.Label(f, text='W 字段:', font=('Arial', 10))
        self.dynamic_w_spinbox = ttk.Spinbox(f, textvariable=self.image_w_bytes, from_=1, to=4, width=8)
        self.dynamic_w_byte_label = ttk.Label(f, text='字节')
        self.dynamic_w_combo = ttk.Combobox(f, textvariable=self.image_w_order, width=8, state='readonly',
                                             values=['小端', '大端'])
        
        # 初始显示固定尺寸模式
        self._update_image_size_mode_ui()
        
        row += 3  # 预留3行空间
        ttk.Separator(f, orient='horizontal').grid(row=row, column=0, columnspan=4, sticky='ew', pady=10)
        
        row += 1
        help_text = tk.Text(f, height=8, width=70, wrap='word', font=('Arial', 9))
        help_text.grid(row=row, column=0, columnspan=4, sticky='we', padx=6, pady=6)
        
        help_content = """图像帧格式说明：

UDP协议格式: [帧头] [图像数据] [帧尾]

【图像格式选项】
系统支持多种像素数据编码格式:

1. 压缩二值(1位) - 8:1压缩，适用于二值图像
   • 8个像素压缩成1字节，MSB优先
   • 数据量: H×W/8 字节
   • 示例(60x120): [A0FFFFA0][900字节][B0B00A0D]

2. 二值图(8位) - 黑白图像，每像素1字节
   • 0=黑色, 255=白色(自动阈值化)
   • 数据量: H×W 字节

3. 灰度图(8位) - 灰度图像，每像素1字节
   • 0-255灰度值
   • 数据量: H×W 字节

4. RGB565 - 彩色图像，每像素2字节
   • 5位R + 6位G + 5位B
   • 数据量: H×W×2 字节
   • 小端序存储

5. RGB888 - 真彩色，每像素3字节
   • 数据量: H×W×3 字节

6. BGR888 - OpenCV格式，每像素3字节
   • 数据量: H×W×3 字节

7. RGBA8888 - 带透明通道，每像素4字节
   • 数据量: H×W×4 字节

【尺寸模式】
• 固定尺寸: 图像数据=像素数据，H/W在配置中指定
• 动态解析: 图像数据=[H字段][W字段][像素数据]

STM32压缩示例(60x120):
  帧头=A0FFFFA0, 帧尾=B0B00A0D
  格式=压缩二值(1位), 固定尺寸=60x120"""
        
        help_text.insert('1.0', help_content)
        help_text.config(state='disabled')
        
        row += 1
        self._btn(f, text='验证图像帧配置', command=self._validate_image_frame, bootstyle='info').grid(row=row, column=1, sticky='we', padx=6, pady=10)
        
        for c in range(4):
            f.grid_columnconfigure(c, weight=1)
        
        return f
    
    def _build_custom_log_frame_tab(self):
        """构建日志帧自定义配置"""
        f = ttk.Frame()
        
        row = 0
        self._add_switch(
            f,
            text='启用日志帧自定义格式',
            var=self.enable_custom_log_frame,
            bootstyle='success',
            grid_kwargs=dict(row=row, column=0, columnspan=4, sticky='w', padx=6, pady=10),
        )
        
        row += 1
        ttk.Separator(f, orient='horizontal').grid(row=row, column=0, columnspan=4, sticky='ew', pady=10)
        
        row += 1
        ttk.Label(f, text='帧头 (Hex):', font=('Arial', 10, 'bold')).grid(row=row, column=0, sticky='e', padx=6, pady=6)
        ttk.Entry(f, textvariable=self.log_frame_header, width=20, font=('Consolas', 10)).grid(row=row, column=1, sticky='w', padx=6)
        ttk.Label(f, text='例: BB66', foreground='gray').grid(row=row, column=2, sticky='w', padx=6)
        
        row += 1
        ttk.Label(f, text='帧尾 (Hex):', font=('Arial', 10, 'bold')).grid(row=row, column=0, sticky='e', padx=6, pady=6)
        ttk.Entry(f, textvariable=self.log_frame_footer, width=20, font=('Consolas', 10)).grid(row=row, column=1, sticky='w', padx=6)
        ttk.Label(f, text='可选', foreground='gray').grid(row=row, column=2, sticky='w', padx=6)
        
        row += 1
        ttk.Separator(f, orient='horizontal').grid(row=row, column=0, columnspan=4, sticky='ew', pady=10)
        
        row += 1
        ttk.Label(f, text='数据格式:', font=('Arial', 10, 'bold')).grid(row=row, column=0, sticky='e', padx=6, pady=6)
        format_combo = ttk.Combobox(f, textvariable=self.log_frame_format, width=18, state='readonly',
                                     values=['标准格式', '纯文本'])
        format_combo.grid(row=row, column=1, sticky='w', padx=6)
        
        row += 1
        ttk.Separator(f, orient='horizontal').grid(row=row, column=0, columnspan=4, sticky='ew', pady=10)
        
        row += 1
        help_text = tk.Text(f, height=10, width=70, wrap='word', font=('Arial', 9))
        help_text.grid(row=row, column=0, columnspan=4, sticky='we', padx=6, pady=6)
        
        help_content = """日志帧格式说明：

UDP协议格式: [帧头] [日志数据] [帧尾]

日志数据部分格式:
  标准格式: [0x02] [LEN-1字节] [日志内容]
    - LEN: 日志内容长度(不含类型/长度字段)
    - 示例: BB66 02 0E [14字节日志] 0D0A

  纯文本格式: [文本内容]
    - 直接发送文本，无需类型、长度字段
    - 示例: BB66 48656C6C6F ("Hello") 0D0A

时间戳说明:
  • 系统自动记录上位机接收时间(精确到微秒)
  • 所有数据按上位机时间戳对齐
  • 无需下位机提供时间戳

建议: 调试时使用纯文本格式更简单"""
        
        help_text.insert('1.0', help_content)
        help_text.config(state='disabled')
        
        row += 1
        self._btn(f, text='验证日志帧配置', command=self._validate_log_frame, bootstyle='info').grid(row=row, column=1, sticky='we', padx=6, pady=10)
        
        for c in range(4):
            f.grid_columnconfigure(c, weight=1)
        
        return f
    
    def _build_log_variables_tab(self):
        """构建日志变量配置标签页"""
        f = ttk.Frame()
        
        # 说明
        info_frame = ttk.Frame(f)
        info_frame.pack(fill='x', padx=6, pady=6)
        
        info_label = ttk.Label(info_frame, text='配置日志变量以实时显示。支持从日志数据包中提取指定字节位置的值。', 
                                font=('Arial', 9), foreground='gray')
        info_label.pack(anchor='w')
        
        ttk.Separator(f, orient='horizontal').pack(fill='x', pady=10)
        
        # 添加变量区域
        add_frame = ttk.LabelFrame(f, text='添加日志变量', padding=10)
        add_frame.pack(fill='x', padx=6, pady=6)
        
        row = 0
        ttk.Label(add_frame, text='变量名称:', font=('Arial', 10)).grid(row=row, column=0, sticky='e', padx=6, pady=6)
        self.log_var_name_entry = ttk.Entry(add_frame, width=20)
        self.log_var_name_entry.grid(row=row, column=1, sticky='w', padx=6)
        ttk.Label(add_frame, text='例: 温度、速度', foreground='gray').grid(row=row, column=2, sticky='w', padx=6)
        
        row += 1
        ttk.Label(add_frame, text='字节位置:', font=('Arial', 10)).grid(row=row, column=0, sticky='e', padx=6, pady=6)
        self.log_var_byte_pos = tk.IntVar(value=0)
        ttk.Spinbox(add_frame, textvariable=self.log_var_byte_pos, from_=0, to=255, width=8).grid(row=row, column=1, sticky='w', padx=6)
        ttk.Label(add_frame, text='从0开始的字节索引', foreground='gray').grid(row=row, column=2, sticky='w', padx=6)
        
        row += 1
        ttk.Label(add_frame, text='数据类型:', font=('Arial', 10)).grid(row=row, column=0, sticky='e', padx=6, pady=6)
        self.log_var_data_type = tk.StringVar(value='uint8')
        type_combo = ttk.Combobox(add_frame, textvariable=self.log_var_data_type, width=18, state='readonly',
                                   values=['uint8', 'int8', 'uint16_le', 'uint16_be', 'int16_le', 'int16_be',
                                           'uint32_le', 'uint32_be', 'int32_le', 'int32_be', 'float_le', 'float_be'])
        type_combo.grid(row=row, column=1, sticky='w', padx=6)
        
        row += 1
        ttk.Label(add_frame, text='显示格式:', font=('Arial', 10)).grid(row=row, column=0, sticky='e', padx=6, pady=6)
        self.log_var_display_format = tk.StringVar(value='{value}')
        format_combo = ttk.Combobox(add_frame, textvariable=self.log_var_display_format, width=18,
                                     values=['{value}', '{value:.2f}', '0x{value:02X}', '0x{value:04X}', '{value}°C', '{value}%'])
        format_combo.grid(row=row, column=1, sticky='w', padx=6)
        ttk.Label(add_frame, text='使用Python格式化语法', foreground='gray').grid(row=row, column=2, sticky='w', padx=6)
        
        row += 1
        btn_frame = ttk.Frame(add_frame)
        btn_frame.grid(row=row, column=1, columnspan=2, sticky='w', padx=6, pady=10)
        self._btn(btn_frame, text='添加变量', command=self._add_log_variable, bootstyle='success').pack(side='left', padx=2)
        self._btn(btn_frame, text='清空所有', command=self._clear_log_variables, bootstyle='warning').pack(side='left', padx=2)
        
        # 变量列表区域
        list_frame = ttk.LabelFrame(f, text='已配置的日志变量', padding=10)
        list_frame.pack(fill='both', expand=True, padx=6, pady=6)
        
        # 创建Treeview显示变量列表
        columns = ('name', 'byte_pos', 'data_type', 'format')
        self.log_vars_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=8)
        
        self.log_vars_tree.heading('name', text='变量名称')
        self.log_vars_tree.heading('byte_pos', text='字节位置')
        self.log_vars_tree.heading('data_type', text='数据类型')
        self.log_vars_tree.heading('format', text='显示格式')
        
        self.log_vars_tree.column('name', width=150)
        self.log_vars_tree.column('byte_pos', width=80)
        self.log_vars_tree.column('data_type', width=100)
        self.log_vars_tree.column('format', width=120)
        
        scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=self.log_vars_tree.yview)
        self.log_vars_tree.configure(yscrollcommand=scrollbar.set)
        
        self.log_vars_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # 删除按钮
        btn_frame2 = ttk.Frame(list_frame)
        btn_frame2.pack(fill='x', pady=(6, 0))
        self._btn(btn_frame2, text='删除选中', command=self._remove_log_variable, bootstyle='danger').pack(side='left', padx=2)
        
        # 初始化日志变量列表
        self.log_variables = []  # [(name, byte_pos, data_type, display_format), ...]
        
        return f

    def _build_tab_video(self):
        f = ttk.Frame()
        row = 0
        ttk.Label(f, text='PNG 目录:').grid(row=row, column=0, sticky='e', padx=6, pady=6)
        ttk.Entry(f, textvariable=self.video_png_dir, width=48).grid(row=row, column=1, columnspan=2, sticky='we')
        self._btn(f, text='选择', command=self._pick_video_png_dir, bootstyle='secondary-outline').grid(row=row, column=3, sticky='w')

        row += 1
        ttk.Label(f, text='输出 MP4:').grid(row=row, column=0, sticky='e', padx=6, pady=6)
        ttk.Entry(f, textvariable=self.video_out, width=48).grid(row=row, column=1, columnspan=2, sticky='we')
        self._btn(f, text='选择', command=self._pick_video_out, bootstyle='secondary-outline').grid(row=row, column=3, sticky='w')

        row += 1
        ttk.Label(f, text='FPS:').grid(row=row, column=0, sticky='e', padx=6, pady=6)
        ttk.Entry(f, textvariable=self.video_fps, width=8).grid(row=row, column=1, sticky='w')
        self._btn(f, text='开始合成', command=self.compose_video, bootstyle='primary').grid(row=row, column=2, sticky='we', padx=6)
        return f

    def _build_tab_align(self):
        f = ttk.Frame()
        row = 0
        ttk.Label(f, text='frames_index.csv:').grid(row=row, column=0, sticky='e', padx=6, pady=6)
        ttk.Entry(f, textvariable=self.frame_index_csv, width=48).grid(row=row, column=1, columnspan=2, sticky='we')
        self._btn(f, text='选择', command=self._pick_frame_csv, bootstyle='secondary-outline').grid(row=row, column=3, sticky='w')

        row += 1
        ttk.Label(f, text='logs.csv:').grid(row=row, column=0, sticky='e', padx=6, pady=6)
        ttk.Entry(f, textvariable=self.log_csv, width=48).grid(row=row, column=1, columnspan=2, sticky='we')
        self._btn(f, text='选择', command=self._pick_log_csv, bootstyle='secondary-outline').grid(row=row, column=3, sticky='w')

        row += 1
        self._btn(f, text='执行对齐', command=self.align_csv, bootstyle='primary').grid(row=row, column=1, sticky='we', padx=6)
        return f

    def _build_tab_scope(self):
        """构建示波器标签页"""
        f = ttk.Frame()
        
        if not HAS_MATPLOTLIB:
            ttk.Label(f, text='示波器不可用：需要安装 matplotlib\n\npip install matplotlib', 
                     font=('Arial', 12), justify='center').pack(expand=True)
            return f
        
        # 使用PanedWindow实现可调节分割
        paned = ttk.PanedWindow(f, orient='vertical')
        paned.pack(fill='both', expand=True, padx=4, pady=4)
        
        # 上部：图表区域（权重较大，可自适应缩放）
        chart_frame = ttk.LabelFrame(paned, text='实时波形', padding=4)
        
        # 创建matplotlib图表 - 使用合理的初始尺寸
        self.scope_fig = Figure(figsize=(8, 5), dpi=100)
        self.scope_ax_time = None
        self.scope_ax_freq = None
        
        # 初始化为单图模式
        self._create_scope_single_plot()
        
        self.scope_canvas = FigureCanvasTkAgg(self.scope_fig, master=chart_frame)
        canvas_widget = self.scope_canvas.get_tk_widget()
        canvas_widget.pack(fill='both', expand=True)
        
        # 保存canvas引用供后续使用
        self._scope_canvas_widget = canvas_widget
        
        # 绑定窗口大小变化事件，确保图表自适应
        self._scope_resize_pending = False
        
        def _on_canvas_configure(event):
            # 使用标志位避免频繁重绘
            if not self._scope_resize_pending:
                self._scope_resize_pending = True
                # 延迟执行以避免频繁调用
                self.after(100, self._resize_scope_canvas)
        
        canvas_widget.bind('<Configure>', _on_canvas_configure)
        
        paned.add(chart_frame, weight=3)
        
        # 下部：控制面板（固定最小高度，带滚动条）
        control_outer = ttk.Frame(paned)
        
        # 创建Canvas和Scrollbar用于滚动
        canvas = tk.Canvas(control_outer, height=180, highlightthickness=0)
        scrollbar = ttk.Scrollbar(control_outer, orient='vertical', command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # 鼠标滚轮支持
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        # 保存引用以便后续可能的清理
        self._scope_mousewheel_handler = _on_mousewheel
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # 控制面板内容
        control_frame = ttk.LabelFrame(scrollable_frame, text='示波器控制', padding=6)
        control_frame.pack(fill='both', expand=True, padx=4, pady=4)
        
        # 第一行：变量管理（紧凑布局）
        row1_frame = ttk.Frame(control_frame)
        row1_frame.pack(fill='x', pady=2)
        
        ttk.Label(row1_frame, text='选择变量:').pack(side='left', padx=(0, 2))
        self.scope_log_var_combo = ttk.Combobox(row1_frame, width=20, state='readonly')
        self.scope_log_var_combo.pack(side='left', padx=2)
        self.scope_log_var_combo.bind('<<ComboboxSelected>>', self._on_scope_log_var_selected)
        
        ttk.Label(row1_frame, text='Bit:').pack(side='left', padx=(6, 2))
        self.scope_bit_entry = ttk.Entry(row1_frame, width=8)
        self.scope_bit_entry.pack(side='left', padx=2)
        ttk.Label(row1_frame, text='(可选: 3 或 2:5 或 0:8:2)', foreground='gray', font=('Arial', 8)).pack(side='left', padx=(0, 4))
        
        self._btn(row1_frame, text='添加', command=self._add_scope_variable, bootstyle='success').pack(side='left', padx=2)
        self._btn(row1_frame, text='删除', command=self._remove_scope_variable, bootstyle='danger').pack(side='left', padx=2)
        self._btn(row1_frame, text='清空', command=self._clear_scope_variables, bootstyle='secondary').pack(side='left', padx=2)
        self._btn(row1_frame, text='刷新列表', command=self._refresh_scope_log_vars, bootstyle='info-outline').pack(side='left', padx=2)
        self._btn(row1_frame, text='?', command=self._show_bit_help, bootstyle='info', width=2).pack(side='left', padx=2)
        
        # 第二行：变量列表
        row2_frame = ttk.Frame(control_frame)
        row2_frame.pack(fill='x', pady=2)
        
        ttk.Label(row2_frame, text='监控变量:').pack(side='top', anchor='w')
        
        list_frame = ttk.Frame(row2_frame)
        list_frame.pack(fill='x', pady=2)
        
        self.scope_var_listbox = tk.Listbox(list_frame, height=2, font=('Consolas', 8))
        self.scope_var_listbox.pack(side='left', fill='both', expand=True)
        
        list_scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=self.scope_var_listbox.yview)
        list_scrollbar.pack(side='left', fill='y')
        self.scope_var_listbox.config(yscrollcommand=list_scrollbar.set)
        
        # 第三行：显示设置（水平排列）
        row3_frame = ttk.Frame(control_frame)
        row3_frame.pack(fill='x', pady=2)
        
        ttk.Label(row3_frame, text='时间窗口:').pack(side='left', padx=(0, 2))
        self.scope_time_window = tk.DoubleVar(value=10.0)
        ttk.Spinbox(row3_frame, textvariable=self.scope_time_window, from_=1, to=60, width=6, increment=1).pack(side='left', padx=2)
        ttk.Label(row3_frame, text='秒').pack(side='left', padx=(0, 8))
        
        ttk.Label(row3_frame, text='刷新率:').pack(side='left', padx=(0, 2))
        self.scope_refresh_rate = tk.IntVar(value=10)
        ttk.Spinbox(row3_frame, textvariable=self.scope_refresh_rate, from_=1, to=60, width=6, increment=5).pack(side='left', padx=2)
        ttk.Label(row3_frame, text='Hz').pack(side='left', padx=(0, 8))
        
        self.scope_auto_scale = tk.BooleanVar(value=True)
        ttk.Checkbutton(row3_frame, text='自动缩放', variable=self.scope_auto_scale).pack(side='left', padx=4)
        
        self._btn(row3_frame, text='清除数据', command=self._clear_scope_data, bootstyle='warning').pack(side='left', padx=4)
        
        # 第四行：FFT功能
        row4_frame = ttk.LabelFrame(control_frame, text='快速傅里叶变换 (FFT)', padding=4)
        row4_frame.pack(fill='x', pady=4)
        
        fft_inner = ttk.Frame(row4_frame)
        fft_inner.pack(fill='x', pady=2)
        
        ttk.Label(fft_inner, text='选择变量:').pack(side='left', padx=(0, 2))
        self.fft_log_var_combo = ttk.Combobox(fft_inner, width=20, state='readonly')
        self.fft_log_var_combo.pack(side='left', padx=2)
        
        ttk.Label(fft_inner, text='采样间隔:').pack(side='left', padx=(8, 2))
        self.fft_sample_interval = tk.DoubleVar(value=10.0)
        ttk.Spinbox(fft_inner, textvariable=self.fft_sample_interval, from_=0.1, to=1000, width=8, increment=1).pack(side='left', padx=2)
        ttk.Label(fft_inner, text='ms').pack(side='left', padx=(0, 8))
        
        self._btn(fft_inner, text='计算FFT', command=self._calculate_fft, bootstyle='info').pack(side='left', padx=2)
        self._btn(fft_inner, text='清除FFT', command=self._clear_fft, bootstyle='secondary').pack(side='left', padx=2)
        
        paned.add(control_outer, weight=1)
        
        # 初始化示波器数据结构
        self.scope_variables = []  # [(byte_idx, bit_idx, name, color), ...]
        self.scope_colors = ['#FF6B6B', '#4ECDC4', '#FFD93D', '#A66FFF', '#6BCF7F', '#FF9F43', '#4A90E2', '#FF6FA3']
        self.scope_color_idx = 0
        
        # FFT相关
        self.fft_active = False  # FFT是否激活
        self.fft_data = {}  # FFT结果缓存 {var_key: (freqs, magnitudes)}
        
        # 启动更新线程
        self._scope_update_job = None
        
        # 初始化时刷新日志变量列表（静默模式，不弹窗）
        self.after(100, lambda: self._refresh_scope_log_vars(show_message=False))
        self.after(100, lambda: self._refresh_fft_log_vars(show_message=False))
        
        # 延迟触发一次图表刷新，确保初始显示正确
        self.after(200, self._initial_scope_refresh)
        
        return f
    
    def _initial_scope_refresh(self):
        """初始刷新示波器图表，确保正确适配"""
        try:
            if hasattr(self, '_scope_canvas_widget'):
                # 更新布局以获取实际尺寸
                self._scope_canvas_widget.update_idletasks()
                # 调用统一的调整大小函数
                self._resize_scope_canvas()
        except Exception as e:
            print(f"[DEBUG] Initial scope refresh: {e}")

    # ---------------- 示波器图表布局管理 ----------------
    def _create_scope_single_plot(self):
        """创建单子图模式（仅时域图）"""
        self.scope_fig.clear()
        self.scope_ax_time = self.scope_fig.add_subplot(111)
        self.scope_ax_time.set_xlabel('时间 (秒)', fontsize=10)
        self.scope_ax_time.set_ylabel('数值', fontsize=10)
        self.scope_ax_time.set_title('时域波形', fontsize=11, fontweight='bold')
        self.scope_ax_time.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        self.scope_ax_freq = None
        self.scope_fig.tight_layout(pad=1.5)
    
    def _create_scope_dual_plot(self):
        """创建双子图模式（时域+频域）"""
        self.scope_fig.clear()
        self.scope_ax_time = self.scope_fig.add_subplot(211)
        self.scope_ax_time.set_xlabel('时间 (秒)', fontsize=10)
        self.scope_ax_time.set_ylabel('数值', fontsize=10)
        self.scope_ax_time.set_title('时域波形', fontsize=11, fontweight='bold')
        self.scope_ax_time.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        self.scope_ax_freq = self.scope_fig.add_subplot(212)
        self.scope_ax_freq.set_xlabel('频率 (Hz)', fontsize=10)
        self.scope_ax_freq.set_ylabel('幅值', fontsize=10)
        self.scope_ax_freq.set_title('频域幅频曲线 (FFT)', fontsize=11, fontweight='bold')
        self.scope_ax_freq.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        self.scope_fig.tight_layout(pad=2.0)
    
    def _resize_scope_canvas(self):
        """调整示波器canvas尺寸以适配窗口"""
        self._scope_resize_pending = False
        
        try:
            if not hasattr(self, '_scope_canvas_widget'):
                return
            
            # 获取canvas的实际尺寸
            w_px = self._scope_canvas_widget.winfo_width()
            h_px = self._scope_canvas_widget.winfo_height()
            
            # 检查尺寸是否有效
            if w_px <= 1 or h_px <= 1:
                return
            
            # 转换为英寸（matplotlib使用英寸作为单位）
            dpi = self.scope_fig.dpi
            w_inch = w_px / dpi
            h_inch = h_px / dpi
            
            # 更新figure尺寸
            self.scope_fig.set_size_inches(w_inch, h_inch, forward=True)
            
            # 重新调整布局
            if self.scope_ax_freq is None:
                self.scope_fig.tight_layout(pad=1.5)
            else:
                self.scope_fig.tight_layout(pad=2.0)
            
            # 重绘canvas
            self.scope_canvas.draw_idle()
            
        except Exception as e:
            print(f"[DEBUG] Resize scope canvas error: {e}")

    # ---------------- 事件处理 ----------------
    def _on_closing(self):
        """窗口关闭事件"""
        # 停止视频接收
        if self.video_receiver and self.video_receiver._running:
            self.stop_run()
        
        # 停止视频更新
        if self._video_update_job:
            self.after_cancel(self._video_update_job)
        
        # 停止示波器更新
        if hasattr(self, '_scope_update_job') and self._scope_update_job:
            self.after_cancel(self._scope_update_job)
        
        self.destroy()
    
    def _show_video_placeholder(self):
        """显示视频占位符"""
        self.video_canvas.delete('all')
        w = self.video_canvas.winfo_width()
        h = self.video_canvas.winfo_height()
        if w > 1 and h > 1:
            self.video_canvas.create_text(
                w // 2, h // 2,
                text='等待视频流...\n请点击"启动监听"开始接收',
                fill='white',
                font=('Arial', 14),
                justify='center'
            )
    
    def _clear_data_display(self):
        """清空原始数据显示"""
        if self.video_receiver:
            with self.video_receiver.data_lock:
                self.video_receiver.recent_data.clear()
        self.data_text.delete('1.0', 'end')
        self._log('已清空原始数据显示')
    
    def _refresh_data_display(self):
        """手动刷新原始数据显示"""
        self._update_data_display()
        self._log('已刷新原始数据显示')
    
    def _update_data_display(self):
        """更新原始数据显示"""
        if not self.video_receiver or not self.video_receiver._running:
            return
        
        # 获取最近的数据
        data_list = []
        with self.video_receiver.data_lock:
            # 只显示最后 N 条
            limit = self.data_display_limit.get()
            data_list = list(self.video_receiver.recent_data[-limit:])
        
        # 获取编码和显示格式
        encoding = self.data_encoding.get().lower()
        if encoding == 'utf-8':
            encoding = 'utf-8'
        elif encoding == 'gbk':
            encoding = 'gbk'
        elif encoding == 'gb2312':
            encoding = 'gb2312'
        elif encoding == 'ascii':
            encoding = 'ascii'
        elif encoding == 'latin-1':
            encoding = 'latin-1'
        elif encoding == 'utf-16':
            encoding = 'utf-16'
        elif encoding == 'utf-32':
            encoding = 'utf-32'
        elif encoding == 'big5':
            encoding = 'big5'
        else:
            encoding = 'utf-8'
        
        display_format = self.data_format.get()
        
        # 更新显示
        self.data_text.delete('1.0', 'end')
        
        for timestamp, ftype, data_hex, info in data_list:
            if display_format == '详细':
                # 详细模式：显示所有信息
                line = f"[{timestamp}] {ftype}\n"
                line += f"  Info: {info}\n"
                
                # 尝试用选定编码解析 hex
                try:
                    data_bytes = bytes.fromhex(data_hex.replace('...', ''))
                    decoded_text = data_bytes.decode(encoding, errors='replace')
                    # 只显示可打印字符，其他用 · 代替
                    decoded_text = ''.join(c if c.isprintable() or c in '\n\r\t' else '·' for c in decoded_text)
                    line += f"  {encoding}: {decoded_text[:100]}" + ('...' if len(decoded_text) > 100 else '') + "\n"
                except:
                    line += f"  {encoding}: <decode error>\n"
                
                line += f"  Hex:  {data_hex}\n"
                line += "-" * 80 + "\n"
                
            elif display_format == '简洁':
                # 简洁模式：只显示时间、类型和简要信息
                line = f"[{timestamp[-15:]}] {ftype:12s} | {info[:60]}\n"
                
            elif display_format == '仅Hex':
                # 仅显示 Hex 数据
                line = f"[{timestamp[-15:]}] {ftype:12s}\n"
                line += f"  {data_hex}\n"
                
            elif display_format == '仅文本':
                # 仅显示解码后的文本
                line = f"[{timestamp[-15:]}] {ftype:12s}\n"
                try:
                    data_bytes = bytes.fromhex(data_hex.replace('...', ''))
                    decoded_text = data_bytes.decode(encoding, errors='replace')
                    decoded_text = ''.join(c if c.isprintable() or c in '\n\r\t' else '·' for c in decoded_text)
                    line += f"  {decoded_text}\n"
                except:
                    line += f"  <decode error>\n"
            else:
                line = f"{timestamp} {ftype} {info}\n"
            
            self.data_text.insert('end', line)
        
        self.data_text.see('end')
    
    def _update_video_display(self):
        """更新视频显示（定期调用）"""
        if not self.video_receiver or not self.video_receiver._running:
            self._video_update_job = None
            return
        
        # 获取当前帧
        frame = None
        with self.video_receiver.frame_lock:
            if self.video_receiver.current_frame is not None:
                frame = self.video_receiver.current_frame.copy()
        
        if frame is not None and HAS_CV2:
            try:
                # 获取画布尺寸
                canvas_w = self.video_canvas.winfo_width()
                canvas_h = self.video_canvas.winfo_height()
                
                if canvas_w > 1 and canvas_h > 1:
                    h, w = frame.shape[:2]
                    
                    # 计算缩放比例（保持宽高比）
                    scale_w = canvas_w / w
                    scale_h = canvas_h / h
                    scale = min(scale_w, scale_h, 4.0)  # 最大放大4倍
                    
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    
                    # 缩放图像
                    if scale > 1.0:
                        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                    else:
                        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    
                    # 转换为 RGB（如果是灰度图）
                    if len(resized.shape) == 2:
                        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
                    else:
                        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    
                    # 转换为 PIL Image
                    img_pil = Image.fromarray(resized)
                    img_tk = ImageTk.PhotoImage(image=img_pil)
                    
                    # 在画布上显示
                    self.video_canvas.delete('all')
                    x = (canvas_w - new_w) // 2
                    y = (canvas_h - new_h) // 2
                    self.video_canvas.create_image(x, y, anchor='nw', image=img_tk)
                    self.video_canvas.image = img_tk  # 保持引用
                    
                    # 绘制统计信息
                    info_y = 10
                    self.video_canvas.create_text(
                        10, info_y,
                        anchor='nw',
                        text=f"帧: {self.video_receiver.frame_counter}",
                        fill='lime',
                        font=('Consolas', 10, 'bold')
                    )
                    info_y += 20
                    self.video_canvas.create_text(
                        10, info_y,
                        anchor='nw',
                        text=f"FPS: {self.video_receiver.fps:.1f}",
                        fill='lime',
                        font=('Consolas', 10, 'bold')
                    )
                    info_y += 20
                    self.video_canvas.create_text(
                        10, info_y,
                        anchor='nw',
                        text=f"大小: {w}x{h}",
                        fill='lime',
                        font=('Consolas', 10, 'bold')
                    )
                    
                    # 更新统计标签
                    error_rate = (self.video_receiver.error_packets / self.video_receiver.total_packets * 100) \
                        if self.video_receiver.total_packets > 0 else 0
                    self.stats_label.config(
                        text=f"帧数: {self.video_receiver.frame_counter} | "
                             f"FPS: {self.video_receiver.fps:.1f} | "
                             f"数据包: {self.video_receiver.total_packets} | "
                             f"错误: {self.video_receiver.error_packets} ({error_rate:.1f}%)"
                    )
            except Exception as e:
                print(f"[ERROR] Display error: {e}")
        
        # 更新原始数据显示（每次视频更新时也更新数据）
        self._update_data_display()
        
        # 继续更新（约30fps）
        self._video_update_job = self.after(33, self._update_video_display)
    
    def _log(self, msg: str):
        try:
            self.out_text.insert('end', msg + '\n')
            self.out_text.see('end')
        except Exception:
            # 当 out_text 是 Text + 手动容器时
            self.out_text.insert('end', msg + '\n')
            self.out_text.see('end')

    def apply_theme(self):
        try:
            self.style.theme_use(self.theme_var.get())
        except Exception:
            messagebox.showerror('错误', f'无法应用主题: {self.theme_var.get()}')

    def _pick_png_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.png_dir.set(d)

    def _pick_log_csv(self):
        fpath = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV', '*.csv'), ('All', '*.*')])
        if fpath:
            self.log_csv.set(fpath)

    def _pick_frame_csv(self):
        fpath = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV', '*.csv'), ('All', '*.*')])
        if fpath:
            self.frame_index_csv.set(fpath)
    
    def _refresh_ips(self):
        """刷新可用的 IP 地址列表"""
        ips = get_local_ips()
        # 找到 IP 输入框并更新其值列表
        for widget in self.winfo_children():
            self._update_ip_combobox(widget, ips)
        self._log(f'已刷新 IP 列表: {", ".join(ips)}')
    
    def _update_ip_combobox(self, widget, ips):
        """递归查找并更新 IP Combobox"""
        if isinstance(widget, ttk.Combobox) and widget['textvariable'] == str(self.ip):
            widget['values'] = ips
            return
        for child in widget.winfo_children():
            self._update_ip_combobox(child, ips)
    
    def _validate_custom_frame(self):
        """验证自定义帧配置"""
        try:
            # 验证帧头
            header = self.frame_header.get().replace(' ', '').strip()
            if not header:
                messagebox.showerror('验证失败', '帧头不能为空！')
                return
            
            # 尝试转换为字节
            try:
                header_bytes = bytes.fromhex(header)
            except ValueError:
                messagebox.showerror('验证失败', f'帧头格式错误："{header}"\n请使用有效的十六进制字符（0-9, A-F）')
                return
            
            # 验证帧尾（可选）
            footer = self.frame_footer.get().replace(' ', '').strip()
            footer_bytes = None
            if footer:
                try:
                    footer_bytes = bytes.fromhex(footer)
                except ValueError:
                    messagebox.showerror('验证失败', f'帧尾格式错误："{footer}"\n请使用有效的十六进制字符（0-9, A-F）')
                    return
            
            # 验证最小长度
            min_len = self.frame_min_length.get()
            if min_len < 1:
                messagebox.showerror('验证失败', '最小帧长度必须大于 0')
                return
            
            # 显示验证结果
            result = f"✓ 配置验证通过！\n\n"
            result += f"帧头：{header} ({len(header_bytes)} 字节)\n"
            result += f"  二进制：{' '.join(f'{b:08b}' for b in header_bytes)}\n"
            result += f"  ASCII： {' '.join(chr(b) if 32 <= b < 127 else '·' for b in header_bytes)}\n\n"
            
            if footer_bytes:
                result += f"帧尾：{footer} ({len(footer_bytes)} 字节)\n"
                result += f"  二进制：{' '.join(f'{b:08b}' for b in footer_bytes)}\n"
                result += f"  ASCII： {' '.join(chr(b) if 32 <= b < 127 else '·' for b in footer_bytes)}\n\n"
            else:
                result += f"帧尾：<无>\n\n"
            
            result += f"最小帧长度：{min_len} 字节\n\n"
            result += f"帧格式示例：\n"
            result += f"  [{header}] [数据...] " + (f"[{footer}]" if footer else "") + "\n\n"
            result += f"建议：启用前请先使用默认模式测试，\n确认数据正常后再切换到自定义格式。"
            
            messagebox.showinfo('验证通过', result)
            self._log(f'✓ 自定义帧配置验证通过: 帧头={header}, 帧尾={footer if footer else "无"}')
            
        except Exception as e:
            messagebox.showerror('验证失败', f'验证过程出错：{str(e)}')
    
    def _reset_custom_frame(self):
        """恢复默认配置"""
        self.enable_custom_frame.set(False)
        self.frame_header.set('AA55')
        self.frame_footer.set('')
        self.frame_min_length.set(3)
        self._log('✓ 已恢复自定义帧默认配置')
        messagebox.showinfo('提示', '已恢复默认配置')
    
    def _on_image_size_mode_changed(self, event=None):
        """图像尺寸模式切换回调"""
        self._update_image_size_mode_ui()
    
    def _update_image_size_mode_ui(self):
        """根据尺寸模式更新UI显示"""
        mode = self.image_size_mode.get()
        
        if mode == '固定尺寸':
            # 显示固定尺寸配置
            self.fixed_size_label.grid(row=self.row_fixed, column=0, sticky='e', padx=6, pady=6)
            self.fixed_frame.grid(row=self.row_fixed, column=1, columnspan=2, sticky='w', padx=6)
            
            # 隐藏动态解析配置
            self.dynamic_h_label.grid_remove()
            self.dynamic_h_spinbox.grid_remove()
            self.dynamic_h_byte_label.grid_remove()
            self.dynamic_h_combo.grid_remove()
            self.dynamic_w_label.grid_remove()
            self.dynamic_w_spinbox.grid_remove()
            self.dynamic_w_byte_label.grid_remove()
            self.dynamic_w_combo.grid_remove()
        else:  # 动态解析
            # 隐藏固定尺寸配置
            self.fixed_size_label.grid_remove()
            self.fixed_frame.grid_remove()
            
            # 显示动态解析配置
            self.dynamic_h_label.grid(row=self.row_dynamic_h, column=0, sticky='e', padx=6, pady=6)
            self.dynamic_h_spinbox.grid(row=self.row_dynamic_h, column=1, sticky='w', padx=(6,2))
            self.dynamic_h_byte_label.grid(row=self.row_dynamic_h, column=1, sticky='w', padx=(80, 0))
            self.dynamic_h_combo.grid(row=self.row_dynamic_h, column=2, sticky='w', padx=6)
            
            self.dynamic_w_label.grid(row=self.row_dynamic_w, column=0, sticky='e', padx=6, pady=6)
            self.dynamic_w_spinbox.grid(row=self.row_dynamic_w, column=1, sticky='w', padx=(6,2))
            self.dynamic_w_byte_label.grid(row=self.row_dynamic_w, column=1, sticky='w', padx=(80, 0))
            self.dynamic_w_combo.grid(row=self.row_dynamic_w, column=2, sticky='w', padx=6)
    
    def _validate_image_frame(self):
        """验证图像帧配置"""
        header = self.image_frame_header.get().replace(' ', '').strip().upper()
        footer = self.image_frame_footer.get().replace(' ', '').strip().upper()
        
        errors = []
        
        if not header:
            errors.append("帧头不能为空")
        else:
            try:
                bytes.fromhex(header)
            except ValueError:
                errors.append(f"帧头格式错误: {header}")
        
        if footer:
            try:
                bytes.fromhex(footer)
            except ValueError:
                errors.append(f"帧尾格式错误: {footer}")
        
        h_bytes = self.image_h_bytes.get()
        w_bytes = self.image_w_bytes.get()
        
        if h_bytes < 1 or h_bytes > 4:
            errors.append("H字段字节数必须在1-4之间")
        if w_bytes < 1 or w_bytes > 4:
            errors.append("W字段字节数必须在1-4之间")
        
        if errors:
            msg = "配置错误:\n" + "\n".join(f"• {e}" for e in errors)
            messagebox.showerror("图像帧配置错误", msg)
        else:
            info = f"✓ 图像帧配置有效!\n\n"
            info += f"协议格式: [帧头][图像数据][帧尾]\n\n"
            info += f"帧头: {header} ({len(bytes.fromhex(header))} 字节)\n"
            if footer:
                info += f"帧尾: {footer} ({len(bytes.fromhex(footer))} 字节)\n"
            else:
                info += f"帧尾: (无)\n"
            info += f"\n图像数据格式:\n"
            info += f"  H字段: {h_bytes}字节 ({self.image_h_order.get()})\n"
            info += f"  W字段: {w_bytes}字节 ({self.image_w_order.get()})\n"
            info += f"  像素数据: H×W 字节\n"
            
            # 生成示例
            info += f"\n示例 (64x48图像):\n"
            if h_bytes == 1:
                h_hex = "40"
            elif h_bytes == 2:
                h_hex = '4000' if self.image_h_order.get()=='小端' else '0040'
            elif h_bytes == 4:
                h_hex = '40000000' if self.image_h_order.get()=='小端' else '00000040'
            else:
                h_hex = "..."
            
            if w_bytes == 1:
                w_hex = "30"
            elif w_bytes == 2:
                w_hex = '3000' if self.image_w_order.get()=='小端' else '0030'
            elif w_bytes == 4:
                w_hex = '30000000' if self.image_w_order.get()=='小端' else '00000030'
            else:
                w_hex = "..."
            
            info += f"  UDP包: {header} {h_hex} {w_hex} [3072字节像素] {footer if footer else ''}"
            
            messagebox.showinfo("图像帧配置验证", info)
    
    def _validate_log_frame(self):
        """验证日志帧配置"""
        header = self.log_frame_header.get().replace(' ', '').strip().upper()
        footer = self.log_frame_footer.get().replace(' ', '').strip().upper()
        
        errors = []
        
        if not header:
            errors.append("帧头不能为空")
        else:
            try:
                bytes.fromhex(header)
            except ValueError:
                errors.append(f"帧头格式错误: {header}")
        
        if footer:
            try:
                bytes.fromhex(footer)
            except ValueError:
                errors.append(f"帧尾格式错误: {footer}")
        
        if errors:
            msg = "配置错误:\n" + "\n".join(f"• {e}" for e in errors)
            messagebox.showerror("日志帧配置错误", msg)
        else:
            info = f"✓ 日志帧配置有效!\n\n"
            info += f"帧头: {header} ({len(bytes.fromhex(header))} 字节)\n"
            if footer:
                info += f"帧尾: {footer} ({len(bytes.fromhex(footer))} 字节)\n"
            else:
                info += f"帧尾: (无)\n"
            info += f"\n数据格式: {self.log_frame_format.get()}\n"
            
            if self.log_frame_format.get() == '标准格式':
                info += f"\n示例:\n  UDP包: {header} 02 05 [5字节数据] {footer if footer else ''}"
            else:
                info += f"\n示例:\n  UDP包: {header} [文本数据] {footer if footer else ''}"
            
            messagebox.showinfo("日志帧配置验证", info)
    
    def _add_log_variable(self):
        """添加日志变量"""
        name = self.log_var_name_entry.get().strip()
        if not name:
            messagebox.showerror("错误", "请输入变量名称")
            return
        
        # 检查是否重名
        for var in self.log_variables:
            if var[0] == name:
                messagebox.showerror("错误", f"变量名称 '{name}' 已存在")
                return
        
        byte_pos = self.log_var_byte_pos.get()
        data_type = self.log_var_data_type.get()
        display_format = self.log_var_display_format.get()
        
        # 添加到列表
        self.log_variables.append((name, byte_pos, data_type, display_format))
        
        # 更新树形视图
        self.log_vars_tree.insert('', 'end', values=(name, byte_pos, data_type, display_format))
        
        # 清空输入框
        self.log_var_name_entry.delete(0, 'end')
        
        # 在日志显示区域创建标签
        self._create_log_label(name)
        
        self._log(f'✓ 已添加日志变量: {name} @ byte[{byte_pos}] ({data_type})')
        
        # 自动刷新示波器的变量列表
        if HAS_MATPLOTLIB and hasattr(self, 'scope_log_var_combo'):
            self._refresh_scope_log_vars()
        # 自动刷新FFT的变量列表
        if HAS_MATPLOTLIB and hasattr(self, 'fft_log_var_combo'):
            self._refresh_fft_log_vars(show_message=False)
    
    def _remove_log_variable(self):
        """删除选中的日志变量"""
        selected = self.log_vars_tree.selection()
        if not selected:
            messagebox.showwarning("提示", "请先选择要删除的变量")
            return
        
        for item in selected:
            values = self.log_vars_tree.item(item, 'values')
            var_name = values[0]
            
            # 从列表中移除
            self.log_variables = [v for v in self.log_variables if v[0] != var_name]
            
            # 从树形视图移除
            self.log_vars_tree.delete(item)
            
            # 从显示区域移除
            if var_name in self.log_labels:
                self.log_labels[var_name].destroy()
                del self.log_labels[var_name]
            if var_name in self.log_values:
                del self.log_values[var_name]
            
            self._log(f'✓ 已删除日志变量: {var_name}')
        
        # 重新排列日志标签
        self._rearrange_log_labels()
        
        # 自动刷新示波器的变量列表
        if HAS_MATPLOTLIB and hasattr(self, 'scope_log_var_combo'):
            self._refresh_scope_log_vars()
        # 自动刷新FFT的变量列表
        if HAS_MATPLOTLIB and hasattr(self, 'fft_log_var_combo'):
            self._refresh_fft_log_vars(show_message=False)
    
    def _clear_log_variables(self):
        """清空所有日志变量"""
        if not self.log_variables:
            return
        
        result = messagebox.askyesno("确认", "确定要清空所有日志变量吗？")
        if not result:
            return
        
        # 清空列表
        self.log_variables.clear()
        
        # 清空树形视图
        for item in self.log_vars_tree.get_children():
            self.log_vars_tree.delete(item)
        
        # 清空显示区域
        for label in self.log_labels.values():
            label.destroy()
        self.log_labels.clear()
        self.log_values.clear()
        
        self._log('✓ 已清空所有日志变量')
        
        # 自动刷新示波器的变量列表
        if HAS_MATPLOTLIB and hasattr(self, 'scope_log_var_combo'):
            self._refresh_scope_log_vars()
        # 自动刷新FFT的变量列表
        if HAS_MATPLOTLIB and hasattr(self, 'fft_log_var_combo'):
            self._refresh_fft_log_vars(show_message=False)
    
    def _create_log_label(self, var_name):
        """在日志显示区域创建标签"""
        # 创建一个标签用于显示日志变量
        label = tk.Label(self.log_canvas, text=f'{var_name}: --', 
                        font=('Consolas', 11, 'bold'), 
                        bg='#1e1e1e', fg='#00ff00',
                        anchor='w', padx=10, pady=5)
        
        self.log_labels[var_name] = label
        self.log_values[var_name] = None
        
        # 重新排列所有标签
        self._rearrange_log_labels()
    
    def _rearrange_log_labels(self):
        """重新排列日志标签"""
        y_offset = 5
        max_width = 0
        
        for var_name in self.log_variables:
            name = var_name[0]
            if name in self.log_labels:
                label = self.log_labels[name]
                self.log_canvas.create_window(5, y_offset, window=label, anchor='nw')
                y_offset += 30
                
                # 更新画布尺寸
                label.update_idletasks()
                max_width = max(max_width, label.winfo_width())
        
        # 更新画布滚动区域
        self.log_canvas.configure(scrollregion=(0, 0, max_width + 20, y_offset + 5))
    
    def _update_log_display(self, log_data: bytes):
        """更新日志显示
        
        Args:
            log_data: 日志帧的payload数据（已去除帧头帧尾）
        """
        if not self.log_variables:
            return
        
        for var_name, byte_pos, data_type, display_format in self.log_variables:
            try:
                value = self._parse_log_value(log_data, byte_pos, data_type)
                if value is not None:
                    # 格式化显示
                    try:
                        display_text = display_format.format(value=value)
                    except:
                        display_text = str(value)
                    
                    # 更新标签
                    if var_name in self.log_labels:
                        self.log_labels[var_name].config(text=f'{var_name}: {display_text}')
                        self.log_values[var_name] = value
            except Exception as e:
                # 解析失败，显示错误
                if var_name in self.log_labels:
                    self.log_labels[var_name].config(text=f'{var_name}: ERR')
    
    def _parse_log_value(self, data: bytes, byte_pos: int, data_type: str):
        """从日志数据中解析指定位置的值
        
        Args:
            data: 日志数据
            byte_pos: 字节位置
            data_type: 数据类型
            
        Returns:
            解析后的值，如果失败返回 None
        """
        if byte_pos >= len(data):
            return None
        
        try:
            if data_type == 'uint8':
                return data[byte_pos]
            elif data_type == 'int8':
                return struct.unpack_from('b', data, byte_pos)[0]
            elif data_type == 'uint16_le':
                if byte_pos + 1 >= len(data):
                    return None
                return struct.unpack_from('<H', data, byte_pos)[0]
            elif data_type == 'uint16_be':
                if byte_pos + 1 >= len(data):
                    return None
                return struct.unpack_from('>H', data, byte_pos)[0]
            elif data_type == 'int16_le':
                if byte_pos + 1 >= len(data):
                    return None
                return struct.unpack_from('<h', data, byte_pos)[0]
            elif data_type == 'int16_be':
                if byte_pos + 1 >= len(data):
                    return None
                return struct.unpack_from('>h', data, byte_pos)[0]
            elif data_type == 'uint32_le':
                if byte_pos + 3 >= len(data):
                    return None
                return struct.unpack_from('<I', data, byte_pos)[0]
            elif data_type == 'uint32_be':
                if byte_pos + 3 >= len(data):
                    return None
                return struct.unpack_from('>I', data, byte_pos)[0]
            elif data_type == 'int32_le':
                if byte_pos + 3 >= len(data):
                    return None
                return struct.unpack_from('<i', data, byte_pos)[0]
            elif data_type == 'int32_be':
                if byte_pos + 3 >= len(data):
                    return None
                return struct.unpack_from('>i', data, byte_pos)[0]
            elif data_type == 'float_le':
                if byte_pos + 3 >= len(data):
                    return None
                return struct.unpack_from('<f', data, byte_pos)[0]
            elif data_type == 'float_be':
                if byte_pos + 3 >= len(data):
                    return None
                return struct.unpack_from('>f', data, byte_pos)[0]
            else:
                return None
        except Exception:
            return None
    
    def _on_log_received(self, log_data: bytes):
        """日志接收回调（在UDP线程中调用，需要线程安全更新GUI）"""
        # 使用after方法在主线程中更新GUI
        self.after(0, lambda: self._update_log_display(log_data))
    
    def _validate_log_frame(self):
        """验证日志帧配置"""
        header = self.log_frame_header.get().replace(' ', '').strip().upper()
        footer = self.log_frame_footer.get().replace(' ', '').strip().upper()
        
        errors = []
        
        if not header:
            errors.append("帧头不能为空")
        else:
            try:
                bytes.fromhex(header)
            except ValueError:
                errors.append(f"帧头格式错误: {header}")
        
        if footer:
            try:
                bytes.fromhex(footer)
            except ValueError:
                errors.append(f"帧尾格式错误: {footer}")
        
        if errors:
            msg = "配置错误:\n" + "\n".join(f"• {e}" for e in errors)
            messagebox.showerror("日志帧配置错误", msg)
        else:
            format_type = self.log_frame_format.get()
            info = f"日志帧配置有效!\n\n"
            info += f"帧头: {header} ({len(bytes.fromhex(header))} 字节)\n"
            if footer:
                info += f"帧尾: {footer} ({len(bytes.fromhex(footer))} 字节)\n"
            else:
                info += f"帧尾: (无)\n"
            info += f"数据格式: {format_type}\n"
            
            if format_type == '标准格式':
                info += "\n标准格式包含: [0x02][长度][时间戳8字节][内容]"
            else:
                info += "\n纯文本格式直接发送文本，系统自动记录接收时间"
            
            messagebox.showinfo("日志帧配置验证", info)

    def _pick_video_png_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.video_png_dir.set(d)

    def _pick_video_out(self):
        fpath = filedialog.asksaveasfilename(defaultextension='.mp4', filetypes=[('MP4', '*.mp4'), ('All', '*.*')])
        if fpath:
            self.video_out.set(fpath)

    def start_run(self):
        """启动 UDP 监听（内嵌模式）"""
        if self.video_receiver and self.video_receiver._running:
            messagebox.showwarning('提示', '监听已在运行')
            return
        
        if not HAS_CV2:
            messagebox.showerror('错误', '未安装 OpenCV，无法显示视频\n请运行: pip install opencv-python pillow')
            return
        
        # 验证 IP 地址
        bind_ip = self.ip.get()
        if bind_ip not in ['0.0.0.0', '127.0.0.1']:
            # 检查是否是本机 IP
            local_ips = get_local_ips()
            if bind_ip not in local_ips:
                result = messagebox.askyesno(
                    '警告：IP 地址可能无效',
                    f'IP 地址 "{bind_ip}" 不在本机 IP 列表中！\n\n'
                    f'本机可用的 IP 地址：\n{chr(10).join(local_ips)}\n\n'
                    f'建议使用 "0.0.0.0" 监听所有网络接口。\n\n'
                    f'是否仍然尝试使用 "{bind_ip}"？'
                )
                if not result:
                    return
        
        # 创建并启动接收器
        # 根据模式选择固定尺寸或动态解析
        if self.image_size_mode.get() == '固定尺寸':
            fixed_h = self.image_fixed_h.get()
            fixed_w = self.image_fixed_w.get()
        else:
            fixed_h = 0  # 0表示动态解析
            fixed_w = 0
        
        self.video_receiver = UdpVideoReceiver(
            ip=bind_ip,
            port=self.port.get(),
            save_png=self.save_png.get(),
            png_dir=self.png_dir.get(),
            log_csv=self.log_csv.get(),
            frame_index_csv=self.frame_index_csv.get(),
            enable_custom_image_frame=self.enable_custom_image_frame.get(),
            image_frame_header=self.image_frame_header.get(),
            image_frame_footer=self.image_frame_footer.get(),
            image_h_bytes=self.image_h_bytes.get(),
            image_w_bytes=self.image_w_bytes.get(),
            image_h_order=self.image_h_order.get(),
            image_w_order=self.image_w_order.get(),
            image_fixed_h=fixed_h,
            image_fixed_w=fixed_w,
            image_format=self.image_format.get(),
            enable_custom_log_frame=self.enable_custom_log_frame.get(),
            log_frame_header=self.log_frame_header.get(),
            log_frame_footer=self.log_frame_footer.get(),
            log_frame_format=self.log_frame_format.get(),
            log_callback=self._on_log_received,  # 添加日志回调
            log_variables=self.log_variables,  # 传递日志变量配置
        )
        
        if self.video_receiver.start():
            self._log(f'✓ 启动监听成功: {bind_ip}:{self.port.get()}')
            if bind_ip == '0.0.0.0':
                self._log(f'  监听所有网络接口，可从任何本机 IP 接收数据')
            # 启动视频显示更新
            self._update_video_display()
        else:
            error_msg = '启动 UDP 监听失败！\n\n'
            error_msg += '常见原因：\n'
            error_msg += f'1. IP 地址 "{bind_ip}" 不是本机 IP\n'
            error_msg += f'2. 端口 {self.port.get()} 已被占用\n'
            error_msg += '3. 防火墙阻止了连接\n\n'
            error_msg += '建议：\n'
            error_msg += '• 使用 "0.0.0.0" 监听所有接口\n'
            error_msg += '• 检查防火墙设置\n'
            error_msg += '• 尝试更换端口号'
            messagebox.showerror('启动失败', error_msg)
            self.video_receiver = None

    def stop_run(self):
        """停止 UDP 监听"""
        if self.video_receiver and self.video_receiver._running:
            self._log('停止监听...')
            self.video_receiver.stop()
            self.video_receiver = None
            
            # 停止视频更新
            if self._video_update_job:
                self.after_cancel(self._video_update_job)
                self._video_update_job = None
            
            # 清空画布
            self._show_video_placeholder()
            self.stats_label.config(text='已停止')
        else:
            messagebox.showinfo('提示', '监听未运行')

    def compose_video(self):
        args = [sys.executable, MAIN_SCRIPT, 'video', '--png-dir', self.video_png_dir.get(), '--out', self.video_out.get(), '--fps', str(self.video_fps.get())]
        self._log('合成视频: ' + ' '.join(args))
        subprocess.Popen(args, cwd=SCRIPT_DIR)

    def align_csv(self):
        out_csv = os.path.join(os.path.dirname(self.log_csv.get()) or os.getcwd(), 'aligned.csv')
        args = [
            sys.executable, MAIN_SCRIPT, 'align',
            '--frames-csv', self.frame_index_csv.get(),
            '--logs-csv', self.log_csv.get(),
            '--out-csv', out_csv,
        ]
        self._log('执行对齐: ' + ' '.join(args))
        subprocess.Popen(args, cwd=SCRIPT_DIR)

    def _add_scope_variable(self):
        """添加示波器监控变量"""
        if not HAS_MATPLOTLIB:
            return
        
        # 从下拉框获取选中的日志变量
        selected = self.scope_log_var_combo.get()
        if not selected:
            messagebox.showerror('错误', '请先从下拉框选择一个日志变量')
            return
        
        # 解析选中的变量信息
        # selected 格式: "变量名 (Byte[X], 类型)"
        # 需要找到对应的log_variables条目
        var_info = None
        for log_var in self.log_variables:
            var_name, byte_pos, data_type, display_format = log_var
            if selected.startswith(var_name + ' '):
                var_info = log_var
                break
        
        if not var_info:
            messagebox.showerror('错误', '无法找到对应的日志变量，请刷新列表后重试')
            return
        
        var_name, byte_pos, data_type, display_format = var_info
        byte_idx = byte_pos
        
        # 获取bit索引或切片（可选）
        bit_str = self.scope_bit_entry.get().strip()
        bit_idx = None
        bit_slice = None
        
        if bit_str:
            # 支持三种格式：
            # 1. 单个位: "3" -> bit[3]
            # 2. 切片: "3:5" -> bit[3:5]（提取bit3和bit4，不包括bit5）
            # 3. 切片带步长: "0:8:2" -> bit[0:8:2]（提取bit0,2,4,6）
            
            if ':' in bit_str:
                # 切片格式
                try:
                    parts = bit_str.split(':')
                    if len(parts) == 2:
                        start = int(parts[0]) if parts[0] else 0
                        end = int(parts[1]) if parts[1] else 8
                        bit_slice = (start, end, 1)
                    elif len(parts) == 3:
                        start = int(parts[0]) if parts[0] else 0
                        end = int(parts[1]) if parts[1] else 8
                        step = int(parts[2]) if parts[2] else 1
                        bit_slice = (start, end, step)
                    else:
                        messagebox.showerror('错误', '位切片格式错误，应为 start:end 或 start:end:step')
                        return
                    
                    # 验证范围
                    if start < 0 or start > 7 or end < 0 or end > 8:
                        messagebox.showerror('错误', '位索引必须在0-7之间（end可以为8）')
                        return
                    if start >= end:
                        messagebox.showerror('错误', 'start必须小于end')
                        return
                    
                except ValueError:
                    messagebox.showerror('错误', '位切片格式错误，应为整数')
                    return
            else:
                # 单个位
                try:
                    bit_idx = int(bit_str)
                    if bit_idx < 0 or bit_idx > 7:
                        messagebox.showerror('错误', 'Bit索引必须在0-7之间')
                        return
                except ValueError:
                    messagebox.showerror('错误', 'Bit索引必须是整数')
                    return
        
        # 构建显示名称
        if bit_slice is not None:
            start, end, step = bit_slice
            if step == 1:
                name = f"{var_name}.Bit[{start}:{end}]"
            else:
                name = f"{var_name}.Bit[{start}:{end}:{step}]"
            # 使用元组作为标识，区别于单个bit
            bit_idx = ('slice', start, end, step)
        elif bit_idx is not None:
            name = f"{var_name}.Bit[{bit_idx}]"
        else:
            name = var_name
        
        # 检查是否已存在
        for var in self.scope_variables:
            if var[0] == byte_idx and var[1] == bit_idx:
                messagebox.showwarning('警告', '该变量已存在')
                return
        
        # 分配颜色
        color = self.scope_colors[self.scope_color_idx % len(self.scope_colors)]
        self.scope_color_idx += 1
        
        # 添加变量
        self.scope_variables.append((byte_idx, bit_idx, name, color))
        
        # 更新列表显示
        if isinstance(bit_idx, tuple) and bit_idx[0] == 'slice':
            _, start, end, step = bit_idx
            if step == 1:
                bit_text = f".Bit[{start}:{end}]"
            else:
                bit_text = f".Bit[{start}:{end}:{step}]"
        elif bit_idx is not None:
            bit_text = f".Bit[{bit_idx}]"
        else:
            bit_text = ""
        
        self.scope_var_listbox.insert('end', f"● {name}  (Byte[{byte_idx}]{bit_text}, {data_type})  —  {color}")
        self.scope_var_listbox.itemconfig('end', foreground=color)
        
        # 初始化数据存储
        if self.video_receiver and self.video_receiver._running:
            with self.video_receiver.scope_lock:
                key = (byte_idx, bit_idx)
                if key not in self.video_receiver.scope_data:
                    from collections import deque
                    self.video_receiver.scope_data[key] = deque(maxlen=10000)
        
        self._log(f'✓ 添加监控变量: {name}')
        
        # 清空bit输入
        self.scope_bit_entry.delete(0, 'end')
        
        # 启动更新
        if not self._scope_update_job:
            self._start_scope_update()
    
    def _show_bit_help(self):
        """显示Bit功能帮助信息"""
        help_text = """
【位（Bit）提取功能说明】

一个字节(Byte)包含8位(Bit)，编号从0到7。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 使用方式：

1️⃣ 提取单个位
   输入：3
   说明：提取第3位的值（0或1）
   
2️⃣ 提取位切片
   输入：2:5
   说明：提取bit2到bit4（不包括bit5）
   结果：将这些位组合成一个值
   例如：byte=0b10110100
        bit[2:5] 提取bit2,3,4 = 0b101 = 5
   
3️⃣ 带步长的切片
   输入：0:8:2
   说明：从bit0到bit7，每隔2位取一个
   结果：提取bit0,2,4,6并组合
   
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 应用场景：

• 状态标志位：某位表示电机使能/刹车等
• 多位数值：某几位组合表示速度档位(0-7)
• 间隔采样：提取奇数位或偶数位的数据

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
示例：
假设 Byte[5] = 0b11010110 (十进制214)

• Bit留空    → 显示整个字节: 214
• Bit填"3"   → 显示第3位: 0
• Bit填"2:6" → 显示bit2-5组合: 0b0101 = 5
• Bit填"0:8:2" → 显示bit0,2,4,6: 0b0111 = 7
        """
        messagebox.showinfo("Bit功能帮助", help_text)
    
    def _refresh_scope_log_vars(self, show_message=True):
        """刷新示波器的日志变量下拉列表
        
        Args:
            show_message: 是否显示提示消息（默认True，初始化时设为False避免弹窗）
        """
        if not HAS_MATPLOTLIB:
            return
        
        # 清空当前列表
        self.scope_log_var_combo['values'] = []
        
        # 如果没有配置日志变量
        if not self.log_variables:
            if show_message:
                messagebox.showinfo('提示', '请先在"自定义帧→日志变量配置"中添加日志变量')
            else:
                # 静默模式，仅记录日志
                self._log('提示：可在"自定义帧→日志变量配置"中添加日志变量')
            return
        
        # 构建下拉选项列表
        options = []
        for var_name, byte_pos, data_type, display_format in self.log_variables:
            option = f"{var_name} (Byte[{byte_pos}], {data_type})"
            options.append(option)
        
        self.scope_log_var_combo['values'] = options
        if options:
            self.scope_log_var_combo.current(0)
        
        if show_message:
            self._log(f'✓ 已刷新示波器变量列表 (共{len(options)}个变量)')
        else:
            self._log(f'已加载 {len(options)} 个日志变量')
    
    def _on_scope_log_var_selected(self, event=None):
        """当选择日志变量时的回调"""
        # 可以在这里添加额外的逻辑，例如显示变量详情
        pass
    
    def _refresh_fft_log_vars(self, show_message=True):
        """刷新FFT的日志变量下拉列表"""
        if not HAS_MATPLOTLIB:
            return
        
        # 清空当前列表
        self.fft_log_var_combo['values'] = []
        
        # 如果没有配置日志变量
        if not self.log_variables:
            if show_message:
                self._log('提示：请先添加日志变量')
            return
        
        # 构建下拉选项列表
        options = []
        for var_name, byte_pos, data_type, display_format in self.log_variables:
            option = f"{var_name} (Byte[{byte_pos}], {data_type})"
            options.append(option)
        
        self.fft_log_var_combo['values'] = options
        if options:
            self.fft_log_var_combo.current(0)
        
        if show_message:
            self._log(f'✓ FFT变量列表已更新')
    
    def _calculate_fft(self):
        """计算并显示FFT"""
        if not HAS_MATPLOTLIB:
            return
        
        # 获取选中的变量
        selected = self.fft_log_var_combo.get()
        if not selected:
            messagebox.showerror('错误', '请先选择要分析的变量')
            return
        
        # 解析选中的变量信息
        var_info = None
        for log_var in self.log_variables:
            var_name, byte_pos, data_type, display_format = log_var
            if selected.startswith(var_name + ' '):
                var_info = log_var
                break
        
        if not var_info:
            messagebox.showerror('错误', '无法找到对应的日志变量')
            return
        
        var_name, byte_pos, data_type, display_format = var_info
        byte_idx = byte_pos
        bit_idx = None  # FFT不支持bit提取，使用整字节
        
        # 获取数据
        if not self.video_receiver or not self.video_receiver._running:
            messagebox.showwarning('提示', '请先启动监听以收集数据')
            return
        
        with self.video_receiver.scope_lock:
            key = (byte_idx, bit_idx)
            if key not in self.video_receiver.scope_data:
                messagebox.showwarning('提示', f'变量 {var_name} 暂无数据，请等待数据采集')
                return
            
            data_deque = self.video_receiver.scope_data[key]
            if len(data_deque) < 10:
                messagebox.showwarning('提示', f'数据点太少（{len(data_deque)}个），需要至少10个数据点')
                return
            
            # 提取时间和值
            times = []
            values = []
            for timestamp, value in data_deque:
                times.append(timestamp)
                values.append(value)
        
        if len(values) < 10:
            messagebox.showwarning('提示', '数据不足，无法进行FFT分析')
            return
        
        try:
            # 导入numpy用于FFT计算
            import numpy as np
            
            # 获取采样间隔（毫秒转秒）
            sample_interval_ms = self.fft_sample_interval.get()
            sample_interval = sample_interval_ms / 1000.0  # 转换为秒
            
            # 计算实际采样率（从数据中）
            time_diffs = np.diff(times)
            if len(time_diffs) > 0:
                actual_sample_rate = 1.0 / np.mean(time_diffs)
                self._log(f'实际采样率: {actual_sample_rate:.2f} Hz (平均间隔: {np.mean(time_diffs)*1000:.2f} ms)')
            else:
                actual_sample_rate = 1.0 / sample_interval
            
            # 使用用户指定的采样间隔作为目标采样率
            target_sample_rate = 1.0 / sample_interval
            
            # 对数据进行重采样（线性插值）
            times_array = np.array(times)
            values_array = np.array(values, dtype=float)
            
            # 创建均匀时间序列
            time_start = times_array[0]
            time_end = times_array[-1]
            n_samples = int((time_end - time_start) / sample_interval)
            
            if n_samples < 10:
                messagebox.showwarning('提示', f'采样间隔过大，仅能生成{n_samples}个样本点，请减小采样间隔')
                return
            
            uniform_times = np.linspace(time_start, time_end, n_samples)
            uniform_values = np.interp(uniform_times, times_array, values_array)
            
            # 去除直流分量
            uniform_values = uniform_values - np.mean(uniform_values)
            
            # 应用汉宁窗减少频谱泄漏
            window = np.hanning(len(uniform_values))
            windowed_values = uniform_values * window
            
            # 执行FFT
            fft_result = np.fft.fft(windowed_values)
            n = len(fft_result)
            
            # 只取正频率部分
            freqs = np.fft.fftfreq(n, sample_interval)[:n//2]
            magnitudes = np.abs(fft_result)[:n//2] * 2 / n  # 归一化
            
            # 保存FFT结果
            self.fft_data[key] = (freqs, magnitudes, var_name)
            self.fft_active = True
            
            self._log(f'✓ 已计算 {var_name} 的FFT (样本数: {n}, 频率分辨率: {freqs[1]:.4f} Hz)')
            
            # 更新显示
            if not self._scope_update_job:
                self._start_scope_update()
            
        except Exception as e:
            messagebox.showerror('FFT计算错误', f'计算FFT时出错：\n{str(e)}')
            self._log(f'[ERROR] FFT计算失败: {e}')
    
    def _clear_fft(self):
        """清除FFT显示"""
        self.fft_active = False
        self.fft_data.clear()
        self._log('✓ 已清除FFT显示')
        
        # 切换回单图模式
        if HAS_MATPLOTLIB:
            self._create_scope_single_plot()
            self.scope_canvas.draw()
    
    def _remove_scope_variable(self):
        """删除选中的监控变量"""
        selection = self.scope_var_listbox.curselection()
        if not selection:
            return
        
        idx = selection[0]
        var = self.scope_variables[idx]
        
        # 从列表删除
        self.scope_variables.pop(idx)
        self.scope_var_listbox.delete(idx)
        
        # 从数据存储删除
        if self.video_receiver:
            with self.video_receiver.scope_lock:
                key = (var[0], var[1])
                if key in self.video_receiver.scope_data:
                    del self.video_receiver.scope_data[key]
        
        self._log(f'✓ 删除监控变量: {var[2]}')
    
    def _clear_scope_variables(self):
        """清空所有监控变量"""
        self.scope_variables.clear()
        self.scope_var_listbox.delete(0, 'end')
        
        if self.video_receiver:
            with self.video_receiver.scope_lock:
                self.video_receiver.scope_data.clear()
        
        self._log('✓ 清空所有监控变量')
    
    def _clear_scope_data(self):
        """清除示波器数据"""
        if self.video_receiver:
            with self.video_receiver.scope_lock:
                for key in self.video_receiver.scope_data:
                    self.video_receiver.scope_data[key].clear()
                self.video_receiver.scope_start_time = time.time()
        
        self._log('✓ 清除示波器数据')
    
    def _start_scope_update(self):
        """启动示波器更新"""
        if not HAS_MATPLOTLIB:
            return
        
        self._update_scope_chart()
    
    def _update_scope_chart(self):
        """更新示波器图表"""
        if not HAS_MATPLOTLIB or not self.video_receiver or not self.video_receiver._running:
            self._scope_update_job = None
            return
        
        try:
            # 检查是否需要切换布局模式
            need_dual_plot = self.fft_active and self.fft_data
            current_is_dual = self.scope_ax_freq is not None
            
            if need_dual_plot and not current_is_dual:
                # 切换到双图模式
                self._create_scope_dual_plot()
            elif not need_dual_plot and current_is_dual:
                # 切换到单图模式
                self._create_scope_single_plot()
            
            # === 更新时域图 ===
            self.scope_ax_time.clear()
            self.scope_ax_time.set_xlabel('时间 (秒)', fontsize=10)
            self.scope_ax_time.set_ylabel('数值', fontsize=10)
            self.scope_ax_time.set_title('时域波形', fontsize=11, fontweight='bold')
            self.scope_ax_time.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            # 获取时间窗口
            time_window = self.scope_time_window.get()
            current_time = time.time()
            start_time = self.video_receiver.scope_start_time
            
            # 绘制每个变量
            has_data = False
            with self.video_receiver.scope_lock:
                for byte_idx, bit_idx, name, color in self.scope_variables:
                    key = (byte_idx, bit_idx)
                    if key not in self.video_receiver.scope_data:
                        continue
                    
                    data = self.video_receiver.scope_data[key]
                    if not data:
                        continue
                    
                    # 过滤时间窗口内的数据
                    times = []
                    values = []
                    for timestamp, value in data:
                        rel_time = timestamp - start_time
                        if current_time - timestamp <= time_window:
                            times.append(rel_time)
                            values.append(value)
                    
                    if times:
                        self.scope_ax_time.plot(times, values, label=name, color=color, linewidth=1.5, marker='o', markersize=2)
                        has_data = True
            
            if has_data:
                self.scope_ax_time.legend(loc='upper right', fontsize=8)
                
                # 设置Y轴范围
                if not self.scope_auto_scale.get():
                    self.scope_ax_time.set_ylim(-10, 270)
            else:
                self.scope_ax_time.text(0.5, 0.5, '等待数据...', 
                                  horizontalalignment='center', verticalalignment='center',
                                  transform=self.scope_ax_time.transAxes, fontsize=12, color='gray')
            
            # === 更新频域图（仅在FFT激活且有数据时） ===
            if self.scope_ax_freq is not None and self.fft_active and self.fft_data:
                self.scope_ax_freq.clear()
                self.scope_ax_freq.set_xlabel('频率 (Hz)', fontsize=10)
                self.scope_ax_freq.set_ylabel('幅值', fontsize=10)
                self.scope_ax_freq.set_title('频域幅频曲线 (FFT)', fontsize=11, fontweight='bold')
                self.scope_ax_freq.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
                # 绘制FFT结果
                for key, (freqs, magnitudes, var_name) in self.fft_data.items():
                    # 找到对应的颜色
                    color = '#4A90E2'  # 默认颜色
                    for byte_idx, bit_idx, name, c in self.scope_variables:
                        if (byte_idx, bit_idx) == key:
                            color = c
                            break
                    
                    self.scope_ax_freq.plot(freqs, magnitudes, label=var_name, color=color, linewidth=1.5)
                
                self.scope_ax_freq.legend(loc='upper right', fontsize=8)
                self.scope_ax_freq.set_xlim(left=0)  # 频率从0开始
                
                # 找出主要频率成分（前3个峰值）
                try:
                    import numpy as np
                    for key, (freqs, magnitudes, var_name) in self.fft_data.items():
                        if len(magnitudes) > 10:
                            # 找到峰值
                            peaks_idx = []
                            for i in range(1, len(magnitudes)-1):
                                if magnitudes[i] > magnitudes[i-1] and magnitudes[i] > magnitudes[i+1]:
                                    peaks_idx.append(i)
                            
                            # 按幅值排序，取前3个
                            peaks_idx = sorted(peaks_idx, key=lambda i: magnitudes[i], reverse=True)[:3]
                            
                            # 标注峰值
                            for idx in peaks_idx:
                                if magnitudes[idx] > np.max(magnitudes) * 0.1:  # 只标注幅值>10%最大值的峰
                                    self.scope_ax_freq.annotate(
                                        f'{freqs[idx]:.2f}Hz',
                                        xy=(freqs[idx], magnitudes[idx]),
                                        xytext=(5, 5), textcoords='offset points',
                                        fontsize=8, color='red',
                                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7)
                                    )
                except ImportError:
                    pass  # numpy不可用时跳过峰值标注
            
            # 调整布局并刷新画布（自动适配当前子图数量）
            self.scope_fig.tight_layout(pad=1.5 if self.scope_ax_freq is None else 2.0)
            self.scope_canvas.draw()
            
        except Exception as e:
            print(f"[ERROR] Scope update error: {e}")
            import traceback
            traceback.print_exc()
        
        # 继续更新
        refresh_interval = int(1000 / self.scope_refresh_rate.get())
        self._scope_update_job = self.after(refresh_interval, self._update_scope_chart)
    



def main():
    app = App()
    app.mainloop()


if __name__ == '__main__':
    main()
