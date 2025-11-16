#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py - 工具函数模块
"""

import numpy as np
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt


def numpy_to_qimage(img: np.ndarray) -> QImage:
    """将 NumPy 图像数组转换为 QImage"""
    if img is None:
        return QImage()
    
    # 处理灰度图
    if len(img.shape) == 2:
        h, w = img.shape
        bytes_per_line = w
        return QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
    
    # 处理 RGB 图像
    elif len(img.shape) == 3:
        h, w, c = img.shape
        if c == 3:
            # RGB 格式
            bytes_per_line = 3 * w
            return QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        elif c == 4:
            # RGBA 格式
            bytes_per_line = 4 * w
            return QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888)
    
    return QImage()


def numpy_to_qpixmap(img: np.ndarray, scale_width: int = 0, scale_height: int = 0) -> QPixmap:
    """将 NumPy 图像转换为 QPixmap，可选缩放"""
    qimage = numpy_to_qimage(img)
    if qimage.isNull():
        return QPixmap()
    
    pixmap = QPixmap.fromImage(qimage)
    
    # 如果指定了缩放尺寸
    if scale_width > 0 and scale_height > 0:
        pixmap = pixmap.scaled(
            scale_width, scale_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
    
    return pixmap


def format_bytes(num_bytes: int) -> str:
    """格式化字节数为人类可读格式"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"


def format_hex(data: bytes, max_length: int = 100) -> str:
    """格式化字节数据为十六进制字符串"""
    hex_str = data.hex().upper()
    if len(hex_str) > max_length * 2:
        return hex_str[:max_length * 2] + '...'
    return hex_str


def parse_hex_input(text: str) -> bytes:
    """解析十六进制输入（支持空格分隔）"""
    text = text.strip().replace(' ', '').replace('\n', '').replace('\r', '')
    try:
        return bytes.fromhex(text)
    except ValueError:
        raise ValueError("无效的十六进制格式")


def validate_ip(ip: str) -> bool:
    """验证 IP 地址格式"""
    if ip == '0.0.0.0':
        return True
    parts = ip.split('.')
    if len(parts) != 4:
        return False
    try:
        return all(0 <= int(part) <= 255 for part in parts)
    except ValueError:
        return False


def validate_port(port: int) -> bool:
    """验证端口号"""
    return 1 <= port <= 65535
