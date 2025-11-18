#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py - 配置管理模块
"""

import socket
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class UdpConfig:
    """UDP 配置"""
    ip: str = '0.0.0.0'
    port: int = 8080
    save_png: bool = False
    png_dir: str = 'frames_png'
    log_csv: str = 'logs.csv'
    frame_index_csv: str = 'frames_index.csv'


@dataclass
class CustomImageFrameConfig:
    """自定义图像帧配置"""
    enabled: bool = False
    header: str = 'A0FFFFA0'
    footer: str = 'B0B00A0D'
    h_bytes: int = 1
    w_bytes: int = 1
    h_order: str = '小端'
    w_order: str = '小端'
    fixed_h: int = 120
    fixed_w: int = 188
    size_mode: str = '固定尺寸'  # 固定尺寸 / 动态解析
    format: str = '压缩二值(1位)'  # 图像格式


@dataclass
class CustomLogFrameConfig:
    """自定义日志帧配置"""
    enabled: bool = False
    header: str = 'BB66'
    footer: str = '0D0A'
    format: str = '标准格式'  # 标准格式 / 纯文本


@dataclass
class LogVariable:
    """日志变量配置"""
    name: str
    byte_pos: int
    data_type: str
    display_format: str


@dataclass
class SendConfig:
    """发送配置"""
    target_ip: str = '192.168.1.100'
    target_port: int = 8080
    frame_header: str = 'AA55'
    frame_footer: str = '0D0A'
    interval: float = 1.0


def get_local_ips() -> List[str]:
    """获取本机所有可用的 IP 地址"""
    ips = ['0.0.0.0', '127.0.0.1']
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None):
            ip = info[4][0]
            if ':' not in ip:  # 只保留 IPv4
                if ip not in ips:
                    ips.append(ip)
    except Exception as e:
        print(f"[WARN] Failed to get local IPs: {e}")
    return ips


# 数据类型选项
DATA_TYPES = [
    'uint8', 'int8',
    'uint16_le', 'uint16_be', 'int16_le', 'int16_be',
    'uint32_le', 'uint32_be', 'int32_le', 'int32_be',
    'float_le', 'float_be'
]

# 图像格式选项
IMAGE_FORMATS = [
    '灰度图(8位)', '二值图(8位)', '二值图(自定义8位)', '压缩二值(1位)',
    'RGB565', 'RGB888', 'BGR888', 'RGBA8888'
]

# 显示格式模板
DISPLAY_FORMAT_TEMPLATES = [
    '{value}', '{value:.2f}', '0x{value:02X}',
    '0x{value:04X}', '{value}°C', '{value}%'
]
