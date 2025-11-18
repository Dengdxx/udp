#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
udp_receiver_qt.py - UDP 接收器（PyQt6 版本）

使用 Qt 信号/槽机制实现线程安全的数据传输
"""

import os
import socket
import struct
import time
import csv
from datetime import datetime
from typing import Optional, Tuple
from collections import deque

import numpy as np
import cv2
from PyQt6.QtCore import QThread, pyqtSignal, QMutex


def sanitize_csv_text(text: str) -> str:
    """清理CSV文本,移除会导致读取问题的特殊字符"""
    if not text:
        return text
    text = text.replace('\x00', '')
    text = text.replace('\x1A', '')
    cleaned = ''.join(c for c in text if ord(c) >= 0x20 or c in '\t\n\r')
    return cleaned


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
    pixels = np.frombuffer(data[3:], dtype=np.uint8)
    img = pixels.reshape((h, w))
    return h, w, img


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


def decode_image_by_format(pixel_data: bytes, h: int, w: int, format_type: str) -> np.ndarray:
    """根据指定格式解码图像数据"""
    pixel_count = h * w
    
    if format_type == '灰度图(8位)':
        if len(pixel_data) < pixel_count:
            raise ValueError(f"gray image data too short: need {pixel_count}, got {len(pixel_data)}")
        img = np.frombuffer(pixel_data[:pixel_count], dtype=np.uint8)
        return img.reshape((h, w))
    
    elif format_type == '二值图(8位)':
        if len(pixel_data) < pixel_count:
            raise ValueError(f"binary image data too short: need {pixel_count}, got {len(pixel_data)}")
        pixels = np.frombuffer(pixel_data[:pixel_count], dtype=np.uint8)
        img = np.where(pixels > 127, 255, 0).astype(np.uint8)
        return img.reshape((h, w))
    
    elif format_type == '二值图(自定义8位)':
        if len(pixel_data) < pixel_count:
            raise ValueError(f"custom 8bit binary image data too short: need {pixel_count}, got {len(pixel_data)}")
        pixels = np.frombuffer(pixel_data[:pixel_count], dtype=np.uint8)
        
        # 创建彩色图像 (h, w, 3)
        img_rgb = np.zeros((pixel_count, 3), dtype=np.uint8)
        
        # 0 和 255 保持黑白
        black_mask = (pixels == 0)
        white_mask = (pixels == 255)
        img_rgb[black_mask] = [0, 0, 0]
        img_rgb[white_mask] = [255, 255, 255]
        
        # 其他值 (1-254) 映射为彩色
        # 使用 HSV 色轮映射: 不同的值映射到不同的色相
        color_mask = ~(black_mask | white_mask)
        color_indices = np.where(color_mask)[0]
        
        if len(color_indices) > 0:
            color_values = pixels[color_indices]
            # 将 1-254 映射到色相环 0-360 度
            # 排除 0 和 255，所以有效范围是 1-254
            hue = ((color_values - 1) * 360.0 / 254.0).astype(np.float32)
            
            # 转换 HSV 到 RGB (S=1.0, V=1.0 for 鲜艳颜色)
            # 使用简化的 HSV->RGB 转换
            h_norm = hue / 60.0
            h_int = h_norm.astype(np.int32) % 6
            f = h_norm - np.floor(h_norm)
            
            for i, idx in enumerate(color_indices):
                hi = h_int[i]
                fi = f[i]
                
                if hi == 0:
                    img_rgb[idx] = [255, int(fi * 255), 0]
                elif hi == 1:
                    img_rgb[idx] = [int((1 - fi) * 255), 255, 0]
                elif hi == 2:
                    img_rgb[idx] = [0, 255, int(fi * 255)]
                elif hi == 3:
                    img_rgb[idx] = [0, int((1 - fi) * 255), 255]
                elif hi == 4:
                    img_rgb[idx] = [int(fi * 255), 0, 255]
                else:  # hi == 5
                    img_rgb[idx] = [255, 0, int((1 - fi) * 255)]
        
        return img_rgb.reshape((h, w, 3))
    
    elif format_type == '压缩二值(1位)':
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
        expected_bytes = pixel_count * 2
        if len(pixel_data) < expected_bytes:
            raise ValueError(f"RGB565 data too short: need {expected_bytes}, got {len(pixel_data)}")
        
        rgb565 = np.frombuffer(pixel_data[:expected_bytes], dtype=np.uint16)
        r = ((rgb565 & 0xF800) >> 11).astype(np.uint8)
        g = ((rgb565 & 0x07E0) >> 5).astype(np.uint8)
        b = (rgb565 & 0x001F).astype(np.uint8)
        
        r = (r << 3) | (r >> 2)
        g = (g << 2) | (g >> 4)
        b = (b << 3) | (b >> 2)
        
        img = np.stack([r, g, b], axis=-1)
        return img.reshape((h, w, 3))
    
    elif format_type == 'RGB888':
        expected_bytes = pixel_count * 3
        if len(pixel_data) < expected_bytes:
            raise ValueError(f"RGB888 data too short: need {expected_bytes}, got {len(pixel_data)}")
        img = np.frombuffer(pixel_data[:expected_bytes], dtype=np.uint8)
        return img.reshape((h, w, 3))
    
    elif format_type == 'BGR888':
        expected_bytes = pixel_count * 3
        if len(pixel_data) < expected_bytes:
            raise ValueError(f"BGR888 data too short: need {expected_bytes}, got {len(pixel_data)}")
        img = np.frombuffer(pixel_data[:expected_bytes], dtype=np.uint8)
        img = img.reshape((h, w, 3))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    elif format_type == 'RGBA8888':
        expected_bytes = pixel_count * 4
        if len(pixel_data) < expected_bytes:
            raise ValueError(f"RGBA8888 data too short: need {expected_bytes}, got {len(pixel_data)}")
        img = np.frombuffer(pixel_data[:expected_bytes], dtype=np.uint8)
        img = img.reshape((h, w, 4))
        return img[:, :, :3]
    
    else:
        raise ValueError(f"Unknown image format: {format_type}")


class UdpReceiverThread(QThread):
    """UDP 接收线程（PyQt6 版本）"""
    
    # 信号定义
    frame_received = pyqtSignal(np.ndarray, int, int, int)  # image, frame_id, h, w
    log_received = pyqtSignal(bytes)  # payload
    stats_updated = pyqtSignal(dict)  # {fps, total_packets, error_packets, frame_counter}
    data_packet_received = pyqtSignal(str, str, str, str)  # timestamp, type, data_hex, info
    scope_data_received = pyqtSignal(int, int, float)  # byte_idx, value, timestamp
    error_occurred = pyqtSignal(str)  # error message
    
    def __init__(self, ip: str, port: int, 
                 save_png: bool = False,
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
                 image_fixed_h: int = 0,
                 image_fixed_w: int = 0,
                 image_format: str = '灰度图(8位)',
                 enable_custom_log_frame: bool = False,
                 log_frame_header: str = '',
                 log_frame_footer: str = '',
                 log_frame_format: str = '标准格式',
                 log_variables: list = None):
        super().__init__()
        
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
        
        # 日志变量配置
        self.log_variables = log_variables if log_variables else []
        
        self._sock: Optional[socket.socket] = None
        self._running = False
        
        # 统计信息
        self.frame_counter = 0
        self.total_packets = 0
        self.error_packets = 0
        self.fps = 0.0
        self._fps_timer = time.time()
        self._fps_frame_count = 0
        
        # 示波器数据
        self.scope_data = {}
        self.scope_lock = QMutex()
        
        # CSV 文件
        self._log_csv_fp: Optional[object] = None
        self._log_writer: Optional[object] = None
        self._frame_index_fp: Optional[object] = None
        self._frame_index_writer: Optional[object] = None
    
    def init_socket(self) -> bool:
        """初始化 socket"""
        try:
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
                csv_header = ["host_recv_iso", "log_text_hex", "log_text_utf8"]
                for var_name, _, _, _ in self.log_variables:
                    csv_header.append(var_name)
                self._log_writer.writerow(csv_header)
            
            frame_csv_exists = os.path.exists(self.frame_index_csv) and os.path.getsize(self.frame_index_csv) > 0
            self._frame_index_fp = open(self.frame_index_csv, 'a', newline='', encoding='utf-8')
            self._frame_index_writer = csv.writer(self._frame_index_fp)
            if not frame_csv_exists:
                self._frame_index_writer.writerow(["frame_id", "host_recv_iso", "png_path", "h", "w"])
            
            return True
        except Exception as e:
            self.error_occurred.emit(f"Socket 初始化失败: {e}")
            return False
    
    def stop(self):
        """停止接收"""
        self._running = False
    
    def cleanup(self):
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
        """更新 FPS"""
        self._fps_frame_count += 1
        elapsed = time.time() - self._fps_timer
        if elapsed >= 1.0:
            self.fps = self._fps_frame_count / elapsed
            self._fps_frame_count = 0
            self._fps_timer = time.time()
            
            # 发送统计信息
            self.stats_updated.emit({
                'fps': self.fps,
                'total_packets': self.total_packets,
                'error_packets': self.error_packets,
                'frame_counter': self.frame_counter
            })
    
    def _parse_custom_image_data(self, data: bytes) -> Optional[Tuple[int, int, np.ndarray]]:
        """解析自定义图像数据"""
        try:
            if self.image_fixed_h > 0 and self.image_fixed_w > 0:
                h = self.image_fixed_h
                w = self.image_fixed_w
                img = decode_image_by_format(data, h, w, self.image_format)
                if img is None:
                    return None
                return h, w, img
            else:
                idx = 0
                if len(data) < self.image_h_bytes + self.image_w_bytes:
                    return None
                
                h_bytes = data[idx:idx+self.image_h_bytes]
                if self.image_h_order == '大端':
                    h = int.from_bytes(h_bytes, 'big')
                else:
                    h = int.from_bytes(h_bytes, 'little')
                idx += self.image_h_bytes
                
                w_bytes = data[idx:idx+self.image_w_bytes]
                if self.image_w_order == '大端':
                    w = int.from_bytes(w_bytes, 'big')
                else:
                    w = int.from_bytes(w_bytes, 'little')
                idx += self.image_w_bytes
                
                pixel_data = data[idx:]
                img = decode_image_by_format(pixel_data, h, w, self.image_format)
                
                if img is None:
                    return None
                return h, w, img
        except Exception as e:
            self.error_occurred.emit(f"图像解析错误: {e}")
            return None
    
    def _parse_custom_frame(self, data: bytes) -> Optional[bytes]:
        """解析自定义帧格式"""
        if not self.enable_custom_image_frame or not self.image_frame_header_bytes:
            return None
        
        header_pos = data.find(self.image_frame_header_bytes)
        if header_pos == -1:
            return None
        
        data_start = header_pos + len(self.image_frame_header_bytes)
        
        if self.image_frame_footer_bytes:
            footer_pos = data.find(self.image_frame_footer_bytes, data_start)
            if footer_pos == -1:
                return None
            frame_data = data[data_start:footer_pos]
        else:
            frame_data = data[data_start:]
        
        if len(frame_data) < 3:
            return None
        
        return frame_data
    
    def _parse_custom_log_frame(self, data: bytes) -> Optional[bytes]:
        """解析自定义日志帧"""
        if not self.enable_custom_log_frame or not self.log_frame_header_bytes:
            return None
        
        header_pos = data.find(self.log_frame_header_bytes)
        if header_pos == -1:
            return None
        
        data_start = header_pos + len(self.log_frame_header_bytes)
        
        if self.log_frame_footer_bytes:
            footer_pos = data.find(self.log_frame_footer_bytes, data_start)
            if footer_pos == -1:
                return None
            frame_data = data[data_start:footer_pos]
        else:
            frame_data = data[data_start:]
        
        if len(frame_data) < 1:
            return None
        
        if self.log_frame_format == '标准格式':
            if len(frame_data) < 2:
                return None
            try:
                length = frame_data[1]
                if len(frame_data) != 1 + 1 + length:
                    return None
                return frame_data[2:2 + length]
            except:
                return None
        else:
            return frame_data
    
    def _parse_log_variables(self, payload: bytes) -> dict:
        """解析日志变量"""
        result = {}
        for var_name, byte_pos, data_type, _ in self.log_variables:
            try:
                value = self._parse_single_log_value(payload, byte_pos, data_type)
                result[var_name] = value if value is not None else ''
            except Exception:
                result[var_name] = ''
        return result
    
    def _parse_single_log_value(self, data: bytes, byte_pos: int, data_type: str):
        """解析单个日志值"""
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
    
    def run(self):
        """线程主循环"""
        if not self.init_socket():
            return
        
        self._running = True
        
        try:
            while self._running:
                try:
                    data, addr = self._sock.recvfrom(65535)
                except socket.timeout:
                    continue
                except Exception as e:
                    if self._running:
                        self.error_occurred.emit(f"接收错误: {e}")
                    break
                
                self.total_packets += 1
                host_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                recv_time = time.time()
                
                if not data:
                    continue
                
                # 提取示波器数据（从原始UDP数据包中）
                # 发射所有字节的值供示波器使用
                for byte_idx in range(len(data)):
                    byte_val = data[byte_idx]
                    self.scope_data_received.emit(byte_idx, byte_val, recv_time)
                
                # 尝试解析自定义图像帧
                if self.enable_custom_image_frame:
                    custom_image_data = self._parse_custom_frame(data)
                    if custom_image_data is not None:
                        result = self._parse_custom_image_data(custom_image_data)
                        if result is not None:
                            h, w, img = result
                            self.frame_counter += 1
                            self._update_fps()
                            
                            # 发送图像信号
                            self.frame_received.emit(img, self.frame_counter, h, w)
                            
                            # 记录数据包
                            format_name = self.image_format
                            if format_name == '压缩二值(1位)':
                                data_size = len(custom_image_data)
                                compression_ratio = (h * w) / data_size if data_size > 0 else 0
                                info = f"{format_name} Frame {self.frame_counter}: {w}x{h}, {data_size} bytes ({compression_ratio:.1f}:1)"
                            else:
                                info = f"{format_name} Frame {self.frame_counter}: {w}x{h}, {len(data)} bytes total"
                            
                            self.data_packet_received.emit(
                                host_iso,
                                'CUSTOM_IMAGE',
                                data[:100].hex() + ('...' if len(data) > 100 else ''),
                                info
                            )
                            
                            # 保存 PNG
                            if self.save_png:
                                png_path = os.path.join(self.png_dir, f"frame_{self.frame_counter:06d}.png")
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
                            text_utf8 = sanitize_csv_text(text_utf8)
                        except:
                            pass
                        text_hex = payload.hex()
                        
                        # 发送日志信号
                        self.log_received.emit(payload)
                        
                        # 记录数据包 - 传递payload的hex，不是整个UDP包的hex
                        payload_hex = payload.hex()
                        if len(payload_hex) > 500:
                            payload_hex = payload_hex[:500] + '...'
                        
                        # 构建更清晰的Info信息
                        if text_utf8 and text_utf8.isprintable():
                            info_text = f"Custom LOG [{self.log_frame_format}]: {text_utf8[:50]}" + ('...' if len(text_utf8) > 50 else '')
                        else:
                            info_text = f"Custom LOG [{self.log_frame_format}]: Hex={payload_hex[:40]}" + ('...' if len(payload_hex) > 40 else '') + f" ({len(payload)} bytes)"
                        
                        self.data_packet_received.emit(
                            host_iso,
                            'CUSTOM_LOG',
                            payload_hex,  # 传递payload的hex，不是整个UDP包
                            info_text
                        )
                        
                        # 写入 CSV
                        if self._log_writer:
                            row_data = [host_iso, text_hex, text_utf8]
                            log_vars = self._parse_log_variables(payload)
                            for var_name, _, _, _ in self.log_variables:
                                row_data.append(log_vars.get(var_name, ''))
                            self._log_writer.writerow(row_data)
                            self._log_csv_fp.flush()
                        
                        continue
                
                # 如果启用了自定义帧但无法解析
                if self.enable_custom_image_frame or self.enable_custom_log_frame:
                    self.error_packets += 1
                    self.data_packet_received.emit(
                        host_iso,
                        'INVALID_CUSTOM',
                        data[:200].hex() + ('...' if len(data) > 200 else ''),
                        f"Failed to parse custom frame: {len(data)} bytes"
                    )
                    continue
                
                # 默认帧格式处理
                ftype = data[0]
                
                try:
                    if ftype == FrameType.IMAGE:
                        h, w, img = parse_image_frame(data)
                        self.frame_counter += 1
                        self._update_fps()
                        
                        self.frame_received.emit(img, self.frame_counter, h, w)
                        
                        self.data_packet_received.emit(
                            host_iso,
                            'IMAGE',
                            data[:100].hex() + ('...' if len(data) > 100 else ''),
                            f"Frame {self.frame_counter}: {w}x{h}, {len(data)} bytes"
                        )
                        
                        png_path = ''
                        if self.save_png:
                            png_path = os.path.join(self.png_dir, f"frame_{self.frame_counter:06d}.png")
                            cv2.imwrite(png_path, img)
                        
                        if self._frame_index_writer:
                            self._frame_index_writer.writerow([self.frame_counter, host_iso, png_path, h, w])
                            self._frame_index_fp.flush()
                    
                    elif ftype == FrameType.BINARY_IMAGE:
                        h, w, img = parse_binary_image_frame(data)
                        self.frame_counter += 1
                        self._update_fps()
                        
                        self.frame_received.emit(img, self.frame_counter, h, w)
                        
                        self.data_packet_received.emit(
                            host_iso,
                            'BINARY_IMAGE',
                            data[:100].hex() + ('...' if len(data) > 100 else ''),
                            f"Binary Frame {self.frame_counter}: {w}x{h}, {len(data)} bytes"
                        )
                        
                        png_path = ''
                        if self.save_png:
                            png_path = os.path.join(self.png_dir, f"frame_{self.frame_counter:06d}.png")
                            cv2.imwrite(png_path, img)
                        
                        if self._frame_index_writer:
                            self._frame_index_writer.writerow([self.frame_counter, host_iso, png_path, h, w])
                            self._frame_index_fp.flush()
                    
                    elif ftype == FrameType.LOG:
                        if len(data) < 1 + 1:
                            raise ValueError("log frame too short")
                        length = data[1]
                        if len(data) != 1 + 1 + length:
                            raise ValueError("log frame size mismatch")
                        payload = data[2:2 + length]
                        
                        text_utf8 = ''
                        try:
                            text_utf8 = payload.decode('utf-8', errors='replace')
                            text_utf8 = sanitize_csv_text(text_utf8)
                        except:
                            pass
                        text_hex = payload.hex()
                        
                        self.log_received.emit(payload)
                        
                        display_hex = data.hex()
                        if len(display_hex) > 500:
                            display_hex = display_hex[:500] + '...'
                        
                        self.data_packet_received.emit(
                            host_iso,
                            'LOG',
                            display_hex,
                            f"LOG: {text_utf8[:50]}" + ('...' if len(text_utf8) > 50 else '')
                        )
                        
                        if self._log_writer:
                            row_data = [host_iso, text_hex, text_utf8]
                            log_vars = self._parse_log_variables(payload)
                            for var_name, _, _, _ in self.log_variables:
                                row_data.append(log_vars.get(var_name, ''))
                            self._log_writer.writerow(row_data)
                            self._log_csv_fp.flush()
                    
                    else:
                        self.error_packets += 1
                        self.data_packet_received.emit(
                            host_iso,
                            f'UNKNOWN(0x{ftype:02X})',
                            data[:100].hex() + ('...' if len(data) > 100 else ''),
                            f"Unknown frame type, {len(data)} bytes"
                        )
                
                except Exception as e:
                    self.error_packets += 1
                    self.error_occurred.emit(f"解析错误: {e}")
                    self.data_packet_received.emit(
                        host_iso,
                        'ERROR',
                        data[:100].hex() + ('...' if len(data) > 100 else ''),
                        f"Parse error: {str(e)}"
                    )
        
        finally:
            self.cleanup()
