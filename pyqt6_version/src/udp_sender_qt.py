#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
udp_sender_qt.py - UDP 发送器（PyQt6 版本）
"""

import socket
from datetime import datetime
from typing import Optional
from PyQt6.QtCore import QThread, pyqtSignal, QTimer


class UdpSenderThread(QThread):
    """UDP 发送线程"""
    
    # 信号定义
    send_completed = pyqtSignal(bool, str)  # success, message
    send_history_updated = pyqtSignal(str, str, str)  # timestamp, data_hex, description
    
    def __init__(self, target_ip: str, target_port: int):
        super().__init__()
        self.target_ip = target_ip
        self.target_port = target_port
        self._sock: Optional[socket.socket] = None
        self.send_count = 0
        
        # 定时发送
        self._timer_interval = 1.0
        self._timer_data = b''
        self._timer_enabled = False
    
    def connect(self) -> bool:
        """连接到目标"""
        try:
            if self._sock:
                self._sock.close()
            
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            return True
        except Exception as e:
            self.send_completed.emit(False, f"连接失败: {e}")
            return False
    
    def send_data(self, data: bytes, description: str = '') -> bool:
        """发送数据"""
        if not self._sock:
            self.send_completed.emit(False, "未连接")
            return False
        
        try:
            self._sock.sendto(data, (self.target_ip, self.target_port))
            self.send_count += 1
            
            # 记录历史
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            data_hex = data[:100].hex() + ('...' if len(data) > 100 else '')
            self.send_history_updated.emit(timestamp, data_hex, description or f"发送 {len(data)} 字节")
            
            self.send_completed.emit(True, f"发送成功: {len(data)} 字节")
            return True
        except Exception as e:
            self.send_completed.emit(False, f"发送失败: {e}")
            return False
    
    def start_timer_send(self, data: bytes, interval: float):
        """启动定时发送"""
        self._timer_data = data
        self._timer_interval = interval
        self._timer_enabled = True
    
    def stop_timer_send(self):
        """停止定时发送"""
        self._timer_enabled = False
    
    def run(self):
        """定时发送循环"""
        while self._timer_enabled:
            if self._timer_data and self._sock:
                self.send_data(self._timer_data, f"定时发送 (间隔={self._timer_interval}s)")
            self.msleep(int(self._timer_interval * 1000))
    
    def close(self):
        """关闭发送器"""
        self.stop_timer_send()
        if self._sock:
            try:
                self._sock.close()
            except:
                pass
            self._sock = None


class UdpSender:
    """UDP 发送器（非线程版本，用于简单发送）"""
    
    def __init__(self):
        self._sock: Optional[socket.socket] = None
        self.target_ip = ''
        self.target_port = 8080
        self.send_count = 0
    
    def connect(self, target_ip: str, target_port: int) -> bool:
        """连接到目标"""
        try:
            if self._sock:
                self._sock.close()
            
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.target_ip = target_ip
            self.target_port = target_port
            return True
        except Exception:
            return False
    
    def send_data(self, data: bytes) -> bool:
        """发送数据"""
        if not self._sock:
            return False
        
        try:
            self._sock.sendto(data, (self.target_ip, self.target_port))
            self.send_count += 1
            return True
        except Exception:
            return False
    
    def build_custom_frame(self, header: str = '', footer: str = '', payload: bytes = b'') -> bytes:
        """构建自定义帧"""
        frame = b''
        if header:
            try:
                frame += bytes.fromhex(header.replace(' ', ''))
            except:
                pass
        frame += payload
        if footer:
            try:
                frame += bytes.fromhex(footer.replace(' ', ''))
            except:
                pass
        return frame
    
    def close(self):
        """关闭"""
        if self._sock:
            try:
                self._sock.close()
            except:
                pass
            self._sock = None
