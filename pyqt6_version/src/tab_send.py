#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tab_send.py - 数据发送标签页
"""

import os
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                              QLabel, QLineEdit, QPushButton, QSpinBox,
                              QGroupBox, QTextEdit, QComboBox, QDoubleSpinBox,
                              QTreeWidget, QTreeWidgetItem)
from PyQt6.QtCore import pyqtSignal

from udp_sender_qt import UdpSender


class SendTab(QWidget):
    """数据发送标签页"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.sender = UdpSender()
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # 连接配置
        conn_group = QGroupBox("连接配置")
        conn_layout = QGridLayout()
        
        conn_layout.addWidget(QLabel("目标 IP:"), 0, 0)
        self.target_ip_edit = QLineEdit("192.168.1.100")
        conn_layout.addWidget(self.target_ip_edit, 0, 1)
        
        conn_layout.addWidget(QLabel("目标端口:"), 0, 2)
        self.target_port_spin = QSpinBox()
        self.target_port_spin.setRange(1, 65535)
        self.target_port_spin.setValue(8080)
        conn_layout.addWidget(self.target_port_spin, 0, 3)
        
        connect_btn = QPushButton("连接")
        connect_btn.clicked.connect(self._connect)
        conn_layout.addWidget(connect_btn, 0, 4)
        
        conn_group.setLayout(conn_layout)
        layout.addWidget(conn_group)
        
        # 帧格式配置
        frame_group = QGroupBox("帧格式")
        frame_layout = QGridLayout()
        
        frame_layout.addWidget(QLabel("帧头 (Hex):"), 0, 0)
        self.header_edit = QLineEdit("AA55")
        frame_layout.addWidget(self.header_edit, 0, 1)
        
        frame_layout.addWidget(QLabel("帧尾 (Hex):"), 0, 2)
        self.footer_edit = QLineEdit("0D0A")
        frame_layout.addWidget(self.footer_edit, 0, 3)
        
        frame_group.setLayout(frame_layout)
        layout.addWidget(frame_group)
        
        # 数据编辑
        data_group = QGroupBox("数据内容")
        data_layout = QVBoxLayout()
        
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("模式:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Hex", "Text"])
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addStretch()
        data_layout.addLayout(mode_layout)
        
        self.data_edit = QTextEdit()
        self.data_edit.setPlaceholderText("输入数据... (Hex模式: AA 55 01 02, Text模式: 直接输入文本)")
        data_layout.addWidget(self.data_edit)
        
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        
        # 发送控制
        send_layout = QHBoxLayout()
        
        send_once_btn = QPushButton("单次发送")
        send_once_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; padding: 8px; }")
        send_once_btn.clicked.connect(self._send_once)
        send_layout.addWidget(send_once_btn)
        
        send_layout.addWidget(QLabel("定时间隔:"))
        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setRange(0.1, 60.0)
        self.interval_spin.setValue(1.0)
        self.interval_spin.setSuffix(" 秒")
        send_layout.addWidget(self.interval_spin)
        
        self.timer_btn = QPushButton("启动定时发送")
        self.timer_btn.clicked.connect(self._toggle_timer)
        send_layout.addWidget(self.timer_btn)
        
        layout.addLayout(send_layout)
        
        # 状态显示
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(100)
        layout.addWidget(self.status_text)
    
    def _connect(self):
        """连接到目标"""
        ip = self.target_ip_edit.text()
        port = self.target_port_spin.value()
        
        if self.sender.connect(ip, port):
            self.status_text.append(f"✓ 已连接到 {ip}:{port}")
        else:
            self.status_text.append(f"✗ 连接失败")
    
    def _send_once(self):
        """单次发送"""
        try:
            # 获取数据
            mode = self.mode_combo.currentText()
            text = self.data_edit.toPlainText()
            
            if mode == "Hex":
                payload = bytes.fromhex(text.replace(' ', '').replace('\n', ''))
            else:
                payload = text.encode('utf-8')
            
            # 构建帧
            header = self.header_edit.text()
            footer = self.footer_edit.text()
            frame = self.sender.build_custom_frame(header, footer, payload)
            
            # 发送
            if self.sender.send_data(frame):
                self.status_text.append(f"✓ 发送成功: {len(frame)} 字节")
            else:
                self.status_text.append(f"✗ 发送失败")
        
        except Exception as e:
            self.status_text.append(f"✗ 错误: {e}")
    
    def _toggle_timer(self):
        """切换定时发送"""
        if self.timer_btn.text() == "启动定时发送":
            self.timer_btn.setText("停止定时发送")
            self.status_text.append("✓ 定时发送已启动")
        else:
            self.timer_btn.setText("启动定时发送")
            self.status_text.append("✓ 定时发送已停止")
