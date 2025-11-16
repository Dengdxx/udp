#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tab_run.py - 运行/监听标签页
"""

import os
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                              QLabel, QLineEdit, QPushButton, QCheckBox,
                              QComboBox, QSpinBox, QGroupBox, QFileDialog)
from PyQt6.QtCore import pyqtSignal

from config import UdpConfig, get_local_ips


class RunTab(QWidget):
    """运行/监听标签页"""
    
    # 信号定义
    start_requested = pyqtSignal(UdpConfig)
    stop_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = UdpConfig()
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # 网络配置组
        network_group = QGroupBox("网络配置")
        network_layout = QGridLayout()
        
        # IP 地址
        network_layout.addWidget(QLabel("绑定 IP:"), 0, 0)
        self.ip_combo = QComboBox()
        self.ip_combo.addItems(get_local_ips())
        self.ip_combo.setCurrentText(self.config.ip)
        self.ip_combo.setEditable(True)
        network_layout.addWidget(self.ip_combo, 0, 1)
        
        refresh_ip_btn = QPushButton("刷新IP")
        refresh_ip_btn.clicked.connect(self._refresh_ips)
        network_layout.addWidget(refresh_ip_btn, 0, 2)
        
        # 端口
        network_layout.addWidget(QLabel("端口:"), 0, 3)
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(self.config.port)
        network_layout.addWidget(self.port_spin, 0, 4)
        
        network_group.setLayout(network_layout)
        layout.addWidget(network_group)
        
        # 保存配置组
        save_group = QGroupBox("保存配置")
        save_layout = QGridLayout()
        
        # 保存 PNG 开关
        self.save_png_check = QCheckBox("保存 PNG")
        self.save_png_check.setChecked(self.config.save_png)
        save_layout.addWidget(self.save_png_check, 0, 0, 1, 2)
        
        # PNG 目录
        save_layout.addWidget(QLabel("PNG 目录:"), 1, 0)
        self.png_dir_edit = QLineEdit(self.config.png_dir)
        save_layout.addWidget(self.png_dir_edit, 1, 1)
        png_dir_btn = QPushButton("浏览...")
        png_dir_btn.clicked.connect(self._browse_png_dir)
        save_layout.addWidget(png_dir_btn, 1, 2)
        
        # 日志 CSV
        save_layout.addWidget(QLabel("日志 CSV:"), 2, 0)
        self.log_csv_edit = QLineEdit(self.config.log_csv)
        save_layout.addWidget(self.log_csv_edit, 2, 1)
        log_csv_btn = QPushButton("浏览...")
        log_csv_btn.clicked.connect(self._browse_log_csv)
        save_layout.addWidget(log_csv_btn, 2, 2)
        
        # 帧索引 CSV
        save_layout.addWidget(QLabel("帧索引 CSV:"), 3, 0)
        self.frame_csv_edit = QLineEdit(self.config.frame_index_csv)
        save_layout.addWidget(self.frame_csv_edit, 3, 1)
        frame_csv_btn = QPushButton("浏览...")
        frame_csv_btn.clicked.connect(self._browse_frame_csv)
        save_layout.addWidget(frame_csv_btn, 3, 2)
        
        save_group.setLayout(save_layout)
        layout.addWidget(save_group)
        
        # 控制按钮
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("启动监听")
        self.start_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; }")
        self.start_btn.clicked.connect(self._on_start)
        button_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("停止监听")
        self.stop_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 10px; }")
        self.stop_btn.clicked.connect(self._on_stop)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)
        
        layout.addLayout(button_layout)
        
        layout.addStretch()
    
    def _refresh_ips(self):
        """刷新 IP 列表"""
        current = self.ip_combo.currentText()
        self.ip_combo.clear()
        self.ip_combo.addItems(get_local_ips())
        # 尝试恢复之前的选择
        index = self.ip_combo.findText(current)
        if index >= 0:
            self.ip_combo.setCurrentIndex(index)
    
    def _browse_png_dir(self):
        """浏览 PNG 目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择 PNG 目录", self.png_dir_edit.text())
        if dir_path:
            self.png_dir_edit.setText(dir_path)
    
    def _browse_log_csv(self):
        """浏览日志 CSV"""
        file_path, _ = QFileDialog.getSaveFileName(self, "选择日志 CSV", self.log_csv_edit.text(), "CSV Files (*.csv)")
        if file_path:
            self.log_csv_edit.setText(file_path)
    
    def _browse_frame_csv(self):
        """浏览帧索引 CSV"""
        file_path, _ = QFileDialog.getSaveFileName(self, "选择帧索引 CSV", self.frame_csv_edit.text(), "CSV Files (*.csv)")
        if file_path:
            self.frame_csv_edit.setText(file_path)
    
    def _on_start(self):
        """启动监听"""
        # 更新配置
        self.config.ip = self.ip_combo.currentText()
        self.config.port = self.port_spin.value()
        self.config.save_png = self.save_png_check.isChecked()
        self.config.png_dir = self.png_dir_edit.text()
        self.config.log_csv = self.log_csv_edit.text()
        self.config.frame_index_csv = self.frame_csv_edit.text()
        
        # 发送信号
        self.start_requested.emit(self.config)
        
        # 更新按钮状态
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
    
    def _on_stop(self):
        """停止监听"""
        self.stop_requested.emit()
        
        # 更新按钮状态
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
    
    def get_config(self) -> UdpConfig:
        """获取当前配置"""
        self.config.ip = self.ip_combo.currentText()
        self.config.port = self.port_spin.value()
        self.config.save_png = self.save_png_check.isChecked()
        self.config.png_dir = self.png_dir_edit.text()
        self.config.log_csv = self.log_csv_edit.text()
        self.config.frame_index_csv = self.frame_csv_edit.text()
        return self.config
