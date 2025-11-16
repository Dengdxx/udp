#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tab_align.py - 数据对齐标签页
"""

import os
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                              QLabel, QLineEdit, QPushButton, QGroupBox,
                              QFileDialog, QTextEdit)
from PyQt6.QtCore import pyqtSignal, QThread

from video_processor import align_frames_and_logs


class AlignThread(QThread):
    """对齐处理线程"""
    progress_updated = pyqtSignal(str)  # message
    finished_signal = pyqtSignal(bool, str)  # success, message
    
    def __init__(self, frames_csv: str, logs_csv: str, output_csv: str):
        super().__init__()
        self.frames_csv = frames_csv
        self.logs_csv = logs_csv
        self.output_csv = output_csv
    
    def run(self):
        self.progress_updated.emit("正在对齐数据...")
        success, message = align_frames_and_logs(self.frames_csv, self.logs_csv, self.output_csv)
        self.finished_signal.emit(success, message)


class AlignTab(QWidget):
    """数据对齐标签页"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.align_thread = None
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # 配置组
        config_group = QGroupBox("对齐配置")
        config_layout = QGridLayout()
        
        # 帧索引 CSV
        config_layout.addWidget(QLabel("帧索引 CSV:"), 0, 0)
        self.frames_csv_edit = QLineEdit(os.path.join(os.getcwd(), 'frames_index.csv'))
        config_layout.addWidget(self.frames_csv_edit, 0, 1)
        frames_browse_btn = QPushButton("浏览...")
        frames_browse_btn.clicked.connect(self._browse_frames_csv)
        config_layout.addWidget(frames_browse_btn, 0, 2)
        
        # 日志 CSV
        config_layout.addWidget(QLabel("日志 CSV:"), 1, 0)
        self.logs_csv_edit = QLineEdit(os.path.join(os.getcwd(), 'logs.csv'))
        config_layout.addWidget(self.logs_csv_edit, 1, 1)
        logs_browse_btn = QPushButton("浏览...")
        logs_browse_btn.clicked.connect(self._browse_logs_csv)
        config_layout.addWidget(logs_browse_btn, 1, 2)
        
        # 输出 CSV
        config_layout.addWidget(QLabel("输出 CSV:"), 2, 0)
        self.output_csv_edit = QLineEdit(os.path.join(os.getcwd(), 'aligned.csv'))
        config_layout.addWidget(self.output_csv_edit, 2, 1)
        output_browse_btn = QPushButton("浏览...")
        output_browse_btn.clicked.connect(self._browse_output_csv)
        config_layout.addWidget(output_browse_btn, 2, 2)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # 说明文本
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setMaximumHeight(120)
        help_text.setHtml("""
        <h3>数据对齐说明</h3>
        <p>此功能将图像帧和日志数据按<b>主机接收时间</b>进行最近邻对齐。</p>
        <ul>
        <li>输入: frames_index.csv（帧索引）和 logs.csv（日志数据）</li>
        <li>输出: aligned.csv（对齐后的数据，包含帧信息和对应的日志）</li>
        <li>时间戳精度: 微秒级</li>
        </ul>
        """)
        layout.addWidget(help_text)
        
        # 状态文本
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(100)
        layout.addWidget(self.status_text)
        
        # 按钮
        button_layout = QHBoxLayout()
        self.align_btn = QPushButton("执行对齐")
        self.align_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; font-weight: bold; padding: 10px; }")
        self.align_btn.clicked.connect(self._align_data)
        button_layout.addWidget(self.align_btn)
        
        layout.addLayout(button_layout)
        layout.addStretch()
    
    def _browse_frames_csv(self):
        """浏览帧索引 CSV"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择帧索引 CSV", self.frames_csv_edit.text(), "CSV Files (*.csv)")
        if file_path:
            self.frames_csv_edit.setText(file_path)
    
    def _browse_logs_csv(self):
        """浏览日志 CSV"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择日志 CSV", self.logs_csv_edit.text(), "CSV Files (*.csv)")
        if file_path:
            self.logs_csv_edit.setText(file_path)
    
    def _browse_output_csv(self):
        """浏览输出 CSV"""
        file_path, _ = QFileDialog.getSaveFileName(self, "保存对齐 CSV", self.output_csv_edit.text(), "CSV Files (*.csv)")
        if file_path:
            self.output_csv_edit.setText(file_path)
    
    def _align_data(self):
        """对齐数据"""
        frames_csv = self.frames_csv_edit.text()
        logs_csv = self.logs_csv_edit.text()
        output_csv = self.output_csv_edit.text()
        
        if not os.path.exists(frames_csv):
            self.status_text.append(f"错误: 帧索引文件不存在: {frames_csv}")
            return
        
        if not os.path.exists(logs_csv):
            self.status_text.append(f"错误: 日志文件不存在: {logs_csv}")
            return
        
        # 禁用按钮
        self.align_btn.setEnabled(False)
        self.status_text.clear()
        self.status_text.append("正在对齐数据...")
        
        # 启动线程
        self.align_thread = AlignThread(frames_csv, logs_csv, output_csv)
        self.align_thread.progress_updated.connect(self._on_progress)
        self.align_thread.finished_signal.connect(self._on_finished)
        self.align_thread.start()
    
    def _on_progress(self, message: str):
        """进度更新"""
        self.status_text.append(message)
    
    def _on_finished(self, success: bool, message: str):
        """对齐完成"""
        self.status_text.append(message)
        if success:
            self.status_text.append("✓ 数据对齐成功！")
        else:
            self.status_text.append("✗ 数据对齐失败")
        
        self.align_btn.setEnabled(True)
        self.align_thread = None
