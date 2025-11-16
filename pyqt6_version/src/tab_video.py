#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tab_video.py - 视频合成标签页
"""

import os
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                              QLabel, QLineEdit, QPushButton, QSpinBox,
                              QGroupBox, QFileDialog, QProgressBar, QTextEdit)
from PyQt6.QtCore import pyqtSignal, QThread

from video_processor import compose_video_from_png


class VideoComposeThread(QThread):
    """视频合成线程"""
    progress_updated = pyqtSignal(int, str)  # progress, message
    finished_signal = pyqtSignal(bool, str)  # success, message
    
    def __init__(self, png_dir: str, output_file: str, fps: int):
        super().__init__()
        self.png_dir = png_dir
        self.output_file = output_file
        self.fps = fps
    
    def run(self):
        self.progress_updated.emit(0, "开始合成视频...")
        success, message = compose_video_from_png(self.png_dir, self.output_file, self.fps)
        self.progress_updated.emit(100, "完成" if success else "失败")
        self.finished_signal.emit(success, message)


class VideoTab(QWidget):
    """视频合成标签页"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.compose_thread = None
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # 配置组
        config_group = QGroupBox("视频合成配置")
        config_layout = QGridLayout()
        
        # PNG 目录
        config_layout.addWidget(QLabel("PNG 目录:"), 0, 0)
        self.png_dir_edit = QLineEdit(os.path.join(os.getcwd(), 'frames_png'))
        config_layout.addWidget(self.png_dir_edit, 0, 1)
        png_browse_btn = QPushButton("浏览...")
        png_browse_btn.clicked.connect(self._browse_png_dir)
        config_layout.addWidget(png_browse_btn, 0, 2)
        
        # 输出文件
        config_layout.addWidget(QLabel("输出 MP4:"), 1, 0)
        self.output_edit = QLineEdit(os.path.join(os.getcwd(), 'output.mp4'))
        config_layout.addWidget(self.output_edit, 1, 1)
        output_browse_btn = QPushButton("浏览...")
        output_browse_btn.clicked.connect(self._browse_output)
        config_layout.addWidget(output_browse_btn, 1, 2)
        
        # FPS
        config_layout.addWidget(QLabel("视频 FPS:"), 2, 0)
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 120)
        self.fps_spin.setValue(30)
        config_layout.addWidget(self.fps_spin, 2, 1)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # 状态文本
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(100)
        layout.addWidget(self.status_text)
        
        # 按钮
        button_layout = QHBoxLayout()
        self.compose_btn = QPushButton("开始合成")
        self.compose_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 10px; }")
        self.compose_btn.clicked.connect(self._compose_video)
        button_layout.addWidget(self.compose_btn)
        
        layout.addLayout(button_layout)
        layout.addStretch()
    
    def _browse_png_dir(self):
        """浏览 PNG 目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择 PNG 目录", self.png_dir_edit.text())
        if dir_path:
            self.png_dir_edit.setText(dir_path)
    
    def _browse_output(self):
        """浏览输出文件"""
        file_path, _ = QFileDialog.getSaveFileName(self, "保存视频文件", self.output_edit.text(), "MP4 Files (*.mp4)")
        if file_path:
            self.output_edit.setText(file_path)
    
    def _compose_video(self):
        """合成视频"""
        png_dir = self.png_dir_edit.text()
        output_file = self.output_edit.text()
        fps = self.fps_spin.value()
        
        if not os.path.exists(png_dir):
            self.status_text.append(f"错误: PNG 目录不存在: {png_dir}")
            return
        
        # 禁用按钮
        self.compose_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_text.clear()
        self.status_text.append(f"正在合成视频...")
        self.status_text.append(f"源目录: {png_dir}")
        self.status_text.append(f"输出文件: {output_file}")
        self.status_text.append(f"FPS: {fps}")
        
        # 启动线程
        self.compose_thread = VideoComposeThread(png_dir, output_file, fps)
        self.compose_thread.progress_updated.connect(self._on_progress)
        self.compose_thread.finished_signal.connect(self._on_finished)
        self.compose_thread.start()
    
    def _on_progress(self, progress: int, message: str):
        """进度更新"""
        self.progress_bar.setValue(progress)
        self.status_text.append(message)
    
    def _on_finished(self, success: bool, message: str):
        """合成完成"""
        self.status_text.append(message)
        if success:
            self.status_text.append("✓ 视频合成成功！")
        else:
            self.status_text.append("✗ 视频合成失败")
        
        self.compose_btn.setEnabled(True)
        self.compose_thread = None
