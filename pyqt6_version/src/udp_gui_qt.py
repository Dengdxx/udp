#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
udp_gui_qt.py - PyQt6 主窗口
"""

import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                              QTabWidget, QLabel, QTextEdit, QSplitter,
                              QMessageBox, QGroupBox, QPushButton, QSpinBox, QComboBox,
                              QMenuBar, QMenu)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QPixmap, QAction
import struct
from collections import deque

import numpy as np

from tab_run import RunTab
from tab_video import VideoTab
from tab_align import AlignTab
from tab_send import SendTab
from tab_custom_frame import CustomFrameTab
from tab_scope import ScopeTab
from udp_receiver_qt import UdpReceiverThread
from utils import numpy_to_qpixmap
from config import UdpConfig


class MainWindow(QMainWindow):
    """主窗口"""
    
    def __init__(self):
        super().__init__()
        self.receiver_thread = None
        self.current_frame = None
        self.stats = {}
        self.current_theme = "dark"  # dark 或 light
        
        self._create_menu()  # 先创建菜单栏
        self._init_ui()
        self._setup_connections()
    
    def _init_ui(self):
        """初始化 UI"""
        self.setWindowTitle("UDP 上位机 - PyQt6 版本")
        self.setGeometry(100, 100, 1600, 1000)
        
        # 设置无边框窗口（隐藏系统标题栏）
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        
        # 启用右键上下文菜单
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        
        # 添加状态栏
        self.statusBar().showMessage("就绪 | 右键显示快捷菜单 | 双击标题栏最大化")
        
        # 窗口拖动相关
        self.dragging = False
        self.drag_position = None
        
        # 中央部件（包含自定义标题栏）
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 自定义标题栏
        self.title_bar = self._create_title_bar()
        main_layout.addWidget(self.title_bar)
        
        # 内容分割器
        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 左侧：控制面板
        left_panel = self._create_left_panel()
        content_splitter.addWidget(left_panel)
        
        # 右侧：视频显示
        right_panel = self._create_right_panel()
        content_splitter.addWidget(right_panel)
        
        # 设置分割比例
        content_splitter.setStretchFactor(0, 1)
        content_splitter.setStretchFactor(1, 2)
        
        main_layout.addWidget(content_splitter)
    
    def _create_title_bar(self):
        """创建自定义标题栏"""
        title_bar = QWidget()
        title_bar.setFixedHeight(45)
        title_bar.setObjectName("titleBar")  # 设置对象名以便主题切换时更新样式
        
        layout = QHBoxLayout(title_bar)
        layout.setContentsMargins(15, 0, 5, 0)
        layout.setSpacing(5)
        
        # 应用图标和标题
        title_label = QLabel("UDP 上位机 - PyQt6")
        title_label.setObjectName("titleLabel")
        layout.addWidget(title_label)
        
        layout.addStretch()
        
        # 主题切换按钮
        theme_btn = QPushButton("🌓")
        theme_btn.setFixedSize(45, 35)
        theme_btn.setToolTip("切换主题")
        theme_btn.setObjectName("themeBtn")
        theme_btn.clicked.connect(self._toggle_theme)
        layout.addWidget(theme_btn)
        
        # 最小化按钮
        min_btn = QPushButton("─")
        min_btn.setFixedSize(45, 35)
        min_btn.setToolTip("最小化")
        min_btn.setObjectName("minBtn")
        min_btn.clicked.connect(self.showMinimized)
        layout.addWidget(min_btn)
        
        # 最大化/还原按钮
        self.max_btn = QPushButton("□")
        self.max_btn.setFixedSize(45, 35)
        self.max_btn.setToolTip("最大化")
        self.max_btn.setObjectName("maxBtn")
        self.max_btn.clicked.connect(self._toggle_maximize)
        layout.addWidget(self.max_btn)
        
        # 关闭按钮
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(45, 35)
        close_btn.setToolTip("关闭")
        close_btn.setObjectName("closeBtn")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
        
        # 启用拖动窗口
        title_bar.mousePressEvent = self._title_bar_mouse_press
        title_bar.mouseMoveEvent = self._title_bar_mouse_move
        title_bar.mouseDoubleClickEvent = lambda e: self._toggle_maximize()
        
        return title_bar
    
    def _toggle_maximize(self):
        """切换最大化/还原"""
        if self.isMaximized():
            self.showNormal()
            self.max_btn.setText("□")
            self.max_btn.setToolTip("最大化")
        else:
            self.showMaximized()
            self.max_btn.setText("❐")
            self.max_btn.setToolTip("还原")
    
    def _title_bar_mouse_press(self, event):
        """标题栏鼠标按下"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
    
    def _title_bar_mouse_move(self, event):
        """标题栏鼠标移动（拖动窗口）"""
        if event.buttons() == Qt.MouseButton.LeftButton and hasattr(self, 'drag_position'):
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()
    
    def _create_left_panel(self):
        """创建左侧控制面板"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 标签页
        self.tab_widget = QTabWidget()
        
        # 运行标签页
        self.run_tab = RunTab()
        self.tab_widget.addTab(self.run_tab, "运行")
        
        # 视频标签页
        self.video_tab = VideoTab()
        self.tab_widget.addTab(self.video_tab, "视频")
        
        # 对齐标签页
        self.align_tab = AlignTab()
        self.tab_widget.addTab(self.align_tab, "对齐")
        
        # 示波器标签页
        self.scope_tab = ScopeTab()
        self.tab_widget.addTab(self.scope_tab, "示波")
        
        # 发送标签页
        self.send_tab = SendTab()
        self.tab_widget.addTab(self.send_tab, "发送")
        
        # 自定义帧标签页
        self.custom_frame_tab = CustomFrameTab()
        self.tab_widget.addTab(self.custom_frame_tab, "自定义帧")
        
        layout.addWidget(self.tab_widget)
        
        # 输出日志
        log_label = QLabel("输出日志:")
        layout.addWidget(log_label)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        layout.addWidget(self.log_text)
        
        return widget
    
    def _create_right_panel(self):
        """创建右侧视频显示面板"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(6)
        
        # 视频显示组
        video_group = QGroupBox("实时视频")
        video_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 12px; }")
        video_layout = QVBoxLayout(video_group)
        
        # 视频画布
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("QLabel { background-color: #1e1e1e; color: #888888; font-size: 14px; }")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setText("等待视频流...\n请点击 \"启动监听\" 开始接收")
        video_layout.addWidget(self.video_label)
        
        # 统计信息
        self.stats_label = QLabel("FPS: 0.0 | 帧数: 0 | 总包: 0 | 错误: 0")
        self.stats_label.setStyleSheet("QLabel { font-size: 10px; font-family: 'Consolas'; padding: 4px; background-color: #2d2d2d; color: #00ff00; }")
        video_layout.addWidget(self.stats_label)
        
        layout.addWidget(video_group, 2)
        
        # 实时日志显示组
        log_group = QGroupBox("实时日志显示")
        log_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 12px; }")
        log_layout = QVBoxLayout(log_group)
        
        self.log_display_widget = QWidget()
        self.log_display_widget.setMinimumHeight(80)
        self.log_display_widget.setMaximumHeight(120)
        self.log_display_widget.setStyleSheet("QWidget { background-color: #1e1e1e; border: 1px solid #3c3c3c; }")
        self.log_display_layout = QHBoxLayout(self.log_display_widget)
        self.log_display_layout.setContentsMargins(8, 4, 8, 4)
        self.log_display_layout.setSpacing(12)
        
        self.log_value_labels = {}  # {var_name: QLabel}
        
        # 默认提示
        self.log_empty_label = QLabel("请在 自定义帧→日志变量配置 中添加变量")
        self.log_empty_label.setStyleSheet("color: #666666; font-size: 10px;")
        self.log_display_layout.addWidget(self.log_empty_label)
        self.log_display_layout.addStretch()
        
        log_layout.addWidget(self.log_display_widget)
        layout.addWidget(log_group)
        
        # 原始数据监视器组
        data_group = QGroupBox("原始数据监视器")
        data_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 12px; }")
        data_layout = QVBoxLayout(data_group)
        
        # 工具栏
        data_toolbar = QHBoxLayout()
        
        clear_data_btn = QPushButton("清空")
        clear_data_btn.clicked.connect(self._clear_data_display)
        clear_data_btn.setMaximumWidth(60)
        data_toolbar.addWidget(clear_data_btn)
        
        refresh_data_btn = QPushButton("刷新")
        refresh_data_btn.clicked.connect(self._refresh_data_display)
        refresh_data_btn.setMaximumWidth(60)
        data_toolbar.addWidget(refresh_data_btn)
        
        data_toolbar.addWidget(QLabel("最大显示:"))
        self.data_display_limit = QSpinBox()
        self.data_display_limit.setRange(10, 100)
        self.data_display_limit.setValue(20)
        self.data_display_limit.setMaximumWidth(70)
        data_toolbar.addWidget(self.data_display_limit)
        
        data_toolbar.addWidget(QLabel("编码:"))
        self.data_encoding = QComboBox()
        self.data_encoding.addItems(['UTF-8', 'GBK', 'GB2312', 'ASCII', 'Latin-1', 'UTF-16', 'UTF-32', 'Big5'])
        self.data_encoding.setMaximumWidth(100)
        self.data_encoding.currentTextChanged.connect(self._refresh_data_display)
        data_toolbar.addWidget(self.data_encoding)
        
        data_toolbar.addWidget(QLabel("格式:"))
        self.data_format = QComboBox()
        self.data_format.addItems(['详细', '简洁', '仅Hex', '仅文本'])
        self.data_format.setMaximumWidth(80)
        self.data_format.currentTextChanged.connect(self._refresh_data_display)
        data_toolbar.addWidget(self.data_format)
        
        data_toolbar.addStretch()
        data_layout.addLayout(data_toolbar)
        
        self.data_text = QTextEdit()
        self.data_text.setReadOnly(True)
        self.data_text.setMaximumHeight(150)
        self.data_text.setStyleSheet("""
            QTextEdit { 
                font-family: 'Consolas', monospace; 
                font-size: 9px; 
                background-color: #1e1e1e; 
                color: #d4d4d4;
                border: 1px solid #3c3c3c;
            }
        """)
        data_layout.addWidget(self.data_text)
        
        layout.addWidget(data_group, 1)
        
        # 初始化数据缓存
        self.recent_data = []  # [(timestamp, ftype, data_hex, info), ...]
        
        return widget
    
    def _setup_connections(self):
        """设置信号连接"""
        # 运行标签页信号
        self.run_tab.start_requested.connect(self._on_start_receiver)
        self.run_tab.stop_requested.connect(self._on_stop_receiver)
        
        # 定时器：更新视频显示
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self._update_video_display)
        self.video_timer.setInterval(33)  # 约 30 FPS
    
    @pyqtSlot(UdpConfig)
    def _on_start_receiver(self, config: UdpConfig):
        """启动接收器"""
        if self.receiver_thread and self.receiver_thread.isRunning():
            self.log("接收器已在运行中")
            return
        
        self.log(f"正在启动 UDP 接收器...")
        self.log(f"绑定地址: {config.ip}:{config.port}")
        self.log(f"保存 PNG: {config.save_png}")
        
        # 获取自定义帧配置
        image_config = self.custom_frame_tab.get_image_config()
        log_config = self.custom_frame_tab.get_log_config()
        
        # 创建接收线程
        self.receiver_thread = UdpReceiverThread(
            ip=config.ip,
            port=config.port,
            save_png=config.save_png,
            png_dir=config.png_dir,
            log_csv=config.log_csv,
            frame_index_csv=config.frame_index_csv,
            enable_custom_image_frame=image_config.enabled,
            image_frame_header=image_config.header,
            image_frame_footer=image_config.footer,
            image_fixed_h=image_config.fixed_h,
            image_fixed_w=image_config.fixed_w,
            image_format=image_config.format,
            enable_custom_log_frame=log_config.enabled,
            log_frame_header=log_config.header,
            log_frame_footer=log_config.footer,
            log_frame_format=log_config.format
        )
        
        # 连接信号
        self.receiver_thread.frame_received.connect(self._on_frame_received)
        self.receiver_thread.log_received.connect(self._on_log_received)
        self.receiver_thread.stats_updated.connect(self._on_stats_updated)
        self.receiver_thread.data_packet_received.connect(self._on_data_packet)
        self.receiver_thread.scope_data_received.connect(self._on_scope_data)
        self.receiver_thread.error_occurred.connect(self._on_error)
        self.receiver_thread.finished.connect(self._on_receiver_finished)
        
        # 初始化日志显示（同步日志变量列表）
        self._update_log_display_vars()
        
        # 更新日志显示变量
        self._update_log_display_vars()
        
        # 同步日志变量到示波器
        self.scope_tab.log_variables = self.custom_frame_tab.get_log_variables()
        self.scope_tab.refresh_log_vars(silent=True)
        
        # 启动线程
        self.receiver_thread.start()
        
        # 启动视频更新定时器
        self.video_timer.start()
        
        self.log("✓ UDP 接收器已启动")
    
    @pyqtSlot()
    def _on_stop_receiver(self):
        """停止接收器"""
        if not self.receiver_thread or not self.receiver_thread.isRunning():
            self.log("接收器未运行")
            return
        
        self.log("正在停止 UDP 接收器...")
        
        # 停止线程
        self.receiver_thread.stop()
        self.receiver_thread.wait(3000)  # 等待最多 3 秒
        
        # 停止视频更新
        self.video_timer.stop()
        
        self.log("✓ UDP 接收器已停止")
    
    @pyqtSlot(np.ndarray, int, int, int)
    def _on_frame_received(self, image: np.ndarray, frame_id: int, h: int, w: int):
        """接收到新帧"""
        self.current_frame = image
    
    @pyqtSlot(dict)
    def _on_stats_updated(self, stats: dict):
        """统计信息更新"""
        self.stats = stats
        fps = stats.get('fps', 0.0)
        frame_counter = stats.get('frame_counter', 0)
        total_packets = stats.get('total_packets', 0)
        error_packets = stats.get('error_packets', 0)
        
        self.stats_label.setText(
            f"FPS: {fps:.1f} | 帧数: {frame_counter} | 总包: {total_packets} | 错误: {error_packets}"
        )
    
    @pyqtSlot(str, str, str, str)
    def _on_data_packet(self, timestamp: str, pkt_type: str, data_hex: str, info: str):
        """接收到数据包"""
        # 保存到缓存
        self.recent_data.append((timestamp, pkt_type, data_hex, info))
        
        # 限制缓存大小
        max_size = 200
        if len(self.recent_data) > max_size:
            self.recent_data = self.recent_data[-max_size:]
        
        # 更新显示
        self._update_data_display()
    
    def _update_data_display(self):
        """更新原始数据显示"""
        # 获取显示限制
        limit = self.data_display_limit.value()
        encoding = self.data_encoding.currentText().lower().replace('-', '')
        display_format = self.data_format.currentText()
        
        # 获取最近的数据
        data_list = self.recent_data[-limit:] if len(self.recent_data) > limit else self.recent_data
        
        # 清空并重新显示
        self.data_text.clear()
        
        for timestamp, ftype, data_hex, info in data_list:
            if display_format == '详细':
                # 详细模式
                line = f"[{timestamp}] {ftype}\n"
                line += f"  Info: {info}\n"
                
                # 尝试解码
                try:
                    if encoding == 'utf8':
                        encoding = 'utf-8'
                    elif encoding == 'gb2312':
                        encoding = 'gb2312'
                    elif encoding == 'gbk':
                        encoding = 'gbk'
                    
                    data_bytes = bytes.fromhex(data_hex.replace('...', ''))
                    decoded_text = data_bytes.decode(encoding, errors='replace')
                    decoded_text = ''.join(c if c.isprintable() or c in '\n\r\t' else '·' for c in decoded_text)
                    line += f"  {encoding.upper()}: {decoded_text[:100]}" + ('...' if len(decoded_text) > 100 else '') + "\n"
                except:
                    line += f"  {encoding.upper()}: <decode error>\n"
                
                line += f"  Hex: {data_hex}\n"
                line += "-" * 80 + "\n"
                
            elif display_format == '简洁':
                # 简洁模式
                line = f"[{timestamp[-15:]}] {ftype:12s} | {info[:60]}\n"
                
            elif display_format == '仅Hex':
                # 仅Hex
                line = f"[{timestamp[-15:]}] {ftype:12s}\n"
                line += f"  {data_hex}\n"
                
            elif display_format == '仅文本':
                # 仅文本
                line = f"[{timestamp[-15:]}] {ftype:12s}\n"
                try:
                    if encoding == 'utf8':
                        encoding = 'utf-8'
                    data_bytes = bytes.fromhex(data_hex.replace('...', ''))
                    decoded_text = data_bytes.decode(encoding, errors='replace')
                    decoded_text = ''.join(c if c.isprintable() or c in '\n\r\t' else '·' for c in decoded_text)
                    line += f"  {decoded_text}\n"
                except:
                    line += f"  <decode error>\n"
            else:
                line = f"{timestamp} {ftype} {info}\n"
            
            self.data_text.insertPlainText(line)
        
        # 滚动到底部
        self.data_text.verticalScrollBar().setValue(
            self.data_text.verticalScrollBar().maximum()
        )
    
    @pyqtSlot(int, int, float)
    def _on_scope_data(self, byte_idx: int, value: int, timestamp: float):
        """接收到示波器数据"""
        # 转发给示波器标签页
        self.scope_tab.add_data_point(byte_idx, value, timestamp)
    
    @pyqtSlot(bytes)
    def _on_log_received(self, payload: bytes):
        """接收到日志数据"""
        self._update_log_display(payload)
    
    def _update_log_display_vars(self):
        """更新日志显示变量（在启动接收器时调用）"""
        # 清空现有标签
        for label in self.log_value_labels.values():
            label.deleteLater()
        self.log_value_labels.clear()
        
        # 获取日志变量列表
        log_vars = self.custom_frame_tab.get_log_variables()
        
        if not log_vars:
            if self.log_empty_label:
                self.log_empty_label.show()
            return
        
        if self.log_empty_label:
            self.log_empty_label.hide()
        
        # 创建新标签
        for var_name, byte_pos, data_type, display_format in log_vars:
            label = QLabel(f"{var_name}: --")
            label.setStyleSheet("QLabel { color: #00d4ff; font-size: 11px; font-family: 'Consolas'; font-weight: bold; }")
            self.log_display_layout.addWidget(label)
            self.log_value_labels[var_name] = label
        
        self.log_display_layout.addStretch()
    
    def _update_log_display(self, log_data: bytes):
        """更新日志显示"""
        log_vars = self.custom_frame_tab.get_log_variables()
        
        for var_name, byte_pos, data_type, display_format in log_vars:
            if var_name not in self.log_value_labels:
                continue
            
            try:
                value = self._parse_log_value(log_data, byte_pos, data_type)
                if value is not None:
                    try:
                        display_text = display_format.format(value=value)
                    except:
                        display_text = str(value)
                    self.log_value_labels[var_name].setText(f"{var_name}: {display_text}")
                else:
                    self.log_value_labels[var_name].setText(f"{var_name}: --")
            except:
                self.log_value_labels[var_name].setText(f"{var_name}: ERR")
    
    def _parse_log_value(self, data: bytes, byte_pos: int, data_type: str):
        """解析日志值"""
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
        except:
            return None
    
    @pyqtSlot(str)
    def _on_error(self, message: str):
        """错误发生"""
        self.log(f"错误: {message}")
    
    @pyqtSlot()
    def _on_receiver_finished(self):
        """接收器线程结束"""
        self.log("接收器线程已结束")
        self.receiver_thread = None
    
    def _create_menu(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 视图菜单
        view_menu = menubar.addMenu("视图")
        
        # 主题切换
        theme_action = QAction("切换主题 (暗色/亮色)", self)
        theme_action.triggered.connect(self._toggle_theme)
        view_menu.addAction(theme_action)
        
        # 全屏模式
        fullscreen_action = QAction("全屏模式 (F11)", self)
        fullscreen_action.setShortcut("F11")
        fullscreen_action.triggered.connect(self._toggle_fullscreen)
        view_menu.addAction(fullscreen_action)
        
        # 隐藏菜单栏
        hide_menu_action = QAction("隐藏菜单栏 (Ctrl+M)", self)
        hide_menu_action.setShortcut("Ctrl+M")
        hide_menu_action.triggered.connect(self._toggle_menubar)
        view_menu.addAction(hide_menu_action)
        
        # 延迟隐藏菜单栏（确保窗口初始化完成后）
        QTimer.singleShot(100, lambda: menubar.hide())
    
    def _toggle_theme(self):
        """切换主题"""
        if self.current_theme == "dark":
            self.current_theme = "light"
            self._apply_light_theme()
        else:
            self.current_theme = "dark"
            self._apply_dark_theme()
    
    def _toggle_fullscreen(self):
        """切换全屏"""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
    
    def _toggle_menubar(self):
        """切换菜单栏显示"""
        menubar = self.menuBar()
        if menubar.isVisible():
            menubar.hide()
            self.statusBar().showMessage("菜单栏已隐藏 | 按 Ctrl+M 或右键显示")
            self.log("菜单栏已隐藏")
        else:
            menubar.show()
            self.statusBar().showMessage("菜单栏已显示 | 按 Ctrl+M 隐藏")
            self.log("菜单栏已显示")
    
    def _show_context_menu(self, pos):
        """显示右键上下文菜单"""
        menu = QMenu(self)
        
        # 主题切换
        theme_text = "切换到亮色主题" if self.current_theme == "dark" else "切换到深色主题"
        theme_action = menu.addAction(theme_text)
        theme_action.triggered.connect(self._toggle_theme)
        
        # 菜单栏
        menubar_text = "隐藏菜单栏" if self.menuBar().isVisible() else "显示菜单栏"
        menubar_action = menu.addAction(menubar_text)
        menubar_action.triggered.connect(self._toggle_menubar)
        
        # 全屏
        fullscreen_text = "退出全屏" if self.isFullScreen() else "进入全屏"
        fullscreen_action = menu.addAction(fullscreen_text)
        fullscreen_action.triggered.connect(self._toggle_fullscreen)
        
        menu.exec(self.mapToGlobal(pos))
    
    def _apply_dark_theme(self):
        """应用深色主题"""
        app = QApplication.instance()
        app.setStyleSheet(get_dark_stylesheet())
        
        # 更新自定义标题栏样式（深色主题）
        self.title_bar.setStyleSheet("""
            QWidget#titleBar {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2d2d30, stop:1 #1e1e1e);
                border-bottom: 1px solid #3c3c3c;
            }
            QLabel#titleLabel {
                color: #e0e0e0;
                font-size: 13px;
                font-weight: 600;
                padding: 0px;
            }
            QPushButton#themeBtn, QPushButton#minBtn, QPushButton#maxBtn {
                background-color: transparent;
                color: #cccccc;
                border: none;
                border-radius: 3px;
                font-size: 16px;
            }
            QPushButton#themeBtn:hover, QPushButton#minBtn:hover, QPushButton#maxBtn:hover {
                background-color: rgba(255, 255, 255, 0.08);
                color: #ffffff;
            }
            QPushButton#themeBtn:pressed, QPushButton#minBtn:pressed, QPushButton#maxBtn:pressed {
                background-color: rgba(255, 255, 255, 0.05);
            }
            QPushButton#closeBtn {
                background-color: transparent;
                color: #cccccc;
                border: none;
                border-radius: 3px;
                font-size: 16px;
            }
            QPushButton#closeBtn:hover {
                background-color: #e81123;
                color: #ffffff;
            }
            QPushButton#closeBtn:pressed {
                background-color: #c50f1f;
            }
        """)
        
        # 更新特定组件的样式（深色主题）
        self.video_label.setStyleSheet("QLabel { background-color: #1e1e1e; color: #888888; font-size: 14px; }")
        self.stats_label.setStyleSheet("QLabel { font-size: 10px; font-family: 'Consolas'; padding: 4px; background-color: #2d2d2d; color: #00ff00; }")
        self.log_display_widget.setStyleSheet("QWidget { background-color: #1e1e1e; border: 1px solid #3c3c3c; }")
        
        # 更新日志标签颜色
        for label in self.log_value_labels.values():
            label.setStyleSheet("QLabel { color: #00d4ff; font-size: 11px; font-family: 'Consolas'; font-weight: bold; }")
        
        self.data_text.setStyleSheet("""
            QTextEdit { 
                font-family: 'Consolas', monospace; 
                font-size: 9px; 
                background-color: #1e1e1e; 
                color: #d4d4d4;
                border: 1px solid #3c3c3c;
            }
        """)
        
        self.statusBar().showMessage("已切换到深色主题")
        self.log("已切换到深色主题")
        # 强制刷新
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()
    
    def _get_dark_stylesheet(self):
        """获取深色主题样式表"""
        return get_dark_stylesheet()
    
    def _apply_light_theme(self):
        """应用亮色主题"""
        app = QApplication.instance()
        app.setStyleSheet(self._get_light_stylesheet())
        
        # 更新自定义标题栏样式（亮色主题）
        self.title_bar.setStyleSheet("""
            QWidget#titleBar {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ffffff, stop:1 #f0f0f0);
                border-bottom: 1px solid #d0d0d0;
            }
            QLabel#titleLabel {
                color: #333333;
                font-size: 13px;
                font-weight: 600;
                padding: 0px;
            }
            QPushButton#themeBtn, QPushButton#minBtn, QPushButton#maxBtn {
                background-color: transparent;
                color: #5a5a5a;
                border: none;
                border-radius: 3px;
                font-size: 16px;
            }
            QPushButton#themeBtn:hover, QPushButton#minBtn:hover, QPushButton#maxBtn:hover {
                background-color: rgba(0, 0, 0, 0.06);
                color: #1a1a1a;
            }
            QPushButton#themeBtn:pressed, QPushButton#minBtn:pressed, QPushButton#maxBtn:pressed {
                background-color: rgba(0, 0, 0, 0.12);
            }
            QPushButton#closeBtn {
                background-color: transparent;
                color: #5a5a5a;
                border: none;
                border-radius: 3px;
                font-size: 16px;
            }
            QPushButton#closeBtn:hover {
                background-color: #e81123;
                color: #ffffff;
            }
            QPushButton#closeBtn:pressed {
                background-color: #c50f1f;
            }
        """)
        
        # 更新特定组件的样式（亮色主题）
        self.video_label.setStyleSheet("QLabel { background-color: #e8e8e8; color: #666666; font-size: 14px; }")
        self.stats_label.setStyleSheet("QLabel { font-size: 10px; font-family: 'Consolas'; padding: 4px; background-color: #e0f7fa; color: #006064; }")
        self.log_display_widget.setStyleSheet("QWidget { background-color: #f9f9f9; border: 1px solid #d0d0d0; }")
        
        # 更新日志标签颜色
        for label in self.log_value_labels.values():
            label.setStyleSheet("QLabel { color: #0078d4; font-size: 11px; font-family: 'Consolas'; font-weight: bold; }")
        
        self.data_text.setStyleSheet("""
            QTextEdit { 
                font-family: 'Consolas', monospace; 
                font-size: 9px; 
                background-color: white; 
                color: #333333;
                border: 1px solid #d0d0d0;
            }
        """)
        
        self.statusBar().showMessage("已切换到亮色主题")
        self.log("已切换到亮色主题")
        # 强制刷新
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()
    
    def _get_light_stylesheet(self):
        """亮色主题样式表"""
        return """
            QMainWindow {
                background-color: #f5f5f5;
            }
            QWidget {
                background-color: #f5f5f5;
                color: #333333;
            }
            QTabWidget::pane {
                border: 1px solid #d0d0d0;
                background-color: #ffffff;
            }
            QTabBar::tab {
                background-color: #e8e8e8;
                color: #333333;
                padding: 8px 16px;
                border: 1px solid #d0d0d0;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #0078d4;
                color: white;
            }
            QTabBar::tab:hover {
                background-color: #d8d8d8;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1084d8;
            }
            QPushButton:pressed {
                background-color: #006cbe;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #888888;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: white;
                color: #333333;
                border: 1px solid #cccccc;
                padding: 4px;
                border-radius: 2px;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border: 1px solid #0078d4;
            }
            QTextEdit {
                background-color: white;
                color: #333333;
                border: 1px solid #cccccc;
                font-family: 'Consolas', monospace;
            }
            QLabel {
                color: #333333;
                background-color: transparent;
            }
            QGroupBox {
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
                font-weight: bold;
                color: #333333;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
                color: #333333;
            }
            QCheckBox {
                color: #333333;
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 1px solid #cccccc;
                border-radius: 3px;
                background-color: white;
            }
            QCheckBox::indicator:checked {
                background-color: #0078d4;
                border-color: #0078d4;
            }
            QTableWidget, QListWidget {
                background-color: white;
                color: #333333;
                border: 1px solid #cccccc;
                gridline-color: #e0e0e0;
            }
            QTableWidget::item, QListWidget::item {
                color: #333333;
            }
            QTableWidget::item:selected, QListWidget::item:selected {
                background-color: #0078d4;
                color: white;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                color: #333333;
                padding: 5px;
                border: 1px solid #d0d0d0;
            }
            QScrollBar:vertical {
                background-color: #f5f5f5;
                width: 12px;
            }
            QScrollBar::handle:vertical {
                background-color: #c0c0c0;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #a0a0a0;
            }
            QScrollBar:horizontal {
                background-color: #f5f5f5;
                height: 12px;
            }
            QScrollBar::handle:horizontal {
                background-color: #c0c0c0;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: #a0a0a0;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #333333;
                margin-right: 5px;
            }
            QSpinBox::up-button, QDoubleSpinBox::up-button,
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                background-color: #e8e8e8;
                border: 1px solid #cccccc;
            }
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
                background-color: #d8d8d8;
            }
        """
    
    def _clear_data_display(self):
        """清空数据显示"""
        self.recent_data.clear()
        self.data_text.clear()
        self.log("已清空原始数据显示")
    
    def _refresh_data_display(self):
        """刷新数据显示"""
        self._update_data_display()
        self.log("已刷新原始数据显示")
    
    def _update_video_display(self):
        """更新视频显示"""
        if self.current_frame is None:
            return
        
        # 获取显示区域大小
        label_width = self.video_label.width()
        label_height = self.video_label.height()
        
        # 转换为 QPixmap 并缩放
        pixmap = numpy_to_qpixmap(self.current_frame, label_width, label_height)
        
        if not pixmap.isNull():
            self.video_label.setPixmap(pixmap)
            self.video_label.setStyleSheet("QLabel { background-color: black; }")
    
    def log(self, message: str):
        """添加日志"""
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        # 停止接收器
        if self.receiver_thread and self.receiver_thread.isRunning():
            self.receiver_thread.stop()
            self.receiver_thread.wait(3000)
        
        # 停止定时器
        if self.video_timer.isActive():
            self.video_timer.stop()
        
        event.accept()


def get_dark_stylesheet():
    """获取深色主题样式表"""
    return """
        QMainWindow {
            background-color: #2b2b2b;
        }
        QWidget {
            background-color: #2b2b2b;
            color: #cccccc;
        }
        QTabWidget::pane {
            border: 1px solid #3c3c3c;
            background-color: #2b2b2b;
        }
        QTabBar::tab {
            background-color: #3c3c3c;
            color: #cccccc;
            padding: 8px 16px;
            border: 1px solid #3c3c3c;
            border-bottom: none;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        QTabBar::tab:selected {
            background-color: #007acc;
            color: white;
        }
        QTabBar::tab:hover {
            background-color: #505050;
        }
        QPushButton {
            background-color: #0e639c;
            color: white;
            border: none;
            padding: 6px 12px;
            border-radius: 3px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #1177bb;
        }
        QPushButton:pressed {
            background-color: #0d5689;
        }
        QPushButton:disabled {
            background-color: #3c3c3c;
            color: #666666;
        }
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            background-color: #3c3c3c;
            color: #cccccc;
            border: 1px solid #555555;
            padding: 4px;
            border-radius: 2px;
        }
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
            border: 1px solid #007acc;
        }
        QTextEdit {
            background-color: #1e1e1e;
            color: #d4d4d4;
            border: 1px solid #3c3c3c;
            font-family: 'Consolas', monospace;
        }
        QGroupBox {
            border: 1px solid #3c3c3c;
            border-radius: 4px;
            margin-top: 8px;
            padding-top: 8px;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 4px;
        }
        QCheckBox {
            color: #cccccc;
            spacing: 6px;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border: 1px solid #555555;
            border-radius: 3px;
            background-color: #3c3c3c;
        }
        QCheckBox::indicator:checked {
            background-color: #007acc;
            border-color: #007acc;
        }
        QListWidget {
            background-color: #1e1e1e;
            color: #cccccc;
            border: 1px solid #3c3c3c;
        }
        QTableWidget {
            background-color: #1e1e1e;
            color: #cccccc;
            gridline-color: #3c3c3c;
            border: 1px solid #3c3c3c;
        }
        QTableWidget::item {
            padding: 4px;
        }
        QTableWidget::item:selected {
            background-color: #007acc;
        }
        QHeaderView::section {
            background-color: #2b2b2b;
            color: #cccccc;
            padding: 5px;
            border: 1px solid #3c3c3c;
        }
        QScrollBar:vertical {
            background-color: #2b2b2b;
            width: 12px;
        }
        QScrollBar::handle:vertical {
            background-color: #555555;
            border-radius: 6px;
        }
        QScrollBar::handle:vertical:hover {
            background-color: #666666;
        }
        QScrollBar:horizontal {
            background-color: #2b2b2b;
            height: 12px;
        }
        QScrollBar::handle:horizontal {
            background-color: #555555;
            border-radius: 6px;
        }
        QScrollBar::handle:horizontal:hover {
            background-color: #666666;
        }
    """

def main():
    """主函数"""
    app = QApplication(sys.argv)
    app.setApplicationName("UDP 上位机 PyQt6")
    app.setStyle("Fusion")  # 使用 Fusion 风格（跨平台一致）
    
    # 应用深色主题
    app.setStyleSheet(get_dark_stylesheet())
    
    window = MainWindow()
    window.show()  # 以正常大小显示,不最大化
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
