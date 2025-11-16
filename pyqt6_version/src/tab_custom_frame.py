#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tab_custom_frame.py - 自定义帧格式配置标签页
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                              QLabel, QLineEdit, QPushButton, QSpinBox,
                              QGroupBox, QComboBox, QCheckBox, QTextEdit,
                              QTabWidget, QFormLayout, QTableWidget, QTableWidgetItem,
                              QHeaderView, QMessageBox)
from PyQt6.QtCore import pyqtSignal

from config import CustomImageFrameConfig, CustomLogFrameConfig, IMAGE_FORMATS


class CustomFrameTab(QWidget):
    """自定义帧格式配置标签页"""
    
    # 信号
    config_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_config = CustomImageFrameConfig()
        self.log_config = CustomLogFrameConfig()
        self.log_variables = []  # [(name, byte_pos, data_type, display_format), ...]
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # 子标签页
        tab_widget = QTabWidget()
        
        # 图像帧配置
        image_tab = self._create_image_frame_config()
        tab_widget.addTab(image_tab, "图像帧配置")
        
        # 日志帧配置
        log_tab = self._create_log_frame_tab()
        tab_widget.addTab(log_tab, "日志帧配置")
        
        # 日志变量配置标签页
        log_vars_tab = self._create_log_variables_config()
        tab_widget.addTab(log_vars_tab, "日志变量配置")
        
        layout.addWidget(tab_widget)
    
    def _create_image_frame_config(self):
        """创建图像帧配置标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 启用开关
        self.image_enable_check = QCheckBox("启用自定义图像帧格式")
        layout.addWidget(self.image_enable_check)
        
        # 帧头帧尾
        frame_group = QGroupBox("帧标识")
        frame_layout = QGridLayout()
        
        frame_layout.addWidget(QLabel("帧头 (Hex):"), 0, 0)
        self.image_header_edit = QLineEdit("A0FFFFA0")
        frame_layout.addWidget(self.image_header_edit, 0, 1)
        
        frame_layout.addWidget(QLabel("帧尾 (Hex):"), 1, 0)
        self.image_footer_edit = QLineEdit("B0B00A0D")
        frame_layout.addWidget(self.image_footer_edit, 1, 1)
        
        frame_group.setLayout(frame_layout)
        layout.addWidget(frame_group)
        
        # 图像格式
        format_group = QGroupBox("图像格式")
        format_layout = QGridLayout()
        
        format_layout.addWidget(QLabel("编码格式:"), 0, 0)
        self.image_format_combo = QComboBox()
        self.image_format_combo.addItems(IMAGE_FORMATS)
        self.image_format_combo.setCurrentText("压缩二值(1位)")
        format_layout.addWidget(self.image_format_combo, 0, 1)
        
        format_group.setLayout(format_layout)
        layout.addWidget(format_group)
        
        # 尺寸模式
        size_group = QGroupBox("尺寸解析")
        size_layout = QGridLayout()
        
        size_layout.addWidget(QLabel("模式:"), 0, 0)
        self.size_mode_combo = QComboBox()
        self.size_mode_combo.addItems(["固定尺寸", "动态解析"])
        size_layout.addWidget(self.size_mode_combo, 0, 1)
        
        # 固定尺寸
        size_layout.addWidget(QLabel("固定高度:"), 1, 0)
        self.fixed_h_spin = QSpinBox()
        self.fixed_h_spin.setRange(1, 1024)
        self.fixed_h_spin.setValue(120)
        size_layout.addWidget(self.fixed_h_spin, 1, 1)
        
        size_layout.addWidget(QLabel("固定宽度:"), 2, 0)
        self.fixed_w_spin = QSpinBox()
        self.fixed_w_spin.setRange(1, 1024)
        self.fixed_w_spin.setValue(188)
        size_layout.addWidget(self.fixed_w_spin, 2, 1)
        
        size_group.setLayout(size_layout)
        layout.addWidget(size_group)
        
        # 说明
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setMaximumHeight(100)
        help_text.setHtml("""
        <b>STM32 压缩示例 (60x120):</b><br>
        帧头=A0FFFFA0, 帧尾=B0B00A0D<br>
        格式=压缩二值(1位), 固定尺寸=60x120<br>
        压缩率: 8:1 (900字节)
        """)
        layout.addWidget(help_text)
        
        layout.addStretch()
        return widget
    
    def _create_log_frame_tab(self):
        """创建日志帧配置标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 启用开关
        self.log_enable_check = QCheckBox("启用自定义日志帧格式")
        layout.addWidget(self.log_enable_check)
        
        # 帧头帧尾
        frame_group = QGroupBox("帧标识")
        frame_layout = QGridLayout()
        
        frame_layout.addWidget(QLabel("帧头 (Hex):"), 0, 0)
        self.log_header_edit = QLineEdit("BB66")
        frame_layout.addWidget(self.log_header_edit, 0, 1)
        
        frame_layout.addWidget(QLabel("帧尾 (Hex):"), 1, 0)
        self.log_footer_edit = QLineEdit("0D0A")
        frame_layout.addWidget(self.log_footer_edit, 1, 1)
        
        frame_group.setLayout(frame_layout)
        layout.addWidget(frame_group)
        
        # 数据格式
        format_group = QGroupBox("数据格式")
        format_layout = QGridLayout()
        
        format_layout.addWidget(QLabel("格式:"), 0, 0)
        self.log_format_combo = QComboBox()
        self.log_format_combo.addItems(["标准格式", "纯文本"])
        format_layout.addWidget(self.log_format_combo, 0, 1)
        
        format_group.setLayout(format_layout)
        layout.addWidget(format_group)
        
        # 说明
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setMaximumHeight(80)
        help_text.setHtml("""
        <b>标准格式:</b> [0x02][LEN][内容]<br>
        <b>纯文本:</b> 直接发送文本，无长度字段
        """)
        layout.addWidget(help_text)
        
        layout.addStretch()
        return widget
    
    def get_image_config(self) -> CustomImageFrameConfig:
        """获取图像帧配置"""
        self.image_config.enabled = self.image_enable_check.isChecked()
        self.image_config.header = self.image_header_edit.text()
        self.image_config.footer = self.image_footer_edit.text()
        self.image_config.format = self.image_format_combo.currentText()
        self.image_config.size_mode = self.size_mode_combo.currentText()
        self.image_config.fixed_h = self.fixed_h_spin.value()
        self.image_config.fixed_w = self.fixed_w_spin.value()
        return self.image_config
    
    def _create_log_variables_config(self):
        """创建日志变量配置标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 说明
        info = QLabel('配置日志变量以实时显示。支持从日志数据包中提取指定字节位置的值。')
        info.setStyleSheet('color: gray; font-size: 10px;')
        layout.addWidget(info)
        
        # 添加变量区域
        add_group = QGroupBox('添加日志变量')
        add_layout = QFormLayout(add_group)
        
        self.log_var_name = QLineEdit()
        self.log_var_name.setPlaceholderText('例: 温度、速度')
        add_layout.addRow('变量名称:', self.log_var_name)
        
        self.log_var_byte_pos = QSpinBox()
        self.log_var_byte_pos.setRange(0, 255)
        add_layout.addRow('字节位置:', self.log_var_byte_pos)
        
        self.log_var_data_type = QComboBox()
        self.log_var_data_type.addItems(['uint8', 'int8', 'uint16_le', 'uint16_be', 
                                         'int16_le', 'int16_be', 'uint32_le', 'uint32_be',
                                         'int32_le', 'int32_be', 'float_le', 'float_be'])
        add_layout.addRow('数据类型:', self.log_var_data_type)
        
        self.log_var_format = QLineEdit('{value}')
        self.log_var_format.setPlaceholderText('{value:.2f}')
        add_layout.addRow('显示格式:', self.log_var_format)
        
        btn_layout = QHBoxLayout()
        add_var_btn = QPushButton('添加变量')
        add_var_btn.clicked.connect(self._add_log_variable)
        btn_layout.addWidget(add_var_btn)
        
        clear_vars_btn = QPushButton('清空所有')
        clear_vars_btn.clicked.connect(self._clear_log_variables)
        btn_layout.addWidget(clear_vars_btn)
        btn_layout.addStretch()
        
        add_layout.addRow('', btn_layout)
        layout.addWidget(add_group)
        
        # 变量列表
        list_group = QGroupBox('已配置的日志变量')
        list_layout = QVBoxLayout(list_group)
        
        self.log_vars_table = QTableWidget()
        self.log_vars_table.setColumnCount(4)
        self.log_vars_table.setHorizontalHeaderLabels(['变量名称', '字节位置', '数据类型', '显示格式'])
        self.log_vars_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.log_vars_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        list_layout.addWidget(self.log_vars_table)
        
        remove_btn = QPushButton('删除选中')
        remove_btn.clicked.connect(self._remove_log_variable)
        list_layout.addWidget(remove_btn)
        
        layout.addWidget(list_group)
        layout.addStretch()
        
        return widget
    
    def _add_log_variable(self):
        """添加日志变量"""
        name = self.log_var_name.text().strip()
        if not name:
            QMessageBox.warning(self, '错误', '请输入变量名称')
            return
        
        # 检查是否已存在
        for var in self.log_variables:
            if var[0] == name:
                QMessageBox.warning(self, '错误', '变量名称已存在')
                return
        
        byte_pos = self.log_var_byte_pos.value()
        data_type = self.log_var_data_type.currentText()
        display_format = self.log_var_format.text().strip() or '{value}'
        
        self.log_variables.append((name, byte_pos, data_type, display_format))
        self._update_log_vars_table()
        
        # 清空输入
        self.log_var_name.clear()
    
    def _remove_log_variable(self):
        """删除选中的日志变量"""
        current_row = self.log_vars_table.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, '提示', '请先选择要删除的变量')
            return
        
        del self.log_variables[current_row]
        self._update_log_vars_table()
    
    def _clear_log_variables(self):
        """清空所有日志变量"""
        if self.log_variables:
            reply = QMessageBox.question(self, '确认', '确定要清空所有日志变量吗？',
                                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.log_variables.clear()
                self._update_log_vars_table()
    
    def _update_log_vars_table(self):
        """更新日志变量表格"""
        self.log_vars_table.setRowCount(len(self.log_variables))
        for i, (name, byte_pos, data_type, display_format) in enumerate(self.log_variables):
            self.log_vars_table.setItem(i, 0, QTableWidgetItem(name))
            self.log_vars_table.setItem(i, 1, QTableWidgetItem(str(byte_pos)))
            self.log_vars_table.setItem(i, 2, QTableWidgetItem(data_type))
            self.log_vars_table.setItem(i, 3, QTableWidgetItem(display_format))
    
    def get_log_variables(self):
        """获取日志变量配置列表"""
        return self.log_variables
    
    def get_log_config(self) -> CustomLogFrameConfig:
        """获取日志帧配置"""
        self.log_config.enabled = self.log_enable_check.isChecked()
        self.log_config.header = self.log_header_edit.text()
        self.log_config.footer = self.log_footer_edit.text()
        self.log_config.format = self.log_format_combo.currentText()
        return self.log_config
