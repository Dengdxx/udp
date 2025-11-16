"""
示波器标签页模块
提供实时波形显示、FFT分析、位提取等功能
"""
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QSpinBox, QDoubleSpinBox, QCheckBox,
                             QComboBox, QLineEdit, QListWidget, QMessageBox,
                             QSplitter, QGroupBox, QFormLayout)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from collections import deque
import numpy as np

try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']  # 中文字体
    matplotlib.rcParams['axes.unicode_minus'] = False  # 负号显示
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class ScopeTab(QWidget):
    """示波器标签页"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        if not HAS_MATPLOTLIB:
            self._create_no_matplotlib_ui()
            return
        
        # 数据结构
        self.scope_variables = []  # [(byte_idx, bit_idx, name, color), ...]
        self.scope_data = {}  # {(byte_idx, bit_idx): deque([(timestamp, value), ...], maxlen=10000)}
        self.scope_start_time = 0
        
        # FFT数据
        self.fft_active = False
        self.fft_data = {}  # {var_key: (freqs, magnitudes)}
        
        # 颜色列表
        self.scope_colors = ['#FF6B6B', '#4ECDC4', '#FFD93D', '#A66FFF', 
                            '#6BCF7F', '#FF9F43', '#4A90E2', '#FF6FA3']
        self.scope_color_idx = 0
        
        # 日志变量列表（外部设置）
        self.log_variables = []
        
        self._create_ui()
        self._start_update_timer()
    
    def _create_no_matplotlib_ui(self):
        """无matplotlib时显示提示"""
        layout = QVBoxLayout(self)
        label = QLabel('示波器不可用：需要安装 matplotlib\n\npip install matplotlib')
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont('Arial', 12)
        label.setFont(font)
        layout.addWidget(label)
    
    def _create_ui(self):
        """创建UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        
        # 垂直分割器：图表区 + 控制区
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # 图表区域
        chart_group = QGroupBox('实时波形')
        chart_layout = QVBoxLayout(chart_group)
        
        # 创建matplotlib图表
        self.fig = Figure(figsize=(8, 5), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        chart_layout.addWidget(self.canvas)
        
        # 初始化为单图模式
        self.ax_time = None
        self.ax_freq = None
        self._create_single_plot()
        
        splitter.addWidget(chart_group)
        
        # 控制区域
        control_group = QGroupBox('示波器控制')
        control_layout = QVBoxLayout(control_group)
        
        # 第一行：变量管理
        row1 = QHBoxLayout()
        row1.addWidget(QLabel('选择变量:'))
        self.log_var_combo = QComboBox()
        self.log_var_combo.setMinimumWidth(200)
        row1.addWidget(self.log_var_combo)
        
        row1.addWidget(QLabel('Bit:'))
        self.bit_entry = QLineEdit()
        self.bit_entry.setMaximumWidth(80)
        self.bit_entry.setPlaceholderText('如: 3 或 2:5')
        row1.addWidget(self.bit_entry)
        
        bit_help_label = QLabel('(可选: 3 或 2:5 或 0:8:2)')
        bit_help_label.setStyleSheet('color: gray; font-size: 9pt;')
        row1.addWidget(bit_help_label)
        
        add_btn = QPushButton('添加')
        add_btn.clicked.connect(self._add_variable)
        row1.addWidget(add_btn)
        
        remove_btn = QPushButton('删除')
        remove_btn.clicked.connect(self._remove_variable)
        row1.addWidget(remove_btn)
        
        clear_btn = QPushButton('清空')
        clear_btn.clicked.connect(self._clear_variables)
        row1.addWidget(clear_btn)
        
        refresh_btn = QPushButton('刷新列表')
        refresh_btn.clicked.connect(self.refresh_log_vars)
        row1.addWidget(refresh_btn)
        
        help_btn = QPushButton('?')
        help_btn.setMaximumWidth(30)
        help_btn.clicked.connect(self._show_bit_help)
        row1.addWidget(help_btn)
        
        row1.addStretch()
        control_layout.addLayout(row1)
        
        # 第二行：监控变量列表
        row2 = QVBoxLayout()
        row2.addWidget(QLabel('监控变量:'))
        self.var_listbox = QListWidget()
        self.var_listbox.setMaximumHeight(80)
        self.var_listbox.setFont(QFont('Consolas', 10))
        row2.addWidget(self.var_listbox)
        control_layout.addLayout(row2)
        
        # 第三行：显示设置
        row3 = QHBoxLayout()
        row3.addWidget(QLabel('时间窗口:'))
        self.time_window = QDoubleSpinBox()
        self.time_window.setRange(1, 60)
        self.time_window.setValue(10.0)
        self.time_window.setSuffix(' 秒')
        row3.addWidget(self.time_window)
        
        row3.addWidget(QLabel('刷新率:'))
        self.refresh_rate = QSpinBox()
        self.refresh_rate.setRange(1, 60)
        self.refresh_rate.setValue(10)
        self.refresh_rate.setSuffix(' Hz')
        row3.addWidget(self.refresh_rate)
        
        self.auto_scale = QCheckBox('自动缩放')
        self.auto_scale.setChecked(True)
        row3.addWidget(self.auto_scale)
        
        clear_data_btn = QPushButton('清除数据')
        clear_data_btn.clicked.connect(self._clear_data)
        row3.addWidget(clear_data_btn)
        
        row3.addStretch()
        control_layout.addLayout(row3)
        
        # 第四行：FFT功能
        fft_group = QGroupBox('快速傅里叶变换 (FFT)')
        fft_layout = QHBoxLayout(fft_group)
        
        fft_layout.addWidget(QLabel('选择变量:'))
        self.fft_var_combo = QComboBox()
        self.fft_var_combo.setMinimumWidth(200)
        fft_layout.addWidget(self.fft_var_combo)
        
        fft_layout.addWidget(QLabel('采样间隔:'))
        self.fft_sample_interval = QDoubleSpinBox()
        self.fft_sample_interval.setRange(0.1, 1000)
        self.fft_sample_interval.setValue(10.0)
        self.fft_sample_interval.setSuffix(' ms')
        fft_layout.addWidget(self.fft_sample_interval)
        
        calc_fft_btn = QPushButton('计算FFT')
        calc_fft_btn.clicked.connect(self._calculate_fft)
        fft_layout.addWidget(calc_fft_btn)
        
        clear_fft_btn = QPushButton('清除FFT')
        clear_fft_btn.clicked.connect(self._clear_fft)
        fft_layout.addWidget(clear_fft_btn)
        
        fft_layout.addStretch()
        control_layout.addWidget(fft_group)
        
        splitter.addWidget(control_group)
        splitter.setStretchFactor(0, 3)  # 图表占更大空间
        splitter.setStretchFactor(1, 1)
        
        layout.addWidget(splitter)
    
    def _create_single_plot(self):
        """创建单图模式（仅时域）"""
        self.fig.clear()
        self.ax_time = self.fig.add_subplot(111)
        self.ax_time.set_xlabel('时间 (秒)', fontsize=10)
        self.ax_time.set_ylabel('数值', fontsize=10)
        self.ax_time.set_title('时域波形', fontsize=11, fontweight='bold')
        self.ax_time.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        self.ax_freq = None
        self.fig.tight_layout(pad=1.5)
        self.canvas.draw()
    
    def _create_dual_plot(self):
        """创建双图模式（时域+频域）"""
        self.fig.clear()
        self.ax_time = self.fig.add_subplot(211)
        self.ax_time.set_xlabel('时间 (秒)', fontsize=10)
        self.ax_time.set_ylabel('数值', fontsize=10)
        self.ax_time.set_title('时域波形', fontsize=11, fontweight='bold')
        self.ax_time.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        self.ax_freq = self.fig.add_subplot(212)
        self.ax_freq.set_xlabel('频率 (Hz)', fontsize=10)
        self.ax_freq.set_ylabel('幅值', fontsize=10)
        self.ax_freq.set_title('频域幅频曲线 (FFT)', fontsize=11, fontweight='bold')
        self.ax_freq.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        self.fig.tight_layout(pad=2.0)
        self.canvas.draw()
    
    def _add_variable(self):
        """添加监控变量"""
        selected = self.log_var_combo.currentText()
        if not selected:
            QMessageBox.warning(self, '错误', '请先从下拉框选择一个日志变量')
            return
        
        # 解析变量信息
        var_info = None
        for log_var in self.log_variables:
            var_name, byte_pos, data_type, display_format = log_var
            if selected.startswith(var_name + ' '):
                var_info = log_var
                break
        
        if not var_info:
            QMessageBox.warning(self, '错误', '无法找到对应的日志变量')
            return
        
        var_name, byte_pos, data_type, display_format = var_info
        byte_idx = byte_pos
        
        # 解析bit索引
        bit_str = self.bit_entry.text().strip()
        bit_idx = None
        
        if bit_str:
            if ':' in bit_str:
                # 切片格式
                try:
                    parts = bit_str.split(':')
                    if len(parts) == 2:
                        start = int(parts[0]) if parts[0] else 0
                        end = int(parts[1]) if parts[1] else 8
                        bit_idx = ('slice', start, end, 1)
                    elif len(parts) == 3:
                        start = int(parts[0]) if parts[0] else 0
                        end = int(parts[1]) if parts[1] else 8
                        step = int(parts[2]) if parts[2] else 1
                        bit_idx = ('slice', start, end, step)
                    else:
                        QMessageBox.warning(self, '错误', '位切片格式错误')
                        return
                    
                    if start < 0 or start > 7 or end < 0 or end > 8 or start >= end:
                        QMessageBox.warning(self, '错误', '位索引范围错误')
                        return
                except ValueError:
                    QMessageBox.warning(self, '错误', '位切片格式错误')
                    return
            else:
                # 单个位
                try:
                    bit_val = int(bit_str)
                    if bit_val < 0 or bit_val > 7:
                        QMessageBox.warning(self, '错误', 'Bit索引必须在0-7之间')
                        return
                    bit_idx = bit_val
                except ValueError:
                    QMessageBox.warning(self, '错误', 'Bit索引必须是整数')
                    return
        
        # 构建显示名称
        if isinstance(bit_idx, tuple) and bit_idx[0] == 'slice':
            _, start, end, step = bit_idx
            if step == 1:
                name = f"{var_name}.Bit[{start}:{end}]"
            else:
                name = f"{var_name}.Bit[{start}:{end}:{step}]"
        elif bit_idx is not None:
            name = f"{var_name}.Bit[{bit_idx}]"
        else:
            name = var_name
        
        # 检查是否已存在
        for var in self.scope_variables:
            if var[0] == byte_idx and var[1] == bit_idx:
                QMessageBox.warning(self, '警告', '该变量已存在')
                return
        
        # 分配颜色
        color = self.scope_colors[self.scope_color_idx % len(self.scope_colors)]
        self.scope_color_idx += 1
        
        # 添加变量
        self.scope_variables.append((byte_idx, bit_idx, name, color))
        
        # 更新列表显示
        self.var_listbox.addItem(f"● {name}  (Byte[{byte_idx}], {data_type})  —  {color}")
        
        # 初始化数据存储
        key = (byte_idx, bit_idx)
        if key not in self.scope_data:
            self.scope_data[key] = deque(maxlen=10000)
        
        # 清空bit输入
        self.bit_entry.clear()
    
    def _remove_variable(self):
        """删除选中的变量"""
        current_row = self.var_listbox.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, '提示', '请先选择要删除的变量')
            return
        
        # 移除变量
        byte_idx, bit_idx, name, color = self.scope_variables[current_row]
        del self.scope_variables[current_row]
        
        # 移除数据
        key = (byte_idx, bit_idx)
        if key in self.scope_data:
            del self.scope_data[key]
        
        # 更新列表
        self.var_listbox.takeItem(current_row)
    
    def _clear_variables(self):
        """清空所有变量"""
        self.scope_variables.clear()
        self.scope_data.clear()
        self.var_listbox.clear()
        self.scope_color_idx = 0
    
    def _clear_data(self):
        """清除数据但保留变量"""
        for key in self.scope_data:
            self.scope_data[key].clear()
        import time
        self.scope_start_time = time.time()
    
    def _show_bit_help(self):
        """显示Bit功能帮助"""
        help_text = """【位（Bit）提取功能说明】

一个字节(Byte)包含8位(Bit)，编号从0到7。

使用方式：
1. 提取单个位：输入 3 → 提取第3位
2. 提取位切片：输入 2:5 → 提取bit2到bit4
3. 带步长切片：输入 0:8:2 → 提取bit0,2,4,6

示例：
假设 Byte[5] = 0b11010110 (十进制214)
• Bit留空    → 显示整个字节: 214
• Bit填"3"   → 显示第3位: 0
• Bit填"2:6" → 显示bit2-5组合: 5
• Bit填"0:8:2" → 显示bit0,2,4,6: 7"""
        
        QMessageBox.information(self, 'Bit功能帮助', help_text)
    
    def refresh_log_vars(self, silent=False):
        """刷新日志变量列表"""
        self.log_var_combo.clear()
        self.fft_var_combo.clear()
        
        if not self.log_variables:
            if not silent:
                QMessageBox.information(self, '提示', 
                    '请先在"自定义帧→日志变量配置"中添加日志变量')
            return
        
        for var_name, byte_pos, data_type, display_format in self.log_variables:
            option = f"{var_name} (Byte[{byte_pos}], {data_type})"
            self.log_var_combo.addItem(option)
            self.fft_var_combo.addItem(option)
    
    def _calculate_fft(self):
        """计算FFT"""
        selected = self.fft_var_combo.currentText()
        if not selected:
            QMessageBox.warning(self, '错误', '请选择变量')
            return
        
        # 查找对应的变量数据
        var_name = selected.split(' ')[0]
        var_key = None
        for byte_idx, bit_idx, name, color in self.scope_variables:
            if name.startswith(var_name):
                var_key = (byte_idx, bit_idx)
                break
        
        if not var_key or var_key not in self.scope_data:
            QMessageBox.warning(self, '错误', '该变量未在监控列表中或无数据')
            return
        
        data_points = list(self.scope_data[var_key])
        if len(data_points) < 10:
            QMessageBox.warning(self, '错误', '数据点太少，无法计算FFT')
            return
        
        # 提取值序列
        values = np.array([v for t, v in data_points])
        
        # 计算FFT
        fft_result = np.fft.fft(values)
        freqs = np.fft.fftfreq(len(values), d=self.fft_sample_interval.value() / 1000.0)
        
        # 只取正频率部分
        positive_freqs = freqs[:len(freqs)//2]
        magnitudes = np.abs(fft_result[:len(fft_result)//2])
        
        self.fft_data[var_key] = (positive_freqs, magnitudes)
        self.fft_active = True
        
        # 切换到双图模式
        if self.ax_freq is None:
            self._create_dual_plot()
    
    def _clear_fft(self):
        """清除FFT"""
        self.fft_data.clear()
        self.fft_active = False
        self._create_single_plot()
    
    def _start_update_timer(self):
        """启动更新定时器"""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_plot)
        self.update_timer.start(100)  # 100ms更新一次
    
    def _update_plot(self):
        """更新图表"""
        if not self.scope_variables:
            return
        
        import time
        current_time = time.time()
        time_window = self.time_window.value()
        
        # 清空时域图
        self.ax_time.clear()
        self.ax_time.set_xlabel('时间 (秒)', fontsize=10)
        self.ax_time.set_ylabel('数值', fontsize=10)
        self.ax_time.set_title('时域波形', fontsize=11, fontweight='bold')
        self.ax_time.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # 绘制每个变量
        for byte_idx, bit_idx, name, color in self.scope_variables:
            key = (byte_idx, bit_idx)
            if key not in self.scope_data:
                continue
            
            data_points = list(self.scope_data[key])
            if not data_points:
                continue
            
            # 过滤时间窗口内的数据
            filtered = [(t, v) for t, v in data_points 
                       if current_time - t <= time_window]
            
            if not filtered:
                continue
            
            times = [t - self.scope_start_time for t, v in filtered]
            values = [v for t, v in filtered]
            
            self.ax_time.plot(times, values, label=name, color=color, linewidth=1.5)
        
        if self.scope_variables:
            self.ax_time.legend(loc='upper right', fontsize=8)
        
        # 如果有FFT数据，绘制频域图
        if self.fft_active and self.ax_freq is not None:
            self.ax_freq.clear()
            self.ax_freq.set_xlabel('频率 (Hz)', fontsize=10)
            self.ax_freq.set_ylabel('幅值', fontsize=10)
            self.ax_freq.set_title('频域幅频曲线 (FFT)', fontsize=11, fontweight='bold')
            self.ax_freq.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            for var_key, (freqs, mags) in self.fft_data.items():
                # 找到对应的变量名和颜色
                for byte_idx, bit_idx, name, color in self.scope_variables:
                    if (byte_idx, bit_idx) == var_key:
                        self.ax_freq.plot(freqs, mags, label=name, color=color, linewidth=1.5)
                        break
            
            if self.fft_data:
                self.ax_freq.legend(loc='upper right', fontsize=8)
        
        self.canvas.draw()
    
    def add_data_point(self, byte_idx, value, timestamp):
        """添加数据点（外部调用）"""
        # 遍历所有监控变量
        for var_byte_idx, bit_idx, name, color in self.scope_variables:
            if var_byte_idx == byte_idx:
                key = (var_byte_idx, bit_idx)
                
                # 计算实际值
                if isinstance(bit_idx, tuple) and bit_idx[0] == 'slice':
                    _, start, end, step = bit_idx
                    # 提取位切片
                    extracted = 0
                    for i, bit in enumerate(range(start, end, step)):
                        if value & (1 << bit):
                            extracted |= (1 << i)
                    actual_value = extracted
                elif bit_idx is not None:
                    # 提取单个位
                    actual_value = 1 if (value & (1 << bit_idx)) else 0
                else:
                    # 整个字节
                    actual_value = value
                
                if key in self.scope_data:
                    self.scope_data[key].append((timestamp, actual_value))
