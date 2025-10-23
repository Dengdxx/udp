#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
udp_gui.py - UDP 上位机 GUI（带 ttkbootstrap 优雅降级）

功能概览：
- 运行/停止 UDP 监听（调用 udp_image_logger.py run）
- PNG 保存目录、日志 CSV、帧索引 CSV 配置
- 一键合成视频（video）
- 一键对齐（align）
- 示波器（scope）：按日志包字节索引选择，支持可选 bit

兼容性：
- 若环境未安装 ttkbootstrap，将自动回退为标准 ttk 外观（功能不变）。
"""

import os
import sys
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import socket
import struct
import time
from datetime import datetime
from typing import Optional, Tuple
import csv

# 尝试导入 ttkbootstrap；失败则降级为标准 ttk
try:
    import ttkbootstrap as tb  # type: ignore
    from ttkbootstrap import ttk  # ttkbootstrap 自带 ttk 封装，支持 bootstyle
    try:
        from ttkbootstrap.scrolled import ScrolledText as TBScrolledText  # 高级滚动文本
    except Exception:
        TBScrolledText = None
    HAS_TTKBOOTSTRAP = True
except Exception:
    tb = None  # type: ignore
    import tkinter.ttk as ttk  # 标准 ttk
    from tkinter.scrolledtext import ScrolledText as TkScrolledText
    TBScrolledText = None
    HAS_TTKBOOTSTRAP = False

# 导入图像处理库
try:
    import cv2
    import numpy as np
    from PIL import Image, ImageTk
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("[WARN] OpenCV or PIL not available, video display disabled")

# Switch 控件（仅在较新 ttkbootstrap 中提供）
Switch = getattr(tb, 'Switch', None) if HAS_TTKBOOTSTRAP else None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_SCRIPT = os.path.join(SCRIPT_DIR, 'udp_image_logger.py')


# ---------------------- 帧解析工具类 ----------------------
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
    pixels = np.frombuffer(data[3:], dtype=np.uint8) if HAS_CV2 else None
    if pixels is not None:
        img = pixels.reshape((h, w))
        return h, w, img
    return h, w, None


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
    
    if not HAS_CV2:
        return h, w, None
    
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


def parse_log_frame(data: bytes) -> Tuple[int, bytes]:
    """解析日志帧 [0x02][LEN][TS_0..TS_7][payload...]"""
    if len(data) < 1 + 1 + 8:
        raise ValueError("log frame too short")
    length = data[1]
    if len(data) != 1 + 1 + 8 + length:
        raise ValueError("log frame size mismatch")
    ts_us = struct.unpack_from('<Q', data, 2)[0]
    payload = data[10:10 + length]
    return ts_us, payload


# ---------------------- UDP 接收线程 ----------------------
class UdpVideoReceiver:
    """UDP 视频接收器，在后台线程接收并更新图像"""
    
    def __init__(self, ip: str, port: int, save_png: bool = False, 
                 png_dir: str = 'frames_png',
                 log_csv: str = 'logs.csv',
                 frame_index_csv: str = 'frames_index.csv'):
        self.ip = ip
        self.port = port
        self.save_png = save_png
        self.png_dir = png_dir
        self.log_csv = log_csv
        self.frame_index_csv = frame_index_csv
        
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        
        # 当前帧数据
        self.current_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()
        
        # 统计信息
        self.frame_counter = 0
        self.total_packets = 0
        self.error_packets = 0
        self.fps = 0.0
        self._fps_timer = time.time()
        self._fps_frame_count = 0
        
        # 原始数据缓存（最近的数据包）
        self.recent_data = []  # 存储最近的数据包 [(timestamp, type, data_hex, parsed_info), ...]
        self.max_recent_data = 100  # 最多保存100条
        self.data_lock = threading.Lock()
        
        # CSV 文件
        self._log_csv_fp: Optional[object] = None
        self._log_writer: Optional[object] = None
        self._frame_index_fp: Optional[object] = None
        self._frame_index_writer: Optional[object] = None
    
    def start(self):
        """启动 UDP 接收线程"""
        if self._running:
            return False
        
        try:
            # 创建 socket
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
                self._log_writer.writerow(["stm32_ts_us", "host_recv_iso", "log_text_hex", "log_text_utf8"])
            
            frame_csv_exists = os.path.exists(self.frame_index_csv) and os.path.getsize(self.frame_index_csv) > 0
            self._frame_index_fp = open(self.frame_index_csv, 'a', newline='', encoding='utf-8')
            self._frame_index_writer = csv.writer(self._frame_index_fp)
            if not frame_csv_exists:
                self._frame_index_writer.writerow(["frame_id", "stm32_ts_us", "host_recv_iso", "png_path", "h", "w"])
            
            # 启动线程
            self._stop_event.clear()
            self._running = True
            self._thread = threading.Thread(target=self._receive_loop, daemon=True)
            self._thread.start()
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to start UDP receiver: {e}")
            self._cleanup()
            return False
    
    def stop(self):
        """停止 UDP 接收"""
        if not self._running:
            return
        
        self._stop_event.set()
        self._running = False
        
        if self._thread:
            self._thread.join(timeout=3.0)
        
        self._cleanup()
    
    def _cleanup(self):
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
        """更新 FPS 统计"""
        self._fps_frame_count += 1
        elapsed = time.time() - self._fps_timer
        if elapsed >= 1.0:
            self.fps = self._fps_frame_count / elapsed
            self._fps_frame_count = 0
            self._fps_timer = time.time()
    
    def _receive_loop(self):
        """接收循环（在后台线程运行）"""
        print(f"[INFO] UDP receiver started on {self.ip}:{self.port}")
        
        while not self._stop_event.is_set():
            try:
                data, addr = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    print(f"[ERROR] Socket error: {e}")
                break
            
            self.total_packets += 1
            host_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            
            if not data:
                continue
            
            ftype = data[0]
            
            try:
                if ftype == FrameType.IMAGE:
                    h, w, img = parse_image_frame(data)
                    if img is not None:
                        self.frame_counter += 1
                        self._update_fps()
                        
                        # 更新当前帧
                        with self.frame_lock:
                            self.current_frame = img.copy()
                        
                        # 记录原始数据
                        with self.data_lock:
                            # 对于图像帧，只保存前 100 字节用于显示
                            self.recent_data.append((
                                host_iso,
                                'IMAGE',
                                data[:100].hex() + ('...' if len(data) > 100 else ''),
                                f"Frame {self.frame_counter}: {w}x{h}, {len(data)} bytes"
                            ))
                            if len(self.recent_data) > self.max_recent_data:
                                self.recent_data.pop(0)
                        
                        # 保存 PNG
                        png_path = ''
                        if self.save_png:
                            png_path = os.path.join(self.png_dir, f"frame_{self.frame_counter:06d}.png")
                            cv2.imwrite(png_path, img)
                        
                        # 记录到 CSV
                        if self._frame_index_writer:
                            self._frame_index_writer.writerow([self.frame_counter, -1, host_iso, png_path, h, w])
                            self._frame_index_fp.flush()
                
                elif ftype == FrameType.BINARY_IMAGE:
                    h, w, img = parse_binary_image_frame(data)
                    if img is not None:
                        self.frame_counter += 1
                        self._update_fps()
                        
                        with self.frame_lock:
                            self.current_frame = img.copy()
                        
                        # 记录原始数据
                        with self.data_lock:
                            # 对于二值图像帧，只保存前 100 字节用于显示
                            self.recent_data.append((
                                host_iso,
                                'BINARY_IMAGE',
                                data[:100].hex() + ('...' if len(data) > 100 else ''),
                                f"Binary Frame {self.frame_counter}: {w}x{h}, {len(data)} bytes"
                            ))
                            if len(self.recent_data) > self.max_recent_data:
                                self.recent_data.pop(0)
                        
                        png_path = ''
                        if self.save_png:
                            png_path = os.path.join(self.png_dir, f"frame_{self.frame_counter:06d}.png")
                            cv2.imwrite(png_path, img)
                        
                        if self._frame_index_writer:
                            self._frame_index_writer.writerow([self.frame_counter, -1, host_iso, png_path, h, w])
                            self._frame_index_fp.flush()
                
                elif ftype == FrameType.LOG:
                    ts_us, payload = parse_log_frame(data)
                    text_utf8 = ''
                    try:
                        text_utf8 = payload.decode('utf-8', errors='replace')
                    except:
                        pass
                    text_hex = payload.hex()
                    
                    # 记录原始数据（LOG 帧保存完整数据，因为通常不大）
                    with self.data_lock:
                        # 限制显示长度但保存完整 hex
                        display_hex = data.hex()
                        if len(display_hex) > 500:  # 如果太长，截断显示
                            display_hex = display_hex[:500] + '...'
                        
                        self.recent_data.append((
                            host_iso,
                            'LOG',
                            display_hex,
                            f"LOG ts={ts_us}us: {text_utf8[:50]}" + ('...' if len(text_utf8) > 50 else '')
                        ))
                        if len(self.recent_data) > self.max_recent_data:
                            self.recent_data.pop(0)
                    
                    if self._log_writer:
                        self._log_writer.writerow([ts_us, host_iso, text_hex, text_utf8])
                        self._log_csv_fp.flush()
                else:
                    # 未知类型
                    self.error_packets += 1
                    with self.data_lock:
                        self.recent_data.append((
                            host_iso,
                            f'UNKNOWN(0x{ftype:02X})',
                            data[:100].hex() + ('...' if len(data) > 100 else ''),
                            f"Unknown frame type, {len(data)} bytes"
                        ))
                        if len(self.recent_data) > self.max_recent_data:
                            self.recent_data.pop(0)
            
            except Exception as e:
                self.error_packets += 1
                print(f"[ERROR] Parse error: {e}")
                with self.data_lock:
                    self.recent_data.append((
                        host_iso,
                        'ERROR',
                        data[:100].hex() + ('...' if len(data) > 100 else ''),
                        f"Parse error: {str(e)}"
                    ))
                    if len(self.recent_data) > self.max_recent_data:
                        self.recent_data.pop(0)
        
        print("[INFO] UDP receiver stopped")



class App(tb.Window if HAS_TTKBOOTSTRAP else tk.Tk):
    def __init__(self):
        if HAS_TTKBOOTSTRAP:
            super().__init__(themename='flatly')  # 可选: superhero, cyborg, darkly, litera, flatly...
            # ttkbootstrap 的 Window 自带 style
        else:
            super().__init__()
            # 标准 ttk 需要手动创建 style
            self.style = ttk.Style(self)

        self.title('UDP 上位机 GUI')
        self.geometry('1200x800')

        self.proc: subprocess.Popen | None = None
        self.scope_proc: subprocess.Popen | None = None
        
        # UDP 视频接收器
        self.video_receiver: Optional[UdpVideoReceiver] = None
        self._video_update_job = None

        # --- 参数 ---
        self.ip = tk.StringVar(value='0.0.0.0')
        self.port = tk.IntVar(value=5005)
        self.show = tk.BooleanVar(value=True)
        self.save_png = tk.BooleanVar(value=False)
        self.png_dir = tk.StringVar(value=os.path.join(os.getcwd(), 'frames_png'))
        self.log_csv = tk.StringVar(value=os.path.join(os.getcwd(), 'logs.csv'))
        self.frame_index_csv = tk.StringVar(value=os.path.join(os.getcwd(), 'frames_index.csv'))

        self.video_png_dir = tk.StringVar(value=os.path.join(os.getcwd(), 'frames_png'))
        self.video_out = tk.StringVar(value=os.path.join(os.getcwd(), 'output.mp4'))
        self.video_fps = tk.IntVar(value=30)

        self.scope_index = tk.IntVar(value=0)
        self.scope_bit = tk.StringVar(value='')  # 允许空，或 0..7
        self.scope_max_points = tk.IntVar(value=2000)

    # 已移除 C 扩展处理选项（ctypes），保持 GUI 简洁

        self._build_ui()
        
        # 绑定关闭事件
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    # ---------- 小部件工厂（处理 bootstyle 兼容） ----------
    def _btn(self, parent, text, command, bootstyle: str | None = None):
        if HAS_TTKBOOTSTRAP:
            return ttk.Button(parent, text=text, command=command, bootstyle=bootstyle)
        return ttk.Button(parent, text=text, command=command)

    def _nb(self, parent):
        if HAS_TTKBOOTSTRAP:
            return ttk.Notebook(parent, bootstyle='primary')
        return ttk.Notebook(parent)

    def _scrolled_text(self, parent):
        if HAS_TTKBOOTSTRAP and TBScrolledText is not None:
            return TBScrolledText(parent, autohide=True, height=8, bootstyle='secondary')
        # 标准 tkinter 的滚动文本
        if not HAS_TTKBOOTSTRAP:
            try:
                from tkinter.scrolledtext import ScrolledText as TkScrolledText  # type: ignore
                return TkScrolledText(parent, height=8)
            except Exception:
                # 兜底：普通 Text + 手动滚动条
                frame = ttk.Frame(parent)
                txt = tk.Text(frame, height=8)
                sb = ttk.Scrollbar(frame, orient='vertical', command=txt.yview)
                txt.configure(yscrollcommand=sb.set)
                txt.pack(side='left', fill='both', expand=True)
                sb.pack(side='right', fill='y')
                txt._container = frame  # type: ignore[attr-defined]
                return txt
        # 若 TBScrolledText 导入失败但处于 ttkbootstrap 环境，用普通 Text
        return tk.Text(parent, height=8)

    def _add_switch(self, parent, text: str, var, bootstyle: str | None, grid_kwargs: dict):
        """创建一个开关；若 ttkbootstrap 不可用或无 Switch，则退化为 Checkbutton。"""
        if HAS_TTKBOOTSTRAP and Switch is not None:
            w = Switch(parent, text=text, variable=var, bootstyle=bootstyle)
        else:
            w = ttk.Checkbutton(parent, text=text, variable=var)
        w.grid(**grid_kwargs)

    # ---------------- UI 构建 ----------------
    def _build_ui(self):
        # 创建主容器：左侧控制面板 + 右侧视频显示
        main_container = ttk.PanedWindow(self, orient='horizontal')
        main_container.pack(fill='both', expand=True, padx=4, pady=4)
        
        # 左侧控制面板
        left_panel = ttk.Frame(main_container)
        main_container.add(left_panel, weight=1)
        
        # 右侧视频显示区域
        right_panel = ttk.Frame(main_container)
        main_container.add(right_panel, weight=2)
        
        # === 左侧面板内容 ===
        # 顶部工具栏（主题切换）
        toolbar = ttk.Frame(left_panel)
        toolbar.pack(fill='x', padx=8, pady=(8, 4))
        ttk.Label(toolbar, text='主题:').pack(side='left')
        current_theme = 'flatly' if HAS_TTKBOOTSTRAP else self.style.theme_use()
        self.theme_var = tk.StringVar(value=current_theme)
        theme_sel = ttk.Combobox(
            toolbar,
            textvariable=self.theme_var,
            width=16,
            state='readonly',
            values=sorted(self.style.theme_names()),
        )
        theme_sel.pack(side='left', padx=(6, 12))
        self._btn(toolbar, text='应用主题', command=self.apply_theme, bootstyle='secondary-outline').pack(side='left')

        # 标签页
        nb = self._nb(left_panel)
        nb.pack(fill='both', expand=True, padx=8, pady=4)

        nb.add(self._build_tab_run(), text='运行')
        nb.add(self._build_tab_video(), text='视频')
        nb.add(self._build_tab_align(), text='对齐')
        nb.add(self._build_tab_scope(), text='示波')
        nb.add(self._build_tab_test(), text='测试')

        # 输出区域（带滚动）
        self.out_text = self._scrolled_text(left_panel)
        container = getattr(self.out_text, '_container', None)
        if container is not None:
            container.pack(fill='both', expand=True, padx=8, pady=(4, 8))
        else:
            self.out_text.pack(fill='both', expand=True, padx=8, pady=(4, 8))
        self._log('提示：点击"启动监听"查看实时视频流。')
        
        # === 右侧视频显示区域 ===
        video_frame = ttk.LabelFrame(right_panel, text='实时视频', padding=10)
        video_frame.pack(fill='both', expand=True, padx=8, pady=(8, 4))
        
        # 视频画布
        self.video_canvas = tk.Canvas(video_frame, bg='black', highlightthickness=0)
        self.video_canvas.pack(fill='both', expand=True)
        
        # 统计信息标签
        stats_frame = ttk.Frame(video_frame)
        stats_frame.pack(fill='x', pady=(8, 0))
        
        self.stats_label = ttk.Label(stats_frame, text='等待视频流...', font=('Consolas', 9))
        self.stats_label.pack()
        
        # 原始数据显示区域
        data_frame = ttk.LabelFrame(right_panel, text='原始数据监视器', padding=10)
        data_frame.pack(fill='both', expand=False, padx=8, pady=(4, 8))
        
        # 工具栏
        data_toolbar = ttk.Frame(data_frame)
        data_toolbar.pack(fill='x', pady=(0, 4))
        
        self._btn(data_toolbar, text='清空', command=self._clear_data_display, bootstyle='secondary-outline').pack(side='left', padx=2)
        self._btn(data_toolbar, text='刷新', command=self._refresh_data_display, bootstyle='info-outline').pack(side='left', padx=2)
        
        ttk.Label(data_toolbar, text='最大显示:').pack(side='left', padx=(10, 2))
        self.data_display_limit = tk.IntVar(value=20)
        ttk.Spinbox(data_toolbar, from_=10, to=100, width=8, textvariable=self.data_display_limit).pack(side='left')
        
        ttk.Label(data_toolbar, text='编码:').pack(side='left', padx=(10, 2))
        self.data_encoding = tk.StringVar(value='UTF-8')
        encoding_combo = ttk.Combobox(
            data_toolbar,
            textvariable=self.data_encoding,
            width=12,
            state='readonly',
            values=['UTF-8', 'GBK', 'GB2312', 'ASCII', 'Latin-1', 'UTF-16', 'UTF-32', 'Big5']
        )
        encoding_combo.pack(side='left', padx=2)
        encoding_combo.bind('<<ComboboxSelected>>', lambda e: self._refresh_data_display())
        
        ttk.Label(data_toolbar, text='显示格式:').pack(side='left', padx=(10, 2))
        self.data_format = tk.StringVar(value='详细')
        format_combo = ttk.Combobox(
            data_toolbar,
            textvariable=self.data_format,
            width=10,
            state='readonly',
            values=['详细', '简洁', '仅Hex', '仅文本']
        )
        format_combo.pack(side='left', padx=2)
        format_combo.bind('<<ComboboxSelected>>', lambda e: self._refresh_data_display())
        
        # 数据显示文本框
        self.data_text = self._scrolled_text(data_frame)
        container = getattr(self.data_text, '_container', None)
        if container is not None:
            container.pack(fill='both', expand=True)
        else:
            self.data_text.pack(fill='both', expand=True)
        
        # 配置文本框样式（等宽字体）
        try:
            self.data_text.configure(font=('Consolas', 9), height=8)
        except:
            pass
        
        # 显示提示信息
        self._show_video_placeholder()

    def _build_tab_run(self):
        f = ttk.Frame()

        row = 0
        ttk.Label(f, text='绑定 IP:').grid(row=row, column=0, sticky='e', padx=6, pady=6)
        ttk.Entry(f, textvariable=self.ip, width=15).grid(row=row, column=1, sticky='w')

        ttk.Label(f, text='端口:').grid(row=row, column=2, sticky='e')
        ttk.Entry(f, textvariable=self.port, width=8).grid(row=row, column=3, sticky='w')

        row += 1
        # 移除 show 开关（因为现在始终在主窗口显示）
        self._add_switch(
            f,
            text='保存 PNG (--save-png)',
            var=self.save_png,
            bootstyle='info',
            grid_kwargs=dict(row=row, column=0, columnspan=2, sticky='w', padx=6),
        )

        row += 1
        ttk.Label(f, text='PNG 目录:').grid(row=row, column=0, sticky='e', padx=6)
        ttk.Entry(f, textvariable=self.png_dir, width=48).grid(row=row, column=1, columnspan=2, sticky='we')
        self._btn(f, text='选择', command=self._pick_png_dir, bootstyle='secondary-outline').grid(row=row, column=3, sticky='w')

        row += 1
        ttk.Label(f, text='日志 CSV:').grid(row=row, column=0, sticky='e', padx=6)
        ttk.Entry(f, textvariable=self.log_csv, width=48).grid(row=row, column=1, columnspan=2, sticky='we')
        self._btn(f, text='选择', command=self._pick_log_csv, bootstyle='secondary-outline').grid(row=row, column=3, sticky='w')

        row += 1
        ttk.Label(f, text='帧索引 CSV:').grid(row=row, column=0, sticky='e', padx=6)
        ttk.Entry(f, textvariable=self.frame_index_csv, width=48).grid(row=row, column=1, columnspan=2, sticky='we')
        self._btn(f, text='选择', command=self._pick_frame_csv, bootstyle='secondary-outline').grid(row=row, column=3, sticky='w')

        row += 1
        self._btn(f, text='启动监听', command=self.start_run, bootstyle='success').grid(row=row, column=1, sticky='we', padx=6, pady=10)
        self._btn(f, text='停止监听', command=self.stop_run, bootstyle='danger').grid(row=row, column=2, sticky='we', padx=6, pady=10)

        for c in range(4):
            f.grid_columnconfigure(c, weight=1)
        return f

    def _build_tab_test(self):
        f = ttk.Frame()
        row = 0
        self._btn(f, text='启动测试服务端', command=self.start_test_host, bootstyle='info').grid(row=row, column=0, sticky='we', padx=6, pady=10)
        self._btn(f, text='启动测试客户端', command=self.start_test_client, bootstyle='info').grid(row=row, column=1, sticky='we', padx=6, pady=10)
        return f

    def _build_tab_video(self):
        f = ttk.Frame()
        row = 0
        ttk.Label(f, text='PNG 目录:').grid(row=row, column=0, sticky='e', padx=6, pady=6)
        ttk.Entry(f, textvariable=self.video_png_dir, width=48).grid(row=row, column=1, columnspan=2, sticky='we')
        self._btn(f, text='选择', command=self._pick_video_png_dir, bootstyle='secondary-outline').grid(row=row, column=3, sticky='w')

        row += 1
        ttk.Label(f, text='输出 MP4:').grid(row=row, column=0, sticky='e', padx=6, pady=6)
        ttk.Entry(f, textvariable=self.video_out, width=48).grid(row=row, column=1, columnspan=2, sticky='we')
        self._btn(f, text='选择', command=self._pick_video_out, bootstyle='secondary-outline').grid(row=row, column=3, sticky='w')

        row += 1
        ttk.Label(f, text='FPS:').grid(row=row, column=0, sticky='e', padx=6, pady=6)
        ttk.Entry(f, textvariable=self.video_fps, width=8).grid(row=row, column=1, sticky='w')
        self._btn(f, text='开始合成', command=self.compose_video, bootstyle='primary').grid(row=row, column=2, sticky='we', padx=6)
        return f

    def _build_tab_align(self):
        f = ttk.Frame()
        row = 0
        ttk.Label(f, text='frames_index.csv:').grid(row=row, column=0, sticky='e', padx=6, pady=6)
        ttk.Entry(f, textvariable=self.frame_index_csv, width=48).grid(row=row, column=1, columnspan=2, sticky='we')
        self._btn(f, text='选择', command=self._pick_frame_csv, bootstyle='secondary-outline').grid(row=row, column=3, sticky='w')

        row += 1
        ttk.Label(f, text='logs.csv:').grid(row=row, column=0, sticky='e', padx=6, pady=6)
        ttk.Entry(f, textvariable=self.log_csv, width=48).grid(row=row, column=1, columnspan=2, sticky='we')
        self._btn(f, text='选择', command=self._pick_log_csv, bootstyle='secondary-outline').grid(row=row, column=3, sticky='w')

        row += 1
        self._btn(f, text='执行对齐', command=self.align_csv, bootstyle='primary').grid(row=row, column=1, sticky='we', padx=6)
        return f

    def _build_tab_scope(self):
        f = ttk.Frame()
        row = 0
        ttk.Label(f, text='绑定 IP:').grid(row=row, column=0, sticky='e', padx=6, pady=6)
        ttk.Entry(f, textvariable=self.ip, width=15).grid(row=row, column=1, sticky='w')

        ttk.Label(f, text='端口:').grid(row=row, column=2, sticky='e')
        ttk.Entry(f, textvariable=self.port, width=8).grid(row=row, column=3, sticky='w')

        row += 1
        ttk.Label(f, text='Byte 索引:').grid(row=row, column=0, sticky='e', padx=6)
        ttk.Entry(f, textvariable=self.scope_index, width=8).grid(row=row, column=1, sticky='w')

        ttk.Label(f, text='Bit (可空 0..7):').grid(row=row, column=2, sticky='e')
        ttk.Entry(f, textvariable=self.scope_bit, width=8).grid(row=row, column=3, sticky='w')

        row += 1
        ttk.Label(f, text='最大点数:').grid(row=row, column=0, sticky='e', padx=6)
        ttk.Entry(f, textvariable=self.scope_max_points, width=10).grid(row=row, column=1, sticky='w')

        row += 1
        self._btn(f, text='启动示波器', command=self.start_scope, bootstyle='success').grid(row=row, column=1, sticky='we', padx=6)
        self._btn(f, text='停止示波器', command=self.stop_scope, bootstyle='danger').grid(row=row, column=2, sticky='we', padx=6)
        for c in range(4):
            f.grid_columnconfigure(c, weight=1)
        return f

    # ---------------- 事件处理 ----------------
    def _on_closing(self):
        """窗口关闭事件"""
        # 停止视频接收
        if self.video_receiver and self.video_receiver._running:
            self.stop_run()
        
        # 停止视频更新
        if self._video_update_job:
            self.after_cancel(self._video_update_job)
        
        self.destroy()
    
    def _show_video_placeholder(self):
        """显示视频占位符"""
        self.video_canvas.delete('all')
        w = self.video_canvas.winfo_width()
        h = self.video_canvas.winfo_height()
        if w > 1 and h > 1:
            self.video_canvas.create_text(
                w // 2, h // 2,
                text='等待视频流...\n请点击"启动监听"开始接收',
                fill='white',
                font=('Arial', 14),
                justify='center'
            )
    
    def _clear_data_display(self):
        """清空原始数据显示"""
        if self.video_receiver:
            with self.video_receiver.data_lock:
                self.video_receiver.recent_data.clear()
        self.data_text.delete('1.0', 'end')
        self._log('已清空原始数据显示')
    
    def _refresh_data_display(self):
        """手动刷新原始数据显示"""
        self._update_data_display()
        self._log('已刷新原始数据显示')
    
    def _update_data_display(self):
        """更新原始数据显示"""
        if not self.video_receiver or not self.video_receiver._running:
            return
        
        # 获取最近的数据
        data_list = []
        with self.video_receiver.data_lock:
            # 只显示最后 N 条
            limit = self.data_display_limit.get()
            data_list = list(self.video_receiver.recent_data[-limit:])
        
        # 获取编码和显示格式
        encoding = self.data_encoding.get().lower()
        if encoding == 'utf-8':
            encoding = 'utf-8'
        elif encoding == 'gbk':
            encoding = 'gbk'
        elif encoding == 'gb2312':
            encoding = 'gb2312'
        elif encoding == 'ascii':
            encoding = 'ascii'
        elif encoding == 'latin-1':
            encoding = 'latin-1'
        elif encoding == 'utf-16':
            encoding = 'utf-16'
        elif encoding == 'utf-32':
            encoding = 'utf-32'
        elif encoding == 'big5':
            encoding = 'big5'
        else:
            encoding = 'utf-8'
        
        display_format = self.data_format.get()
        
        # 更新显示
        self.data_text.delete('1.0', 'end')
        
        for timestamp, ftype, data_hex, info in data_list:
            if display_format == '详细':
                # 详细模式：显示所有信息
                line = f"[{timestamp}] {ftype}\n"
                line += f"  Info: {info}\n"
                
                # 尝试用选定编码解析 hex
                try:
                    data_bytes = bytes.fromhex(data_hex.replace('...', ''))
                    decoded_text = data_bytes.decode(encoding, errors='replace')
                    # 只显示可打印字符，其他用 · 代替
                    decoded_text = ''.join(c if c.isprintable() or c in '\n\r\t' else '·' for c in decoded_text)
                    line += f"  {encoding}: {decoded_text[:100]}" + ('...' if len(decoded_text) > 100 else '') + "\n"
                except:
                    line += f"  {encoding}: <decode error>\n"
                
                line += f"  Hex:  {data_hex}\n"
                line += "-" * 80 + "\n"
                
            elif display_format == '简洁':
                # 简洁模式：只显示时间、类型和简要信息
                line = f"[{timestamp[-15:]}] {ftype:12s} | {info[:60]}\n"
                
            elif display_format == '仅Hex':
                # 仅显示 Hex 数据
                line = f"[{timestamp[-15:]}] {ftype:12s}\n"
                line += f"  {data_hex}\n"
                
            elif display_format == '仅文本':
                # 仅显示解码后的文本
                line = f"[{timestamp[-15:]}] {ftype:12s}\n"
                try:
                    data_bytes = bytes.fromhex(data_hex.replace('...', ''))
                    decoded_text = data_bytes.decode(encoding, errors='replace')
                    decoded_text = ''.join(c if c.isprintable() or c in '\n\r\t' else '·' for c in decoded_text)
                    line += f"  {decoded_text}\n"
                except:
                    line += f"  <decode error>\n"
            else:
                line = f"{timestamp} {ftype} {info}\n"
            
            self.data_text.insert('end', line)
        
        self.data_text.see('end')
    
    def _update_video_display(self):
        """更新视频显示（定期调用）"""
        if not self.video_receiver or not self.video_receiver._running:
            self._video_update_job = None
            return
        
        # 获取当前帧
        frame = None
        with self.video_receiver.frame_lock:
            if self.video_receiver.current_frame is not None:
                frame = self.video_receiver.current_frame.copy()
        
        if frame is not None and HAS_CV2:
            try:
                # 获取画布尺寸
                canvas_w = self.video_canvas.winfo_width()
                canvas_h = self.video_canvas.winfo_height()
                
                if canvas_w > 1 and canvas_h > 1:
                    h, w = frame.shape[:2]
                    
                    # 计算缩放比例（保持宽高比）
                    scale_w = canvas_w / w
                    scale_h = canvas_h / h
                    scale = min(scale_w, scale_h, 4.0)  # 最大放大4倍
                    
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    
                    # 缩放图像
                    if scale > 1.0:
                        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                    else:
                        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    
                    # 转换为 RGB（如果是灰度图）
                    if len(resized.shape) == 2:
                        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
                    else:
                        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    
                    # 转换为 PIL Image
                    img_pil = Image.fromarray(resized)
                    img_tk = ImageTk.PhotoImage(image=img_pil)
                    
                    # 在画布上显示
                    self.video_canvas.delete('all')
                    x = (canvas_w - new_w) // 2
                    y = (canvas_h - new_h) // 2
                    self.video_canvas.create_image(x, y, anchor='nw', image=img_tk)
                    self.video_canvas.image = img_tk  # 保持引用
                    
                    # 绘制统计信息
                    info_y = 10
                    self.video_canvas.create_text(
                        10, info_y,
                        anchor='nw',
                        text=f"帧: {self.video_receiver.frame_counter}",
                        fill='lime',
                        font=('Consolas', 10, 'bold')
                    )
                    info_y += 20
                    self.video_canvas.create_text(
                        10, info_y,
                        anchor='nw',
                        text=f"FPS: {self.video_receiver.fps:.1f}",
                        fill='lime',
                        font=('Consolas', 10, 'bold')
                    )
                    info_y += 20
                    self.video_canvas.create_text(
                        10, info_y,
                        anchor='nw',
                        text=f"大小: {w}x{h}",
                        fill='lime',
                        font=('Consolas', 10, 'bold')
                    )
                    
                    # 更新统计标签
                    error_rate = (self.video_receiver.error_packets / self.video_receiver.total_packets * 100) \
                        if self.video_receiver.total_packets > 0 else 0
                    self.stats_label.config(
                        text=f"帧数: {self.video_receiver.frame_counter} | "
                             f"FPS: {self.video_receiver.fps:.1f} | "
                             f"数据包: {self.video_receiver.total_packets} | "
                             f"错误: {self.video_receiver.error_packets} ({error_rate:.1f}%)"
                    )
            except Exception as e:
                print(f"[ERROR] Display error: {e}")
        
        # 更新原始数据显示（每次视频更新时也更新数据）
        self._update_data_display()
        
        # 继续更新（约30fps）
        self._video_update_job = self.after(33, self._update_video_display)
    
    def _log(self, msg: str):
        try:
            self.out_text.insert('end', msg + '\n')
            self.out_text.see('end')
        except Exception:
            # 当 out_text 是 Text + 手动容器时
            self.out_text.insert('end', msg + '\n')
            self.out_text.see('end')

    def apply_theme(self):
        try:
            self.style.theme_use(self.theme_var.get())
        except Exception:
            messagebox.showerror('错误', f'无法应用主题: {self.theme_var.get()}')

    def _pick_png_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.png_dir.set(d)

    def _pick_log_csv(self):
        fpath = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV', '*.csv'), ('All', '*.*')])
        if fpath:
            self.log_csv.set(fpath)

    def _pick_frame_csv(self):
        fpath = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV', '*.csv'), ('All', '*.*')])
        if fpath:
            self.frame_index_csv.set(fpath)

    def _pick_video_png_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.video_png_dir.set(d)

    def _pick_video_out(self):
        fpath = filedialog.asksaveasfilename(defaultextension='.mp4', filetypes=[('MP4', '*.mp4'), ('All', '*.*')])
        if fpath:
            self.video_out.set(fpath)

    def start_test_host(self):
        self._log('启动测试服务端: ' + ' '.join([sys.executable, 'test_udp_host.py']))
        subprocess.Popen([sys.executable, 'test_udp_host.py'], cwd=SCRIPT_DIR)

    def start_test_client(self):
        self._log('启动测试客户端: ' + ' '.join([sys.executable, 'test_udp_client.py']))
        subprocess.Popen([sys.executable, 'test_udp_client.py'], cwd=SCRIPT_DIR)

    def start_run(self):
        """启动 UDP 监听（内嵌模式）"""
        if self.video_receiver and self.video_receiver._running:
            messagebox.showwarning('提示', '监听已在运行')
            return
        
        if not HAS_CV2:
            messagebox.showerror('错误', '未安装 OpenCV，无法显示视频\n请运行: pip install opencv-python pillow')
            return
        
        # 创建并启动接收器
        self.video_receiver = UdpVideoReceiver(
            ip=self.ip.get(),
            port=self.port.get(),
            save_png=self.save_png.get(),
            png_dir=self.png_dir.get(),
            log_csv=self.log_csv.get(),
            frame_index_csv=self.frame_index_csv.get()
        )
        
        if self.video_receiver.start():
            self._log(f'启动监听: {self.ip.get()}:{self.port.get()}')
            # 启动视频显示更新
            self._update_video_display()
        else:
            messagebox.showerror('错误', '启动 UDP 监听失败')
            self.video_receiver = None

    def stop_run(self):
        """停止 UDP 监听"""
        if self.video_receiver and self.video_receiver._running:
            self._log('停止监听...')
            self.video_receiver.stop()
            self.video_receiver = None
            
            # 停止视频更新
            if self._video_update_job:
                self.after_cancel(self._video_update_job)
                self._video_update_job = None
            
            # 清空画布
            self._show_video_placeholder()
            self.stats_label.config(text='已停止')
        else:
            messagebox.showinfo('提示', '监听未运行')

    def compose_video(self):
        args = [sys.executable, MAIN_SCRIPT, 'video', '--png-dir', self.video_png_dir.get(), '--out', self.video_out.get(), '--fps', str(self.video_fps.get())]
        self._log('合成视频: ' + ' '.join(args))
        subprocess.Popen(args, cwd=SCRIPT_DIR)

    def align_csv(self):
        out_csv = os.path.join(os.path.dirname(self.log_csv.get()) or os.getcwd(), 'aligned.csv')
        args = [
            sys.executable, MAIN_SCRIPT, 'align',
            '--frames-csv', self.frame_index_csv.get(),
            '--logs-csv', self.log_csv.get(),
            '--out-csv', out_csv,
        ]
        self._log('执行对齐: ' + ' '.join(args))
        subprocess.Popen(args, cwd=SCRIPT_DIR)

    def start_scope(self):
        if self.scope_proc and self.scope_proc.poll() is None:
            messagebox.showwarning('提示', '示波器已在运行')
            return
        bit = self.scope_bit.get().strip()
        args = [
            sys.executable, MAIN_SCRIPT, 'scope',
            '--ip', self.ip.get(), '--port', str(self.port.get()),
            '--index', str(self.scope_index.get()), '--max-points', str(self.scope_max_points.get()),
        ]
        if bit != '':
            args += ['--bit', bit]
        self._log('启动示波器: ' + ' '.join(args))
        self.scope_proc = subprocess.Popen(args, cwd=SCRIPT_DIR)

    def stop_scope(self):
        if self.scope_proc and self.scope_proc.poll() is None:
            self._log('停止示波器...')
            self.scope_proc.terminate()
            threading.Thread(target=self._wait_kill_scope, daemon=True).start()
        else:
            messagebox.showinfo('提示', '示波器未运行')
    
    def _wait_kill_scope(self):
        """等待并强制结束示波器进程"""
        if self.scope_proc:
            try:
                self.scope_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._log('强制结束示波器进程')
                self.scope_proc.kill()


def main():
    app = App()
    app.mainloop()


if __name__ == '__main__':
    main()
