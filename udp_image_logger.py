#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
udp_image_logger.py

Windows 上位机：
- 监听 UDP 端口
- 解析自定义帧：
  * 0x01: [0x01][H][W][pixels...]   (8-bit 灰度或二值)
  * 0x02: [0x02][LEN][TS_0..TS_7][payload...]  (LEN=日志内容字节数，TS 小端 64bit 微秒)
- 实时显示图像，可选保存 PNG
- 将日志及其时间戳写入 CSV
- 支持将 PNG 序列合成为 MP4
- 图像与日志按 STM32 时间戳对齐（本机记录接收时刻以备分析）

注意：默认假设单帧 UDP 数据完整承载一帧（推荐在 STM32 侧保证单帧<=MTU 或自行做分片重组）。如需跨包重组，可扩展 FrameAssembler。
"""

import argparse
import csv
import os
import socket
import struct
import sys
import threading
import time
from datetime import datetime
from typing import Optional, Tuple

import cv2
import numpy as np

# ---------------------- 辅助函数 ----------------------

def sanitize_csv_text(text: str) -> str:
    """清理CSV文本,移除会导致读取问题的特殊字符
    
    Args:
        text: 原始文本
    
    Returns:
        清理后的文本,移除NULL、EOF等特殊字符
    """
    if not text:
        return text
    
    # 移除NULL字符(0x00)和EOF字符(0x1A)
    # 这些字符会导致C++的std::getline提前终止
    text = text.replace('\x00', '')  # NULL
    text = text.replace('\x1A', '')  # EOF/SUB (Ctrl+Z)
    
    # 可选:也移除其他控制字符(保留换行、制表等常用字符)
    # 移除 0x01-0x08, 0x0B-0x0C, 0x0E-0x1F (保留 \t=0x09, \n=0x0A, \r=0x0D)
    cleaned = ''.join(c for c in text if ord(c) >= 0x20 or c in '\t\n\r')
    
    return cleaned
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------------------- 配置 ----------------------
DEFAULT_UDP_PORT = 5005
DEFAULT_BIND_IP = "0.0.0.0"
PNG_DIR_DEFAULT = "frames_png"
LOG_CSV_DEFAULT = "logs.csv"
FRAME_INDEX_CSV = "frames_index.csv"  # 记录每帧图像的STM32时间戳与主机接收时间
VIDEO_OUT_DEFAULT = "output.mp4"
DEFAULT_SCOPE_MAX_POINTS = 2000


# ---------------------- 日志示波器（实时） ----------------------
class LogScopeReceiver:
    def __init__(self, ip: str, port: int, index: int, bit: int = -1, max_points: int = DEFAULT_SCOPE_MAX_POINTS):
        self.ip = ip
        self.port = port
        self.index = index
        self.bit = bit
        self.max_points = max_points
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((self.ip, self.port))
        self._sock.settimeout(1.0)
        self._stop = threading.Event()
        self.x = deque(maxlen=max_points)  # seconds (relative to first ts)
        self.y = deque(maxlen=max_points)  # value (0..255) or bit (0/1)
        self._t0: Optional[int] = None  # first STM32 ts_us

    def stop(self):
        self._stop.set()

    def close(self):
        try:
            self._sock.close()
        except Exception:
            pass

    def _append_point(self, ts_us: int, payload: bytes):
        if self.index >= len(payload):
            return
        v = payload[self.index]
        if 0 <= self.bit <= 7:
            v = (v >> self.bit) & 0x01
        if self._t0 is None:
            self._t0 = ts_us
        t_rel = (ts_us - self._t0) / 1e6
        self.x.append(t_rel)
        self.y.append(v)

    def thread_loop(self):
        print(f"[INFO] Scope listening on {self.ip}:{self.port}, index={self.index}, bit={self.bit}")
        try:
            while not self._stop.is_set():
                try:
                    data, _ = self._sock.recvfrom(65535)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not data:
                    continue
                if data[0] != FrameType.LOG:
                    continue
                try:
                    ts_us, payload = parse_log_frame(data)
                    self._append_point(ts_us, payload)
                except Exception:
                    continue
        finally:
            self.close()


def run_scope(ip: str, port: int, index: int, bit: int, max_points: int, ymin: Optional[float], ymax: Optional[float], title: Optional[str]):
    rx = LogScopeReceiver(ip, port, index, bit, max_points)
    th = threading.Thread(target=rx.thread_loop, daemon=True)
    th.start()

    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(10, 4))
    ln, = ax.plot([], [], lw=1.2)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('value')
    if title:
        ax.set_title(title)
    if ymin is not None or ymax is not None:
        ax.set_ylim(bottom=ymin if ymin is not None else 0, top=ymax if ymax is not None else 255 if bit < 0 else 1)

    def init():
        ln.set_data([], [])
        return ln,

    def update(_):
        if len(rx.x) >= 2:
            ax.set_xlim(max(0.0, rx.x[0]), rx.x[-1] if rx.x[-1] > 1 else 1)
        ln.set_data(list(rx.x), list(rx.y))
        return ln,

    ani = FuncAnimation(fig, update, init_func=init, interval=50, blit=True)
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        rx.stop()
        rx.close()

# ---------------------- 工具函数 ----------------------

def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")


def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# ---------------------- 帧解析 ----------------------
class FrameType:
    IMAGE = 0x01
    LOG = 0x02
    BINARY_IMAGE = 0x03  # 二值图（8像素压缩为1字节）


def parse_image_frame(data: bytes) -> Tuple[int, int, np.ndarray]:
    """解析图像帧 [0x01][H][W][pixels...]
    H、W 取 1 字节，像素长度应为 H*W 字节。
    返回: (H, W, img)
    """
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


def parse_log_frame(data: bytes) -> Tuple[int, bytes]:
    """解析日志帧 [0x02][LEN][TS_0..TS_7][payload...]
    LEN 为日志内容字节数（不含类型/LEN/时间戳），TS 为 8 字节小端。
    返回: (timestamp_us, payload)
    """
    if len(data) < 1 + 1 + 8:
        raise ValueError("log frame too short")
    length = data[1]
    if len(data) != 1 + 1 + 8 + length:
        raise ValueError("log frame size mismatch")
    ts_us = struct.unpack_from('<Q', data, 2)[0]
    payload = data[10:10 + length]
    return ts_us, payload


def parse_binary_image_frame(data: bytes) -> Tuple[int, int, np.ndarray]:
    """解析二值图像帧 [0x03][H][W][compressed_pixels...]
    H、W 取 1 字节，像素数据压缩（8像素1字节，MSB优先）
    返回: (H, W, img) - img为0或255的灰度图
    """
    if len(data) < 3:
        raise ValueError("binary image frame too short")
    h = data[1]
    w = data[2]
    pixel_count = h * w
    expected_bytes = 1 + 1 + 1 + ((pixel_count + 7) // 8)  # 向上取整
    if len(data) < expected_bytes:
        raise ValueError(f"binary image frame size mismatch: got {len(data)}, expect {expected_bytes}")
    
    # 解压缩：8像素1字节 -> 8个像素
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


# ---------------------- UDP 监听与处理 ----------------------
class UdpReceiver:
    def __init__(self, ip: str, port: int, 
                 show: bool = True,
                 save_png: bool = False,
                 png_dir: str = PNG_DIR_DEFAULT,
                 log_csv: str = LOG_CSV_DEFAULT,
                 frame_index_csv: str = FRAME_INDEX_CSV):
        self.ip = ip
        self.port = port
        self.show = show
        self.save_png = save_png
        self.png_dir = png_dir
        self.log_csv = log_csv
        self.frame_index_csv = frame_index_csv

        ensure_dir(self.png_dir)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((self.ip, self.port))
        self._sock.settimeout(1.0)
        self._stop = threading.Event()

        # CSV 初始化（修复：检查文件是否存在）
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

        self._frame_counter = 0
        self._last_image_ts_us: Optional[int] = None
        
        # 统计信息
        self._fps_timer = time.time()
        self._fps_frame_count = 0
        self._current_fps = 0.0
        self._total_packets = 0
        self._error_packets = 0

    def close(self):
        try:
            self._sock.close()
        finally:
            try:
                self._log_csv_fp.close()
            except Exception:
                pass
            try:
                self._frame_index_fp.close()
            except Exception:
                pass

    def stop(self):
        self._stop.set()
    
    def _update_fps(self):
        """更新FPS统计"""
        self._fps_frame_count += 1
        elapsed = time.time() - self._fps_timer
        if elapsed >= 1.0:  # 每秒更新一次
            self._current_fps = self._fps_frame_count / elapsed
            self._fps_frame_count = 0
            self._fps_timer = time.time()
    
    def _display_image(self, img: np.ndarray, frame_id: int):
        """实时显示图像，带统计信息和自适应缩放"""
        h, w = img.shape[:2]
        
        # 自适应缩放：如果图像太小，放大显示
        display_img = img.copy()
        scale = 1.0
        if w < 320 or h < 240:
            scale = max(320 / w, 240 / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            display_img = cv2.resize(display_img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # 如果图像太大，缩小显示
        elif w > 1280 or h > 720:
            scale = min(1280 / w, 720 / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            display_img = cv2.resize(display_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 转为彩色以便绘制彩色文字
        if len(display_img.shape) == 2:
            display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
        
        # 绘制统计信息
        error_rate = (self._error_packets / self._total_packets * 100) if self._total_packets > 0 else 0
        info_lines = [
            f"Frame: {frame_id}",
            f"FPS: {self._current_fps:.1f}",
            f"Size: {w}x{h} (Scale: {scale:.2f}x)",
            f"Packets: {self._total_packets}",
            f"Errors: {self._error_packets} ({error_rate:.1f}%)"
        ]
        
        y_offset = 20
        for line in info_lines:
            cv2.putText(display_img, line, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            y_offset += 20
        
        cv2.imshow('UDP Image Stream', display_img)
        # 1ms 响应键盘
        if cv2.waitKey(1) & 0xFF == 27:  # ESC 退出
            print("[INFO] ESC pressed, stopping...")
            self.stop()
            return True
        return False

    def loop(self):
        print(f"[INFO] Listening UDP on {self.ip}:{self.port}")
        try:
            while not self._stop.is_set():
                try:
                    data, addr = self._sock.recvfrom(65535)
                except socket.timeout:
                    continue
                except OSError:
                    break

                self._total_packets += 1
                host_iso = now_iso()
                if not data:
                    continue
                ftype = data[0]
                try:
                    if ftype == FrameType.IMAGE:
                        h, w, img = parse_image_frame(data)
                        self._frame_counter += 1
                        frame_id = self._frame_counter
                        self._update_fps()

                        # 实时显示（带统计信息）
                        if self.show:
                            if self._display_image(img, frame_id):
                                break

                        png_path = ''
                        if self.save_png:
                            png_path = os.path.join(self.png_dir, f"frame_{frame_id:06d}.png")
                            cv2.imwrite(png_path, img)

                        # 可选：图像帧中没有STM32时间戳，这里记录为-1；若你在图像帧中另加TS字段，可在此提取
                        stm32_ts_us = -1
                        self._frame_index_writer.writerow([frame_id, stm32_ts_us, host_iso, png_path, h, w])
                        self._frame_index_fp.flush()

                    elif ftype == FrameType.BINARY_IMAGE:
                        h, w, img = parse_binary_image_frame(data)
                        self._frame_counter += 1
                        frame_id = self._frame_counter
                        self._update_fps()

                        # 实时显示
                        if self.show:
                            if self._display_image(img, frame_id):
                                break

                        png_path = ''
                        if self.save_png:
                            png_path = os.path.join(self.png_dir, f"frame_{frame_id:06d}.png")
                            cv2.imwrite(png_path, img)

                        stm32_ts_us = -1
                        self._frame_index_writer.writerow([frame_id, stm32_ts_us, host_iso, png_path, h, w])
                        self._frame_index_fp.flush()

                    elif ftype == FrameType.LOG:
                        ts_us, payload = parse_log_frame(data)
                        text_utf8 = ''
                        try:
                            text_utf8 = payload.decode('utf-8', errors='replace')
                            # 清理特殊字符,避免CSV读取问题
                            text_utf8 = sanitize_csv_text(text_utf8)
                        except Exception:
                            text_utf8 = ''
                        text_hex = payload.hex()
                        self._log_writer.writerow([ts_us, host_iso, text_hex, text_utf8])
                        self._log_csv_fp.flush()
                    else:
                        print(f"[WARN] Unknown frame type: 0x{ftype:02X} (len={len(data)})")
                        self._error_packets += 1
                except Exception as e:
                    print(f"[ERROR] Parse error: {e}; from {addr}, len={len(data)})")
                    self._error_packets += 1
        finally:
            self.close()
            if self.show:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass


# ---------------------- PNG -> MP4 合成 ----------------------

def compose_video_from_png(png_dir: str, video_out: str, fps: int = 30):
    images = sorted([f for f in os.listdir(png_dir) if f.lower().endswith('.png')])
    if not images:
        print(f"[WARN] No PNGs found in {png_dir}")
        return
    first = cv2.imread(os.path.join(png_dir, images[0]), cv2.IMREAD_GRAYSCALE)
    if first is None:
        print(f"[ERROR] Cannot read first image: {images[0]}")
        return
    h, w = first.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # 修复：灰度视频应该写入灰度图，不需要转BGR
    writer = cv2.VideoWriter(video_out, fourcc, fps, (w, h), isColor=False)
    for name in images:
        img = cv2.imread(os.path.join(png_dir, name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if img.shape != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
        writer.write(img)  # 直接写入灰度图
    writer.release()
    print(f"[INFO] Video saved to {video_out}")


# ---------------------- 对齐：按主机接收时间 ----------------------
def _parse_iso(s: str) -> float:
    """将 ISO 字符串转为 POSIX 秒(float)。"""
    try:
        dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        # 兜底：没有微秒
        dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    return dt.timestamp()


def align_frames_and_logs(frames_csv: str, logs_csv: str, out_csv: str = "aligned.csv"):
    """按主机接收时间(host_recv_iso)进行最近邻对齐，生成 aligned.csv。
    输出字段：
    frame_id, png_path, frame_host_iso, h, w, log_host_iso, log_stm32_ts_us, log_text_utf8, host_dt_diff_ms
    """
    frames = []
    logs = []

    # 读 frames_index.csv
    with open(frames_csv, 'r', encoding='utf-8', newline='') as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                ts = _parse_iso(row.get('host_recv_iso', ''))
                frames.append({
                    'frame_id': int(row.get('frame_id', '0')),
                    'png_path': row.get('png_path', ''),
                    'host_iso': row.get('host_recv_iso', ''),
                    'host_ts': ts,
                    'h': int(row.get('h', '0')),
                    'w': int(row.get('w', '0')),
                })
            except Exception:
                continue

    # 读 logs.csv
    log_headers = []  # 保存logs.csv的所有列名
    with open(logs_csv, 'r', encoding='utf-8', newline='') as f:
        rdr = csv.DictReader(f)
        log_headers = rdr.fieldnames or []  # 获取所有列名
        for row in rdr:
            try:
                ts_host = _parse_iso(row.get('host_recv_iso', ''))
                log_entry = {
                    'stm32_ts_us': int(row.get('stm32_ts_us', '-1')),
                    'host_iso': row.get('host_recv_iso', ''),
                    'host_ts': ts_host,
                    'log_text_utf8': row.get('log_text_utf8', ''),
                }
                # 保存所有额外字段(自定义变量)
                for key in row.keys():
                    if key not in ['stm32_ts_us', 'host_recv_iso', 'log_text_utf8', 'log_hex']:
                        log_entry[key] = row.get(key, '')
                logs.append(log_entry)
            except Exception:
                continue

    logs.sort(key=lambda x: x['host_ts'])

    def find_nearest_log(ts_host: float):
        if not logs:
            return None
        # 二分搜索最近邻
        lo, hi = 0, len(logs) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if logs[mid]['host_ts'] < ts_host:
                lo = mid + 1
            else:
                hi = mid
        cand_idx = max(0, min(lo, len(logs) - 1))
        cand = logs[cand_idx]
        # 比较前一个是否更近
        if cand_idx > 0 and abs(logs[cand_idx - 1]['host_ts'] - ts_host) < abs(cand['host_ts'] - ts_host):
            cand = logs[cand_idx - 1]
        return cand

    # 收集所有自定义变量列名(除了固定字段)
    custom_var_names = []
    for key in log_headers:
        if key not in ['stm32_ts_us', 'host_recv_iso', 'log_text_utf8', 'log_hex']:
            custom_var_names.append(key)
    
    with open(out_csv, 'w', encoding='utf-8', newline='') as f:
        wtr = csv.writer(f)
        # 动态生成表头:固定列 + 自定义变量列
        header = ["frame_id", "png_path", "frame_host_iso", "h", "w",
                  "log_host_iso", "log_stm32_ts_us", "log_text_utf8", "host_dt_diff_ms"]
        header.extend(custom_var_names)  # 添加自定义变量列
        wtr.writerow(header)
        
        for fr in frames:
            lg = find_nearest_log(fr['host_ts'])
            if lg is None:
                # 没有匹配的日志
                row = [fr['frame_id'], fr['png_path'], fr['host_iso'], fr['h'], fr['w'],
                       '', '', '', '']
                row.extend([''] * len(custom_var_names))  # 空的自定义变量
                wtr.writerow(row)
            else:
                diff_ms = (fr['host_ts'] - lg['host_ts']) * 1000.0
                row = [fr['frame_id'], fr['png_path'], fr['host_iso'], fr['h'], fr['w'],
                       lg['host_iso'], lg['stm32_ts_us'], lg['log_text_utf8'], f"{diff_ms:.3f}"]
                # 添加自定义变量的值
                for var_name in custom_var_names:
                    row.append(lg.get(var_name, ''))
                wtr.writerow(row)
    print(f"[INFO] Aligned CSV saved to {out_csv}")

# ---------------------- CLI ----------------------

def main():
    parser = argparse.ArgumentParser(description="UDP image/log host for STM32 frames")
    subparsers = parser.add_subparsers(dest='cmd')

    # run
    p_run = subparsers.add_parser('run', help='run UDP listener')
    p_run.add_argument('--ip', default=DEFAULT_BIND_IP, help='bind ip (default 0.0.0.0)')
    p_run.add_argument('--port', type=int, default=DEFAULT_UDP_PORT, help='udp port (default 5005)')
    p_run.add_argument('--show', action='store_true', help='show image window')
    p_run.add_argument('--save-png', action='store_true', help='save frames as PNG')
    p_run.add_argument('--png-dir', default=PNG_DIR_DEFAULT, help='png output directory')
    p_run.add_argument('--log-csv', default=LOG_CSV_DEFAULT, help='log csv path')
    p_run.add_argument('--frame-index-csv', default=FRAME_INDEX_CSV, help='frame index csv path')

    # compose
    p_vid = subparsers.add_parser('video', help='compose PNGs to MP4')
    p_vid.add_argument('--png-dir', default=PNG_DIR_DEFAULT)
    p_vid.add_argument('--out', default=VIDEO_OUT_DEFAULT)
    p_vid.add_argument('--fps', type=int, default=30)

    # align
    p_align = subparsers.add_parser('align', help='align frames_index.csv and logs.csv by host time')
    p_align.add_argument('--frames-csv', default=FRAME_INDEX_CSV, help='frames_index.csv path')
    p_align.add_argument('--logs-csv', default=LOG_CSV_DEFAULT, help='logs.csv path')
    p_align.add_argument('--out-csv', default='aligned.csv', help='output csv path')

    # scope
    p_scope = subparsers.add_parser('scope', help='plot i-th byte (and optional bit) from log payload vs STM32 timestamp')
    p_scope.add_argument('--ip', default=DEFAULT_BIND_IP, help='bind ip (default 0.0.0.0)')
    p_scope.add_argument('--port', type=int, default=DEFAULT_UDP_PORT, help='udp port (default 5005)')
    p_scope.add_argument('--index', type=int, required=True, help='byte index in log payload (0-based)')
    p_scope.add_argument('--bit', type=int, default=-1, help='bit index in selected byte (0..7), -1 to plot full byte')
    p_scope.add_argument('--max-points', type=int, default=DEFAULT_SCOPE_MAX_POINTS, help='max points in rolling window')
    p_scope.add_argument('--ymin', type=float, default=None, help='y-axis min')
    p_scope.add_argument('--ymax', type=float, default=None, help='y-axis max')
    p_scope.add_argument('--title', type=str, default=None, help='plot title')

    args = parser.parse_args()

    if args.cmd == 'run':
        recv = UdpReceiver(ip=args.ip, port=args.port, show=args.show, save_png=args.save_png,
                           png_dir=args.png_dir, log_csv=args.log_csv, frame_index_csv=args.frame_index_csv)
        try:
            recv.loop()
        except KeyboardInterrupt:
            print("\n[INFO] KeyboardInterrupt, exiting...")
        finally:
            recv.stop()
    elif args.cmd == 'video':
        compose_video_from_png(args.png_dir, args.out, args.fps)
    elif args.cmd == 'align':
        # 使用默认文件名或用户指定
        frames_csv = getattr(args, 'frames_csv', FRAME_INDEX_CSV)
        logs_csv = getattr(args, 'logs_csv', LOG_CSV_DEFAULT)
        out_csv = getattr(args, 'out_csv', 'aligned.csv')
        align_frames_and_logs(frames_csv, logs_csv, out_csv)
    elif args.cmd == 'scope':
        run_scope(args.ip, args.port, args.index, args.bit, args.max_points, args.ymin, args.ymax, args.title)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
