#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
video_processor.py - 视频处理工具
"""

import os
import csv
from datetime import datetime
import cv2
import numpy as np


def compose_video_from_png(png_dir: str, video_out: str, fps: int = 30) -> tuple[bool, str]:
    """从 PNG 序列合成视频"""
    try:
        images = sorted([f for f in os.listdir(png_dir) if f.lower().endswith('.png')])
        if not images:
            return False, f"在 {png_dir} 中未找到 PNG 文件"
        
        first = cv2.imread(os.path.join(png_dir, images[0]), cv2.IMREAD_GRAYSCALE)
        if first is None:
            return False, f"无法读取第一张图像: {images[0]}"
        
        h, w = first.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_out, fourcc, fps, (w, h), isColor=False)
        
        for name in images:
            img = cv2.imread(os.path.join(png_dir, name), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape != (h, w):
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
            writer.write(img)
        
        writer.release()
        return True, f"视频已保存到 {video_out}"
    except Exception as e:
        return False, f"视频合成失败: {e}"


def _parse_iso(s: str) -> float:
    """将 ISO 字符串转为 POSIX 秒"""
    try:
        dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    return dt.timestamp()


def align_frames_and_logs(frames_csv: str, logs_csv: str, out_csv: str = "aligned.csv") -> tuple[bool, str]:
    """对齐帧和日志"""
    try:
        frames = []
        logs = []
        
        # 读取帧索引
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
        
        # 读取日志
        log_headers = []
        with open(logs_csv, 'r', encoding='utf-8', newline='') as f:
            rdr = csv.DictReader(f)
            log_headers = rdr.fieldnames or []
            for row in rdr:
                try:
                    ts_host = _parse_iso(row.get('host_recv_iso', ''))
                    log_entry = {
                        'stm32_ts_us': int(row.get('stm32_ts_us', '-1')),
                        'host_iso': row.get('host_recv_iso', ''),
                        'host_ts': ts_host,
                        'log_text_utf8': row.get('log_text_utf8', ''),
                    }
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
            lo, hi = 0, len(logs) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if logs[mid]['host_ts'] < ts_host:
                    lo = mid + 1
                else:
                    hi = mid
            cand_idx = max(0, min(lo, len(logs) - 1))
            cand = logs[cand_idx]
            if cand_idx > 0 and abs(logs[cand_idx - 1]['host_ts'] - ts_host) < abs(cand['host_ts'] - ts_host):
                cand = logs[cand_idx - 1]
            return cand
        
        # 收集自定义变量列名
        custom_var_names = []
        for key in log_headers:
            if key not in ['stm32_ts_us', 'host_recv_iso', 'log_text_utf8', 'log_hex']:
                custom_var_names.append(key)
        
        with open(out_csv, 'w', encoding='utf-8', newline='') as f:
            wtr = csv.writer(f)
            header = ["frame_id", "png_path", "frame_host_iso", "h", "w",
                      "log_host_iso", "log_stm32_ts_us", "log_text_utf8", "host_dt_diff_ms"]
            header.extend(custom_var_names)
            wtr.writerow(header)
            
            for fr in frames:
                lg = find_nearest_log(fr['host_ts'])
                if lg is None:
                    row = [fr['frame_id'], fr['png_path'], fr['host_iso'], fr['h'], fr['w'],
                           '', '', '', '']
                    row.extend([''] * len(custom_var_names))
                    wtr.writerow(row)
                else:
                    diff_ms = (fr['host_ts'] - lg['host_ts']) * 1000.0
                    row = [fr['frame_id'], fr['png_path'], fr['host_iso'], fr['h'], fr['w'],
                           lg['host_iso'], lg['stm32_ts_us'], lg['log_text_utf8'], f"{diff_ms:.3f}"]
                    for var_name in custom_var_names:
                        row.append(lg.get(var_name, ''))
                    wtr.writerow(row)
        
        return True, f"对齐的 CSV 已保存到 {out_csv}"
    except Exception as e:
        return False, f"对齐失败: {e}"
