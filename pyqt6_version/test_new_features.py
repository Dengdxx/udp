#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_new_features.py - 测试新功能：二值图(自定义8位) 和 MP4 录制

使用方法：
1. 启动此测试脚本：python test_new_features.py
2. 启动 UDP 上位机，配置：
   - IP: 127.0.0.1
   - Port: 8080
   - 自定义图像帧启用
   - 帧头: AAAA
   - 帧尾: BBBB
   - 编码格式: 二值图(自定义8位)
   - 固定尺寸: 60x80
3. 观察上位机显示的彩色索引图
4. 点击"开始录制"按钮测试 MP4 录制功能
"""

import socket
import time
import numpy as np

def create_test_custom_8bit_frame(frame_id):
    """创建测试用的自定义8位二值图帧"""
    h, w = 60, 80
    
    # 创建测试图像
    img = np.zeros((h, w), dtype=np.uint8)
    
    # 在不同区域填充不同的值
    # 0 = 黑色
    img[0:15, :] = 0
    
    # 255 = 白色
    img[15:30, :] = 255
    
    # 1-254 = 彩色（根据 HSV 色轮映射）
    # 创建渐变色带
    for i in range(30, 60):
        # 从 1 到 254 渐变
        value = int(1 + (i - 30) * (253 / 30))
        img[i, :] = value
    
    # 添加一些图案
    # 在彩色区域画一个圆形
    center_x, center_y = 45, 40
    radius = 10
    for y in range(h):
        for x in range(w):
            if (x - center_y)**2 + (y - center_x)**2 < radius**2:
                # 圆形内部用另一种颜色
                img[y, x] = min(254, img[y, x] + 50)
    
    # 添加动画效果（随帧变化）
    shift = (frame_id * 5) % 254
    img[30:60, :] = np.clip(img[30:60, :] + shift, 1, 254).astype(np.uint8)
    
    return img.tobytes()

def send_test_frames():
    """发送测试帧到 UDP 上位机"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target = ('127.0.0.1', 8080)
    
    header = bytes.fromhex('AAAA')
    footer = bytes.fromhex('BBBB')
    
    print("开始发送测试帧...")
    print("配置上位机:")
    print("  - 自定义图像帧: 启用")
    print("  - 帧头: AAAA")
    print("  - 帧尾: BBBB")
    print("  - 编码格式: 二值图(自定义8位)")
    print("  - 固定尺寸: 60x80")
    print("\n按 Ctrl+C 停止\n")
    
    frame_id = 0
    try:
        while True:
            # 创建测试帧
            pixel_data = create_test_custom_8bit_frame(frame_id)
            
            # 组装数据包: [帧头][像素数据][帧尾]
            packet = header + pixel_data + footer
            
            # 发送
            sock.sendto(packet, target)
            
            frame_id += 1
            if frame_id % 10 == 0:
                print(f"已发送 {frame_id} 帧... (数据包大小: {len(packet)} 字节)")
            
            # 控制帧率 (约 15 FPS)
            time.sleep(1.0 / 15)
    
    except KeyboardInterrupt:
        print(f"\n停止发送。共发送 {frame_id} 帧")
    finally:
        sock.close()

def send_simple_gradient_frames():
    """发送简单的渐变色测试帧"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target = ('127.0.0.1', 8080)
    
    header = bytes.fromhex('AAAA')
    footer = bytes.fromhex('BBBB')
    
    h, w = 60, 80
    
    print("发送简单渐变色测试...")
    print("这个测试会显示完整的色轮效果\n")
    
    frame_id = 0
    try:
        while True:
            # 创建从左到右的渐变
            img = np.zeros((h, w), dtype=np.uint8)
            
            # 上半部分：0 到 255 的渐变（会看到黑→彩色→白）
            for x in range(w):
                value = int(x * 255 / (w - 1))
                img[0:h//2, x] = value
            
            # 下半部分：只有彩色部分 (1-254)
            for x in range(w):
                value = int(1 + x * 253 / (w - 1))
                img[h//2:h, x] = value
            
            pixel_data = img.tobytes()
            packet = header + pixel_data + footer
            
            sock.sendto(packet, target)
            
            frame_id += 1
            if frame_id % 30 == 0:
                print(f"已发送 {frame_id} 帧")
            
            time.sleep(1.0 / 10)
    
    except KeyboardInterrupt:
        print(f"\n停止发送。共发送 {frame_id} 帧")
    finally:
        sock.close()

if __name__ == '__main__':
    import sys
    
    print("=" * 60)
    print("测试新功能：二值图(自定义8位) 编码格式")
    print("=" * 60)
    print("\n选择测试模式:")
    print("1. 动画测试（动态彩色效果）")
    print("2. 渐变测试（色轮展示）")
    
    choice = input("\n请选择 (1/2，默认=1): ").strip() or '1'
    
    print("\n提示: 录制 MP4 测试步骤:")
    print("  1. 在上位机中点击「启动监听」")
    print("  2. 看到彩色图像后，点击「⏺ 开始录制」")
    print("  3. 等待几秒后点击「⏹ 停止录制」")
    print("  4. 查看生成的 MP4 文件\n")
    
    input("按 Enter 开始发送...")
    
    if choice == '2':
        send_simple_gradient_frames()
    else:
        send_test_frames()
