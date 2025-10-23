#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_udp_client.py - UDP上位机测试客户端

模拟STM32发送不同类型的帧，用于测试上位机功能
"""

import socket
import struct
import time
import numpy as np

HOST = '127.0.0.1'
PORT = 5005

def send_gray_image(sock, height=60, width=120):
    """发送灰度图像帧 [0x01][H][W][pixels...]"""
    frame_type = 0x01
    
    # 生成测试图像：渐变条纹
    img = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        img[i, :] = int(255 * i / height)
    
    # 构造帧
    data = bytes([frame_type, height, width]) + img.tobytes()
    sock.sendto(data, (HOST, PORT))
    print(f"[INFO] Sent gray image: {height}x{width}, {len(data)} bytes")

def send_binary_image(sock, height=60, width=120):
    """发送压缩二值图像帧 [0x03][H][W][compressed...]"""
    frame_type = 0x03
    
    # 生成测试图像：棋盘格
    img = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            if (i // 10 + j // 10) % 2 == 0:
                img[i, j] = 1
    
    # 压缩：8像素 -> 1字节
    pixel_count = height * width
    compressed_size = (pixel_count + 7) // 8
    compressed = bytearray(compressed_size)
    
    for idx in range(pixel_count):
        if img.flat[idx] > 0:
            byte_idx = idx // 8
            bit_idx = idx % 8
            compressed[byte_idx] |= (1 << (7 - bit_idx))
    
    # 构造帧
    data = bytes([frame_type, height, width]) + bytes(compressed)
    sock.sendto(data, (HOST, PORT))
    print(f"[INFO] Sent binary image: {height}x{width}, {len(data)} bytes (compressed from {pixel_count})")

def send_log(sock, message: str):
    """发送日志帧 [0x02][LEN][TS_0..TS_7][payload...]"""
    frame_type = 0x02
    payload = message.encode('utf-8')
    length = len(payload)
    
    if length > 255:
        print(f"[WARN] Log message too long: {length} bytes, truncating to 255")
        payload = payload[:255]
        length = 255
    
    # 模拟STM32时间戳（微秒）
    timestamp_us = int(time.time() * 1e6) % (2**64)
    ts_bytes = struct.pack('<Q', timestamp_us)
    
    # 构造帧
    data = bytes([frame_type, length]) + ts_bytes + payload
    sock.sendto(data, (HOST, PORT))
    print(f"[INFO] Sent log: '{message}' ({length} bytes)")

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    print("=" * 60)
    print("UDP测试客户端 - 模拟STM32发送数据")
    print(f"目标地址: {HOST}:{PORT}")
    print("=" * 60)
    print()
    
    try:
        # 测试1：发送灰度图像
        print("[TEST 1] 发送灰度图像...")
        for i in range(5):
            send_gray_image(sock, height=60, width=120)
            time.sleep(0.1)
        
        time.sleep(1)
        
        # 测试2：发送二值图像
        print("\n[TEST 2] 发送压缩二值图像...")
        for i in range(5):
            send_binary_image(sock, height=60, width=120)
            time.sleep(0.1)
        
        time.sleep(1)
        
        # 测试3：发送日志
        print("\n[TEST 3] 发送日志消息...")
        logs = [
            "System initialized",
            "Camera started",
            "Image processing: OK",
            "Speed: 1.23 m/s",
            "Temperature: 45.6 C"
        ]
        for log in logs:
            send_log(sock, log)
            time.sleep(0.2)
        
        time.sleep(1)
        
        # 测试4：混合发送
        print("\n[TEST 4] 混合发送（模拟真实场景）...")
        for i in range(20):
            if i % 5 == 0:
                send_log(sock, f"Frame {i}: Processing...")
            
            if i % 2 == 0:
                send_gray_image(sock, height=60, width=120)
            else:
                send_binary_image(sock, height=60, width=120)
            
            time.sleep(0.05)  # 20fps
        
        print("\n" + "=" * 60)
        print("测试完成！")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n[INFO] 测试中断")
    finally:
        sock.close()

if __name__ == '__main__':
    main()
