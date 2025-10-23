#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_stm32_protocol.py - STM32协议测试客户端

模拟STM32通过WiFi模块发送图像数据，测试新的协议格式：
[帧头] + [图像数据] + [帧尾]
"""

import socket
import time
import numpy as np

HOST = '127.0.0.1'
PORT = 5005

# STM32 WiFi模块使用的帧头帧尾
FRAME_HEADER = bytes.fromhex('A0FFFFA0')  # 0xa0, 0xff, 0xff, 0xa0
FRAME_FOOTER = bytes.fromhex('B0B00A0D')  # 0xb0, 0xb0, 0x0a, 0x0d

def send_gray_image_stm32_format(sock, height=60, width=120):
    """
    发送灰度图像 - STM32格式 (固定尺寸模式)
    格式: [帧头4字节] + [纯像素数据] + [帧尾4字节]
    注意: 不包含H/W字段，上位机需要手动配置图像尺寸
    """
    # 生成测试图像：渐变条纹
    img = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        img[i, :] = int(255 * i / height)
    
    # 构造数据包 (只有像素数据)
    packet = FRAME_HEADER + img.tobytes() + FRAME_FOOTER
    
    sock.sendto(packet, (HOST, PORT))
    print(f"[INFO] Sent STM32 format gray image (FIXED SIZE): {height}x{width}")
    print(f"       Header: {FRAME_HEADER.hex()}")
    print(f"       Image Data: {len(img.tobytes())} bytes (pure pixels)")
    print(f"       Footer: {FRAME_FOOTER.hex()}")
    print(f"       Total: {len(packet)} bytes")
    print(f"       上位机配置: 固定尺寸 H={height}, W={width}")

def send_gray_image_2byte_hw(sock, height=256, width=320):
    """
    发送灰度图像 - 2字节H/W格式 (动态解析模式)
    格式: [帧头] + [H-2字节小端] + [W-2字节小端] + [像素数据] + [帧尾]
    """
    # 生成测试图像：棋盘格
    img = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            if (i // 32 + j // 32) % 2 == 0:
                img[i, j] = 200
            else:
                img[i, j] = 50
    
    # 构造数据包 (小端序)
    h_bytes = height.to_bytes(2, 'little')
    w_bytes = width.to_bytes(2, 'little')
    image_data = h_bytes + w_bytes + img.tobytes()
    packet = FRAME_HEADER + image_data + FRAME_FOOTER
    
    sock.sendto(packet, (HOST, PORT))
    print(f"[INFO] Sent 2-byte H/W image (DYNAMIC): {height}x{width}")
    print(f"       Header: {FRAME_HEADER.hex()}")
    print(f"       H (little-endian): {h_bytes.hex()}")
    print(f"       W (little-endian): {w_bytes.hex()}")
    print(f"       Footer: {FRAME_FOOTER.hex()}")
    print(f"       Total: {len(packet)} bytes")
    print(f"       上位机配置: 动态解析, H=2字节小端, W=2字节小端")

def send_compressed_binary_image(sock, height=60, width=120):
    """
    发送压缩二值图像 - 8个像素压缩成1字节 (固定尺寸模式)
    格式: [帧头] + [压缩像素数据] + [帧尾]
    注意: 不包含H/W字段
    """
    # 生成二值图像：圆形
    img = np.zeros((height, width), dtype=np.uint8)
    center_y, center_x = height // 2, width // 2
    radius = min(height, width) // 3
    
    for i in range(height):
        for j in range(width):
            if (i - center_y)**2 + (j - center_x)**2 <= radius**2:
                img[i, j] = 1
    
    # 压缩：8个像素 -> 1字节
    pixel_count = height * width
    compressed_bytes = (pixel_count + 7) // 8
    compressed = bytearray(compressed_bytes)
    
    for idx in range(pixel_count):
        if img.flat[idx] > 0:
            byte_idx = idx // 8
            bit_idx = idx % 8
            compressed[byte_idx] |= (1 << (7 - bit_idx))
    
    # 构造数据包 (只有压缩数据)
    packet = FRAME_HEADER + bytes(compressed) + FRAME_FOOTER
    
    sock.sendto(packet, (HOST, PORT))
    print(f"[INFO] Sent compressed binary image (FIXED SIZE): {height}x{width}")
    print(f"       Compressed: {pixel_count} pixels -> {compressed_bytes} bytes")
    print(f"       Total: {len(packet)} bytes")
    print(f"       上位机配置: 固定尺寸 H={height}, W={width}")

def main():
    print("=" * 60)
    print("STM32 协议测试客户端")
    print("=" * 60)
    print(f"\n目标地址: {HOST}:{PORT}")
    print(f"帧头: {FRAME_HEADER.hex().upper()}")
    print(f"帧尾: {FRAME_FOOTER.hex().upper()}")
    print("\n请确保上位机GUI已启动并配置:")
    print("  【固定尺寸模式】(STM32默认)")
    print("  1. 启用图像帧自定义格式")
    print("  2. 帧头设置为: A0FFFFA0")
    print("  3. 帧尾设置为: B0B00A0D")
    print("  4. 尺寸解析模式: 固定尺寸")
    print("  5. 图像尺寸: H=60, W=120 (根据测试调整)")
    print("\n  【动态解析模式】(带H/W字段)")
    print("  1-3同上")
    print("  4. 尺寸解析模式: 动态解析")
    print("  5. H/W字段: 根据测试配置(1-4字节)")
    print("\n" + "=" * 60 + "\n")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    try:
        while True:
            print("\n请选择测试类型:")
            print("  1. STM32灰度图像 (60x120, 固定尺寸)")
            print("  2. STM32灰度图像 (80x160, 固定尺寸)")
            print("  3. 动态解析图像 (256x320, 2字节H/W)")
            print("  4. 压缩二值图像 (60x120, 固定尺寸)")
            print("  5. 连续发送模式")
            print("  q. 退出")
            
            choice = input("\n输入选择: ").strip()
            
            if choice == '1':
                send_gray_image_stm32_format(sock, 60, 120)
            elif choice == '2':
                send_gray_image_stm32_format(sock, 80, 160)
            elif choice == '3':
                send_gray_image_2byte_hw(sock, 256, 320)
            elif choice == '4':
                send_compressed_binary_image(sock, 60, 120)
            elif choice == '5':
                print("\n开始连续发送 (Ctrl+C停止)...")
                frame_count = 0
                try:
                    while True:
                        # 交替发送不同图像
                        if frame_count % 2 == 0:
                            send_gray_image_stm32_format(sock, 60, 120)
                        else:
                            send_gray_image_stm32_format(sock, 80, 160)
                        frame_count += 1
                        time.sleep(0.1)  # 10 FPS
                except KeyboardInterrupt:
                    print(f"\n已发送 {frame_count} 帧")
            elif choice.lower() == 'q':
                break
            else:
                print("无效选择，请重试")
                
    except KeyboardInterrupt:
        print("\n\n程序已中断")
    finally:
        sock.close()
        print("连接已关闭")

if __name__ == '__main__':
    main()
