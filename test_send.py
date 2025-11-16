#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试UDP发送功能的脚本
"""

import socket
import time
import threading

def test_receiver():
    """测试接收器 - 接收发送的数据"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', 8080))
    sock.settimeout(1.0)
    
    print("[测试接收器] 已启动，监听端口 8080")
    
    try:
        while True:
            try:
                data, addr = sock.recvfrom(65535)
                print(f"\n[接收] 从 {addr} 收到 {len(data)} 字节:")
                print(f"  Hex: {data.hex()}")
                try:
                    text = data.decode('utf-8', errors='replace')
                    print(f"  Text: {text}")
                except:
                    pass
            except socket.timeout:
                continue
            except KeyboardInterrupt:
                break
    finally:
        sock.close()
        print("\n[测试接收器] 已停止")

if __name__ == '__main__':
    print("=" * 60)
    print("UDP 发送功能测试工具")
    print("=" * 60)
    print("\n使用说明:")
    print("1. 运行此脚本启动测试接收器")
    print("2. 启动 udp_gui.py")
    print("3. 在GUI的'发送'标签页中:")
    print("   - 目标IP: 127.0.0.1")
    print("   - 目标端口: 8080")
    print("   - 点击'连接'按钮")
    print("   - 在'数据编辑'中输入数据")
    print("   - 点击'单次发送'或'启动定时发送'")
    print("4. 观察此窗口的接收输出")
    print("\n按 Ctrl+C 退出\n")
    print("=" * 60)
    
    try:
        test_receiver()
    except KeyboardInterrupt:
        print("\n程序已退出")
