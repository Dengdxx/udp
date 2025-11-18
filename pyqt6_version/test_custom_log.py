#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_custom_log.py - 测试自定义日志帧

测试场景：
1. 纯文本模式 - ASCII 文本
2. 纯文本模式 - 二进制数据
3. 标准格式 - 带长度字段
"""

import socket
import time

def test_custom_log_pure_text():
    """测试纯文本模式"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target = ('127.0.0.1', 8080)
    
    header = bytes.fromhex('BB66')
    footer = bytes.fromhex('0D0A')
    
    print("=" * 60)
    print("测试自定义日志帧 - 纯文本模式")
    print("=" * 60)
    print("\n配置上位机:")
    print("  - 自定义日志帧: 启用")
    print("  - 帧头: BB66")
    print("  - 帧尾: 0D0A")
    print("  - 格式: 纯文本")
    print("\n测试数据:")
    
    test_cases = [
        ("Hello World", "纯ASCII文本"),
        ("温度:25.5°C", "UTF-8中文文本"),
        (b"\xAA\xBB\xCC\xDD", "二进制数据(不可打印)"),
        ("Speed: 100 km/h", "传感器数据"),
        (b"\x01\x02\x03\x04\x05", "原始字节序列"),
    ]
    
    for i, (payload, desc) in enumerate(test_cases, 1):
        if isinstance(payload, str):
            payload_bytes = payload.encode('utf-8')
        else:
            payload_bytes = payload
        
        packet = header + payload_bytes + footer
        sock.sendto(packet, target)
        
        print(f"\n{i}. {desc}")
        print(f"   Payload: {payload if isinstance(payload, str) else payload_bytes.hex()}")
        print(f"   包结构: {header.hex()} + {payload_bytes.hex()} + {footer.hex()}")
        print(f"   总长度: {len(packet)} bytes")
        
        time.sleep(1)
    
    print("\n\n✓ 测试完成！")
    print("请在上位机的「原始数据监视器」中查看解码结果。")
    sock.close()

def test_custom_log_standard_format():
    """测试标准格式"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target = ('127.0.0.1', 8080)
    
    header = bytes.fromhex('BB66')
    footer = bytes.fromhex('0D0A')
    
    print("\n" + "=" * 60)
    print("测试自定义日志帧 - 标准格式")
    print("=" * 60)
    print("\n配置上位机:")
    print("  - 自定义日志帧: 启用")
    print("  - 帧头: BB66")
    print("  - 帧尾: 0D0A")
    print("  - 格式: 标准格式")
    print("\n标准格式结构: [0x02][LEN][内容]")
    print("\n测试数据:")
    
    test_cases = [
        ("Test 123", "简单文本"),
        ("LOG: System OK", "带前缀的日志"),
        (b"\xDE\xAD\xBE\xEF", "二进制数据"),
    ]
    
    for i, (payload, desc) in enumerate(test_cases, 1):
        if isinstance(payload, str):
            payload_bytes = payload.encode('utf-8')
        else:
            payload_bytes = payload
        
        # 标准格式: [0x02][长度][内容]
        length = len(payload_bytes)
        standard_payload = bytes([0x02, length]) + payload_bytes
        packet = header + standard_payload + footer
        
        sock.sendto(packet, target)
        
        print(f"\n{i}. {desc}")
        print(f"   原始内容: {payload if isinstance(payload, str) else payload_bytes.hex()}")
        print(f"   标准格式: 02 {length:02X} {payload_bytes.hex()}")
        print(f"   完整包: {packet.hex()}")
        
        time.sleep(1)
    
    print("\n\n✓ 测试完成！")
    sock.close()

if __name__ == '__main__':
    import sys
    
    print("选择测试模式:")
    print("1. 纯文本模式测试")
    print("2. 标准格式测试")
    print("3. 全部测试")
    
    choice = input("\n请选择 (1/2/3，默认=1): ").strip() or '1'
    
    print("\n提示: 请先在上位机中启动监听！")
    input("按 Enter 开始测试...")
    
    if choice == '1':
        test_custom_log_pure_text()
    elif choice == '2':
        test_custom_log_standard_format()
    elif choice == '3':
        test_custom_log_pure_text()
        time.sleep(2)
        test_custom_log_standard_format()
    else:
        print("无效选择")
