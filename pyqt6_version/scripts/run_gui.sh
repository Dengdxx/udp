#!/bin/bash
# Linux/Mac shell script to run PyQt6 GUI

echo "========================================"
echo " UDP 上位机 PyQt6 版本"
echo "========================================"
echo ""

# 检查 conda 是否可用
if ! command -v conda &> /dev/null; then
    echo "[错误] 未找到 conda 命令"
    echo "请确保 Anaconda/Miniconda 已安装并添加到 PATH"
    exit 1
fi

echo "[1/3] 激活 conda 环境: smartcar-udp-pyqt6"
eval "$(conda shell.bash hook)"
conda activate smartcar-udp-pyqt6

if [ $? -ne 0 ]; then
    echo ""
    echo "[错误] 环境 'smartcar-udp-pyqt6' 不存在"
    echo ""
    echo "请先创建环境:"
    echo "  cd pyqt6_version"
    echo "  conda env create -f environment.yml"
    echo ""
    exit 1
fi

echo "[2/3] 检查依赖..."
python -c "import PyQt6; import cv2; import numpy" 2>/dev/null

if [ $? -ne 0 ]; then
    echo ""
    echo "[错误] 缺少必要的 Python 包"
    echo "正在安装依赖..."
    pip install -r requirements.txt
    
    if [ $? -ne 0 ]; then
        echo "[错误] 依赖安装失败"
        exit 1
    fi
fi

echo "[3/3] 启动 GUI..."
echo ""
python src/udp_gui_qt.py

if [ $? -ne 0 ]; then
    echo ""
    echo "[错误] 程序异常退出"
fi
