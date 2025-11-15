#!/bin/bash
# UDP GUI 启动脚本

echo "正在启动 UDP GUI..."

# 检查是否在 conda 环境中
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "激活 conda 环境..."
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate smartcar-udp-host
fi

# 检查 ttkbootstrap 是否安装
python -c "import ttkbootstrap" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "正在安装 ttkbootstrap..."
    pip install ttkbootstrap
fi

# 启动 GUI
echo "启动 UDP GUI..."
python src/udp_gui.py