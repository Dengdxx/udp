@echo off
chcp 65001 >nul
REM Windows batch script to run PyQt6 GUI
REM 激活 conda 环境并运行程序

echo ========================================
echo  UDP 上位机 PyQt6 版本
echo ========================================
echo.

REM 检查 conda 是否可用
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [错误] 未找到 conda 命令
    echo 请确保 Anaconda/Miniconda 已安装并添加到 PATH
    pause
    exit /b 1
)

echo [1/3] 激活 conda 环境: smartcar-udp-pyqt6
call conda activate smartcar-udp-pyqt6
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [错误] 环境 'smartcar-udp-pyqt6' 不存在
    echo.
    echo 请先创建环境:
    echo   cd pyqt6_version
    echo   conda env create -f environment.yml
    echo.
    pause
    exit /b 1
)

echo [2/3] 检查依赖...
python -c "import PyQt6; import cv2; import numpy" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [错误] 缺少必要的 Python 包
    echo 正在安装依赖...
    pip install -r ..\requirements.txt
    if %ERRORLEVEL% NEQ 0 (
        echo [错误] 依赖安装失败
        pause
        exit /b 1
    )
)

echo [3/3] 启动 GUI...
echo.
python ..\src\udp_gui_qt.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [错误] 程序异常退出
    pause
)
