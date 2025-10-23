@echo off
REM UDP上位机GUI启动脚本
REM 自动调用src目录中的Python脚本

echo ========================================
echo    UDP 上位机 GUI
echo ========================================
echo.

REM 检查Python环境
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python！请先安装Python 3.11+
    echo.
    pause
    exit /b 1
)

REM 检查源文件
if not exist "src\udp_gui.py" (
    echo [错误] 找不到 src\udp_gui.py
    echo 请确保在项目根目录运行此脚本
    pause
    exit /b 1
)

echo [信息] 启动UDP上位机GUI...
echo.

REM 启动GUI程序
python src\udp_gui.py

if errorlevel 1 (
    echo.
    echo [错误] 程序异常退出
    pause
)
