@echo off
chcp 65001 >nul
REM ========================================
REM UDP上位机GUI一键启动脚本
REM ========================================

echo.
echo ========================================
echo    UDP图传上位机GUI启动中...
echo ========================================
echo.

REM 激活conda环境
echo [1/2] 激活conda环境: smartcar-udp-host
call conda activate smartcar-udp-host
if errorlevel 1 (
    echo [错误] 无法激活conda环境 smartcar-udp-host
    echo.
    echo 请先创建conda环境:
    echo   conda env create -f environment.yml
    echo.
    pause
    exit /b 1
)

echo [成功] conda环境已激活
echo.

REM 启动GUI
echo [2/2] 启动GUI界面...
echo.

python udp_gui.py

echo.
echo ========================================
echo    GUI已关闭
echo ========================================
echo.
pause
