@echo off
chcp 65001 >nul
echo ========================================
echo    智能挂号助手 - 启动脚本
echo ========================================
echo.

rem 设置 Python 路径
set PYTHONPATH=

rem 使用系统 Python 启动
python main.py

pause

