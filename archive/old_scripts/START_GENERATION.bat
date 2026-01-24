@echo off
chcp 65001 >nul
cd /d C:\datacollection

echo ============================================================
echo             多模型数据集生成系统 - 新API版本
echo ============================================================
echo.
echo API端点: https://wzw.pp.ua/v1
echo 配置模型: deepseek-v3.2-chat, claude-sonnet-4-5, qwen-max-latest, GLM-4.7
echo.
echo 开始运行...
echo ============================================================
echo.

python "new data collection.py"

echo.
echo ============================================================
echo 程序运行完成！
echo 数据集保存在: auto_generated_datasets\
echo ============================================================
pause
