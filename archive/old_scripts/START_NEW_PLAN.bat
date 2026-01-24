@echo off
chcp 65001 >nul
cd /d C:\datacollection

echo ============================================================
echo       新方案数据集生成 - 修复版 (deepseek/qwen/claude)
echo ============================================================
echo.
echo 模型配置:
echo   - deepseek: 10000 条
echo   - qwen: 8000 条
echo   - claude: 7000 条
echo   - 总计: 25000 条
echo.
echo 开始运行...
echo ============================================================
echo.

set PYTHONUNBUFFERED=1
python -u "new data collection.py"

echo.
echo ============================================================
echo 程序运行完成！
echo 数据集保存在: new_plan_datasets\
echo ============================================================
pause
