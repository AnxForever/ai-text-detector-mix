@echo off
cd /d C:\datacollection
echo Starting new plan data generation...
echo Configuration:
echo   - Custom:   10000 samples
echo   - DeepSeek: 8000 samples
echo   - Qwen:     7000 samples
echo   - Total:    25000 samples
echo.
echo Output will be saved to: generation.log
echo.
start /B .venv\bin\python.exe "new data collection.py" > generation.log 2>&1
echo Process started. Check generation.log for progress.
echo.
tail -f generation.log
