#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""新方案数据集生成启动脚本"""
import subprocess
import sys
import os
import pathlib

work_dir = pathlib.Path(__file__).parent.absolute()
os.chdir(work_dir)

print("=" * 70)
print("          新方案数据集生成 - 启动中")
print("=" * 70)
print(f"配置:")
print(f"  - Custom:   10000条")
print(f"  - DeepSeek: 8000条")
print(f"  - Qwen:     7000条")
print(f"  - 总计:     25000条")
print("=" * 70)
print()

# 使用虚拟环境的Python
python_exe = work_dir / '.venv' / 'Scripts' / 'python.exe'
script = work_dir / 'new data collection.py'

# 启动程序
process = subprocess.Popen(
    [str(python_exe), str(script)],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    encoding='utf-8',
    errors='replace'
)

# 实时输出
for line in process.stdout:
    print(line.rstrip())

process.wait()
print("\n" + "=" * 70)
print(f"程序退出码: {process.returncode}")
print("=" * 70)
