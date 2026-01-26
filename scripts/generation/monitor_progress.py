#!/usr/bin/env python3
"""监控混合数据集生成进度"""
import json, os, time
from datetime import datetime

FILES = {
    "C2": "datasets/hybrid/c2_continuation.json",
    "C3": "datasets/hybrid/c3_edited.json",
    "C4": "datasets/hybrid/c4_polished.json"
}

while True:
    os.system('clear')
    print(f"{'='*60}")
    print(f"混合数据集生成进度监控 - {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}\n")
    
    total = 0
    for cat, path in FILES.items():
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                count = len(data)
                total += count
                status = "✓" if count >= 100 else "⏳"
                print(f"{status} {cat}: {count:3d} 条")
            except:
                print(f"⚠ {cat}: 文件损坏")
        else:
            print(f"⏳ {cat}:   0 条 (未生成)")
    
    print(f"\n总计: {total} 条")
    print(f"\n按Ctrl+C退出监控")
    time.sleep(5)
