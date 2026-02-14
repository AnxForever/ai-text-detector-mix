#!/usr/bin/env python3
"""
训练状态监控脚本
检查当前 BERT 训练进度和系统状态
"""
import os
import re
import subprocess

def check_training_status():
    """检查训练状态."""
    log_path = '/tmp/train_balanced3.log'

    if not os.path.exists(log_path):
        print("❌ 训练日志不存在")
        return

    # 检查进程
    result = subprocess.run(
        ['pgrep', '-f', 'train_balanced'],
        capture_output=True, text=True
    )
    if not result.stdout.strip():
        print("⚠️ 训练进程未运行")
    else:
        print(f"✓ 训练进程运行中 (PID: {result.stdout.strip()})")

    # 读取日志
    with open(log_path, 'rb') as f:
        content = f.read().decode('utf-8', errors='ignore')

    # 解析进度
    progress_matches = re.findall(r'Training:\s+(\d+)%.*?(\d+)/28810', content)
    epoch_matches = re.findall(r'=== Epoch (\d+)/(\d+) ===', content)
    loss_matches = re.findall(r'Loss: ([\d.]+)', content)
    acc_matches = re.findall(r'Val Acc: ([\d.]+)', content)
    speed_matches = re.findall(r'([\d.]+)it/s', content)
    best_matches = re.findall(r'最佳模型已保存.*acc=([\d.]+)', content)

    print("\n" + "=" * 50)
    print("训练进度")
    print("=" * 50)

    if epoch_matches:
        current_epoch, total_epochs = epoch_matches[-1]
        print(f"当前 Epoch: {current_epoch}/{total_epochs}")

    if progress_matches:
        pct, step = progress_matches[-1]
        print(f"Epoch 内进度: {pct}% ({step}/28810 steps)")

        if speed_matches:
            speed = float(speed_matches[-1])
            remaining_in_epoch = 28810 - int(step)
            if epoch_matches:
                remaining_epochs = int(total_epochs) - int(current_epoch)
                remaining_steps = remaining_in_epoch + remaining_epochs * 28810
                remaining_min = remaining_steps / speed / 60
                print(f"速度: {speed:.1f} it/s")
                print(f"预估剩余: {remaining_min:.0f} 分钟")

    if loss_matches:
        print(f"\n最近 Loss: {loss_matches[-1]}")

    if acc_matches:
        print(f"最近 Val Acc: {float(acc_matches[-1])*100:.2f}%")

    if best_matches:
        print(f"当前最佳: {float(best_matches[-1])*100:.2f}%")

    # GPU 状态
    print("\n" + "=" * 50)
    print("系统状态")
    print("=" * 50)

    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            if len(parts) >= 3:
                used, total, util = parts
                print(f"GPU 显存: {used}MB / {total}MB")
                print(f"GPU 利用率: {util}%")
    except Exception:
        pass


if __name__ == '__main__':
    check_training_status()
