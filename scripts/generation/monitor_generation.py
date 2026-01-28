#!/usr/bin/env python3
"""
Monitor ongoing scenario fill generation progress.

Usage:
    python scripts/generation/monitor_generation.py [output_dir]
"""
import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
import sys


def count_records(jsonl_path: Path) -> int:
    """Count records in a JSONL file."""
    if not jsonl_path.exists():
        return 0
    count = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def analyze_directory(output_dir: Path):
    """Analyze generation progress in output directory."""
    # Find all part files
    part_files = sorted(output_dir.glob("*_part*.jsonl"))
    rejected_files = sorted(output_dir.glob("*_rejected.jsonl"))

    if not part_files and not rejected_files:
        print("❌ No generation files found in directory")
        return

    # Count total records
    total_success = 0
    scenario_counts = Counter()
    style_counts = Counter()
    model_counts = Counter()

    for part_file in part_files:
        with part_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        total_success += 1
                        scenario_counts[record.get("scenario_id", "?")] += 1
                        style_counts[record.get("style_plan", "?")] += 1
                        model_counts[record.get("model", "?")] += 1
                    except:
                        pass

    # Count rejected
    total_rejected = 0
    reject_reasons = Counter()
    reject_scenarios = Counter()

    for rejected_file in rejected_files:
        with rejected_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        total_rejected += 1
                        reject_reasons[record.get("reason", "?")] += 1
                        reject_scenarios[record.get("scenario_id", "?")] += 1
                    except:
                        pass

    # Display results
    print("=" * 60)
    print("📊 生成进度监控")
    print("=" * 60)
    print(f"\n✅ 成功生成: {total_success:,}")
    print(f"❌ 被拒绝: {total_rejected:,}")
    total = total_success + total_rejected
    if total > 0:
        print(f"📈 成功率: {100 * total_success / total:.1f}%")

    print(f"\n📁 文件数: {len(part_files)} 个part, {len(rejected_files)} 个rejected")

    # Scenario distribution
    if scenario_counts:
        print(f"\n🎯 场景分布 (成功):")
        for scenario, count in sorted(scenario_counts.items()):
            print(f"   {scenario}: {count:,}")

    # Style distribution
    if style_counts:
        print(f"\n✍️  样式分布 (成功):")
        for style, count in sorted(style_counts.items()):
            print(f"   {style}: {count:,}")

    # Model distribution (top 5)
    if model_counts:
        print(f"\n🤖 模型分布 (Top 5):")
        for model, count in model_counts.most_common(5):
            print(f"   {model}: {count:,}")

    # Rejection reasons
    if reject_reasons:
        print(f"\n⚠️  拒绝原因:")
        for reason, count in reject_reasons.most_common():
            pct = 100 * count / total_rejected if total_rejected > 0 else 0
            print(f"   {reason}: {count:,} ({pct:.1f}%)")

    # Rejected scenarios
    if reject_scenarios:
        print(f"\n❌ 拒绝场景分布:")
        for scenario, count in sorted(reject_scenarios.items()):
            print(f"   {scenario}: {count:,}")

    print("\n" + "=" * 60)


def watch_mode(output_dir: Path, interval: int = 10):
    """Watch directory and update stats periodically."""
    print(f"👀 监控模式启动 (每{interval}秒刷新)")
    print(f"📂 监控目录: {output_dir}")
    print("按 Ctrl+C 停止\n")

    try:
        while True:
            # Clear screen (works on most terminals)
            print("\033[2J\033[H", end="")
            analyze_directory(output_dir)
            print(f"\n⏰ 下次更新: {interval}秒后...")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\n✋ 监控已停止")


def main():
    parser = argparse.ArgumentParser(
        description="Monitor scenario fill generation progress"
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="datasets/generated/scenario_fill/2026-01-27_10h_multi_proxies",
        help="Output directory to monitor (default: 2026-01-27_10h_multi_proxies)",
    )
    parser.add_argument(
        "-w", "--watch",
        action="store_true",
        help="Watch mode: continuously update stats",
    )
    parser.add_argument(
        "-i", "--interval",
        type=int,
        default=10,
        help="Update interval in seconds for watch mode (default: 10)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"❌ 目录不存在: {output_dir}", file=sys.stderr)
        return 1

    if args.watch:
        watch_mode(output_dir, args.interval)
    else:
        analyze_directory(output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
