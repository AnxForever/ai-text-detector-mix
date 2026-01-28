#!/usr/bin/env python3
"""Remove explicit AI/refusal phrases from final_clean splits."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

DEFAULT_PHRASES = [
    "作为一个AI",
    "作为AI",
    "作为一个人工智能",
    "AI语言模型",
    "As an AI",
    "as an AI",
    "I cannot",
    "I can't",
    "无法回答",
    "不能回答",
]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Clean final_clean explicit phrases.")
    parser.add_argument(
        "--input-dir",
        default=str(PROJECT_ROOT / "datasets" / "final_clean"),
        help="Input final_clean directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "datasets" / "final_clean_phrase_clean"),
        help="Output directory for cleaned splits.",
    )
    parser.add_argument(
        "--report-dir",
        default=str(PROJECT_ROOT / "docs" / "plans" / "audit_reports"),
        help="Directory to write markdown report.",
    )
    parser.add_argument(
        "--phrases",
        default="",
        help="Optional comma-separated phrases to remove.",
    )
    return parser.parse_args()


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file with basic error handling."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing file: {path}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to read CSV: {path}") from exc


def remove_phrases(df: pd.DataFrame, phrases: List[str]) -> pd.DataFrame:
    """Remove rows containing explicit AI/refusal phrases."""
    if not phrases:
        return df.copy()
    pattern = "|".join(phrases)
    mask = ~df["text"].astype(str).str.contains(pattern, regex=True, case=False, na=False)
    return df.loc[mask].copy()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    phrases = DEFAULT_PHRASES
    if args.phrases:
        phrases = [p.strip() for p in args.phrases.split(",") if p.strip()]

    splits = ["train.csv", "val.csv", "test.csv"]
    summary: Dict[str, Dict[str, int]] = {}

    for split in splits:
        path = input_dir / split
        df = load_csv(path)
        before = len(df)
        df = remove_phrases(df, phrases)
        after = len(df)
        summary[split] = {"before": before, "after": after, "removed": before - after}
        df.to_csv(output_dir / split, index=False)

    log_path = output_dir / "cleaning_log.json"
    log_path.write_text(
        json.dumps(
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "summary": summary,
                "phrases": phrases,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    report_lines = [
        "# final_clean 口癖清理报告",
        "",
        f"> 生成日期: {datetime.now().strftime('%Y-%m-%d')}",
        "",
        f"- 输入: {input_dir}",
        f"- 输出: {output_dir}",
        "",
        "## 各 split 统计",
    ]
    for split, stats in summary.items():
        report_lines.append(
            f"- {split}: before {stats['before']}, removed {stats['removed']}, after {stats['after']}"
        )
    report_path = report_dir / f"{datetime.now().strftime('%Y-%m-%d')}_final_clean_phrase_clean_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(str(output_dir))


if __name__ == "__main__":
    main()
