#!/usr/bin/env python3
"""Score dataset quality/difficulty and route into Core/Hard/Review/Reject pools."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

PHRASES = [
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

AI_SOURCE_HINTS = ["gpt", "qwen", "deepseek", "glm", "gemini", "claude", "openai"]
HUMAN_SOURCE_HINTS = ["human", "thucnews", "hc3", "wikipedia", "news"]

LENGTH_BUCKETS: List[Tuple[int, int, str]] = [
    (0, 80, "<80"),
    (80, 200, "80-200"),
    (200, 500, "200-500"),
    (500, 1000, "500-1000"),
    (1000, 2000, "1000-2000"),
    (2000, 10**9, "2000+"),
]

RE_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
RE_PHONE = re.compile(r"1[3-9]\d{9}")
RE_ID = re.compile(r"\b\d{17}[\dXx]\b")
RE_CJK = re.compile(r"[\u4e00-\u9fff]")
RE_ENG = re.compile(r"[A-Za-z]")
RE_DIGIT = re.compile(r"\d")
RE_PUNCT = re.compile(r"[\s\.,;:!?、，。？！；：\(\)\[\]\{\}\"'“”‘’…·—\-]")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Score and route dataset.")
    parser.add_argument(
        "--input",
        default="",
        help="Input CSV path (single file).",
    )
    parser.add_argument(
        "--input-dir",
        default="",
        help="Input directory containing train/val/test CSV files.",
    )
    parser.add_argument(
        "--dataset-name",
        default="dataset",
        help="Dataset name for output directory naming.",
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "datasets" / "routed"),
        help="Root output directory.",
    )
    parser.add_argument(
        "--report-dir",
        default=str(PROJECT_ROOT / "docs" / "plans" / "audit_reports"),
        help="Directory for markdown reports.",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=80,
        help="Minimum length for Q_length.",
    )
    parser.add_argument(
        "--rule-version",
        default="v1",
        help="Rule version identifier.",
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


def load_inputs(input_path: str, input_dir: str) -> pd.DataFrame:
    """Load input CSV(s)."""
    if input_dir:
        dir_path = Path(input_dir)
        frames: List[pd.DataFrame] = []
        for split in ["train.csv", "val.csv", "test.csv"]:
            split_path = dir_path / split
            if split_path.exists():
                df = load_csv(split_path)
                df["_split"] = split.replace(".csv", "")
                frames.append(df)
        if not frames:
            raise FileNotFoundError(f"No CSV files found in {input_dir}")
        return pd.concat(frames, ignore_index=True)
    if input_path:
        return load_csv(Path(input_path))
    raise ValueError("Either --input or --input-dir must be provided.")


def bucket_length(length: int) -> str:
    """Map length to bucket name."""
    for low, high, name in LENGTH_BUCKETS:
        if low <= length < high:
            return name
    return "unknown"


def compute_valid_ratios(text: str) -> Dict[str, float]:
    """Compute basic character ratios for quality checks."""
    total = max(len(text), 1)
    cjk = len(RE_CJK.findall(text))
    eng = len(RE_ENG.findall(text))
    digit = len(RE_DIGIT.findall(text))
    punct = len(RE_PUNCT.findall(text))
    valid = cjk + eng + digit + punct
    return {
        "valid_ratio": valid / total,
        "cjk_ratio": cjk / total,
        "eng_ratio": eng / total,
    }


def detect_quality_flags(text: str, length: int, min_length: int, dup: bool) -> List[str]:
    """Detect quality flags based on rules."""
    flags: List[str] = []
    ratios = compute_valid_ratios(text)

    if length < min_length:
        flags.append("q_truncated")
    if "[SEP]" in text:
        flags.append("q_sep_contamination")
    if any(p in text for p in PHRASES):
        flags.append("q_boilerplate")
    if ratios["valid_ratio"] < 0.6:
        flags.append("q_garbage")
    if (ratios["cjk_ratio"] + ratios["eng_ratio"]) < 0.15:
        flags.append("q_format_abnormal")
    if ratios["cjk_ratio"] < 0.1 and ratios["eng_ratio"] > 0.5:
        flags.append("q_lang_mismatch")
    if dup:
        flags.append("q_duplicate")
    if RE_EMAIL.search(text) or RE_PHONE.search(text) or RE_ID.search(text):
        flags.append("q_pii_risk")

    return flags


def score_q(text: str, length: int, flags: List[str], min_length: int, source: str) -> float:
    """Compute quality score Q."""
    ratios = compute_valid_ratios(text)
    q_clean = ratios["valid_ratio"]
    penalties = {
        "q_garbage": 0.3,
        "q_format_abnormal": 0.2,
        "q_lang_mismatch": 0.2,
        "q_boilerplate": 0.1,
        "q_sep_contamination": 0.3,
        "q_duplicate": 0.2,
        "q_truncated": 0.2,
        "q_pii_risk": 0.2,
    }
    for flag, penalty in penalties.items():
        if flag in flags:
            q_clean -= penalty
    q_clean = max(0.0, min(1.0, q_clean))

    if length < min_length:
        q_length = 0.0
    elif length < 200:
        q_length = 0.6
    elif length < 500:
        q_length = 0.8
    elif length < 1000:
        q_length = 0.9
    else:
        q_length = 1.0

    source_lower = (source or "").lower()
    if any(k in source_lower for k in AI_SOURCE_HINTS + HUMAN_SOURCE_HINTS):
        q_prov = 0.8
    elif source_lower:
        q_prov = 0.7
    else:
        q_prov = 0.5

    q_score = 0.4 * q_clean + 0.3 * q_length + 0.3 * q_prov
    return round(max(0.0, min(1.0, q_score)), 4)


def map_label(label: str, category: str) -> str:
    """Map raw label/category to y_main."""
    if category in {"C2", "C3", "C4"}:
        return "MIXED"
    if label in {"1", "AI", "ai"}:
        return "AI"
    if label in {"0", "Human", "human"}:
        return "HUMAN"
    return "UNCERTAIN"


def compute_y_conf(y_main: str, flags: List[str], source: str) -> Tuple[float, List[str]]:
    """Compute label confidence score and evidence."""
    evidence: List[str] = []
    if y_main == "UNCERTAIN":
        return 0.4, ["y_source_unknown"]

    source_lower = (source or "").lower()
    base = 0.7
    if source_lower:
        evidence.append("y_source_trust")
        base += 0.1
    else:
        evidence.append("y_source_unknown")

    if "q_boilerplate" in flags:
        if y_main == "AI":
            evidence.append("y_rule_support")
            base += 0.1
        else:
            evidence.append("y_conflict")
            base -= 0.3

    if any(k in source_lower for k in AI_SOURCE_HINTS) and y_main == "AI":
        evidence.append("y_source_trust")
        base += 0.1
    if any(k in source_lower for k in HUMAN_SOURCE_HINTS) and y_main == "HUMAN":
        evidence.append("y_source_trust")
        base += 0.1

    return round(max(0.0, min(1.0, base)), 4), evidence


def compute_difficulty(
    pred_prob: Optional[float], style: str
) -> Tuple[Optional[float], List[str]]:
    """Compute difficulty score D."""
    if pred_prob is None:
        return None, []
    margin = abs(2 * pred_prob - 1)
    d_uncertainty = 1 - margin
    d_style_prior = 0.05 if style in {"technical_doc", "list", "readme"} else 0.0
    d_score = 0.7 * d_uncertainty + 0.3 * d_style_prior
    flags = ["d_uncertain"] if d_uncertainty >= 0.6 else []
    if d_style_prior > 0:
        flags.append("d_style_prior")
    return round(max(0.0, min(1.0, d_score)), 4), flags


def route_sample(
    y_main: str, q_score: float, y_conf: float, flags: List[str], d_score: Optional[float]
) -> str:
    """Route sample into pools based on scores."""
    if y_main == "MIXED":
        return "review"
    if "q_sep_contamination" in flags or q_score < 0.3:
        return "reject"
    if y_conf < 0.6 or q_score < 0.6:
        return "review"
    if d_score is not None and d_score >= 0.75:
        return "hard"
    if d_score is None and y_main == "MIXED":
        return "review"
    return "core"


def main() -> None:
    """Main entry point."""
    args = parse_args()
    df = load_inputs(args.input, args.input_dir)

    if "text" not in df.columns:
        raise RuntimeError("Missing required column: text")

    df["text"] = df["text"].astype(str)
    df["label"] = df.get("label", "").astype(str)
    if "category" in df.columns:
        df["category"] = df["category"].astype(str)
    else:
        df["category"] = ""
    df["source"] = df.get("source", "").astype(str)
    if "style" in df.columns:
        df["style"] = df["style"].astype(str)
    else:
        df["style"] = ""

    hashes = df["text"].apply(lambda x: hash(x))
    dup_counts = hashes.value_counts()
    dup_flags = hashes.map(lambda h: dup_counts.get(h, 0) > 1)

    rows = []
    pool_counts = Counter()
    q_flag_counts = Counter()
    d_flag_counts = Counter()
    y_conflicts = 0

    difficulty_enabled = "pred_ai_prob" in df.columns

    for idx, row in df.iterrows():
        text = row["text"]
        length = len(text)
        length_bucket = bucket_length(length)
        flags = detect_quality_flags(text, length, args.min_length, bool(dup_flags.iloc[idx]))
        for flag in flags:
            q_flag_counts[flag] += 1

        q_score = score_q(text, length, flags, args.min_length, row["source"])
        y_main = map_label(row["label"], row["category"])
        y_conf, y_evidence = compute_y_conf(y_main, flags, row["source"])

        if "y_conflict" in y_evidence:
            y_conflicts += 1

        pred_prob = None
        if difficulty_enabled:
            try:
                pred_prob = float(row["pred_ai_prob"])
            except ValueError:
                pred_prob = None

        d_score, d_flags = compute_difficulty(pred_prob, row["style"])
        for flag in d_flags:
            d_flag_counts[flag] += 1

        pool = route_sample(y_main, q_score, y_conf, flags, d_score)
        pool_counts[pool] += 1

        rows.append(
            {
                **row.to_dict(),
                "y_main": y_main,
                "length_bucket": length_bucket,
                "Q": q_score,
                "D": d_score if d_score is not None else "",
                "y_conf": y_conf,
                "q_flags": "|".join(flags),
                "d_flags": "|".join(d_flags),
                "y_evidence": "|".join(y_evidence),
                "routing": pool,
                "rule_version": args.rule_version,
            }
        )

    out_df = pd.DataFrame(rows)

    output_root = Path(args.output_root) / args.dataset_name
    output_root.mkdir(parents=True, exist_ok=True)
    pools_dir = output_root / "pools"
    pools_dir.mkdir(parents=True, exist_ok=True)

    out_df.to_csv(output_root / "scored_all.csv", index=False)

    for pool in ["core", "hard", "review", "reject"]:
        pool_df = out_df[out_df["routing"] == pool]
        pool_df.to_csv(pools_dir / f"{pool}.csv", index=False)

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_name": args.dataset_name,
        "rows": len(out_df),
        "pool_counts": dict(pool_counts),
        "q_flag_counts": dict(q_flag_counts),
        "d_flag_counts": dict(d_flag_counts),
        "y_conflicts": y_conflicts,
        "rule_version": args.rule_version,
    }

    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{args.dataset_name}_score_route_report.md"
    lines = [
        "# 数据评价系统评分与分流报告",
        "",
        f"> 生成日期: {datetime.now().strftime('%Y-%m-%d')}",
        f"> 数据集: {args.dataset_name}",
        f"> 规则版本: {args.rule_version}",
        f"> 难度评分启用: {difficulty_enabled}",
        "",
        f"- 总样本: {len(out_df)}",
        f"- Core: {pool_counts.get('core', 0)}",
        f"- Hard: {pool_counts.get('hard', 0)}",
        f"- Review: {pool_counts.get('review', 0)}",
        f"- Reject: {pool_counts.get('reject', 0)}",
        f"- Label Conflicts: {y_conflicts}",
        "",
        "## 质量原因码统计",
    ]
    for key, value in q_flag_counts.most_common():
        lines.append(f"- {key}: {value}")

    lines.append("")
    lines.append("## 难度原因码统计")
    for key, value in d_flag_counts.most_common():
        lines.append(f"- {key}: {value}")

    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(str(report_path))


if __name__ == "__main__":
    main()
