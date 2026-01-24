#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter refusal/template responses from a CSV dataset.

Default input:  new_plan_datasets/parallel_dataset.csv
Default output: new_plan_datasets/parallel_dataset_cleaned.csv
"""
import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_INPUT = Path("new_plan_datasets/parallel_dataset.csv")
DEFAULT_OUTPUT = Path("new_plan_datasets/parallel_dataset_cleaned.csv")
DEFAULT_REPORT = Path("new_plan_datasets/cleaning_refusal_report.json")


RULES: List[Dict[str, str]] = [
    {
        "label": "en_refusal_core",
        "pattern": r"\bI (?:can't|cannot|won't|will not|am unable to|am not able to)\b",
        "scope": "all",
    },
    {
        "label": "en_refusal_help",
        "pattern": r"\bI (?:can't|cannot|won't|will not) (?:help|assist|comply|provide|fulfill)\b",
        "scope": "all",
    },
    {
        "label": "en_refusal_help_alt",
        "pattern": r"\bI (?:can't|cannot) (?:help|assist) (?:with|you)\b",
        "scope": "head",
    },
    {
        "label": "en_refusal_provide",
        "pattern": r"\bI (?:can't|cannot) provide\b",
        "scope": "head",
    },
    {
        "label": "en_refusal_comply",
        "pattern": r"\bI (?:can't|cannot) comply\b",
        "scope": "head",
    },
    {
        "label": "en_refusal_fulfill",
        "pattern": r"\bI (?:can't|cannot) fulfill\b",
        "scope": "head",
    },
    {
        "label": "en_refusal_complete",
        "pattern": r"\bI (?:can't|cannot) (?:complete|perform)\b",
        "scope": "head",
    },
    {"label": "en_apology", "pattern": r"\bI(?:'m| am) sorry\b", "scope": "head"},
    {"label": "en_apology_but", "pattern": r"\bI(?:'m| am) sorry,? but\b", "scope": "head"},
    {"label": "en_apology_alt", "pattern": r"\bI apologize\b", "scope": "head"},
    {"label": "en_ai_disclaimer", "pattern": r"\bAs an AI\b", "scope": "head"},
    {"label": "en_ai_model", "pattern": r"\bAs an AI (?:assistant|model)\b", "scope": "head"},
    {"label": "en_lm_disclaimer", "pattern": r"\bAs a (?:large )?language model\b", "scope": "head"},
    {"label": "en_ai_identity", "pattern": r"\bI(?:'m| am) an AI\b", "scope": "head"},
    {"label": "en_ai_identity_alt", "pattern": r"\bI(?:'m| am) a (?:language model|virtual assistant)\b", "scope": "head"},
    {"label": "en_access", "pattern": r"\bI (?:do not|don't) have (?:access|the ability)\b", "scope": "head"},
    {"label": "en_access_alt", "pattern": r"\bI (?:cannot|can't) access\b", "scope": "head"},
    {"label": "en_refusal_style", "pattern": r"\bI appreciate you reaching out\b", "scope": "head"},
    {"label": "en_refusal_style_alt", "pattern": r"\bI appreciate your interest\b", "scope": "head"},
    {
        "label": "en_refusal_redirect",
        "pattern": r"\bI need to (?:redirect|decline|refuse) (?:this|that) request\b",
        "scope": "head",
    },
    {
        "label": "en_policy",
        "pattern": r"\bThis request (?:violates|is not allowed).*(?:policy|policies|guidelines|safety|rules)\b",
        "scope": "head",
    },
    {"label": "en_policy_alt", "pattern": r"\bI (?:cannot|can't) comply with (?:this|that) request\b", "scope": "head"},
    {"label": "en_policy_alt2", "pattern": r"\bI must decline\b", "scope": "head"},
    {"label": "en_scope", "pattern": r"\bI (?:can only|am only able to)\b", "scope": "head"},
    {"label": "en_refusal_request", "pattern": r"\bI (?:can't|cannot) (?:help|assist) with that request\b", "scope": "head"},
    {"label": "zh_ai_disclaimer", "pattern": r"作为(一个)?AI", "scope": "head"},
    {"label": "zh_ai_assistant", "pattern": r"作为(一个)?AI助手", "scope": "head"},
    {"label": "zh_ai_alt", "pattern": r"作为(一个)?人工智能", "scope": "head"},
    {"label": "zh_ai_model", "pattern": r"作为(一个)?(语言模型|大语言模型|大型语言模型|AI模型)", "scope": "head"},
    {"label": "zh_ai_identity", "pattern": r"我是(一个)?(AI|人工智能|语言模型)", "scope": "head"},
    {"label": "zh_ai_identity_alt", "pattern": r"我只是(一个)?(AI|人工智能|语言模型)", "scope": "head"},
    {"label": "zh_i_am_ai", "pattern": r"我是(一个)?AI", "scope": "head"},
    {"label": "zh_apology", "pattern": r"(很抱歉|抱歉|对不起)", "scope": "head"},
    {"label": "zh_refusal_core", "pattern": r"我(无法|不能|不便|不支持|不提供)", "scope": "head"},
    {"label": "zh_refusal_support", "pattern": r"无法(提供|满足|协助|帮助)", "scope": "head"},
    {"label": "zh_refusal_support_alt", "pattern": r"不能(提供|满足|协助|帮助)", "scope": "head"},
    {"label": "zh_refusal_support_alt2", "pattern": r"不便(提供|满足|协助|帮助)", "scope": "head"},
    {"label": "zh_refusal_help", "pattern": r"我(无法|不能)帮助", "scope": "head"},
    {"label": "zh_refusal_help_alt", "pattern": r"我(无法|不能)为你", "scope": "head"},
    {"label": "zh_refusal_request", "pattern": r"无法满足(你的|该|此)请求", "scope": "head"},
    {"label": "zh_refusal_request_alt", "pattern": r"不支持(你的|该|此)请求", "scope": "head"},
    {"label": "zh_refusal_request_alt2", "pattern": r"无法处理(你的|该|此)请求", "scope": "head"},
    {
        "label": "zh_refusal_policy",
        "pattern": r"(?:违反|根据).{0,20}(?:政策|规定|准则|安全).{0,20}(?:无法|不能|不支持|不提供|不便|不予|拒绝|不允许|不被允许)",
        "scope": "head",
    },
    {
        "label": "zh_refusal_policy_alt",
        "pattern": r"(?:出于|基于).{0,20}(?:安全|政策|规定|合规).{0,20}(?:无法|不能|不支持|不提供|不便|不予|拒绝|不允许|不被允许)",
        "scope": "head",
    },
    {"label": "zh_refusal_policy_alt2", "pattern": r"(该|此)请求.*(不被允许|不允许|不合规)", "scope": "head"},
    {"label": "zh_refusal_deny", "pattern": r"不予(提供|支持|协助)", "scope": "head"},
]


def compile_rules() -> List[Tuple[str, re.Pattern, str]]:
    compiled: List[Tuple[str, re.Pattern, str]] = []
    for rule in RULES:
        compiled.append(
            (
                rule["label"],
                re.compile(rule["pattern"], re.IGNORECASE),
                rule["scope"],
            )
        )
    return compiled


def safe_snippet(text: str, limit: int = 160) -> str:
    snippet = text.replace("\n", " ").replace("\r", " ").strip()
    return snippet[:limit]


def match_rule(text: str, compiled: List[Tuple[str, re.Pattern, str]], head_chars: int) -> Optional[str]:
    if not text:
        return "empty_text"
    head = text[:head_chars]
    for label, pattern, scope in compiled:
        target = text if scope == "all" else head
        if pattern.search(target):
            return label
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter refusal/template responses from a CSV dataset.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input CSV path")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output CSV path")
    parser.add_argument("--report", default=str(DEFAULT_REPORT), help="Report JSON path")
    parser.add_argument("--text-column", default="text_content", help="Column name to scan")
    parser.add_argument("--head-chars", type=int, default=260, help="Head chars used for head-only rules")
    parser.add_argument("--sample", type=int, default=5, help="Sample removed rows for report")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report)

    if not input_path.exists():
        print(f"Input not found: {input_path}")
        return 1

    compiled = compile_rules()
    total = 0
    kept = 0
    removed = 0
    reason_counts: Dict[str, int] = {}
    samples: List[Dict[str, str]] = []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8-sig", newline="") as f_in:
        reader = csv.DictReader(f_in)
        if not reader.fieldnames:
            print("Input CSV missing header.")
            return 1
        if args.text_column not in reader.fieldnames:
            print(f"Missing column: {args.text_column}")
            return 1

        with output_path.open("w", encoding="utf-8-sig", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
            writer.writeheader()

            for row in reader:
                total += 1
                text = row.get(args.text_column, "")
                reason = match_rule(str(text), compiled, args.head_chars)
                if reason:
                    removed += 1
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
                    if len(samples) < args.sample:
                        samples.append(
                            {
                                "reason": reason,
                                "source_api": row.get("source_api", ""),
                                "source_model": row.get("source_model", ""),
                                "text_len": str(len(str(text))),
                                "text_snippet": safe_snippet(str(text)),
                            }
                        )
                    continue

                writer.writerow(row)
                kept += 1

    report = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "total_rows": total,
        "kept_rows": kept,
        "removed_rows": removed,
        "removed_pct": round(removed / total * 100, 2) if total else 0.0,
        "head_chars": args.head_chars,
        "reason_counts": reason_counts,
        "samples": samples,
        "generated_at": datetime.now().isoformat(),
    }

    with report_path.open("w", encoding="utf-8") as f_report:
        json.dump(report, f_report, indent=2, ensure_ascii=False)

    print(f"Done. kept={kept} removed={removed} total={total}")
    print(f"Output: {output_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
