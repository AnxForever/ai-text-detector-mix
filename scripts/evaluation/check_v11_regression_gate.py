#!/usr/bin/env python3
"""Check whether a candidate model passes the V11 regression gate."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate rollback gate for V11 candidate.")
    parser.add_argument(
        "--baseline-json",
        default=str(ROOT / "models/bert_v10_augmented/eval_comparison.json"),
        help="Baseline comparison JSON path.",
    )
    parser.add_argument(
        "--candidate-json",
        default=str(ROOT / "models/bert_v11_candidate/eval_comparison.json"),
        help="Candidate comparison JSON path.",
    )
    parser.add_argument(
        "--baseline-key",
        default="bert_v10_augmented",
        help="Model key in baseline JSON.",
    )
    parser.add_argument(
        "--candidate-key",
        default="bert_v11_candidate",
        help="Model key in candidate JSON.",
    )
    parser.add_argument(
        "--max-three-set-degradation",
        type=float,
        default=0.5,
        help="Maximum allowed drop in three_set_avg (percentage points).",
    )
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "docs/plans/v11_regression_gate.json"),
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "docs/plans/v11_regression_gate.md"),
        help="Output Markdown report path.",
    )
    return parser.parse_args()


def load_model_metrics(path: Path, model_key: str) -> Dict[str, object]:
    """Load one model entry from an eval comparison JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Comparison JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if model_key not in payload:
        keys = ", ".join(sorted(payload.keys()))
        raise KeyError(f"Model key `{model_key}` not found in {path}. Available: {keys}")
    return payload[model_key]


def dataset_delta(
    baseline: Dict[str, object],
    candidate: Dict[str, object],
    dataset_key: str,
) -> Dict[str, float]:
    """Extract per-dataset deltas for accuracy/precision/recall/f1."""
    base_ds = baseline.get(dataset_key, {})
    cand_ds = candidate.get(dataset_key, {})

    output = {"dataset": dataset_key}
    for metric in ("accuracy", "precision", "recall", "f1"):
        b = float(base_ds.get(metric, 0.0))
        c = float(cand_ds.get(metric, 0.0))
        output[f"{metric}_baseline"] = b
        output[f"{metric}_candidate"] = c
        output[f"{metric}_delta"] = round(c - b, 4)
    return output


def evaluate_gate(
    baseline: Dict[str, object],
    candidate: Dict[str, object],
    max_three_set_degradation: float,
) -> Dict[str, object]:
    """Apply rollback decision rule."""
    baseline_avg = float(baseline.get("three_set_avg", 0.0))
    candidate_avg = float(candidate.get("three_set_avg", 0.0))
    avg_delta = round(candidate_avg - baseline_avg, 4)
    degradation = round(baseline_avg - candidate_avg, 4)

    datasets = [
        dataset_delta(baseline, candidate, "core_v1_test_clean"),
        dataset_delta(baseline, candidate, "independent_data"),
        dataset_delta(baseline, candidate, "merged_v2_val_clean"),
    ]

    pass_gate = degradation <= max_three_set_degradation
    if pass_gate:
        decision = "keep_candidate"
        reason = (
            f"three_set_avg degradation {degradation:.4f} <= "
            f"threshold {max_three_set_degradation:.4f}"
        )
    else:
        decision = "rollback_to_baseline"
        reason = (
            f"three_set_avg degradation {degradation:.4f} > "
            f"threshold {max_three_set_degradation:.4f}"
        )

    return {
        "baseline_three_set_avg": baseline_avg,
        "candidate_three_set_avg": candidate_avg,
        "three_set_avg_delta": avg_delta,
        "three_set_avg_degradation": degradation,
        "max_three_set_degradation": max_three_set_degradation,
        "pass_gate": pass_gate,
        "decision": decision,
        "reason": reason,
        "dataset_deltas": datasets,
    }


def render_markdown(report: Dict[str, object]) -> str:
    """Render Markdown output."""
    lines = [
        "# V11 Regression Gate",
        "",
        f"- Generated at: {report['generated_at']}",
        f"- Baseline: `{report['baseline_model_key']}` from `{report['baseline_json']}`",
        f"- Candidate: `{report['candidate_model_key']}` from `{report['candidate_json']}`",
        "",
        "## Gate result",
        "",
        f"- Decision: **{report['gate']['decision']}**",
        f"- Pass gate: `{report['gate']['pass_gate']}`",
        f"- Reason: {report['gate']['reason']}",
        f"- Baseline three_set_avg: {report['gate']['baseline_three_set_avg']}",
        f"- Candidate three_set_avg: {report['gate']['candidate_three_set_avg']}",
        f"- Delta (candidate - baseline): {report['gate']['three_set_avg_delta']}",
        "",
        "## Per-set deltas",
        "",
        "| dataset | accuracy_delta | precision_delta | recall_delta | f1_delta |",
        "|---|---:|---:|---:|---:|",
    ]

    for row in report["gate"]["dataset_deltas"]:
        lines.append(
            f"| {row['dataset']} | {row['accuracy_delta']} | {row['precision_delta']} | "
            f"{row['recall_delta']} | {row['f1_delta']} |"
        )

    lines.extend(
        [
            "",
            "## Rollback policy",
            "",
            "- If `three_set_avg` degrades by more than 0.5 points, keep V10 and rollback.",
            "- Analyze degradation sources (domain, length bucket, unknown-source mix) before retrying.",
        ]
    )
    return "\n".join(lines) + "\n"


def ensure_parent(path: Path) -> None:
    """Create parent directory if missing."""
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Entry point."""
    args = parse_args()
    baseline_json = Path(args.baseline_json)
    candidate_json = Path(args.candidate_json)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)

    baseline_metrics = load_model_metrics(baseline_json, args.baseline_key)
    candidate_metrics = load_model_metrics(candidate_json, args.candidate_key)
    gate = evaluate_gate(
        baseline=baseline_metrics,
        candidate=candidate_metrics,
        max_three_set_degradation=args.max_three_set_degradation,
    )

    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "baseline_json": str(baseline_json),
        "candidate_json": str(candidate_json),
        "baseline_model_key": args.baseline_key,
        "candidate_model_key": args.candidate_key,
        "gate": gate,
    }

    ensure_parent(output_json)
    ensure_parent(output_md)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(render_markdown(report), encoding="utf-8")

    print(f"[OK] JSON report: {output_json}")
    print(f"[OK] Markdown report: {output_md}")
    print(f"[INFO] decision: {gate['decision']} ({gate['reason']})")


if __name__ == "__main__":
    main()

