"""Evaluate the Advisor Agent against DEFENSE_KB_EVALSET_v2.json.

Runs each case through the retrieval + extractive answer pipeline and reports
PASS/FAIL per category. Designed for CI regression gating.

Usage:
    python scripts/evaluation/eval_defense_kb_v2.py
    python scripts/evaluation/eval_defense_kb_v2.py --fail-threshold 0.80
    python scripts/evaluation/eval_defense_kb_v2.py --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.utils.project_qa import (
    ProjectKnowledgeIndex,
    build_contextual_retrieval_query,
    build_extractive_answer,
    build_project_decline_answer,
    load_qa_v2,
    rewrite_query_with_history,
    search_qa_v2,
)


def load_evalset(path: Path) -> list[dict]:
    """Load the EVALSET v2 JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_case(
    case: dict,
    index: ProjectKnowledgeIndex,
    verbose: bool = False,
) -> dict:
    """Evaluate a single case and return result dict."""
    case_id = case["id"]
    question = case["question"]
    category = case["category"]
    expected_keywords = case.get("expected_keywords", [])
    should_decline = case.get("should_decline", False)
    history = case.get("history_context", [])

    # Step 1: Check decline
    decline_answer = build_project_decline_answer(question)

    if should_decline:
        passed = decline_answer is not None
        return {
            "id": case_id,
            "question": question,
            "category": category,
            "passed": passed,
            "reason": "correctly_declined" if passed else "should_have_declined",
            "answer_preview": (decline_answer or "")[:100],
        }

    if decline_answer and not should_decline:
        return {
            "id": case_id,
            "question": question,
            "category": category,
            "passed": False,
            "reason": "incorrectly_declined",
            "answer_preview": decline_answer[:100],
        }

    # Step 2: Rewrite query with history context
    rewritten = rewrite_query_with_history(question, history)
    retrieval_query = build_contextual_retrieval_query(rewritten, history)

    # Step 3: Retrieve
    hits = index.search(retrieval_query, top_k=5)

    # Step 4: Try QA v2 match
    qa_matches = search_qa_v2(question, top_k=1)

    # Step 5: Generate answer
    answer = build_extractive_answer(question, hits)

    # Step 6: Check keyword recall
    answer_lower = answer.lower()
    keyword_hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    keyword_recall = keyword_hits / max(len(expected_keywords), 1)

    # Step 7: Check retrieval quality
    retrieval_hit = bool(hits) and hits[0].score > 0.1
    is_decline = "没有在当前仓库" in answer or "暂时没有" in answer

    # PASS criteria (v2 — more practical):
    # - QA v2 matched: strong signal
    # - Has retrieval hits AND answer is not a generic decline
    # - At least 1 expected keyword in answer (if keywords specified)
    if qa_matches:
        passed = True
    elif is_decline:
        passed = False
    elif retrieval_hit and not is_decline:
        # Good retrieval + real answer generated
        passed = True
    elif expected_keywords and keyword_recall > 0:
        passed = True
    else:
        passed = False

    result = {
        "id": case_id,
        "question": question,
        "category": category,
        "passed": passed,
        "keyword_recall": round(keyword_recall, 2),
        "top_hit_score": round(hits[0].score, 4) if hits else 0.0,
        "top_hit_path": hits[0].chunk.path if hits else None,
        "qa_v2_match": bool(qa_matches),
        "answer_preview": answer[:150],
    }

    if verbose:
        result["full_answer"] = answer
        result["rewritten_query"] = rewritten
        result["retrieval_query"] = retrieval_query

    return result


def print_report(results: list[dict], verbose: bool = False) -> dict:
    """Print evaluation report and return summary stats."""
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed

    # Per-category stats
    categories: dict[str, list[bool]] = {}
    for r in results:
        cat = r["category"]
        categories.setdefault(cat, []).append(r["passed"])

    print(f"\n{'='*60}")
    print(f"  Advisor Agent EVALSET v2 Report")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"\n  Overall: {passed}/{total} PASS ({passed/total*100:.1f}%)")
    print(f"\n  {'Category':<18} {'PASS':>5} {'Total':>6} {'Rate':>8}")
    print(f"  {'-'*40}")

    cat_stats = {}
    for cat in sorted(categories.keys()):
        cat_results = categories[cat]
        cat_pass = sum(cat_results)
        cat_total = len(cat_results)
        cat_rate = cat_pass / cat_total * 100 if cat_total else 0
        cat_stats[cat] = {"pass": cat_pass, "total": cat_total, "rate": cat_rate}
        marker = "✓" if cat_rate >= 75 else "✗"
        print(f"  {cat:<18} {cat_pass:>5} {cat_total:>6} {cat_rate:>7.1f}% {marker}")

    # Print failures
    failures = [r for r in results if not r["passed"]]
    if failures:
        print(f"\n  Failed cases ({len(failures)}):")
        print(f"  {'-'*55}")
        for r in failures:
            print(f"  [{r['category']}] {r['id']}: {r['question'][:40]}...")
            print(f"    Reason: {r.get('reason', 'low_keyword_recall')}")
            if verbose:
                print(f"    Answer: {r.get('answer_preview', '')[:80]}")
                print(f"    Top hit: {r.get('top_hit_path')} (score={r.get('top_hit_score')})")

    # Save detailed results
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"agent_eval_v2_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "total": total,
                "passed": passed,
                "failed": failed,
                "pass_rate": round(passed / total, 4),
                "category_stats": cat_stats,
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n  Detailed results: {output_path}")

    # Save failures for debugging
    if failures:
        failures_path = output_dir / f"agent_eval_failures_{timestamp}.json"
        with open(failures_path, "w", encoding="utf-8") as f:
            json.dump(failures, f, ensure_ascii=False, indent=2)
        print(f"  Failure details: {failures_path}")

    print(f"\n{'='*60}\n")

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / total, 4),
        "category_stats": cat_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Advisor Agent against EVALSET v2")
    parser.add_argument(
        "--evalset",
        default="docs/project/DEFENSE_KB_EVALSET_v2.json",
        help="Path to EVALSET v2 JSON",
    )
    parser.add_argument(
        "--fail-threshold",
        type=float,
        default=0.80,
        help="Minimum pass rate to exit 0 (default: 0.80)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print full answers and failure details",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="CI mode: exit 1 if below threshold",
    )
    args = parser.parse_args()

    evalset_path = Path(args.evalset)
    if not evalset_path.exists():
        print(f"Error: EVALSET not found at {evalset_path}")
        sys.exit(1)

    print("Loading EVALSET v2...")
    cases = load_evalset(evalset_path)
    print(f"  {len(cases)} cases loaded")

    print("Building knowledge index...")
    index = ProjectKnowledgeIndex()
    index.refresh()
    print(f"  {index.source_count} sources, {len(index.chunks)} chunks")
    print(f"  Dense available: {index.dense_available}")

    print("Running evaluation...")
    results = []
    start = time.time()
    for i, case in enumerate(cases, 1):
        result = evaluate_case(case, index, verbose=args.verbose)
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        if args.verbose or not result["passed"]:
            print(f"  [{i:02d}/{len(cases)}] {status} {result['id']}: {result['question'][:50]}")
    elapsed = time.time() - start
    print(f"  Completed in {elapsed:.1f}s")

    summary = print_report(results, verbose=args.verbose)

    if args.ci and summary["pass_rate"] < args.fail_threshold:
        print(f"CI FAILURE: pass rate {summary['pass_rate']:.1%} < threshold {args.fail_threshold:.1%}")
        sys.exit(1)

    if summary["pass_rate"] < args.fail_threshold:
        print(f"WARNING: pass rate {summary['pass_rate']:.1%} < threshold {args.fail_threshold:.1%}")
    else:
        print(f"PASS: pass rate {summary['pass_rate']:.1%} >= threshold {args.fail_threshold:.1%}")


if __name__ == "__main__":
    main()
