#!/usr/bin/env python3
"""
Evaluate AI-text detectors with sliding-window aggregation for long texts.

Supported aggregation methods:
    - first
    - mean
    - max
    - topk_mean
    - mean_logit
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATASETS: Sequence[Tuple[str, str]] = (
    ("core_v1_test_clean", "datasets/eval/fair_test/core_v1_test_clean.csv"),
    ("independent_data", "datasets/eval/fair_test/independent_data.csv"),
    ("merged_v2_val_clean", "datasets/eval/fair_test/merged_v2_val_clean.csv"),
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate one model with sliding-window aggregation on one or more CSV datasets."
    )
    parser.add_argument(
        "--model-path",
        default=str(ROOT / "models" / "bert_v11c_boundary_fix"),
        help="Path to the classification model directory.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help=(
            "Dataset spec in the form name=relative_or_absolute_csv_path. "
            "Can be passed multiple times. Defaults to the three fair_test CSVs."
        ),
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=256,
        help="Token window size including special tokens.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=128,
        help="Sliding stride in tokens before adding special tokens.",
    )
    parser.add_argument(
        "--aggregation",
        choices=["first", "mean", "max", "topk_mean", "mean_logit"],
        default="topk_mean",
        help="Window aggregation strategy.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="Top-k windows used by topk_mean aggregation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for window inference.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device.",
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=5,
        help="Minimum count before exporting per-group metrics.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional per-dataset row limit for quick debugging (0 means full dataset).",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional explicit JSON output path. Defaults to evaluation_results/.",
    )
    parser.add_argument(
        "--output-predictions",
        default="",
        help="Optional explicit CSV path for per-text predictions. Defaults to evaluation_results/.",
    )
    return parser.parse_args()


def resolve_datasets(dataset_args: Sequence[str]) -> List[Tuple[str, Path]]:
    """Resolve datasets from CLI args or defaults."""
    if not dataset_args:
        return [(name, ROOT / rel_path) for name, rel_path in DEFAULT_DATASETS]

    datasets: List[Tuple[str, Path]] = []
    for item in dataset_args:
        if "=" not in item:
            raise ValueError(f"Invalid --dataset spec: {item}")
        name, raw_path = item.split("=", 1)
        path = Path(raw_path)
        if not path.is_absolute():
            path = ROOT / raw_path
        datasets.append((name, path))
    return datasets


def load_dataset(path: Path, limit: int = 0) -> pd.DataFrame:
    """Load one CSV dataset for evaluation."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception as exc:
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc

    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    out = df.copy()
    out["text"] = out["text"].astype(str)
    out["label"] = out["label"].astype(int)
    out["char_len"] = out["text"].str.len()
    if limit > 0:
        out = out.head(limit).copy()
    return out.reset_index(drop=True)


def compute_metrics(labels: Sequence[int], preds: Sequence[int]) -> Dict[str, object]:
    """Compute binary classification metrics."""
    labels_np = np.array(labels, dtype=int)
    preds_np = np.array(preds, dtype=int)
    accuracy = accuracy_score(labels_np, preds_np)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_np,
        preds_np,
        average="binary",
        zero_division=0,
    )
    cm = confusion_matrix(labels_np, preds_np, labels=[0, 1])
    return {
        "accuracy": round(float(accuracy) * 100, 2),
        "precision": round(float(precision) * 100, 2),
        "recall": round(float(recall) * 100, 2),
        "f1": round(float(f1) * 100, 2),
        "samples": int(len(labels_np)),
        "tn": int(cm[0][0]),
        "fp": int(cm[0][1]),
        "fn": int(cm[1][0]),
        "tp": int(cm[1][1]),
    }


class SlidingWindowPredictor:
    """Run BERT classification on one text via multiple windows."""

    def __init__(
        self,
        model: BertForSequenceClassification,
        tokenizer: BertTokenizer,
        device: torch.device,
        window_size: int,
        stride: int,
        aggregation: str,
        topk: int,
        batch_size: int,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.window_size = window_size
        self.stride = stride
        self.aggregation = aggregation
        self.topk = topk
        self.batch_size = batch_size
        self.special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
        self.payload_size = max(1, window_size - self.special_tokens)
        if getattr(self.tokenizer, "model_max_length", 0) < 1_000_000:
            self.tokenizer.model_max_length = 1_000_000

    def make_windows(self, text: str) -> List[Dict[str, object]]:
        """Split one text into token windows."""
        encoded_text = self.tokenizer(
            str(text),
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            verbose=False,
        )
        token_ids = encoded_text["input_ids"]
        if not token_ids:
            token_ids = []

        windows: List[Dict[str, object]] = []
        start = 0
        while True:
            end = start + self.payload_size
            chunk = token_ids[start:end]
            encoded = self.tokenizer.prepare_for_model(
                chunk,
                max_length=self.window_size,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_token_type_ids=True,
            )
            windows.append(
                {
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                    "token_type_ids": encoded.get("token_type_ids"),
                    "start_token": start,
                    "end_token": min(len(token_ids), end),
                }
            )
            if end >= len(token_ids):
                break
            start += max(1, self.stride)
        return windows

    def run_windows(self, windows: Sequence[Dict[str, object]]) -> torch.Tensor:
        """Infer logits for a sequence of windows."""
        all_logits: List[torch.Tensor] = []
        for start in range(0, len(windows), self.batch_size):
            batch = windows[start : start + self.batch_size]
            inputs: Dict[str, torch.Tensor] = {
                "input_ids": torch.tensor(
                    [item["input_ids"] for item in batch],
                    dtype=torch.long,
                    device=self.device,
                ),
                "attention_mask": torch.tensor(
                    [item["attention_mask"] for item in batch],
                    dtype=torch.long,
                    device=self.device,
                ),
            }
            token_type_ids = [item["token_type_ids"] for item in batch]
            if token_type_ids and token_type_ids[0] is not None:
                inputs["token_type_ids"] = torch.tensor(
                    token_type_ids,
                    dtype=torch.long,
                    device=self.device,
                )

            with torch.no_grad():
                outputs = self.model(**inputs)
            all_logits.append(outputs.logits.cpu())
        return torch.cat(all_logits, dim=0)

    def aggregate_probability(self, logits: torch.Tensor) -> float:
        """Aggregate per-window logits into one AI probability."""
        probs = torch.softmax(logits, dim=1)[:, 1]
        if self.aggregation == "first":
            return float(probs[0].item())
        if self.aggregation == "mean":
            return float(probs.mean().item())
        if self.aggregation == "max":
            return float(probs.max().item())
        if self.aggregation == "topk_mean":
            k = max(1, min(self.topk, len(probs)))
            values = torch.topk(probs, k=k).values
            return float(values.mean().item())
        if self.aggregation == "mean_logit":
            mean_logits = logits.mean(dim=0, keepdim=True)
            return float(torch.softmax(mean_logits, dim=1)[0, 1].item())
        raise ValueError(f"Unsupported aggregation: {self.aggregation}")

    def predict_text(self, text: str) -> Dict[str, object]:
        """Predict one text using window aggregation."""
        windows = self.make_windows(text)
        logits = self.run_windows(windows)
        probs = torch.softmax(logits, dim=1)[:, 1]
        first_ai_prob = float(probs[0].item())
        agg_ai_prob = self.aggregate_probability(logits)
        return {
            "pred": int(agg_ai_prob >= 0.5),
            "ai_prob": round(agg_ai_prob, 6),
            "first_pred": int(first_ai_prob >= 0.5),
            "first_ai_prob": round(first_ai_prob, 6),
            "window_count": int(len(windows)),
            "max_window_ai_prob": round(float(probs.max().item()), 6),
            "mean_window_ai_prob": round(float(probs.mean().item()), 6),
        }


def dataset_group_metrics(
    df: pd.DataFrame,
    group_col: str,
    pred_col: str,
    min_group_size: int,
) -> Dict[str, Dict[str, object]]:
    """Compute metrics for one grouping column."""
    if group_col not in df.columns:
        return {}

    result: Dict[str, Dict[str, object]] = {}
    grouped = df.groupby(group_col, dropna=False)
    for group_name, part in grouped:
        if len(part) < min_group_size:
            continue
        result[str(group_name)] = compute_metrics(part["label"], part[pred_col])
    return result


def evaluate_dataset(
    dataset_name: str,
    df: pd.DataFrame,
    predictor: SlidingWindowPredictor,
    min_group_size: int,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    """Evaluate one dataset and return summary plus per-text predictions."""
    rows: List[Dict[str, object]] = []
    started = time.time()

    iterator = tqdm(
        df.to_dict(orient="records"),
        desc=f"Eval {dataset_name}",
        leave=False,
    )
    for record in iterator:
        pred_result = predictor.predict_text(record["text"])
        row = dict(record)
        row.update(pred_result)
        row["dataset_name"] = dataset_name
        rows.append(row)

    pred_df = pd.DataFrame(rows)
    elapsed = time.time() - started

    overall = compute_metrics(pred_df["label"], pred_df["pred"])
    first_window = compute_metrics(pred_df["label"], pred_df["first_pred"])
    changed = int((pred_df["pred"] != pred_df["first_pred"]).sum())
    improved = int(
        ((pred_df["pred"] == pred_df["label"]) & (pred_df["first_pred"] != pred_df["label"])).sum()
    )
    worsened = int(
        ((pred_df["pred"] != pred_df["label"]) & (pred_df["first_pred"] == pred_df["label"])).sum()
    )

    summary: Dict[str, object] = {
        "dataset_name": dataset_name,
        "elapsed_sec": round(elapsed, 1),
        "overall": overall,
        "first_window_baseline": first_window,
        "changed_predictions": changed,
        "improved_vs_first": improved,
        "worsened_vs_first": worsened,
        "avg_char_len": round(float(pred_df["char_len"].mean()), 1),
        "avg_window_count": round(float(pred_df["window_count"].mean()), 2),
        "max_window_count": int(pred_df["window_count"].max()),
    }

    for group_col in ("subset", "source", "category"):
        group_metrics = dataset_group_metrics(pred_df, group_col, "pred", min_group_size)
        if group_metrics:
            summary[f"by_{group_col}"] = group_metrics

    return summary, pred_df


def default_json_path(args: argparse.Namespace) -> Path:
    """Generate default JSON output path."""
    model_name = Path(args.model_path).name
    out_dir = ROOT / "evaluation_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    file_name = (
        f"sliding_window_eval_{model_name}_{args.aggregation}"
        f"_w{args.window_size}_s{args.stride}.json"
    )
    return out_dir / file_name


def default_predictions_path(args: argparse.Namespace) -> Path:
    """Generate default predictions CSV path."""
    model_name = Path(args.model_path).name
    out_dir = ROOT / "evaluation_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    file_name = (
        f"sliding_window_preds_{model_name}_{args.aggregation}"
        f"_w{args.window_size}_s{args.stride}.csv"
    )
    return out_dir / file_name


def main() -> None:
    """Entry point."""
    args = parse_args()
    datasets = resolve_datasets(args.dataset)

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = ROOT / args.model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    device = torch.device(args.device)
    print(f"Loading model from: {model_path}")
    print(f"Device: {device}")
    print(
        f"Aggregation: {args.aggregation}, window={args.window_size}, "
        f"stride={args.stride}, topk={args.topk}"
    )

    tokenizer = BertTokenizer.from_pretrained(str(model_path))
    model = BertForSequenceClassification.from_pretrained(str(model_path)).to(device)
    model.eval()

    predictor = SlidingWindowPredictor(
        model=model,
        tokenizer=tokenizer,
        device=device,
        window_size=args.window_size,
        stride=args.stride,
        aggregation=args.aggregation,
        topk=args.topk,
        batch_size=args.batch_size,
    )

    summaries: Dict[str, object] = {
        "model_path": str(model_path),
        "aggregation": args.aggregation,
        "window_size": args.window_size,
        "stride": args.stride,
        "topk": args.topk,
        "device": str(device),
        "datasets": {},
    }
    prediction_frames: List[pd.DataFrame] = []

    for dataset_name, dataset_path in datasets:
        print(f"\nEvaluating dataset: {dataset_name} ({dataset_path})")
        df = load_dataset(dataset_path, limit=args.limit)
        summary, pred_df = evaluate_dataset(
            dataset_name=dataset_name,
            df=df,
            predictor=predictor,
            min_group_size=args.min_group_size,
        )
        summaries["datasets"][dataset_name] = summary
        prediction_frames.append(pred_df)
        print(
            f"  sliding={summary['overall']['accuracy']:.2f}% "
            f"vs first={summary['first_window_baseline']['accuracy']:.2f}% "
            f"changed={summary['changed_predictions']}"
        )

    all_predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()

    output_json = Path(args.output_json) if args.output_json else default_json_path(args)
    output_preds = (
        Path(args.output_predictions)
        if args.output_predictions
        else default_predictions_path(args)
    )
    if not output_json.is_absolute():
        output_json = ROOT / output_json
    if not output_preds.is_absolute():
        output_preds = ROOT / output_preds
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_preds.parent.mkdir(parents=True, exist_ok=True)

    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2, ensure_ascii=False)
    all_predictions.to_csv(output_preds, index=False, encoding="utf-8-sig")

    print("\nSaved results:")
    print(f"  JSON: {output_json}")
    print(f"  CSV : {output_preds}")


if __name__ == "__main__":
    main()
