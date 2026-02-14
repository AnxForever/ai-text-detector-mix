"""A/B verify base model bias on formal Human vs casual/defective AI.

This script compares two models on targeted subsets to validate whether a base
model has learned superficial shortcuts (e.g., "fluent/complete => AI").
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer

# Add project root to sys.path for local imports (per AGENTS guidelines).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))


@dataclass
class DatasetSlice:
    name: str
    texts: List[str]
    labels: List[int]


def load_jsonl_texts(
    path: Path,
    label: int,
    max_samples: int | None = None,
    text_fields: Tuple[str, ...] = ("text", "content"),
) -> DatasetSlice:
    """Load a JSONL file into (texts, labels) with a fixed label."""
    texts: List[str] = []
    labels: List[int] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = ""
            for field in text_fields:
                if isinstance(obj.get(field), str):
                    text = obj[field]
                    break
            if not text:
                continue
            texts.append(text)
            labels.append(label)
            if max_samples and len(texts) >= max_samples:
                break
    return DatasetSlice(name=path.name, texts=texts, labels=labels)


def load_core_v2_category(
    core_v2_dir: Path,
    categories: Iterable[str],
    label: int,
    max_samples: int | None = None,
) -> DatasetSlice:
    """Load texts from core_v2 train/val/test filtered by category and label."""
    frames: List[pd.DataFrame] = []
    for split in ("train.csv", "val.csv", "test.csv"):
        path = core_v2_dir / split
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"No core_v2 CSVs found in {core_v2_dir}")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["label"] == label]
    df = df[df["category"].isin(list(categories))]
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    return DatasetSlice(name=f"core_v2_{'_'.join(categories)}", texts=texts, labels=labels)


def batch_iter(items: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def evaluate_model(
    model_path: Path,
    tokenizer: BertTokenizer,
    dataset: DatasetSlice,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> Dict[str, float]:
    """Evaluate model and return metrics on a dataset slice."""
    model = BertForSequenceClassification.from_pretrained(str(model_path)).to(device)
    model.eval()

    preds: List[int] = []
    labels = dataset.labels

    with torch.no_grad():
        for batch_texts in tqdm(
            list(batch_iter(dataset.texts, batch_size)),
            desc=f"{model_path.name}:{dataset.name}",
        ):
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            outputs = model(**enc)
            batch_preds = outputs.logits.argmax(dim=-1).cpu().tolist()
            preds.extend(batch_preds)

    total = len(labels)
    if total == 0:
        return {"total": 0}
    correct = sum(int(p == y) for p, y in zip(preds, labels))
    acc = correct / total

    # Class-specific rates
    human_total = sum(1 for y in labels if y == 0)
    ai_total = sum(1 for y in labels if y == 1)
    human_correct = sum(1 for p, y in zip(preds, labels) if y == 0 and p == 0)
    ai_correct = sum(1 for p, y in zip(preds, labels) if y == 1 and p == 1)

    metrics: Dict[str, float] = {
        "total": total,
        "accuracy": acc,
    }
    if human_total:
        metrics["human_acc"] = human_correct / human_total
        metrics["human_fp_rate"] = 1.0 - metrics["human_acc"]
    if ai_total:
        metrics["ai_acc"] = ai_correct / ai_total
        metrics["ai_fn_rate"] = 1.0 - metrics["ai_acc"]

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify base model bias via A/B test")
    parser.add_argument(
        "--model-a",
        default="models/bert_v2_with_sep",
        help="Model A path (base model to verify)",
    )
    parser.add_argument(
        "--model-b",
        default="models/bert_v2_balanced",
        help="Model B path (comparison model)",
    )
    parser.add_argument(
        "--formal-human",
        default="datasets/human_formal_samples.jsonl",
        help="Formal Human JSONL path",
    )
    parser.add_argument(
        "--core-v2-dir",
        default="datasets/active/core_v2",
        help="core_v2 directory containing train/val/test CSVs",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--output",
        default="evaluation_results/base_model_bias_ab.json",
        help="Output JSON path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_samples = args.max_samples if args.max_samples > 0 else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    formal_human = load_jsonl_texts(
        Path(args.formal_human),
        label=0,
        max_samples=max_samples,
    )
    casual_ai = load_core_v2_category(
        Path(args.core_v2_dir),
        categories=["casual_ai"],
        label=1,
        max_samples=max_samples,
    )
    defective_ai = load_core_v2_category(
        Path(args.core_v2_dir),
        categories=["defective_ai"],
        label=1,
        max_samples=max_samples,
    )

    datasets = [formal_human, casual_ai, defective_ai]

    results: Dict[str, Dict[str, Dict[str, float]]] = {
        "model_a": {},
        "model_b": {},
    }

    for model_key, model_path in (
        ("model_a", Path(args.model_a)),
        ("model_b", Path(args.model_b)),
    ):
        for ds in datasets:
            metrics = evaluate_model(
                model_path,
                tokenizer,
                ds,
                device,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )
            results[model_key][ds.name] = metrics

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== A/B bias check summary ===")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nOK Results saved: {output_path}")


if __name__ == "__main__":
    main()
