#!/usr/bin/env python3
"""Evaluate one or more models on fair_test and emit gate-compatible JSON."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

EVAL_FILES: Sequence[Tuple[str, str]] = (
    ("core_v1_test_clean", "core_v1_test_clean.csv"),
    ("independent_data", "independent_data.csv"),
    ("merged_v2_val_clean", "merged_v2_val_clean.csv"),
)


class TextDataset(Dataset):
    """Simple text classification dataset wrapper."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate fair_test sets for gate comparison."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help=(
            "Model specs. Accepts `model_name` (resolved to models/<name>) or "
            "`model_name=/abs/or/rel/path`."
        ),
    )
    parser.add_argument(
        "--eval-dir",
        default=str(ROOT / "datasets/eval/fair_test"),
        help="Directory containing fair_test CSV files.",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Output comparison JSON path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Tokenizer max sequence length.",
    )
    return parser.parse_args()


def parse_model_spec(spec: str) -> tuple[str, Path]:
    """Parse one model spec string into model_key and model_path."""
    if "=" in spec:
        model_key, model_path_str = spec.split("=", 1)
        model_key = model_key.strip()
        model_path = Path(model_path_str.strip()).expanduser()
    else:
        model_key = spec.strip()
        model_path = ROOT / "models" / model_key

    if not model_key:
        raise ValueError(f"Invalid model spec: {spec}")
    return model_key, model_path


def predict_logits(
    model: BertForSequenceClassification,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect logits and labels for one dataset."""
    model.eval()
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"]
            outputs = model(input_ids=ids, attention_mask=mask)
            all_logits.append(outputs.logits.cpu())
            all_labels.append(labels)

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def eval_one_set(
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    df: pd.DataFrame,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> tuple[Dict[str, float], torch.Tensor, torch.Tensor]:
    """Evaluate one model on one dataframe."""
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    ds = TextDataset(texts=texts, labels=labels, tokenizer=tokenizer, max_len=max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    t0 = time.time()
    logits, labels_tensor = predict_logits(model=model, loader=loader, device=device)
    elapsed = time.time() - t0

    preds = logits.argmax(dim=1).numpy()
    labels_np = labels_tensor.numpy()
    acc = accuracy_score(labels_np, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels_np, preds, average="binary")

    metrics = {
        "accuracy": round(acc * 100.0, 2),
        "precision": round(p * 100.0, 2),
        "recall": round(r * 100.0, 2),
        "f1": round(f1 * 100.0, 2),
        "samples": int(len(df)),
        "time_sec": round(elapsed, 1),
    }
    return metrics, logits, labels_tensor


def eval_independent_by_source(
    df: pd.DataFrame,
    preds: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, Dict[str, float]]:
    """Compute independent_data source-wise accuracy stats."""
    labels_np = labels.numpy()
    preds_np = preds.numpy()
    out: Dict[str, Dict[str, float]] = {}
    if "source" not in df.columns:
        return out

    for source in sorted(df["source"].dropna().unique()):
        mask = df["source"].eq(source).values
        count = int(mask.sum())
        if count == 0:
            continue
        src_acc = float((preds_np[mask] == labels_np[mask]).mean() * 100.0)
        out[str(source)] = {"accuracy": round(src_acc, 2), "count": count}
    return out


def evaluate_model(
    model_key: str,
    model_path: Path,
    eval_dir: Path,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> Dict[str, object]:
    """Evaluate one model across three fair_test sets."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found for {model_key}: {model_path}")

    tokenizer = BertTokenizer.from_pretrained(str(model_path))
    model = BertForSequenceClassification.from_pretrained(str(model_path)).to(device)
    model.eval()

    result: Dict[str, object] = {"model": model_key}
    acc_list: List[float] = []

    for set_name, filename in EVAL_FILES:
        csv_path = eval_dir / filename
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing eval file: {csv_path}")

        df = pd.read_csv(csv_path, encoding="utf-8-sig").dropna(subset=["text", "label"])
        metrics, logits, labels = eval_one_set(
            model=model,
            tokenizer=tokenizer,
            df=df,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )
        result[set_name] = metrics
        acc_list.append(metrics["accuracy"])

        if set_name == "independent_data":
            preds = logits.argmax(dim=1)
            result["independent_data_by_source"] = eval_independent_by_source(df, preds, labels)
            result["independent_data_errors"] = int((preds.numpy() != labels.numpy()).sum())

    result["three_set_avg"] = round(sum(acc_list) / len(acc_list), 2) if acc_list else 0.0
    return result


def main() -> None:
    """Entry point."""
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    all_results: Dict[str, object] = {}
    for spec in args.models:
        model_key, model_path = parse_model_spec(spec)
        print(f"[INFO] evaluating {model_key} from {model_path}")
        all_results[model_key] = evaluate_model(
            model_key=model_key,
            model_path=model_path,
            eval_dir=eval_dir,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        print(
            f"[OK] {model_key}: three_set_avg={all_results[model_key]['three_set_avg']}"
        )

    output_json.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] output: {output_json}")


if __name__ == "__main__":
    main()

