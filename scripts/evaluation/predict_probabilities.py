#!/usr/bin/env python3
"""Generate prediction probabilities and append pred_ai_prob column."""

from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path
from typing import List

import pandas as pd
import torch
from transformers import BertForSequenceClassification, BertTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate pred_ai_prob for dataset.")
    parser.add_argument(
        "--model-path",
        default=str(PROJECT_ROOT / "models" / "bert_improved" / "best_model"),
        help="Path to model directory.",
    )
    parser.add_argument(
        "--input",
        default="",
        help="Input CSV file.",
    )
    parser.add_argument(
        "--input-dir",
        default="",
        help="Input directory containing train/val/test CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "datasets" / "pred_probs"),
        help="Output directory for CSV files with pred_ai_prob column.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap on number of samples (0 = all).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device selection. auto uses cuda if available.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=2,
        help="Limit CPU threads (affects OMP/MKL).",
    )
    parser.add_argument(
        "--sleep-ms",
        type=int,
        default=0,
        help="Sleep between batches to reduce GPU utilization (milliseconds).",
    )
    return parser.parse_args()


def load_model(
    model_path: Path, device_choice: str
) -> tuple[BertForSequenceClassification, BertTokenizer, str]:
    """Load model and tokenizer."""
    if device_choice == "cuda":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device_choice == "cpu":
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def predict_probs(
    texts: List[str],
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    device: str,
    batch_size: int,
    sleep_ms: int,
) -> List[float]:
    """Predict AI probabilities for a list of texts."""
    probs: List[float] = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encodings = tokenizer(
            batch_texts,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            probs.extend(batch_probs.tolist())
        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)
    return probs


def process_file(
    input_path: Path,
    output_path: Path,
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    device: str,
    batch_size: int,
    max_samples: int,
    sleep_ms: int,
) -> None:
    """Process a single CSV file and write output with pred_ai_prob."""
    df = pd.read_csv(input_path)
    if "text" not in df.columns:
        raise RuntimeError(f"Missing text column in {input_path}")

    if max_samples and len(df) > max_samples:
        df = df.iloc[:max_samples].copy()

    texts = df["text"].astype(str).tolist()
    df["pred_ai_prob"] = predict_probs(texts, model, tokenizer, device, batch_size, sleep_ms)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    """Main entry point."""
    args = parse_args()
    model_path = Path(args.model_path)
    # Limit CPU threads to avoid saturating the machine
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
        torch.set_num_interop_threads(max(1, args.num_threads // 2))
    model, tokenizer, device = load_model(model_path, args.device)

    if args.input_dir:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir) / input_dir.name
        for split in ["train.csv", "val.csv", "test.csv"]:
            input_path = input_dir / split
            if input_path.exists():
                output_path = output_dir / split
                process_file(
                    input_path,
                    output_path,
                    model,
                    tokenizer,
                    device,
                    args.batch_size,
                    args.max_samples,
                    args.sleep_ms,
                )
        print(str(output_dir))
        return

    if args.input:
        input_path = Path(args.input)
        output_dir = Path(args.output_dir)
        output_path = output_dir / input_path.name
        process_file(
            input_path,
            output_path,
            model,
            tokenizer,
            device,
            args.batch_size,
            args.max_samples,
            args.sleep_ms,
        )
        print(str(output_path))
        return

    raise ValueError("Either --input or --input-dir must be provided.")


if __name__ == "__main__":
    main()
