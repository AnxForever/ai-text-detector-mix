"""Compare FP32 vs INT8 quantized classifier on the v1 held-out test split.

Loads both ``bert_v11c_boundary_fix`` (FP32) and ``bert_v11c_int8`` (INT8) and
runs them over the same texts, then reports the accuracy / F1 delta and the
wall-clock speedup.  Padding matches production (``padding=True`` dynamic).

Run from the repo root:
    python scripts/evaluation/compare_quant_accuracy.py
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertForSequenceClassification, BertTokenizer

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
FP32_PATH = ROOT / "models" / "bert_v11c_boundary_fix"
INT8_PATH = ROOT / "models" / "bert_v11c_int8"
TEST_CSV = ROOT / "datasets" / "eval" / "splits" / "v1" / "id_test_final_clean_phrase_clean.csv"

MAX_LENGTH = 256
BATCH_SIZE = 8
TEMPERATURE = 0.8165


def load_int8_model(path: Path) -> torch.nn.Module:
    model = torch.load(path / "quantized_model.pt", map_location="cpu", weights_only=False)
    model.eval()
    return model


def run_model(
    model: torch.nn.Module, tokenizer: BertTokenizer, texts: list[str]
) -> tuple[list[int], float]:
    preds: list[int] = []
    total = 0.0
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            enc = tokenizer(
                batch,
                max_length=MAX_LENGTH,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            start = time.time()
            out = model(**enc)
            total += time.time() - start
            scaled = out.logits / TEMPERATURE
            preds.extend(scaled.argmax(dim=-1).tolist())
    return preds, total


def main() -> None:
    logger.info("Loading test CSV: %s", TEST_CSV.name)
    df = pd.read_csv(TEST_CSV).dropna(subset=["text", "label"])
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    logger.info("  N = %d   balance = %s", len(texts), df["label"].value_counts().to_dict())

    logger.info("\nTokenizer + FP32 model: %s", FP32_PATH.name)
    tokenizer = BertTokenizer.from_pretrained(str(FP32_PATH))
    fp32_model = BertForSequenceClassification.from_pretrained(str(FP32_PATH))

    logger.info("INT8 model: %s", INT8_PATH.name)
    int8_model = load_int8_model(INT8_PATH)

    logger.info("\n===== FP32 forward =====")
    fp32_preds, fp32_time = run_model(fp32_model, tokenizer, texts)
    fp32_acc = accuracy_score(labels, fp32_preds)
    fp32_f1 = f1_score(labels, fp32_preds, average="binary")
    logger.info("  accuracy=%.4f   f1=%.4f   forward-time=%.1fs", fp32_acc, fp32_f1, fp32_time)

    logger.info("\n===== INT8 forward =====")
    int8_preds, int8_time = run_model(int8_model, tokenizer, texts)
    int8_acc = accuracy_score(labels, int8_preds)
    int8_f1 = f1_score(labels, int8_preds, average="binary")
    logger.info("  accuracy=%.4f   f1=%.4f   forward-time=%.1fs", int8_acc, int8_f1, int8_time)

    logger.info("\n===== Summary =====")
    logger.info(
        "  accuracy: FP32 %.4f -> INT8 %.4f   Δ=%+.4f",
        fp32_acc,
        int8_acc,
        int8_acc - fp32_acc,
    )
    logger.info(
        "  f1      : FP32 %.4f -> INT8 %.4f   Δ=%+.4f",
        fp32_f1,
        int8_f1,
        int8_f1 - fp32_f1,
    )
    speedup = fp32_time / int8_time if int8_time else float("inf")
    logger.info(
        "  time    : FP32 %.1fs -> INT8 %.1fs   speedup=%.2fx",
        fp32_time,
        int8_time,
        speedup,
    )

    # Agreement
    agree = sum(1 for a, b in zip(fp32_preds, int8_preds, strict=True) if a == b)
    logger.info("  agreement: %d / %d (%.2f%%)", agree, len(texts), 100.0 * agree / len(texts))


if __name__ == "__main__":
    main()
