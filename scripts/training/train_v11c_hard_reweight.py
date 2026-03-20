#!/usr/bin/env python3
"""
Train a hard-reweight variant on top of the current V11c setup.

Goal:
    Keep the V11c training recipe stable while amplifying hard subsets
    identified from existing `source` metadata.

Default behavior:
    - Reuse V11c train/val data and hyperparameters
    - Continue from the current `bert_v11c_boundary_fix` checkpoint by default
    - Upweight hard subsets in the sampler
    - Keep loss weighting neutral unless explicitly enabled
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.evaluation.build_hard_set import assign_subset


DEFAULT_CONFIG: Dict[str, object] = {
    "base_model": str(ROOT / "models" / "bert_v11c_boundary_fix"),
    "train_data": str(ROOT / "datasets" / "merged_v2" / "train_v11c_candidate.csv"),
    "val_data": str(ROOT / "datasets" / "merged_v2" / "val.csv"),
    "output": str(ROOT / "models" / "bert_v11c_hard_reweight"),
    "batch_size": 8,
    "max_length": 256,
    "epochs": 5,
    "learning_rate": 1e-5,
    "label_smoothing": 0.05,
    "length_penalty_weight": 0.1,
    "accum_steps": 4,
    "patience": 2,
    "min_text_length": 10,
    "max_text_length": 5000,
    "hard_sampler_multiplier": 2.0,
    "other_sampler_multiplier": 1.15,
    "hard_loss_multiplier": 1.0,
    "other_loss_multiplier": 1.0,
    "human_hard_sampler_multiplier": None,
    "ai_hard_sampler_multiplier": None,
    "human_other_sampler_multiplier": None,
    "ai_other_sampler_multiplier": None,
    "human_hard_loss_multiplier": None,
    "ai_hard_loss_multiplier": None,
    "human_other_loss_multiplier": None,
    "ai_other_loss_multiplier": None,
    "seed": 42,
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a V11c hard-reweight variant with source-aware subset weights."
    )
    parser.add_argument("--base-model", default=DEFAULT_CONFIG["base_model"])
    parser.add_argument("--train-data", default=DEFAULT_CONFIG["train_data"])
    parser.add_argument("--val-data", default=DEFAULT_CONFIG["val_data"])
    parser.add_argument("--output", default=DEFAULT_CONFIG["output"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--max-length", type=int, default=DEFAULT_CONFIG["max_length"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--label-smoothing", type=float, default=DEFAULT_CONFIG["label_smoothing"])
    parser.add_argument(
        "--length-penalty-weight",
        type=float,
        default=DEFAULT_CONFIG["length_penalty_weight"],
    )
    parser.add_argument("--accum-steps", type=int, default=DEFAULT_CONFIG["accum_steps"])
    parser.add_argument("--patience", type=int, default=DEFAULT_CONFIG["patience"])
    parser.add_argument("--min-text-length", type=int, default=DEFAULT_CONFIG["min_text_length"])
    parser.add_argument("--max-text-length", type=int, default=DEFAULT_CONFIG["max_text_length"])
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=0,
        help="Optional cap on train rows for smoke tests (0 means full train set).",
    )
    parser.add_argument(
        "--max-val-rows",
        type=int,
        default=0,
        help="Optional cap on val rows for smoke tests (0 means full val set).",
    )
    parser.add_argument(
        "--hard-sampler-multiplier",
        type=float,
        default=DEFAULT_CONFIG["hard_sampler_multiplier"],
        help="Multiplier for human_hard / ai_hard in the sampler.",
    )
    parser.add_argument(
        "--other-sampler-multiplier",
        type=float,
        default=DEFAULT_CONFIG["other_sampler_multiplier"],
        help="Multiplier for *_other subsets in the sampler.",
    )
    parser.add_argument(
        "--human-hard-sampler-multiplier",
        type=float,
        default=DEFAULT_CONFIG["human_hard_sampler_multiplier"],
        help="Override sampler multiplier for human_hard.",
    )
    parser.add_argument(
        "--ai-hard-sampler-multiplier",
        type=float,
        default=DEFAULT_CONFIG["ai_hard_sampler_multiplier"],
        help="Override sampler multiplier for ai_hard.",
    )
    parser.add_argument(
        "--human-other-sampler-multiplier",
        type=float,
        default=DEFAULT_CONFIG["human_other_sampler_multiplier"],
        help="Override sampler multiplier for human_other.",
    )
    parser.add_argument(
        "--ai-other-sampler-multiplier",
        type=float,
        default=DEFAULT_CONFIG["ai_other_sampler_multiplier"],
        help="Override sampler multiplier for ai_other.",
    )
    parser.add_argument(
        "--hard-loss-multiplier",
        type=float,
        default=DEFAULT_CONFIG["hard_loss_multiplier"],
        help="Optional loss multiplier for human_hard / ai_hard.",
    )
    parser.add_argument(
        "--other-loss-multiplier",
        type=float,
        default=DEFAULT_CONFIG["other_loss_multiplier"],
        help="Optional loss multiplier for *_other subsets.",
    )
    parser.add_argument(
        "--human-hard-loss-multiplier",
        type=float,
        default=DEFAULT_CONFIG["human_hard_loss_multiplier"],
        help="Override loss multiplier for human_hard.",
    )
    parser.add_argument(
        "--ai-hard-loss-multiplier",
        type=float,
        default=DEFAULT_CONFIG["ai_hard_loss_multiplier"],
        help="Override loss multiplier for ai_hard.",
    )
    parser.add_argument(
        "--human-other-loss-multiplier",
        type=float,
        default=DEFAULT_CONFIG["human_other_loss_multiplier"],
        help="Override loss multiplier for human_other.",
    )
    parser.add_argument(
        "--ai-other-loss-multiplier",
        type=float,
        default=DEFAULT_CONFIG["ai_other_loss_multiplier"],
        help="Override loss multiplier for ai_other.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Print subset stats and exit without training.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_config(args: argparse.Namespace) -> Dict[str, object]:
    """Convert argparse namespace into a config dict."""
    return {
        "base_model": args.base_model,
        "train_data": args.train_data,
        "val_data": args.val_data,
        "output": args.output,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "label_smoothing": args.label_smoothing,
        "length_penalty_weight": args.length_penalty_weight,
        "accum_steps": args.accum_steps,
        "patience": args.patience,
        "min_text_length": args.min_text_length,
        "max_text_length": args.max_text_length,
        "max_train_rows": args.max_train_rows,
        "max_val_rows": args.max_val_rows,
        "hard_sampler_multiplier": args.hard_sampler_multiplier,
        "other_sampler_multiplier": args.other_sampler_multiplier,
        "hard_loss_multiplier": args.hard_loss_multiplier,
        "other_loss_multiplier": args.other_loss_multiplier,
        "human_hard_sampler_multiplier": args.human_hard_sampler_multiplier,
        "ai_hard_sampler_multiplier": args.ai_hard_sampler_multiplier,
        "human_other_sampler_multiplier": args.human_other_sampler_multiplier,
        "ai_other_sampler_multiplier": args.ai_other_sampler_multiplier,
        "human_hard_loss_multiplier": args.human_hard_loss_multiplier,
        "ai_hard_loss_multiplier": args.ai_hard_loss_multiplier,
        "human_other_loss_multiplier": args.human_other_loss_multiplier,
        "ai_other_loss_multiplier": args.ai_other_loss_multiplier,
        "seed": args.seed,
    }


def load_and_clean(path: str, min_len: int, max_len: int) -> pd.DataFrame:
    """Load CSV, clean text, and remove obvious template leakage phrases."""
    df = pd.read_csv(path)
    before = len(df)
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    df["char_len"] = df["text"].str.len()
    df = df[(df["char_len"] >= min_len) & (df["char_len"] <= max_len)]
    df = df.drop_duplicates(subset=["text"], keep="first")
    df["label"] = df["label"].astype(int)

    ai_phrases = ["作为一个AI", "作为AI", "AI语言模型", "As an AI"]
    for phrase in ai_phrases:
        df = df[~df["text"].str.contains(phrase, na=False)]

    if "source" not in df.columns:
        df["source"] = "unknown"
    else:
        df["source"] = df["source"].fillna("unknown").astype(str)

    print(f"  {path}: {before} -> {len(df)} (cleaned {before - len(df)})")
    return df.reset_index(drop=True)


def subset_weight_maps(cfg: Dict[str, object]) -> tuple[Dict[str, float], Dict[str, float]]:
    """Build exact subset weight maps with optional asymmetric overrides."""
    sampler_map = {
        "human_easy": 1.0,
        "ai_easy": 1.0,
        "human_hard": float(cfg["hard_sampler_multiplier"]),
        "ai_hard": float(cfg["hard_sampler_multiplier"]),
        "human_other": float(cfg["other_sampler_multiplier"]),
        "ai_other": float(cfg["other_sampler_multiplier"]),
    }
    loss_map = {
        "human_easy": 1.0,
        "ai_easy": 1.0,
        "human_hard": float(cfg["hard_loss_multiplier"]),
        "ai_hard": float(cfg["hard_loss_multiplier"]),
        "human_other": float(cfg["other_loss_multiplier"]),
        "ai_other": float(cfg["other_loss_multiplier"]),
    }

    override_fields = {
        "human_hard": ("human_hard_sampler_multiplier", "human_hard_loss_multiplier"),
        "ai_hard": ("ai_hard_sampler_multiplier", "ai_hard_loss_multiplier"),
        "human_other": ("human_other_sampler_multiplier", "human_other_loss_multiplier"),
        "ai_other": ("ai_other_sampler_multiplier", "ai_other_loss_multiplier"),
    }
    for subset, (sampler_key, loss_key) in override_fields.items():
        if cfg[sampler_key] is not None:
            sampler_map[subset] = float(cfg[sampler_key])
        if cfg[loss_key] is not None:
            loss_map[subset] = float(cfg[loss_key])

    return sampler_map, loss_map


def annotate_subset_weights(
    df: pd.DataFrame,
    sampler_map: Dict[str, float],
    loss_map: Dict[str, float],
) -> pd.DataFrame:
    """Annotate hard/easy subsets and associated training weights."""
    out = df.copy()
    subset_pairs = out.apply(
        lambda row: assign_subset(int(row["label"]), str(row["source"])),
        axis=1,
    )
    out["subset"] = subset_pairs.map(lambda item: item[0])
    out["subset_reason"] = subset_pairs.map(lambda item: item[1])
    out["sampler_weight"] = out["subset"].map(lambda value: float(sampler_map.get(value, 1.0)))
    out["loss_weight"] = out["subset"].map(lambda value: float(loss_map.get(value, 1.0)))
    return out


def maybe_limit_rows(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Optionally truncate rows for a faster smoke run."""
    if limit and limit > 0:
        return df.head(limit).copy().reset_index(drop=True)
    return df


class HardWeightedTextDataset(Dataset):
    """Dataset with per-sample hard weighting metadata."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        sampler_weights: List[float],
        loss_weights: List[float],
        tokenizer: BertTokenizer,
        max_len: int,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.sampler_weights = sampler_weights
        self.loss_weights = loss_weights
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.lengths = [len(str(text)) for text in texts]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "length": torch.tensor(self.lengths[idx], dtype=torch.float),
            "loss_weight": torch.tensor(self.loss_weights[idx], dtype=torch.float),
            "sampler_weight": torch.tensor(self.sampler_weights[idx], dtype=torch.float),
        }


class HardAwareLengthAwareLabelSmoothingLoss(nn.Module):
    """Length-aware label-smoothed CE with optional per-sample loss multipliers."""

    def __init__(self, label_smoothing: float = 0.05, length_penalty_weight: float = 0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(
            reduction="none",
            label_smoothing=label_smoothing,
        )
        self.length_penalty_weight = length_penalty_weight

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lengths: torch.Tensor,
        loss_weights: torch.Tensor | None = None,
        mean_length: float = 500.0,
    ) -> torch.Tensor:
        """Compute one scalar loss."""
        ce = self.ce(logits, labels)
        length_ratio = lengths / mean_length
        base_weight = 1.0 / (
            1.0 + self.length_penalty_weight * torch.abs(torch.log(length_ratio + 1e-6))
        )
        if loss_weights is not None:
            base_weight = base_weight * loss_weights
        return (ce * base_weight).mean()


def build_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    """Build the V11c sampler plus hard-subset multiplier."""
    bins = [0, 100, 200, 500, 1000, 2000, float("inf")]
    out = df.copy()
    out["len_bin"] = pd.cut(out["char_len"], bins=bins, labels=False)

    bin_weights = 1.0 / out["len_bin"].value_counts()
    label_weights = 1.0 / out["label"].value_counts()

    weights = out["len_bin"].map(bin_weights).values.astype(np.float64)
    weights *= out["label"].map(label_weights).values.astype(np.float64)
    weights *= out["sampler_weight"].values.astype(np.float64)

    return WeightedRandomSampler(weights, num_samples=len(out), replacement=True)


def train_one_epoch(
    model: BertForSequenceClassification,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    loss_fn: HardAwareLengthAwareLabelSmoothingLoss,
    accum_steps: int,
) -> tuple[float, float]:
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc="  Train")
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        lengths = batch["length"].to(device)
        loss_weights = batch["loss_weight"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels, lengths, loss_weights) / accum_steps
        loss.backward()

        if (step + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
        preds = outputs.logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (step + 1) % 500 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        pbar.set_postfix(
            loss=f"{total_loss / (step + 1):.4f}",
            acc=f"{correct / total:.4f}",
        )

    # Flush trailing gradients when len(loader) is not divisible by accum_steps
    if len(loader) > 0 and (step + 1) % accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    if len(loader) == 0:
        return 0.0, 0.0
    return total_loss / len(loader), correct / total if total > 0 else 0.0


def evaluate(
    model: BertForSequenceClassification,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float, float, float]:
    """Evaluate on one loader."""
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_lengths: List[float] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Eval"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            total_loss += ce(outputs.logits, labels).item()

            all_preds.extend(outputs.logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_lengths.extend(batch["length"].numpy())

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    lengths = np.array(all_lengths)

    if len(loader) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    loss = total_loss / len(loader)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
    )

    print("\n  Length-segmented accuracy:")
    for name, lower, upper in [
        ("Short <128", 0, 128),
        ("128-256", 128, 256),
        ("256-512", 256, 512),
        ("512+", 512, 99999),
    ]:
        mask = (lengths >= lower) & (lengths < upper)
        if mask.sum() > 0:
            print(f"    {name}: {(preds[mask] == labels[mask]).mean():.4f} ({mask.sum()} samples)")

    return loss, acc, precision, recall, f1


def subset_summary(df: pd.DataFrame) -> Dict[str, object]:
    """Collect subset and source diagnostics."""
    return {
        "subset_counts": df["subset"].value_counts().sort_index().to_dict(),
        "subset_by_label": (
            df.groupby(["subset", "label"], dropna=False)
            .size()
            .reset_index(name="count")
            .to_dict(orient="records")
        ),
        "top_sources": df["source"].value_counts().head(20).to_dict(),
        "sampler_weight_stats": (
            df.groupby("subset", dropna=False)["sampler_weight"]
            .agg(["count", "mean", "min", "max"])
            .round(4)
            .reset_index()
            .to_dict(orient="records")
        ),
        "loss_weight_stats": (
            df.groupby("subset", dropna=False)["loss_weight"]
            .agg(["count", "mean", "min", "max"])
            .round(4)
            .reset_index()
            .to_dict(orient="records")
        ),
    }


def main() -> None:
    """Entry point."""
    args = parse_args()
    cfg = build_config(args)
    set_seed(int(cfg["seed"]))

    print("=" * 70)
    print("  V11c Hard-Reweight Training")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    effective_batch = int(cfg["batch_size"]) * int(cfg["accum_steps"])
    print("\nConfig:")
    for key, value in cfg.items():
        print(f"  {key}: {value}")
    print(f"  effective_batch: {effective_batch}")

    print(f"\n>>> Base model configured: {cfg['base_model']}")

    print("\n>>> Loading data")
    train_df = load_and_clean(
        str(cfg["train_data"]),
        int(cfg["min_text_length"]),
        int(cfg["max_text_length"]),
    )
    val_df = load_and_clean(
        str(cfg["val_data"]),
        int(cfg["min_text_length"]),
        int(cfg["max_text_length"]),
    )

    train_sampler_map, train_loss_map = subset_weight_maps(cfg)
    neutral_sampler_map = {subset: 1.0 for subset in train_sampler_map}
    neutral_loss_map = {subset: 1.0 for subset in train_loss_map}

    print("\nEffective subset weights:")
    print(f"  sampler: {train_sampler_map}")
    print(f"  loss:    {train_loss_map}")

    train_df = annotate_subset_weights(
        train_df,
        sampler_map=train_sampler_map,
        loss_map=train_loss_map,
    )
    val_df = annotate_subset_weights(
        val_df,
        sampler_map=neutral_sampler_map,
        loss_map=neutral_loss_map,
    )
    train_df = maybe_limit_rows(train_df, int(cfg["max_train_rows"]))
    val_df = maybe_limit_rows(val_df, int(cfg["max_val_rows"]))

    print("\nData stats:")
    print(
        f"  Train: {len(train_df)} (AI: {(train_df['label'] == 1).sum()}, "
        f"Human: {(train_df['label'] == 0).sum()})"
    )
    print(
        f"  Val:   {len(val_df)} (AI: {(val_df['label'] == 1).sum()}, "
        f"Human: {(val_df['label'] == 0).sum()})"
    )

    print("\nTrain subset counts:")
    for subset, count in train_df["subset"].value_counts().sort_index().items():
        print(f"  {subset}: {count}")

    if args.inspect_only:
        print("\nInspect-only mode: no training started.")
        return

    base_model_ref = str(cfg["base_model"])
    if ("/" in base_model_ref or "\\" in base_model_ref or base_model_ref.startswith(".")) and not Path(
        base_model_ref
    ).exists():
        raise FileNotFoundError(
            f"Base model path not found: {base_model_ref}. "
            "Pass --base-model with an existing local checkpoint or a cached HF model id."
        )

    print(f"\n>>> Loading tokenizer/model from: {cfg['base_model']}")
    tokenizer = BertTokenizer.from_pretrained(base_model_ref)

    train_ds = HardWeightedTextDataset(
        texts=train_df["text"].tolist(),
        labels=train_df["label"].tolist(),
        sampler_weights=train_df["sampler_weight"].tolist(),
        loss_weights=train_df["loss_weight"].tolist(),
        tokenizer=tokenizer,
        max_len=int(cfg["max_length"]),
    )
    val_ds = HardWeightedTextDataset(
        texts=val_df["text"].tolist(),
        labels=val_df["label"].tolist(),
        sampler_weights=[1.0] * len(val_df),
        loss_weights=[1.0] * len(val_df),
        tokenizer=tokenizer,
        max_len=int(cfg["max_length"]),
    )

    sampler = build_sampler(train_df)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        sampler=sampler,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=0,
    )

    model = BertForSequenceClassification.from_pretrained(
        base_model_ref,
        num_labels=2,
    ).to(device)
    loss_fn = HardAwareLengthAwareLabelSmoothingLoss(
        label_smoothing=float(cfg["label_smoothing"]),
        length_penalty_weight=float(cfg["length_penalty_weight"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=0.01,
    )

    total_steps = len(train_loader) * int(cfg["epochs"]) // int(cfg["accum_steps"])
    warmup_steps = total_steps // 10
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    output_dir = Path(str(cfg["output"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  Start training")
    print("=" * 70)

    best_acc = 0.0
    patience_counter = 0
    epoch_results: List[Dict[str, object]] = []

    for epoch in range(int(cfg["epochs"])):
        started = time.time()
        print(f"\n{'─' * 70}")
        print(f"  Epoch {epoch + 1}/{cfg['epochs']}")
        print(f"{'─' * 70}")

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            loss_fn=loss_fn,
            accum_steps=int(cfg["accum_steps"]),
        )
        val_loss, val_acc, val_p, val_r, val_f1 = evaluate(model, val_loader, device)
        elapsed = time.time() - started

        print(f"\n  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        print(f"          P: {val_p:.4f}, R: {val_r:.4f}, F1: {val_f1:.4f}")
        print(f"  Time: {elapsed:.1f}s")

        epoch_results.append(
            {
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 4),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 4),
                "val_precision": round(val_p, 4),
                "val_recall": round(val_r, 4),
                "val_f1": round(val_f1, 4),
                "time_sec": round(elapsed, 1),
            }
        )

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            model.save_pretrained(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))
            print(f"  >>> Saved best model (Val Acc={val_acc:.4f})")
        else:
            patience_counter += 1
            print(f"  Val Acc not improved ({patience_counter}/{cfg['patience']})")
            if patience_counter >= int(cfg["patience"]):
                print(f"\n  Early stopping at epoch {epoch + 1}")
                break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    log = {
        "version": "v11c_hard_reweight",
        "strategy": "v11c_sampler_reweight_with_optional_loss_weight",
        "description": (
            "Reuse V11c training config and amplify hard subsets inferred from source tags. "
            "Default is sampler-only hard reweight."
        ),
        "base_model": cfg["base_model"],
        "config": cfg,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "train_subset_summary": subset_summary(train_df),
        "val_subset_summary": subset_summary(val_df),
        "best_val_acc": round(best_acc, 5),
        "results": epoch_results,
    }
    log_path = output_dir / "training_log.json"
    with log_path.open("w", encoding="utf-8") as handle:
        json.dump(log, handle, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"  Training complete! Best Val Acc: {best_acc:.4f}")
    print(f"  Model: {output_dir}")
    print(f"  Log: {log_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
