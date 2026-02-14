#!/usr/bin/env python3
"""Train V11d with configurable mode (fast patch or full retrain)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.training import train_v11c


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train V11d model.")
    parser.add_argument(
        "--mode",
        choices=["fast", "full"],
        default="fast",
        help="fast: continue from V11c for quick validation; full: retrain from V7 baseline.",
    )
    parser.add_argument(
        "--train-data",
        default=str(ROOT / "datasets/merged_v2/train_v11d_candidate.csv"),
        help="Training CSV path.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output model path. Defaults depend on mode.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate.",
    )
    return parser.parse_args()


def main() -> None:
    """Reuse V11c trainer with V11d data and mode-specific config."""
    args = parse_args()

    train_v11c.CONFIG["train_data"] = str(Path(args.train_data))

    if args.mode == "fast":
        train_v11c.CONFIG["base_model"] = str(ROOT / "models/bert_v11c_boundary_fix")
        train_v11c.CONFIG["epochs"] = 1
        train_v11c.CONFIG["learning_rate"] = 5e-6
        train_v11c.CONFIG["patience"] = 1
        default_output = ROOT / "models/bert_v11d_gemini_patch_fast"
    else:
        train_v11c.CONFIG["base_model"] = str(ROOT / "models/bert_v7_improved")
        train_v11c.CONFIG["epochs"] = 5
        train_v11c.CONFIG["learning_rate"] = 1e-5
        train_v11c.CONFIG["patience"] = 2
        default_output = ROOT / "models/bert_v11d_gemini_patch"

    if args.epochs is not None:
        train_v11c.CONFIG["epochs"] = int(args.epochs)
    if args.learning_rate is not None:
        train_v11c.CONFIG["learning_rate"] = float(args.learning_rate)

    train_v11c.CONFIG["output"] = str(Path(args.output) if args.output else default_output)

    print(
        "[INFO] V11d training wrapper\n"
        f"  mode={args.mode}\n"
        f"  base_model={train_v11c.CONFIG['base_model']}\n"
        f"  train_data={train_v11c.CONFIG['train_data']}\n"
        f"  epochs={train_v11c.CONFIG['epochs']}\n"
        f"  learning_rate={train_v11c.CONFIG['learning_rate']}\n"
        f"  output={train_v11c.CONFIG['output']}"
    )
    train_v11c.main()


if __name__ == "__main__":
    main()
