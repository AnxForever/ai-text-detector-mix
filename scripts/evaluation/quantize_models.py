"""Dynamic INT8 quantization for the two BERT models used by the detector.

Dynamic quantization converts `nn.Linear` layer weights to INT8 at save time
while keeping activations in FP32 (quantized on the fly during forward).

Expected gains on CPU-only servers:
  * ~4x smaller on-disk footprint (~400MB -> ~100MB per model).
  * 1.3-1.8x faster inference without AVX-512 VNNI; 2-3x with VNNI.
  * Accuracy drop typically <0.5% for BERT sequence/token classification.

Output format:
  - ``quantized_model.pt``  whole module saved via ``torch.save`` so we can
    round-trip without rebuilding the quantized wrapper ourselves.
  - Original ``config.json`` / tokenizer files are copied alongside so the
    tokenizer still loads via ``AutoTokenizer.from_pretrained``.

Safetensors cannot serialize quantized tensors, hence the plain ``.pt`` file.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import torch
from transformers import BertForSequenceClassification, BertForTokenClassification

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

TOKENIZER_FILES = (
    "config.json",
    "tokenizer_config.json",
    "vocab.txt",
    "special_tokens_map.json",
    "tokenizer.json",
)


def directory_size_mb(path: Path) -> float:
    """Sum all regular file sizes under ``path`` in MB."""
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024 / 1024


def copy_tokenizer_assets(src: Path, dst: Path) -> None:
    for name in TOKENIZER_FILES:
        src_file = src / name
        if src_file.exists():
            shutil.copy2(src_file, dst / name)


def quantize_one(
    model_cls: type,
    src: Path,
    dst: Path,
    *,
    label: str,
) -> tuple[float, float]:
    logger.info("\n===== %s =====", label)
    logger.info("  src: %s", src)
    logger.info("  dst: %s", dst)

    logger.info("  loading FP32 model (attn_implementation=eager for cross-version compat)...")
    # SDPA wrappers differ across transformers major versions; eager uses the
    # classic ``BertSelfAttention`` class which is stable across 4.x and 5.x.
    model = model_cls.from_pretrained(str(src), attn_implementation="eager")
    model.eval()

    logger.info("  running quantize_dynamic on nn.Linear layers (qint8)...")
    quantized = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )

    dst.mkdir(parents=True, exist_ok=True)

    out_file = dst / "quantized_state_dict.pt"
    logger.info("  saving state_dict to %s (tensor-only, cross-version)...", out_file)
    # Save ONLY the state_dict (tensors) — avoids pickling Python classes that
    # differ across transformers 4.x vs 5.x. Loader rebuilds the quantized
    # wrapper from the FP32 checkpoint then ``load_state_dict`` into it.
    torch.save(quantized.state_dict(), out_file)

    copy_tokenizer_assets(src, dst)

    before_mb = directory_size_mb(src)
    after_mb = directory_size_mb(dst)
    ratio = before_mb / after_mb if after_mb else float("inf")
    logger.info("  size: %6.1f MB -> %6.1f MB (%.2fx smaller)", before_mb, after_mb, ratio)
    return before_mb, after_mb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Root directory that contains bert_v11c_boundary_fix/ and bert_span_detector/",
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir).resolve()
    if not models_dir.is_dir():
        raise SystemExit(f"models dir not found: {models_dir}")

    totals_before = 0.0
    totals_after = 0.0

    before, after = quantize_one(
        BertForSequenceClassification,
        models_dir / "bert_v11c_boundary_fix",
        models_dir / "bert_v11c_int8",
        label="Classifier (bert_v11c_boundary_fix)",
    )
    totals_before += before
    totals_after += after

    before, after = quantize_one(
        BertForTokenClassification,
        models_dir / "bert_span_detector",
        models_dir / "bert_span_int8",
        label="Span detector (bert_span_detector)",
    )
    totals_before += before
    totals_after += after

    logger.info("\n===== Totals =====")
    logger.info("  before: %6.1f MB", totals_before)
    logger.info("  after : %6.1f MB", totals_after)
    ratio = totals_before / totals_after if totals_after else float("inf")
    logger.info(
        "  saved : %6.1f MB (%.2fx)",
        totals_before - totals_after,
        ratio,
    )


if __name__ == "__main__":
    main()
