"""Upload bert_v11c_boundary_fix to HuggingFace Hub.

Target repo: AnxForever/chinese-ai-detector-bert (overwrites existing v10).
Run AFTER `hf auth login` is complete.

Usage:
    source ~/datacollection/.venv/bin/activate
    python scripts/deployment/upload_v11c_to_hf.py --dry-run   # preview
    python scripts/deployment/upload_v11c_to_hf.py             # actually upload
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from huggingface_hub import HfApi, upload_folder

LOGGER = logging.getLogger(__name__)

REPO_ID = "AnxForever/chinese-ai-detector-bert"
LOCAL_DIR = Path.home() / "datacollection" / "models" / "bert_v11c_boundary_fix"

ALLOW_PATTERNS = [
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.txt",
    "README.md",
    "eval_comparison.json",
    "eval_perclass.json",
    "training_log.json",
]

COMMIT_MESSAGE = (
    "Upgrade to v11c boundary-fix "
    "(three-set avg 98.56%, independent eval 98.57%, validation 98.75%)"
)


def _verify_local_files(local_dir: Path) -> list[Path]:
    missing: list[str] = []
    present: list[Path] = []
    for name in ALLOW_PATTERNS:
        fp = local_dir / name
        if fp.exists():
            present.append(fp)
        else:
            missing.append(name)
    if missing:
        raise FileNotFoundError(
            f"Missing required files in {local_dir}: {missing}"
        )
    return present


def _verify_auth() -> str:
    api = HfApi()
    info = api.whoami()
    username = info.get("name") or info.get("fullname") or "<unknown>"
    LOGGER.info("HF authenticated as: %s", username)
    return username


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify files and auth, but do not upload.",
    )
    parser.add_argument(
        "--repo-id",
        default=REPO_ID,
        help=f"Target HF repo (default: {REPO_ID})",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not LOCAL_DIR.exists():
        LOGGER.error("Local model dir not found: %s", LOCAL_DIR)
        return 1

    try:
        present = _verify_local_files(LOCAL_DIR)
    except FileNotFoundError as e:
        LOGGER.error(str(e))
        return 1

    total_bytes = sum(p.stat().st_size for p in present)
    LOGGER.info(
        "Found %d files in %s (total %.1f MB)",
        len(present),
        LOCAL_DIR,
        total_bytes / 1024 / 1024,
    )
    for p in present:
        LOGGER.info("  - %s (%.1f KB)", p.name, p.stat().st_size / 1024)

    try:
        _verify_auth()
    except Exception as e:
        LOGGER.error("HF auth check failed: %s", e)
        LOGGER.error("Run `hf auth login` first (needs Write-scope token).")
        return 1

    if args.dry_run:
        LOGGER.info("[DRY RUN] Would upload to %s. Skipping actual upload.", args.repo_id)
        return 0

    LOGGER.info("Uploading folder %s -> %s ...", LOCAL_DIR, args.repo_id)
    commit = upload_folder(
        repo_id=args.repo_id,
        folder_path=str(LOCAL_DIR),
        repo_type="model",
        allow_patterns=ALLOW_PATTERNS,
        commit_message=COMMIT_MESSAGE,
    )
    LOGGER.info("Upload complete. Commit: %s", commit.commit_url)
    LOGGER.info("Model page: https://huggingface.co/%s", args.repo_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
