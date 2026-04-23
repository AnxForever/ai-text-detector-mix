"""Utilities for the manual verification feedback loop.

This module persists post-detection human confirmations so misclassified
samples can be routed back into future dataset curation.
"""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.utils.paths import PATHS

_FILE_LOCK = threading.Lock()
_VALID_LABELS = {"human", "ai", "mixed"}


def normalize_label(label: str) -> str:
    """Normalize a feedback label to the canonical lowercase form."""
    normalized = label.strip().lower()
    if normalized not in _VALID_LABELS:
        raise ValueError(
            f"Invalid label '{label}'. Expected one of: {', '.join(sorted(_VALID_LABELS))}."
        )
    return normalized


def normalize_tags(tags: list[str] | None) -> list[str]:
    """Deduplicate and sanitize tags while preserving user intent."""
    if not tags:
        return []

    result: list[str] = []
    seen: set[str] = set()

    for tag in tags:
        cleaned = tag.strip()
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)

    return result


def derive_feedback_tags(
    predicted_label: str,
    confirmed_label: str,
    tags: list[str] | None = None,
) -> list[str]:
    """Attach useful system tags on top of user-provided tags."""
    normalized_tags = normalize_tags(tags)
    derived = ["manual_feedback"]

    if predicted_label == "ai" and confirmed_label == "human":
        derived.append("false_positive")
    elif predicted_label == "human" and confirmed_label == "ai":
        derived.append("false_negative")
    elif predicted_label != confirmed_label:
        derived.append("label_corrected")

    return normalize_tags(normalized_tags + derived)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append a single JSON record to a JSONL file safely within-process."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _FILE_LOCK:
        with path.open("a", encoding="utf-8") as file_obj:
            json.dump(record, file_obj, ensure_ascii=False)
            file_obj.write("\n")


def canonicalize_feedback_text(text: str) -> str:
    """Normalize whitespace so repeated submissions map to the same sample."""
    return " ".join(text.split())


def build_feedback_dedup_key(text: str) -> str:
    """Build a stable hash key for deduplicating training samples."""
    canonical_text = canonicalize_feedback_text(text)
    return hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()


def _load_existing_dedup_keys(path: Path) -> set[str]:
    """Load existing dedup keys from a JSONL dataset."""
    if not path.exists():
        return set()

    dedup_keys: set[str] = set()
    with path.open("r", encoding="utf-8") as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            dedup_key = row.get("dedup_key")
            if isinstance(dedup_key, str) and dedup_key:
                dedup_keys.add(dedup_key)
                continue

            text = row.get("text")
            if isinstance(text, str) and text.strip():
                dedup_keys.add(build_feedback_dedup_key(text))

    return dedup_keys


def _extract_record_dedup_key(row: dict[str, Any]) -> str | None:
    """Extract a stable dedup key from a stored JSONL row."""
    dedup_key = row.get("dedup_key")
    if isinstance(dedup_key, str) and dedup_key:
        return dedup_key

    text = row.get("text")
    if isinstance(text, str) and text.strip():
        return build_feedback_dedup_key(text)

    return None


def _load_latest_records_by_dedup_key(path: Path) -> dict[str, dict[str, Any]]:
    """Load the latest stored feedback record for each dedup key."""
    if not path.exists():
        return {}

    latest_records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            dedup_key = _extract_record_dedup_key(row)
            if dedup_key:
                latest_records[dedup_key] = row

    return latest_records


def _load_conflicted_dedup_keys(path: Path) -> set[str]:
    """Load dedup keys that have conflicting manual confirmations."""
    if not path.exists():
        return set()

    conflicted_keys: set[str] = set()
    with path.open("r", encoding="utf-8") as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            dedup_key = _extract_record_dedup_key(row)
            if dedup_key:
                conflicted_keys.add(dedup_key)

    return conflicted_keys


def lookup_feedback_override(
    *,
    text: str,
    output_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Return an exact-match manual correction record when it is safe to reuse.

    Only exact text matches are eligible. If the sample has conflicting manual
    confirmations, this helper returns ``None`` so the detector keeps using the
    model output until the conflict is reviewed offline.
    """
    cleaned_text = text.strip()
    if not cleaned_text:
        return None

    stored_dir = output_dir or PATHS.feedback_loop_dir
    corrections_path = stored_dir / "misclassified_samples.jsonl"
    conflicts_path = stored_dir / "feedback_conflicts.jsonl"
    dedup_key = build_feedback_dedup_key(cleaned_text)

    if dedup_key in _load_conflicted_dedup_keys(conflicts_path):
        return None

    latest_records = _load_latest_records_by_dedup_key(corrections_path)
    record = latest_records.get(dedup_key)
    if record is None:
        return None

    confirmed_label = record.get("confirmed_label")
    if not isinstance(confirmed_label, str):
        return None

    try:
        normalized_label = normalize_label(confirmed_label)
    except ValueError:
        return None

    boundary = record.get("boundary")
    return {
        "feedback_id": record.get("feedback_id"),
        "dedup_key": dedup_key,
        "confirmed_label": normalized_label,
        "boundary": boundary if isinstance(boundary, int) and boundary >= 0 else None,
        "domain_hint": record.get("domain_hint"),
        "source": "manual_feedback_exact_match",
    }


def persist_feedback(
    *,
    text: str,
    predicted_label: str,
    confirmed_correct: bool,
    confirmed_label: str | None = None,
    tags: list[str] | None = None,
    note: str | None = None,
    source: str,
    model_version: str | None = None,
    confidence: float | None = None,
    ai_percentage: int | None = None,
    human_percentage: int | None = None,
    boundary: int | None = None,
    domain_hint: str | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Persist a unique correction sample for future retraining.

    Returns:
        Metadata describing where the records were stored.
    """
    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError("text must not be blank")

    normalized_predicted = normalize_label(predicted_label)
    normalized_confirmed = (
        normalize_label(confirmed_label)
        if confirmed_label is not None
        else normalized_predicted
    )

    timestamp = datetime.now().isoformat()
    feedback_id = uuid.uuid4().hex
    stored_dir = output_dir or PATHS.feedback_loop_dir
    corrections_path = stored_dir / "misclassified_samples.jsonl"
    conflicts_path = stored_dir / "feedback_conflicts.jsonl"
    dedup_key = build_feedback_dedup_key(cleaned_text)
    feedback_tags = derive_feedback_tags(
        predicted_label=normalized_predicted,
        confirmed_label=normalized_confirmed,
        tags=tags,
    )

    record = {
        "feedback_id": feedback_id,
        "created_at": timestamp,
        "text": cleaned_text,
        "predicted_label": normalized_predicted,
        "confirmed_correct": confirmed_correct,
        "confirmed_label": normalized_confirmed,
        "tags": feedback_tags,
        "note": note.strip() if note else None,
        "source": source,
        "model_version": model_version,
        "confidence": confidence,
        "ai_percentage": ai_percentage,
        "human_percentage": human_percentage,
        "boundary": boundary,
        "domain_hint": domain_hint,
        "dedup_key": dedup_key,
        "dataset_type": "manual_correction",
    }

    if not confirmed_correct:
        corrections_path.parent.mkdir(parents=True, exist_ok=True)
        with _FILE_LOCK:
            existing_records = _load_latest_records_by_dedup_key(corrections_path)
            existing_record = existing_records.get(dedup_key)

            if existing_record is None:
                with corrections_path.open("a", encoding="utf-8") as file_obj:
                    json.dump(record, file_obj, ensure_ascii=False)
                    file_obj.write("\n")
                misclassified_saved = True
                conflict_detected = False
            else:
                existing_confirmed_label = existing_record.get("confirmed_label")
                if existing_confirmed_label == normalized_confirmed:
                    misclassified_saved = False
                    conflict_detected = False
                else:
                    conflict_record = {
                        **record,
                        "event_type": "conflicting_manual_feedback",
                        "existing_feedback_id": existing_record.get("feedback_id"),
                        "existing_confirmed_label": existing_confirmed_label,
                    }
                    with conflicts_path.open("a", encoding="utf-8") as file_obj:
                        json.dump(conflict_record, file_obj, ensure_ascii=False)
                        file_obj.write("\n")
                    misclassified_saved = False
                    conflict_detected = True
    else:
        misclassified_saved = False
        conflict_detected = False

    return {
        "feedback_id": feedback_id,
        "created_at": timestamp,
        "stored_dir": str(stored_dir),
        "corrections_path": str(corrections_path),
        "conflicts_path": str(conflicts_path),
        "misclassified_saved": misclassified_saved,
        "duplicate_skipped": not confirmed_correct and not misclassified_saved,
        "conflict_detected": conflict_detected,
    }
