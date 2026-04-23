"""Tests for the manual feedback loop persistence helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.utils.feedback_loop import (
    build_feedback_dedup_key,
    canonicalize_feedback_text,
    derive_feedback_tags,
    lookup_feedback_override,
    normalize_label,
    normalize_tags,
    persist_feedback,
)


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dictionaries."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class TestNormalizeLabel:
    """normalize_label: canonicalize supported labels."""

    def test_accepts_supported_labels(self):
        assert normalize_label("AI") == "ai"
        assert normalize_label("human") == "human"
        assert normalize_label(" mixed ") == "mixed"

    def test_rejects_invalid_label(self):
        with pytest.raises(ValueError):
            normalize_label("robot")


class TestNormalizeTags:
    """normalize_tags: remove blanks and duplicates."""

    def test_deduplicates_and_keeps_order(self):
        assert normalize_tags(["formal", "Formal", "demo", ""]) == ["formal", "demo"]

    def test_none_returns_empty(self):
        assert normalize_tags(None) == []


class TestDeriveFeedbackTags:
    """derive_feedback_tags: add system tags for routing."""

    def test_false_positive_tag(self):
        tags = derive_feedback_tags("ai", "human", ["formal"])
        assert "manual_feedback" in tags
        assert "false_positive" in tags
        assert "formal" in tags

    def test_false_negative_tag(self):
        tags = derive_feedback_tags("human", "ai", ["casual"])
        assert "false_negative" in tags


class TestFeedbackDedup:
    """Dedup helpers for manual correction dataset."""

    def test_canonicalize_feedback_text_collapses_whitespace(self):
        assert canonicalize_feedback_text("  第一行\n第二行\t  结束  ") == "第一行 第二行 结束"

    def test_build_feedback_dedup_key_same_text_same_hash(self):
        assert build_feedback_dedup_key("同一段文本") == build_feedback_dedup_key(" 同一段文本 ")


class TestPersistFeedback:
    """persist_feedback: keep only unique misclassified samples."""

    def test_correct_prediction_does_not_write_training_dataset(self, tmp_path):
        result = persist_feedback(
            text="这是一段正常文本。",
            predicted_label="human",
            confirmed_correct=True,
            source="test",
            output_dir=tmp_path,
        )

        corrections = tmp_path / "misclassified_samples.jsonl"

        assert not corrections.exists()
        assert result["misclassified_saved"] is False

    def test_incorrect_prediction_writes_correction_dataset(self, tmp_path):
        result = persist_feedback(
            text="这段文本被误判了。",
            predicted_label="ai",
            confirmed_correct=False,
            confirmed_label="human",
            tags=["formal", "false_positive_case"],
            note="正式通知被误判",
            source="test",
            output_dir=tmp_path,
        )

        corrections = tmp_path / "misclassified_samples.jsonl"

        assert corrections.exists()
        assert result["misclassified_saved"] is True

        correction_rows = read_jsonl(corrections)

        assert len(correction_rows) == 1
        assert correction_rows[0]["confirmed_label"] == "human"
        assert "manual_feedback" in correction_rows[0]["tags"]
        assert "false_positive" in correction_rows[0]["tags"]
        assert correction_rows[0]["dataset_type"] == "manual_correction"
        assert correction_rows[0]["dedup_key"] == build_feedback_dedup_key("这段文本被误判了。")

    def test_duplicate_misclassified_text_is_not_saved_twice(self, tmp_path):
        first = persist_feedback(
            text="重复误判样本",
            predicted_label="ai",
            confirmed_correct=False,
            confirmed_label="human",
            source="test",
            output_dir=tmp_path,
        )
        second = persist_feedback(
            text="  重复误判样本  ",
            predicted_label="ai",
            confirmed_correct=False,
            confirmed_label="human",
            source="test",
            output_dir=tmp_path,
        )

        correction_rows = read_jsonl(tmp_path / "misclassified_samples.jsonl")

        assert first["misclassified_saved"] is True
        assert second["misclassified_saved"] is False
        assert second["duplicate_skipped"] is True
        assert len(correction_rows) == 1

    def test_conflicting_duplicate_disables_exact_match_override(self, tmp_path):
        first = persist_feedback(
            text="需要人工确认冲突的样本",
            predicted_label="ai",
            confirmed_correct=False,
            confirmed_label="human",
            source="test",
            output_dir=tmp_path,
        )
        second = persist_feedback(
            text="需要人工确认冲突的样本",
            predicted_label="ai",
            confirmed_correct=False,
            confirmed_label="ai",
            source="test",
            output_dir=tmp_path,
        )

        conflict_rows = read_jsonl(tmp_path / "feedback_conflicts.jsonl")

        assert first["misclassified_saved"] is True
        assert second["misclassified_saved"] is False
        assert second["conflict_detected"] is True
        assert len(conflict_rows) == 1
        assert lookup_feedback_override(text="需要人工确认冲突的样本", output_dir=tmp_path) is None

    def test_lookup_feedback_override_returns_exact_match_record(self, tmp_path):
        persist_feedback(
            text="这段文本已经被人工确认纠正。",
            predicted_label="ai",
            confirmed_correct=False,
            confirmed_label="human",
            source="test",
            output_dir=tmp_path,
        )

        override = lookup_feedback_override(
            text="  这段文本已经被人工确认纠正。 ",
            output_dir=tmp_path,
        )

        assert override is not None
        assert override["confirmed_label"] == "human"
        assert override["source"] == "manual_feedback_exact_match"

    def test_blank_text_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            persist_feedback(
                text="   ",
                predicted_label="human",
                confirmed_correct=True,
                source="test",
                output_dir=tmp_path,
            )
