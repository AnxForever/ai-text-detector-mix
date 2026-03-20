"""Tests for training utilities — Dataset, LengthWeightedLoss, BERTTrainer config.

These tests mock PyTorch/Transformers to avoid loading actual model weights.
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from scripts.training.length_weighted_loss import LengthWeightedLoss


class TestLengthWeightedLoss:
    """LengthWeightedLoss: short-text-weighted cross entropy."""

    def test_output_shape_mean(self):
        loss_fn = LengthWeightedLoss(alpha=0.3, max_length=512, reduction="mean")
        logits = torch.randn(4, 2)
        labels = torch.tensor([0, 1, 0, 1])
        lengths = torch.tensor([50, 100, 200, 400], dtype=torch.float)

        loss = loss_fn(logits, labels, lengths=lengths)
        assert loss.shape == ()  # scalar
        assert loss.item() > 0

    def test_output_shape_none(self):
        loss_fn = LengthWeightedLoss(alpha=0.3, max_length=512, reduction="none")
        logits = torch.randn(4, 2)
        labels = torch.tensor([0, 1, 0, 1])
        lengths = torch.tensor([50, 100, 200, 400], dtype=torch.float)

        loss = loss_fn(logits, labels, lengths=lengths)
        assert loss.shape == (4,)

    def test_alpha_zero_equals_standard_ce(self):
        """When alpha=0, should behave like standard CrossEntropyLoss."""
        loss_fn = LengthWeightedLoss(alpha=0.0, max_length=512, reduction="mean")
        ce_fn = nn.CrossEntropyLoss(reduction="mean")

        torch.manual_seed(42)
        logits = torch.randn(8, 2)
        labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
        lengths = torch.tensor([50, 100, 200, 400, 50, 100, 200, 400], dtype=torch.float)

        weighted_loss = loss_fn(logits, labels, lengths=lengths)
        standard_loss = ce_fn(logits, labels)

        assert abs(weighted_loss.item() - standard_loss.item()) < 1e-4

    def test_short_texts_get_higher_weight(self):
        """Shorter texts should receive higher loss weight."""
        loss_fn = LengthWeightedLoss(alpha=0.5, max_length=512, reduction="none")

        # Same logits/labels, different lengths
        logits = torch.tensor([[2.0, -1.0], [2.0, -1.0]])  # both predict class 0
        labels = torch.tensor([1, 1])  # both wrong

        short_loss = loss_fn(logits, labels, lengths=torch.tensor([50.0, 50.0]))
        long_loss = loss_fn(logits, labels, lengths=torch.tensor([500.0, 500.0]))

        # Short text loss should be >= long text loss
        assert short_loss[0].item() >= long_loss[0].item()

    def test_with_attention_mask(self):
        """Can compute lengths from attention_mask when lengths not provided."""
        loss_fn = LengthWeightedLoss(alpha=0.3, max_length=8, reduction="mean")
        logits = torch.randn(2, 2)
        labels = torch.tensor([0, 1])
        mask = torch.tensor([
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ])

        loss = loss_fn(logits, labels, attention_masks=mask)
        assert loss.item() > 0


# =========================================================================
# TextDataset (from eval_complete.py)
# =========================================================================
class TestTextDataset:
    """TextDataset: basic BERT dataset wrapper."""

    def test_dataset_length(self):
        from scripts.evaluation.eval_complete import TextDataset

        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": torch.zeros(1, 32, dtype=torch.long),
            "attention_mask": torch.ones(1, 32, dtype=torch.long),
        }

        texts = ["text1", "text2", "text3"]
        labels = [0, 1, 0]
        ds = TextDataset(texts, labels, tokenizer, max_len=32)

        assert len(ds) == 3

    def test_dataset_getitem(self):
        from scripts.evaluation.eval_complete import TextDataset

        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": torch.zeros(1, 32, dtype=torch.long),
            "attention_mask": torch.ones(1, 32, dtype=torch.long),
        }

        ds = TextDataset(["hello"], [1], tokenizer, max_len=32)
        item = ds[0]

        assert "input_ids" in item
        assert "attention_mask" in item
        assert "label" in item
        assert item["label"].item() == 1


# =========================================================================
# BERTTrainer configuration
# =========================================================================

# The training module imports scripts.bert_prep which may not exist (archived).
# We use a session-scoped fixture to mock it before the module is imported.

@pytest.fixture(autouse=True, scope="session")
def _mock_bert_prep_modules():
    """Inject mock modules for scripts.bert_prep before any BERTTrainer import.

    Also mock transformers if not installed, so the module can be imported
    in environments without GPU dependencies.
    """
    import sys
    mock = MagicMock()
    injected_keys: list[str] = []

    for key in ("scripts.bert_prep", "scripts.bert_prep.create_bert_dataset"):
        if key not in sys.modules:
            sys.modules[key] = mock
            injected_keys.append(key)

    # Mock transformers if not installed
    if "transformers" not in sys.modules:
        try:
            import transformers  # noqa: F401
        except ImportError:
            tx_mock = MagicMock()
            for tx_key in ("transformers",):
                sys.modules[tx_key] = tx_mock
                injected_keys.append(tx_key)

    yield

    for key in injected_keys:
        sys.modules.pop(key, None)


class TestBERTTrainerConfig:
    """BERTTrainer: verify configuration without loading real models."""

    def _make_trainer(self, **kwargs):
        """Helper: create a BERTTrainer with mocked model loading."""
        import scripts.training.train_bert_improved as mod

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        with patch.object(mod, "BertTokenizer") as tok_cls, \
             patch.object(mod, "BertForSequenceClassification") as mdl_cls:
            mdl_cls.from_pretrained.return_value = mock_model
            tok_cls.from_pretrained.return_value = MagicMock()
            return mod.BERTTrainer(**kwargs)

    def test_trainer_init_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = self._make_trainer(output_dir=tmpdir)

            assert trainer.max_length == 512
            assert trainer.batch_size == 16
            assert trainer.learning_rate == 2e-5
            assert trainer.num_epochs == 5
            assert trainer.use_length_weighted_loss is True
            assert trainer.loss_alpha == 0.3

    def test_trainer_custom_params(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = self._make_trainer(
                max_length=256,
                batch_size=8,
                learning_rate=3e-5,
                num_epochs=3,
                use_length_weighted_loss=False,
                output_dir=tmpdir,
            )

            assert trainer.max_length == 256
            assert trainer.batch_size == 8
            assert trainer.learning_rate == 3e-5
            assert trainer.num_epochs == 3
            assert trainer.use_length_weighted_loss is False

    def test_trainer_creates_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "new_dir")
            self._make_trainer(output_dir=out)
            assert os.path.isdir(out)

    def test_trainer_history_initialized(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = self._make_trainer(output_dir=tmpdir)

            for key in ["train_loss", "train_acc", "val_loss", "val_acc", "val_f1"]:
                assert key in trainer.history
                assert trainer.history[key] == []
