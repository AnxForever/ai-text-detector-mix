"""Tests for scripts/utils/ modules — paths, schemas, log."""

from __future__ import annotations

import tempfile

import pandas as pd
import pytest
from pydantic import ValidationError

from scripts.utils.paths import PATHS, PROJECT_ROOT
from scripts.utils.schemas import DetectionSample, HybridSample, TrainingConfig, validate_dataframe


# =========================================================================
# PATHS
# =========================================================================
class TestPaths:
    """Centralized path management."""

    def test_project_root_exists(self):
        assert PROJECT_ROOT.exists()
        assert (PROJECT_ROOT / "api").exists()

    def test_datasets_dir(self):
        assert PATHS.datasets_dir.name == "datasets"

    def test_models_dir(self):
        assert PATHS.models_dir.name == "models"

    def test_classifier_model_default(self):
        assert "bert_v11c_boundary_fix" in str(PATHS.classifier_model)

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("DC_DATASETS_DIR", "/tmp/custom_datasets")
        from scripts.utils import paths as paths_mod

        # Re-instantiate to pick up env
        p = paths_mod._Paths()
        assert str(p.datasets_dir) == "/tmp/custom_datasets"

    def test_ensure_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = PATHS.ensure_dir(PROJECT_ROOT / tmpdir / "sub" / "dir")
            assert new_dir.exists()

    def test_require_file_raises(self):
        with pytest.raises(FileNotFoundError, match="Required file"):
            PATHS.require_file(PROJECT_ROOT / "nonexistent_file.xyz", "test file")

    def test_require_file_passes(self):
        result = PATHS.require_file(PROJECT_ROOT / "CLAUDE.md", "project docs")
        assert result.exists()


# =========================================================================
# SCHEMAS
# =========================================================================
class TestDetectionSample:
    """DetectionSample: text + label validation."""

    def test_valid_sample(self):
        s = DetectionSample(text="hello world", label=0)
        assert s.text == "hello world"
        assert s.label == 0

    def test_ai_label(self):
        s = DetectionSample(text="AI text", label=1)
        assert s.label == 1

    def test_empty_text_rejected(self):
        with pytest.raises(ValidationError):
            DetectionSample(text="", label=0)

    def test_blank_text_rejected(self):
        with pytest.raises(ValidationError):
            DetectionSample(text="   ", label=0)

    def test_invalid_label_rejected(self):
        with pytest.raises(ValidationError):
            DetectionSample(text="hello", label=2)

    def test_negative_label_rejected(self):
        with pytest.raises(ValidationError):
            DetectionSample(text="hello", label=-1)


class TestHybridSample:
    """HybridSample: text + label + category validation."""

    def test_valid_c2(self):
        s = HybridSample(text="mixed text", label=1, category="C2")
        assert s.category == "C2"

    def test_valid_human(self):
        s = HybridSample(text="human text", label=0, category="Human")
        assert s.category == "Human"

    def test_invalid_category(self):
        with pytest.raises(ValidationError):
            HybridSample(text="text", label=0, category="C5")


class TestTrainingConfig:
    """TrainingConfig: hyperparameter bounds."""

    def test_defaults(self):
        c = TrainingConfig()
        assert c.model_name == "bert-base-chinese"
        assert c.max_length == 512
        assert c.batch_size == 16

    def test_custom_values(self):
        c = TrainingConfig(max_length=256, batch_size=8, num_epochs=3)
        assert c.max_length == 256

    def test_invalid_batch_size(self):
        with pytest.raises(ValidationError):
            TrainingConfig(batch_size=0)

    def test_invalid_learning_rate(self):
        with pytest.raises(ValidationError):
            TrainingConfig(learning_rate=-0.01)


class TestValidateDataframe:
    """validate_dataframe: bulk validation."""

    def test_valid_df(self):
        df = pd.DataFrame({"text": ["a", "b", "c"], "label": [0, 1, 0]})
        errors = validate_dataframe(df, DetectionSample)
        assert errors == []

    def test_invalid_rows(self):
        df = pd.DataFrame({"text": ["a", "", "c"], "label": [0, 1, 0]})
        errors = validate_dataframe(df, DetectionSample)
        assert len(errors) > 0

    def test_missing_columns(self):
        df = pd.DataFrame({"text": ["a"]})
        errors = validate_dataframe(df, DetectionSample, required_columns=["text", "label"])
        assert len(errors) == 1
        assert "Missing columns" in errors[0]["error"]

    def test_sample_size(self):
        df = pd.DataFrame({"text": [f"text_{i}" for i in range(100)], "label": [0] * 100})
        errors = validate_dataframe(df, DetectionSample, sample_size=10)
        assert errors == []
