"""Centralized path management for the datacollection project.

All dataset/model paths go through here. Supports environment variable
overrides for portability across environments.

Usage:
    from scripts.utils.paths import PATHS

    df = pd.read_csv(PATHS.core_v1_train)
    model = load_model(PATHS.classifier_model)
"""

from __future__ import annotations

import os
from pathlib import Path


def _project_root() -> Path:
    """Resolve project root (two levels up from this file)."""
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = _project_root()


class _Paths:
    """Lazy path resolver with environment variable overrides.

    Each property returns a :class:`pathlib.Path`. The corresponding
    environment variable (if set) takes precedence over the default.
    """

    # -- Datasets --------------------------------------------------------
    @property
    def datasets_dir(self) -> Path:
        return Path(os.getenv("DC_DATASETS_DIR", PROJECT_ROOT / "datasets"))

    @property
    def core_v1_dir(self) -> Path:
        return self.datasets_dir / "active" / "core_v1"

    @property
    def core_v1_train(self) -> Path:
        return self.core_v1_dir / "train.csv"

    @property
    def core_v1_val(self) -> Path:
        return self.core_v1_dir / "val.csv"

    @property
    def core_v1_test(self) -> Path:
        return self.core_v1_dir / "test.csv"

    @property
    def merged_v2_dir(self) -> Path:
        return self.datasets_dir / "merged_v2"

    @property
    def hybrid_dir(self) -> Path:
        return self.datasets_dir / "mixed" / "hybrid"

    # -- Models ----------------------------------------------------------
    @property
    def models_dir(self) -> Path:
        return Path(os.getenv("DC_MODELS_DIR", PROJECT_ROOT / "models"))

    @property
    def classifier_model(self) -> Path:
        return Path(
            os.getenv("DETECTOR_CLASSIFIER_MODEL", self.models_dir / "bert_v11c_boundary_fix")
        )

    @property
    def span_model(self) -> Path:
        return Path(
            os.getenv("DETECTOR_SPAN_MODEL", self.models_dir / "bert_span_detector")
        )

    # -- Config ----------------------------------------------------------
    @property
    def config_dir(self) -> Path:
        return PROJECT_ROOT / "config"

    @property
    def api_local_config(self) -> Path:
        return self.config_dir / "api.local.json"

    # -- Helpers ---------------------------------------------------------
    def ensure_dir(self, path: Path) -> Path:
        """Create directory if it doesn't exist and return it."""
        path.mkdir(parents=True, exist_ok=True)
        return path

    def require_file(self, path: Path, description: str = "") -> Path:
        """Raise FileNotFoundError if path doesn't exist."""
        if not path.exists():
            desc = f" ({description})" if description else ""
            raise FileNotFoundError(f"Required file not found{desc}: {path}")
        return path


PATHS = _Paths()
