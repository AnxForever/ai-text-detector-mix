"""Data validation schemas for the datacollection project.

Pydantic models for validating dataset records, API payloads, and
training configurations. Use these at data-loading boundaries to
catch schema violations early.

Usage:
    from scripts.utils.schemas import DetectionSample, validate_dataframe

    # Validate a single record
    sample = DetectionSample(text="some text", label=1)

    # Validate a whole dataframe
    errors = validate_dataframe(df, DetectionSample)
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class DetectionSample(BaseModel):
    """A single text classification sample."""

    text: str = Field(min_length=1)
    label: int = Field(ge=0, le=1)

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank")
        return v


class HybridSample(BaseModel):
    """A hybrid (mixed human+AI) dataset sample."""

    text: str = Field(min_length=1)
    label: int = Field(ge=0, le=1)
    category: str = Field(pattern=r"^(C2|C3|C4|Human|AI)$")


class TrainingConfig(BaseModel):
    """Training hyperparameter configuration."""

    model_name: str = "bert-base-chinese"
    max_length: int = Field(default=512, ge=32, le=2048)
    batch_size: int = Field(default=16, ge=1, le=512)
    learning_rate: float = Field(default=2e-5, gt=0, lt=1)
    num_epochs: int = Field(default=5, ge=1, le=100)
    warmup_steps: int = Field(default=500, ge=0)
    weight_decay: float = Field(default=0.01, ge=0, le=1)


def validate_dataframe(
    df: pd.DataFrame,
    schema: type[BaseModel],
    required_columns: list[str] | None = None,
    sample_size: int | None = None,
) -> list[dict[str, Any]]:
    """Validate a DataFrame against a Pydantic schema.

    Args:
        df: DataFrame to validate.
        schema: Pydantic model class to validate each row against.
        required_columns: Column names that must exist (checked first).
        sample_size: If set, validate only a random sample of rows.

    Returns:
        List of error dicts ``{"row": int, "error": str}`` (empty = all valid).
    """
    errors: list[dict[str, Any]] = []

    # Check required columns
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            errors.append({"row": -1, "error": f"Missing columns: {missing}"})
            return errors

    target = df.sample(n=sample_size) if sample_size and sample_size < len(df) else df

    for idx, row in target.iterrows():
        try:
            schema(**row.to_dict())
        except Exception as e:
            errors.append({"row": idx, "error": str(e)})

    if errors:
        logger.warning("Validation found %d errors in %d rows", len(errors), len(target))
    else:
        logger.info("All %d rows passed validation", len(target))

    return errors
