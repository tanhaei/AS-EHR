"""Experiment configuration loading and validation.

The public repository deliberately separates values reported in the manuscript
from values that must be recovered from the original training logs.  This
module never silently invents a missing experimental parameter.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


class ConfigError(ValueError):
    """Raised when a configuration cannot support the requested operation."""


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ConfigError("top-level configuration must be a JSON object")
    return config


def deep_get(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def require(config: dict[str, Any], keys: Iterable[str]) -> None:
    missing = [key for key in keys if deep_get(config, key) in (None, "", [])]
    if missing:
        raise ConfigError("missing required configuration values: " + ", ".join(missing))


def validate_for_training(config: dict[str, Any]) -> None:
    require(
        config,
        [
            "model.model_id",
            "data.train_jsonl",
            "data.validation_jsonl",
            "training.output_dir",
            "training.learning_rate",
            "training.epochs",
            "training.batch_size",
            "training.gradient_accumulation_steps",
            "lora.rank",
            "lora.alpha",
            "lora.target_modules",
        ],
    )


def validate_for_inference(config: dict[str, Any]) -> None:
    require(config, ["model.model_id", "data.input_jsonl", "inference.output_jsonl"])

