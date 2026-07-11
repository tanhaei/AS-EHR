"""Configuration provenance and validation tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ehr_summ.config import ConfigError, load_config, validate_for_training

ROOT = Path(__file__).resolve().parents[1]


def test_reported_config_does_not_invent_missing_values():
    config = load_config(ROOT / "configs/paper_reported.json")
    assert config["model"]["model_id"] is None
    try:
        validate_for_training(config)
    except ConfigError:
        pass
    else:
        raise AssertionError("incomplete paper-reported config must not validate for training")


def test_training_example_is_structurally_complete():
    config = load_config(ROOT / "configs/medgemma_lora.example.json")
    validate_for_training(config)

