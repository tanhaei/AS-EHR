#!/usr/bin/env python3
"""Validate or execute the optional MedGemma LoRA training path."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ehr_summ.config import load_config, validate_for_training
from ehr_summ.training import train_lora


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    validate_for_training(config)
    if args.validate_only:
        print("Training configuration is complete. No model was loaded.")
        return 0
    train_lora(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

