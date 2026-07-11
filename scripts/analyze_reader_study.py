#!/usr/bin/env python3
"""Analyze blinded paired clinician ratings from a de-identified matrix."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from ehr_summ.reader_study import mixed_effects_utility, paired_outcomes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mixed-effects", action="store_true")
    args = parser.parse_args()
    frame = pd.read_csv(args.ratings)
    result = {"paired": [row.to_dict() for row in paired_outcomes(frame)]}
    if args.mixed_effects:
        result["mixed_effects_utility"] = mixed_effects_utility(frame)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

