#!/usr/bin/env python3
"""Validate public schemas, IDs, language labels, and patient-level leakage."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ehr_summ.schemas import load_records
from ehr_summ.splits import assert_no_patient_overlap


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train")
    parser.add_argument("--tune")
    parser.add_argument("--test")
    parser.add_argument("--records")
    args = parser.parse_args()
    if args.records:
        records = load_records(args.records)
        print(f"Valid records: {len(records)}")
        return 0
    if not all([args.train, args.tune, args.test]):
        parser.error("provide --records or all of --train/--tune/--test")
    splits = {name: load_records(path) for name, path in
              {"train": args.train, "tune": args.tune, "test": args.test}.items()}
    assert_no_patient_overlap(splits)
    print("Patient-level split validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

