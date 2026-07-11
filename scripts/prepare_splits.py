#!/usr/bin/env python3
"""Create deterministic patient-level train/tune/test JSONL splits."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ehr_summ.schemas import load_records, write_jsonl
from ehr_summ.splits import patient_level_split


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--tune", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    splits = patient_level_split(load_records(args.input), train=args.train, tune=args.tune, seed=args.seed)
    output = Path(args.output_dir)
    for name, records in splits.items():
        write_jsonl(output / f"{name}.jsonl", (record.to_dict() for record in records))
        print(f"{name}: {len(records)} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

