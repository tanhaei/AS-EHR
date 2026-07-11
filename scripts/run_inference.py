#!/usr/bin/env python3
"""Run specialty-aware or matched specialty-agnostic summarization."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ehr_summ.config import deep_get, load_config
from ehr_summ.engine import build_engine
from ehr_summ.schemas import load_records, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--backend", choices=["dry-run", "medgemma"], default="dry-run")
    parser.add_argument(
        "--condition",
        choices=["specialty-aware", "specialty-agnostic"],
        default="specialty-aware",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    records = load_records(deep_get(config, "data.input_jsonl"))
    engine = build_engine(config, args.backend)
    predictions = [engine.summarize(record, args.condition) for record in records]
    output = deep_get(config, f"inference.outputs.{args.condition}")
    if not output:
        raise ValueError(f"missing inference.outputs.{args.condition}")
    write_jsonl(output, (prediction.__dict__ for prediction in predictions))
    print(f"Wrote {len(predictions)} predictions to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
