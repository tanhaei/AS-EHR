#!/usr/bin/env python3
"""Benchmark raw end-to-end latency and persist every observation."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ehr_summ.latency import benchmark
from ehr_summ.modeling import DryRunSummarizer, MedGemmaSummarizer
from ehr_summ.schemas import load_records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--raw-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--backend", choices=["dry-run", "medgemma"], default="dry-run")
    parser.add_argument("--model-id", default="google/medgemma-27b-it")
    parser.add_argument("--adapter-path")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    record = load_records(args.records)[0]
    summarizer = (DryRunSummarizer() if args.backend == "dry-run"
                  else MedGemmaSummarizer(args.model_id, adapter_path=args.adapter_path))
    rows, summary = benchmark(lambda: summarizer.summarize(record, record.features),
                              warmup=args.warmup, repeats=args.repeats)
    raw_path = Path(args.raw_csv)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["iteration", "elapsed_seconds"])
        writer.writeheader()
        writer.writerows(row.to_dict() for row in rows)
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

