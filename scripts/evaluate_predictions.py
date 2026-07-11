#!/usr/bin/env python3
"""Evaluate real or dry-run predictions and retain record-level evidence."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from ehr_summ.evaluation import aggregate_evaluations, evaluate_record
from ehr_summ.schemas import load_predictions, load_records
from ehr_summ.stats import cluster_bootstrap_ci


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--per-record-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()
    records = {record.record_id: record for record in load_records(args.records)}
    predictions = load_predictions(args.predictions)
    if {item.record_id for item in predictions} != set(records):
        raise ValueError("prediction and record IDs must match exactly")
    evaluations = [evaluate_record(records[prediction.record_id], prediction) for prediction in predictions]
    frame = pd.DataFrame([row.to_dict() for row in evaluations])
    per_record = Path(args.per_record_csv)
    per_record.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(per_record, index=False)
    summary = aggregate_evaluations(evaluations)
    _, low, high = cluster_bootstrap_ci(
        frame["f1"].to_numpy(), frame["patient_id_hash"].to_numpy(), n_boot=1000, seed=2026
    )
    summary["f1_95_ci"] = [low, high]
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
