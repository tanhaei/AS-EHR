#!/usr/bin/env python3
"""Paired comparison from genuine per-record score artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from ehr_summ.stats import benjamini_hochberg, cluster_bootstrap_ci, paired_test


REQUIRED = {"record_id", "patient_id_hash", "specialty", "f1"}


def _load(path: str, suffix: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    if frame["record_id"].duplicated().any():
        raise ValueError(f"duplicate record_id in {path}")
    return frame[list(REQUIRED)].rename(columns={"f1": f"f1_{suffix}"})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposed", required=True)
    parser.add_argument("--comparator", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    proposed = _load(args.proposed, "proposed")
    comparator = _load(args.comparator, "comparator")
    merged = proposed.merge(comparator, on=["record_id", "patient_id_hash", "specialty"], how="inner")
    if len(merged) != len(proposed) or len(merged) != len(comparator):
        raise ValueError("proposed and comparator record sets do not match exactly")

    def analyze(frame: pd.DataFrame) -> dict:
        test = paired_test(frame["f1_proposed"], frame["f1_comparator"])
        differences = frame["f1_proposed"].to_numpy() - frame["f1_comparator"].to_numpy()
        estimate, low, high = cluster_bootstrap_ci(
            differences, frame["patient_id_hash"].to_numpy(), n_boot=2000, seed=2026
        )
        return {
            "n_records": int(len(frame)),
            "proposed_mean": float(frame["f1_proposed"].mean()),
            "comparator_mean": float(frame["f1_comparator"].mean()),
            "mean_paired_difference": estimate,
            "difference_95_ci": [low, high],
            "test": test.test,
            "p_value": test.p_value,
            "effect_name": test.effect_name,
            "effect_size": test.effect_size,
        }

    result = {"pooled": analyze(merged), "specialties": {}}
    p_values, names = [], []
    for specialty, group in merged.groupby("specialty", sort=True):
        outcome = analyze(group)
        result["specialties"][specialty] = outcome
        names.append(specialty)
        p_values.append(outcome["p_value"])
    adjusted, rejected = benjamini_hochberg(p_values, q=0.05)
    for name, q_value, reject in zip(names, adjusted, rejected):
        result["specialties"][name]["q_value"] = q_value
        result["specialties"][name]["fdr_significant_0_05"] = bool(reject)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
