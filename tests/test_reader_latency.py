"""Reader-study and latency evidence tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from ehr_summ.latency import benchmark
from ehr_summ.reader_study import paired_outcomes, validate_ratings

ROOT = Path(__file__).resolve().parents[1]


def test_reader_example_is_paired_and_valid():
    frame = pd.read_csv(ROOT / "data/examples/reader_ratings.csv")
    validate_ratings(frame)
    outcomes = {row.outcome: row for row in paired_outcomes(frame)}
    assert outcomes["utility"].n_pairs == 6
    assert outcomes["utility"].absolute_difference > 0
    assert outcomes["time_seconds"].absolute_difference < 0


def test_latency_keeps_raw_observations():
    rows, summary = benchmark(lambda: sum(range(10)), warmup=1, repeats=5)
    assert len(rows) == 5
    assert summary["repeats"] == 5
    assert summary["p95_seconds"] >= summary["median_seconds"]

