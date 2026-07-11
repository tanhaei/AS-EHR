"""Dry-run selection, safety layer, and explicit denominator tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from ehr_summ.embeddings import HashingEmbedder
from ehr_summ.evaluation import aggregate_evaluations, evaluate_record
from ehr_summ.modeling import DryRunSummarizer
from ehr_summ.pipeline import select_specialty_features
from ehr_summ.schemas import SummaryPrediction, load_records

ROOT = Path(__file__).resolve().parents[1]


def test_safety_layer_never_drops_mandatory_features():
    record = load_records(ROOT / "data/examples/records.jsonl")[0]
    embedder = HashingEmbedder(64)
    selected = select_specialty_features(record, np.ones(64), embedder, tau=1.1, K=0, safety_layer=True)
    assert {item.feature_id for item in selected} == {
        item.feature_id for item in record.features if item.mandatory
    }


def test_record_evaluation_counts_unsupported_and_omitted():
    record = load_records(ROOT / "data/examples/records.jsonl")[0]
    prediction = SummaryPrediction(
        record.record_id, "test", "پرفشاری خون", ["CARD-DX-1", "NOT-A-SOURCE"], "test-model"
    )
    row = evaluate_record(record, prediction)
    assert row.unsupported_feature_count == 1
    assert row.mandatory_omission_count == 2
    assert row.code_switch_total == 2


def test_dry_run_aggregation_uses_term_denominators():
    records = load_records(ROOT / "data/examples/records.jsonl")
    summarizer = DryRunSummarizer()
    rows = [evaluate_record(record, summarizer.summarize(record, record.features)) for record in records]
    summary = aggregate_evaluations(rows)
    assert summary["n_records"] == 6
    assert summary["code_switch_total"] == 12
    assert 0 <= summary["code_switch_accuracy"] <= 1

