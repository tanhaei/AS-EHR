"""End-to-end reproduction tests: paper aggregates and synthetic calibration."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from ehr_summ import data_loader as dl
from ehr_summ.metrics import macro_average, weighted_average
from ehr_summ.stats import cohens_d_paired, paired_test
from ehr_summ.synthetic import generate_all_pairs


def test_corpus_totals():
    dist = dl.specialty_distribution()
    assert int(dist["n_records"].sum()) == 1000
    cs = (dist["n_records"] * dist["pct_code_switched"] / 100.0).sum()
    assert abs(cs - 707) < 1.0


def test_pooled_and_macro_metrics():
    m = dl.specialty_metrics()
    n = dl.specialty_distribution()["n_records"]
    assert abs(weighted_average(m["f1"], n) - 0.83) < 0.005
    assert abs(macro_average(m["rouge_n"]) - 0.855) < 1e-6


def test_reader_study_totals():
    f = dl.expert_feedback()
    assert int(f["n_experts"].sum()) == 130
    assert abs(weighted_average(f["satisfaction_mean"], f["n_experts"]) - 4.4) < 0.05
    assert abs(weighted_average(f["coverage_mean"], f["n_experts"]) - 4.5) < 0.05


def test_f1_ci_table_pooled():
    ci = dl.f1_confidence_intervals().set_index("specialty").loc["Pooled"]
    assert abs(ci["f1"] - 0.83) < 1e-9
    assert abs(ci["ci_low"] - 0.825) < 1e-9 and abs(ci["ci_high"] - 0.839) < 1e-9


def test_synthetic_reproduces_pooled_inference():
    pairs = generate_all_pairs(seed=0)
    proposed = np.concatenate([p.proposed for p in pairs.values()])
    baseline = np.concatenate([p.baseline for p in pairs.values()])
    assert proposed.size == 1000
    assert abs(proposed.mean() - 0.83) < 0.01
    assert abs(baseline.mean() - 0.72) < 0.01
    res = paired_test(proposed, baseline)
    assert res.p_value < 0.001
    d = cohens_d_paired(proposed, baseline)
    assert 1.0 <= d <= 1.15  # paper reports 1.07


def test_synthetic_per_specialty_means_match_table5():
    base = dl.baseline_comparison().set_index("specialty")
    pairs = generate_all_pairs(seed=0)
    for name, ps in pairs.items():
        assert abs(ps.proposed.mean() - base.loc[name, "proposed_f1"]) < 0.01
        assert abs(ps.baseline.mean() - base.loc[name, "biobert_f1"]) < 0.01


def test_packaged_aggregate_tables_match_repository_copies():
    root = Path(__file__).resolve().parents[1]
    packaged = root / "src/ehr_summ/paper_data"
    for source in (root / "data").glob("*.csv"):
        target = packaged / source.name
        assert target.exists()
        assert source.read_bytes() == target.read_bytes()
