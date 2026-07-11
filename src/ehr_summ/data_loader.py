"""Load the paper's reported results (CSVs under ``data/``)."""
from __future__ import annotations

import os
from importlib.resources import files

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
# repo_root/src/ehr_summ/data_loader.py  ->  repo_root/data
DATA_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "data"))


def _load(name: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, name)
    if os.path.exists(path):
        return pd.read_csv(path)
    resource = files("ehr_summ").joinpath("paper_data", name)
    if not resource.is_file():
        raise FileNotFoundError(f"missing data file: {path}")
    with resource.open("rb") as handle:
        return pd.read_csv(handle)


def specialty_distribution() -> pd.DataFrame:   # Table 3
    return _load("specialty_distribution.csv")


def threshold_sensitivity() -> pd.DataFrame:    # Table 4
    return _load("threshold_sensitivity.csv")


def baseline_comparison() -> pd.DataFrame:      # Table 5
    return _load("baseline_comparison.csv")


def specialty_metrics() -> pd.DataFrame:        # Table 6
    return _load("specialty_metrics.csv")


def expert_feedback() -> pd.DataFrame:          # Table 7
    return _load("expert_feedback.csv")


def f1_confidence_intervals() -> pd.DataFrame:  # Table A.8
    return _load("f1_confidence_intervals.csv")


def rouge_l_w() -> pd.DataFrame:                # Appendix
    return _load("rouge_l_w.csv")


def specialty_embedding_construction() -> pd.DataFrame:  # Appendix
    return _load("specialty_embedding_construction.csv")
