"""Reader-study validation and paired/mixed-effects analysis."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from .stats import paired_test


REQUIRED_COLUMNS = {
    "evaluator_hash", "record_id", "condition", "specialty",
    "relevance", "clarity", "utility", "time_seconds",
}


@dataclass
class ReaderOutcome:
    outcome: str
    n_pairs: int
    aware_mean: float
    agnostic_mean: float
    absolute_difference: float
    test: str
    p_value: float
    effect_name: str
    effect_size: float

    def to_dict(self) -> dict:
        return asdict(self)


def validate_ratings(frame: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"reader ratings missing columns: {sorted(missing)}")
    if not set(frame["condition"]).issubset({"specialty-aware", "specialty-agnostic"}):
        raise ValueError("condition must be specialty-aware or specialty-agnostic")
    rating_columns = ["relevance", "clarity", "utility"]
    if not frame[rating_columns].apply(lambda series: series.between(1, 5).all()).all():
        raise ValueError("reader ratings must be between 1 and 5")
    if frame["time_seconds"].isna().any() or (frame["time_seconds"] < 0).any():
        raise ValueError("time_seconds must be non-negative")
    duplicate = frame.duplicated(["evaluator_hash", "record_id", "condition"])
    if duplicate.any():
        raise ValueError("duplicate evaluator-record-condition rating")


def paired_outcomes(frame: pd.DataFrame) -> list[ReaderOutcome]:
    validate_ratings(frame)
    index = ["evaluator_hash", "record_id"]
    results: list[ReaderOutcome] = []
    for outcome in ["relevance", "clarity", "utility", "time_seconds"]:
        wide = frame.pivot(index=index, columns="condition", values=outcome).dropna()
        aware = wide["specialty-aware"].to_numpy(float)
        agnostic = wide["specialty-agnostic"].to_numpy(float)
        test = paired_test(aware, agnostic)
        results.append(
            ReaderOutcome(outcome, len(wide), float(aware.mean()), float(agnostic.mean()),
                          float((aware - agnostic).mean()), test.test, test.p_value,
                          test.effect_name, test.effect_size)
        )
    return results


def mixed_effects_utility(frame: pd.DataFrame) -> dict:
    """Fit utility ~ condition with evaluator and record random effects.

    statsmodels is optional so the dependency-light test suite remains fast.
    """
    validate_ratings(frame)
    try:
        import statsmodels.formula.api as smf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install the 'analysis' optional dependencies") from exc
    data = frame.copy()
    data["aware"] = (data["condition"] == "specialty-aware").astype(int)
    model = smf.mixedlm("utility ~ aware", data, groups=data["evaluator_hash"],
                        vc_formula={"record": "0 + C(record_id)"})
    fit = model.fit(reml=False, method="lbfgs")
    return {
        "coefficient": float(fit.params["aware"]),
        "standard_error": float(fit.bse["aware"]),
        "p_value": float(fit.pvalues["aware"]),
        "confidence_interval": [float(value) for value in fit.conf_int().loc["aware"]],
        "n_ratings": int(len(data)),
    }
