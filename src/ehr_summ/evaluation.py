"""Per-record evaluation with explicit clinical-error denominators."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from .metrics import prf_from_sets, text_rouge
from .preprocessing import normalize_persian
from .schemas import ClinicalRecord, SummaryPrediction


@dataclass
class RecordEvaluation:
    record_id: str
    patient_id_hash: str
    specialty: str
    site_id: str
    language_profile: str
    condition: str
    precision: float
    recall: float
    f1: float
    rouge_1: float | None
    rouge_2: float | None
    rouge_l: float | None
    unsupported_feature_count: int
    predicted_feature_count: int
    mandatory_omission_count: int
    mandatory_feature_count: int
    code_switch_correct: int
    code_switch_total: int

    def to_dict(self) -> dict:
        return asdict(self)


def _code_switch_counts(record: ClinicalRecord, summary: str) -> tuple[int, int]:
    normalized = normalize_persian(summary).casefold()
    correct = 0
    for term in record.code_switched_terms:
        acceptable = term.get("acceptable", []) or [term.get("source", "")]
        if any(normalize_persian(str(item)).casefold() in normalized for item in acceptable if item):
            correct += 1
    return correct, len(record.code_switched_terms)


def evaluate_record(record: ClinicalRecord, prediction: SummaryPrediction) -> RecordEvaluation:
    source_ids = {feature.feature_id for feature in record.features}
    predicted_ids = set(prediction.included_feature_ids)
    required_ids = set(record.required_feature_ids)
    p, r, f1 = prf_from_sets(predicted_ids, required_ids)
    mandatory = {feature.feature_id for feature in record.features if feature.mandatory}
    rouge = text_rouge(prediction.summary, record.reference_summary) if record.reference_summary else {}
    code_correct, code_total = _code_switch_counts(record, prediction.summary)
    return RecordEvaluation(
        record_id=record.record_id,
        patient_id_hash=record.patient_id_hash,
        specialty=record.specialty,
        site_id=record.site_id,
        language_profile=record.language_profile,
        condition=prediction.condition,
        precision=p,
        recall=r,
        f1=f1,
        rouge_1=rouge.get("rouge_1"),
        rouge_2=rouge.get("rouge_2"),
        rouge_l=rouge.get("rouge_l"),
        unsupported_feature_count=len(predicted_ids - source_ids),
        predicted_feature_count=len(predicted_ids),
        mandatory_omission_count=len(mandatory - predicted_ids),
        mandatory_feature_count=len(mandatory),
        code_switch_correct=code_correct,
        code_switch_total=code_total,
    )


def aggregate_evaluations(rows: Iterable[RecordEvaluation]) -> dict:
    rows = list(rows)
    if not rows:
        raise ValueError("no evaluation rows")
    numeric = ("precision", "recall", "f1", "rouge_1", "rouge_2", "rouge_l")
    result = {key: float(np.mean([getattr(row, key) for row in rows if getattr(row, key) is not None]))
              for key in numeric}
    unsupported = sum(row.unsupported_feature_count for row in rows)
    predicted = sum(row.predicted_feature_count for row in rows)
    mandatory_omissions = sum(row.mandatory_omission_count for row in rows)
    mandatory_total = sum(row.mandatory_feature_count for row in rows)
    cs_correct = sum(row.code_switch_correct for row in rows)
    cs_total = sum(row.code_switch_total for row in rows)
    result.update(
        n_records=len(rows),
        unsupported_feature_rate=unsupported / predicted if predicted else 0.0,
        mandatory_omission_rate=mandatory_omissions / mandatory_total if mandatory_total else 0.0,
        code_switch_accuracy=cs_correct / cs_total if cs_total else None,
        code_switch_correct=cs_correct,
        code_switch_total=cs_total,
    )
    return result
