"""Typed, PHI-free interchange schemas for the public pipeline."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


ALLOWED_LANGUAGE_PROFILES = {"persian", "english", "code-switched", "unknown"}


@dataclass
class ClinicalFeature:
    feature_id: str
    text: str
    category: str
    code: str | None = None
    code_system: str | None = None
    value: str | float | int | None = None
    unit: str | None = None
    timestamp: str | None = None
    mandatory: bool = False
    source: str = "structured"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ClinicalFeature":
        return cls(**value)


@dataclass
class ClinicalRecord:
    record_id: str
    patient_id_hash: str
    specialty: str
    site_id: str
    note: str
    language_profile: str = "unknown"
    features: list[ClinicalFeature] = field(default_factory=list)
    reference_summary: str | None = None
    required_feature_ids: list[str] = field(default_factory=list)
    code_switched_terms: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.record_id or not self.patient_id_hash:
            raise ValueError("record_id and patient_id_hash are required")
        if self.language_profile not in ALLOWED_LANGUAGE_PROFILES:
            raise ValueError(f"unsupported language_profile: {self.language_profile}")
        ids = [feature.feature_id for feature in self.features]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate feature_id in record {self.record_id}")
        unknown = set(self.required_feature_ids) - set(ids)
        if unknown:
            raise ValueError(f"required_feature_ids not present in features: {sorted(unknown)}")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ClinicalRecord":
        payload = dict(value)
        payload["features"] = [ClinicalFeature.from_dict(item) for item in payload.get("features", [])]
        record = cls(**payload)
        record.validate()
        return record

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SummaryPrediction:
    record_id: str
    condition: str
    summary: str
    included_feature_ids: list[str]
    model_id: str
    latency_seconds: float | None = None
    raw_output: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SummaryPrediction":
        return cls(**value)


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc


def load_records(path: str | Path) -> list[ClinicalRecord]:
    return [ClinicalRecord.from_dict(row) for row in read_jsonl(path)]


def load_predictions(path: str | Path) -> list[SummaryPrediction]:
    return [SummaryPrediction.from_dict(row) for row in read_jsonl(path)]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

