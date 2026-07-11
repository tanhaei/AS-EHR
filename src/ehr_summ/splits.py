"""Leakage-resistant patient-level data splitting."""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Iterable

from .schemas import ClinicalRecord


def _bucket(patient_id_hash: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{patient_id_hash}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def patient_level_split(records: Iterable[ClinicalRecord], *, train: float = 0.70,
                        tune: float = 0.15, seed: int = 2026) -> dict[str, list[ClinicalRecord]]:
    if not 0 < train < 1 or not 0 <= tune < 1 or train + tune >= 1:
        raise ValueError("train and tune fractions must leave a non-empty test fraction")
    result = {"train": [], "tune": [], "test": []}
    for record in records:
        value = _bucket(record.patient_id_hash, seed)
        split = "train" if value < train else "tune" if value < train + tune else "test"
        result[split].append(record)
    assert_no_patient_overlap(result)
    return result


def assert_no_patient_overlap(splits: dict[str, Iterable[ClinicalRecord]]) -> None:
    ownership: dict[str, str] = {}
    for split_name, records in splits.items():
        for record in records:
            previous = ownership.setdefault(record.patient_id_hash, split_name)
            if previous != split_name:
                raise ValueError(f"patient leakage: {record.patient_id_hash} in {previous} and {split_name}")

