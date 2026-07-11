"""Public record-schema and patient-level split tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ehr_summ.schemas import load_records
from ehr_summ.splits import assert_no_patient_overlap, patient_level_split

ROOT = Path(__file__).resolve().parents[1]


def test_example_records_validate():
    records = load_records(ROOT / "data/examples/records.jsonl")
    assert len(records) == 6
    assert all(record.metadata.get("synthetic") is True for record in records)


def test_patient_level_split_has_no_overlap():
    splits = patient_level_split(load_records(ROOT / "data/examples/records.jsonl"), seed=7)
    assert_no_patient_overlap(splits)
    assert sum(len(rows) for rows in splits.values()) == 6


def test_patient_split_is_deterministic():
    records = load_records(ROOT / "data/examples/records.jsonl")
    first = patient_level_split(records, seed=99)
    second = patient_level_split(records, seed=99)
    assert {name: [row.record_id for row in rows] for name, rows in first.items()} == {
        name: [row.record_id for row in rows] for name, rows in second.items()
    }

