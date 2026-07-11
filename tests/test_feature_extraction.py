"""Configurable EHR adapter tests with synthetic payloads only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ehr_summ.feature_extraction import adapt_ehr_payload, hash_patient_identifier
from ehr_summ.schemas import read_jsonl

ROOT = Path(__file__).resolve().parents[1]


def test_patient_hash_is_salted_and_deterministic():
    first = hash_patient_identifier("patient", "salt-a")
    second = hash_patient_identifier("patient", "salt-a")
    other = hash_patient_identifier("patient", "salt-b")
    assert first == second and first != other


def test_ehr_adapter_redacts_and_extracts_structured_features():
    payload = next(read_jsonl(ROOT / "data/examples/raw_ehr_export.jsonl"))
    record = adapt_ehr_payload(payload, patient_salt="test-only-salt")
    assert "test@example.com" not in record.note
    assert len(record.features) == 3
    assert any(feature.mandatory for feature in record.features)

