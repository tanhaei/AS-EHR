"""Configurable adapter from structured EHR payloads to canonical features."""
from __future__ import annotations

import hashlib
import hmac
from typing import Any, Iterable

from .deidentification import redact_direct_identifiers
from .knowledge_base import KnowledgeConcept
from .preprocessing import normalize_persian
from .schemas import ClinicalFeature, ClinicalRecord


DEFAULT_COLLECTIONS = {
    "diagnoses": "diagnosis",
    "medications": "medication",
    "laboratory_results": "laboratory",
    "procedures": "procedure",
    "allergies": "allergy",
}


def hash_patient_identifier(identifier: str, salt: str) -> str:
    if not identifier or not salt:
        raise ValueError("identifier and an institution-held salt are required")
    return hmac.new(salt.encode(), identifier.encode(), hashlib.sha256).hexdigest()


def _stable_feature_id(record_id: str, category: str, index: int, item: dict[str, Any]) -> str:
    source = f"{record_id}|{category}|{index}|{item.get('code')}|{item.get('descriptor') or item.get('name')}"
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:20]


def _feature_text(item: dict[str, Any]) -> str:
    descriptor = item.get("descriptor") or item.get("name") or item.get("text")
    if not descriptor:
        raise ValueError("structured item requires descriptor, name, or text")
    parts = [str(descriptor)]
    if item.get("value") is not None:
        parts.append(str(item["value"]))
    if item.get("unit"):
        parts.append(str(item["unit"]))
    return normalize_persian(" ".join(parts))


def structured_features(payload: dict[str, Any], record_id: str,
                        collections: dict[str, str] | None = None) -> list[ClinicalFeature]:
    features: list[ClinicalFeature] = []
    for source_key, category in (collections or DEFAULT_COLLECTIONS).items():
        for index, item in enumerate(payload.get(source_key, []) or []):
            if not isinstance(item, dict):
                raise ValueError(f"{source_key}[{index}] must be an object")
            features.append(
                ClinicalFeature(
                    feature_id=str(
                        item.get("feature_id") or _stable_feature_id(record_id, category, index, item)
                    ),
                    text=_feature_text(item),
                    category=category,
                    code=str(item["code"]) if item.get("code") is not None else None,
                    code_system=str(item["code_system"]) if item.get("code_system") else None,
                    value=item.get("value"),
                    unit=item.get("unit"),
                    timestamp=item.get("timestamp"),
                    mandatory=bool(item.get("mandatory", category == "allergy")),
                    source="structured",
                )
            )
    return features


def note_concept_features(
    note: str, concepts: Iterable[KnowledgeConcept], record_id: str
) -> list[ClinicalFeature]:
    normalized = normalize_persian(note).casefold()
    features: list[ClinicalFeature] = []
    for concept in concepts:
        terms = (concept.descriptor, *concept.synonyms)
        if any(normalize_persian(term).casefold() in normalized for term in terms):
            feature_id = hashlib.sha256(f"{record_id}|note|{concept.concept_id}".encode()).hexdigest()[:20]
            features.append(
                ClinicalFeature(feature_id, concept.descriptor, "note_concept",
                                code=concept.concept_id, mandatory=concept.mandatory, source="unstructured")
            )
    return features


def adapt_ehr_payload(payload: dict[str, Any], *, patient_salt: str,
                      concepts: Iterable[KnowledgeConcept] = (),
                      field_map: dict[str, str] | None = None) -> ClinicalRecord:
    """Convert a configurable EHR export object to the canonical schema."""
    mapping = {
        "record_id": "record_id",
        "patient_id": "patient_id",
        "specialty": "specialty",
        "site_id": "site_id",
        "note": "note",
    }
    mapping.update(field_map or {})
    record_id = str(payload[mapping["record_id"]])
    redacted = redact_direct_identifiers(str(payload.get(mapping["note"], ""))).text
    features = structured_features(payload, record_id)
    features.extend(note_concept_features(redacted, concepts, record_id))
    deduplicated = {feature.feature_id: feature for feature in features}
    record = ClinicalRecord(
        record_id=record_id,
        patient_id_hash=hash_patient_identifier(str(payload[mapping["patient_id"]]), patient_salt),
        specialty=str(payload[mapping["specialty"]]),
        site_id=str(payload[mapping["site_id"]]),
        note=redacted,
        features=list(deduplicated.values()),
        metadata={"adapted_from_ehr": True},
    )
    record.validate()
    return record
