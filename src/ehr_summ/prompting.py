"""Grounded prompt construction for specialty-aware summaries."""
from __future__ import annotations

import json
from typing import Sequence

from .schemas import ClinicalFeature, ClinicalRecord


SYSTEM_PROMPT = """You are a clinical summarization component inside an EHR.
Use only the supplied record. Do not infer absent diagnoses, values, dates, or
medication doses. Preserve negation and temporality. Return valid JSON with
exactly two keys: summary and included_feature_ids. The second value must list
only source feature IDs that support the summary."""


def build_user_prompt(record: ClinicalRecord, retained: Sequence[ClinicalFeature]) -> str:
    payload = {
        "target_specialty": record.specialty,
        "language_profile": record.language_profile,
        "note": record.note,
        "features": [feature.__dict__ for feature in retained],
        "requirements": {
            "language": "Persian unless an English clinical term should be preserved",
            "length_words": [150, 250],
            "mandatory_feature_ids": [feature.feature_id for feature in retained if feature.mandatory],
        },
    }
    return "Summarize this record:\n" + json.dumps(payload, ensure_ascii=False, sort_keys=True)

