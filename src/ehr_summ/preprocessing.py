"""Deterministic normalization for Persian and Persian-English clinical text."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import replace

from .schemas import ClinicalFeature, ClinicalRecord


_ARABIC_TO_PERSIAN = str.maketrans(
    {
        "ي": "ی",
        "ى": "ی",
        "ئ": "ی",
        "ك": "ک",
        "ة": "ه",
        "ۀ": "ه",
        "ؤ": "و",
        "ـ": "",
    }
)
_DIGITS_TO_ASCII = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_DIACRITICS = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
_SPACE = re.compile(r"[ \t\r\f\v]+")
_PERSIAN = re.compile(r"[\u0600-\u06ff]")
_LATIN = re.compile(r"[A-Za-z]")
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-/][A-Za-z0-9]+)*|[\u0600-\u06ff\u200c]+|\d+(?:[.,]\d+)?")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?؟])\s+|\n+")


def normalize_persian(text: str, *, ascii_digits: bool = True) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.translate(_ARABIC_TO_PERSIAN)
    text = _DIACRITICS.sub("", text)
    if ascii_digits:
        text = text.translate(_DIGITS_TO_ASCII)
    text = re.sub(r"\s*\u200c\s*", "\u200c", text)
    text = _SPACE.sub(" ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def detect_language_profile(text: str) -> str:
    persian_count = len(_PERSIAN.findall(text or ""))
    latin_count = len(_LATIN.findall(text or ""))
    if persian_count and latin_count:
        return "code-switched"
    if persian_count:
        return "persian"
    if latin_count:
        return "english"
    return "unknown"


def tokenize_clinical(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _TOKEN.finditer(normalize_persian(text))]


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_BOUNDARY.split(normalize_persian(text)) if part.strip()]


def normalize_feature(feature: ClinicalFeature) -> ClinicalFeature:
    value = normalize_persian(str(feature.value)) if feature.value is not None else None
    return replace(
        feature,
        text=normalize_persian(feature.text),
        code=feature.code.strip() if feature.code else None,
        code_system=feature.code_system.upper().strip() if feature.code_system else None,
        value=value,
        unit=normalize_persian(feature.unit) if feature.unit else None,
    )


def normalize_record(record: ClinicalRecord) -> ClinicalRecord:
    note = normalize_persian(record.note)
    profile = record.language_profile
    if profile == "unknown":
        profile = detect_language_profile(note)
    normalized = replace(record, note=note, language_profile=profile,
                         features=[normalize_feature(feature) for feature in record.features])
    normalized.validate()
    return normalized
