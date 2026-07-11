"""Conservative demonstration de-identification utilities.

This is a public reference implementation, not a certified clinical
de-identification system.  Institutions must validate recall on their own data
and retain their governed production de-identification process.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RedactionReport:
    text: str
    counts: dict[str, int]


PATTERNS = {
    "EMAIL": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "URL": re.compile(r"https?://\S+|www\.\S+", re.I),
    "PHONE": re.compile(r"(?<!\d)(?:\+?98|0)?9\d{9}(?!\d)"),
    "NATIONAL_ID": re.compile(r"(?<!\d)\d{10}(?!\d)"),
}


def redact_direct_identifiers(text: str) -> RedactionReport:
    counts: dict[str, int] = {}
    result = text or ""
    for label, pattern in PATTERNS.items():
        result, count = pattern.subn(f"[{label}]", result)
        counts[label] = count
    return RedactionReport(result, counts)

