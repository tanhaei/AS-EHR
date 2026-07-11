"""Persian normalization, tokenization, and direct-identifier tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ehr_summ.deidentification import redact_direct_identifiers
from ehr_summ.preprocessing import detect_language_profile, normalize_persian, tokenize_clinical


def test_arabic_character_and_digit_normalization():
    assert normalize_persian("كتاب ۱۲٣") == "کتاب 123"


def test_code_switch_detection():
    assert detect_language_profile("بیمار Losartan مصرف می‌کند") == "code-switched"
    assert detect_language_profile("فقط متن فارسی") == "persian"
    assert detect_language_profile("English note") == "english"


def test_clinical_tokenizer_preserves_mixed_terms():
    tokens = tokenize_clinical("HbA1c برابر 8.2% است")
    assert "hba1c" in tokens and "8.2" in tokens


def test_direct_identifier_redaction():
    report = redact_direct_identifiers("email test@example.com phone 09121234567")
    assert "test@example.com" not in report.text
    assert "09121234567" not in report.text
    assert report.counts["EMAIL"] == 1 and report.counts["PHONE"] == 1

