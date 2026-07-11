#!/usr/bin/env python3
"""Adapt a governed JSONL EHR export to the canonical pipeline schema."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ehr_summ.feature_extraction import adapt_ehr_payload
from ehr_summ.knowledge_base import load_knowledge_base
from ehr_summ.schemas import read_jsonl, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--knowledge-base", required=True)
    parser.add_argument("--field-map-json")
    parser.add_argument("--salt-env", default="EHR_PATIENT_HASH_SALT")
    args = parser.parse_args()
    salt = os.environ.get(args.salt_env)
    if not salt:
        raise RuntimeError(
            f"set the institution-held salt in environment variable {args.salt_env}"
        )
    field_map = (
        json.loads(Path(args.field_map_json).read_text(encoding="utf-8"))
        if args.field_map_json
        else None
    )
    concepts = load_knowledge_base(args.knowledge_base)
    records = [
        adapt_ehr_payload(payload, patient_salt=salt, concepts=concepts, field_map=field_map)
        for payload in read_jsonl(args.input)
    ]
    write_jsonl(args.output, (record.to_dict() for record in records))
    print(f"Wrote {len(records)} canonical records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
