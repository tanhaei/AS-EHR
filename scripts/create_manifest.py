#!/usr/bin/env python3
"""Create a SHA-256 manifest for governed split/config/evidence artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    rows = []
    for value in args.files:
        path = Path(value)
        rows.append({"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"artifacts": rows}, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {len(rows)} hashes to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

