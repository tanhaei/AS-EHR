#!/usr/bin/env python3
"""Minimal stdlib test runner so the suite runs without pytest installed.

Discovers every ``test_*`` function in ``tests/test_*.py`` and executes it.
With pytest available you can instead just run ``pytest -q``.

Run:  python tests/run_tests.py
"""
from __future__ import annotations

import importlib.util
import traceback
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    files = sorted(p for p in TESTS_DIR.glob("test_*.py"))
    passed = failed = 0
    failures: list[str] = []

    for f in files:
        mod = _load_module(f)
        fns = sorted(name for name in dir(mod) if name.startswith("test_") and callable(getattr(mod, name)))
        for name in fns:
            try:
                getattr(mod, name)()
                passed += 1
                print(f"  PASS  {f.name}::{name}")
            except Exception:  # noqa: BLE001
                failed += 1
                failures.append(f"{f.name}::{name}\n{traceback.format_exc()}")
                print(f"  FAIL  {f.name}::{name}")

    print(f"\n{passed} passed, {failed} failed")
    if failures:
        print("\n===== FAILURE DETAILS =====")
        for fail in failures:
            print(fail)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
