#!/usr/bin/env bash
# Convenience: run every reproduction script + the test suite.
set -e
echo "==> reproduce_tables";        python scripts/reproduce_tables.py
echo; echo "==> threshold_sweep";   python scripts/run_threshold_sweep.py
echo; echo "==> statistical_analysis"; python scripts/run_statistical_analysis.py
echo; echo "==> make_figures";      python scripts/make_figures.py
echo; echo "==> tests";             python tests/run_tests.py
