"""Run phases 0→5 in order. Use this after config is finalized."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PHASES = [
    "00_setup_validate.py",
    "01_original_baseline.py",
    "02_temporal_features.py",
    "03_ep_iail.py",
    "04_evaluation_suite.py",
    "05_write_report.py",
]


def main() -> None:
    here = Path(__file__).resolve().parent
    for name in PHASES:
        print("=" * 72)
        print("RUNNING", name)
        print("=" * 72)
        runpy.run_path(str(here / name), run_name="__main__")


if __name__ == "__main__":
    main()
