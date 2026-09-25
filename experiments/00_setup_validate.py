"""
Phase 0 — Setup & data validation.

What this phase does
--------------------
Confirms the PaySim file is readable, computes *all* dataset statistics
from the file (never from the paper), and writes a quality report.

Why it exists
-------------
Every later table must be traceable to this scan. If fraud+legit != n,
we abort.

How to run
----------
python experiments/00_setup_validate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_validation import validate_paysim, write_quality_report
from src.utils import get_logger, load_config, results_paths, save_json, set_global_seed


def main() -> None:
    cfg = load_config()
    set_global_seed(int(cfg["project"]["seed"]))
    paths = results_paths(cfg)
    logger = get_logger("phase0", paths["logs"] / "phase0.log", cfg["logging"]["level"])
    logger.info("Phase 0: scanning %s", cfg["paths"]["raw_csv"])
    report = validate_paysim(cfg)
    out = paths["metrics"] / "phase0_data_quality.json"
    write_quality_report(report, out)
    # Human-readable companion
    save_json(
        {
            "headline": {
                "n_rows": report["n_rows"],
                "n_fraud": report["class_counts"]["fraud"],
                "n_legit": report["class_counts"]["legitimate"],
                "fraud_rate": report["fraud_rate"],
                "imbalance_ratio": report["imbalance_ratio_legit_per_fraud"],
                "step_range": report["step_range"],
                "type_distribution": report["type_distribution"],
            }
        },
        paths["tables"] / "phase0_headline.json",
    )
    logger.info(
        "n=%s fraud=%s rate=%.6f",
        f"{report['n_rows']:,}",
        f"{report['class_counts']['fraud']:,}",
        report["fraud_rate"],
    )
    print("Phase 0 complete:", out)


if __name__ == "__main__":
    main()
