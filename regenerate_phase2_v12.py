"""
regenerate_phase2_v12.py — Regenerate phase2_v2_results with v1.2 pipeline
============================================================================
Re-runs KMPipeline on all 13 gold-tier trials and saves curves/IPD/summary
to phase2_v2_results/ for R cross-validation.
"""

import os
import sys
import io
import time
from pathlib import Path

WASSERSTEIN_DIR = Path(__file__).parent
sys.path.insert(0, str(WASSERSTEIN_DIR))

from km_pipeline import KMPipeline, _safe_stem, write_ipd_csv, write_curves_csv, write_summary_json

# Same 13 gold trials as regression_validate_13.py
GOLD_PDFS = [
    "Cardiovasc electrophysiol - 2015 - HUNTER - Point\u2010by\u2010Point Radiofrequency Ablation Versus the Cryoballoon or a Novel.pdf",
    "Cardiovasc electrophysiol - 2023 - Mililis - Radiofrequency versus cryoballoon catheter ablation in patients with.pdf",
    "NEJMoa1602014.pdf",
    "andrade-et-al-cryoballoon-or-radiofrequency-ablation-for-atrial-fibrillation-assessed-by-continuous-monitoring.pdf",
    "chun-et-al-cryoballoon-versus-laserballoon.pdf",
    "ehaf451.pdf",
    "euaf066.pdf",
    "eut398.pdf",
    "euu064.pdf",
    "pak-et-al-cryoballoon-versus-high-power-short-duration-radiofrequency-ablation-for-pulmonary-vein-isolation-in-patients.pdf",
    "reddy-et-al-pulsed-field-vs-conventional-thermal-ablation-for-paroxysmal-atrial-fibrillation.pdf",
    "s12872-017-0566-6.pdf",
    "schmidt-et-al-laser-balloon-or-wide-area-circumferential-irrigated-radiofrequency-ablation-for-persistent-atrial.pdf",
]

PDF_DIR = Path(r"C:\Users\user\Downloads\Ablation")
OUTPUT_DIR = WASSERSTEIN_DIR / "phase2_v2_results"


def main():
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_start = time.time()
    for i, pdf_name in enumerate(GOLD_PDFS, 1):
        pdf_path = PDF_DIR / pdf_name
        stem = _safe_stem(pdf_name)
        print(f"\n[{i}/13] {stem}")

        if not pdf_path.exists():
            print(f"  SKIP: PDF not found")
            continue

        pipeline = KMPipeline(dpi=300, max_pages=12, n_per_arm=100)
        t0 = time.time()
        try:
            result = pipeline.extract(str(pdf_path))
            elapsed = time.time() - t0
            print(f"  HR={result.hr}, IPD={result.total_ipd_records}, Time={elapsed:.0f}s")

            # Write outputs
            write_summary_json(result, str(OUTPUT_DIR / f"{stem}_summary.json"))
            if result.has_ipd:
                write_ipd_csv(result, str(OUTPUT_DIR / f"{stem}_ipd.csv"))
            if result.curve1_times:
                write_curves_csv(result, str(OUTPUT_DIR / f"{stem}_curves.csv"))
        except Exception as e:
            print(f"  ERROR: {e}")

    total = time.time() - total_start
    print(f"\nDone. Total time: {total:.0f}s ({total/60:.1f} min)")


if __name__ == '__main__':
    main()
