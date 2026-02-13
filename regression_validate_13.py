"""
regression_validate_13.py — Phase 2 Regression Validation
==========================================================
Runs km_pipeline on the 13 gold-tier trials and compares HR extraction
against ground truth. Reports pass/fail per trial and overall.

Criteria (from PLAN_SIRATAL_MUSTAQEEM.md):
  - All 13 must extract successfully
  - All 13 HRs within reported CI (100% within CI)
  - 12/13 should have relative error <10%
"""

import os
import sys
import io
import json
import time
from pathlib import Path
from datetime import datetime, timezone

# UTF-8 stdout for Unicode filenames
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add wasserstein dir to path
WASSERSTEIN_DIR = Path(__file__).parent
sys.path.insert(0, str(WASSERSTEIN_DIR))

from km_pipeline import KMPipeline, _safe_stem, write_summary_json

# ---------------------------------------------------------------------------
# Ground truth for the 13 gold-tier trials
# ---------------------------------------------------------------------------
GOLD_TRIALS = [
    {
        "name": "HUNTER",
        "pdf": "Cardiovasc electrophysiol - 2015 - HUNTER - Point\u2010by\u2010Point Radiofrequency Ablation Versus the Cryoballoon or a Novel.pdf",
        "gt_hr": 0.53, "gt_ci_lower": 0.25, "gt_ci_upper": 1.1,
        "hr_source": "derived",
    },
    {
        "name": "MILILIS-PERS",
        "pdf": "Cardiovasc electrophysiol - 2023 - Mililis - Radiofrequency versus cryoballoon catheter ablation in patients with.pdf",
        "gt_hr": 0.96, "gt_ci_lower": 0.5, "gt_ci_upper": 1.8,
        "hr_source": "derived",
    },
    {
        "name": "FIRE AND ICE",
        "pdf": "NEJMoa1602014.pdf",
        "gt_hr": 0.96, "gt_ci_lower": 0.76, "gt_ci_upper": 1.22,
        "hr_source": "reported",
    },
    {
        "name": "CIRCA-DOSE",
        "pdf": "andrade-et-al-cryoballoon-or-radiofrequency-ablation-for-atrial-fibrillation-assessed-by-continuous-monitoring.pdf",
        "gt_hr": 1.08, "gt_ci_lower": 0.78, "gt_ci_upper": 1.5,
        "hr_source": "reported",
    },
    {
        "name": "FROZEN AF",
        "pdf": "chun-et-al-cryoballoon-versus-laserballoon.pdf",
        "gt_hr": 0.9, "gt_ci_lower": 0.5, "gt_ci_upper": 1.6,
        "hr_source": "derived",
    },
    {
        "name": "CRRF-PeAF",
        "pdf": "ehaf451.pdf",
        "gt_hr": 0.99, "gt_ci_lower": 0.69, "gt_ci_upper": 1.43,
        "hr_source": "reported",
    },
    {
        "name": "HIPAF",
        "pdf": "euaf066.pdf",
        "gt_hr": 1.47, "gt_ci_lower": 0.7, "gt_ci_upper": 2.5,
        "hr_source": "derived",
    },
    {
        "name": "PVAC-CPVI",
        "pdf": "eut398.pdf",
        "gt_hr": 0.72, "gt_ci_lower": 0.35, "gt_ci_upper": 1.5,
        "hr_source": "derived",
    },
    {
        "name": "WACA-PVAC",
        "pdf": "euu064.pdf",
        "gt_hr": 1.14, "gt_ci_lower": 0.6, "gt_ci_upper": 2.1,
        "hr_source": "derived",
    },
    {
        "name": "CRAVE",
        "pdf": "pak-et-al-cryoballoon-versus-high-power-short-duration-radiofrequency-ablation-for-pulmonary-vein-isolation-in-patients.pdf",
        "gt_hr": 0.91, "gt_ci_lower": 0.45, "gt_ci_upper": 1.85,
        "hr_source": "derived",
    },
    {
        "name": "ADVENT",
        "pdf": "reddy-et-al-pulsed-field-vs-conventional-thermal-ablation-for-paroxysmal-atrial-fibrillation.pdf",
        "gt_hr": 0.92, "gt_ci_lower": 0.55, "gt_ci_upper": 1.5,
        "hr_source": "derived",
    },
    {
        "name": "FREEZEAF-30M",
        "pdf": "s12872-017-0566-6.pdf",
        "gt_hr": 1.06, "gt_ci_lower": 0.55, "gt_ci_upper": 2.0,
        "hr_source": "derived",
    },
    {
        "name": "LBRF-PERSISTENT",
        "pdf": "schmidt-et-al-laser-balloon-or-wide-area-circumferential-irrigated-radiofrequency-ablation-for-persistent-atrial.pdf",
        "gt_hr": 0.93, "gt_ci_lower": 0.5, "gt_ci_upper": 1.7,
        "hr_source": "derived",
    },
]

PDF_DIR = Path(r"C:\Users\user\Downloads\Ablation")
OUTPUT_DIR = WASSERSTEIN_DIR / "regression_13_results"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Verify all PDFs exist
    missing = []
    for trial in GOLD_TRIALS:
        pdf_path = PDF_DIR / trial["pdf"]
        if not pdf_path.exists():
            missing.append(trial["name"])
    if missing:
        print(f"ERROR: Missing PDFs for: {missing}")
        sys.exit(1)

    print("=" * 70)
    print("REGRESSION VALIDATION: 13 Gold-Tier Trials")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    pipeline = KMPipeline(dpi=300, max_pages=12, n_per_arm=100)
    results = []
    total_start = time.time()

    for i, trial in enumerate(GOLD_TRIALS, 1):
        pdf_path = PDF_DIR / trial["pdf"]
        name = trial["name"]
        gt_hr = trial["gt_hr"]
        gt_lo = trial["gt_ci_lower"]
        gt_hi = trial["gt_ci_upper"]

        print(f"\n[{i}/13] {name}")
        print(f"  PDF: {trial['pdf'][:60]}...")
        print(f"  Ground Truth: HR={gt_hr} [{gt_lo}, {gt_hi}]")

        t0 = time.time()
        try:
            result = pipeline.extract(str(pdf_path))
            elapsed = time.time() - t0

            if result.succeeded and result.hr is not None:
                ext_hr = result.hr
                ext_lo = result.ci_lower
                ext_hi = result.ci_upper

                # Check: extracted HR within ground truth CI?
                within_ci = gt_lo <= ext_hr <= gt_hi
                # Check: also consider reciprocal (arm ordering)
                recip_hr = 1.0 / ext_hr if ext_hr > 0 else None
                within_ci_recip = (recip_hr is not None and
                                   gt_lo <= recip_hr <= gt_hi)

                # Use whichever orientation is closer to GT
                if not within_ci and within_ci_recip:
                    # Reciprocal is better match
                    rel_error = abs(recip_hr - gt_hr) / gt_hr * 100
                    orientation = "reciprocal"
                    final_hr = recip_hr
                else:
                    rel_error = abs(ext_hr - gt_hr) / gt_hr * 100
                    orientation = "direct"
                    final_hr = ext_hr

                within_ci_final = gt_lo <= final_hr <= gt_hi

                status = "PASS" if within_ci_final else "FAIL"
                error_ok = rel_error < 10.0

                ci_str = (f"[{ext_lo:.3f}, {ext_hi:.3f}]"
                          if ext_lo is not None and ext_hi is not None
                          else "[N/A]")
                print(f"  Extracted: HR={ext_hr:.3f} {ci_str}")
                if orientation == "reciprocal":
                    print(f"  Reciprocal: HR={recip_hr:.3f} (better match)")
                print(f"  Rel Error: {rel_error:.1f}% {'OK' if error_ok else '>10%'}")
                print(f"  Within CI: {within_ci_final} -> {status}")
                print(f"  Time: {elapsed:.1f}s")

                # Save per-trial summary
                stem = _safe_stem(trial["pdf"])
                write_summary_json(result, str(OUTPUT_DIR / f"{stem}_summary.json"))

                results.append({
                    "name": name,
                    "pdf": trial["pdf"],
                    "gt_hr": gt_hr,
                    "gt_ci": [gt_lo, gt_hi],
                    "ext_hr": ext_hr,
                    "ext_ci": [ext_lo, ext_hi],
                    "final_hr": final_hr,
                    "orientation": orientation,
                    "rel_error_pct": round(rel_error, 2),
                    "within_ci": within_ci_final,
                    "error_under_10pct": error_ok,
                    "ipd_records": result.total_ipd_records,
                    "time_s": round(elapsed, 1),
                    "status": status,
                })
            else:
                elapsed = time.time() - t0
                warnings = result.warnings if result.warnings else ["Unknown failure"]
                print(f"  FAILED: {warnings[0]}")
                print(f"  Time: {elapsed:.1f}s")
                results.append({
                    "name": name,
                    "pdf": trial["pdf"],
                    "gt_hr": gt_hr,
                    "ext_hr": None,
                    "within_ci": False,
                    "error_under_10pct": False,
                    "time_s": round(elapsed, 1),
                    "status": "FAIL_EXTRACT",
                    "error": warnings[0],
                })

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  EXCEPTION: {e}")
            print(f"  Time: {elapsed:.1f}s")
            results.append({
                "name": name,
                "pdf": trial["pdf"],
                "gt_hr": gt_hr,
                "ext_hr": None,
                "within_ci": False,
                "error_under_10pct": False,
                "time_s": round(elapsed, 1),
                "status": "EXCEPTION",
                "error": str(e),
            })

    total_elapsed = time.time() - total_start

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    n_extracted = sum(1 for r in results if r["status"] != "FAIL_EXTRACT" and r["status"] != "EXCEPTION")
    n_within_ci = sum(1 for r in results if r.get("within_ci", False))
    n_under_10 = sum(1 for r in results if r.get("error_under_10pct", False))
    errors = [r.get("rel_error_pct", None) for r in results if r.get("rel_error_pct") is not None]
    mean_error = sum(errors) / len(errors) if errors else 0
    median_error = sorted(errors)[len(errors) // 2] if errors else 0

    print("\n" + "=" * 70)
    print("REGRESSION VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  Extracted:    {n_extracted}/13")
    print(f"  Within CI:    {n_within_ci}/13 (target: 13/13)")
    print(f"  Error <10%:   {n_under_10}/13 (target: 12/13)")
    print(f"  Mean Error:   {mean_error:.1f}%")
    print(f"  Median Error: {median_error:.1f}%")
    print(f"  Total Time:   {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print()

    # Per-trial table
    print(f"{'Trial':<20} {'GT HR':>6} {'Ext HR':>7} {'Err%':>6} {'CI?':>4} {'Status':>8} {'Time':>6}")
    print("-" * 70)
    for r in results:
        name = r["name"][:19]
        gt = f"{r['gt_hr']:.2f}"
        ext = f"{r.get('final_hr', r.get('ext_hr', 'N/A'))}" if r.get("final_hr") else "N/A"
        if isinstance(ext, float):
            ext = f"{ext:.3f}"
        err = f"{r['rel_error_pct']:.1f}" if r.get("rel_error_pct") is not None else "N/A"
        ci = "Y" if r.get("within_ci") else "N"
        status = r["status"]
        tm = f"{r['time_s']:.0f}s"
        print(f"{name:<20} {gt:>6} {ext:>7} {err:>6} {ci:>4} {status:>8} {tm:>6}")

    # Overall verdict
    regression_pass = (n_extracted == 13 and n_within_ci >= 13 and n_under_10 >= 12)
    print()
    if regression_pass:
        print("VERDICT: PASS — No regression detected")
    else:
        print("VERDICT: FAIL — Regression detected!")
        if n_extracted < 13:
            print(f"  - Only {n_extracted}/13 extracted (need 13/13)")
        if n_within_ci < 13:
            print(f"  - Only {n_within_ci}/13 within CI (need 13/13)")
        if n_under_10 < 12:
            print(f"  - Only {n_under_10}/13 under 10% error (need 12/13)")

    # Save full results
    report = {
        "validation": "regression_13_gold_trials",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_time_s": round(total_elapsed, 1),
        "n_extracted": n_extracted,
        "n_within_ci": n_within_ci,
        "n_under_10pct": n_under_10,
        "mean_error_pct": round(mean_error, 2),
        "median_error_pct": round(median_error, 2),
        "verdict": "PASS" if regression_pass else "FAIL",
        "trials": results,
    }
    report_path = OUTPUT_DIR / "regression_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report: {report_path}")


if __name__ == '__main__':
    main()
