"""
Pipeline Validation v3.1
========================

Validates the Robust KM Pipeline against ground truth RCT data.
Generates a TruthCert-compatible report with provenance for every number.

Reports separate metrics for:
  - "Reported" ground truth (HRs from paper text — high-confidence)
  - "Derived" ground truth (HRs from event rates — lower-confidence)
  - Overall

All outputs are UNCERTIFIED.
"""

import sys
sys.path.insert(0, r"C:\Users\user\Downloads\wasserstein")

import os
import json
import time
import numpy as np
from datetime import datetime
from robust_km_pipeline import RobustKMPipeline, HRExtractionResult
from rct_ground_truth import get_available, RCTGroundTruth
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("validate")


def validate_single(pipeline: RobustKMPipeline,
                    gt: RCTGroundTruth) -> dict:
    """Validate pipeline on one trial. Returns result dict."""
    src = "R" if gt.hr_source == "reported" else "D"
    print(f"\n{'='*60}")
    print(f"  {gt.name}  [{src}]")
    print(f"  Ground truth: HR={gt.hr} ({gt.ci_lower}-{gt.ci_upper})"
          f"  source={gt.hr_source}")
    print(f"{'='*60}")

    t0 = time.time()
    result = pipeline.extract_hr(gt.pdf_path)
    elapsed = time.time() - t0

    entry = {
        "trial": gt.name,
        "hr_true": gt.hr,
        "ci_lower_true": gt.ci_lower,
        "ci_upper_true": gt.ci_upper,
        "hr_source": gt.hr_source,
        "elapsed_s": round(elapsed, 1),
    }

    if result.succeeded:
        hr = result.hr
        error_pct = abs(hr - gt.hr) / gt.hr * 100
        within_ci = gt.ci_lower <= hr <= gt.ci_upper
        status = "OK" if within_ci else "MISS"

        entry.update({
            "status": status,
            "hr_extracted": hr,
            "ci_lower_ext": result.ci_lower,
            "ci_upper_ext": result.ci_upper,
            "error_pct": round(error_pct, 2),
            "within_ci": within_ci,
            "confidence": result.confidence,
            "method": result.provenance.orientation_method,
            "text_hr": result.provenance.text_hr_found,
            "pair": result.provenance.pair_description,
            "pair_rank": result.provenance.pair_rank,
            "orientation": result.provenance.orientation,
            "hr_fwd": result.provenance.hr_fwd,
            "hr_inv": result.provenance.hr_inv,
            "n_curves": result.provenance.total_curves_extracted,
            "pdf_hash": result.provenance.pdf_sha256,
            "warnings": result.warnings,
            "certification": result.certification_status,
        })

        print(f"  Extracted:  HR={hr:.4f}  Error={error_pct:.1f}%  {status}")
        print(f"  Method:     {result.provenance.orientation_method}")
        print(f"  Text HR:    {result.provenance.text_hr_found}")
        print(f"  Pair:       {result.provenance.pair_description}")
        print(f"  HR fwd/inv: {result.provenance.hr_fwd} / "
              f"{result.provenance.hr_inv}")
        print(f"  Confidence: {result.confidence}")
        print(f"  Time:       {elapsed:.1f}s")
        if result.warnings:
            for w in result.warnings:
                print(f"  WARNING: {w}")
    else:
        entry.update({
            "status": "FAILED",
            "hr_extracted": None,
            "error_pct": None,
            "within_ci": False,
            "error_msg": result.error,
        })
        print(f"  FAILED: {result.error}")

    return entry


def make_serializable(obj):
    """Convert numpy/bool types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, bool):
        return obj
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj


def print_report(results: list):
    """Print summary report with separate sections for reported vs derived."""
    total = len(results)
    successful = [r for r in results if r.get("hr_extracted") is not None]
    within_ci = [r for r in successful if r["within_ci"]]
    errors = [r["error_pct"] for r in successful]

    # Split by source
    reported = [r for r in results if r.get("hr_source") == "reported"]
    derived = [r for r in results if r.get("hr_source") == "derived"]

    rep_success = [r for r in reported if r.get("hr_extracted") is not None]
    rep_ci = [r for r in rep_success if r["within_ci"]]
    rep_errors = [r["error_pct"] for r in rep_success]

    der_success = [r for r in derived if r.get("hr_extracted") is not None]
    der_ci = [r for r in der_success if r["within_ci"]]
    der_errors = [r["error_pct"] for r in der_success]

    text_matched = [r for r in successful
                    if r.get("method") == "text_hr_match"]

    print("\n" + "=" * 70)
    print("  VALIDATION REPORT v3.1  (UNCERTIFIED)")
    print("=" * 70)

    print(f"\n  Pipeline version:   3.0 (two-stage pair selection)")
    print(f"  Timestamp:          {datetime.now().isoformat()}")
    print(f"  Trials tested:      {total} "
          f"({len(reported)} reported, {len(derived)} derived)")
    print(f"  HR extracted:       {len(successful)}/{total} "
          f"({len(successful)/total*100:.0f}%)")
    print(f"  Within GT CI:       {len(within_ci)}/{len(successful)} "
          f"({len(within_ci)/len(successful)*100:.0f}%)" if successful else "")

    # Reported-only metrics (high confidence ground truth)
    if rep_success:
        print(f"\n  --- REPORTED ground truth ({len(reported)} trials) ---")
        print(f"  HR extracted:       {len(rep_success)}/{len(reported)}")
        print(f"  Within reported CI: {len(rep_ci)}/{len(rep_success)} "
              f"({len(rep_ci)/len(rep_success)*100:.0f}%)")
        if rep_errors:
            print(f"  Mean error:         {np.mean(rep_errors):.1f}%")
            print(f"  Median error:       {np.median(rep_errors):.1f}%")

    # Derived metrics (lower confidence ground truth)
    if der_success:
        print(f"\n  --- DERIVED ground truth ({len(derived)} trials) ---")
        print(f"  HR extracted:       {len(der_success)}/{len(derived)}")
        print(f"  Within derived CI:  {len(der_ci)}/{len(der_success)} "
              f"({len(der_ci)/len(der_success)*100:.0f}%)")
        if der_errors:
            print(f"  Mean error:         {np.mean(der_errors):.1f}%")
            print(f"  Median error:       {np.median(der_errors):.1f}%")

    # Overall error distribution
    if errors:
        under_10 = [r for r in successful if r["error_pct"] < 10]
        under_15 = [r for r in successful if r["error_pct"] < 15]
        under_20 = [r for r in successful if r["error_pct"] < 20]
        print(f"\n  Overall error distribution:")
        print(f"    <10%:  {len(under_10)}/{len(successful)}")
        print(f"    <15%:  {len(under_15)}/{len(successful)}")
        print(f"    <20%:  {len(under_20)}/{len(successful)}")
        print(f"    Mean:  {np.mean(errors):.1f}%")
        print(f"    Median:{np.median(errors):.1f}%")
        print(f"    Max:   {np.max(errors):.1f}%")

    print(f"\n  Text HR cross-validation: "
          f"{len(text_matched)}/{len(successful)} trials")

    print(f"\n  Per-trial results:")
    print(f"  {'Trial':<22s} {'Src':>3s} {'HR ext':>7s} {'HR true':>7s} "
          f"{'Error':>7s} {'CI?':>4s} {'Method':<16s} {'Conf':>5s}")
    print(f"  {'-'*22} {'-'*3} {'-'*7} {'-'*7} {'-'*7} {'-'*4} "
          f"{'-'*16} {'-'*5}")

    for r in results:
        src = "R" if r.get("hr_source") == "reported" else "D"
        if r.get("hr_extracted") is not None:
            ci_str = "OK" if r["within_ci"] else "MISS"
            print(f"  {r['trial']:<22s} {src:>3s} {r['hr_extracted']:7.3f} "
                  f"{r['hr_true']:7.3f} {r['error_pct']:6.1f}% "
                  f"{ci_str:>4s} {r.get('method',''):16s} "
                  f"{r.get('confidence',0):5.2f}")
        else:
            print(f"  {r['trial']:<22s} {src:>3s} {'FAILED':>7s}")

    print(f"\n  Certification: ALL OUTPUTS ARE UNCERTIFIED")
    print("=" * 70)


def main():
    print("=" * 70)
    print("  KM Pipeline v3.1 Validation (corrected ground truth)")
    print(f"  {datetime.now().isoformat()}")
    print("=" * 70)

    available = get_available()
    reported = [g for g in available if g.hr_source == "reported"]
    derived = [g for g in available if g.hr_source == "derived"]
    print(f"\n  {len(available)} trials available "
          f"({len(reported)} reported, {len(derived)} derived)\n")

    pipeline = RobustKMPipeline(top_k_pairs=3)
    results = []

    for gt in available:
        entry = validate_single(pipeline, gt)
        results.append(entry)

    print_report(results)

    # Save JSON report
    out_dir = os.path.join(os.path.dirname(__file__), "validation_results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir,
        f"v3.1_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    report = make_serializable({
        "pipeline_version": "3.1",
        "timestamp": datetime.now().isoformat(),
        "n_trials": len(results),
        "n_reported": len(reported),
        "n_derived": len(derived),
        "n_successful": len([r for r in results
                            if r.get("hr_extracted") is not None]),
        "n_within_ci": len([r for r in results
                           if r.get("within_ci")]),
        "certification_status": "UNCERTIFIED",
        "results": results,
    })

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Report saved: {out_path}")
    return results


if __name__ == "__main__":
    main()
