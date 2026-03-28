"""
Phase 4 batch runner — processes trials in chunks to avoid segfaults.
Saves results incrementally to phase4_results/batch_N.json.

Usage:
  python phase4_batch_runner.py [--batch N] [--batch-size M]

Default: runs batch 0 (trials 0-14). Run with --batch 1, 2, ... for more.
"""

import sys
import io
import json
import time
import platform
import argparse
from pathlib import Path

if sys.platform == 'win32':
    def _safe_wmi_query(*args, **kwargs):
        raise OSError("WMI bypassed")
    platform._wmi_query = _safe_wmi_query

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                  errors='replace')

sys.path.insert(0, str(Path(__file__).parent))
from km_pipeline import KMPipeline
from phase4_orientation import resolve_phase4_hr_orientation

DIR_MAP = {
    'oncology': Path(r'C:\Users\user\oncology_rcts'),
    'cardiology': Path(r'C:\Users\user\cardiology_rcts'),
    'diabetes': Path(r'C:\Users\user\diabetes_rcts'),
    'respiratory': Path(r'C:\Users\user\respiratory_rcts'),
    'neurology': Path(r'C:\Users\user\neurology_rcts'),
    'infectious': Path(r'C:\Users\user\infectious_rcts'),
    'rheumatology': Path(r'C:\Users\user\rheumatology_rcts'),
}

RESULTS_DIR = Path(__file__).parent / 'phase4_results'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=15)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(Path(__file__).parent / 'selected_phase4_trials.json') as f:
        all_trials = json.load(f)

    start = args.batch * args.batch_size
    end = min(start + args.batch_size, len(all_trials))
    batch = all_trials[start:end]

    if not batch:
        print(f"No trials in batch {args.batch} (start={start}, total={len(all_trials)})")
        sys.exit(0)

    print(f"=== Batch {args.batch}: trials {start}-{end-1} ({len(batch)} trials) ===")

    results = []
    for i, trial in enumerate(batch):
        name = trial['name']
        pdf_dir_key = trial.get('pdf_dir')
        pdf_dir = DIR_MAP.get(pdf_dir_key)
        pdf_path = pdf_dir / trial['pdf'] if pdf_dir else None
        gt_hr = trial['gt_hr']
        gt_lo = trial['gt_ci_lower']
        gt_hi = trial['gt_ci_upper']
        hr_source = trial.get('hr_source')
        area = trial.get('area', trial.get('pdf_dir', 'unknown'))
        endpoint = trial.get('outcome_type')

        idx = start + i
        print(f"\n[{idx}/{len(all_trials)}] {name} ({area})")
        print(f"  GT: HR={gt_hr} [{gt_lo}, {gt_hi}]")

        if pdf_dir is None:
            print(f"  FAILED: unknown pdf_dir '{pdf_dir_key}'")
            results.append({
                'name': name, 'area': area,
                'gt_hr': gt_hr, 'gt_ci': [gt_lo, gt_hi],
                'ext_hr': None, 'within_ci': False,
                'status': 'FAIL_CONFIG',
                'error': f"unknown pdf_dir '{pdf_dir_key}'",
                'time_s': 0.0,
            })
            continue

        if not pdf_path.exists():
            print(f"  FAILED: missing PDF {pdf_path}")
            results.append({
                'name': name, 'area': area,
                'gt_hr': gt_hr, 'gt_ci': [gt_lo, gt_hi],
                'ext_hr': None, 'within_ci': False,
                'status': 'FAIL_CONFIG',
                'error': f"missing PDF: {pdf_path}",
                'time_s': 0.0,
            })
            continue

        t0 = time.time()
        try:
            pipeline = KMPipeline(dpi=300, max_pages=12, n_per_arm=100)
            result = pipeline.extract(str(pdf_path), target_endpoint=endpoint)
            elapsed = time.time() - t0

            if result.hr is not None:
                ext_hr = float(result.hr)
                final_hr, within_ci, orientation = resolve_phase4_hr_orientation(
                    ext_hr, gt_lo, gt_hi, result.hr_method, hr_source
                )

                rel_error = abs(final_hr - gt_hr) / gt_hr * 100 if gt_hr != 0 else 0
                status = "PASS" if within_ci else "FAIL"
                print(f"  Extracted: HR={ext_hr:.3f} ({orientation})")
                print(f"  Error: {rel_error:.1f}% | Within CI: {within_ci} -> {status}")
                print(f"  Method: {result.hr_method} | Time: {elapsed:.0f}s")

                results.append({
                    'name': name, 'area': area,
                    'gt_hr': gt_hr, 'gt_ci': [gt_lo, gt_hi],
                    'ext_hr': ext_hr, 'final_hr': final_hr,
                    'orientation': orientation,
                    'rel_error_pct': round(rel_error, 2),
                    'within_ci': within_ci,
                    'status': status,
                    'hr_method': result.hr_method,
                    'time_s': round(elapsed, 1),
                })
            else:
                elapsed = time.time() - t0
                warnings = result.warnings if result.warnings else ["Unknown"]
                print(f"  FAILED: {warnings[0]}")
                results.append({
                    'name': name, 'area': area,
                    'gt_hr': gt_hr, 'gt_ci': [gt_lo, gt_hi],
                    'ext_hr': None, 'within_ci': False,
                    'status': 'FAIL_EXTRACT',
                    'error': warnings[0],
                    'time_s': round(elapsed, 1),
                })

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  EXCEPTION: {e}")
            results.append({
                    'name': name, 'area': area,
                    'gt_hr': gt_hr, 'ext_hr': None,
                    'gt_ci': [gt_lo, gt_hi],
                    'within_ci': False, 'status': 'EXCEPTION',
                    'error': str(e), 'time_s': round(elapsed, 1),
                })

    # Save batch results
    out_path = RESULTS_DIR / f'batch_{args.batch}.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    n_pass = sum(1 for r in results if r.get('within_ci'))
    n_total = len(results)
    errors = [r['rel_error_pct'] for r in results if r.get('rel_error_pct') is not None]
    mean_err = sum(errors) / len(errors) if errors else 0
    print(f"\n=== Batch {args.batch} Summary ===")
    print(f"  {n_pass}/{n_total} within CI ({100*n_pass/n_total:.0f}%)")
    print(f"  Mean error: {mean_err:.1f}%")
    print(f"  Saved: {out_path}")


if __name__ == '__main__':
    main()
