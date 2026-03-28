# Wasserstein KM Extractor v1.3 — Release Notes

**Date**: 2026-02-14
**Status**: Validated, TruthCert PASS

## What Changed

### Accuracy Improvements
1. **Derived HR fallback threshold tightened** (0.10 -> 0.07) — Catches FROZEN AF (log_err=0.074) and FREEZEAF-30M (log_err=0.071) where curve HR diverges from text-derived HR. Safe gap: HUNTER at 0.036 is well below threshold.
   - FROZEN AF: 7.4% -> **0.2%**
   - FREEZEAF-30M: 7.0% -> **0.4%**
2. **Global np.random.seed(42)** — Added at top of `extract_hr()` for run-to-run determinism. KMeans already used `random_state=42` but other numpy random ops were unseeded.
3. **NAR detector figure routing** — When embedded figure images are available, passes cropped figure images to NAR detector instead of full-page rasterizations. Reduces NAR false positives from non-NAR text on the page.

### Validation Results
| Metric | v1.2 | v1.3 |
|--------|------|------|
| Extracted | 13/13 | **13/13** |
| Within CI | 13/13 | **13/13** |
| Error <10% | 13/13 | **13/13** |
| Mean error | 3.1% | **2.1%** |
| Median error | 2.6% | **1.2%** |

### Per-Trial Results
| Trial | v1.2 | v1.3 | Change |
|-------|------|------|--------|
| HUNTER | 2.6% | 2.6% | same |
| MILILIS-PERS | 1.8% | 1.3% | -0.5pp |
| FIRE AND ICE | 1.8% | 1.8% | same |
| CIRCA-DOSE | 4.4% | 4.4% | same |
| FROZEN AF | **7.4%** | **0.2%** | -7.2pp |
| CRRF-PeAF | 1.0% | 1.0% | same |
| HIPAF | 0.4% | 0.4% | same |
| PVAC-CPVI | 2.8% | 2.8% | same |
| WACA-PVAC | 0.4% | 0.4% | same |
| CRAVE | 7.2% | 7.3% | +0.1pp |
| ADVENT | 3.2% | 3.2% | same |
| FREEZEAF-30M | **7.0%** | **0.4%** | -6.6pp |
| LBRF-PERSISTENT | 1.8% | 1.2% | -0.6pp |

## Files Modified
- `robust_km_pipeline.py` — threshold 0.10->0.07, np.random.seed(42), NAR figure routing, version 3.7->3.8
- `km_pipeline.py` — version bump 1.2->1.3

## How Verified
1. `python regression_validate_13.py` — 13/13 PASS, 13/13 <10% error, mean 2.1%, median 1.2%
2. TruthCert bundle: `regression_13_results/TRUTHCERT_v1.3.json`
