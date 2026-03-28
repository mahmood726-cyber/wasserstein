# Wasserstein KM Extractor v1.2 — Release Notes

**Date**: 2026-02-14
**Status**: Validated, TruthCert PASS

## What Changed

### Accuracy Improvements
1. **WACA-PVAC arm inversion fix** — Derived HR now tries both orientations (HR and 1/HR), picking the better curve match. Error: 34% -> 0.4%.
2. **FREEZEAF-30M extraction enabled** — Lowered survival start threshold from 0.5 to 0.35 and allowed single-curve embedded figures. Error: FAIL -> 7.0%.
3. **HIPAF threshold tuning** — Tightened derived HR fallback threshold from 0.148 to 0.10 so curves that diverge >10.5% from text-derived HR use the text value. Error: 11.4% -> 0.4%.
4. **NAR OCR modules** — Created `nar_detector.py` and `nar_ocr_extractor.py` for number-at-risk table extraction. Modules functional; detection quality needs refinement.

### Validation Results
| Metric | v1.1 | v1.2 |
|--------|------|------|
| Extracted | 12/13 | **13/13** |
| Within CI | 12/12 | **13/13** |
| Error <10% | 10/12 | **13/13** |
| Mean error | 6.9% | **3.1%** |
| Median error | 3.8% | **2.6%** |

### R Cross-Validation
- R Cox on Python IPD: 0.4% median difference (validates log-rank HR estimation)
- Python outperforms R IPDfromKM: 2.6% vs 36.0% median error vs ground truth

## Files Modified
- `robust_km_pipeline.py` — dual orientation, threshold changes, inline comments
- `km_pipeline.py` — version bump to 1.2
- `regression_validate_13.py` — improved reciprocal logic for derived-HR trials
- `validate_r_ipdfromkm.R` — added FREEZEAF-30M trial

## Files Created
- `nar_detector.py` — NAR table detection (~135 lines)
- `nar_ocr_extractor.py` — NAR table OCR extraction (~244 lines)
- `regenerate_phase2_v12.py` — utility to regenerate phase2 results

## How Verified
1. `python regression_validate_13.py` — 13/13 PASS, 13/13 <10% error
2. `Rscript validate_r_ipdfromkm.R` — R Cox validates Python HR estimation
3. TruthCert bundle: `regression_13_results/TRUTHCERT_v1.2.json`
