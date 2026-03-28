# CLAUDE.md — Wasserstein KM Extractor v1.4

## Project Overview
**KM curve digitizer**: PDF → rasterize → HSV color detect → Guyot IPD → log-rank → HR.
Extracts Kaplan-Meier survival curves from published PDF figures and derives hazard ratios.

## File Structure (15 core Python files)
```
km_pipeline.py                 CLI entry point (v1.4)
robust_km_pipeline.py          Full extraction engine v3.8
improved_hr_estimation.py      Log-rank test + HR calculation
simple_multicurve_handler.py   Multi-arm curve separation (HSV + KMeans)
improved_guyot_algorithm.py    Guyot IPD reconstruction
enhanced_ocr_axis.py           Axis calibration (OCR + heuristic)
figure_classifier.py           KM curve detection vs other figure types
enhanced_curve_extractor.py    HSV color-based curve extraction
pdf_extractor.py               PDF rasterization (PyMuPDF)
pdf_figure_extractor.py        PDF figure region extraction
legend_extractor.py            Arm label extraction from legends
nar_detector.py                Number-at-risk table detection (v1.2)
nar_ocr_extractor.py           NAR table OCR extraction (v1.2)
ground_truth_database.py       1,836 validated entries
regression_validate_13.py      Regression validation script
```

## Pipeline Stages
1. **Rasterize**: PDF pages → 300 DPI images (PyMuPDF)
2. **Classify**: detect KM curve figures vs bar charts, flow diagrams
3. **Calibrate**: OCR + heuristic axis detection (time + survival)
4. **Extract curves**: HSV color segmentation → pixel coordinates
5. **Separate arms**: KMeans clustering + fallback cascade
6. **Reconstruct IPD**: Guyot algorithm (time, event, arm)
7. **Estimate HR**: Log-rank test → HR + 95% CI + p-value

## Validation Results (v1.4, 2026-02-15)
- **40 trials, 11 therapeutic areas**: 36/40 (90%) within CI, 30/40 <10% error, median 2.5%
- **13/13 gold AF trials**: 100% within CI, 13/13 <10% error, median 1.2% (unchanged from v1.3)
- **Cardio/Metabolic (12)**: 12/12 within CI, mean 2.8%
- **Phase 2 Diverse (15)**: 11/15 within CI, 4 failures from wrong text HR in review papers
- **Performance**: 15-137s per PDF (median ~58s)
- **F5 threshold**: 0.15 (rejects curve HR diverging >16% from text HR)
- **R cross-validated**: Python pipeline vs R IPDfromKM v0.1.10 (Python outperforms R: 1.2% vs 10.4% median error)

## Critical Warnings
- **Non-inferiority trials** (HR~1.0): do NOT penalize overlapping curves
- **Derived HR arm ordering**: text order may be RECIPROCAL of ground truth
- **Recurrence rates**: must be COMPLEMENTED (1-rate) before HR calculation
- **Citation filter**: skip HRs from cited references ([18], "Author et al.")
- **Event-rate pattern**: only "X% vs Y%", NOT loose "X% and Y%"
- **Text normalization**: NEVER split common words like "freedom"
- **NAR OCR timeout**: daemon thread + 8s per-page limit
- **`threading.Thread`**: must use `daemon=True`
- **Overlap penalty**: 0.01 threshold (non-inferiority curves ARE nearly identical)

## Do NOT
- Apply aggressive cross-page penalties without isolated testing
- Use plausibility tiers that override good curve matches
- Disable NAR without measuring accuracy impact
- Use aggressive text normalization that splits real words
- Use derived HR from event-rate derivation as orientation anchor
- Add `was\s+achieved\s+in` to event-rate verb list (HUNTER false positive)

## Dependencies
- numpy, opencv-python, PyMuPDF, scipy (required)
- scikit-learn (optional: multi-arm color clustering)
- pytesseract + Tesseract OCR (optional: axis label OCR)

## Workflow Rules

### Regression Prevention
Before optimization changes, save accuracy snapshot. After each change, compare. If any trial regresses >2%, rollback immediately. Never apply aggressive heuristics without isolated testing.

### Data Integrity
Never fabricate identifiers (NCT IDs, DOIs, PMIDs). Verify against existing data files before use.

### Fix Completeness
Fix ALL issues in one pass. Re-run validation before declaring done. Track fixes with IDs.

### TruthCert Protocol
All outputs carry `certification_status = UNCERTIFIED`. Certified claims require evidence locators, content hashes, transformation steps, and validator outcomes. Memory references BLOCK certification.
