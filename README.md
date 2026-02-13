# Wasserstein KM Extractor

Extract individual patient data (IPD) and hazard ratios from Kaplan-Meier survival curves in published PDF articles.

## What it does

Given an RCT PDF with KM curves, the pipeline:

1. **Rasterizes** PDF pages to images
2. **Detects** KM curve figures (vs bar charts, flow diagrams, etc.)
3. **Calibrates** time and survival axes via OCR
4. **Extracts** curve coordinates using HSV color detection
5. **Separates** overlapping arms in multi-curve figures
6. **Reconstructs IPD** using the Guyot algorithm (time, event, arm)
7. **Estimates HR** via log-rank test with 95% CI

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+. Optional: install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) for improved axis label reading.

## Usage

**Single PDF:**
```bash
python km_pipeline.py paper.pdf --output results/
```

**Batch (all PDFs in a directory):**
```bash
python km_pipeline.py pdfs/ --batch --output results/
```

**Options:**
```
--output, -o DIR       Output directory (default: current)
--format, -f FORMAT    csv, json, or all (default: all)
--batch, -b            Process all PDFs in input directory
--dpi DPI              Rasterization DPI (default: 300)
--max-pages N          Max pages to scan (default: 12)
--n-per-arm N          Assumed patients per arm (default: 100)
--quiet, -q            Suppress console output
--verbose, -v          Debug logging
```

## Output

For each PDF, the pipeline produces:

- **`{name}_ipd.csv`** — Reconstructed IPD: `time, event, arm` per patient
- **`{name}_curves.csv`** — Digitized KM curve coordinates
- **`{name}_summary.json`** — HR, 95% CI, p-value, provenance, warnings

Batch mode additionally produces **`batch_summary.json`** with per-PDF results.

## Validation

Validated on 13 gold-tier RCTs (cardiac ablation trials):

| Metric | Result |
|--------|--------|
| Extraction success | 13/13 (100%) |
| HR within reported CI | 13/13 (100%) |
| Relative error <10% | 12/13 (92%) |
| Median error | 1.7% |
| Mean error | 5.4% |

Batch validation on 27 real PDFs: 26/27 success (96.3%).

## Architecture

```
km_pipeline.py                 CLI entry point
robust_km_pipeline.py          Full extraction engine (v3.8)
improved_hr_estimation.py      Log-rank test + HR calculation
simple_multicurve_handler.py   Multi-arm curve separation
improved_guyot_algorithm.py    Guyot IPD reconstruction
enhanced_ocr_axis.py           Axis calibration (OCR)
figure_classifier.py           KM curve detection
enhanced_curve_extractor.py    HSV color-based curve extraction
pdf_extractor.py               PDF rasterization
pdf_figure_extractor.py        PDF figure extraction
legend_extractor.py            Arm label extraction
ground_truth_database.py       Validation reference (1,836 entries)
```

## Limitations

- Designed for two-arm KM curves; three+ arms may not separate cleanly
- Derived HRs (from event rates) lack confidence intervals
- Very low-resolution or heavily compressed PDF figures may fail
- Numbers-at-risk extraction requires Tesseract OCR
- All outputs carry `certification_status = UNCERTIFIED` per TruthCert protocol

## License

Research use. See CLAUDE.md for project governance.
