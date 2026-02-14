"""
km_pipeline.py — Siratal Mustaqeem: The Straight Path
=====================================================

Single entry point for KM curve → IPD reconstruction from RCT PDFs.

Usage:
    python km_pipeline.py input.pdf
    python km_pipeline.py input.pdf --output results/
    python km_pipeline.py input.pdf --output results/ --format csv json
    python km_pipeline.py folder_of_pdfs/ --batch

Pipeline:
    PDF → rasterize → HSV color detect → curve pairing →
    Guyot IPD reconstruction → log-rank → HR + CI

Outputs:
    - IPD data (time, event, arm) as CSV
    - HR summary (HR, CI, p-value, provenance) as JSON
    - Console summary

Built on robust_km_pipeline.py v3.7 (validated on 13 RCTs, 100% within CI).
All outputs carry certification_status = UNCERTIFIED per TruthCert protocol.

Author: Wasserstein KM Extractor
Date: February 2026
"""

import os
import sys
import json
import csv
import argparse
import logging
import re
import time
import unicodedata
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

# Core pipeline
from robust_km_pipeline import RobustKMPipeline, HRExtractionResult
from improved_hr_estimation import (
    reconstruct_ipd_guyot, log_rank_test, IPDRecord, HRResult
)
from simple_multicurve_handler import SimpleMultiCurveHandler

# Optional: PyMuPDF for IPD re-extraction
try:
    import fitz
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('km_pipeline')


def _safe_stem(filename: str) -> str:
    """Sanitize filename stem for safe output on all platforms.

    Replaces Unicode hyphens/dashes/special chars with ASCII equivalents.
    """
    stem = Path(filename).stem
    # Normalize Unicode (NFKD decomposes special chars)
    stem = unicodedata.normalize('NFKD', stem)
    # Replace common Unicode hyphens/dashes with ASCII hyphen
    stem = re.sub(r'[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]', '-', stem)
    # Remove any remaining non-ASCII characters
    stem = stem.encode('ascii', 'replace').decode('ascii')
    return stem


# ---------------------------------------------------------------------------
# Data classes for pipeline output
# ---------------------------------------------------------------------------

@dataclass
class IPDExport:
    """Reconstructed individual patient data for one arm."""
    arm_label: str           # "treatment" or "control"
    arm_index: int           # 0 or 1
    n_patients: int
    n_events: int
    n_censored: int
    records: List[Dict]      # [{time, event, arm}]


@dataclass
class PipelineResult:
    """Complete pipeline output: HR + IPD + provenance."""
    # Source
    pdf_path: str
    pdf_name: str

    # HR result
    hr: Optional[float]
    ci_lower: Optional[float]
    ci_upper: Optional[float]
    p_value: Optional[float]
    hr_method: str

    # IPD data
    ipd_arm1: Optional[IPDExport]
    ipd_arm2: Optional[IPDExport]
    total_ipd_records: int

    # Curve data (for downstream use)
    curve1_times: Optional[List[float]]
    curve1_survivals: Optional[List[float]]
    curve2_times: Optional[List[float]]
    curve2_survivals: Optional[List[float]]

    # Metadata
    confidence: float
    orientation_method: str
    text_hr: Optional[float]
    n_curves_found: int
    n_pages_scanned: int
    processing_time_s: float
    warnings: List[str]
    certification_status: str = "UNCERTIFIED"
    pipeline_version: str = "1.0"
    timestamp: str = ""

    @property
    def succeeded(self) -> bool:
        return self.hr is not None

    @property
    def has_ipd(self) -> bool:
        return self.total_ipd_records > 0

    def to_dict(self) -> Dict:
        d = {}
        for k, v in self.__dict__.items():
            if k == 'pdf_path':
                # Only emit basename, not full absolute path (P1-10)
                d[k] = Path(v).name if v else v
            elif k.startswith('curve') and v is not None:
                # Truncate curve data for JSON readability
                d[k] = v[:10] if len(v) > 10 else v
                d[k + '_length'] = len(v)
            elif isinstance(v, IPDExport):
                d[k] = {
                    'arm_label': v.arm_label,
                    'arm_index': v.arm_index,
                    'n_patients': v.n_patients,
                    'n_events': v.n_events,
                    'n_censored': v.n_censored,
                    'n_records': len(v.records),
                }
            else:
                d[k] = v
        return d


# ---------------------------------------------------------------------------
# Extended pipeline that captures IPD
# ---------------------------------------------------------------------------

class KMPipeline(RobustKMPipeline):
    """Extended pipeline that also reconstructs and exports IPD data.

    Inherits the full HR extraction logic from RobustKMPipeline v3.7,
    then re-runs Guyot reconstruction on the winning pair to capture
    the IPD records (which RobustKMPipeline discards after log-rank).
    """

    def __init__(self, dpi: int = 300, max_pages: int = 12,
                 min_curve_points: int = 10, top_k_pairs: int = 3,
                 n_per_arm: int = 100):
        super().__init__(dpi=dpi, max_pages=max_pages,
                         min_curve_points=min_curve_points,
                         top_k_pairs=top_k_pairs)
        self.n_per_arm = n_per_arm
        self._cached_all_curves = None  # Cache from extract_hr

    def _extract_all_curves(self, pages, handler, **kwargs):
        """Override to cache curves for IPD re-extraction."""
        curves = super()._extract_all_curves(pages, handler, **kwargs)
        self._cached_all_curves = curves
        return curves

    def extract(self, pdf_path: str) -> PipelineResult:
        """Full extraction: HR + IPD from a single PDF.

        Returns PipelineResult with both HR and reconstructed IPD data.
        """
        self._cached_all_curves = None  # Clear stale cache from prior PDFs (P0-2)
        t0 = time.time()
        pdf_path = str(Path(pdf_path).resolve())
        pdf_name = Path(pdf_path).name

        if not os.path.exists(pdf_path):
            return self._empty_result(pdf_path, pdf_name,
                                      error=f"File not found: {pdf_path}")

        # Step 1: Run the full HR extraction pipeline
        logger.info(f"Processing: {pdf_name}")
        hr_result = self.extract_hr(pdf_path)

        if not hr_result.succeeded:
            elapsed = time.time() - t0
            # If curve extraction failed but text HR was found, return that
            text_hr = hr_result.provenance.text_hr_found
            if text_hr is not None:
                logger.info(f"No curve pairs but text HR={text_hr:.3f} found")
                return PipelineResult(
                    pdf_path=pdf_path, pdf_name=pdf_name,
                    hr=round(text_hr, 3),
                    ci_lower=None, ci_upper=None, p_value=None,
                    hr_method="text_derived_only",
                    ipd_arm1=None, ipd_arm2=None, total_ipd_records=0,
                    curve1_times=None, curve1_survivals=None,
                    curve2_times=None, curve2_survivals=None,
                    confidence=0.3,
                    orientation_method="text_only",
                    text_hr=text_hr,
                    n_curves_found=hr_result.provenance.total_curves_extracted,
                    n_pages_scanned=hr_result.provenance.pages_scanned,
                    processing_time_s=round(elapsed, 2),
                    warnings=["Text-derived HR only, no curve pair for IPD"],
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            return self._empty_result(
                pdf_path, pdf_name,
                error=hr_result.error or "HR extraction failed",
                n_pages=hr_result.provenance.pages_scanned,
                n_curves=hr_result.provenance.total_curves_extracted,
                processing_time=elapsed,
                warnings=hr_result.warnings,
            )

        # Step 2: Re-extract the winning curve pair for IPD reconstruction
        # We need to re-rasterize and find the winning pair's curves
        ipd_arm1, ipd_arm2, c1_data, c2_data = self._reconstruct_ipd(
            pdf_path, hr_result
        )

        elapsed = time.time() - t0

        return PipelineResult(
            pdf_path=pdf_path,
            pdf_name=pdf_name,
            hr=hr_result.hr,
            ci_lower=hr_result.ci_lower,
            ci_upper=hr_result.ci_upper,
            p_value=getattr(hr_result, 'p_value', None),
            hr_method=hr_result.provenance.orientation_method,
            ipd_arm1=ipd_arm1,
            ipd_arm2=ipd_arm2,
            total_ipd_records=(
                (len(ipd_arm1.records) if ipd_arm1 else 0) +
                (len(ipd_arm2.records) if ipd_arm2 else 0)
            ),
            curve1_times=c1_data[0] if c1_data else None,
            curve1_survivals=c1_data[1] if c1_data else None,
            curve2_times=c2_data[0] if c2_data else None,
            curve2_survivals=c2_data[1] if c2_data else None,
            confidence=hr_result.confidence,
            orientation_method=hr_result.provenance.orientation_method,
            text_hr=hr_result.provenance.text_hr_found,
            n_curves_found=hr_result.provenance.total_curves_extracted,
            n_pages_scanned=hr_result.provenance.pages_scanned,
            processing_time_s=round(elapsed, 2),
            warnings=hr_result.warnings,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _reconstruct_ipd(self, pdf_path: str, hr_result: HRExtractionResult
                         ) -> Tuple[Optional[IPDExport], Optional[IPDExport],
                                    Optional[Tuple], Optional[Tuple]]:
        """Re-extract the winning pair and reconstruct IPD.

        This re-runs curve extraction to find the pair described in
        hr_result.provenance.pair_description, then runs Guyot IPD
        reconstruction to get the actual patient-level data.
        """
        try:
            # Use cached curves from extract_hr (avoids re-rasterization)
            all_curves = self._cached_all_curves
            if not all_curves or len(all_curves) < 2:
                logger.warning("Could not re-extract curves for IPD")
                return None, None, None, None

            # Find the winning pair by matching the provenance description
            pair_desc = hr_result.provenance.pair_description
            orientation = hr_result.provenance.orientation

            # Re-score and find best pair (same logic as extract_hr)
            top_pairs = self._select_top_pairs(all_curves, 10)
            c1, c2 = None, None

            # Try to match by description
            for curve_a, curve_b, score in top_pairs:
                desc = (f"{curve_a['color_name']}(p{curve_a['page']}) vs "
                        f"{curve_b['color_name']}(p{curve_b['page']})")
                if desc == pair_desc:
                    c1, c2 = curve_a, curve_b
                    break

            # Fallback: use top-scoring pair
            if c1 is None and top_pairs:
                c1, c2, _ = top_pairs[0]

            if c1 is None or c2 is None:
                return None, None, None, None

            # Apply orientation (which curve is treatment vs control)
            if orientation == "inverse":
                c1, c2 = c2, c1

            t1 = np.array(c1['times'])
            s1 = np.array(c1['survivals'])
            t2 = np.array(c2['times'])
            s2 = np.array(c2['survivals'])

            # Extract NAR data if available
            n1, nrt1, nrv1 = self._extract_nar_for_curve(c1, arm_index=0)
            n2, nrt2, nrv2 = self._extract_nar_for_curve(c2, arm_index=1)

            # Reconstruct IPD for each arm
            # Use NAR-derived n when available; fall back to n_per_arm
            ipd1_records = reconstruct_ipd_guyot(
                t1, s1,
                n_risk_times=nrt1, n_risk_values=nrv1,
                total_n=n1 if nrv1 is not None else self.n_per_arm,
            )
            ipd2_records = reconstruct_ipd_guyot(
                t2, s2,
                n_risk_times=nrt2, n_risk_values=nrv2,
                total_n=n2 if nrv2 is not None else self.n_per_arm,
            )

            # Label arms
            for rec in ipd1_records:
                rec.arm = 1  # treatment
            for rec in ipd2_records:
                rec.arm = 0  # control

            # Build exports
            arm1 = IPDExport(
                arm_label="treatment",
                arm_index=1,
                n_patients=len(ipd1_records),
                n_events=sum(1 for r in ipd1_records if r.event == 1),
                n_censored=sum(1 for r in ipd1_records if r.event == 0),
                records=[{'time': round(r.time, 4), 'event': r.event,
                          'arm': r.arm} for r in ipd1_records],
            )
            arm2 = IPDExport(
                arm_label="control",
                arm_index=0,
                n_patients=len(ipd2_records),
                n_events=sum(1 for r in ipd2_records if r.event == 1),
                n_censored=sum(1 for r in ipd2_records if r.event == 0),
                records=[{'time': round(r.time, 4), 'event': r.event,
                          'arm': r.arm} for r in ipd2_records],
            )

            c1_data = (t1.tolist(), s1.tolist())
            c2_data = (t2.tolist(), s2.tolist())

            logger.info(
                f"IPD reconstructed: {arm1.n_patients} treatment "
                f"({arm1.n_events} events), {arm2.n_patients} control "
                f"({arm2.n_events} events)"
            )

            return arm1, arm2, c1_data, c2_data

        except Exception as e:
            logger.error(f"IPD reconstruction failed: {e}")
            return None, None, None, None

    def _empty_result(self, pdf_path, pdf_name, error="", n_pages=0,
                      n_curves=0, processing_time=0.0, warnings=None):
        return PipelineResult(
            pdf_path=pdf_path,
            pdf_name=pdf_name,
            hr=None, ci_lower=None, ci_upper=None, p_value=None,
            hr_method="none",
            ipd_arm1=None, ipd_arm2=None, total_ipd_records=0,
            curve1_times=None, curve1_survivals=None,
            curve2_times=None, curve2_survivals=None,
            confidence=0.0,
            orientation_method="none",
            text_hr=None,
            n_curves_found=n_curves,
            n_pages_scanned=n_pages,
            processing_time_s=processing_time,
            warnings=warnings or [error],
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def write_ipd_csv(result: PipelineResult, output_path: str):
    """Write reconstructed IPD to CSV file."""
    rows = []
    if result.ipd_arm1:
        rows.extend(result.ipd_arm1.records)
    if result.ipd_arm2:
        rows.extend(result.ipd_arm2.records)

    if not rows:
        logger.warning("No IPD records to write")
        return

    # Sort by time, then arm
    rows.sort(key=lambda r: (r['time'], r['arm']))

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['time', 'event', 'arm'])
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"IPD CSV written: {output_path} ({len(rows)} records)")


def write_curves_csv(result: PipelineResult, output_path: str):
    """Write digitized curve data to CSV (for plotting/validation)."""
    rows = []
    if result.curve1_times and result.curve1_survivals:
        for t, s in zip(result.curve1_times, result.curve1_survivals):
            rows.append({'time': round(t, 6), 'survival': round(s, 6),
                         'arm': 'treatment'})
    if result.curve2_times and result.curve2_survivals:
        for t, s in zip(result.curve2_times, result.curve2_survivals):
            rows.append({'time': round(t, 6), 'survival': round(s, 6),
                         'arm': 'control'})

    if not rows:
        return

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['time', 'survival', 'arm'])
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Curves CSV written: {output_path} ({len(rows)} points)")


def write_summary_json(result: PipelineResult, output_path: str):
    """Write full result summary as JSON."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=2, default=str)
    logger.info(f"Summary JSON written: {output_path}")


def print_summary(result: PipelineResult):
    """Print human-readable summary to console."""
    print()
    print("=" * 60)
    print(f"  KM Pipeline Result: {result.pdf_name}")
    print("=" * 60)

    if result.succeeded:
        ci_str = (f"(95% CI: {result.ci_lower:.4f} - {result.ci_upper:.4f})"
                  if result.ci_lower is not None else "(CI: not available)")
        print(f"  HR:           {result.hr:.4f} {ci_str}")
        print(f"  Method:       {result.orientation_method}")
        if result.text_hr:
            print(f"  Text HR:      {result.text_hr}")
        print(f"  Confidence:   {result.confidence:.2f}")
        print(f"  IPD Records:  {result.total_ipd_records}")
        if result.ipd_arm1:
            print(f"    Treatment:  {result.ipd_arm1.n_patients} pts "
                  f"({result.ipd_arm1.n_events} events, "
                  f"{result.ipd_arm1.n_censored} censored)")
        if result.ipd_arm2:
            print(f"    Control:    {result.ipd_arm2.n_patients} pts "
                  f"({result.ipd_arm2.n_events} events, "
                  f"{result.ipd_arm2.n_censored} censored)")
        print(f"  Curves Found: {result.n_curves_found}")
        print(f"  Pages:        {result.n_pages_scanned}")
        print(f"  Time:         {result.processing_time_s:.1f}s")
        print(f"  Status:       {result.certification_status}")
    else:
        print(f"  FAILED: {result.warnings}")

    if result.warnings:
        print(f"  Warnings:     {'; '.join(result.warnings)}")
    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_batch(input_dir: str, output_dir: str, **kwargs) -> List[Dict]:
    """Process all PDFs in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(input_path.glob('*.pdf'))
    if not pdfs:
        logger.error(f"No PDFs found in {input_dir}")
        return []

    logger.info(f"Batch processing: {len(pdfs)} PDFs")
    pipeline = KMPipeline(**kwargs)
    results = []

    for i, pdf in enumerate(pdfs, 1):
        logger.info(f"[{i}/{len(pdfs)}] {pdf.name}")
        try:
            result = pipeline.extract(str(pdf))

            # Write outputs for each PDF
            stem = _safe_stem(pdf.name)
            if result.has_ipd:
                write_ipd_csv(result, str(output_path / f"{stem}_ipd.csv"))
                write_curves_csv(result, str(output_path / f"{stem}_curves.csv"))

            write_summary_json(result, str(output_path / f"{stem}_summary.json"))
            print_summary(result)

            results.append({
                'pdf': pdf.name,
                'hr': result.hr,
                'ci_lower': result.ci_lower,
                'ci_upper': result.ci_upper,
                'ipd_records': result.total_ipd_records,
                'succeeded': result.succeeded,
                'time_s': result.processing_time_s,
                'error': result.warnings[0] if not result.succeeded and result.warnings else None,
            })
        except Exception as e:
            logger.error(f"  Error: {e}")
            results.append({
                'pdf': pdf.name, 'hr': None, 'succeeded': False,
                'error': str(e),
            })

    # Write batch summary
    summary_path = output_path / "batch_summary.json"
    succeeded = sum(1 for r in results if r.get('succeeded'))
    batch_summary = {
        'total': len(results),
        'succeeded': succeeded,
        'failed': len(results) - succeeded,
        'success_rate': f"{succeeded / len(results) * 100:.1f}%",
        'results': results,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(batch_summary, f, indent=2, default=str)

    print(f"\nBatch complete: {succeeded}/{len(results)} succeeded "
          f"({succeeded / len(results) * 100:.1f}%)")
    print(f"Results in: {output_path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='KM Pipeline: Extract IPD from Kaplan-Meier curves in RCT PDFs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python km_pipeline.py paper.pdf
  python km_pipeline.py paper.pdf --output results/
  python km_pipeline.py paper.pdf --output results/ --format csv json
  python km_pipeline.py pdfs/ --batch --output results/
  python km_pipeline.py paper.pdf --dpi 200 --max-pages 6
        """,
    )
    parser.add_argument('input', help='PDF file or directory (with --batch)')
    parser.add_argument('--output', '-o', default='.',
                        help='Output directory (default: current dir)')
    parser.add_argument('--format', '-f', nargs='+',
                        choices=['csv', 'json', 'all'],
                        default=['all'],
                        help='Output format(s): csv, json, all (default: all)')
    parser.add_argument('--batch', '-b', action='store_true',
                        help='Process all PDFs in input directory')
    parser.add_argument('--dpi', type=int, default=300,
                        help='Rasterization DPI (default: 300)')
    parser.add_argument('--max-pages', type=int, default=12,
                        help='Max pages to scan (default: 12)')
    parser.add_argument('--n-per-arm', type=int, default=100,
                        help='Assumed patients per arm for IPD (default: 100)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress console output')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    pipeline_kwargs = {
        'dpi': args.dpi,
        'max_pages': args.max_pages,
        'n_per_arm': args.n_per_arm,
    }

    formats = set(args.format)
    if 'all' in formats:
        formats = {'csv', 'json'}

    # Batch mode
    if args.batch:
        if not os.path.isdir(args.input):
            print(f"Error: {args.input} is not a directory", file=sys.stderr)
            sys.exit(1)
        process_batch(args.input, args.output, **pipeline_kwargs)
        return

    # Single file mode
    if not os.path.isfile(args.input):
        print(f"Error: {args.input} is not a file", file=sys.stderr)
        sys.exit(1)

    pipeline = KMPipeline(**pipeline_kwargs)
    result = pipeline.extract(args.input)

    # Print summary
    if not args.quiet:
        print_summary(result)

    # Write outputs
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(args.input)

    if result.succeeded:
        if 'csv' in formats and result.has_ipd:
            write_ipd_csv(result, str(output_dir / f"{stem}_ipd.csv"))
            write_curves_csv(result, str(output_dir / f"{stem}_curves.csv"))
        if 'json' in formats:
            write_summary_json(result, str(output_dir / f"{stem}_summary.json"))
    else:
        # Always write summary even on failure (for diagnostics)
        if 'json' in formats:
            write_summary_json(result, str(output_dir / f"{stem}_summary.json"))
        sys.exit(1)


if __name__ == '__main__':
    import io
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding='utf-8', errors='replace')
    main()
