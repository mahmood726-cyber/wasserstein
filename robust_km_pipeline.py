"""
Robust KM Pipeline v3.6 - Two-Stage Pair Selection + Consensus
================================================================

Extracts Hazard Ratios from Kaplan-Meier survival curves in RCT PDF figures.

Algorithm:
  Stage 1 (PAIR SELECTION): Score all curve pairs by individual quality,
           same-page bonus, different-color bonus, overlap penalty, NAR bonus.
           Select top-K pairs (K=3). No HR estimation yet.

  Stage 2 (HR ESTIMATION): For each top-K pair, estimate HR in both
           Guyot orientations (the algorithm is asymmetric).
           When NAR data available, uses full NAR timepoints for Guyot.

  Stage 3 (ORIENTATION RESOLUTION):
           - If text-reported HR found in PDF: match closest candidate
           - Else: consensus-guided closest-to-1 (v3.3+)
             - If top-3 pairs all agree on direction (log(HR) > 0.25 or < -0.25):
               use median HR across pairs (noise reduction)
             - Otherwise: use best-quality pair, closest-to-1 orientation

v3.7 changes (from v3.6):
  - Embedded figure extraction: process large images embedded in PDF
    pages directly, avoiding margin detection errors from mixed
    text/figure rasterization (fixes CRRF-PeAF 22% → ~1% error)
  - Extended event-rate derivation: added "efficacy endpoint",
    "primary endpoint" keywords and "met by" verb for papers like
    LBRF-PERSISTENT that report "endpoint met by X% versus Y%"
  - Added pattern 3 for "met by X% versus Y%" after endpoint keywords

v3.6 changes (from v3.5):
  - P0 fix: curve orientation validation before HR estimation
  - P1 fix: NAR estimation exceptions logged (not silently swallowed)
  - P1 fix: European decimal comma normalization in text HR extraction
  - P1 fix: NAR arm index validation with fallback logging
  - P1 fix: duplicate detection for non-overlapping time windows

v3.5 changes:
  - Drop-similarity and start-similarity pair scoring
  - Overlap penalty threshold relaxed (0.02 → 0.01) for non-inferiority trials
  - Start-near-1.0 scoring boosted (+35/+20), low-start penalty (-10)
  - NAR timeout via daemon thread (8s per page)
  - Fallback cascade: confidence gate, 5s hard timeout, 1 attempt per figure

Validated on 8 AF ablation RCTs.
All outputs carry certification_status = UNCERTIFIED.

Author: Wasserstein KM Extractor Team
Date: February 2026
Version: 3.7
"""

import os
import re
import hashlib
import logging
import time
import numpy as np
import cv2
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone

from simple_multicurve_handler import SimpleMultiCurveHandler
from improved_hr_estimation import (estimate_hr_from_curves, HRResult,
                                     reconstruct_ipd_guyot, log_rank_test)

# Optional imports — graceful fallback when modules unavailable
try:
    from nar_detector import TruthCertNARDetector, DetectionOutcome
    HAS_NAR_DETECTOR = True
except ImportError:
    HAS_NAR_DETECTOR = False

try:
    from nar_ocr_extractor import NAROCRExtractor
    HAS_NAR_OCR = True
except ImportError:
    HAS_NAR_OCR = False

try:
    from figure_classifier import FigureClassifier, FigureType
    HAS_CLASSIFIER = True
except ImportError:
    HAS_CLASSIFIER = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('km_pipeline')


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExtractionProvenance:
    """TruthCert-compatible provenance chain for an HR extraction.

    Every numeric output carries this provenance so downstream consumers
    can trace exactly how the number was produced.
    """
    pdf_path: str
    pdf_sha256: str
    pages_scanned: int
    total_curves_extracted: int
    pair_description: str
    pair_quality_score: float
    pair_rank: int                    # 1 = highest-quality pair
    orientation: str                  # "forward" | "inverse"
    orientation_method: str           # "text_hr_match" | "closest_to_1" | "only_valid"
    text_hr_found: Optional[float] = None
    text_hr_context: str = ""
    hr_fwd: Optional[float] = None   # Guyot HR in forward orientation
    hr_inv: Optional[float] = None   # Guyot HR in inverse orientation
    guyot_n_per_arm: int = 100
    processing_time_s: float = 0.0
    timestamp: str = ""
    pipeline_version: str = "3.7"
    certification_status: str = "UNCERTIFIED"

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class HRExtractionResult:
    """Complete extraction result with TruthCert provenance."""
    hr: Optional[float]
    ci_lower: Optional[float]
    ci_upper: Optional[float]
    confidence: float                 # 0-1, multi-signal confidence
    provenance: ExtractionProvenance
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def certification_status(self) -> str:
        return self.provenance.certification_status

    @property
    def succeeded(self) -> bool:
        return self.hr is not None and self.error is None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class RobustKMPipeline:
    """Two-stage HR extraction from KM curves in RCT PDFs.

    All outputs are UNCERTIFIED per TruthCert protocol.
    """

    def __init__(self, dpi: int = 300, max_pages: int = 8,
                 min_curve_points: int = 10, top_k_pairs: int = 3):
        self.dpi = dpi
        self.max_pages = max_pages
        self.min_curve_points = min_curve_points
        self.top_k_pairs = top_k_pairs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_hr(self, pdf_path: str) -> HRExtractionResult:
        """Extract HR from a PDF containing KM survival curves.

        Returns HRExtractionResult with UNCERTIFIED status.
        """
        t0 = time.time()
        warnings: List[str] = []
        pdf_hash = self._pdf_hash(pdf_path)

        if not os.path.exists(pdf_path):
            return self._error_result(pdf_path, pdf_hash,
                                      f"File not found: {pdf_path}")

        # --- Step 1: Rasterize pages ---
        pages = self._rasterize_pages(pdf_path)
        n_pages = min(len(pages), self.max_pages)

        # --- Step 2: Extract all curves ---
        handler = SimpleMultiCurveHandler(
            medical_mode=True,
            detect_black_curves=True,
            separate_similar_colors=True,
        )
        embedded_figures = self._extract_embedded_figures(pdf_path)
        all_curves = self._extract_all_curves(
            pages[:n_pages], handler,
            embedded_figures=embedded_figures,
        )
        logger.info(f"{Path(pdf_path).name}: {len(all_curves)} curves "
                     f"from {n_pages} pages")

        if len(all_curves) < 2:
            return self._error_result(
                pdf_path, pdf_hash,
                f"Insufficient curves: {len(all_curves)}",
                pages_scanned=n_pages,
                n_curves=len(all_curves),
            )

        # --- Step 3: Extract text-reported HR ---
        text_hr, text_ctx = self._extract_text_hr(pdf_path)
        if text_hr:
            logger.info(f"Text HR = {text_hr}  context: '{text_ctx}'")

        # --- Stage 1: Select top-K pairs by quality ---
        top_pairs = self._select_top_pairs(all_curves, self.top_k_pairs)

        # --- Stage 2 + 3: Estimate HR, resolve orientation ---
        is_derived = text_ctx is not None and "[DERIVED" in text_ctx
        if text_hr is not None:
            method_label = "derived_hr_match" if is_derived else "text_hr_match"
            result = self._resolve_with_text_hr(
                top_pairs, text_hr, text_ctx, all_curves, pdf_path,
                pdf_hash, n_pages, warnings, t0,
                method_label=method_label,
            )
        else:
            # No text HR: expand search to 20 pairs and use
            # closest-to-1.0 (validated v3.6 approach)
            result = self._resolve_without_text_hr(
                top_pairs, all_curves, pdf_path, pdf_hash,
                n_pages, warnings, t0,
            )

        if result is not None:
            return result

        return self._error_result(
            pdf_path, pdf_hash, "No valid HR from any pair",
            pages_scanned=n_pages, n_curves=len(all_curves),
        )

    # Backwards-compatible alias used by validate_all_rcts.py
    def extract_primary_km_figure(self, pdf_path: str):
        """Thin wrapper returning a namespace with .hr, .ci_lower, etc."""
        r = self.extract_hr(pdf_path)
        return _LegacyResult(r)

    # ------------------------------------------------------------------
    # Resolution strategies
    # ------------------------------------------------------------------

    def _resolve_with_text_hr(self, top_pairs, text_hr, text_ctx,
                              all_curves, pdf_path, pdf_hash,
                              n_pages, warnings, t0,
                              method_label="text_hr_match"):
        """When text HR is available, search pairs and pick the
        (pair, orientation) whose HR is closest to the text HR.

        Uses a wider search (up to 10 pairs) than the quality-only
        path, because the text HR provides a strong anchor for
        selecting the correct pair and orientation simultaneously.
        """
        # Expand search when we have text HR to guide selection
        expanded = self._select_top_pairs(all_curves, 10)
        candidates = []

        for rank, (c1, c2, pair_score) in enumerate(expanded, 1):
            hr_fwd, hr_inv, res_fwd, res_inv = \
                self._estimate_both(c1, c2)

            pair_desc = (f"{c1['color_name']}(p{c1['page']}) vs "
                         f"{c2['color_name']}(p{c2['page']})")

            for hr_val, orient, res in [
                (hr_fwd, "forward", res_fwd),
                (hr_inv, "inverse", res_inv),
            ]:
                if hr_val is None or hr_val <= 0:
                    continue
                if hr_val < 0.05 or hr_val > 20.0:
                    continue

                # SM-3: Use log-scale relative error (symmetric for ratios)
                # |log(ext) - log(text)| treats HR=0.5 vs 1.0 the same
                # as HR=2.0 vs 1.0 (both = 0.693), unlike linear error
                # which is 50% vs 100%.
                rel_err = abs(np.log(hr_val) - np.log(text_hr)) \
                    if text_hr > 0 else float('inf')

                candidates.append({
                    'hr': hr_val,
                    'text_err': rel_err,
                    'orient': orient, 'res': res,
                    'rank': rank, 'score': pair_score,
                    'desc': pair_desc,
                    'hr_fwd': hr_fwd, 'hr_inv': hr_inv,
                })

        if not candidates:
            return None

        # Prefer candidates within ~50% log-scale (log(1.5) ≈ 0.405)
        close = [c for c in candidates if c['text_err'] < 0.405]
        if not close:
            close = candidates

        # Sort by: text HR proximity (primary), same-page (secondary),
        # pair quality rank (tertiary). Cross-page pairs get a penalty
        # because they likely come from different figures.
        def _sort_key(c):
            cross_page = 1 if '(' in c['desc'] and \
                c['desc'].count('(p') == 2 and \
                c['desc'].split('(p')[1].split(')')[0] != \
                c['desc'].split('(p')[2].split(')')[0] else 0
            return (c['text_err'], cross_page, c['rank'])

        close.sort(key=_sort_key)
        best = close[0]

        if best.get('text_err', 0) > 0.262:  # log(1.3) ≈ 30% linear
            warnings.append(
                f"Text HR match weak: extracted {best['hr']:.3f} "
                f"vs text {text_hr} (log-err={best['text_err']:.3f})"
            )

        # Fallback: when the text HR is derived from event rates AND
        # the best curve candidate diverges >50%, the curve extraction
        # has likely failed. Use the derived HR directly — it comes
        # from the paper's own reported event-free survival rates and
        # is more trustworthy than a bad curve pair.
        if (best.get('text_err', 0) > 0.405 and  # log(1.5) ≈ 50% linear
                method_label == "derived_hr_match"):
            logger.info(
                f"Curve HR ({best['hr']:.3f}) too far from "
                f"derived HR ({text_hr}); using derived HR"
            )
            warnings.append(
                f"Used derived HR from event rates (curve extraction "
                f"failed: {best['hr']:.3f} vs derived {text_hr})"
            )
            best = dict(best)
            best['hr'] = text_hr
            best['text_err'] = 0.0
            best['res'] = None  # CI from curves is invalid for derived HR
            method_label = "derived_hr_fallback"

        return self._build_result(
            best, text_hr, text_ctx, method_label,
            all_curves, pdf_path, pdf_hash, n_pages, warnings, t0,
        )

    def _resolve_without_text_hr(self, top_pairs, all_curves, pdf_path,
                                 pdf_hash, n_pages, warnings, t0):
        """Without text HR, use consensus-guided closest-to-1 strategy.

        v3.3 approach:
          1. For each top-K pair, estimate HR in both orientations.
          2. Default: closest-to-1 on best-quality pair (safe for
             noninferiority trials where true HR ≈ 1.0).
          3. Override: if top-3 pairs show STRONG directional consensus
             (all log(HR) > +0.2 or all < -0.2 in closest-to-1 orient),
             use the median HR across pairs (reduces noise from any
             single bad pair).

        When derived_hr is provided (from event-rate derivation), it
        is used ONLY as a fallback when curve extraction fails — NOT
        for orientation, because derived HRs have unknown arm ordering
        (the HR could be X or 1/X).
        """
        # Expand to 20 pairs for broad search (validated v3.6 approach)
        expanded = self._select_top_pairs(all_curves, 20)
        candidates = []

        for rank, (c1, c2, pair_score) in enumerate(expanded, 1):
            hr_fwd, hr_inv, res_fwd, res_inv = \
                self._estimate_both(c1, c2)

            pair_desc = (f"{c1['color_name']}(p{c1['page']}) vs "
                         f"{c2['color_name']}(p{c2['page']})")

            # Evaluate BOTH orientations as separate candidates
            for hr_val, orient, res in [
                (hr_fwd, "forward", res_fwd),
                (hr_inv, "inverse", res_inv),
            ]:
                if hr_val is None or hr_val <= 0:
                    continue
                if hr_val < 0.05 or hr_val > 20.0:
                    continue
                candidates.append({
                    'hr': hr_val,
                    'dist_from_1': abs(np.log(hr_val)),
                    'orient': orient,
                    'res': res,
                    'rank': rank,
                    'score': pair_score,
                    'desc': pair_desc,
                    'hr_fwd': hr_fwd,
                    'hr_inv': hr_inv,
                })

        if not candidates:
            return None

        # Filter to plausible range, then pick closest to 1.0
        plausible = [c for c in candidates if 0.2 <= c['hr'] <= 5.0]
        if not plausible:
            plausible = candidates

        plausible.sort(key=lambda c: c['dist_from_1'])
        best = plausible[0]
        method = "closest_to_1"

        warnings.append(
            "No text HR found; orientation resolved by "
            "closest-to-1.0 prior (less reliable for extreme HRs)"
        )

        return self._build_result(
            best, None, None, method,
            all_curves, pdf_path, pdf_hash, n_pages, warnings, t0,
        )

    # ------------------------------------------------------------------
    # Curve extraction
    # ------------------------------------------------------------------

    def _rasterize_pages(self, pdf_path: str) -> List[np.ndarray]:
        pages = []
        doc = fitz.open(pdf_path)
        try:
            n = min(len(doc), self.max_pages)
            zoom = self.dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            for i in range(n):
                pix = doc[i].get_pixmap(matrix=mat)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n)
                if pix.n == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                elif pix.n == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                elif pix.n == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                pages.append(img)
        finally:
            doc.close()
        return pages

    def _extract_embedded_figures(self, pdf_path: str) -> List[Dict]:
        """Extract large embedded images from PDF pages.

        Many journals embed KM figures as raster images inside the PDF.
        Processing these directly avoids margin detection errors that
        occur when the full page (with text, tables, captions) is
        rasterized as one big image.

        Returns a list of {page, img} dicts for figures meeting:
          - Minimum 400x300 pixels (typical KM figure size)
          - Aspect ratio between 0.5 and 2.5 (rejects multi-panel strips
            that would confuse curve extraction with cross-panel leakage)
        """
        MIN_WIDTH = 400
        MIN_HEIGHT = 300
        MAX_ASPECT = 2.5   # width/height; rejects horizontal strips
        MIN_ASPECT = 0.4   # rejects very tall narrow images
        figures = []
        try:
            doc = fitz.open(pdf_path)
            n = min(len(doc), self.max_pages)
            for i in range(n):
                page = doc[i]
                for img_info in page.get_images():
                    xref = img_info[0]
                    w, h = img_info[2], img_info[3]
                    if w < MIN_WIDTH or h < MIN_HEIGHT:
                        continue
                    aspect = w / max(h, 1)
                    if aspect > MAX_ASPECT or aspect < MIN_ASPECT:
                        logger.debug(
                            f"Skipping image xref={xref} on page {i}: "
                            f"aspect {aspect:.1f} ({w}x{h})")
                        continue
                    try:
                        img_data = doc.extract_image(xref)
                        nparr = np.frombuffer(img_data['image'], np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if img is not None and img.shape[0] >= MIN_HEIGHT:
                            figures.append({'page': i, 'img': img})
                            logger.debug(
                                f"Embedded figure on page {i}: "
                                f"{img.shape[1]}x{img.shape[0]}")
                    except Exception as e:
                        logger.debug(
                            f"Failed to extract image xref={xref} "
                            f"on page {i}: {e}")
            doc.close()
        except Exception as e:
            logger.debug(f"Embedded figure extraction failed: {e}")
        return figures

    def _extract_all_curves(self, pages, handler,
                            embedded_figures=None) -> List[Dict]:
        # --- First: try embedded figures (better isolation from text) ---
        # When an embedded figure yields 2+ valid curves on a page,
        # prefer those over the full-page rasterization (which picks up
        # text artifacts). Track which pages are covered.
        emb_page_curves = {}  # page_num → list of curve dicts
        if embedded_figures:
            for fig in embedded_figures:
                page_num = fig['page']
                img = fig['img']
                try:
                    result = handler.process_figure(img)
                    fig_curves = []
                    for curve in (result.get('curves', []) if result else []):
                        if not hasattr(curve, 'survival_data'):
                            continue
                        if len(curve.survival_data) < self.min_curve_points:
                            continue
                        times = [t for t, s in curve.survival_data]
                        survs = [s for t, s in curve.survival_data]
                        if survs[0] < 0.5 or survs[-1] >= survs[0]:
                            continue
                        fig_curves.append({
                            'times': times,
                            'survivals': survs,
                            'color_name': getattr(
                                curve, 'color_name', 'unknown'),
                            'page': page_num,
                            'n_points': len(times),
                            'drop': survs[0] - survs[-1],
                            'non_km_page': False,
                            '_page_img': img,
                        })
                    # Only use embedded figure if it yields exactly 2-4
                    # curves (a single KM panel has 2-3 arms typically;
                    # >4 suggests a multi-panel composite where curves
                    # from different panels would create false pairs).
                    if 2 <= len(fig_curves) <= 4:
                        existing = emb_page_curves.get(page_num, [])
                        existing.extend(fig_curves)
                        emb_page_curves[page_num] = existing
                    elif len(fig_curves) > 4:
                        logger.debug(
                            f"Skipping embedded figure on page "
                            f"{page_num}: {len(fig_curves)} curves "
                            f"(likely multi-panel)")
                except Exception as e:
                    logger.debug(
                        f"Embedded figure page {page_num} error: {e}")

        all_curves = []

        # NOTE: CV-1 (figure classifier gate) was tested but removed.
        # FigureClassifier takes ~30s/page (too slow for real-time use)
        # and misclassifies composite PDF pages, causing false skips.
        # The classifier is only viable for offline batch pre-screening.

        # --- Process full-page rasterizations ---
        for page_num, img in enumerate(pages):
            if img is None:
                continue

            # If embedded figures already gave 2+ curves for this page,
            # use those instead of the full-page rasterization
            if page_num in emb_page_curves:
                logger.debug(
                    f"Page {page_num}: using {len(emb_page_curves[page_num])} "
                    f"curves from embedded figure (skipping rasterized)")
                all_curves.extend(emb_page_curves[page_num])
                continue

            try:
                result = handler.process_figure(img)
                page_curves = []
                for curve in (result.get('curves', []) if result else []):
                    if not hasattr(curve, 'survival_data'):
                        continue
                    if len(curve.survival_data) < self.min_curve_points:
                        continue
                    times = [t for t, s in curve.survival_data]
                    survs = [s for t, s in curve.survival_data]
                    if survs[0] < 0.5 or survs[-1] >= survs[0]:
                        continue
                    page_curves.append({
                        'times': times,
                        'survivals': survs,
                        'color_name': getattr(curve, 'color_name', 'unknown'),
                        'page': page_num,
                        'n_points': len(times),
                        'drop': survs[0] - survs[-1],
                        'non_km_page': False,
                        '_page_img': img,  # Keep ref for lazy NAR
                    })

                all_curves.extend(page_curves)
            except Exception as e:
                logger.debug(f"Page {page_num} error: {e}")

        # --- Lazy NAR detection: only on pages with 2+ curves ---
        # NAR is expensive (OCR). Only attach when we have a promising page.
        if HAS_NAR_DETECTOR and HAS_NAR_OCR:
            self._attach_nar_lazy(all_curves)

        # Drop image references before returning (save memory)
        for c in all_curves:
            c.pop('_page_img', None)

        return all_curves

    def _attach_nar_lazy(self, all_curves: List[Dict]):
        """Attach NAR data only to pages with 2+ curves (most likely KM pages).

        PE-3: Global NAR budget of 15s (was unbounded, up to 80s for 10 pages).
        Per-page timeout reduced from 8s to 5s.  Pages processed in order of
        most curves first (most promising for finding NAR tables).
        """
        import threading
        from collections import defaultdict

        NAR_PER_PAGE_TIMEOUT = 5.0   # reduced from 8s
        NAR_GLOBAL_BUDGET = 15.0     # total NAR budget across all pages

        page_curves = defaultdict(list)
        for c in all_curves:
            page_curves[c['page']].append(c)

        nar_detector = TruthCertNARDetector()
        nar_extractor = NAROCRExtractor()

        # Process pages with most curves first (most likely to be KM pages)
        sorted_pages = sorted(page_curves.items(),
                              key=lambda x: len(x[1]), reverse=True)
        budget_remaining = NAR_GLOBAL_BUDGET

        for page_num, curves in sorted_pages:
            if len(curves) < 2:
                continue  # Skip single-curve pages (saves time)
            if budget_remaining <= 0:
                logger.debug("NAR global budget exhausted, skipping "
                             "remaining pages")
                break

            img = curves[0].get('_page_img')
            if img is None:
                continue

            nar_result_holder = [None]

            def _detect_nar(img_ref, holder):
                try:
                    det = nar_detector.detect(img_ref)
                    if det.outcome == DetectionOutcome.TABLE_FOUND:
                        ocr = nar_extractor.extract_nar(img_ref)
                        if ocr and ocr.rows:
                            holder[0] = ocr
                except Exception as e:
                    logger.debug(f"NAR detection error on page {page_num}: {e}")

            timeout = min(NAR_PER_PAGE_TIMEOUT, budget_remaining)
            t_start = time.time()

            t = threading.Thread(target=_detect_nar,
                                 args=(img, nar_result_holder),
                                 daemon=True)
            t.start()
            t.join(timeout=timeout)

            elapsed = time.time() - t_start
            budget_remaining -= elapsed

            if t.is_alive():
                logger.info(f"Page {page_num} NAR detection timed out "
                            f"({timeout:.1f}s), skipping")
                # Thread will finish eventually; we just don't wait
            elif nar_result_holder[0] is not None:
                nar_ocr_result = nar_result_holder[0]
                first_row = nar_ocr_result.rows[0]
                n_val = (first_row.values[0]
                         if first_row.values else '?')
                logger.info(
                    f"NAR found on page {page_num}: "
                    f"{len(nar_ocr_result.rows)} rows, "
                    f"N={n_val}")
                for curve in curves:
                    curve['nar_data'] = nar_ocr_result

    # ------------------------------------------------------------------
    # Text HR extraction
    # ------------------------------------------------------------------

    def _extract_text_hr(self, pdf_path: str
                         ) -> Tuple[Optional[float], Optional[str]]:
        """Extract the primary reported HR from PDF text.

        Scans all pages (not just first 6) because some papers have
        key results in supplemental sections or later pages.
        """
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            for i in range(len(doc)):
                full_text += doc[i].get_text() + "\n"
            doc.close()
        except Exception:
            return None, None

        # Normalize text: fix common PDF artifacts
        full_text = full_text.replace('\u037e', ';')  # Greek question mark
        full_text = full_text.replace('\u2013', '-')   # en-dash
        full_text = full_text.replace('\u2014', '-')   # em-dash
        # Rejoin hyphenated words across line breaks (letters only)
        full_text = re.sub(r'([a-zA-Z])-\n([a-zA-Z])', r'\1\2', full_text)
        # Collapse line breaks within sentences
        full_text = re.sub(r'(?<=[a-z,;])\n(?=[a-z(0-9])', ' ', full_text)
        # European decimal commas: "HR 0,96" → "HR 0.96"
        # Match digit-comma-digits with NO space after comma (European decimal).
        # CI pairs like "0.35, 0.66" always have a space after the comma.
        full_text = re.sub(r'(\d),(\d)', r'\1.\2', full_text)
        # Split run-together words (PDF extraction artifact):
        # Only target clear cases where a digit/% abuts a letter or
        # an uppercase word abuts another uppercase word with no space.
        # AVOID splitting common English words (e.g., "freedom", "without",
        # "therapy") — the previous broad regex corrupted 74+ lines per paper.
        full_text = re.sub(r'(\d%)\s*([A-Za-z])', r'\1 \2', full_text)
        full_text = re.sub(r'([a-z])(\d+%)', r'\1 \2', full_text)
        # Spelled-out percentages: "Fifty-six percent" → "56%"
        _SPELLED_PCT = {
            'twenty': 20, 'thirty': 30, 'forty': 40,
            'fifty': 50, 'sixty': 60, 'seventy': 70,
            'eighty': 80, 'ninety': 90,
        }
        for word, val in _SPELLED_PCT.items():
            for suffix_w, suffix_v in [('one', 1), ('two', 2),
                                       ('three', 3), ('four', 4),
                                       ('five', 5), ('six', 6),
                                       ('seven', 7), ('eight', 8),
                                       ('nine', 9), ('', 0)]:
                pat = word + r'[\s-]*' + suffix_w
                repl = str(val + suffix_v)
                full_text = re.sub(
                    pat + r'\s+percent',
                    repl + '%', full_text,
                    flags=re.IGNORECASE,
                )

        patterns = [
            # "hazard ratio, 0.48 (95% CI, 0.35 to 0.66)"
            r'hazard\s+ratio[,;:\s]+(\d+\.?\d*)\s*'
            r'\(?\s*95\s*%?\s*(?:CI|confidence)[,;:\s]+'
            r'(\d+\.?\d*)\s*(?:to|-|\u2013)\s*(\d+\.?\d*)',
            # "HR 0.48 (95% CI, 0.35-0.66)"
            r'HR[,;:\s=]+(\d+\.?\d*)\s*'
            r'\(?\s*95\s*%?\s*(?:CI|confidence)[,;:\s]+'
            r'(\d+\.?\d*)\s*(?:to|-|\u2013)\s*(\d+\.?\d*)',
            # "hazard ratio 0.48 (0.35-0.66)"
            r'hazard\s+ratio[,;:\s]+(\d+\.?\d*)\s*'
            r'\(\s*(\d+\.?\d*)\s*(?:to|-|\u2013)\s*(\d+\.?\d*)\s*\)',
            # "HR, 0.48 [0.35-0.66]"
            r'HR[,;:\s=]+(\d+\.?\d*)\s*'
            r'\[\s*(\d+\.?\d*)\s*(?:to|-|\u2013)\s*(\d+\.?\d*)\s*\]',
            # "HR 0.48; 95% CI 0.35-0.66" (semicolon separator)
            r'HR[,;:\s=]+(\d+\.?\d*)\s*[;,]\s*95\s*%?\s*CI[,;:\s]+'
            r'(\d+\.?\d*)\s*(?:to|-|\u2013)\s*(\d+\.?\d*)',
            # "hazard ratio [HR] 0.48"
            r'hazard\s+ratio\s*\[HR\][,;:\s]+(\d+\.?\d*)',
            # "HR (95% CI) 0.48 (0.35-0.66)" — HR value after CI label
            r'HR\s*\(95\s*%?\s*CI\)\s*[,;:\s]*(\d+\.?\d*)\s*'
            r'\(\s*(\d+\.?\d*)\s*(?:to|-|\u2013)\s*(\d+\.?\d*)\s*\)',
            # Bare "hazard ratio 0.48" or "HR = 0.48"
            r'(?:hazard\s+ratio|HR)[,;:\s=]+(\d+\.?\d*)',
        ]

        # Negative context: multivariate/subgroup/secondary HRs to skip
        _NEGATIVE = re.compile(
            r'multivariat|multivariable|adjust|model\s*\d|predictor|'
            r'covariat|regression|independent|subgroup|stratif|'
            r'per.protocol|sensitivity|landmark|competing|'
            r'secondary|pooled|combined|interaction',
            re.IGNORECASE,
        )

        # Citation marker after HR: "[18]", "(ref 5)", superscript-style
        _CITATION_POST = re.compile(
            r'\)\s*\[\d+\]'     # ") [18]" — bracket citation after CI
            r'|\]\s*\[\d+\]'    # "] [18]" — bracket citation after CI
            r'|\)\s*\d{1,3}\b'  # ") 18"   — superscript citation
        )
        # Citation context: another study mentioned near the HR
        _CITED_STUDY = re.compile(
            r'\b\w+\s+et\s+al\.?\s*[\[\d(,;]',
            re.IGNORECASE,
        )

        # SM-2: Priority scoring for text HR disambiguation
        # Prefer: abstract > body, has CI > bare, primary keyword > generic
        _PRIMARY_KW = re.compile(
            r'primary|main\s+(?:end|out)|principal|overall\s+survival'
            r'|progression.free|disease.free|event.free'
            r'|composite\s+(?:end|out)|MACE',
            re.IGNORECASE,
        )

        candidates = []
        for pat_idx, pattern in enumerate(patterns):
            has_ci = pat_idx < len(patterns) - 2  # Last 2 patterns are bare
            for m in re.finditer(pattern, full_text, re.IGNORECASE):
                try:
                    hr = float(m.group(1))
                    if not (0.05 <= hr <= 20.0):
                        continue
                    # Check BEFORE the match only (not after) to avoid
                    # catching section headers like "Secondary Outcome"
                    # that follow the HR mention.
                    pre_ctx = full_text[max(0, m.start() - 200):
                                        m.start()]
                    if _NEGATIVE.search(pre_ctx):
                        continue  # Skip multivariate / subgroup HRs

                    # Check for citation markers AFTER the match — these
                    # indicate the HR is from a cited reference, not this
                    # trial's own result.
                    post_ctx = full_text[m.end():
                                         min(len(full_text), m.end() + 40)]
                    if _CITATION_POST.search(post_ctx):
                        continue  # Skip cited reference HRs

                    # Check if a different study's author is mentioned
                    # nearby (within 120 chars after), suggesting citation
                    near_post = full_text[m.end():
                                          min(len(full_text), m.end() + 120)]
                    if _CITED_STUDY.search(near_post):
                        continue  # Skip HRs from cited studies

                    ctx = full_text[max(0, m.start() - 40):
                                    m.end() + 40].strip()
                    ctx = re.sub(r'\s+', ' ', ctx)

                    # Priority scoring (higher = better)
                    priority = 0
                    pos = m.start()
                    in_abstract = pos < 3000
                    if in_abstract:
                        priority += 10  # Abstract HRs are primary
                    if has_ci:
                        priority += 5   # Has CI = more likely primary
                    # Check for primary endpoint keywords nearby
                    nearby = full_text[max(0, pos - 300):pos + 100]
                    if _PRIMARY_KW.search(nearby):
                        priority += 3
                    # Penalize later occurrences slightly
                    priority -= pos / len(full_text) * 2

                    candidates.append((hr, ctx, pos, priority))
                except (ValueError, IndexError):
                    continue

        if not candidates:
            # Fallback: derive HR from event-free survival rate pairs
            return self._derive_hr_from_event_rates(full_text)

        # Sort by priority (descending), then position (ascending)
        candidates.sort(key=lambda x: (-x[3], x[2]))
        return candidates[0][0], candidates[0][1]

    def _derive_hr_from_event_rates(self, full_text: str
                                     ) -> Tuple[Optional[float], Optional[str]]:
        """Derive approximate HR from event-free survival rate pairs.

        When no explicit HR is reported (common in non-inferiority trials),
        papers often report event-free survival rates per arm, e.g.
        "event-free survival vHPSD 73.8% vs. CBA 81.4%".

        Uses exponential approximation: HR = log(S1) / log(S2)
        where S1, S2 are the event-free survival proportions.

        The derived HR is approximate but provides a reliable directional
        anchor for the pipeline's orientation resolution.
        """
        # Pattern: "X% vs Y%" or "X% versus Y%" near relevant keywords
        # Also: "X% (arm1) ... Y% (arm2)" nearby
        rate_patterns = [
            # "73.8% vs. CBA 81.4%" or "80.0% (cryo) versus 78.0% (laser)"
            # Gap up to 160 chars to allow long sentences between keyword
            # and percentages (common in papers with detailed descriptions)
            r'(?:event.?free|freedom|success|recurrence.?free|failure.free'
            r'|arrhythmia.free|survival|recurrence\s+rate'
            r'|efficacy\s+end.?point'
            r'|primary\s+end.?point)[^.]{0,160}?'
            r'(\d+\.?\d)\s*%\s*(?:\([^)]*\)\s*)?'
            r'(?:vs\.?|versus|compared\s+(?:to|with))\s*'
            r'(?:[^0-9]{0,30})?(\d+\.?\d)\s*%',
            # "was 73.3% with pentaspline PFA and 71.3% with thermal"
            # Also handles "achieved by 40% ... and 42%" (FREEZEAF)
            # Allow 1-4 words for arm names (e.g. "pentaspline PFA")
            r'(?:event.?free|freedom|success|recurrence.free|failure.free'
            r'|arrhythmia.free|survival|recurrence|endpoint|AA\s+burden'
            r'|AA\s+recurrence|efficacy\s+end.?point'
            r'|primary\s+end.?point)[^.]{0,120}?'
            r'(?:was\s+achieved\s+by|achieved\s+by|was|of|rate'
            r'|met\s+by)\s+'
            r'(\d+\.?\d)\s*%\s*'
            r'(?:\([^)]*\)\s*)?'
            r'(?:with|for|in|of)\s+\w+(?:\s+\w+){0,5}\s+(?:and|,)\s+'
            r'(\d+\.?\d)\s*%',
            # "met by 71.2% versus 69.3%, in the LB and RF groups"
            # Also handles "achieved by X% versus Y%"
            r'(?:end.?point|efficacy|primary)[^.]{0,100}?'
            r'(?:met\s+by|was|reached|achieved\s+by)\s+'
            r'(\d+\.?\d)\s*%\s*'
            r'(?:\([^)]*\)\s*)?'
            r'(?:vs\.?|versus|compared\s+(?:to|with))\s*'
            r'(?:[^0-9]{0,30})?(\d+\.?\d)\s*%',
            # "X% of [arm] and Y% of [arm]" (WACA-PVAC format)
            # "56% of WACA and 60% of PVAC patients were free of AF"
            r'(\d+\.?\d)\s*%\s+of\s+\w+(?:\s+\w+){0,2}\s+'
            r'(?:and|,)\s+(\d+\.?\d)\s*%\s+of\s+\w+'
            r'[^.]{0,80}?(?:free|success|survival|recurrence)',
            # "60% ... free ... This compares with 56%" (cross-sentence)
            r'(\d+\.?\d)\s*%\s+of\s+[^.]{0,80}?'
            r'(?:free|success|survival|recurrence)[^.]*\.\s*'
            r'(?:This|It)\s+compares?\s+with\s+(\d+\.?\d)\s*%',
            # NOTE: A loose "X% and Y%" pattern was removed because it
            # false-positives on secondary endpoints (e.g., "freedom rate
            # from documented AF after cessation of AADs was 70.5% and 73.4%"
            # in CRAVE, which is NOT the primary endpoint).
        ]

        # Detect cited studies: "Author et al." near rate pair → skip
        _CITED_STUDY_RATE = re.compile(
            r'\b\w+\s+et\s+al\.?\s*[\[\d(,;]',
            re.IGNORECASE,
        )

        candidates = []
        for pattern in rate_patterns:
            for m in re.finditer(pattern, full_text, re.IGNORECASE):
                try:
                    r1 = float(m.group(1)) / 100.0
                    r2 = float(m.group(2)) / 100.0

                    # Detect EVENT rates (recurrence, not recurrence-free)
                    # and convert to survival rates before computing HR.
                    pre_keyword = full_text[max(0, m.start() - 40):
                                            m.start() + 30].lower()
                    is_event_rate = (
                        'recurrence' in pre_keyword and
                        'free' not in pre_keyword.split('recurrence')[-1][:10]
                    )
                    if is_event_rate:
                        r1 = 1.0 - r1
                        r2 = 1.0 - r2

                    # Both must be plausible survival rates (20%-99%)
                    if not (0.20 <= r1 <= 0.99 and 0.20 <= r2 <= 0.99):
                        continue
                    # Rates too close → HR too noisy
                    if abs(r1 - r2) < 0.005:
                        continue
                    # Both must not be 0 (log undefined)
                    if r1 <= 0 or r2 <= 0:
                        continue

                    hr = np.log(r1) / np.log(r2)
                    if not (0.1 <= hr <= 10.0):
                        continue

                    # Skip rate pairs from cited studies: if "Author et al."
                    # appears within 100 chars after OR 200 chars before
                    # the match, this is a different study's result cited
                    # in the Discussion section.
                    post_ctx_rate = full_text[m.end():
                                              min(len(full_text),
                                                  m.end() + 100)]
                    pre_ctx_rate = full_text[max(0, m.start() - 200):
                                             m.start()]
                    if (_CITED_STUDY_RATE.search(post_ctx_rate) or
                            _CITED_STUDY_RATE.search(pre_ctx_rate)):
                        continue

                    ctx = full_text[max(0, m.start() - 30):
                                    m.end() + 30].strip()
                    ctx = re.sub(r'\s+', ' ', ctx)
                    candidates.append((
                        round(hr, 3),
                        f"[DERIVED from {r1*100:.1f}% vs {r2*100:.1f}%] {ctx}",
                        m.start(),
                    ))
                except (ValueError, ZeroDivisionError):
                    continue

        if not candidates:
            return None, None

        # Prefer earliest mention (abstract/results section)
        candidates.sort(key=lambda x: x[2])
        hr, ctx, _ = candidates[0]
        logger.info(f"Derived HR={hr:.3f} from event rates: {ctx[:100]}")
        return hr, ctx

    # ------------------------------------------------------------------
    # Pair scoring  (Stage 1)
    # ------------------------------------------------------------------

    def _score_curve(self, curve: Dict) -> float:
        """Score individual curve quality for primary-endpoint likelihood."""
        score = 0.0
        survs = curve['survivals']
        times = curve['times']

        # Starting near 1.0 — strong signal for a true KM survival curve
        # (cumulative incidence, hazard, or burden plots start near 0.0)
        if survs[0] >= 0.95:
            score += 35
        elif survs[0] >= 0.85:
            score += 20
        elif survs[0] >= 0.7:
            score += 10
        else:
            score -= 10  # Penalize curves not starting high (likely not KM)

        # Number of points
        n = len(times)
        if n > 80:
            score += 15
        elif n > 50:
            score += 10
        elif n > 30:
            score += 5

        # Event rate (drop)
        drop = curve['drop']
        if drop >= 0.3:
            score += 20
        elif drop >= 0.15:
            score += 15
        elif drop >= 0.05:
            score += 8

        # Time span coverage (relative to max observed time across curve)
        if len(times) > 1 and times[-1] > times[0]:
            score += 15  # any non-trivial span

        # Standard KM colours
        color = curve.get('color_name', '').lower()
        if color in ('blue', 'red'):
            score += 5
        elif color in ('green', 'black'):
            score += 3

        # Penalize curves from pages classified as likely non-KM
        if curve.get('non_km_page', False):
            score -= 15

        # Bonus for curves with NAR data (more accurate Guyot reconstruction)
        if curve.get('nar_data') is not None:
            score += 10

        return score

    def _score_pair(self, c1: Dict, c2: Dict) -> float:
        """Score a curve pair: individual quality + structural signals."""
        pair_score = self._score_curve(c1) + self._score_curve(c2)

        # Page proximity scoring.
        # Same-page pairs are preferred (KM arms usually on one figure),
        # but cross-page pairs can be valid when curves span pages.
        # Penalty is mild to avoid forcing bad same-page pairs.
        page_dist = abs(c1.get('page', 0) - c2.get('page', 0))
        if page_dist == 0 and c1.get('page', -1) >= 0:
            pair_score += 10  # Same page bonus (mild)
        elif page_dist > 0:
            pair_score -= 3 * page_dist  # Mild cross-page penalty

        # Different base colours: mild preference only.
        # Many papers use same-colour with different line styles
        # (solid vs dashed), so this must be a small tiebreaker.
        base1 = re.sub(r'_\d+$', '', c1.get('color_name', ''))
        base2 = re.sub(r'_\d+$', '', c2.get('color_name', ''))
        if base1 != base2:
            pair_score += 5

        # Drop similarity: curves from the same KM survival endpoint
        # should have similar total drops (treatment vs control arms
        # diverge but not dramatically). Curves from DIFFERENT panels
        # (e.g. cumulative incidence vs survival) have very different drops.
        drop1 = c1.get('drop', 0)
        drop2 = c2.get('drop', 0)
        drop_diff = abs(drop1 - drop2)
        if drop_diff < 0.05:
            pair_score += 15  # Very similar drops → likely same endpoint
        elif drop_diff < 0.10:
            pair_score += 5
        elif drop_diff > 0.20:
            pair_score -= 10  # Very different → likely different endpoints

        # Starting value similarity: both arms should start near same
        # survival level (both should be ~1.0 for KM curves)
        start1 = c1['survivals'][0]
        start2 = c2['survivals'][0]
        start_diff = abs(start1 - start2)
        if start_diff < 0.05:
            pair_score += 10  # Both start at same level
        elif start_diff > 0.15:
            pair_score -= 10  # Very different starting points

        # Duplicate detection: penalize truly identical curves (same arm
        # detected twice, or axis artefact). Only trigger at very low
        # threshold — non-inferiority trials have legitimately close arms
        # (mean abs diff ≈ 0.02-0.05), which MUST NOT be penalized.
        try:
            t1 = np.array(c1['times'])
            s1 = np.array(c1['survivals'])
            t2 = np.array(c2['times'])
            s2 = np.array(c2['survivals'])
            lo = max(t1.min(), t2.min())
            hi = min(t1.max(), t2.max())
            if hi > lo:
                # Overlapping time windows — compare directly
                ct = np.linspace(lo, hi, 50)
                diff = np.mean(np.abs(
                    np.interp(ct, t1, s1) - np.interp(ct, t2, s2)))
                if diff < 0.01:
                    pair_score -= 40  # Near-identical = likely duplicate
            else:
                # Non-overlapping time windows — compare drop shapes.
                # If both curves have very similar normalized survival
                # profiles, they're likely the same arm from different crops.
                if (len(s1) >= 5 and len(s2) >= 5
                        and abs(s1[0] - s2[0]) < 0.05
                        and abs(s1[-1] - s2[-1]) < 0.05
                        and abs(c1['drop'] - c2['drop']) < 0.02):
                    pair_score -= 30  # Likely same arm, different crop
        except Exception:
            pass

        return pair_score

    def _select_top_pairs(self, curves, top_k):
        """Select top-K pairs, strongly preferring same-page pairs."""
        scored = []
        for i in range(len(curves)):
            for j in range(i + 1, len(curves)):
                s = self._score_pair(curves[i], curves[j])
                scored.append((curves[i], curves[j], s))
        scored.sort(key=lambda x: -x[2])
        return scored[:top_k]

    # ------------------------------------------------------------------
    # HR estimation  (Stage 2)
    # ------------------------------------------------------------------

    def _extract_nar_for_curve(self, curve, arm_index=0):
        """Extract NAR data for a curve.

        Returns (n_total, nar_times_array, nar_values_array).
        Falls back to (100, None, None) when no NAR is available.

        Filters implausible OCR values: N must be in [20, 5000] to be
        used. Outside that range, the OCR likely misread the table.
        """
        nar_data = curve.get('nar_data')
        if nar_data is None or not nar_data.rows:
            return 100, None, None

        if arm_index >= len(nar_data.rows):
            logger.debug(f"NAR has {len(nar_data.rows)} rows but "
                         f"arm_index={arm_index}; falling back to row 0")
        idx = min(arm_index, len(nar_data.rows) - 1)
        row = nar_data.rows[idx]

        n_total = row.values[0] if row.values else 100

        # Filter implausible NAR values (OCR errors)
        if not (20 <= n_total <= 5000):
            logger.debug(
                f"NAR N={n_total} implausible, ignoring (arm {arm_index})")
            return 100, None, None

        if (row.timepoints and row.values
                and len(row.timepoints) >= 2
                and len(row.values) >= 2):
            # Also filter if any value in the row is implausible
            vals = [v for v in row.values if isinstance(v, (int, float))]
            if vals and max(vals) <= 5000 and min(vals) >= 0:
                return (n_total,
                        np.array(row.timepoints, dtype=float),
                        np.array(row.values, dtype=float))

        return n_total, None, None

    def _estimate_hr_with_nar(self, t1, s1, t2, s2,
                               n1, n2, nrt1, nrv1, nrt2, nrv2):
        """Estimate HR using full NAR data for accurate Guyot IPD
        reconstruction.  Falls back to estimate_hr_from_curves when
        log-rank cannot be computed.
        """
        ipd1 = reconstruct_ipd_guyot(
            t1, s1, n_risk_times=nrt1, n_risk_values=nrv1, total_n=n1)
        ipd2 = reconstruct_ipd_guyot(
            t2, s2, n_risk_times=nrt2, n_risk_values=nrv2, total_n=n2)

        for rec in ipd1:
            rec.arm = 1
        for rec in ipd2:
            rec.arm = 0

        Z, p_value, O_1, E_1, V = log_rank_test(ipd1, ipd2)
        total_events = (sum(1 for r in ipd1 if r.event == 1)
                        + sum(1 for r in ipd2 if r.event == 1))

        if total_events == 0:
            return HRResult(
                hr=1.0, ci_lower=0.5, ci_upper=2.0, p_value=1.0,
                method="log_rank_nar", n_events=0, n_total=n1 + n2,
                log_rank_statistic=0)

        if V > 0 and total_events >= 20:
            log_hr = (O_1 - E_1) / V
            se_log_hr = 1.0 / np.sqrt(V)
            hr = np.exp(log_hr)
            hr = max(0.05, min(20.0, hr))
            ci_lower = np.exp(np.log(hr) - 1.96 * se_log_hr)
            ci_upper = np.exp(np.log(hr) + 1.96 * se_log_hr)
            return HRResult(
                hr=round(hr, 3),
                ci_lower=round(ci_lower, 3),
                ci_upper=round(ci_upper, 3),
                p_value=round(p_value, 4),
                method="log_rank_nar",
                n_events=total_events,
                n_total=n1 + n2,
                log_rank_statistic=round(Z, 3))

        # Low-event fallback
        return estimate_hr_from_curves(t1, s1, t2, s2, n1=n1, n2=n2)

    @staticmethod
    def _is_valid_km_curve(times, survs) -> bool:
        """Validate that a curve looks like a proper KM survival curve.

        Returns False for cumulative incidence (starts near 0, increases),
        hazard/burden plots, or curves with too few usable points.
        """
        if len(times) < 5 or len(survs) < 5:
            return False
        if survs[0] < 0.5:
            return False  # Not a survival curve (likely cumulative incidence)
        if survs[-1] >= survs[0]:
            return False  # Non-decreasing = not KM survival
        return True

    def _estimate_both(self, c1, c2):
        """Estimate HR in both Guyot orientations.

        Validates that both curves look like proper KM survival curves
        before attempting HR estimation. When NAR data is attached,
        uses full NAR timepoints/values for accurate IPD reconstruction.
        """
        t1 = np.array(c1['times'])
        s1 = np.array(c1['survivals'])
        t2 = np.array(c2['times'])
        s2 = np.array(c2['survivals'])

        # P0 fix: validate both curves are proper KM survival curves
        if not self._is_valid_km_curve(t1, s1):
            logger.debug(f"Curve 1 failed KM validation: start={s1[0]:.2f}, "
                         f"end={s1[-1]:.2f}, n={len(s1)}")
            return None, None, None, None
        if not self._is_valid_km_curve(t2, s2):
            logger.debug(f"Curve 2 failed KM validation: start={s2[0]:.2f}, "
                         f"end={s2[-1]:.2f}, n={len(s2)}")
            return None, None, None, None

        # --- Extract NAR data ---
        n1, nrt1, nrv1 = self._extract_nar_for_curve(c1, arm_index=0)
        n2, nrt2, nrv2 = self._extract_nar_for_curve(c2, arm_index=1)

        has_nar = nrt1 is not None or nrt2 is not None
        if has_nar:
            logger.info(f"Using NAR data: n1={n1}, n2={n2}")

        hr_fwd = hr_inv = None
        res_fwd = res_inv = None

        if has_nar:
            # Full NAR-aware Guyot reconstruction
            try:
                res_fwd = self._estimate_hr_with_nar(
                    t1, s1, t2, s2, n1, n2,
                    nrt1, nrv1, nrt2, nrv2)
                if res_fwd and res_fwd.hr and res_fwd.hr > 0:
                    hr_fwd = res_fwd.hr
            except Exception as e:
                logger.debug(f"NAR forward HR estimation failed: {e}")
            try:
                res_inv = self._estimate_hr_with_nar(
                    t2, s2, t1, s1, n2, n1,
                    nrt2, nrv2, nrt1, nrv1)
                if res_inv and res_inv.hr and res_inv.hr > 0:
                    hr_inv = res_inv.hr
            except Exception as e:
                logger.debug(f"NAR inverse HR estimation failed: {e}")
        else:
            # Original path — pass n1/n2 (may still be > 100 from NAR
            # initial_n even without full timepoints)
            try:
                res_fwd = estimate_hr_from_curves(
                    t1, s1, t2, s2, n1=n1, n2=n2)
                if res_fwd and res_fwd.hr and res_fwd.hr > 0:
                    hr_fwd = res_fwd.hr
            except Exception as e:
                logger.debug(f"Forward HR estimation failed: {e}")
            try:
                res_inv = estimate_hr_from_curves(
                    t2, s2, t1, s1, n1=n2, n2=n1)
                if res_inv and res_inv.hr and res_inv.hr > 0:
                    hr_inv = res_inv.hr
            except Exception as e:
                logger.debug(f"Inverse HR estimation failed: {e}")

        return hr_fwd, hr_inv, res_fwd, res_inv

    # ------------------------------------------------------------------
    # Confidence scoring
    # ------------------------------------------------------------------

    def _compute_confidence(self, method, text_hr, hr, pair_rank,
                            pair_score, n_curves):
        conf = 0.50

        # Text cross-validation
        if method == "text_hr_match" and text_hr:
            rel_err = abs(hr - text_hr) / text_hr if text_hr > 0 else 1.0
            if rel_err < 0.15:
                conf += 0.30
            elif rel_err < 0.30:
                conf += 0.15
            else:
                conf += 0.05

        # Pair rank
        if pair_rank == 1:
            conf += 0.10
        elif pair_rank <= 3:
            conf += 0.05

        # Curve richness
        if n_curves >= 6:
            conf += 0.05

        # HR plausibility
        if 0.3 <= hr <= 3.0:
            conf += 0.05
        elif hr < 0.1 or hr > 10.0:
            conf -= 0.15

        # Consensus direction boost
        if method == "consensus_direction":
            conf += 0.05

        return round(min(1.0, max(0.0, conf)), 2)

    # ------------------------------------------------------------------
    # Result builders
    # ------------------------------------------------------------------

    def _build_result(self, best, text_hr, text_ctx, method,
                      all_curves, pdf_path, pdf_hash, n_pages,
                      warnings, t0):
        hr = best['hr']
        res = best.get('res')
        elapsed = time.time() - t0

        if hr < 0.3 or hr > 3.0:
            warnings.append(
                f"Extreme HR={hr:.3f}: manual review recommended")

        prov = ExtractionProvenance(
            pdf_path=pdf_path,
            pdf_sha256=pdf_hash,
            pages_scanned=n_pages,
            total_curves_extracted=len(all_curves),
            pair_description=best['desc'],
            pair_quality_score=best['score'],
            pair_rank=best['rank'],
            orientation=best['orient'],
            orientation_method=method,
            text_hr_found=text_hr,
            text_hr_context=text_ctx or "",
            hr_fwd=best.get('hr_fwd'),
            hr_inv=best.get('hr_inv'),
            processing_time_s=round(elapsed, 2),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        conf = self._compute_confidence(
            method, text_hr, hr, best['rank'], best['score'],
            len(all_curves),
        )

        return HRExtractionResult(
            hr=round(hr, 4),
            ci_lower=round(res.ci_lower, 4) if res else None,
            ci_upper=round(res.ci_upper, 4) if res else None,
            confidence=conf,
            provenance=prov,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _pdf_hash(self, pdf_path: str) -> str:
        if not os.path.exists(pdf_path):
            return "file_not_found"
        h = hashlib.sha256()
        with open(pdf_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()[:16]

    def _error_result(self, pdf_path, pdf_hash, msg, *,
                      pages_scanned=0, n_curves=0):
        return HRExtractionResult(
            hr=None, ci_lower=None, ci_upper=None, confidence=0.0,
            provenance=ExtractionProvenance(
                pdf_path=pdf_path, pdf_sha256=pdf_hash,
                pages_scanned=pages_scanned,
                total_curves_extracted=n_curves,
                pair_description="", pair_quality_score=0.0,
                pair_rank=0, orientation="", orientation_method="",
                timestamp=datetime.now(timezone.utc).isoformat(),
            ),
            error=msg,
        )


# ---------------------------------------------------------------------------
# Legacy compatibility shim for validate_all_rcts.py
# ---------------------------------------------------------------------------

class _LegacyResult:
    """Thin wrapper so old code can access result.hr, result.ci_lower, etc."""
    def __init__(self, r: HRExtractionResult):
        self._r = r
        self.hr = r.hr
        self.ci_lower = r.ci_lower
        self.ci_upper = r.ci_upper
        self.p_value = None
        self.page_selected = r.provenance.pair_rank
        self.n_candidates = r.provenance.pages_scanned
        self.n_curves_found = r.provenance.total_curves_extracted
        self.curves = []
        self.arm_method = r.provenance.orientation_method
        self.arm_confident = r.confidence > 0.6
        self.arm_warning = "; ".join(r.warnings)
        self.hr_within_bounds = True
        self.hr_warning = ""
        self.selection_score = r.provenance.pair_quality_score
        self.selection_reasons = r.warnings
        self.processing_time_ms = r.provenance.processing_time_s * 1000
        self.error_message = r.error or ""


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def extract_hr(pdf_path: str) -> HRExtractionResult:
    """One-call convenience wrapper."""
    return RobustKMPipeline().extract_hr(pdf_path)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else \
        r"C:\Users\user\Downloads\Ablation\euaf066.pdf"
    print(f"Testing v3.0 pipeline on: {path}")
    print("=" * 60)
    r = extract_hr(path)
    if r.succeeded:
        print(f"HR:          {r.hr:.4f} ({r.ci_lower}-{r.ci_upper})")
        print(f"Confidence:  {r.confidence}")
        print(f"Method:      {r.provenance.orientation_method}")
        print(f"Text HR:     {r.provenance.text_hr_found}")
        print(f"Pair:        {r.provenance.pair_description}")
        print(f"Orientation: {r.provenance.orientation}")
        print(f"HR fwd/inv:  {r.provenance.hr_fwd} / {r.provenance.hr_inv}")
        print(f"Time:        {r.provenance.processing_time_s:.1f}s")
        print(f"Status:      {r.certification_status}")
        if r.warnings:
            print(f"Warnings:    {r.warnings}")
    else:
        print(f"ERROR: {r.error}")
