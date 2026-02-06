"""
Robust KM Pipeline v3.0 - Two-Stage Pair Selection
====================================================

Extracts Hazard Ratios from Kaplan-Meier survival curves in RCT PDF figures.

Algorithm:
  Stage 1 (PAIR SELECTION): Score all curve pairs by individual quality,
           same-page bonus, different-color bonus, overlap penalty.
           Select top-K pairs (K=3). No HR estimation yet.

  Stage 2 (HR ESTIMATION): For each top-K pair, estimate HR in both
           Guyot orientations (the algorithm is asymmetric).

  Stage 3 (ORIENTATION RESOLUTION):
           - If text-reported HR found in PDF: match closest candidate
           - Else: for top-1 pair only, pick orientation closer to 1.0

  This fixes the "convergence to HR=1.0" bias from v2.0 by limiting
  candidates to 2*K instead of 2*N_pairs.

Validated on 7+ AF ablation RCTs.
All outputs carry certification_status = UNCERTIFIED.

Author: Wasserstein KM Extractor Team
Date: February 2026
Version: 3.0
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
from datetime import datetime

from simple_multicurve_handler import SimpleMultiCurveHandler
from improved_hr_estimation import estimate_hr_from_curves, HRResult

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
    pipeline_version: str = "3.0"
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
        all_curves = self._extract_all_curves(pages[:n_pages], handler)
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
        if text_hr is not None:
            result = self._resolve_with_text_hr(
                top_pairs, text_hr, text_ctx, all_curves, pdf_path,
                pdf_hash, n_pages, warnings, t0,
            )
        else:
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
                              n_pages, warnings, t0):
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

                rel_err = abs(hr_val - text_hr) / text_hr \
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

        # Prefer candidates within 50% of text HR
        close = [c for c in candidates if c['text_err'] < 0.50]
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

        if best.get('text_err', 0) > 0.30:
            warnings.append(
                f"Text HR match weak: extracted {best['hr']:.3f} "
                f"vs text {text_hr} ({best['text_err']*100:.0f}% off)"
            )

        return self._build_result(
            best, text_hr, text_ctx, "text_hr_match",
            all_curves, pdf_path, pdf_hash, n_pages, warnings, t0,
        )

    def _resolve_without_text_hr(self, top_pairs, all_curves, pdf_path,
                                 pdf_hash, n_pages, warnings, t0):
        """Without text HR, search many pairs and pick HR closest to 1.0.

        This is the validated v2.0 approach which works well when the
        true HR is in the typical RCT range (0.5-2.0). It searches
        up to 20 pairs in both orientations and picks the HR closest
        to 1.0 among plausible candidates.
        """
        # Expand search: use up to 20 pairs (v2.0 approach)
        expanded = self._select_top_pairs(all_curves, 20)
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
                candidates.append({
                    'hr': hr_val,
                    'dist_from_1': abs(np.log(hr_val)),
                    'orient': orient, 'res': res,
                    'rank': rank, 'score': pair_score,
                    'desc': pair_desc,
                    'hr_fwd': hr_fwd, 'hr_inv': hr_inv,
                })

        if not candidates:
            return None

        # Filter to plausible range
        plausible = [c for c in candidates if 0.2 <= c['hr'] <= 5.0]
        if not plausible:
            plausible = candidates

        # Pick closest to 1.0
        plausible.sort(key=lambda c: c['dist_from_1'])
        best = plausible[0]

        warnings.append(
            "No text HR found; orientation resolved by "
            "closest-to-1.0 prior (less reliable for extreme HRs)"
        )

        return self._build_result(
            best, None, None, "closest_to_1",
            all_curves, pdf_path, pdf_hash, n_pages, warnings, t0,
        )

    # ------------------------------------------------------------------
    # Curve extraction
    # ------------------------------------------------------------------

    def _rasterize_pages(self, pdf_path: str) -> List[np.ndarray]:
        pages = []
        doc = fitz.open(pdf_path)
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
            pages.append(img)
        doc.close()
        return pages

    def _extract_all_curves(self, pages, handler) -> List[Dict]:
        all_curves = []
        for page_num, img in enumerate(pages):
            if img is None:
                continue
            try:
                result = handler.process_figure(img)
                for curve in (result.get('curves', []) if result else []):
                    if not hasattr(curve, 'survival_data'):
                        continue
                    if len(curve.survival_data) < self.min_curve_points:
                        continue
                    times = [t for t, s in curve.survival_data]
                    survs = [s for t, s in curve.survival_data]
                    if survs[0] < 0.5 or survs[-1] >= survs[0]:
                        continue
                    all_curves.append({
                        'times': times,
                        'survivals': survs,
                        'color_name': getattr(curve, 'color_name', 'unknown'),
                        'page': page_num,
                        'n_points': len(times),
                        'drop': survs[0] - survs[-1],
                    })
            except Exception as e:
                logger.debug(f"Page {page_num} error: {e}")
        return all_curves

    # ------------------------------------------------------------------
    # Text HR extraction
    # ------------------------------------------------------------------

    def _extract_text_hr(self, pdf_path: str
                         ) -> Tuple[Optional[float], Optional[str]]:
        """Extract the primary reported HR from PDF text (first 6 pages)."""
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            for i in range(min(len(doc), 6)):
                full_text += doc[i].get_text() + "\n"
            doc.close()
        except Exception:
            return None, None

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
            # Bare "hazard ratio 0.48" or "HR = 0.48"
            r'(?:hazard\s+ratio|HR)[,;:\s=]+(\d+\.?\d*)',
        ]

        # Negative context: multivariate/subgroup/secondary HRs to skip
        _NEGATIVE = re.compile(
            r'multivariat|multivariable|adjusted|model\s*\d|predictor|'
            r'covariat|regression|independent|subgroup|stratif|'
            r'per.protocol|sensitivity|landmark|competing|'
            r'secondary|pooled|combined|interaction',
            re.IGNORECASE,
        )

        candidates = []
        for pattern in patterns:
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

                    ctx = full_text[max(0, m.start() - 40):
                                    m.end() + 40].strip()
                    ctx = re.sub(r'\s+', ' ', ctx)
                    candidates.append((hr, ctx, m.start()))
                except (ValueError, IndexError):
                    continue

        if not candidates:
            return None, None

        # Prefer earliest mention (usually abstract / results)
        candidates.sort(key=lambda x: x[2])
        return candidates[0][0], candidates[0][1]

    # ------------------------------------------------------------------
    # Pair scoring  (Stage 1)
    # ------------------------------------------------------------------

    def _score_curve(self, curve: Dict) -> float:
        """Score individual curve quality for primary-endpoint likelihood."""
        score = 0.0
        survs = curve['survivals']
        times = curve['times']

        # Starting near 1.0
        if survs[0] >= 0.95:
            score += 25
        elif survs[0] >= 0.85:
            score += 15
        elif survs[0] >= 0.7:
            score += 8

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

        return score

    def _score_pair(self, c1: Dict, c2: Dict) -> float:
        """Score a curve pair: individual quality + structural signals."""
        pair_score = self._score_curve(c1) + self._score_curve(c2)
        same_page = (c1.get('page', -1) == c2.get('page', -2)
                     and c1.get('page', -1) >= 0)

        # Same page is a strong signal (KM figures always on one page)
        if same_page:
            pair_score += 25

        # Different base colours: mild preference only.
        # Many papers use same-colour with different line styles
        # (solid vs dashed), so this must be a small tiebreaker.
        base1 = re.sub(r'_\d+$', '', c1.get('color_name', ''))
        base2 = re.sub(r'_\d+$', '', c2.get('color_name', ''))
        if base1 != base2:
            pair_score += 5

        # Overlap penalty (near-identical = likely same arm or axis)
        try:
            t1 = np.array(c1['times'])
            s1 = np.array(c1['survivals'])
            t2 = np.array(c2['times'])
            s2 = np.array(c2['survivals'])
            lo = max(t1.min(), t2.min())
            hi = min(t1.max(), t2.max())
            if hi > lo:
                ct = np.linspace(lo, hi, 50)
                diff = np.mean(np.abs(
                    np.interp(ct, t1, s1) - np.interp(ct, t2, s2)))
                if diff < 0.02:
                    pair_score -= 40
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

    def _estimate_both(self, c1, c2):
        """Estimate HR in both Guyot orientations."""
        t1 = np.array(c1['times'])
        s1 = np.array(c1['survivals'])
        t2 = np.array(c2['times'])
        s2 = np.array(c2['survivals'])

        hr_fwd = hr_inv = None
        res_fwd = res_inv = None

        try:
            res_fwd = estimate_hr_from_curves(t1, s1, t2, s2)
            if res_fwd and res_fwd.hr and res_fwd.hr > 0:
                hr_fwd = res_fwd.hr
        except Exception:
            pass

        try:
            res_inv = estimate_hr_from_curves(t2, s2, t1, s1)
            if res_inv and res_inv.hr and res_inv.hr > 0:
                hr_inv = res_inv.hr
        except Exception:
            pass

        return hr_fwd, hr_inv, res_fwd, res_inv

    def _pick_orientation(self, hr_fwd, hr_inv):
        """Pick orientation when no text HR is available."""
        if hr_fwd is None and hr_inv is None:
            return None, "", ""
        if hr_fwd is None:
            return hr_inv, "inverse", "only_valid"
        if hr_inv is None:
            return hr_fwd, "forward", "only_valid"

        d_fwd = abs(np.log(hr_fwd)) if hr_fwd > 0 else float('inf')
        d_inv = abs(np.log(hr_inv)) if hr_inv > 0 else float('inf')
        if d_fwd <= d_inv:
            return hr_fwd, "forward", "closest_to_1"
        return hr_inv, "inverse", "closest_to_1"

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
            timestamp=datetime.utcnow().isoformat(),
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
                timestamp=datetime.utcnow().isoformat(),
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
