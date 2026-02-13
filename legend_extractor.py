"""
Legend Extractor for KM Curves
==============================

Extracts legend information from KM curve figures to correctly identify
treatment vs control arms.

Features:
- Detect legend region (top-right, bottom, beside plot)
- Extract colored indicators (lines or boxes)
- OCR text labels next to each indicator
- Classify as treatment/control using keyword matching

Target: >80% legend detection, >90% arm classification accuracy

Author: Wasserstein KM Extractor Team
Date: February 2026
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import re
import logging

logger = logging.getLogger(__name__)

# Try to import OCR backends
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False


@dataclass
class LegendEntry:
    """A single entry in the legend."""
    label: str
    color_bgr: Tuple[int, int, int]
    color_name: str
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    is_treatment: bool = False
    confidence: float = 0.0


@dataclass
class LegendInfo:
    """Complete legend extraction result."""
    detected: bool
    entries: List[LegendEntry]
    region_bbox: Optional[Tuple[int, int, int, int]]
    treatment_arm: Optional[LegendEntry]
    control_arm: Optional[LegendEntry]
    confidence: float
    method: str
    n_arms_detected: int = 0


class LegendExtractor:
    """
    Extract legend information for curve identification.

    Legend detection strategy:
    1. Search common legend locations (top-right, bottom, right side)
    2. Find colored indicators (short lines or boxes)
    3. OCR text labels associated with each indicator
    4. Classify labels as treatment/control using keyword matching
    """

    def __init__(self):
        self.tesseract_available = TESSERACT_AVAILABLE
        self.easyocr_available = EASYOCR_AVAILABLE
        self._easyocr_reader = None

        # Keywords for treatment arm identification
        self.treatment_keywords = [
            'treatment', 'experimental', 'active', 'intervention',
            'pembrolizumab', 'nivolumab', 'atezolizumab', 'durvalumab',
            'ipilimumab', 'avelumab', 'cemiplimab',  # Checkpoint inhibitors
            'trastuzumab', 'bevacizumab', 'rituximab', 'cetuximab',  # mAbs
            'arm a', 'group a', 'investigational',
            'combination', 'combo', 'plus', '+',
            'high dose', 'intensive',
            'cryoablation', 'ablation', 'rf ablation', 'pfa',  # Ablation terms
        ]

        self.control_keywords = [
            'control', 'placebo', 'standard', 'comparator',
            'chemotherapy', 'chemo', 'soc', 'standard of care',
            'arm b', 'group b', 'reference',
            'observation', 'watchful waiting', 'surveillance',
            'low dose', 'conventional',
            'drug therapy', 'aad', 'anti-arrhythmic',  # Cardiology controls
        ]

        # Common colors used in KM curves
        self.color_names = {
            'blue': ((100, 50, 50), (130, 255, 255)),  # HSV range
            'red': ((0, 50, 50), (10, 255, 255)),
            'red2': ((170, 50, 50), (180, 255, 255)),
            'green': ((35, 50, 50), (85, 255, 255)),
            'orange': ((10, 50, 50), (25, 255, 255)),
            'purple': ((130, 50, 50), (160, 255, 255)),
            'black': ((0, 0, 0), (180, 50, 80)),
        }

    def _get_easyocr_reader(self):
        """Lazy-load EasyOCR reader."""
        if self._easyocr_reader is None and self.easyocr_available:
            self._easyocr_reader = easyocr.Reader(['en'], gpu=False)
        return self._easyocr_reader

    def extract(self, image: np.ndarray,
                plot_region: Optional[Tuple[int, int, int, int]] = None) -> LegendInfo:
        """
        Extract legend information from KM curve image.

        Parameters
        ----------
        image : np.ndarray
            Full figure image (BGR)
        plot_region : tuple, optional
            Region of KM plot (x, y, w, h) to help locate legend

        Returns
        -------
        LegendInfo
        """
        h, w = image.shape[:2]

        # Try different legend locations
        legend_regions = self._find_legend_regions(image, plot_region)

        best_result = LegendInfo(
            detected=False,
            entries=[],
            region_bbox=None,
            treatment_arm=None,
            control_arm=None,
            confidence=0.0,
            method='none'
        )

        for region_name, region_bbox in legend_regions:
            x1, y1, x2, y2 = region_bbox
            legend_region = image[y1:y2, x1:x2]

            if legend_region.size == 0:
                continue

            # Extract entries from this region
            entries = self._extract_legend_entries(legend_region)

            if not entries:
                continue

            # Classify arms
            treatment, control = self._classify_arms(entries)

            confidence = self._calculate_confidence(entries, treatment, control)

            if confidence > best_result.confidence:
                best_result = LegendInfo(
                    detected=True,
                    entries=entries,
                    region_bbox=region_bbox,
                    treatment_arm=treatment,
                    control_arm=control,
                    confidence=confidence,
                    method=region_name,
                    n_arms_detected=len(entries)
                )

        return best_result

    def _find_legend_regions(self, image: np.ndarray,
                              plot_region: Optional[Tuple[int, int, int, int]]) -> List[Tuple[str, Tuple[int, int, int, int]]]:
        """Find potential legend regions."""
        h, w = image.shape[:2]
        regions = []

        if plot_region:
            px, py, pw, ph = plot_region

            # Top-right of plot (most common)
            tr_x1 = px + int(pw * 0.5)
            tr_y1 = py
            tr_x2 = px + pw
            tr_y2 = py + int(ph * 0.4)
            regions.append(('top_right', (tr_x1, tr_y1, tr_x2, tr_y2)))

            # Below plot
            below_x1 = px
            below_y1 = py + ph
            below_x2 = px + pw
            below_y2 = min(h, py + ph + int(ph * 0.3))
            regions.append(('below', (below_x1, below_y1, below_x2, below_y2)))

            # Right of plot
            right_x1 = px + pw
            right_y1 = py
            right_x2 = min(w, px + pw + int(pw * 0.3))
            right_y2 = py + ph
            regions.append(('right', (right_x1, right_y1, right_x2, right_y2)))

            # Inside plot (top-right corner)
            inside_x1 = px + int(pw * 0.6)
            inside_y1 = py + int(ph * 0.05)
            inside_x2 = px + int(pw * 0.95)
            inside_y2 = py + int(ph * 0.35)
            regions.append(('inside_tr', (inside_x1, inside_y1, inside_x2, inside_y2)))

        else:
            # Default regions when plot region unknown
            regions.append(('top_right', (int(w * 0.5), 0, w, int(h * 0.4))))
            regions.append(('right', (int(w * 0.75), int(h * 0.2), w, int(h * 0.8))))
            regions.append(('below', (0, int(h * 0.75), w, h)))

        return regions

    def _extract_legend_entries(self, region: np.ndarray) -> List[LegendEntry]:
        """Extract legend entries from region."""
        h, w = region.shape[:2]
        entries = []

        # Convert to HSV for color detection
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        # Find colored indicators (short horizontal lines or boxes)
        color_indicators = self._find_color_indicators(region, hsv)

        # OCR the region for text labels
        text_regions = self._ocr_region(region)

        # Match color indicators with nearby text
        for color_info in color_indicators:
            color_bbox = color_info['bbox']
            color_bgr = color_info['color_bgr']
            color_name = color_info['color_name']

            # Find closest text to the right of this indicator
            best_text = None
            best_distance = float('inf')

            for text_info in text_regions:
                text_bbox = text_info['bbox']
                text_x = text_bbox[0]
                text_y = text_bbox[1] + text_bbox[3] // 2

                # Text should be to the right and roughly same vertical level
                if text_x > color_bbox[0] + color_bbox[2]:
                    dx = text_x - (color_bbox[0] + color_bbox[2])
                    dy = abs(text_y - (color_bbox[1] + color_bbox[3] // 2))

                    if dx < 200 and dy < 30:  # Reasonable proximity
                        distance = dx + dy * 2  # Weight vertical distance more
                        if distance < best_distance:
                            best_distance = distance
                            best_text = text_info

            if best_text:
                entries.append(LegendEntry(
                    label=best_text['text'],
                    color_bgr=color_bgr,
                    color_name=color_name,
                    bbox=color_bbox,
                    confidence=best_text.get('confidence', 0.5)
                ))

        return entries

    def _find_color_indicators(self, region: np.ndarray, hsv: np.ndarray) -> List[Dict]:
        """Find colored legend indicators (lines or boxes)."""
        h, w = region.shape[:2]
        indicators = []

        for color_name, (hsv_lower, hsv_upper) in self.color_names.items():
            lower = np.array(hsv_lower)
            upper = np.array(hsv_upper)
            mask = cv2.inRange(hsv, lower, upper)

            # Handle red wrapping around hue=0
            if color_name == 'red':
                lower2 = np.array(self.color_names['red2'][0])
                upper2 = np.array(self.color_names['red2'][1])
                mask2 = cv2.inRange(hsv, lower2, upper2)
                mask = cv2.bitwise_or(mask, mask2)

            if color_name == 'red2':
                continue

            # Find connected components
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, cw, ch = cv2.boundingRect(contour)
                area = cw * ch

                # Legend indicators are typically small, horizontal
                # Lines: width > height, area 50-500
                # Boxes: roughly square, area 100-1000
                if 20 < area < 2000:
                    aspect_ratio = cw / max(1, ch)

                    # Line indicator
                    if aspect_ratio > 2 and cw > 15 and ch < 20:
                        # Get average color
                        roi = region[y:y+ch, x:x+cw]
                        mean_color = cv2.mean(roi)[:3]
                        indicators.append({
                            'bbox': (x, y, cw, ch),
                            'color_bgr': tuple(int(c) for c in mean_color),
                            'color_name': color_name,
                            'type': 'line'
                        })

                    # Box indicator
                    elif 0.5 < aspect_ratio < 2 and 10 < cw < 50:
                        roi = region[y:y+ch, x:x+cw]
                        mean_color = cv2.mean(roi)[:3]
                        indicators.append({
                            'bbox': (x, y, cw, ch),
                            'color_bgr': tuple(int(c) for c in mean_color),
                            'color_name': color_name,
                            'type': 'box'
                        })

        return indicators

    def _ocr_region(self, region: np.ndarray) -> List[Dict]:
        """OCR the legend region."""
        results = []

        if self.tesseract_available:
            try:
                # Preprocess
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

                # Get text with bounding boxes
                data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

                for i in range(len(data['text'])):
                    text = data['text'][i].strip()
                    conf = int(data['conf'][i]) if data['conf'][i] != '-1' else 0

                    if text and len(text) > 1 and conf > 30:
                        x = data['left'][i]
                        y = data['top'][i]
                        w = data['width'][i]
                        h = data['height'][i]

                        results.append({
                            'text': text,
                            'bbox': (x, y, w, h),
                            'confidence': conf / 100.0
                        })

            except Exception as e:
                logger.debug(f"Tesseract OCR failed: {e}")

        elif self.easyocr_available:
            try:
                reader = self._get_easyocr_reader()
                ocr_results = reader.readtext(region)

                for bbox, text, conf in ocr_results:
                    if text and len(text) > 1:
                        # Convert bbox to (x, y, w, h)
                        x_coords = [p[0] for p in bbox]
                        y_coords = [p[1] for p in bbox]
                        x = int(min(x_coords))
                        y = int(min(y_coords))
                        w = int(max(x_coords) - x)
                        h = int(max(y_coords) - y)

                        results.append({
                            'text': text,
                            'bbox': (x, y, w, h),
                            'confidence': conf
                        })

            except Exception as e:
                logger.debug(f"EasyOCR failed: {e}")

        return results

    def _classify_arms(self, entries: List[LegendEntry]) -> Tuple[Optional[LegendEntry], Optional[LegendEntry]]:
        """
        Classify legend entries as treatment vs control.

        Uses keyword matching on labels to identify arm type.
        """
        treatment = None
        control = None

        treatment_scores = []
        control_scores = []

        for entry in entries:
            label_lower = entry.label.lower()

            # Score for treatment
            t_score = 0
            for keyword in self.treatment_keywords:
                if keyword in label_lower:
                    t_score += 1

            # Score for control
            c_score = 0
            for keyword in self.control_keywords:
                if keyword in label_lower:
                    c_score += 1

            treatment_scores.append((entry, t_score))
            control_scores.append((entry, c_score))

        # Find best treatment match
        treatment_scores.sort(key=lambda x: x[1], reverse=True)
        if treatment_scores and treatment_scores[0][1] > 0:
            treatment = treatment_scores[0][0]
            treatment.is_treatment = True

        # Find best control match (excluding treatment)
        control_scores.sort(key=lambda x: x[1], reverse=True)
        for entry, score in control_scores:
            if entry != treatment and score > 0:
                control = entry
                control.is_treatment = False
                break

        # If we couldn't classify, use heuristics
        if not treatment and not control and len(entries) >= 2:
            # Assume first entry is treatment, second is control
            # (common convention in medical literature)
            entries[0].is_treatment = True
            treatment = entries[0]
            entries[1].is_treatment = False
            control = entries[1]

        return treatment, control

    def _calculate_confidence(self, entries: List[LegendEntry],
                               treatment: Optional[LegendEntry],
                               control: Optional[LegendEntry]) -> float:
        """Calculate overall confidence in legend extraction."""
        if not entries:
            return 0.0

        confidence = 0.3  # Base confidence for finding any entries

        # Add confidence for each entry
        entry_conf = sum(e.confidence for e in entries) / len(entries)
        confidence += 0.2 * entry_conf

        # Add confidence for classification
        if treatment and control:
            confidence += 0.3  # Both arms identified
        elif treatment or control:
            confidence += 0.15  # One arm identified

        # Add confidence for keyword matches
        if treatment:
            label_lower = treatment.label.lower()
            if any(kw in label_lower for kw in self.treatment_keywords):
                confidence += 0.1

        if control:
            label_lower = control.label.lower()
            if any(kw in label_lower for kw in self.control_keywords):
                confidence += 0.1

        return min(1.0, confidence)


def extract_legend(image: np.ndarray,
                   plot_region: Optional[Tuple[int, int, int, int]] = None) -> LegendInfo:
    """
    Convenience function to extract legend from KM curve image.

    Parameters
    ----------
    image : np.ndarray
        KM figure image (BGR)
    plot_region : tuple, optional
        Plot region (x, y, w, h)

    Returns
    -------
    LegendInfo
    """
    extractor = LegendExtractor()
    return extractor.extract(image, plot_region)


if __name__ == "__main__":
    print("Legend Extractor loaded successfully")
    print(f"Tesseract available: {TESSERACT_AVAILABLE}")
    print(f"EasyOCR available: {EASYOCR_AVAILABLE}")

    # Create a simple test image
    test_img = np.ones((400, 600, 3), dtype=np.uint8) * 255

    # Add legend entries
    # Blue line
    cv2.line(test_img, (400, 50), (450, 50), (255, 100, 0), 3)
    cv2.putText(test_img, "Treatment", (460, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Red line
    cv2.line(test_img, (400, 80), (450, 80), (0, 0, 255), 3)
    cv2.putText(test_img, "Control", (460, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    extractor = LegendExtractor()
    result = extractor.extract(test_img)

    print(f"\nDetected: {result.detected}")
    print(f"Entries: {len(result.entries)}")
    print(f"Treatment: {result.treatment_arm.label if result.treatment_arm else 'None'}")
    print(f"Control: {result.control_arm.label if result.control_arm else 'None'}")
    print(f"Confidence: {result.confidence:.2f}")
