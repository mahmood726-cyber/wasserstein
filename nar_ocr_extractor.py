# sentinel:skip-file — hardcoded paths / templated placeholders are fixture/registry/audit-narrative data for this repo's research workflow, not portable application configuration. Same pattern as push_all_repos.py and E156 workbook files.
"""NAR (Number at Risk) OCR extractor for KM curve figures.

Reads number-at-risk tables from the bottom region of KM figure images
using pytesseract OCR. Returns structured NARRow data that integrates
with the Guyot IPD reconstruction algorithm.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Common OCR misreads for digits
OCR_DIGIT_FIXES = {
    'O': '0', 'o': '0', 'Q': '0',
    'l': '1', 'I': '1', '|': '1',
    'S': '5', 's': '5',
    'B': '8',
    'Z': '2', 'z': '2',
    'g': '9',
}


@dataclass
class NARRow:
    """A single row from a Number-at-Risk table.

    Attributes:
        label: Arm label (e.g., "RF", "CB", "Treatment", "Control").
        timepoints: Time values corresponding to each count (e.g., [0, 6, 12, 18]).
        values: Patient counts at each timepoint (e.g., [147, 130, 118]).
    """
    label: str
    timepoints: List[float] = field(default_factory=list)
    values: List[int] = field(default_factory=list)


@dataclass
class NAROCRResult:
    """Result from NAR OCR extraction.

    Attributes:
        rows: List of NARRow, one per arm/group in the table.
    """
    rows: List[NARRow] = field(default_factory=list)


class NAROCRExtractor:
    """Extract Number-at-Risk tables from KM figure images using OCR.

    Crops the bottom portion of the image, binarizes it, detects
    text rows via horizontal projection, and OCRs each row to
    extract arm labels and patient counts.
    """

    def __init__(self, bottom_fraction: float = 0.30,
                 max_n: int = 10000):
        self.bottom_fraction = bottom_fraction
        self.max_n = max_n

    def extract_nar(self, image: np.ndarray) -> Optional[NAROCRResult]:
        """Extract NAR table from image.

        Args:
            image: BGR or grayscale image (numpy array).

        Returns:
            NAROCRResult with rows, or None if extraction fails.
        """
        if image is None or image.size == 0:
            return None

        h, w = image.shape[:2]
        y_start = int(h * (1.0 - self.bottom_fraction))
        crop = image[y_start:, :]

        # Convert to grayscale
        if len(crop.shape) == 3:
            import cv2
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop.copy()

        # Binarize with Otsu
        gray = _preprocess(gray)

        # Detect text row regions via horizontal projection
        row_regions = self._detect_row_regions(gray)
        if len(row_regions) < 1:
            logger.debug("NAR OCR: no text rows detected")
            return None

        # OCR each row
        rows = []
        for y1, y2 in row_regions:
            row_img = gray[y1:y2, :]
            row_data = self._ocr_single_row(row_img)
            if row_data is not None:
                rows.append(row_data)

        if not rows:
            logger.debug("NAR OCR: no valid rows parsed")
            return None

        # Assign timepoints if not already set
        self._assign_timepoints(rows)

        # Validate rows
        valid_rows = [r for r in rows if self._validate_row(r)]
        if not valid_rows:
            logger.debug("NAR OCR: no rows passed validation")
            return None

        logger.debug(f"NAR OCR: extracted {len(valid_rows)} valid rows")
        return NAROCRResult(rows=valid_rows)

    def _detect_row_regions(self, gray: np.ndarray) -> List[tuple]:
        """Find horizontal text row regions using projection profile."""
        h, w = gray.shape[:2]
        if h < 10 or w < 10:
            return []

        # Invert: text pixels become white
        binary_inv = 255 - gray

        # Horizontal projection
        proj = np.sum(binary_inv, axis=1).astype(float)

        # Smooth
        kernel_size = max(3, h // 30)
        if kernel_size % 2 == 0:
            kernel_size += 1
        try:
            from scipy.ndimage import uniform_filter1d
            proj = uniform_filter1d(proj, size=kernel_size)
        except ImportError:
            pass

        # Threshold: rows with significant text
        threshold = np.mean(proj) + 0.3 * np.std(proj)
        above = proj > threshold

        # Find contiguous regions
        regions = []
        in_region = False
        start = 0
        for i in range(len(above)):
            if above[i] and not in_region:
                start = i
                in_region = True
            elif not above[i] and in_region:
                if i - start >= 5:  # minimum row height
                    regions.append((max(0, start - 2), min(h, i + 2)))
                in_region = False
        if in_region and len(above) - start >= 5:
            regions.append((max(0, start - 2), h))

        # Skip the first region if it looks like axis labels or title
        # (NAR rows are typically the 2nd+ text rows below the axis)
        if len(regions) > 2:
            # Check if first region is a header ("Number at risk", etc.)
            first_h = regions[0][1] - regions[0][0]
            others_avg_h = np.mean([r[1] - r[0] for r in regions[1:]])
            if first_h > 1.5 * others_avg_h:
                regions = regions[1:]

        return regions

    def _ocr_single_row(self, row_img: np.ndarray) -> Optional[NARRow]:
        """OCR a single row image and parse label + values."""
        try:
            import pytesseract
            text = pytesseract.image_to_string(
                row_img, config='--psm 7 --oem 3'
            ).strip()
        except Exception as e:
            logger.debug(f"NAR row OCR error: {e}")
            return None

        if not text or len(text) < 3:
            return None

        # Apply OCR digit fixes
        text_fixed = self._fix_ocr_digits(text)

        # Parse: first non-numeric token(s) = label, rest = numbers
        return self._parse_row_text(text_fixed)

    def _fix_ocr_digits(self, text: str) -> str:
        """Apply common OCR digit corrections.

        Only fixes characters that appear within numeric contexts
        (adjacent to digits or in number sequences).
        """
        # Fix isolated OCR misreads within number sequences
        tokens = text.split()
        fixed_tokens = []
        for token in tokens:
            # If token looks mostly numeric, apply digit fixes
            digit_count = sum(1 for c in token if c.isdigit())
            if digit_count > 0 and digit_count >= len(token) * 0.5:
                fixed = []
                for c in token:
                    if c in OCR_DIGIT_FIXES:
                        fixed.append(OCR_DIGIT_FIXES[c])
                    else:
                        fixed.append(c)
                fixed_tokens.append(''.join(fixed))
            else:
                fixed_tokens.append(token)
        return ' '.join(fixed_tokens)

    def _parse_row_text(self, text: str) -> Optional[NARRow]:
        """Parse OCR text into label + integer values.

        Expected format: "Label  123  110  95  82  70"
        or: "123  110  95  82  70" (no label)
        """
        # Split into tokens
        tokens = re.split(r'\s+', text.strip())
        if not tokens:
            return None

        label_parts = []
        values = []

        for token in tokens:
            # Try to parse as integer
            cleaned = re.sub(r'[,.](\d{3})', r'\1', token)  # remove thousands sep
            cleaned = re.sub(r'[^0-9-]', '', cleaned)
            if cleaned and cleaned not in ('-', ''):
                try:
                    val = int(cleaned)
                    if 0 <= val <= self.max_n:
                        values.append(val)
                        continue
                except ValueError:
                    pass

            # Not a valid number — part of label
            if not values:  # only accept label tokens before numbers
                label_parts.append(token)

        label = ' '.join(label_parts).strip() if label_parts else "Arm"

        if len(values) < 2:
            return None

        return NARRow(label=label, timepoints=[], values=values)

    def _assign_timepoints(self, rows: List[NARRow]):
        """Assign timepoints to rows that don't have them.

        Uses equal spacing from 0 based on the number of values.
        This is a heuristic; proper alignment requires x-axis calibration.
        """
        for row in rows:
            if not row.timepoints and row.values:
                n = len(row.values)
                # Default: assume equal spacing, total span unknown
                # Use index as placeholder (0, 1, 2, ...)
                row.timepoints = list(range(n))

    def _validate_row(self, row: NARRow) -> bool:
        """Validate a NAR row for plausibility."""
        if len(row.values) < 2:
            return False

        # All values must be non-negative integers in range
        if any(v < 0 or v > self.max_n for v in row.values):
            return False

        # First value (baseline N) should be reasonable
        if row.values[0] < 1:
            return False

        # Values should be generally non-increasing (allow small increases
        # from OCR errors: up to 10% of the first value)
        max_increase = max(1, int(row.values[0] * 0.1))
        for i in range(1, len(row.values)):
            if row.values[i] > row.values[i - 1] + max_increase:
                logger.debug(
                    f"NAR row '{row.label}': value {row.values[i]} > "
                    f"{row.values[i-1]} + {max_increase} at index {i}")
                return False

        return True


def _preprocess(gray: np.ndarray) -> np.ndarray:
    """Preprocess grayscale image for OCR: Otsu binarization."""
    try:
        import cv2
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    except ImportError:
        threshold = int(np.mean(gray))
        return np.where(gray > threshold, 255, 0).astype(np.uint8)
