"""NAR (Number at Risk) table detector for KM curve figures.

Detects whether a KM figure image contains a Number-at-Risk table,
typically located below the x-axis. Uses OCR keyword search and
structural analysis (horizontal text row detection via projection).
"""

import logging
from enum import Enum
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Keywords that indicate a NAR table (case-insensitive)
NAR_KEYWORDS = [
    "at risk", "no.", "number at risk", "patients at risk",
    "n at risk", "no at risk", "subjects at risk",
    "numbers at risk", "no. at risk",
]


class DetectionOutcome(Enum):
    TABLE_FOUND = "table_found"
    NO_TABLE = "no_table"


@dataclass
class DetectionResult:
    outcome: DetectionOutcome
    confidence: float = 0.0
    region_y_start: Optional[int] = None  # top of NAR region in pixels
    n_text_rows: int = 0


class TruthCertNARDetector:
    """Detect Number-at-Risk tables in KM figure images.

    Crops the bottom portion of the image (where NAR tables typically
    appear below the x-axis) and uses OCR + structural analysis to
    determine if a table is present.
    """

    def __init__(self, bottom_fraction: float = 0.35,
                 min_text_rows: int = 2):
        self.bottom_fraction = bottom_fraction
        self.min_text_rows = min_text_rows

    def detect(self, image: np.ndarray) -> DetectionResult:
        """Detect whether image contains a NAR table.

        Args:
            image: BGR or grayscale image (numpy array).

        Returns:
            DetectionResult with outcome and metadata.
        """
        if image is None or image.size == 0:
            return DetectionResult(outcome=DetectionOutcome.NO_TABLE)

        h, w = image.shape[:2]
        y_start = int(h * (1.0 - self.bottom_fraction))
        crop = image[y_start:, :]

        # Convert to grayscale if needed
        if len(crop.shape) == 3:
            import cv2
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop

        # --- Method 1: OCR keyword search ---
        keyword_found = self._check_keywords_ocr(gray)

        # --- Method 2: Structural analysis (horizontal text rows) ---
        n_rows = self._count_text_rows(gray)

        if keyword_found:
            return DetectionResult(
                outcome=DetectionOutcome.TABLE_FOUND,
                confidence=0.9,
                region_y_start=y_start,
                n_text_rows=n_rows,
            )

        if n_rows >= self.min_text_rows:
            return DetectionResult(
                outcome=DetectionOutcome.TABLE_FOUND,
                confidence=0.6,
                region_y_start=y_start,
                n_text_rows=n_rows,
            )

        return DetectionResult(outcome=DetectionOutcome.NO_TABLE)

    def _check_keywords_ocr(self, gray: np.ndarray) -> bool:
        """OCR the bottom region and search for NAR keywords."""
        try:
            import pytesseract
            text = pytesseract.image_to_string(
                gray, config='--psm 6 --oem 3'
            ).lower()
            for kw in NAR_KEYWORDS:
                if kw in text:
                    logger.debug(f"NAR keyword found: '{kw}'")
                    return True
        except Exception as e:
            logger.debug(f"NAR OCR keyword check failed: {e}")
        return False

    def _count_text_rows(self, gray: np.ndarray) -> int:
        """Count horizontal text rows using projection profile.

        Projects pixel intensities horizontally and counts peaks
        (dark text rows) in the projection.
        """
        h, w = gray.shape[:2]
        if h < 10 or w < 10:
            return 0

        # Binarize: dark text on light background
        _, binary = _threshold_otsu(gray)
        # Invert so text pixels = 1
        binary_inv = 255 - binary

        # Horizontal projection: sum of white (text) pixels per row
        proj = np.sum(binary_inv, axis=1).astype(float)

        # Smooth projection to reduce noise
        kernel_size = max(3, h // 40)
        if kernel_size % 2 == 0:
            kernel_size += 1
        from scipy.ndimage import uniform_filter1d
        proj_smooth = uniform_filter1d(proj, size=kernel_size)

        # Find peaks: rows with significant text content
        threshold = np.mean(proj_smooth) + 0.5 * np.std(proj_smooth)
        above = proj_smooth > threshold

        # Count transitions from below to above threshold (row starts)
        transitions = np.diff(above.astype(int))
        n_rows = np.sum(transitions == 1)

        logger.debug(f"NAR structural: {n_rows} text rows detected")
        return n_rows


def _threshold_otsu(gray: np.ndarray):
    """Otsu thresholding (cv2-free fallback)."""
    try:
        import cv2
        return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    except ImportError:
        # Manual Otsu fallback
        threshold = int(np.mean(gray))
        binary = np.where(gray > threshold, 255, 0).astype(np.uint8)
        return threshold, binary
