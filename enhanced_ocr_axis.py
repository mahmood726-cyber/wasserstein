#!/usr/bin/env python3
"""
Enhanced OCR Axis Detection
===========================

Advanced axis detection using Tesseract OCR with improved algorithms:
1. Multi-region analysis - scan axis regions with multiple strategies
2. Adaptive thresholding - handle various image qualities
3. Label detection - identify "Time", "Survival", "Months", etc.
4. Scale inference - detect linear vs log scales
5. Confidence scoring - report reliability of detection

Author: Wasserstein KM Extractor Project
Date: January 2026
"""

import cv2
import numpy as np
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import pytesseract
try:
    import pytesseract
    TESSERACT_AVAILABLE = True

    # Common Windows Tesseract paths
    common_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Users\user\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
    ]
    for path in common_paths:
        if Path(path).exists():
            pytesseract.pytesseract.tesseract_cmd = path
            break
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("pytesseract not installed")

# Try to import EasyOCR
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.debug("easyocr not installed")


@dataclass
class AxisDetectionResult:
    """Result of axis detection"""
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    x_label: str = "Time"
    y_label: str = "Survival"
    x_unit: str = ""
    y_unit: str = ""
    x_ticks: List[float] = field(default_factory=list)
    y_ticks: List[float] = field(default_factory=list)
    x_scale: str = "linear"  # "linear" or "log"
    y_scale: str = "linear"
    confidence: float = 0.0
    method: str = "unknown"
    plot_region: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h
    calibrated: bool = False  # True if actual axis values detected, False if normalized
    strategies_tried: List[str] = field(default_factory=list)


@dataclass
class CalibratedAxisInfo:
    """Calibrated axis information for coordinate conversion."""
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    x_unit: str = "months"
    calibrated: bool = False
    confidence: float = 0.0

    def pixel_to_time(self, x_pixel: int, plot_x_min: int, plot_x_max: int) -> float:
        """Convert x pixel to calibrated time value."""
        if not self.calibrated:
            return (x_pixel - plot_x_min) / max(1, plot_x_max - plot_x_min)
        plot_width = max(1, plot_x_max - plot_x_min)
        normalized = (x_pixel - plot_x_min) / plot_width
        return self.x_min + normalized * (self.x_max - self.x_min)

    def pixel_to_survival(self, y_pixel: int, plot_y_min: int, plot_y_max: int) -> float:
        """Convert y pixel to calibrated survival value."""
        plot_height = max(1, plot_y_max - plot_y_min)
        normalized = 1.0 - (y_pixel - plot_y_min) / plot_height
        if not self.calibrated:
            return max(0, min(1, normalized))
        return self.y_min + normalized * (self.y_max - self.y_min)


class EnhancedAxisDetector:
    """
    Advanced axis detection for KM curves using OCR and image analysis.

    Multi-strategy detection for robust axis calibration:
    1. Tesseract OCR with preprocessing
    2. EasyOCR as fallback
    3. Tick mark interpolation
    4. NAR timepoint integration (if available)
    5. Heuristic estimation from image content
    """

    def __init__(self):
        self.tesseract_available = TESSERACT_AVAILABLE
        self.easyocr_available = EASYOCR_AVAILABLE
        self._easyocr_reader = None

        # Common axis labels for KM curves
        self.x_labels = [
            'time', 'months', 'years', 'days', 'weeks',
            'follow-up', 'survival time', 'pfs', 'os'
        ]
        self.y_labels = [
            'survival', 'probability', 'proportion', 'cumulative',
            'event-free', 'pfs', 'os', 'disease-free', 'overall survival'
        ]

        # Common x-axis max values for clinical trials (months)
        self.common_x_max_values = [6, 12, 18, 24, 30, 36, 48, 60, 72, 84, 96, 100, 120]

    def _get_easyocr_reader(self):
        """Lazy-load EasyOCR reader."""
        if self._easyocr_reader is None and self.easyocr_available:
            self._easyocr_reader = easyocr.Reader(['en'], gpu=False)
        return self._easyocr_reader

    def detect_axes(self, image_path: str) -> AxisDetectionResult:
        """
        Main function to detect axis information from KM curve image.

        Args:
            image_path: Path to the image file

        Returns:
            AxisDetectionResult with detected axis information
        """
        logger.info(f"Detecting axes from: {image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return AxisDetectionResult(confidence=0.0, method="failed")

        result = AxisDetectionResult()
        confidence_scores = []

        # Step 1: Detect plot region
        plot_region = self._detect_plot_region(image)
        result.plot_region = plot_region
        logger.info(f"Plot region: {plot_region}")

        # Step 2: Extract and analyze axis regions
        y_region = self._extract_y_axis_region(image, plot_region)
        x_region = self._extract_x_axis_region(image, plot_region)

        # Step 3: OCR on axis regions
        if self.tesseract_available:
            # Y-axis analysis
            y_result = self._analyze_y_axis(y_region)
            result.y_min = y_result['min']
            result.y_max = y_result['max']
            result.y_ticks = y_result['ticks']
            result.y_label = y_result['label']
            confidence_scores.append(y_result['confidence'])
            logger.info(f"Y-axis: {result.y_min} to {result.y_max}")

            # X-axis analysis
            x_result = self._analyze_x_axis(x_region)
            result.x_min = x_result['min']
            result.x_max = x_result['max']
            result.x_ticks = x_result['ticks']
            result.x_label = x_result['label']
            result.x_unit = x_result.get('unit', '')
            confidence_scores.append(x_result['confidence'])
            logger.info(f"X-axis: {result.x_min} to {result.x_max}")

            # Step 4: Detect scale type
            result.x_scale = self._detect_scale_type(result.x_ticks)
            result.y_scale = "linear"  # Survival is always linear

        # Step 5: Fallback using image analysis if OCR failed
        if not result.x_ticks and not result.y_ticks:
            heuristic_result = self._heuristic_axis_detection(image)
            result.x_max = heuristic_result.get('x_max', 1.0)
            result.y_max = 1.0
            confidence_scores.append(0.4)
            result.method = "heuristic"
        else:
            result.method = "ocr"

        # Calculate overall confidence
        if confidence_scores:
            result.confidence = sum(confidence_scores) / len(confidence_scores)
        else:
            result.confidence = 0.3

        return result

    def calibrated_axis_detection(self, image: np.ndarray,
                                   nar_timepoints: Optional[List[float]] = None,
                                   caption: Optional[str] = None) -> CalibratedAxisInfo:
        """
        Multi-strategy axis detection with validation.

        Tries multiple strategies in order of reliability:
        1. Caption-based extraction (if caption provided)
        2. Tesseract OCR with preprocessing
        3. EasyOCR as fallback
        4. Tick mark interpolation
        5. NAR timepoints integration
        6. Heuristic estimation

        Parameters
        ----------
        image : np.ndarray
            KM curve image (BGR)
        nar_timepoints : list, optional
            NAR table timepoints if available (e.g., [0, 12, 24, 36])
        caption : str, optional
            Figure caption text

        Returns
        -------
        CalibratedAxisInfo
        """
        strategies_tried = []
        best_result = CalibratedAxisInfo()
        best_confidence = 0.0

        # Detect plot region first
        plot_region = self._detect_plot_region(image)
        x_region = self._extract_x_axis_region(image, plot_region)

        # Strategy 0: Caption-based extraction (if caption provided)
        if caption:
            strategies_tried.append('caption')
            caption_result = self._extract_from_caption(caption)
            if caption_result:
                caption_conf = caption_result.get('confidence', 0.6)
                if caption_conf > best_confidence:
                    best_result = CalibratedAxisInfo(
                        x_min=0.0,
                        x_max=caption_result['x_max'],
                        y_min=0.0,
                        y_max=1.0,
                        x_unit=caption_result.get('x_unit', 'months'),
                        calibrated=True,
                        confidence=caption_conf
                    )
                    best_confidence = caption_conf
                    logger.info(f"Caption calibration: x_max={caption_result['x_max']}, "
                               f"source={caption_result.get('source_type', 'unknown')}, "
                               f"conf={caption_conf:.2f}")

        # Strategy 1: Tesseract OCR with preprocessing
        if self.tesseract_available:
            strategies_tried.append('tesseract')
            try:
                result = self._ocr_with_tesseract(x_region)
                if result['confidence'] > best_confidence and result['ticks']:
                    best_result = CalibratedAxisInfo(
                        x_min=result['min'],
                        x_max=result['max'],
                        y_min=0.0,
                        y_max=1.0,
                        x_unit=result.get('unit', 'months'),
                        calibrated=True,
                        confidence=result['confidence']
                    )
                    best_confidence = result['confidence']
                    logger.info(f"Tesseract calibration: x_max={result['max']}, conf={result['confidence']:.2f}")
            except Exception as e:
                logger.debug(f"Tesseract failed: {e}")

        # Strategy 2: EasyOCR as fallback
        if self.easyocr_available and best_confidence < 0.7:
            strategies_tried.append('easyocr')
            try:
                result = self._ocr_with_easyocr(x_region)
                if result['confidence'] > best_confidence and result['ticks']:
                    best_result = CalibratedAxisInfo(
                        x_min=result['min'],
                        x_max=result['max'],
                        y_min=0.0,
                        y_max=1.0,
                        x_unit=result.get('unit', 'months'),
                        calibrated=True,
                        confidence=result['confidence']
                    )
                    best_confidence = result['confidence']
                    logger.info(f"EasyOCR calibration: x_max={result['max']}, conf={result['confidence']:.2f}")
            except Exception as e:
                logger.debug(f"EasyOCR failed: {e}")

        # Strategy 3: Tick mark interpolation
        if best_confidence < 0.6:
            strategies_tried.append('tick_interpolation')
            try:
                result = self._tick_mark_interpolation(image, plot_region)
                if result['confidence'] > best_confidence:
                    best_result = CalibratedAxisInfo(
                        x_min=0.0,
                        x_max=result['x_max'],
                        y_min=0.0,
                        y_max=1.0,
                        x_unit='months',
                        calibrated=True,
                        confidence=result['confidence']
                    )
                    best_confidence = result['confidence']
                    logger.info(f"Tick interpolation: x_max={result['x_max']}, conf={result['confidence']:.2f}")
            except Exception as e:
                logger.debug(f"Tick interpolation failed: {e}")

        # Strategy 4: Use NAR timepoints if available
        if nar_timepoints and best_confidence < 0.8:
            strategies_tried.append('nar_timepoints')
            if len(nar_timepoints) >= 2:
                x_max = max(nar_timepoints)
                nar_confidence = min(0.9, 0.5 + 0.1 * len(nar_timepoints))
                if nar_confidence > best_confidence:
                    best_result = CalibratedAxisInfo(
                        x_min=min(nar_timepoints),
                        x_max=x_max,
                        y_min=0.0,
                        y_max=1.0,
                        x_unit='months',
                        calibrated=True,
                        confidence=nar_confidence
                    )
                    best_confidence = nar_confidence
                    logger.info(f"NAR timepoints calibration: x_max={x_max}, conf={nar_confidence:.2f}")

        # Strategy 5: Heuristic estimation from image content
        if best_confidence < 0.5:
            strategies_tried.append('heuristic')
            heuristic_result = self._heuristic_axis_detection(image)
            heur_confidence = 0.4
            if heur_confidence > best_confidence:
                best_result = CalibratedAxisInfo(
                    x_min=0.0,
                    x_max=heuristic_result.get('x_max', 24),
                    y_min=0.0,
                    y_max=1.0,
                    x_unit='months',
                    calibrated=False,  # Mark as uncalibrated
                    confidence=heur_confidence
                )
                best_confidence = heur_confidence
                logger.info(f"Heuristic calibration: x_max={heuristic_result.get('x_max', 24)}")

        # Validate: x_max should be reasonable for clinical trials
        if best_result.x_max > 200 or best_result.x_max < 1:
            logger.warning(f"Unusual x_max detected: {best_result.x_max}, using closest common value")
            best_result.x_max = min(self.common_x_max_values,
                                    key=lambda x: abs(x - best_result.x_max))
            best_result.confidence *= 0.7

        # Final validation: cross-check with NAR timepoints if available
        if nar_timepoints and len(nar_timepoints) >= 2:
            best_result = self._validate_with_nar(best_result, nar_timepoints)
            strategies_tried.append('nar_validation')

        logger.debug(f"Calibration strategies tried: {strategies_tried}")
        return best_result

    def _ocr_with_tesseract(self, region: np.ndarray) -> Dict:
        """OCR using Tesseract with preprocessing."""
        result = {
            'min': 0.0,
            'max': 1.0,
            'ticks': [],
            'unit': 'months',
            'confidence': 0.0
        }

        if region.size == 0:
            return result

        # Preprocess
        processed = self._preprocess_for_ocr(region)

        # Try multiple PSM modes for robustness
        configs = [
            '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.',
            '--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.',
            '--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789.'
        ]

        all_numbers = []
        for config in configs:
            try:
                text = pytesseract.image_to_string(processed, config=config)
                numbers = self._extract_numbers(text, axis='x')
                all_numbers.extend(numbers)
            except Exception:
                continue

        if all_numbers:
            # Remove duplicates and sort
            all_numbers = sorted(set(all_numbers))

            # Filter for reasonable clinical values
            clinical_numbers = [n for n in all_numbers if 0 <= n <= 150]

            if clinical_numbers:
                result['ticks'] = clinical_numbers
                result['min'] = min(clinical_numbers)
                result['max'] = max(clinical_numbers)
                result['confidence'] = min(0.85, 0.4 + 0.1 * len(clinical_numbers))

        return result

    def _ocr_with_easyocr(self, region: np.ndarray) -> Dict:
        """OCR using EasyOCR."""
        result = {
            'min': 0.0,
            'max': 1.0,
            'ticks': [],
            'unit': 'months',
            'confidence': 0.0
        }

        if region.size == 0:
            return result

        reader = self._get_easyocr_reader()
        if reader is None:
            return result

        try:
            # EasyOCR detection
            ocr_results = reader.readtext(region, detail=1)

            numbers = []
            confidences = []
            for bbox, text, conf in ocr_results:
                # Extract numbers from text
                nums = re.findall(r'\d+\.?\d*', text)
                for num_str in nums:
                    try:
                        num = float(num_str)
                        if 0 <= num <= 150:  # Reasonable clinical range
                            numbers.append(num)
                            confidences.append(conf)
                    except ValueError:
                        continue

            if numbers:
                result['ticks'] = sorted(set(numbers))
                result['min'] = min(numbers)
                result['max'] = max(numbers)
                result['confidence'] = np.mean(confidences) if confidences else 0.5

        except Exception as e:
            logger.debug(f"EasyOCR error: {e}")

        return result

    def _tick_mark_interpolation(self, image: np.ndarray,
                                  plot_region: Tuple[int, int, int, int]) -> Dict:
        """Detect tick marks and interpolate axis values."""
        result = {
            'x_max': 24,
            'confidence': 0.0
        }

        h, w = image.shape[:2]
        px, py, pw, ph = plot_region

        # Extract bottom edge where tick marks would be
        bottom_region = image[py + ph - 20:py + ph + 30, px:px + pw]
        if bottom_region.size == 0:
            return result

        gray = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)

        # Detect vertical lines (tick marks)
        edges = cv2.Canny(gray, 50, 150)

        # Find vertical line positions
        vertical_kernel = np.ones((15, 1), np.uint8)
        v_lines = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, vertical_kernel)

        # Project to find tick positions
        v_proj = np.sum(v_lines, axis=0)
        threshold = np.max(v_proj) * 0.3

        tick_positions = []
        in_tick = False
        tick_start = 0
        for x, val in enumerate(v_proj):
            if val > threshold and not in_tick:
                tick_start = x
                in_tick = True
            elif val <= threshold and in_tick:
                tick_positions.append((tick_start + x) // 2)
                in_tick = False

        # Estimate x_max from number of ticks
        n_ticks = len(tick_positions)
        if n_ticks >= 3:
            # Assume regular spacing and common clinical values
            # Common patterns: 0,12,24,36 (4 ticks), 0,6,12,18,24 (5 ticks)
            tick_interval_estimates = {
                4: [12, 24, 36],  # 3 intervals
                5: [6, 12, 18, 24, 30],  # 4 intervals
                6: [6, 12, 18, 24, 30],  # 5 intervals
                7: [10, 20, 30, 40, 50, 60],  # 6 intervals
            }

            if n_ticks in tick_interval_estimates:
                # Use middle estimate
                estimated_max = tick_interval_estimates[n_ticks][len(tick_interval_estimates[n_ticks])//2]
            else:
                # Rough estimate: assume 10-12 months per tick
                estimated_max = (n_ticks - 1) * 12

            # Find closest common value
            result['x_max'] = min(self.common_x_max_values,
                                  key=lambda x: abs(x - estimated_max))
            result['confidence'] = min(0.6, 0.3 + 0.05 * n_ticks)

        return result

    def _detect_plot_region(self, image: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Detect the plot area within the image.

        Returns (x, y, width, height) of the plot region.
        """
        h, w = image.shape[:2]

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect edges
        edges = cv2.Canny(gray, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find largest rectangular contour (likely the plot area)
            largest_area = 0
            best_rect = (0, 0, w, h)

            for contour in contours:
                x, y, cw, ch = cv2.boundingRect(contour)
                area = cw * ch

                # Filter: should be reasonably sized and roughly rectangular
                if area > largest_area and cw > w * 0.3 and ch > h * 0.3:
                    largest_area = area
                    best_rect = (x, y, cw, ch)

            return best_rect

        # Default: estimate based on typical KM plot layout
        # Usually: 15% left margin, 10% right, 15% bottom, 10% top
        x = int(w * 0.15)
        y = int(h * 0.10)
        plot_w = int(w * 0.75)
        plot_h = int(h * 0.75)

        return (x, y, plot_w, plot_h)

    def _extract_y_axis_region(
        self,
        image: np.ndarray,
        plot_region: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """Extract the Y-axis region (left of plot)"""
        h, w = image.shape[:2]
        px, py, pw, ph = plot_region

        # Y-axis is to the left of the plot
        x_start = 0
        x_end = px
        y_start = py
        y_end = py + ph

        # Ensure valid bounds
        x_end = max(x_end, 1)
        y_end = min(y_end, h)

        return image[y_start:y_end, x_start:x_end]

    def _extract_x_axis_region(
        self,
        image: np.ndarray,
        plot_region: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """Extract the X-axis region (below plot)"""
        h, w = image.shape[:2]
        px, py, pw, ph = plot_region

        # X-axis is below the plot
        x_start = px
        x_end = px + pw
        y_start = py + ph
        y_end = h

        # Ensure valid bounds
        x_end = min(x_end, w)
        y_end = min(y_end, h)

        return image[y_start:y_end, x_start:x_end]

    def _preprocess_for_ocr(self, region: np.ndarray) -> np.ndarray:
        """Preprocess image region for better OCR"""
        if region.size == 0:
            return region

        # Convert to grayscale
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region

        # Increase contrast
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # Denoise
        thresh = cv2.medianBlur(thresh, 3)

        # Scale up for better OCR
        scale = 2
        thresh = cv2.resize(thresh, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        return thresh

    def _analyze_y_axis(self, region: np.ndarray) -> Dict:
        """Analyze Y-axis region to extract ticks and labels"""
        result = {
            'min': 0.0,
            'max': 1.0,
            'ticks': [],
            'label': 'Survival',
            'confidence': 0.5
        }

        if region.size == 0 or not self.tesseract_available:
            return result

        # Preprocess
        processed = self._preprocess_for_ocr(region)

        # OCR configuration for numbers
        config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.%'

        try:
            text = pytesseract.image_to_string(processed, config=config)
            numbers = self._extract_numbers(text, axis='y')

            if numbers:
                result['ticks'] = sorted(numbers)
                result['min'] = min(numbers)
                result['max'] = max(numbers)
                result['confidence'] = 0.8

                # Validate: Y-axis should be 0-1 or 0-100 for survival
                if result['max'] > 1.5 and result['max'] <= 100:
                    # Percentage scale - normalize
                    result['ticks'] = [t / 100 for t in result['ticks']]
                    result['min'] /= 100
                    result['max'] /= 100

                # Force valid range for survival
                result['min'] = max(0, result['min'])
                result['max'] = min(1, max(result['max'], 1))

        except Exception as e:
            logger.warning(f"Y-axis OCR failed: {e}")

        # Try to detect label
        try:
            config_text = '--oem 3 --psm 6'
            text = pytesseract.image_to_string(region, config=config_text)
            for label in self.y_labels:
                if label.lower() in text.lower():
                    result['label'] = label.title()
                    break
        except (pytesseract.TesseractError, OSError) as e:
            pass  # OCR may fail on low quality regions

        return result

    def _analyze_x_axis(self, region: np.ndarray) -> Dict:
        """Analyze X-axis region to extract ticks and labels"""
        result = {
            'min': 0.0,
            'max': 1.0,
            'ticks': [],
            'label': 'Time',
            'unit': '',
            'confidence': 0.5
        }

        if region.size == 0 or not self.tesseract_available:
            return result

        # Preprocess
        processed = self._preprocess_for_ocr(region)

        # OCR configuration for numbers
        config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.'

        try:
            text = pytesseract.image_to_string(processed, config=config)
            numbers = self._extract_numbers(text, axis='x')

            if numbers:
                result['ticks'] = sorted(numbers)
                result['min'] = min(numbers)
                result['max'] = max(numbers)
                result['confidence'] = 0.8

        except Exception as e:
            logger.warning(f"X-axis OCR failed: {e}")

        # Try to detect label and unit
        try:
            config_text = '--oem 3 --psm 6'
            text = pytesseract.image_to_string(region, config=config_text).lower()

            for label in self.x_labels:
                if label in text:
                    result['label'] = label.title()
                    break

            # Detect time unit
            if 'month' in text:
                result['unit'] = 'months'
            elif 'year' in text:
                result['unit'] = 'years'
            elif 'day' in text:
                result['unit'] = 'days'
            elif 'week' in text:
                result['unit'] = 'weeks'

        except (pytesseract.TesseractError, OSError) as e:
            pass  # Label detection failed, use defaults

        return result

    def _extract_numbers(self, text: str, axis: str = 'x') -> List[float]:
        """Extract numbers from OCR text"""
        numbers = []

        # Find all numbers (including decimals)
        pattern = r'[-+]?\d*\.?\d+'
        matches = re.findall(pattern, text)

        for match in matches:
            try:
                num = float(match)

                # Filter based on axis
                if axis == 'y':
                    # Y-axis: survival probability (0-1 or 0-100)
                    if 0 <= num <= 100:
                        numbers.append(num)
                else:
                    # X-axis: time values (can vary widely)
                    if 0 <= num <= 10000:
                        numbers.append(num)

            except ValueError:
                continue

        return sorted(set(numbers))

    def _detect_scale_type(self, ticks: List[float]) -> str:
        """Detect if scale is linear or logarithmic"""
        if len(ticks) < 3:
            return "linear"

        # Check if differences are roughly equal (linear)
        diffs = np.diff(ticks)
        if len(diffs) > 0:
            diff_std = np.std(diffs) / (np.mean(diffs) + 1e-10)
            if diff_std < 0.3:
                return "linear"

        # Check if ratios are roughly equal (log)
        ratios = [ticks[i+1] / ticks[i] if ticks[i] > 0 else 0 for i in range(len(ticks)-1)]
        if ratios:
            ratio_std = np.std(ratios) / (np.mean(ratios) + 1e-10)
            if ratio_std < 0.3:
                return "log"

        return "linear"

    def _heuristic_axis_detection(self, image: np.ndarray) -> Dict:
        """
        Robust fallback heuristic detection when OCR fails.

        Uses adaptive image analysis to estimate axis ranges.
        Multiple threshold strategies for robustness.
        """
        h, w = image.shape[:2]

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Strategy 1: Otsu's automatic thresholding (most robust)
        otsu_thresh, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        logger.debug(f"Otsu threshold: {otsu_thresh}")

        # Strategy 2: Adaptive thresholding for varying illumination
        binary_adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 5
        )

        # Strategy 3: Histogram-based percentile threshold
        # Find threshold that captures top 20% darkest pixels
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        cumsum = np.cumsum(hist)
        total_pixels = cumsum[-1]
        percentile_thresh = np.searchsorted(cumsum, total_pixels * 0.2)
        _, binary_percentile = cv2.threshold(gray, percentile_thresh, 255, cv2.THRESH_BINARY_INV)

        # Combine strategies: use intersection for robustness
        # Pixel is content if detected by at least 2 of 3 methods
        combined = ((binary_otsu > 0).astype(int) +
                   (binary_adaptive > 0).astype(int) +
                   (binary_percentile > 0).astype(int))
        binary = (combined >= 2).astype(np.uint8) * 255

        # Find rightmost curve point for x_max estimate
        cols_with_content = np.where(np.any(binary > 0, axis=0))[0]
        rows_with_content = np.where(np.any(binary > 0, axis=1))[0]

        if len(cols_with_content) > 0:
            # Filter out axis lines (usually at edges)
            content_center = cols_with_content[(cols_with_content > w * 0.15) &
                                               (cols_with_content < w * 0.95)]
            if len(content_center) > 0:
                rightmost = content_center[-1]
            else:
                rightmost = cols_with_content[-1]

            # Estimate x_max based on position in plot area
            plot_width_fraction = (rightmost - w * 0.15) / (w * 0.8)
            plot_width_fraction = np.clip(plot_width_fraction, 0.1, 1.0)

            # Common x_max values for KM curves
            common_x_max = [12, 24, 36, 48, 60, 100, 120]
            # Estimate based on content extent
            x_max_estimate = round(plot_width_fraction * 100 / 12) * 12
            # Find closest common value
            x_max_estimate = min(common_x_max, key=lambda x: abs(x - max(12, x_max_estimate)))
        else:
            x_max_estimate = 24  # Safe default for most oncology curves

        # Estimate y_max from content
        if len(rows_with_content) > 0:
            # Top of content should be near y=1.0
            topmost = rows_with_content[0]
            bottommost = rows_with_content[-1]
            # Check if content starts near top (survival starts at 1.0)
            starts_at_top = topmost < h * 0.2
        else:
            starts_at_top = True

        return {
            'x_min': 0,
            'x_max': x_max_estimate,
            'y_min': 0,
            'y_max': 1.0,
            'otsu_threshold': float(otsu_thresh),
            'percentile_threshold': float(percentile_thresh),
            'content_detected': len(cols_with_content) > 0
        }

    def _validate_with_nar(self, ocr_result: CalibratedAxisInfo,
                           nar_timepoints: List[float]) -> CalibratedAxisInfo:
        """
        Validate OCR axis detection using NAR timepoints.

        NAR (number-at-risk) timepoints typically span most of the x-axis.
        If there's a significant discrepancy between OCR x_max and NAR max,
        adjust the calibration.

        Parameters
        ----------
        ocr_result : CalibratedAxisInfo
            OCR-based axis detection result
        nar_timepoints : list
            Timepoints from NAR table (e.g., [0, 12, 24, 36, 48])

        Returns
        -------
        CalibratedAxisInfo with potentially adjusted values
        """
        if not nar_timepoints or len(nar_timepoints) < 2:
            return ocr_result

        nar_max = max(nar_timepoints)
        ocr_max = ocr_result.x_max

        if nar_max <= 0 or ocr_max <= 0:
            return ocr_result

        # NAR should span 70-100% of x-axis typically
        ratio = nar_max / ocr_max

        if 0.7 < ratio < 1.1:
            # OCR result is consistent with NAR - increase confidence
            ocr_result.confidence = min(1.0, ocr_result.confidence + 0.15)
            logger.debug(f"NAR validates OCR: ratio={ratio:.2f}, confidence boosted")
        elif ratio > 1.1:
            # NAR extends beyond OCR x_max - OCR likely underestimated
            # Adjust x_max to accommodate NAR with margin
            ocr_result.x_max = nar_max * 1.1
            ocr_result.confidence *= 0.8
            ocr_result.calibrated = True  # Trust NAR-based calibration

            # Round to nearest common value if close
            closest_common = min(self.common_x_max_values,
                                 key=lambda x: abs(x - ocr_result.x_max))
            if abs(closest_common - ocr_result.x_max) / closest_common < 0.1:
                ocr_result.x_max = closest_common

            logger.info(f"NAR extends beyond OCR: adjusted x_max to {ocr_result.x_max}")
        elif ratio < 0.5:
            # NAR is much smaller than OCR x_max - OCR likely overestimated
            # This could mean OCR picked up unrelated numbers
            ocr_result.x_max = nar_max * 1.15
            ocr_result.confidence *= 0.7

            closest_common = min(self.common_x_max_values,
                                 key=lambda x: abs(x - ocr_result.x_max))
            if abs(closest_common - ocr_result.x_max) / closest_common < 0.1:
                ocr_result.x_max = closest_common

            logger.info(f"NAR much smaller than OCR: adjusted x_max to {ocr_result.x_max}")
        else:
            # NAR is 50-70% of OCR - might be OK but reduce confidence slightly
            ocr_result.confidence *= 0.9
            logger.debug(f"NAR partial coverage: ratio={ratio:.2f}")

        return ocr_result

    def _extract_from_caption(self, caption: str) -> Optional[Dict]:
        """
        Extract axis information from figure caption.

        Parses captions for follow-up time and axis range information.

        Parameters
        ----------
        caption : str
            Figure caption text

        Returns
        -------
        dict or None: {'x_max': value, 'x_unit': 'months', 'source': 'caption'}
        """
        import re

        if not caption:
            return None

        caption_lower = caption.lower()

        # Patterns to extract follow-up time or axis range
        patterns = [
            # "follow-up of 36 months" or "36-month follow-up"
            (r'follow[- ]?up.*?(\d+)\s*(months?|years?|weeks?|days?)', 'follow_up'),
            (r'(\d+)[- ]?(months?|years?)\s*(?:of\s*)?follow[- ]?up', 'follow_up'),

            # "median follow-up was 24 months"
            (r'median\s+follow[- ]?up\s+(?:was|of|:)?\s*(\d+\.?\d*)\s*(months?|years?)', 'median_follow_up'),

            # "over 60 months" or "up to 60 months"
            (r'(?:over|up\s+to|through)\s+(\d+)\s*(months?|years?)', 'duration'),

            # "x-axis shows time in months (0-36)"
            (r'x[- ]?axis.*?(\d+)\s*(months?|years?)', 'x_axis'),

            # "time, months" with nearby number
            (r'time.*?(\d+)\s*(months?|years?)', 'time'),

            # Just numbers followed by months/years at end of sentence
            (r'(\d+)\s*(months?|years?)(?:\s*[\.;,)])', 'duration_end'),

            # "at 24 months" or "by 36 months"
            (r'(?:at|by|after)\s+(\d+)\s*(months?|years?)', 'timepoint'),
        ]

        results = []
        for pattern, source_type in patterns:
            matches = re.findall(pattern, caption_lower)
            for match in matches:
                try:
                    value = float(match[0])
                    unit = match[1].lower().rstrip('s')  # Normalize unit

                    # Convert to months
                    if unit == 'year':
                        value *= 12
                    elif unit == 'week':
                        value /= 4.33
                    elif unit == 'day':
                        value /= 30.44

                    # Reasonable range for clinical trials
                    if 3 <= value <= 150:
                        results.append({
                            'value': value,
                            'source_type': source_type,
                            'original_unit': unit
                        })
                except (ValueError, IndexError):
                    continue

        if not results:
            return None

        # Prioritize results
        # follow_up and median_follow_up are most reliable for x_max
        priority_order = ['median_follow_up', 'follow_up', 'duration', 'x_axis', 'time', 'timepoint', 'duration_end']

        for priority in priority_order:
            for r in results:
                if r['source_type'] == priority:
                    # Round to common clinical value
                    x_max = r['value']
                    closest_common = min(self.common_x_max_values,
                                         key=lambda x: abs(x - x_max))

                    # If very close to common value, use it
                    if abs(closest_common - x_max) / closest_common < 0.15:
                        x_max = closest_common

                    return {
                        'x_max': x_max,
                        'x_unit': 'months',
                        'source': 'caption',
                        'source_type': r['source_type'],
                        'confidence': 0.8 if priority in ['median_follow_up', 'follow_up'] else 0.6
                    }

        # Fallback: use the largest value found
        max_result = max(results, key=lambda r: r['value'])
        return {
            'x_max': max_result['value'],
            'x_unit': 'months',
            'source': 'caption',
            'source_type': max_result['source_type'],
            'confidence': 0.5
        }


def detect_axis_info(image_path: str) -> Dict:
    """
    Convenience function for axis detection.

    Args:
        image_path: Path to KM curve image

    Returns:
        Dictionary with axis information
    """
    detector = EnhancedAxisDetector()
    result = detector.detect_axes(image_path)

    return {
        'x_min': result.x_min,
        'x_max': result.x_max,
        'y_min': result.y_min,
        'y_max': result.y_max,
        'x_label': result.x_label,
        'y_label': result.y_label,
        'x_unit': result.x_unit,
        'x_ticks': result.x_ticks,
        'y_ticks': result.y_ticks,
        'x_scale': result.x_scale,
        'y_scale': result.y_scale,
        'confidence': result.confidence,
        'method': result.method,
        'plot_region': result.plot_region
    }


def test_axis_detector():
    """Test the enhanced axis detector"""
    print("=" * 60)
    print("TESTING ENHANCED AXIS DETECTOR")
    print("=" * 60)

    detector = EnhancedAxisDetector()

    # Test on validation images
    test_dirs = [
        Path("validation_ground_truth"),
        Path("ml_models/synthetic_training")
    ]

    test_images = []
    for test_dir in test_dirs:
        if test_dir.exists():
            test_images.extend(list(test_dir.glob("*.png"))[:3])

    if not test_images:
        print("No test images found")
        return

    print(f"Found {len(test_images)} test images")

    for img_path in test_images:
        print(f"\n{'='*40}")
        print(f"Testing: {img_path.name}")

        result = detector.detect_axes(str(img_path))

        print(f"  X-axis: {result.x_min} to {result.x_max} ({result.x_label})")
        print(f"  Y-axis: {result.y_min} to {result.y_max} ({result.y_label})")
        print(f"  X ticks: {result.x_ticks[:5]}...")
        print(f"  Y ticks: {result.y_ticks[:5]}...")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Method: {result.method}")

    print("\n" + "=" * 60)
    print("AXIS DETECTION TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_axis_detector()
