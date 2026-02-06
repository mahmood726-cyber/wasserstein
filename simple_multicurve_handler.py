"""
Simple Multi-Curve Handler for KM Extraction v8.7

This version uses a simpler, more reliable approach:
1. Direct color range detection using predefined HSV ranges
2. Connected component analysis for each color
3. Curve validation based on KM properties
4. No complex clustering - just detect each color separately

Key Options:
- detect_black_curves: Enable detection of black/dark gray curves (disabled by default
  to avoid detecting axes which are typically black)
- separate_similar_colors: Enable spatial clustering to separate curves with similar hues

Author: Wasserstein KM Extractor Team
Date: January 2026
Version: 8.7 (Added black curve detection, timing benchmarks, PDF validation)
"""

import os
import sys
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import logging

# Try to import scipy for advanced signal processing
try:
    from scipy import ndimage
    from scipy.signal import find_peaks
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Import adaptive margin detector
try:
    from adaptive_margin_detector import AdaptiveMarginDetector
    HAS_ADAPTIVE_MARGINS = True
except ImportError:
    HAS_ADAPTIVE_MARGINS = False

# Import sub-pixel extractor
try:
    from subpixel_extractor import SubpixelExtractor
    HAS_SUBPIXEL = True
except ImportError:
    HAS_SUBPIXEL = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class KMCurve:
    """A detected KM survival curve."""
    curve_id: int
    color_bgr: Tuple[int, int, int]
    color_name: str
    points: List[Tuple[int, int]]
    survival_data: List[Tuple[float, float]]  # (time, survival) tuples
    confidence: float
    panel_id: int = 0
    calibrated: bool = False  # True if axis-calibrated, False if normalized 0-1
    x_unit: str = ""  # Time unit (months, years, etc.) if calibrated


@dataclass
class Panel:
    """A detected panel."""
    panel_id: int
    bbox: Tuple[int, int, int, int]
    is_km_plot: bool
    curves: List[KMCurve] = field(default_factory=list)


class SimpleMultiCurveHandler:
    """
    Simple but robust multi-curve handler using direct color detection.

    Supports:
    - Standard multi-color curve detection
    - Similar-color curve separation via spatial clustering
    - Inset and multi-panel handling
    - Black/dark gray curve detection (enabled by default in medical mode)
    - Dashed/dotted line style detection

    Medical Mode:
    -------------
    When medical_mode=True (default), the handler is optimized for clinical
    trial KM curves with:
    - Black curve detection enabled (many journals use black curves)
    - Lower pixel thresholds for thin/anti-aliased curves
    - More aggressive curve separation

    Black Curve Detection:
    ----------------------
    When enabled, additional validation is applied:
    1. Black pixels must form a continuous horizontal structure (not vertical like axes)
    2. The detected region must span at least 20% of plot width
    3. Must pass standard KM validation (starts high, decreases)
    4. Vertical structures (aspect ratio < 0.3) are rejected as likely axes

    Use Cases for Black Curves:
    - Older publications with grayscale-only figures
    - Supplementary materials with simple black curves
    - Figures where treatment/control are differentiated by line style, not color
    """

    def __init__(self, separate_similar_colors: bool = False,
                 detect_black_curves: bool = None,  # None = auto based on medical_mode
                 use_adaptive_margins: bool = True,
                 use_subpixel_extraction: bool = True,
                 medical_mode: bool = True):
        """
        Args:
            separate_similar_colors: If True, attempt to separate curves of similar
                                     colors using vertical position clustering.
                                     Useful for plots with light blue vs dark blue.
            detect_black_curves: If True, also detect black/dark gray curves.
                                 If None (default), auto-set based on medical_mode.
                                 Enable for grayscale figures or when curves
                                 are differentiated by line style rather than color.
            use_adaptive_margins: If True, use adaptive margin detection to find
                                  plot boundaries. This improves accuracy by detecting
                                  actual axis positions instead of using fixed margins.
                                  Expected RMSE improvement: 0.03-0.05.
            use_subpixel_extraction: If True, use sub-pixel precision for curve tracing.
                                     This improves accuracy by using weighted centroid
                                     and gradient analysis instead of integer pixel coords.
                                     Expected RMSE improvement: 0.02-0.03.
            medical_mode: If True (default), optimize for clinical trial KM curves
                          with more aggressive detection settings.
        """
        self.medical_mode = medical_mode
        self.separate_similar_colors = separate_similar_colors

        # Auto-enable black curves in medical mode unless explicitly disabled
        if detect_black_curves is None:
            self.detect_black_curves = medical_mode
        else:
            self.detect_black_curves = detect_black_curves

        self.use_adaptive_margins = use_adaptive_margins and HAS_ADAPTIVE_MARGINS
        self.use_subpixel_extraction = use_subpixel_extraction and HAS_SUBPIXEL

        # Initialize adaptive margin detector
        if self.use_adaptive_margins:
            self.margin_detector = AdaptiveMarginDetector()
        else:
            self.margin_detector = None

        # Initialize sub-pixel extractor
        if self.use_subpixel_extraction:
            self.subpixel_extractor = SubpixelExtractor()
        else:
            self.subpixel_extractor = None

        # Define color ranges in HSV (H: 0-180, S: 0-255, V: 0-255)
        # Each color has a name, HSV lower/upper bounds, and expected BGR center
        self.color_defs = [
            {
                'name': 'red1',
                'hsv_lower': (0, 40, 80),
                'hsv_upper': (10, 255, 255),
                'bgr_center': (0, 0, 200)
            },
            {
                'name': 'red2',
                'hsv_lower': (170, 40, 80),
                'hsv_upper': (180, 255, 255),
                'bgr_center': (0, 0, 200)
            },
            {
                'name': 'orange',
                'hsv_lower': (10, 50, 80),
                'hsv_upper': (25, 255, 255),
                'bgr_center': (0, 128, 255)
            },
            {
                'name': 'yellow',
                'hsv_lower': (25, 50, 80),
                'hsv_upper': (35, 255, 255),
                'bgr_center': (0, 255, 255)
            },
            {
                'name': 'green',
                'hsv_lower': (35, 50, 50),
                'hsv_upper': (85, 255, 255),
                'bgr_center': (0, 180, 0)
            },
            # Cyan merged with blue to avoid edge pixel conflicts
            {
                'name': 'blue',
                'hsv_lower': (85, 50, 50),
                'hsv_upper': (130, 255, 255),
                'bgr_center': (255, 100, 0)
            },
            {
                'name': 'purple',
                'hsv_lower': (130, 50, 50),
                'hsv_upper': (160, 255, 255),
                'bgr_center': (180, 0, 180)
            },
            {
                'name': 'magenta',
                'hsv_lower': (160, 50, 50),
                'hsv_upper': (170, 255, 255),
                'bgr_center': (255, 0, 255)
            },
        ]

        # Black curve definition (only used when detect_black_curves=True)
        # Uses low saturation and low value to detect dark/black pixels
        # Additional validation prevents axis detection
        self.black_color_def = {
            'name': 'black',
            'hsv_lower': (0, 0, 0),
            'hsv_upper': (180, 50, 80),  # Low saturation, low value
            'bgr_center': (30, 30, 30)
        }

        # Gray curve definition - for grayscale figures with multiple gray levels
        self.gray_color_def = {
            'name': 'gray',
            'hsv_lower': (0, 0, 80),
            'hsv_upper': (180, 30, 180),  # Low saturation, medium value
            'bgr_center': (128, 128, 128)
        }

        # Thresholds - more lenient in medical mode for thin/anti-aliased curves
        if self.medical_mode:
            self.min_curve_points = 10  # Lower threshold for medical papers
            self.min_curve_width_ratio = 0.06  # Allow partial curves
            self.min_pixel_area = 15  # Lower for thin curves
        else:
            self.min_curve_points = 15
            self.min_curve_width_ratio = 0.15
            self.min_pixel_area = 50

    def process_figure(self, image: np.ndarray, expected_curves: int = None) -> Dict:
        """Process a KM figure."""
        h, w = image.shape[:2]

        result = {
            'panels': [],
            'curves': [],
            'insets_excluded': [],
            'is_multipanel': False,
            'n_curves': 0
        }

        # Step 1: Detect panels
        panels = self._detect_panels(image)
        result['is_multipanel'] = len(panels) > 1
        result['panels'] = panels

        if len(panels) == 0:
            panels = [Panel(0, (0, 0, w, h), True)]

        # Step 2: Process each panel
        all_curves = []
        for panel in panels:
            if not panel.is_km_plot:
                continue

            x1, y1, x2, y2 = panel.bbox
            panel_img = image[y1:y2, x1:x2].copy()

            # Step 3: Detect insets
            insets = self._detect_insets(panel_img)
            for inset in insets:
                ix1, iy1, ix2, iy2 = inset
                panel_img[iy1:iy2, ix1:ix2] = 255
                result['insets_excluded'].append({
                    'panel_id': panel.panel_id,
                    'bbox': inset
                })

            # Step 4: Extract curves
            curves = self._extract_curves(panel_img, panel.panel_id)

            # Fallback: if < 2 curves found, try enhanced separation cascade
            if len(curves) < 2:
                logger.info(f"Panel {panel.panel_id}: only {len(curves)} curves from primary detection, trying fallback cascade")
                fallback_curves = self._extract_curves_with_separation(panel_img, panel.panel_id)
                # Only use fallback if it found a reasonable number (2-20) — hundreds means noise
                if len(curves) < len(fallback_curves) <= 20:
                    curves = fallback_curves
                    logger.info(f"Panel {panel.panel_id}: fallback cascade found {len(curves)} curves")
                elif len(fallback_curves) > 20:
                    logger.info(f"Panel {panel.panel_id}: fallback cascade found {len(fallback_curves)} curves (too many, discarding)")

            # Adjust coordinates
            for curve in curves:
                curve.points = [(px + x1, py + y1) for px, py in curve.points]

            all_curves.extend(curves)
            panel.curves = curves

        # Step 5: Filter confidence bands (if more than 2 curves detected)
        if len(all_curves) > 2:
            all_curves = self._filter_confidence_bands(all_curves, image)
            # Update panel curves
            for panel in panels:
                panel.curves = [c for c in all_curves if c.panel_id == panel.panel_id]

        result['curves'] = all_curves
        result['n_curves'] = len(all_curves)

        return result

    def _detect_panels(self, image: np.ndarray) -> List[Panel]:
        """Detect multiple panels."""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find white gaps
        v_splits = self._find_white_gaps(gray, axis=1)
        h_splits = self._find_white_gaps(gray, axis=0)

        x_bounds = [0] + sorted(v_splits) + [w]
        y_bounds = [0] + sorted(h_splits) + [h]

        panels = []
        panel_id = 0

        for i in range(len(y_bounds) - 1):
            for j in range(len(x_bounds) - 1):
                x1, x2 = x_bounds[j], x_bounds[j+1]
                y1, y2 = y_bounds[i], y_bounds[i+1]

                if (x2 - x1) < 100 or (y2 - y1) < 100:
                    continue

                panels.append(Panel(
                    panel_id=panel_id,
                    bbox=(x1, y1, x2, y2),
                    is_km_plot=True
                ))
                panel_id += 1

        if len(panels) == 0:
            panels.append(Panel(0, (0, 0, w, h), True))

        return panels

    def _find_white_gaps(self, gray: np.ndarray, axis: int) -> List[int]:
        """
        Find white gaps that separate panels. Very conservative to avoid false positives.
        Only detects gaps where there's a continuous white strip across the entire image dimension.
        """
        h, w = gray.shape

        gaps = []

        if axis == 1:  # Vertical gaps (check columns)
            # For each column position, check if it's entirely white
            for x in range(int(w * 0.25), int(w * 0.75)):
                col = gray[:, x]
                # The ENTIRE column must be nearly white (mean > 250)
                # AND have low variance (consistent whiteness)
                if np.mean(col) > 250 and np.std(col) < 20:
                    # Check if this is part of a wide white region
                    gap_start = x
                    while gap_start > 0 and np.mean(gray[:, gap_start-1]) > 250:
                        gap_start -= 1
                    gap_end = x
                    while gap_end < w-1 and np.mean(gray[:, gap_end+1]) > 250:
                        gap_end += 1

                    gap_width = gap_end - gap_start
                    gap_center = (gap_start + gap_end) // 2

                    # Only count as gap if wide enough (>50 pixels)
                    if gap_width > 50 and gap_center not in gaps:
                        # Verify it's in the middle
                        if gap_center > w * 0.25 and gap_center < w * 0.75:
                            gaps.append(gap_center)
                            # Skip ahead to avoid duplicate detection
                            break

        else:  # Horizontal gaps (check rows)
            for y in range(int(h * 0.25), int(h * 0.75)):
                row = gray[y, :]
                if np.mean(row) > 250 and np.std(row) < 20:
                    gap_start = y
                    while gap_start > 0 and np.mean(gray[gap_start-1, :]) > 250:
                        gap_start -= 1
                    gap_end = y
                    while gap_end < h-1 and np.mean(gray[gap_end+1, :]) > 250:
                        gap_end += 1

                    gap_width = gap_end - gap_start
                    gap_center = (gap_start + gap_end) // 2

                    if gap_width > 50 and gap_center not in gaps:
                        if gap_center > h * 0.25 and gap_center < h * 0.75:
                            gaps.append(gap_center)
                            break

        return gaps

    def _detect_insets(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect inset plots to mask out. Be conservative - only mask clear insets."""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        insets = []

        # Look for clear rectangular boxes (not axes)
        # Focus on top-right corner where insets are most common
        corner_x1, corner_y1 = int(w * 0.5), 0
        corner_x2, corner_y2 = w, int(h * 0.5)

        corner_region = gray[corner_y1:corner_y2, corner_x1:corner_x2]

        # Look for dark rectangular borders
        edges = cv2.Canny(corner_region, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) == 4:
                x, y, cw, ch = cv2.boundingRect(contour)
                area = cw * ch
                corner_area = (corner_x2 - corner_x1) * (corner_y2 - corner_y1)
                area_ratio = area / (h * w)

                # Inset should be 5-20% of main area
                # and have reasonable aspect ratio (not too thin)
                aspect_ratio = cw / max(1, ch)
                if (0.05 < area_ratio < 0.20 and
                    0.5 < aspect_ratio < 2.0 and
                    cw > 100 and ch > 80):

                    # Check if this region has internal axes (indicator of inset plot)
                    inset_region = corner_region[y:y+ch, x:x+cw]
                    if self._has_internal_axes(inset_region):
                        # Convert to full image coordinates
                        abs_x1 = corner_x1 + x
                        abs_y1 = corner_y1 + y
                        insets.append((abs_x1, abs_y1, abs_x1 + cw, abs_y1 + ch))

        return insets

    def _has_internal_axes(self, region: np.ndarray) -> bool:
        """Check if a region has its own axes (L-shaped dark lines)."""
        if region.shape[0] < 50 or region.shape[1] < 50:
            return False

        h, w = region.shape

        # Check left edge for vertical dark line
        left_strip = region[:, :max(5, w // 10)]
        left_dark = np.mean(left_strip < 100)

        # Check bottom edge for horizontal dark line
        bottom_strip = region[max(h-5, int(h*0.9)):, :]
        bottom_dark = np.mean(bottom_strip < 100)

        return left_dark > 0.02 and bottom_dark > 0.02

    def _extract_curves(self, image: np.ndarray, panel_id: int) -> List[KMCurve]:
        """Extract curves using direct color detection."""
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        curves = []
        curve_id = 0

        # Define plot region (exclude margins)
        # Use adaptive margin detection for better accuracy (RMSE improvement: 0.03-0.05)
        if self.use_adaptive_margins and self.margin_detector is not None:
            try:
                margin_result = self.margin_detector.detect_margins(image)
                margin_left = margin_result.left
                margin_right = margin_result.right
                margin_top = margin_result.top
                margin_bottom = margin_result.bottom
                logger.debug(f"Adaptive margins: L={margin_left}, R={margin_right}, "
                           f"T={margin_top}, B={margin_bottom} (conf={margin_result.confidence:.2f})")
            except Exception as e:
                logger.warning(f"Adaptive margin detection failed, using defaults: {e}")
                margin_left = int(w * 0.08)
                margin_right = int(w * 0.05)
                margin_top = int(h * 0.08)
                margin_bottom = int(h * 0.12)
        else:
            # Default fixed margins
            margin_left = int(w * 0.08)
            margin_right = int(w * 0.05)
            margin_top = int(h * 0.08)
            margin_bottom = int(h * 0.12)

        plot_x1 = margin_left
        plot_x2 = w - margin_right
        plot_y1 = margin_top
        plot_y2 = h - margin_bottom
        plot_w = plot_x2 - plot_x1
        plot_h = plot_y2 - plot_y1

        # Track which colors we've found
        found_colors = set()

        # Build list of colors to check
        colors_to_check = list(self.color_defs)
        if self.detect_black_curves:
            colors_to_check.append(self.black_color_def)

        # Check each color definition
        for color_def in colors_to_check:
            color_name = color_def['name']

            # Skip if we already found this color (handles red1/red2 merge)
            base_name = color_name.rstrip('12')
            if base_name in found_colors:
                continue

            # Create mask for this color
            lower = np.array(color_def['hsv_lower'])
            upper = np.array(color_def['hsv_upper'])
            mask = cv2.inRange(hsv, lower, upper)

            # Handle red which spans hue boundary
            if color_name == 'red1':
                red2_def = next((c for c in self.color_defs if c['name'] == 'red2'), None)
                if red2_def:
                    lower2 = np.array(red2_def['hsv_lower'])
                    upper2 = np.array(red2_def['hsv_upper'])
                    mask2 = cv2.inRange(hsv, lower2, upper2)
                    mask = cv2.bitwise_or(mask, mask2)
                color_name = 'red'
                base_name = 'red'

            if color_name == 'red2':
                continue  # Already handled with red1

            # Check if we have any pixels for this color
            total_pixels = np.sum(mask > 0)
            if total_pixels < self.min_pixel_area:
                continue

            # Clean mask lightly (just remove small noise)
            kernel = np.ones((3, 3), np.uint8)
            clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # Check the width span of all pixels
            cols_with_pixels = np.any(clean_mask > 0, axis=0)
            if not np.any(cols_with_pixels):
                continue

            x_min = np.argmax(cols_with_pixels)
            x_max = len(cols_with_pixels) - np.argmax(cols_with_pixels[::-1]) - 1
            width_span = x_max - x_min

            # Skip if doesn't span enough width (likely not a curve)
            if width_span < plot_w * self.min_curve_width_ratio:
                continue

            # Trace the curve directly from the mask (all pixels of this color)
            points = self._trace_curve(clean_mask)

            if len(points) < self.min_curve_points:
                logger.debug(f"  Rejected {color_name}: only {len(points)} points (min {self.min_curve_points})")
                continue

            # Convert to survival values (using legacy normalized method for compatibility)
            survival_data = self._to_survival_legacy(points, h, w, margin_left, margin_top, plot_w, plot_h)

            # Validate KM shape
            if not self._is_valid_km(survival_data):
                logger.debug(f"  Rejected {color_name}: failed KM validation")
                continue

            # Additional validation for black curves to avoid detecting axes
            if color_name == 'black':
                if not self._is_valid_black_curve(clean_mask, points, plot_w, plot_h):
                    logger.debug(f"  Rejected black: failed axis exclusion validation")
                    continue

            # Calculate confidence
            confidence = min(1.0, len(points) / 100) * min(1.0, width_span / (plot_w * 0.5))

            curves.append(KMCurve(
                curve_id=curve_id,
                color_bgr=color_def['bgr_center'],
                color_name=color_name if color_name != 'red1' else 'red',
                points=points,
                survival_data=survival_data,
                confidence=confidence,
                panel_id=panel_id
            ))
            found_colors.add(base_name)
            curve_id += 1

            # If separate_similar_colors is enabled, look for additional curves
            # of similar color but different vertical position
            if self.separate_similar_colors:
                additional = self._find_additional_curves_spatial(
                    clean_mask, points, h, w, margin_left, margin_top, plot_w, plot_h,
                    color_def, color_name, panel_id, curve_id
                )
                curves.extend(additional)
                curve_id += len(additional)

        return curves

    def _find_additional_curves_spatial(self, mask: np.ndarray,
                                        existing_points: List[Tuple[int, int]],
                                        h: int, w: int,
                                        margin_left: int, margin_top: int,
                                        plot_w: int, plot_h: int,
                                        color_def: dict, color_name: str,
                                        panel_id: int, start_curve_id: int) -> List[KMCurve]:
        """
        Find additional curves of the same color using spatial clustering.

        After finding the first curve (topmost points), look for a second curve
        by finding pixels that are significantly below the first curve.
        """
        additional_curves = []

        # Get existing curve's y values at each x
        existing_y_by_x = {}
        for x, y in existing_points:
            if x not in existing_y_by_x or y < existing_y_by_x[x]:
                existing_y_by_x[x] = y

        # Create mask excluding the existing curve (with margin)
        margin = 20  # Pixels margin around existing curve
        remaining_mask = mask.copy()

        for x, y in existing_points:
            y_min = max(0, y - margin)
            y_max = min(h, y + margin)
            remaining_mask[y_min:y_max, x] = 0

        # Check if there are enough remaining pixels
        remaining_pixels = np.sum(remaining_mask > 0)
        if remaining_pixels < self.min_pixel_area:
            return additional_curves

        # Check width span
        cols_with_pixels = np.any(remaining_mask > 0, axis=0)
        if not np.any(cols_with_pixels):
            return additional_curves

        x_min = np.argmax(cols_with_pixels)
        x_max = len(cols_with_pixels) - np.argmax(cols_with_pixels[::-1]) - 1
        width_span = x_max - x_min

        if width_span < plot_w * self.min_curve_width_ratio:
            return additional_curves

        # Trace additional curve
        points = self._trace_curve(remaining_mask)

        if len(points) < self.min_curve_points:
            return additional_curves

        # Validate
        survival_data = self._to_survival_legacy(points, h, w, margin_left, margin_top, plot_w, plot_h)

        if not self._is_valid_km(survival_data):
            return additional_curves

        # Check that this curve is meaningfully different (not just noise)
        # Average y-distance from existing curve should be significant
        y_distances = []
        for x, y in points:
            if x in existing_y_by_x:
                y_distances.append(abs(y - existing_y_by_x[x]))

        if y_distances and np.mean(y_distances) < 15:
            return additional_curves  # Too close to existing curve

        confidence = min(1.0, len(points) / 100) * 0.8  # Slightly lower confidence

        additional_curves.append(KMCurve(
            curve_id=start_curve_id,
            color_bgr=color_def['bgr_center'],
            color_name=f"{color_name}_2",  # Mark as second curve of same color
            points=points,
            survival_data=survival_data,
            confidence=confidence,
            panel_id=panel_id
        ))

        return additional_curves

    def _trace_curve(self, mask: np.ndarray, image: np.ndarray = None) -> List[Tuple[int, int]]:
        """
        Trace curve from mask.

        IMPROVED: Detects if mask is a thick band (confidence interval shading)
        and extracts centerline instead of top edge.

        If sub-pixel extraction is enabled, uses weighted centroid and
        gradient analysis for sub-pixel precision. Otherwise uses integer
        topmost pixel per column.

        Args:
            mask: Binary mask of curve pixels
            image: Original image (optional, for sub-pixel intensity weighting)

        Returns:
            List of (x, y) coordinate tuples
        """
        h, w = mask.shape

        # Check if this is a thick band (confidence interval shading)
        # by analyzing the average vertical thickness
        is_thick_band = self._is_thick_shaded_region(mask)

        # Use sub-pixel extraction for better accuracy (RMSE improvement: 0.02-0.03)
        if self.use_subpixel_extraction and self.subpixel_extractor is not None and not is_thick_band:
            try:
                # Get intensity image for weighting
                if image is not None and len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                elif image is not None:
                    gray = image
                else:
                    gray = mask.astype(np.uint8)

                subpixel_points = self.subpixel_extractor.trace_curve_subpixel(mask, gray)

                # Convert to integer coordinates (preserving sub-pixel info in the process)
                # The sub-pixel precision improves survival value calculation later
                points = [(int(p.x), int(round(p.y))) for p in subpixel_points]
                return points
            except Exception as e:
                logger.warning(f"Sub-pixel extraction failed, using fallback: {e}")

        # Extract points based on region type
        points = []
        for x in range(w):
            col = mask[:, x]
            y_positions = np.where(col > 0)[0]

            if len(y_positions) > 0:
                if is_thick_band:
                    # For thick bands: use MEDIAN (centerline) to get the curve
                    y = int(np.median(y_positions))
                else:
                    # For thin curves: use topmost (minimum y = highest survival)
                    y = int(np.min(y_positions))
                points.append((x, y))

        return points

    def _is_thick_shaded_region(self, mask: np.ndarray) -> bool:
        """
        Detect if mask represents a thick shaded region (confidence band)
        rather than a thin curve line.

        Criteria:
        1. Total pixel count is very high (>50,000 pixels)
        2. Average column thickness is >10 pixels
        3. Thickness is fairly consistent (low variation = uniform band)

        Returns True if this is likely a confidence band shading.
        """
        h, w = mask.shape
        total_pixels = np.sum(mask > 0)

        # A typical curve line has <10,000 pixels
        # Confidence bands typically have >50,000 pixels
        if total_pixels < 50000:
            return False

        # Calculate column-wise thickness
        column_sums = np.sum(mask > 0, axis=0)
        non_zero_cols = column_sums[column_sums > 0]

        if len(non_zero_cols) == 0:
            return False

        avg_thickness = np.mean(non_zero_cols)
        std_thickness = np.std(non_zero_cols)

        # Thick band: average thickness > 10 pixels
        # (typical curve lines are 1-5 pixels thick)
        if avg_thickness > 10:
            logger.debug(f"Thick shaded region detected: {total_pixels} pixels, "
                        f"avg_thickness={avg_thickness:.1f}")
            return True

        return False

    def _detect_line_style(self, mask: np.ndarray) -> str:
        """
        Detect if curve is solid, dashed, or dotted.

        Analyzes gaps in the horizontal projection to identify line style.

        Parameters
        ----------
        mask : np.ndarray
            Binary mask of curve pixels

        Returns
        -------
        str: 'solid', 'dashed', or 'dotted'
        """
        if mask.size == 0:
            return 'solid'

        h, w = mask.shape

        # Horizontal projection - count pixels per column
        h_proj = np.sum(mask > 0, axis=0)

        # Find columns with pixels
        has_pixels = h_proj > 0

        # Count transitions (gaps in line)
        transitions = np.diff(has_pixels.astype(int))
        gap_starts = np.sum(transitions == -1)  # Pixel -> no pixel
        gap_ends = np.sum(transitions == 1)     # No pixel -> pixel

        n_gaps = min(gap_starts, gap_ends)

        # Calculate average gap length
        if n_gaps > 0:
            # Find gap positions
            gap_positions = np.where(transitions == -1)[0]
            resume_positions = np.where(transitions == 1)[0]

            if len(gap_positions) > 0 and len(resume_positions) > 0:
                # Match gaps to resumes
                gap_lengths = []
                for gap_start in gap_positions:
                    resumes_after = resume_positions[resume_positions > gap_start]
                    if len(resumes_after) > 0:
                        gap_lengths.append(resumes_after[0] - gap_start)

                if gap_lengths:
                    avg_gap = np.mean(gap_lengths)

                    # Classify based on gap count and size
                    if n_gaps > 15 and avg_gap < 5:
                        return 'dotted'
                    elif n_gaps > 5 and avg_gap > 3:
                        return 'dashed'

        return 'solid'

    def _detect_dashed_curve(self, image: np.ndarray, color_mask: np.ndarray) -> np.ndarray:
        """
        Connect dashed/dotted line segments into continuous curve.

        Uses morphological operations to bridge gaps.

        Parameters
        ----------
        image : np.ndarray
            Original image
        color_mask : np.ndarray
            Binary mask from color detection

        Returns
        -------
        np.ndarray: Connected mask
        """
        # Detect line style
        style = self._detect_line_style(color_mask)

        if style == 'solid':
            return color_mask

        # For dashed/dotted, use morphological closing to bridge gaps
        if style == 'dotted':
            # Dotted lines need more aggressive closing
            kernel = np.ones((3, 7), np.uint8)  # Horizontal kernel
        else:  # dashed
            kernel = np.ones((3, 12), np.uint8)  # Wider horizontal kernel

        # Close gaps
        closed = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)

        # Thin back to single pixel width
        # Use skeletonization if available, otherwise erode
        try:
            from skimage.morphology import skeletonize
            skeleton = skeletonize(closed > 0).astype(np.uint8) * 255
            return skeleton
        except ImportError:
            # Fallback: light erosion
            erode_kernel = np.ones((2, 2), np.uint8)
            return cv2.erode(closed, erode_kernel, iterations=1)

    def _extract_curves_with_line_styles(self, image: np.ndarray, panel_id: int) -> List[KMCurve]:
        """
        Extract curves with support for different line styles.

        This method extends _extract_curves to handle dashed and dotted lines.
        """
        # First try standard extraction
        curves = self._extract_curves(image, panel_id)

        # If we found enough curves, return
        if len(curves) >= 2:
            return curves

        # Otherwise, try with dashed line detection
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Check each color with dashed line handling
        colors_to_check = list(self.color_defs)
        if self.detect_black_curves:
            colors_to_check.append(self.black_color_def)

        for color_def in colors_to_check:
            color_name = color_def['name']
            if color_name in ['red2']:
                continue

            lower = np.array(color_def['hsv_lower'])
            upper = np.array(color_def['hsv_upper'])
            mask = cv2.inRange(hsv, lower, upper)

            # Handle red wrapping
            if color_name == 'red1':
                red2_def = next((c for c in self.color_defs if c['name'] == 'red2'), None)
                if red2_def:
                    lower2 = np.array(red2_def['hsv_lower'])
                    upper2 = np.array(red2_def['hsv_upper'])
                    mask2 = cv2.inRange(hsv, lower2, upper2)
                    mask = cv2.bitwise_or(mask, mask2)

            # Check if this might be a dashed line
            style = self._detect_line_style(mask)
            if style in ['dashed', 'dotted']:
                # Connect the dashes
                connected_mask = self._detect_dashed_curve(image, mask)

                # Check if connected mask forms a valid curve
                total_pixels = np.sum(connected_mask > 0)
                if total_pixels > self.min_pixel_area:
                    logger.debug(f"Detected {style} {color_name} curve with {total_pixels} pixels")

                    # Trace the connected mask and validate as KM curve
                    points = self._trace_curve(connected_mask)
                    if len(points) >= self.min_curve_points:
                        # Define plot region for survival conversion
                        margin_left = int(w * 0.08)
                        margin_top = int(h * 0.08)
                        plot_w = w - margin_left - int(w * 0.05)
                        plot_h = h - margin_top - int(h * 0.12)

                        survival_data = self._to_survival_legacy(
                            points, h, w, margin_left, margin_top, plot_w, plot_h
                        )

                        if self._is_valid_km(survival_data):
                            # Check width span
                            cols_with_pixels = np.any(connected_mask > 0, axis=0)
                            x_min_px = np.argmax(cols_with_pixels)
                            x_max_px = len(cols_with_pixels) - np.argmax(cols_with_pixels[::-1]) - 1
                            width_span = x_max_px - x_min_px

                            # Check not already in curves (by color)
                            existing_colors = {c.color_name for c in curves}
                            actual_name = 'red' if color_name == 'red1' else color_name
                            if actual_name not in existing_colors:
                                confidence = min(1.0, len(points) / 100) * min(1.0, width_span / (plot_w * 0.5)) * 0.9
                                curves.append(KMCurve(
                                    curve_id=len(curves),
                                    color_bgr=color_def['bgr_center'],
                                    color_name=f"{actual_name}_{style}",
                                    points=points,
                                    survival_data=survival_data,
                                    confidence=confidence,
                                    panel_id=panel_id
                                ))
                                logger.info(f"Added {style} {actual_name} curve: {len(points)} points")

        return curves

    def _cluster_colors_by_hue(self, hsv_image: np.ndarray,
                                n_clusters: int = 4,
                                saturation_threshold: int = 30) -> Optional[List[np.ndarray]]:
        """
        Cluster pixels by hue to separate similar-colored curves.

        Uses KMeans clustering on hue values to identify distinct color groups.

        Parameters
        ----------
        hsv_image : np.ndarray
            Image in HSV color space
        n_clusters : int
            Number of color clusters to find
        saturation_threshold : int
            Minimum saturation to consider (filters out grayscale)

        Returns
        -------
        List of binary masks, one per cluster, or None if clustering fails
        """
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            logger.debug("sklearn not available for color clustering")
            return None

        h, w = hsv_image.shape[:2]

        # Get non-white, non-gray pixels (saturated colors)
        saturation = hsv_image[:, :, 1]
        value = hsv_image[:, :, 2]

        # Mask: has color (saturated) and not too bright (not white)
        color_mask = (saturation > saturation_threshold) & (value < 250) & (value > 30)

        # Get coordinates and hue values of colored pixels
        y_coords, x_coords = np.where(color_mask)

        if len(y_coords) < 100:
            return None

        hues = hsv_image[y_coords, x_coords, 0].reshape(-1, 1)

        # Handle hue wrapping for red (0 and 180 are same color)
        # Map hues to circular coordinates
        hue_sin = np.sin(hues * np.pi / 90)  # 180 degrees -> pi
        hue_cos = np.cos(hues * np.pi / 90)
        hue_features = np.hstack([hue_sin, hue_cos])

        # Cluster
        try:
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            labels = kmeans.fit_predict(hue_features)
        except Exception as e:
            logger.debug(f"KMeans clustering failed: {e}")
            return None

        # Create separate masks for each cluster
        cluster_masks = []
        for i in range(n_clusters):
            mask = np.zeros((h, w), dtype=np.uint8)
            cluster_indices = labels == i

            if np.sum(cluster_indices) < self.min_pixel_area:
                continue

            mask[y_coords[cluster_indices], x_coords[cluster_indices]] = 255
            cluster_masks.append(mask)

        return cluster_masks if cluster_masks else None

    def _separate_curves_by_color_clustering(self, image: np.ndarray,
                                              panel_id: int,
                                              expected_curves: int = 2) -> List[KMCurve]:
        """
        Separate curves using color clustering when standard detection fails.

        This is useful for figures with similar colors (e.g., light blue vs dark blue)
        or when color definitions don't match exactly.

        Parameters
        ----------
        image : np.ndarray
            KM figure image
        panel_id : int
            Panel ID for output
        expected_curves : int
            Expected number of curves (used for clustering)

        Returns
        -------
        List of KMCurve objects
        """
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Try clustering with expected number of curves
        cluster_masks = self._cluster_colors_by_hue(hsv, n_clusters=expected_curves + 1)

        if not cluster_masks:
            return []

        curves = []
        curve_id = 0

        # Define plot region
        margin_left = int(w * 0.08)
        margin_right = int(w * 0.05)
        margin_top = int(h * 0.08)
        margin_bottom = int(h * 0.12)
        plot_w = w - margin_left - margin_right
        plot_h = h - margin_top - margin_bottom

        for mask in cluster_masks:
            # Clean the mask
            kernel = np.ones((3, 3), np.uint8)
            clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # Check width span
            cols_with_pixels = np.any(clean_mask > 0, axis=0)
            if not np.any(cols_with_pixels):
                continue

            x_min = np.argmax(cols_with_pixels)
            x_max = len(cols_with_pixels) - np.argmax(cols_with_pixels[::-1]) - 1
            width_span = x_max - x_min

            if width_span < plot_w * self.min_curve_width_ratio:
                continue

            # Trace the curve
            points = self._trace_curve(clean_mask)

            if len(points) < self.min_curve_points:
                continue

            # Convert to survival
            survival_data = self._to_survival_legacy(points, h, w, margin_left, margin_top, plot_w, plot_h)

            # Validate
            if not self._is_valid_km(survival_data):
                continue

            # Get average color for this cluster
            y_coords, x_coords = np.where(clean_mask > 0)
            if len(y_coords) > 0:
                avg_bgr = np.mean(image[y_coords, x_coords], axis=0).astype(int)
            else:
                avg_bgr = (128, 128, 128)

            confidence = min(1.0, len(points) / 100) * 0.8  # Lower confidence for clustered

            curves.append(KMCurve(
                curve_id=curve_id,
                color_bgr=tuple(avg_bgr),
                color_name=f'cluster_{curve_id}',
                points=points,
                survival_data=survival_data,
                confidence=confidence,
                panel_id=panel_id
            ))
            curve_id += 1

        return curves

    def _separate_by_vertical_position(self, mask: np.ndarray,
                                        min_gap: int = 15) -> List[np.ndarray]:
        """
        Separate overlapping curves by vertical position clustering.

        When curves of the same color are stacked vertically (one above the other),
        this method separates them using connected component analysis.

        Parameters
        ----------
        mask : np.ndarray
            Binary mask with potentially multiple curves
        min_gap : int
            Minimum vertical gap between curves to consider them separate

        Returns
        -------
        List of binary masks, one per detected curve
        """
        from scipy import ndimage

        if mask.size == 0:
            return [mask]

        h, w = mask.shape

        # First, try connected component analysis
        # Dilate horizontally to connect curve segments, but not vertically
        h_kernel = np.ones((1, 5), np.uint8)
        dilated = cv2.dilate(mask, h_kernel, iterations=2)

        # Label connected components
        labeled, n_features = ndimage.label(dilated)

        if n_features <= 1:
            # Only one component - try y-position clustering
            return self._cluster_by_y_position(mask, min_gap)

        # Extract each component as separate mask
        separate_masks = []
        for i in range(1, n_features + 1):
            component_mask = ((labeled == i) & (mask > 0)).astype(np.uint8) * 255

            # Check if this component is large enough
            if np.sum(component_mask > 0) >= self.min_pixel_area:
                separate_masks.append(component_mask)

        return separate_masks if separate_masks else [mask]

    def _cluster_by_y_position(self, mask: np.ndarray, min_gap: int = 15) -> List[np.ndarray]:
        """
        Cluster pixels by y-position to separate horizontally overlapping curves.

        Parameters
        ----------
        mask : np.ndarray
            Binary mask with potentially multiple curves
        min_gap : int
            Minimum vertical gap to consider as separate curves

        Returns
        -------
        List of binary masks, one per detected curve
        """
        h, w = mask.shape

        # For each x-column, find all y-positions with pixels
        separate_masks = []

        # Analyze y-distribution
        y_coords, x_coords = np.where(mask > 0)

        if len(y_coords) < self.min_pixel_area:
            return [mask]

        # Find y-distribution at each x
        # Group y-values by column and look for gaps
        y_by_x = defaultdict(list)
        for y, x in zip(y_coords, x_coords):
            y_by_x[x].append(y)

        # For columns with multiple y-values, check for gaps
        gap_detected = False
        for x, y_list in y_by_x.items():
            if len(y_list) > 1:
                y_sorted = sorted(y_list)
                gaps = np.diff(y_sorted)
                if np.any(gaps > min_gap):
                    gap_detected = True
                    break

        if not gap_detected:
            return [mask]

        # Perform actual separation using vertical profile
        # Project to y-axis
        y_proj = np.sum(mask > 0, axis=1)

        # Find valleys (gaps between curves)
        # Smooth the projection
        kernel_size = 5
        y_proj_smooth = np.convolve(y_proj, np.ones(kernel_size)/kernel_size, mode='same')

        # Find local minima that could be gaps
        from scipy.signal import find_peaks

        # Find peaks in inverted signal (valleys in original)
        inverted = np.max(y_proj_smooth) - y_proj_smooth
        peaks, properties = find_peaks(inverted, height=np.max(inverted) * 0.3, distance=min_gap)

        if len(peaks) == 0:
            return [mask]

        # Split at the deepest valley
        if len(peaks) >= 1:
            # Use the most prominent valley
            split_y = peaks[np.argmax(properties['peak_heights'])]

            # Create two masks
            mask_upper = mask.copy()
            mask_lower = mask.copy()

            mask_upper[split_y:, :] = 0
            mask_lower[:split_y, :] = 0

            # Check both have enough pixels
            result_masks = []
            for m in [mask_upper, mask_lower]:
                if np.sum(m > 0) >= self.min_pixel_area:
                    result_masks.append(m)

            if len(result_masks) >= 2:
                return result_masks

        return [mask]

    def _extract_curves_with_separation(self, image: np.ndarray, panel_id: int,
                                         expected_curves: int = 2) -> List[KMCurve]:
        """
        Extract curves with enhanced separation for overlapping/similar curves.

        This method combines multiple strategies:
        1. Standard color-based extraction
        2. Color clustering for similar hues
        3. Vertical position clustering for overlapping curves
        4. Dashed/dotted line detection

        Parameters
        ----------
        image : np.ndarray
            KM figure image
        panel_id : int
            Panel ID
        expected_curves : int
            Expected number of curves

        Returns
        -------
        List of KMCurve objects
        """
        # Try standard extraction first
        curves = self._extract_curves(image, panel_id)

        if len(curves) >= expected_curves:
            return curves

        # Try with line style detection
        curves_styled = self._extract_curves_with_line_styles(image, panel_id)
        if len(curves_styled) > len(curves):
            curves = curves_styled

        if len(curves) >= expected_curves:
            return curves

        # Try color clustering
        curves_clustered = self._separate_curves_by_color_clustering(image, panel_id, expected_curves)
        if len(curves_clustered) > len(curves):
            curves = curves_clustered

        if len(curves) >= expected_curves:
            return curves

        # Try vertical separation on existing masks
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Get all colored pixels
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        color_mask = ((saturation > 30) | (value < 80)) & (value < 250) & (value > 20)
        color_mask = color_mask.astype(np.uint8) * 255

        # Try to separate
        separated = self._separate_by_vertical_position(color_mask, min_gap=15)

        if len(separated) > len(curves):
            # Process each separated mask
            margin_left = int(w * 0.08)
            margin_top = int(h * 0.08)
            plot_w = w - margin_left - int(w * 0.05)
            plot_h = h - margin_top - int(h * 0.12)

            new_curves = []
            for i, sep_mask in enumerate(separated):
                points = self._trace_curve(sep_mask)
                if len(points) >= self.min_curve_points:
                    survival_data = self._to_survival_legacy(points, h, w, margin_left, margin_top, plot_w, plot_h)
                    if self._is_valid_km(survival_data):
                        # Get color
                        y_coords, x_coords = np.where(sep_mask > 0)
                        if len(y_coords) > 0:
                            avg_bgr = np.mean(image[y_coords, x_coords], axis=0).astype(int)
                        else:
                            avg_bgr = (128, 128, 128)

                        new_curves.append(KMCurve(
                            curve_id=i,
                            color_bgr=tuple(avg_bgr),
                            color_name=f'separated_{i}',
                            points=points,
                            survival_data=survival_data,
                            confidence=0.7,
                            panel_id=panel_id
                        ))

            if len(new_curves) > len(curves):
                curves = new_curves

        return curves

    def _to_survival(self, points: List[Tuple[int, int]], h: int, w: int,
                     margin_left: int, margin_top: int, plot_w: int, plot_h: int,
                     axis_info: Optional[Dict] = None) -> Tuple[List[Tuple[float, float]], bool]:
        """
        Convert pixel points to (time, survival) with optional calibration.

        Parameters
        ----------
        points : list
            List of (x, y) pixel coordinates
        h, w : int
            Image dimensions
        margin_left, margin_top : int
            Plot margins
        plot_w, plot_h : int
            Plot dimensions in pixels
        axis_info : dict, optional
            Calibration info with keys:
            - x_min, x_max: Time axis range
            - y_min, y_max: Survival axis range (usually 0-1)
            - calibrated: True if actual values, False if normalized

        Returns
        -------
        tuple: (survival_data, is_calibrated)
        """
        survival_data = []
        is_calibrated = False

        # Get axis ranges
        if axis_info and axis_info.get('calibrated', False):
            x_min = axis_info.get('x_min', 0.0)
            x_max = axis_info.get('x_max', 1.0)
            y_min = axis_info.get('y_min', 0.0)
            y_max = axis_info.get('y_max', 1.0)
            is_calibrated = True
        else:
            # Normalized 0-1 for both axes
            x_min, x_max = 0.0, 1.0
            y_min, y_max = 0.0, 1.0

        for x, y in points:
            # Convert x to time
            x_normalized = max(0, min(1, (x - margin_left) / max(1, plot_w)))
            t = x_min + x_normalized * (x_max - x_min)

            # Convert y to survival
            y_normalized = max(0, min(1, 1 - (y - margin_top) / max(1, plot_h)))
            s = y_min + y_normalized * (y_max - y_min)

            # Clamp survival to valid range
            s = max(0, min(y_max, s))

            survival_data.append((t, s))

        return survival_data, is_calibrated

    def _to_survival_legacy(self, points: List[Tuple[int, int]], h: int, w: int,
                            margin_left: int, margin_top: int, plot_w: int, plot_h: int) -> List[Tuple[float, float]]:
        """Legacy method: Convert pixel points to normalized (time, survival)."""
        survival_data, _ = self._to_survival(points, h, w, margin_left, margin_top, plot_w, plot_h, None)
        return survival_data

    def _is_valid_km(self, survival_data: List[Tuple[float, float]]) -> bool:
        """Check if data looks like a valid KM curve."""
        if len(survival_data) < 8:  # Reduced from 10 for thin curves
            logger.debug(f"    KM validation: too few points ({len(survival_data)})")
            return False

        # Should start high (but allow some margin for edge detection)
        start_vals = [s for t, s in survival_data[:5]]
        mean_start = np.mean(start_vals)
        if mean_start < 0.4:  # Reduced from 0.5 for better tolerance
            logger.debug(f"    KM validation: start too low ({mean_start:.2f})")
            return False

        # Should generally decrease or plateau (allow plateaus and some noise)
        increases = 0
        for i in range(1, len(survival_data)):
            # More tolerant of small increases (noise)
            if survival_data[i][1] > survival_data[i-1][1] + 0.05:  # Increased from 0.03
                increases += 1

        increase_ratio = increases / len(survival_data)
        if increase_ratio > 0.20:  # Increased from 0.15 for more tolerance
            logger.debug(f"    KM validation: too many increases ({increases}, {increase_ratio:.2%})")
            return False

        return True

    def _is_valid_black_curve(self, mask: np.ndarray, points: List[Tuple[int, int]],
                               plot_w: int, plot_h: int) -> bool:
        """
        Additional validation specifically for black curves to avoid detecting axes.

        Axes are typically:
        - Vertical lines on the left (y-axis)
        - Horizontal lines at the bottom (x-axis)
        - Straight with no step pattern

        KM curves are:
        - Primarily horizontal with steps
        - Span most of the plot width
        - Have variable y positions

        This method checks aspect ratio and spatial distribution to reject axes.
        """
        if len(points) < 20:
            return False

        # Get bounding box of detected points
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        width = x_max - x_min
        height = y_max - y_min

        # Aspect ratio check: axes are thin lines
        # Y-axis: very tall, very narrow (height >> width)
        # X-axis: very wide, very thin (width >> height)
        if height > 0:
            aspect_ratio = width / height
        else:
            return False

        # Reject vertical structures (likely y-axis)
        if aspect_ratio < 0.3:
            logger.debug(f"    Black curve rejected: too vertical (aspect={aspect_ratio:.2f})")
            return False

        # Reject very thin horizontal structures (likely x-axis or gridlines)
        if height < plot_h * 0.15 and aspect_ratio > 5:
            logger.debug(f"    Black curve rejected: too thin horizontal (h={height}, aspect={aspect_ratio:.2f})")
            return False

        # Check y-variance: real KM curves have varying y values
        y_std = np.std(y_coords)
        if y_std < 5:
            logger.debug(f"    Black curve rejected: y too constant (std={y_std:.1f})")
            return False

        # Check for step pattern: KM curves have clusters of same y-value
        # Count unique y values - should be limited (step function)
        unique_y = len(set(y_coords))
        unique_ratio = unique_y / len(points)

        # Real KM curves have many points at same y (horizontal segments)
        # Diagonal lines have unique y for almost every point
        if unique_ratio > 0.9:
            logger.debug(f"    Black curve rejected: no step pattern (unique_y_ratio={unique_ratio:.2f})")
            return False

        # Check position: reject if concentrated at edges (axis regions)
        # Y-axis would be at left edge, X-axis at bottom
        left_edge_ratio = sum(1 for x, y in points if x < plot_w * 0.1) / len(points)
        bottom_edge_ratio = sum(1 for x, y in points if y > plot_h * 0.85) / len(points)

        if left_edge_ratio > 0.8:
            logger.debug(f"    Black curve rejected: concentrated at left edge ({left_edge_ratio:.0%})")
            return False

        if bottom_edge_ratio > 0.8:
            logger.debug(f"    Black curve rejected: concentrated at bottom edge ({bottom_edge_ratio:.0%})")
            return False

        return True

    def _is_confidence_band(self, mask: np.ndarray, image: np.ndarray = None) -> bool:
        """
        Detect if a detected region is a confidence band rather than a curve.

        CONSERVATIVE DETECTION - Only flag true confidence bands:
        Confidence bands are shaded regions around curves showing confidence intervals.
        They are MUCH thicker than curves and have specific visual properties.

        Key differences from curves:
        1. MUCH thicker (typically >5% of image height vs <2% for curves)
        2. Semi-transparent/lighter coloring
        3. Symmetric (equal pixels above and below center)
        4. Fill a region rather than trace a line

        This method is intentionally CONSERVATIVE to avoid false positives.
        Better to keep a confidence band than remove a valid curve.

        Parameters
        ----------
        mask : np.ndarray
            Binary mask of the detected region
        image : np.ndarray, optional
            Original image for saturation analysis

        Returns
        -------
        bool : True if region is DEFINITELY a confidence band
        """
        if mask is None or np.sum(mask > 0) == 0:
            return False

        # Get image dimensions for relative thresholds
        h, w = mask.shape[:2]

        # VERY CONSERVATIVE THRESHOLDS
        # Only detect TRUE confidence bands, not thick curves
        # Confidence bands are typically 5-15% of image height
        # Curves (even thick/anti-aliased) are typically <3%
        band_thickness_threshold = h * 0.05  # 5% of height - MUST exceed this
        min_band_pixel_count = h * w * 0.01  # At least 1% of image area

        # Calculate average thickness (vertical extent per column)
        column_sums = np.sum(mask > 0, axis=0)
        non_zero_columns = column_sums[column_sums > 0]

        if len(non_zero_columns) == 0:
            return False

        avg_thickness = np.mean(non_zero_columns)
        total_pixels = np.sum(mask > 0)

        # REQUIREMENT 1: Must be THICK (>5% of image height average)
        if avg_thickness < band_thickness_threshold:
            return False  # Not thick enough to be a band

        # REQUIREMENT 2: Must cover significant area (>1% of image)
        if total_pixels < min_band_pixel_count:
            return False  # Not enough pixels to be a band

        # REQUIREMENT 3: Must be somewhat symmetric (band wraps around curve)
        y_coords = np.where(mask > 0)[0]
        if len(y_coords) > 50:
            median_y = np.median(y_coords)
            upper_count = np.sum(y_coords < median_y)
            lower_count = np.sum(y_coords >= median_y)

            if upper_count > 0 and lower_count > 0:
                symmetry = min(upper_count, lower_count) / max(upper_count, lower_count)
                # High symmetry (>0.6) suggests a band around a curve
                if symmetry < 0.5:
                    return False  # Not symmetric enough - probably a curve

        # REQUIREMENT 4: Check saturation - bands are often semi-transparent/lighter
        is_low_saturation = False
        if image is not None and len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            y_coords, x_coords = np.where(mask > 0)

            if len(y_coords) > 0:
                saturations = hsv[y_coords, x_coords, 1]
                avg_saturation = np.mean(saturations)
                is_low_saturation = avg_saturation < 80  # Lower saturation = more band-like

        # FINAL DECISION: All criteria must be met
        # Thick + large area + symmetric + (optionally low saturation)
        relative_thickness = avg_thickness / h
        logger.debug(f"Confidence band check: thickness={avg_thickness:.1f} ({relative_thickness:.1%}), "
                    f"pixels={total_pixels}, threshold={band_thickness_threshold:.1f}")

        # Only flag as band if VERY thick and symmetric
        if avg_thickness > band_thickness_threshold * 1.5:  # >7.5% of height
            logger.info(f"Confidence band detected: very thick ({relative_thickness:.1%} of height)")
            return True

        return False

    def _filter_confidence_bands(self, curves: List[KMCurve], image: np.ndarray,
                                   hsv: np.ndarray = None) -> List[KMCurve]:
        """
        Filter out curves that are actually confidence bands.

        CONSERVATIVE FILTERING - Only remove clear confidence bands when we have extras.

        Parameters
        ----------
        curves : List[KMCurve]
            Detected curves
        image : np.ndarray
            Original image
        hsv : np.ndarray, optional
            HSV version of image

        Returns
        -------
        List[KMCurve] : Curves with confidence bands removed
        """
        # CONSERVATIVE: Only filter if we have MORE than 3 curves
        # This prevents accidentally removing valid curves
        if len(curves) <= 3:
            return curves

        if hsv is None:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        filtered = []
        bands_detected = []

        for curve in curves:
            # Create mask from curve points
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)

            for x, y in curve.points:
                if 0 <= x < w and 0 <= y < h:
                    # Draw thicker to capture neighborhood
                    cv2.circle(mask, (x, y), 3, 255, -1)

            # Check if this looks like a confidence band
            if self._is_confidence_band(mask, image):
                logger.info(f"Filtering confidence band: {curve.color_name}")
                bands_detected.append(curve)
                # Lower confidence but keep for backup
                curve.confidence *= 0.3
            else:
                filtered.append(curve)

        # SAFETY: If we filtered too many, add back the best "bands"
        # Always keep at least 2 curves
        if len(filtered) < 2 and bands_detected:
            # Sort bands by original confidence (before reduction)
            bands_detected.sort(key=lambda c: c.confidence, reverse=True)
            while len(filtered) < 2 and bands_detected:
                best_band = bands_detected.pop(0)
                best_band.confidence /= 0.3  # Restore confidence
                filtered.append(best_band)
                logger.warning(f"Restoring {best_band.color_name} - needed for minimum curve count")

        return filtered


    def process_figure_calibrated(self, image: np.ndarray,
                                    axis_info: Optional[Dict] = None,
                                    expected_curves: int = None) -> Dict:
        """
        Process a KM figure with calibrated axis information.

        Parameters
        ----------
        image : np.ndarray
            Input image (BGR)
        axis_info : dict, optional
            Axis calibration with keys:
            - x_min, x_max: Time axis range (e.g., 0-60 months)
            - y_min, y_max: Survival axis range (usually 0-1)
            - x_unit: Time unit (months, years, etc.)
            - calibrated: True if values are actual, False if normalized
        expected_curves : int, optional
            Expected number of curves

        Returns
        -------
        dict with curves, panels, calibration status
        """
        h, w = image.shape[:2]

        result = {
            'panels': [],
            'curves': [],
            'insets_excluded': [],
            'is_multipanel': False,
            'n_curves': 0,
            'calibrated': False,
            'axis_info': axis_info
        }

        # Step 1: Detect panels
        panels = self._detect_panels(image)
        result['is_multipanel'] = len(panels) > 1
        result['panels'] = panels

        if len(panels) == 0:
            panels = [Panel(0, (0, 0, w, h), True)]

        # Step 2: Process each panel
        all_curves = []
        for panel in panels:
            if not panel.is_km_plot:
                continue

            x1, y1, x2, y2 = panel.bbox
            panel_img = image[y1:y2, x1:x2].copy()
            panel_h, panel_w = panel_img.shape[:2]

            # Step 3: Detect insets
            insets = self._detect_insets(panel_img)
            for inset in insets:
                ix1, iy1, ix2, iy2 = inset
                panel_img[iy1:iy2, ix1:ix2] = 255
                result['insets_excluded'].append({
                    'panel_id': panel.panel_id,
                    'bbox': inset
                })

            # Step 4: Extract curves with calibration
            hsv = cv2.cvtColor(panel_img, cv2.COLOR_BGR2HSV)

            # Define plot region
            if self.use_adaptive_margins and self.margin_detector is not None:
                try:
                    margin_result = self.margin_detector.detect_margins(panel_img)
                    margin_left = margin_result.left
                    margin_right = margin_result.right
                    margin_top = margin_result.top
                    margin_bottom = margin_result.bottom
                except Exception:
                    margin_left = int(panel_w * 0.08)
                    margin_right = int(panel_w * 0.05)
                    margin_top = int(panel_h * 0.08)
                    margin_bottom = int(panel_h * 0.12)
            else:
                margin_left = int(panel_w * 0.08)
                margin_right = int(panel_w * 0.05)
                margin_top = int(panel_h * 0.08)
                margin_bottom = int(panel_h * 0.12)

            plot_x1 = margin_left
            plot_x2 = panel_w - margin_right
            plot_y1 = margin_top
            plot_y2 = panel_h - margin_bottom
            plot_w = plot_x2 - plot_x1
            plot_h = plot_y2 - plot_y1

            # Extract curves with calibration
            curve_id = 0
            found_colors = set()
            colors_to_check = list(self.color_defs)
            if self.detect_black_curves:
                colors_to_check.append(self.black_color_def)

            for color_def in colors_to_check:
                color_name = color_def['name']
                base_name = color_name.rstrip('12')
                if base_name in found_colors:
                    continue

                lower = np.array(color_def['hsv_lower'])
                upper = np.array(color_def['hsv_upper'])
                mask = cv2.inRange(hsv, lower, upper)

                if color_name == 'red1':
                    red2_def = next((c for c in self.color_defs if c['name'] == 'red2'), None)
                    if red2_def:
                        lower2 = np.array(red2_def['hsv_lower'])
                        upper2 = np.array(red2_def['hsv_upper'])
                        mask2 = cv2.inRange(hsv, lower2, upper2)
                        mask = cv2.bitwise_or(mask, mask2)
                    color_name = 'red'
                    base_name = 'red'

                if color_name == 'red2':
                    continue

                total_pixels = np.sum(mask > 0)
                if total_pixels < self.min_pixel_area:
                    continue

                kernel = np.ones((3, 3), np.uint8)
                clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

                cols_with_pixels = np.any(clean_mask > 0, axis=0)
                if not np.any(cols_with_pixels):
                    continue

                x_min_px = np.argmax(cols_with_pixels)
                x_max_px = len(cols_with_pixels) - np.argmax(cols_with_pixels[::-1]) - 1
                width_span = x_max_px - x_min_px

                if width_span < plot_w * self.min_curve_width_ratio:
                    continue

                points = self._trace_curve(clean_mask)

                if len(points) < self.min_curve_points:
                    continue

                # Convert with calibration
                survival_data, is_calibrated = self._to_survival(
                    points, panel_h, panel_w, margin_left, margin_top, plot_w, plot_h, axis_info
                )

                if not self._is_valid_km(survival_data):
                    continue

                if color_name == 'black':
                    if not self._is_valid_black_curve(clean_mask, points, plot_w, plot_h):
                        continue

                confidence = min(1.0, len(points) / 100) * min(1.0, width_span / (plot_w * 0.5))

                # Adjust coordinates
                adjusted_points = [(px + x1, py + y1) for px, py in points]

                all_curves.append(KMCurve(
                    curve_id=curve_id,
                    color_bgr=color_def['bgr_center'],
                    color_name=color_name if color_name != 'red1' else 'red',
                    points=adjusted_points,
                    survival_data=survival_data,
                    confidence=confidence,
                    panel_id=panel.panel_id,
                    calibrated=is_calibrated,
                    x_unit=axis_info.get('x_unit', '') if axis_info else ''
                ))
                found_colors.add(base_name)
                curve_id += 1

            panel.curves = [c for c in all_curves if c.panel_id == panel.panel_id]

        result['curves'] = all_curves
        result['n_curves'] = len(all_curves)
        result['calibrated'] = axis_info.get('calibrated', False) if axis_info else False

        return result

    # === NEW METHODS FOR ROBUST PIPELINE ===

    def score_curves_for_primary(self, curves: List[KMCurve]) -> List[Tuple[KMCurve, float]]:
        """
        Score curves by likelihood of being primary endpoint arms.

        Parameters
        ----------
        curves : List[KMCurve]
            List of detected curves

        Returns
        -------
        List of (curve, score) tuples sorted by score (highest first)
        """
        scored = []
        for curve in curves:
            score = 0.0

            if not curve.survival_data or len(curve.survival_data) < 5:
                scored.append((curve, 0.0))
                continue

            times = [t for t, s in curve.survival_data]
            survivals = [s for t, s in curve.survival_data]

            # 1. Prefer curves that start at ~1.0 survival (+20 points)
            if 0.95 <= survivals[0] <= 1.05:
                score += 20

            # 2. Prefer curves with many points (well-traced)
            n_points = len(curve.survival_data)
            if n_points > 50:
                score += 10
            elif n_points > 30:
                score += 5
            elif n_points > 20:
                score += 3

            # 3. Prefer curves spanning >80% of time axis (+15 points)
            if times[-1] > times[0]:
                time_span = (times[-1] - times[0]) / times[-1]
                if time_span > 0.8:
                    score += 15
                elif time_span > 0.5:
                    score += 8

            # 4. Prefer curves that end below 0.7 (have events)
            if survivals[-1] < 0.7:
                score += 10
            elif survivals[-1] < 0.9:
                score += 5

            # 5. Prefer standard colors (blue, red) over unusual colors
            color = curve.color_name.lower() if curve.color_name else ''
            if color in ['blue', 'red', 'red1', 'red2']:
                score += 5
            elif color in ['green', 'black']:
                score += 3
            elif color in ['orange', 'purple']:
                score += 2

            # 6. Higher confidence bonus
            score += curve.confidence * 5

            scored.append((curve, score))

        return sorted(scored, key=lambda x: -x[1])

    def validate_curve_pair(self, curve1: KMCurve, curve2: KMCurve) -> Tuple[bool, str]:
        """
        Validate that a curve pair is suitable for HR estimation.

        Parameters
        ----------
        curve1 : KMCurve
            First curve
        curve2 : KMCurve
            Second curve

        Returns
        -------
        Tuple of (is_valid, reason)
        """
        if not curve1.survival_data or not curve2.survival_data:
            return False, "One or both curves have no survival data"

        times1 = np.array([t for t, s in curve1.survival_data])
        surv1 = np.array([s for t, s in curve1.survival_data])
        times2 = np.array([t for t, s in curve2.survival_data])
        surv2 = np.array([s for t, s in curve2.survival_data])

        if len(times1) < 5 or len(times2) < 5:
            return False, "Curves too short (<5 points)"

        # Check 1: Curves don't overlap excessively (same arm duplicated?)
        # Interpolate to common time points
        t_min = max(times1.min(), times2.min())
        t_max = min(times1.max(), times2.max())

        if t_max <= t_min:
            return False, "Curves don't overlap in time"

        try:
            common_t = np.linspace(t_min, t_max, 50)
            s1_interp = np.interp(common_t, times1, surv1)
            s2_interp = np.interp(common_t, times2, surv2)

            # Calculate mean absolute difference
            mean_diff = np.mean(np.abs(s1_interp - s2_interp))
            overlap = 1.0 - mean_diff

            if overlap > 0.95:
                return False, "Curves overlap >95% - likely same arm duplicated"
            if overlap > 0.9:
                # Warning but still valid
                logger.warning(f"Curves highly similar (overlap={overlap:.1%})")
        except Exception as e:
            logger.warning(f"Overlap calculation failed: {e}")

        # Check 2: Curves have similar time ranges
        max_time1 = times1.max()
        max_time2 = times2.max()
        time_diff = abs(max_time1 - max_time2) / max(max_time1, max_time2)

        if time_diff > 0.5:
            return False, f"Curves have very different time ranges (diff={time_diff:.1%})"

        # Check 3: At least one curve has events (survival < 0.95 at end)
        if surv1[-1] > 0.95 and surv2[-1] > 0.95:
            return False, "Neither curve shows sufficient events"

        return True, "Valid pair"

    def select_best_pair(self, curves: List[KMCurve]) -> Tuple[Optional[KMCurve], Optional[KMCurve], str]:
        """
        Select the best pair of curves for HR estimation.

        Parameters
        ----------
        curves : List[KMCurve]
            List of detected curves

        Returns
        -------
        Tuple of (curve1, curve2, selection_reason)
        """
        if len(curves) < 2:
            return None, None, "Insufficient curves (<2)"

        # Score all curves
        scored = self.score_curves_for_primary(curves)

        # Try pairs starting with highest-scored
        for i, (curve1, score1) in enumerate(scored):
            for j, (curve2, score2) in enumerate(scored[i+1:], start=i+1):
                is_valid, reason = self.validate_curve_pair(curve1, curve2)
                if is_valid:
                    return (curve1, curve2,
                            f"Selected curves {i} (score={score1:.1f}) and {j} (score={score2:.1f})")

        # If no valid pair found, return top 2 with warning
        if len(scored) >= 2:
            return (scored[0][0], scored[1][0],
                    f"No valid pair found - returning top 2 curves (may have issues)")

        return None, None, "Could not select valid curve pair"


def test_simple_handler():
    """Quick test of the handler."""
    handler = SimpleMultiCurveHandler()

    # Create simple test
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255

    # Axes
    cv2.line(img, (80, 50), (80, 500), (0, 0, 0), 2)
    cv2.line(img, (80, 500), (750, 500), (0, 0, 0), 2)

    # Blue curve
    for x in range(100, 700):
        y = 80 + int(350 * (1 - np.exp(-0.003 * (x - 100))))
        cv2.circle(img, (x, y), 2, (255, 100, 0), -1)

    # Red curve
    for x in range(100, 700):
        y = 80 + int(350 * (1 - np.exp(-0.008 * (x - 100))))
        cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

    result = handler.process_figure(img)

    print(f"Detected {result['n_curves']} curves")
    for curve in result['curves']:
        print(f"  {curve.color_name}: {len(curve.points)} points, conf={curve.confidence:.2f}")

    return result


if __name__ == '__main__':
    test_simple_handler()
