"""
Figure Type Classifier
======================

Classify extracted figures by type (KM curve, forest plot, etc.)

Author: Wasserstein KM Extractor Team
Date: January 2026
Version: 1.0
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict
from enum import Enum
import re

# Try to import pytesseract for OCR-based detection
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


class FigureType(Enum):
    """Types of figures commonly found in medical papers."""
    KAPLAN_MEIER = "kaplan_meier"
    FOREST_PLOT = "forest_plot"
    BAR_CHART = "bar_chart"
    LINE_CHART = "line_chart"
    SCATTER_PLOT = "scatter_plot"
    FLOW_DIAGRAM = "flow_diagram"
    TABLE_IMAGE = "table_image"
    UNKNOWN = "unknown"


@dataclass
class ClassificationResult:
    """Result of figure classification."""
    figure_type: FigureType
    confidence: float
    features: Dict[str, float]
    is_survival_curve: bool
    secondary_type: Optional[FigureType] = None


class FigureClassifier:
    """
    Classify figures by type using image features.

    Detection Strategy:
    - Kaplan-Meier: Step function patterns, survival axis (0-1 or 0-100%)
    - Forest plot: Diamond markers, horizontal lines, vertical reference line
    - Bar chart: Vertical/horizontal bars
    - Scatter: Random dot patterns
    """

    def __init__(self, use_ocr: bool = False):
        """
        Initialize classifier.

        Parameters
        ----------
        use_ocr : bool
            If True, use OCR for keyword detection (slower but more accurate).
            Default False for speed.
        """
        self.min_confidence = 0.5
        self.use_ocr = use_ocr

    def classify(self, image: np.ndarray) -> ClassificationResult:
        """
        Classify a figure image.

        Parameters
        ----------
        image : np.ndarray
            BGR image of the figure

        Returns
        -------
        ClassificationResult
            Classification with confidence and features
        """
        if image is None or image.size == 0:
            return ClassificationResult(
                figure_type=FigureType.UNKNOWN,
                confidence=0.0,
                features={},
                is_survival_curve=False
            )

        # Extract features
        features = self._extract_features(image)

        # Score each type
        scores = {
            FigureType.KAPLAN_MEIER: self._score_kaplan_meier(image, features),
            FigureType.FOREST_PLOT: self._score_forest_plot(image, features),
            FigureType.BAR_CHART: self._score_bar_chart(image, features),
            FigureType.LINE_CHART: self._score_line_chart(image, features),
            FigureType.SCATTER_PLOT: self._score_scatter_plot(image, features),
            FigureType.TABLE_IMAGE: self._score_table(image, features),
        }

        # Find best match
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]

        # Get secondary type
        sorted_types = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        secondary = sorted_types[1][0] if len(sorted_types) > 1 else None

        # Determine if survival curve (lowered threshold for real PDF detection)
        # CRITICAL: Reject text-heavy images which are often misclassified as KM
        text_density = features.get('text_density', 0)
        image_size = features.get('size', 0)
        min_km_size = 200 * 200  # At least 200x200 pixels for a real KM figure

        # ENHANCED: Apply forest plot rejection rule
        # If forest_score > km_score, this is NOT a KM curve
        km_score = scores.get(FigureType.KAPLAN_MEIER, 0)
        forest_score = scores.get(FigureType.FOREST_PLOT, 0)

        forest_rejected = forest_score > km_score and forest_score > 0.4

        is_survival = (best_type == FigureType.KAPLAN_MEIER and
                       best_score > 0.5 and
                       text_density < 0.25 and    # Reject if >25% text
                       image_size > min_km_size and  # Reject small images
                       not forest_rejected)  # Reject if forest plot score higher

        # Override best_type if forest plot rejection triggered
        if forest_rejected and best_type == FigureType.KAPLAN_MEIER:
            best_type = FigureType.FOREST_PLOT
            best_score = forest_score

        if best_score < self.min_confidence:
            best_type = FigureType.UNKNOWN

        return ClassificationResult(
            figure_type=best_type,
            confidence=best_score,
            features=features,
            is_survival_curve=is_survival,
            secondary_type=secondary
        )

    def _extract_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract features for classification."""
        features = {}

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape[:2]

        # Basic features
        features['aspect_ratio'] = w / h if h > 0 else 1.0
        features['size'] = w * h

        # Edge features
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / edges.size

        # Line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)

        if lines is not None:
            features['n_lines'] = len(lines)

            # Classify lines by orientation
            horizontal = 0
            vertical = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if angle < 15 or angle > 165:
                    horizontal += 1
                elif 75 < angle < 105:
                    vertical += 1

            features['horizontal_lines'] = horizontal
            features['vertical_lines'] = vertical
            features['h_v_ratio'] = horizontal / (vertical + 1)
        else:
            features['n_lines'] = 0
            features['horizontal_lines'] = 0
            features['vertical_lines'] = 0
            features['h_v_ratio'] = 1.0

        # Color features
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            features['mean_saturation'] = np.mean(hsv[:, :, 1])
            features['color_variance'] = np.var(hsv[:, :, 0])

            # Count distinct colors
            colors = self._count_distinct_colors(image)
            features['n_colors'] = colors
        else:
            features['mean_saturation'] = 0
            features['color_variance'] = 0
            features['n_colors'] = 1

        # Step function detection (for KM curves)
        features['step_score'] = self._detect_step_pattern(gray)

        # Text density
        features['text_density'] = self._estimate_text_density(gray)

        return features

    def _score_kaplan_meier(self, image: np.ndarray, features: Dict) -> float:
        """Score likelihood of being a Kaplan-Meier curve."""
        score = 0.0

        # KM curves typically have:
        # 1. Step function pattern (HIGHEST weight - most distinctive KM feature)
        step_score = features.get('step_score', 0)
        score += step_score * 0.45  # Increased from 0.3 - step pattern is key indicator

        # 2. Aspect ratio (pages are typically portrait or landscape)
        ar = features.get('aspect_ratio', 1.0)
        if 0.5 < ar < 2.5:  # Relaxed
            score += 0.08

        # 3. Both horizontal and vertical lines (axes)
        h_lines = features.get('horizontal_lines', 0)
        v_lines = features.get('vertical_lines', 0)
        if h_lines > 0 and v_lines > 0:
            score += 0.08

        # 4. Has colored content (relaxed - full pages have many colors)
        n_colors = features.get('n_colors', 0)
        if n_colors >= 2:  # Relaxed from 2-10
            score += 0.06

        # 5. Some colored content (curves are usually colored)
        if features.get('mean_saturation', 0) > 10:  # Relaxed from 15
            score += 0.06

        # 6. Not extremely text-heavy
        if features.get('text_density', 1) < 0.4:  # Relaxed from 0.35
            score += 0.04

        # 7. Detect declining curve pattern
        declining_score = self._detect_declining_pattern(image)
        score += declining_score * 0.1

        # 8. Detect multiple colored curves
        multi_curve_score = self._detect_multiple_curves(image)
        score += multi_curve_score * 0.08

        # 9. High step score bonus (strong KM indicator even without other features)
        if step_score > 0.5:
            score += 0.15  # Bonus for high step score - distinctive KM feature

        # 10. OCR-based survival keyword detection (high weight, but slow)
        if self.use_ocr:
            keyword_score = self._detect_survival_keywords(image)
            score += keyword_score * 0.25

        return min(1.0, score)

    def _score_forest_plot(self, image: np.ndarray, features: Dict) -> float:
        """
        Score likelihood of being a forest plot.

        Enhanced detection for:
        - Vertical reference line at HR=1.0
        - Many small horizontal lines with diamonds
        - High text density (>30%)
        - Tall/narrow aspect ratio (>2.0)
        """
        score = 0.0

        # Forest plots typically have:
        # 1. Many horizontal lines (effect estimates)
        h_lines = features.get('horizontal_lines', 0)
        if h_lines > 5:
            score += 0.25
        if h_lines > 10:
            score += 0.1  # Extra bonus for many lines

        # 2. Vertical reference line at HR=1.0 (null effect)
        v_lines = features.get('vertical_lines', 0)
        if v_lines >= 1:
            score += 0.1
        # Enhanced: Detect prominent central vertical line
        central_line_score = self._detect_central_vertical_line(image)
        score += central_line_score * 0.15

        # 3. Aspect ratio >2.0 (forest plots are typically tall/narrow)
        ar = features.get('aspect_ratio', 1.0)
        if ar > 2.0:
            score += 0.15
        elif ar > 1.5:
            score += 0.08
        elif ar < 0.5:
            score += 0.08  # Or very wide

        # 4. Diamond markers (summary effect)
        diamond_score = self._detect_diamond_markers(image)
        score += diamond_score * 0.25

        # 5. HIGH text density (>30% is strong forest plot indicator)
        text_density = features.get('text_density', 0)
        if text_density > 0.3:
            score += 0.2
        elif text_density > 0.2:
            score += 0.1

        # 6. Low step pattern (not survival curve)
        if features.get('step_score', 1) < 0.3:
            score += 0.1

        # 7. Many small horizontal segments (effect estimate bars)
        small_h_segments = self._detect_small_horizontal_segments(image)
        score += small_h_segments * 0.15

        # 8. Low color variety (forest plots usually black/white)
        n_colors = features.get('n_colors', 10)
        if n_colors < 5:
            score += 0.05

        return min(1.0, score)

    def _detect_central_vertical_line(self, image: np.ndarray) -> float:
        """
        Detect a prominent central vertical line (HR=1.0 reference).

        Forest plots have a vertical line at HR=1.0 that runs the height of the plot.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape

        # Look at vertical column sums in central region
        center_region = gray[:, w//3:2*w//3]

        # Find columns with high contrast (dark line on light background)
        col_variance = np.var(center_region, axis=0)
        col_darkness = np.mean(255 - center_region, axis=0)

        # Strong vertical line = high variance and relatively dark
        if len(col_variance) > 0:
            max_var_idx = np.argmax(col_variance)
            if col_variance[max_var_idx] > 1000 and col_darkness[max_var_idx] > 50:
                return 0.8  # Strong central line detected

            # Check for any prominent vertical structures
            prominent_lines = np.sum(col_variance > np.mean(col_variance) + 2 * np.std(col_variance))
            if prominent_lines > 0:
                return 0.4

        return 0.0

    def _detect_small_horizontal_segments(self, image: np.ndarray) -> float:
        """
        Detect many small horizontal line segments (effect estimate bars).

        Forest plots have many short horizontal lines representing CIs.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Detect horizontal lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 20, minLineLength=10, maxLineGap=5)

        if lines is None:
            return 0.0

        # Count short horizontal lines (10-150 pixels)
        short_h_lines = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            # Horizontal and short
            if (angle < 15 or angle > 165) and 10 < length < 150:
                short_h_lines += 1

        # Many short horizontal lines suggests forest plot
        if short_h_lines > 20:
            return 0.8
        elif short_h_lines > 10:
            return 0.5
        elif short_h_lines > 5:
            return 0.3

        return 0.0

    def _score_bar_chart(self, image: np.ndarray, features: Dict) -> float:
        """Score likelihood of being a bar chart."""
        score = 0.0

        # Bar charts have:
        # 1. Many vertical OR horizontal lines (bars)
        h_lines = features.get('horizontal_lines', 0)
        v_lines = features.get('vertical_lines', 0)
        if v_lines > 3 or h_lines > 3:
            score += 0.3

        # 2. Distinct colored regions
        if features.get('n_colors', 0) >= 2:
            score += 0.2

        # 3. High saturation (colored bars)
        if features.get('mean_saturation', 0) > 40:
            score += 0.2

        # 4. Low step pattern
        if features.get('step_score', 1) < 0.2:
            score += 0.15

        # 5. Not too many lines overall
        if features.get('n_lines', 100) < 50:
            score += 0.15

        return min(1.0, score)

    def _score_line_chart(self, image: np.ndarray, features: Dict) -> float:
        """Score likelihood of being a general line chart."""
        score = 0.0

        # Line charts have:
        # 1. Lines present
        if features.get('n_lines', 0) > 2:
            score += 0.15

        # 2. Colors
        if features.get('n_colors', 0) >= 2:
            score += 0.15

        # 3. Axes (horizontal and vertical)
        if features.get('horizontal_lines', 0) > 0 and features.get('vertical_lines', 0) > 0:
            score += 0.15

        # 4. Low step pattern (smooth lines) - but not TOO low (that's scatter)
        step = features.get('step_score', 0)
        if 0.1 < step < 0.5:
            score += 0.15

        # 5. Reasonable aspect ratio
        ar = features.get('aspect_ratio', 1.0)
        if 0.5 < ar < 2.5:
            score += 0.1

        # PENALTY: If this looks like a KM curve (declining pattern), reduce line_chart score
        declining_score = self._detect_declining_pattern(image)
        multi_curve = self._detect_multiple_curves(image)
        if declining_score > 0.4 and multi_curve > 0.4:
            score -= 0.2  # Strong KM indicators, penalize line_chart

        return max(0.0, min(0.8, score))  # Cap at 0.8 to allow KM to win

    def _score_scatter_plot(self, image: np.ndarray, features: Dict) -> float:
        """Score likelihood of being a scatter plot."""
        score = 0.0

        # Scatter plots have:
        # 1. Low line count (dots, not lines)
        if features.get('n_lines', 100) < 20:
            score += 0.3

        # 2. Points detected
        point_score = self._detect_scatter_points(image)
        score += point_score * 0.4

        # 3. Axes present
        if features.get('horizontal_lines', 0) > 0 and features.get('vertical_lines', 0) > 0:
            score += 0.15

        # 4. Low step pattern
        if features.get('step_score', 1) < 0.2:
            score += 0.15

        return min(1.0, score)

    def _score_table(self, image: np.ndarray, features: Dict) -> float:
        """Score likelihood of being a table image."""
        score = 0.0

        # Tables have:
        # 1. Many horizontal lines
        if features.get('horizontal_lines', 0) > 3:
            score += 0.3

        # 2. Many vertical lines
        if features.get('vertical_lines', 0) > 3:
            score += 0.3

        # 3. High text density
        if features.get('text_density', 0) > 0.3:
            score += 0.2

        # 4. Low color variation
        if features.get('mean_saturation', 100) < 20:
            score += 0.1

        # 5. Grid pattern
        if features.get('horizontal_lines', 0) > 2 and features.get('vertical_lines', 0) > 2:
            h = features.get('horizontal_lines', 0)
            v = features.get('vertical_lines', 0)
            if 0.3 < h / (v + 1) < 3.0:
                score += 0.1

        return min(1.0, score)

    def _detect_step_pattern(self, gray: np.ndarray) -> float:
        """Detect step function pattern characteristic of KM curves."""
        h, w = gray.shape

        # Find horizontal segments by looking for constant-y regions
        edges = cv2.Canny(gray, 50, 150)

        # Look for horizontal edges followed by vertical edges
        horizontal_kernel = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
        vertical_kernel = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)

        h_edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        v_edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)

        # Step pattern has alternating H and V edges
        h_count = np.sum(h_edges > 0)
        v_count = np.sum(v_edges > 0)

        if h_count + v_count == 0:
            return 0.0

        # Ratio should be roughly balanced for step functions
        ratio = min(h_count, v_count) / max(h_count, v_count)

        # Also check for the step-down pattern (survival curves go down)
        # Look at right-to-left trend
        left_half = gray[:, :w//2]
        right_half = gray[:, w//2:]
        if np.mean(left_half) < np.mean(right_half):  # Curve going down left-to-right
            ratio *= 1.2

        return min(1.0, ratio)

    def _detect_declining_pattern(self, image: np.ndarray) -> float:
        """
        Detect declining curve pattern characteristic of survival curves.
        KM curves typically start high (top-left) and decline to lower right.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape

        # Exclude margins (axes area)
        margin_x = int(w * 0.15)
        margin_y_top = int(h * 0.1)
        margin_y_bottom = int(h * 0.15)

        plot_area = gray[margin_y_top:h-margin_y_bottom, margin_x:w-margin_x]
        if plot_area.size == 0:
            return 0.0

        ph, pw = plot_area.shape

        # Divide into vertical strips and find the "content center" in each
        n_strips = 10
        strip_width = pw // n_strips
        centers = []

        for i in range(n_strips):
            strip = plot_area[:, i*strip_width:(i+1)*strip_width]
            # Find where content is (darker pixels = curve content)
            col_profile = np.mean(strip, axis=1)
            # Look for darkest region
            dark_threshold = np.percentile(col_profile, 30)
            dark_rows = np.where(col_profile < dark_threshold)[0]
            if len(dark_rows) > 0:
                # Center of dark content
                center_y = np.mean(dark_rows)
                centers.append(center_y)
            else:
                centers.append(ph / 2)

        if len(centers) < 2:
            return 0.0

        # Check if centers trend downward (in image coords, down = increasing y)
        # But for survival, high survival = top of plot, so we want centers to increase
        centers = np.array(centers)

        # Calculate trend
        x_vals = np.arange(len(centers))
        if len(centers) > 2:
            slope, _ = np.polyfit(x_vals, centers, 1)
            # Positive slope means curve going down (survival decreasing)
            if slope > 0:
                # Normalize: slope of 5+ pixels per strip is strong decline
                return min(1.0, slope / 5.0)

        return 0.0

    def _detect_multiple_curves(self, image: np.ndarray) -> float:
        """
        Detect presence of multiple colored curves (common in KM plots with 2+ arms).
        """
        if len(image.shape) != 3:
            return 0.0

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w = image.shape[:2]

        # Exclude margins
        margin_x = int(w * 0.15)
        margin_y_top = int(h * 0.1)
        margin_y_bottom = int(h * 0.15)

        plot_hsv = hsv[margin_y_top:h-margin_y_bottom, margin_x:w-margin_x]
        if plot_hsv.size == 0:
            return 0.0

        # Find distinct hue clusters with sufficient saturation
        saturation_mask = plot_hsv[:, :, 1] > 50  # Reasonably saturated
        value_mask = plot_hsv[:, :, 2] > 30  # Not too dark
        color_mask = saturation_mask & value_mask

        if np.sum(color_mask) < 100:
            return 0.0

        hues = plot_hsv[:, :, 0][color_mask]

        # Count distinct hue peaks
        hist, bins = np.histogram(hues, bins=18, range=(0, 180))  # 10-degree bins

        # Find significant peaks (>5% of max)
        threshold = max(hist) * 0.15
        n_peaks = np.sum(hist > threshold)

        # 2-4 distinct hue peaks suggests multiple curves
        if 2 <= n_peaks <= 6:
            return min(1.0, n_peaks / 4.0)
        elif n_peaks == 1:
            return 0.3  # Single curve still possible

        return 0.0

    def _detect_survival_keywords(self, image: np.ndarray) -> float:
        """
        Use OCR to detect survival-related keywords in the figure.
        Returns a score 0-1 based on presence of KM-related terms.
        """
        if not TESSERACT_AVAILABLE:
            return 0.0

        try:
            # Convert to grayscale for OCR
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

            # Enhance contrast for better OCR
            gray = cv2.equalizeHist(gray)

            # Run OCR
            text = pytesseract.image_to_string(gray, config='--psm 6').lower()

            # Define survival-related keywords with weights
            keywords = {
                'survival': 0.25,
                'probability': 0.20,
                'at risk': 0.25,
                'number at risk': 0.30,
                'hazard': 0.15,
                'kaplan': 0.30,
                'meier': 0.30,
                'months': 0.10,
                'years': 0.10,
                'time': 0.05,
                'event': 0.10,
                'death': 0.15,
                'mortality': 0.15,
                'freedom': 0.15,  # "freedom from event"
                'cumulative': 0.15,
                '%': 0.05,
            }

            score = 0.0
            for keyword, weight in keywords.items():
                if keyword in text:
                    score += weight

            return min(1.0, score)

        except Exception:
            return 0.0

    def _detect_diamond_markers(self, image: np.ndarray) -> float:
        """Detect diamond-shaped markers (forest plot summary)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Diamond detection using template matching or contour shape
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        diamond_count = 0
        for contour in contours:
            # Approximate contour
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Diamonds have 4 vertices
            if len(approx) == 4:
                # Check if roughly diamond-shaped (symmetric)
                area = cv2.contourArea(contour)
                if 50 < area < 5000:  # Reasonable size for marker
                    diamond_count += 1

        # Normalize by image size
        score = min(1.0, diamond_count / 10)
        return score

    def _detect_scatter_points(self, image: np.ndarray) -> float:
        """Detect scattered point pattern."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Use blob detection
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 5
        params.maxArea = 500
        params.filterByCircularity = True
        params.minCircularity = 0.3

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(gray)

        # Many small blobs suggest scatter plot
        n_points = len(keypoints)
        score = min(1.0, n_points / 50)  # 50+ points is high confidence
        return score

    def _count_distinct_colors(self, image: np.ndarray, n_bins: int = 8) -> int:
        """Count number of distinct colors in image."""
        if len(image.shape) != 3:
            return 1

        # Quantize colors
        quantized = (image // (256 // n_bins)).astype(np.uint8)

        # Count unique color combinations
        colors = quantized.reshape(-1, 3)
        unique = np.unique(colors, axis=0)

        return len(unique)

    def _estimate_text_density(self, gray: np.ndarray) -> float:
        """Estimate how much of the image is text."""
        # Text typically has high contrast and specific patterns
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Connected components - text has many small components
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(255 - binary)

        # Count small components (likely text)
        text_pixels = 0
        for i in range(1, n_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if 10 < area < 500:  # Character-sized
                text_pixels += area

        return text_pixels / gray.size


def classify_figures(images: List[np.ndarray]) -> List[ClassificationResult]:
    """Classify a list of figure images."""
    classifier = FigureClassifier()
    return [classifier.classify(img) for img in images]


if __name__ == "__main__":
    # Test with a sample image
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        if img is not None:
            classifier = FigureClassifier()
            result = classifier.classify(img)
            print(f"Type: {result.figure_type.value}")
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Is survival curve: {result.is_survival_curve}")
            print(f"Features: {result.features}")
        else:
            print(f"Could not load image: {sys.argv[1]}")
    else:
        print("Usage: python figure_classifier.py <image_path>")
