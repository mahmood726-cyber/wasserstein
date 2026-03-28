"""
PDF Figure Extractor for Wasserstein KM Extractor v4.0

Extract KM figures directly from PDF publications:
1. Detect figure regions in PDF pages
2. Extract as high-resolution images
3. Filter for potential KM plots
4. Batch process multiple PDFs

Dependencies:
- PyMuPDF (fitz): pip install PyMuPDF
- pdf2image (optional): pip install pdf2image

Author: Wasserstein KM Extractor Team
Date: January 2026
"""

import os
import sys
import io
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# Try to import PDF libraries
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    print("Warning: PyMuPDF not installed. Install with: pip install PyMuPDF")

import numpy as np
import cv2


@dataclass
class ExtractedFigure:
    """An extracted figure from a PDF."""
    page_number: int
    figure_index: int
    image: np.ndarray
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    width: int
    height: int
    is_potential_km: bool
    confidence: float
    caption: Optional[str] = None


@dataclass
class PDFExtractionResult:
    """Result of extracting figures from a PDF."""
    pdf_path: str
    total_pages: int
    total_figures: int
    km_candidates: int
    figures: List[ExtractedFigure]
    extraction_time: float
    errors: List[str] = field(default_factory=list)


class PDFExtractor:
    """
    Extract figures from PDF documents.

    Focuses on identifying potential Kaplan-Meier survival curves.

    Usage:
        extractor = PDFExtractor()
        result = extractor.extract_from_pdf("paper.pdf")

        for fig in result.figures:
            if fig.is_potential_km:
                # Process with KM extractor
                km_result = handler.process_figure(fig.image)
    """

    # Minimum dimensions for a valid figure
    MIN_WIDTH = 200
    MIN_HEIGHT = 150

    # Aspect ratio bounds for KM plots (width/height)
    KM_ASPECT_MIN = 0.8
    KM_ASPECT_MAX = 2.5

    def __init__(
        self,
        min_width: int = 200,
        min_height: int = 150,
        dpi: int = 150,
        extract_captions: bool = True
    ):
        """
        Initialize PDF extractor.

        Args:
            min_width: Minimum figure width in pixels
            min_height: Minimum figure height in pixels
            dpi: Resolution for rendering PDF pages
            extract_captions: Attempt to extract figure captions
        """
        if not HAS_PYMUPDF:
            raise ImportError("PyMuPDF is required. Install with: pip install PyMuPDF")

        self.min_width = min_width
        self.min_height = min_height
        self.dpi = dpi
        self.extract_captions = extract_captions

    def extract_from_pdf(
        self,
        pdf_path: str,
        pages: Optional[List[int]] = None,
        filter_km_only: bool = False
    ) -> PDFExtractionResult:
        """
        Extract figures from a PDF file.

        Args:
            pdf_path: Path to PDF file
            pages: Specific pages to extract (None = all pages)
            filter_km_only: Only return potential KM plots

        Returns:
            PDFExtractionResult with extracted figures
        """
        import time
        start_time = time.time()

        figures = []
        errors = []

        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)

            # Determine which pages to process
            if pages is None:
                pages_to_process = range(total_pages)
            else:
                pages_to_process = [p for p in pages if 0 <= p < total_pages]

            for page_num in pages_to_process:
                try:
                    page_figures = self._extract_from_page(doc, page_num)
                    figures.extend(page_figures)
                except Exception as e:
                    errors.append(f"Page {page_num}: {str(e)}")

            doc.close()

        except Exception as e:
            errors.append(f"PDF open error: {str(e)}")
            total_pages = 0

        # Filter for KM candidates if requested
        if filter_km_only:
            figures = [f for f in figures if f.is_potential_km]

        km_candidates = sum(1 for f in figures if f.is_potential_km)

        return PDFExtractionResult(
            pdf_path=pdf_path,
            total_pages=total_pages,
            total_figures=len(figures),
            km_candidates=km_candidates,
            figures=figures,
            extraction_time=time.time() - start_time,
            errors=errors
        )

    def _extract_from_page(
        self,
        doc: 'fitz.Document',
        page_num: int
    ) -> List[ExtractedFigure]:
        """Extract figures from a single page."""
        page = doc[page_num]
        figures = []

        # Method 1: Extract embedded images
        image_list = page.get_images()

        for img_index, img_info in enumerate(image_list):
            try:
                xref = img_info[0]
                base_image = doc.extract_image(xref)

                if base_image:
                    image_bytes = base_image["image"]

                    # Convert to numpy array
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if img is not None:
                        h, w = img.shape[:2]

                        # Check minimum size
                        if w >= self.min_width and h >= self.min_height:
                            is_km, confidence = self._is_potential_km_plot(img)

                            figures.append(ExtractedFigure(
                                page_number=page_num,
                                figure_index=img_index,
                                image=img,
                                bbox=(0, 0, w, h),  # Embedded images don't have bbox
                                width=w,
                                height=h,
                                is_potential_km=is_km,
                                confidence=confidence
                            ))
            except Exception:
                continue

        # Method 2: Render page and detect figure regions
        # This catches figures that aren't embedded as separate images
        if len(figures) == 0:
            try:
                # Render page at specified DPI
                mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
                pix = page.get_pixmap(matrix=mat)

                # Convert to numpy array
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )

                # Convert to BGR if needed
                if pix.n == 4:  # RGBA
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                elif pix.n == 1:  # Grayscale
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                # Detect figure regions
                regions = self._detect_figure_regions(img)

                for idx, (x1, y1, x2, y2) in enumerate(regions):
                    region_img = img[y1:y2, x1:x2].copy()
                    h, w = region_img.shape[:2]

                    if w >= self.min_width and h >= self.min_height:
                        is_km, confidence = self._is_potential_km_plot(region_img)

                        figures.append(ExtractedFigure(
                            page_number=page_num,
                            figure_index=len(figures),
                            image=region_img,
                            bbox=(x1, y1, x2, y2),
                            width=w,
                            height=h,
                            is_potential_km=is_km,
                            confidence=confidence
                        ))
            except Exception:
                pass

        return figures

    def _detect_figure_regions(
        self,
        page_img: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect figure regions in a rendered page.

        Returns list of (x1, y1, x2, y2) bounding boxes.
        """
        h, w = page_img.shape[:2]
        gray = cv2.cvtColor(page_img, cv2.COLOR_BGR2GRAY)

        # Find non-white regions (potential figures)
        _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

        # Morphological operations to connect figure elements
        kernel = np.ones((20, 20), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=3)

        # Find contours
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        regions = []
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)

            # Filter by size and aspect ratio
            if cw >= self.min_width and ch >= self.min_height:
                aspect = cw / ch
                if 0.5 < aspect < 3.0:  # Reasonable figure aspect ratio
                    # Add some padding
                    pad = 10
                    x1 = max(0, x - pad)
                    y1 = max(0, y - pad)
                    x2 = min(w, x + cw + pad)
                    y2 = min(h, y + ch + pad)
                    regions.append((x1, y1, x2, y2))

        return regions

    def _is_potential_km_plot(self, img: np.ndarray) -> Tuple[bool, float]:
        """
        Determine if an image is likely a Kaplan-Meier plot.

        Returns (is_potential_km, confidence).

        Heuristics:
        1. Has axes (L-shaped dark lines)
        2. Has step-like patterns
        3. Has survival-like y-axis (0-1 or 0-100)
        4. Aspect ratio typical for KM plots
        """
        h, w = img.shape[:2]
        confidence = 0.0

        # Check aspect ratio
        aspect = w / h
        if self.KM_ASPECT_MIN <= aspect <= self.KM_ASPECT_MAX:
            confidence += 0.2

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Check for axes (dark L-shape in lower-left)
        left_strip = gray[:, :int(w * 0.15)]
        bottom_strip = gray[int(h * 0.85):, :]

        left_dark = np.mean(left_strip < 100)
        bottom_dark = np.mean(bottom_strip < 100)

        if left_dark > 0.01 and bottom_dark > 0.01:
            confidence += 0.3

        # Check for colored curves (non-gray pixels)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        colored_ratio = np.mean(saturation > 30)

        if 0.01 < colored_ratio < 0.3:
            confidence += 0.2

        # Check for step-like patterns (horizontal lines)
        edges = cv2.Canny(gray, 50, 150)
        horizontal_kernel = np.array([[1, 1, 1, 1, 1]])
        horizontal = cv2.filter2D(edges, -1, horizontal_kernel)
        horizontal_ratio = np.mean(horizontal > 100)

        if horizontal_ratio > 0.01:
            confidence += 0.2

        # Check for white/light background
        white_ratio = np.mean(gray > 240)
        if white_ratio > 0.5:
            confidence += 0.1

        is_km = confidence >= 0.5

        return is_km, min(1.0, confidence)

    def save_figures(
        self,
        result: PDFExtractionResult,
        output_dir: str,
        km_only: bool = True,
        format: str = 'png'
    ) -> List[str]:
        """
        Save extracted figures to disk.

        Args:
            result: PDFExtractionResult from extract_from_pdf
            output_dir: Directory to save images
            km_only: Only save potential KM plots
            format: Image format (png, jpg)

        Returns:
            List of saved file paths
        """
        os.makedirs(output_dir, exist_ok=True)

        saved_paths = []
        saved_figures = []
        pdf_name = os.path.splitext(os.path.basename(result.pdf_path))[0]

        for fig in result.figures:
            if km_only and not fig.is_potential_km:
                continue

            filename = f"{pdf_name}_p{fig.page_number}_f{fig.figure_index}.{format}"
            filepath = os.path.join(output_dir, filename)

            if cv2.imwrite(filepath, fig.image):
                saved_paths.append(filepath)
                saved_figures.append(fig)

        # Save metadata
        metadata = {
            'pdf_path': result.pdf_path,
            'extraction_date': datetime.now().isoformat(),
            'total_figures': result.total_figures,
            'km_candidates': result.km_candidates,
            'figures': [
                {
                    'filename': os.path.basename(p),
                    'page': fig.page_number,
                    'is_km': fig.is_potential_km,
                    'confidence': fig.confidence
                }
                for p, fig in zip(saved_paths, saved_figures)
            ]
        }

        metadata_path = os.path.join(output_dir, f"{pdf_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return saved_paths


def extract_figures_from_pdf(
    pdf_path: str,
    output_dir: Optional[str] = None,
    km_only: bool = True
) -> PDFExtractionResult:
    """
    Convenience function to extract figures from a PDF.

    Args:
        pdf_path: Path to PDF file
        output_dir: Optional directory to save images
        km_only: Only extract potential KM plots

    Returns:
        PDFExtractionResult
    """
    extractor = PDFExtractor()
    result = extractor.extract_from_pdf(pdf_path, filter_km_only=km_only)

    if output_dir:
        extractor.save_figures(result, output_dir, km_only=km_only)

    return result


def process_pdf_folder(
    folder_path: str,
    output_dir: str,
    km_only: bool = True
) -> Dict:
    """
    Process all PDFs in a folder.

    Args:
        folder_path: Folder containing PDFs
        output_dir: Directory to save extracted figures
        km_only: Only extract potential KM plots

    Returns:
        Summary dictionary
    """
    import glob

    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))

    results = {
        'total_pdfs': len(pdf_files),
        'total_figures': 0,
        'total_km_candidates': 0,
        'pdfs': []
    }

    extractor = PDFExtractor()

    for pdf_path in pdf_files:
        print(f"Processing: {os.path.basename(pdf_path)}")

        result = extractor.extract_from_pdf(pdf_path, filter_km_only=km_only)

        pdf_output = os.path.join(output_dir, os.path.splitext(os.path.basename(pdf_path))[0])
        saved = extractor.save_figures(result, pdf_output, km_only=km_only)

        results['total_figures'] += result.total_figures
        results['total_km_candidates'] += result.km_candidates
        results['pdfs'].append({
            'pdf': os.path.basename(pdf_path),
            'figures': result.total_figures,
            'km_candidates': result.km_candidates,
            'saved': len(saved)
        })

    # Save summary
    summary_path = os.path.join(output_dir, 'extraction_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)

    return results


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract KM figures from PDFs')
    parser.add_argument('input', help='PDF file or folder')
    parser.add_argument('-o', '--output', default='extracted_figures',
                       help='Output directory')
    parser.add_argument('--all', action='store_true',
                       help='Extract all figures, not just KM candidates')

    args = parser.parse_args()

    if os.path.isfile(args.input):
        result = extract_figures_from_pdf(
            args.input,
            args.output,
            km_only=not args.all
        )
        print(f"Extracted {result.km_candidates}/{result.total_figures} KM candidates")
    elif os.path.isdir(args.input):
        results = process_pdf_folder(
            args.input,
            args.output,
            km_only=not args.all
        )
        print(f"Processed {results['total_pdfs']} PDFs")
        print(f"Found {results['total_km_candidates']}/{results['total_figures']} KM candidates")
    else:
        print(f"Error: {args.input} not found")
        sys.exit(1)
