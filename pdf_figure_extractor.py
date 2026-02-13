"""
PDF Figure Extractor
====================

Extract figures and images from PDF documents for KM curve analysis.

Supports:
- Embedded images (JPEG, PNG)
- Vector graphics conversion
- Page-to-image rendering
- Figure region detection

Author: Wasserstein KM Extractor Team
Date: January 2026
Version: 1.0
"""

import os
import io
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

# PDF libraries
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

import cv2


@dataclass
class ExtractedFigure:
    """Represents an extracted figure from a PDF."""
    image: np.ndarray
    page_number: int
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    width: int
    height: int
    dpi: int
    source_method: str  # 'pymupdf', 'pdfplumber', 'pdf2image', 'page_render'
    figure_index: int
    caption: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PDFExtractionResult:
    """Result of PDF figure extraction."""
    pdf_path: str
    total_pages: int
    figures: List[ExtractedFigure]
    page_images: List[np.ndarray]  # Full page renders
    extraction_method: str
    success: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class PDFFigureExtractor:
    """
    Extract figures from PDF documents.

    Tries multiple methods in order of preference:
    1. PyMuPDF (fitz) - Best for embedded images
    2. pdfplumber - Good for structured PDFs
    3. pdf2image - Renders full pages as images
    """

    def __init__(self, dpi: int = 200, min_figure_size: int = 100):
        """
        Initialize extractor.

        Parameters
        ----------
        dpi : int
            Resolution for page rendering (default: 200)
        min_figure_size : int
            Minimum width/height to consider as a figure (default: 100px)
        """
        self.dpi = dpi
        self.min_figure_size = min_figure_size
        self._check_dependencies()

    def _check_dependencies(self):
        """Check available PDF libraries."""
        self.available_methods = []
        if PYMUPDF_AVAILABLE:
            self.available_methods.append('pymupdf')
        if PDFPLUMBER_AVAILABLE:
            self.available_methods.append('pdfplumber')
        if PDF2IMAGE_AVAILABLE:
            self.available_methods.append('pdf2image')

        if not self.available_methods:
            raise ImportError(
                "No PDF library available. Install one of: "
                "PyMuPDF (fitz), pdfplumber, pdf2image"
            )

    def extract(self, pdf_path: str) -> PDFExtractionResult:
        """
        Extract all figures from a PDF.

        Parameters
        ----------
        pdf_path : str
            Path to PDF file

        Returns
        -------
        PDFExtractionResult
            Contains extracted figures and page images
        """
        if not os.path.exists(pdf_path):
            return PDFExtractionResult(
                pdf_path=pdf_path,
                total_pages=0,
                figures=[],
                page_images=[],
                extraction_method='none',
                success=False,
                errors=[f"File not found: {pdf_path}"]
            )

        # Try PyMuPDF first (best quality)
        if PYMUPDF_AVAILABLE:
            try:
                return self._extract_with_pymupdf(pdf_path)
            except Exception as e:
                pass  # Fall through to next method

        # Try pdfplumber
        if PDFPLUMBER_AVAILABLE:
            try:
                return self._extract_with_pdfplumber(pdf_path)
            except Exception as e:
                pass

        # Fall back to pdf2image (page rendering)
        if PDF2IMAGE_AVAILABLE:
            try:
                return self._extract_with_pdf2image(pdf_path)
            except Exception as e:
                return PDFExtractionResult(
                    pdf_path=pdf_path,
                    total_pages=0,
                    figures=[],
                    page_images=[],
                    extraction_method='failed',
                    success=False,
                    errors=[str(e)]
                )

        return PDFExtractionResult(
            pdf_path=pdf_path,
            total_pages=0,
            figures=[],
            page_images=[],
            extraction_method='none',
            success=False,
            errors=["No extraction method available"]
        )

    def _extract_with_pymupdf(self, pdf_path: str) -> PDFExtractionResult:
        """Extract using PyMuPDF (fitz)."""
        doc = fitz.open(pdf_path)
        figures = []
        page_images = []
        errors = []
        warnings = []

        for page_num in range(len(doc)):
            page = doc[page_num]

            # Render full page
            try:
                pix = page.get_pixmap(dpi=self.dpi)
                page_img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )
                if pix.n == 4:  # RGBA
                    page_img = cv2.cvtColor(page_img, cv2.COLOR_RGBA2BGR)
                elif pix.n == 3:  # RGB
                    page_img = cv2.cvtColor(page_img, cv2.COLOR_RGB2BGR)
                page_images.append(page_img)
            except Exception as e:
                warnings.append(f"Page {page_num + 1} render failed: {e}")
                page_images.append(None)

            # Extract embedded images
            try:
                image_list = page.get_images()
                for img_index, img_info in enumerate(image_list):
                    try:
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]

                        # Convert to numpy array
                        img_array = np.frombuffer(image_bytes, dtype=np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                        if img is not None and img.shape[0] >= self.min_figure_size and img.shape[1] >= self.min_figure_size:
                            # Get image position on page
                            for item in page.get_image_info():
                                if item.get('xref') == xref:
                                    bbox = item.get('bbox', (0, 0, img.shape[1], img.shape[0]))
                                    break
                            else:
                                bbox = (0, 0, img.shape[1], img.shape[0])

                            figures.append(ExtractedFigure(
                                image=img,
                                page_number=page_num + 1,
                                bbox=bbox,
                                width=img.shape[1],
                                height=img.shape[0],
                                dpi=self.dpi,
                                source_method='pymupdf',
                                figure_index=len(figures),
                                metadata={'xref': xref, 'format': base_image.get('ext', 'unknown')}
                            ))
                    except Exception as e:
                        warnings.append(f"Image {img_index} on page {page_num + 1} failed: {e}")
            except Exception as e:
                warnings.append(f"Image extraction on page {page_num + 1} failed: {e}")

        doc.close()

        # If no embedded images found, detect figures from page renders
        if not figures and page_images:
            figures = self._detect_figures_from_pages(page_images)

        return PDFExtractionResult(
            pdf_path=pdf_path,
            total_pages=len(page_images),
            figures=figures,
            page_images=[p for p in page_images if p is not None],
            extraction_method='pymupdf',
            success=True,
            errors=errors,
            warnings=warnings
        )

    def _extract_with_pdfplumber(self, pdf_path: str) -> PDFExtractionResult:
        """Extract using pdfplumber."""
        figures = []
        page_images = []
        errors = []
        warnings = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Render page to image
                try:
                    img = page.to_image(resolution=self.dpi)
                    # Convert PIL to numpy
                    page_img = np.array(img.original)
                    if len(page_img.shape) == 3 and page_img.shape[2] == 4:
                        page_img = cv2.cvtColor(page_img, cv2.COLOR_RGBA2BGR)
                    elif len(page_img.shape) == 3 and page_img.shape[2] == 3:
                        page_img = cv2.cvtColor(page_img, cv2.COLOR_RGB2BGR)
                    page_images.append(page_img)
                except Exception as e:
                    warnings.append(f"Page {page_num + 1} render failed: {e}")

                # Extract images from page
                try:
                    if hasattr(page, 'images'):
                        for img_index, img_info in enumerate(page.images):
                            try:
                                # pdfplumber provides image data directly
                                img_data = img_info.get('stream', None)
                                if img_data:
                                    img_bytes = img_data.get_data()
                                    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                                    if img is not None and img.shape[0] >= self.min_figure_size:
                                        bbox = (
                                            img_info.get('x0', 0),
                                            img_info.get('top', 0),
                                            img_info.get('x1', img.shape[1]),
                                            img_info.get('bottom', img.shape[0])
                                        )
                                        figures.append(ExtractedFigure(
                                            image=img,
                                            page_number=page_num + 1,
                                            bbox=bbox,
                                            width=img.shape[1],
                                            height=img.shape[0],
                                            dpi=self.dpi,
                                            source_method='pdfplumber',
                                            figure_index=len(figures)
                                        ))
                            except Exception as e:
                                warnings.append(f"Image {img_index} on page {page_num + 1}: {e}")
                except Exception as e:
                    pass  # Some pages may not have images

        # If no embedded images, detect from page renders
        if not figures and page_images:
            figures = self._detect_figures_from_pages(page_images)

        return PDFExtractionResult(
            pdf_path=pdf_path,
            total_pages=len(page_images),
            figures=figures,
            page_images=page_images,
            extraction_method='pdfplumber',
            success=len(page_images) > 0,
            errors=errors,
            warnings=warnings
        )

    def _extract_with_pdf2image(self, pdf_path: str) -> PDFExtractionResult:
        """Extract using pdf2image (full page rendering)."""
        figures = []
        page_images = []
        errors = []
        warnings = []

        try:
            # Convert PDF pages to images
            pil_images = convert_from_path(pdf_path, dpi=self.dpi)

            for page_num, pil_img in enumerate(pil_images):
                # Convert PIL to numpy BGR
                img_array = np.array(pil_img)
                if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                    page_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                else:
                    page_img = img_array
                page_images.append(page_img)

            # Detect figures from page images
            figures = self._detect_figures_from_pages(page_images)

        except Exception as e:
            errors.append(f"pdf2image conversion failed: {e}")

        return PDFExtractionResult(
            pdf_path=pdf_path,
            total_pages=len(page_images),
            figures=figures,
            page_images=page_images,
            extraction_method='pdf2image',
            success=len(page_images) > 0,
            errors=errors,
            warnings=warnings
        )

    def _detect_figures_from_pages(self, page_images: List[np.ndarray]) -> List[ExtractedFigure]:
        """
        Detect figure regions from full page images.

        Uses contour detection to find rectangular regions that
        might be figures/charts.
        """
        figures = []

        for page_num, page_img in enumerate(page_images):
            if page_img is None:
                continue

            # Convert to grayscale
            gray = cv2.cvtColor(page_img, cv2.COLOR_BGR2GRAY)

            # Edge detection
            edges = cv2.Canny(gray, 50, 150)

            # Dilate to connect nearby edges
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=2)

            # Find contours
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                # Filter by size (figures are typically substantial)
                if w >= self.min_figure_size and h >= self.min_figure_size:
                    # Filter by aspect ratio (figures are typically not too extreme)
                    aspect_ratio = w / h
                    if 0.2 < aspect_ratio < 5.0:
                        # Extract region
                        figure_img = page_img[y:y+h, x:x+w].copy()

                        # Check if this looks like a figure (not just text)
                        if self._is_likely_figure(figure_img):
                            figures.append(ExtractedFigure(
                                image=figure_img,
                                page_number=page_num + 1,
                                bbox=(x, y, x + w, y + h),
                                width=w,
                                height=h,
                                dpi=self.dpi,
                                source_method='page_detect',
                                figure_index=len(figures),
                                confidence=0.7  # Lower confidence for detected regions
                            ))

        return figures

    def _is_likely_figure(self, img: np.ndarray) -> bool:
        """
        Check if an image region is likely a figure (not just text).

        Figures typically have:
        - More color variation than text
        - Horizontal/vertical lines (axes)
        - Less text density
        """
        if img is None or img.size == 0:
            return False

        # Check color variation
        if len(img.shape) == 3:
            # Color image - check for colored pixels
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1]
            color_ratio = np.sum(saturation > 30) / saturation.size
            if color_ratio > 0.05:  # Has some colored content
                return True

        # Check for lines (axes)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        edges = cv2.Canny(gray, 50, 150)

        # Detect horizontal and vertical lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
        if lines is not None and len(lines) > 2:
            return True

        # Check for low text density (figures have more white space)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        white_ratio = np.sum(binary == 255) / binary.size
        if 0.3 < white_ratio < 0.95:  # Not all text, not all white
            return True

        return False

    def extract_figure_at_coordinates(self, pdf_path: str, page_num: int,
                                       bbox: Tuple[float, float, float, float]) -> Optional[ExtractedFigure]:
        """
        Extract a specific region from a PDF page.

        Parameters
        ----------
        pdf_path : str
            Path to PDF
        page_num : int
            Page number (1-indexed)
        bbox : tuple
            Bounding box (x0, y0, x1, y1) in PDF coordinates

        Returns
        -------
        ExtractedFigure or None
        """
        result = self.extract(pdf_path)
        if not result.success or page_num > len(result.page_images):
            return None

        page_img = result.page_images[page_num - 1]
        x0, y0, x1, y1 = [int(c * self.dpi / 72) for c in bbox]  # Convert PDF points to pixels

        # Ensure bounds are valid
        y0, y1 = max(0, y0), min(page_img.shape[0], y1)
        x0, x1 = max(0, x0), min(page_img.shape[1], x1)

        if x1 <= x0 or y1 <= y0:
            return None

        figure_img = page_img[y0:y1, x0:x1].copy()

        return ExtractedFigure(
            image=figure_img,
            page_number=page_num,
            bbox=bbox,
            width=x1 - x0,
            height=y1 - y0,
            dpi=self.dpi,
            source_method='manual_crop',
            figure_index=0
        )


def test_pdf_extraction(pdf_path: str):
    """Test PDF extraction on a file."""
    print(f"Testing PDF extraction: {pdf_path}")
    print("=" * 60)

    extractor = PDFFigureExtractor(dpi=150)
    result = extractor.extract(pdf_path)

    print(f"Success: {result.success}")
    print(f"Method: {result.extraction_method}")
    print(f"Total pages: {result.total_pages}")
    print(f"Figures found: {len(result.figures)}")
    print(f"Page images: {len(result.page_images)}")

    if result.errors:
        print(f"Errors: {result.errors}")
    if result.warnings:
        print(f"Warnings: {result.warnings[:5]}...")  # First 5

    for i, fig in enumerate(result.figures[:5]):  # First 5 figures
        print(f"\nFigure {i + 1}:")
        print(f"  Page: {fig.page_number}")
        print(f"  Size: {fig.width} x {fig.height}")
        print(f"  Method: {fig.source_method}")

    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_pdf_extraction(sys.argv[1])
    else:
        print("Usage: python pdf_figure_extractor.py <pdf_path>")
