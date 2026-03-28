import json

import numpy as np

from pdf_extractor import ExtractedFigure, PDFExtractionResult, PDFExtractor


def test_save_figures_metadata_matches_saved_figures_with_km_filter(tmp_path):
    extractor = object.__new__(PDFExtractor)
    img = np.zeros((10, 10, 3), dtype=np.uint8)

    figures = [
        ExtractedFigure(
            page_number=0,
            figure_index=0,
            image=img,
            bbox=(0, 0, 10, 10),
            width=10,
            height=10,
            is_potential_km=False,
            confidence=0.2,
        ),
        ExtractedFigure(
            page_number=3,
            figure_index=1,
            image=img,
            bbox=(0, 0, 10, 10),
            width=10,
            height=10,
            is_potential_km=True,
            confidence=0.8,
        ),
        ExtractedFigure(
            page_number=7,
            figure_index=2,
            image=img,
            bbox=(0, 0, 10, 10),
            width=10,
            height=10,
            is_potential_km=True,
            confidence=0.9,
        ),
    ]

    result = PDFExtractionResult(
        pdf_path="paper.pdf",
        total_pages=12,
        total_figures=3,
        km_candidates=2,
        figures=figures,
        extraction_time=0.1,
    )

    saved_paths = extractor.save_figures(result, str(tmp_path), km_only=True)
    assert len(saved_paths) == 2

    metadata_path = tmp_path / "paper_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert [f["page"] for f in metadata["figures"]] == [3, 7]
    assert [f["is_km"] for f in metadata["figures"]] == [True, True]
    assert [f["confidence"] for f in metadata["figures"]] == [0.8, 0.9]


def test_save_figures_skips_failed_writes(monkeypatch, tmp_path):
    extractor = object.__new__(PDFExtractor)
    img = np.zeros((10, 10, 3), dtype=np.uint8)

    figures = [
        ExtractedFigure(
            page_number=1,
            figure_index=0,
            image=img,
            bbox=(0, 0, 10, 10),
            width=10,
            height=10,
            is_potential_km=True,
            confidence=0.7,
        ),
        ExtractedFigure(
            page_number=2,
            figure_index=1,
            image=img,
            bbox=(0, 0, 10, 10),
            width=10,
            height=10,
            is_potential_km=True,
            confidence=0.9,
        ),
    ]
    result = PDFExtractionResult(
        pdf_path="paper.pdf",
        total_pages=2,
        total_figures=2,
        km_candidates=2,
        figures=figures,
        extraction_time=0.1,
    )

    calls = {"n": 0}

    def _fake_imwrite(path, image):
        calls["n"] += 1
        return calls["n"] == 2

    monkeypatch.setattr("pdf_extractor.cv2.imwrite", _fake_imwrite)

    saved_paths = extractor.save_figures(result, str(tmp_path), km_only=True)
    assert len(saved_paths) == 1

    metadata = json.loads((tmp_path / "paper_metadata.json").read_text(encoding="utf-8"))
    assert len(metadata["figures"]) == 1
    assert metadata["figures"][0]["page"] == 2
