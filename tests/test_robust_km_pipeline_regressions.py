import pytest

pytest.importorskip("fitz")
pytest.importorskip("cv2")

import robust_km_pipeline as rkp


class _DummyCurve:
    def __init__(self, color_name):
        self.color_name = color_name
        self.survival_data = [
            (0.0, 1.00),
            (1.0, 0.95),
            (2.0, 0.87),
            (3.0, 0.78),
            (4.0, 0.69),
            (5.0, 0.60),
        ]


class _DummyHandler:
    def process_figure(self, img):
        if img == "embedded_page_0":
            return {"curves": [_DummyCurve("blue")]}
        if img == "raster_page_0":
            return {"curves": [_DummyCurve("red")]}
        return {"curves": []}


def test_single_embedded_curve_does_not_skip_rasterized_scan(monkeypatch):
    monkeypatch.setattr(rkp, "HAS_NAR_DETECTOR", False)
    monkeypatch.setattr(rkp, "HAS_NAR_OCR", False)

    pipeline = rkp.RobustKMPipeline(min_curve_points=5)
    handler = _DummyHandler()

    curves = pipeline._extract_all_curves(
        pages=["raster_page_0"],
        handler=handler,
        embedded_figures=[{"page": 0, "img": "embedded_page_0"}],
    )

    assert len(curves) == 2
    assert {c["color_name"] for c in curves} == {"blue", "red"}
