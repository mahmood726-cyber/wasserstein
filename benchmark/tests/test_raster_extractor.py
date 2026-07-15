"""Test the raster-OCR KM extractor (scanned-figure path). Renders a raster figure, extracts
it via easyocr + sub-pixel HSV tracing, asserts near-exact recovery + legend orientation.
Slow (easyocr/torch on CPU). Run: pytest benchmark/tests/test_raster_extractor.py
"""
import os, sys, tempfile
import numpy as np
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import synth_km_generator as G
from faithful_guyot import reconstruct_arm_faithful
from km_metrics import plot_metrics

try:
    import easyocr  # noqa: F401
    import raster_km_extractor as RK
    extract_raster_km = RK.extract_raster_km
    HAVE_OCR = True
except Exception:
    HAVE_OCR = False


@pytest.mark.skipif(not HAVE_OCR, reason="easyocr not installed")
def test_raster_ocr_recovers_curves_and_orientation():
    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(11)
        rows, meta = G.simulate_trial(2, 250, "administrative", rng)
        pdf = os.path.join(tmp, "p.pdf"); png = os.path.join(tmp, "p.png")
        G.render_figure(rows, meta, pdf, png, "raster_150dpi", "full_0_1", True)
        ex = extract_raster_km(png)
        assert ex is not None and ex["n_arms"] == 2
        assert ex["orientation_from_legend"] is True
        recon = []
        for a in ex["arms"]:
            role = a["role"]
            N = len([r for r in rows if r["arm"] == role])
            ipd = reconstruct_arm_faithful(np.array(a["times"]), np.array(a["survivals"]), N,
                                           follow_up=meta["horizon"])
            for r in ipd:
                recon.append({"time": r["time"], "status": r["status"], "arm": role})
        pm = plot_metrics(rows, recon)
        assert pm["iae"] < 0.02, f"raster-OCR IAE {pm['iae']} should beat the 0.018 target"


def test_raster_legend_phrase_groups_multiword_labels():
    import raster_km_extractor as RK

    labels = [
        {"text": "Usual", "x0": 100.0, "x1": 126.0, "cx": 113.0, "cy": 20.0},
        {"text": "care", "x0": 131.0, "x1": 151.0, "cx": 141.0, "cy": 20.5},
        {"text": "Active", "x0": 100.0, "x1": 130.0, "cx": 115.0, "cy": 40.0},
        {"text": "treatment", "x0": 135.0, "x1": 185.0, "cx": 160.0, "cy": 40.0},
    ]
    rows = RK._label_rows(labels)
    roles = {row["text"]: RK._role_from_label(row["text"]) for row in rows}
    assert roles["Usual care"] == 0
    assert roles["Active treatment"] == 1
