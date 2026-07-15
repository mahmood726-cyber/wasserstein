"""Tests for the vector-path KM extractor (L8). Generates a vector figure, extracts it, and
asserts near-exact digitization + legend-based orientation. Run: pytest benchmark/tests/
"""
import os, sys, json, tempfile
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import synth_km_generator as G
from vector_km_extractor import extract_vector_km
from faithful_guyot import reconstruct_arm_faithful
from km_metrics import plot_metrics


def _make_plot(tmp, arms=2, n=150, cens="administrative", yaxis="full_0_1", seed=7):
    rng = np.random.default_rng(seed)
    rows, meta = G.simulate_trial(arms, n, cens, rng)
    pdf = os.path.join(tmp, "p.pdf"); png = os.path.join(tmp, "p.png")
    G.render_figure(rows, meta, pdf, png, "clean_600dpi", yaxis, True)
    return rows, meta, pdf


def test_vector_extract_recovers_arms_and_orientation():
    with tempfile.TemporaryDirectory() as tmp:
        rows, meta, pdf = _make_plot(tmp, arms=2)
        ex = extract_vector_km(pdf)
        assert ex is not None and ex["n_arms"] == 2
        assert ex["orientation_from_legend"] is True
        roles = sorted(a["role"] for a in ex["arms"])
        assert roles == [0, 1], f"legend roles not resolved: {roles}"
        for a in ex["arms"]:
            s = np.array(a["survivals"])
            assert s[0] > 0.98, "survival should start at ~1.0"
            assert all(s[i] <= s[i - 1] + 1e-9 for i in range(1, len(s))), "must be non-increasing"


def test_vector_extract_axis_calibration_matches_horizon():
    with tempfile.TemporaryDirectory() as tmp:
        rows, meta, pdf = _make_plot(tmp, arms=2)
        ex = extract_vector_km(pdf)
        # x-axis calibrated from real ticks: max tick near the plotted horizon
        assert abs(ex["x_range"][1] - round(meta["horizon"] / 10) * 10) <= 12


def test_vector_path_reconstruction_is_near_exact():
    with tempfile.TemporaryDirectory() as tmp:
        rows, meta, pdf = _make_plot(tmp, arms=2, n=250)
        ex = extract_vector_km(pdf)
        recon = []
        for a in ex["arms"]:
            role = a["role"]
            N = len([r for r in rows if r["arm"] == role])
            ipd = reconstruct_arm_faithful(np.array(a["times"]), np.array(a["survivals"]), N,
                                           follow_up=meta["horizon"])
            for r in ipd:
                recon.append({"time": r["time"], "status": r["status"], "arm": role})
        pm = plot_metrics(rows, recon)
        assert pm["iae"] < 0.02, f"vector IAE {pm['iae']} should beat the 0.018 SOTA target"


def test_percent_axis_detected_and_scaled():
    with tempfile.TemporaryDirectory() as tmp:
        rows, meta, pdf = _make_plot(tmp, arms=2, yaxis="full_0_100pct")
        ex = extract_vector_km(pdf)
        assert ex["y_is_percent"] is True
        for a in ex["arms"]:
            assert max(a["survivals"]) <= 1.001, "percent axis must be scaled back to [0,1]"


def test_vector_legend_roles_use_multiword_labels(monkeypatch):
    class Pt:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    class Page:
        def get_drawings(self):
            return [
                {"color": (0, 0, 1), "items": [("m", Pt(30, 20)), ("l", Pt(80, 40)), ("l", Pt(130, 60))]},
                {"color": (1, 0, 0), "items": [("m", Pt(30, 22)), ("l", Pt(80, 45)), ("l", Pt(130, 70))]},
                {"color": (0, 0, 1), "items": [("m", Pt(150, 30)), ("l", Pt(164, 30))]},
                {"color": (1, 0, 0), "items": [("m", Pt(150, 47)), ("l", Pt(164, 47))]},
            ]

        def get_text(self, _kind):
            # x axis row: value increases with x
            # y axis column: value decreases with y because PDF y grows downward
            return [
                (26, 120, 34, 128, "0", 0, 0, 0),
                (76, 120, 84, 128, "10", 0, 0, 1),
                (126, 120, 134, 128, "20", 0, 0, 2),
                (6, 116, 14, 124, "0", 0, 1, 0),
                (6, 66, 14, 74, "0.5", 0, 1, 1),
                (6, 16, 14, 24, "1", 0, 1, 2),
                (170, 25, 195, 35, "Usual", 0, 2, 0),
                (200, 25, 220, 35, "care", 0, 2, 1),
                (170, 42, 198, 52, "Active", 0, 3, 0),
                (203, 42, 250, 52, "treatment", 0, 3, 1),
            ]

    class Doc:
        def __getitem__(self, _index):
            return Page()

        def close(self):
            return None

    import fitz
    monkeypatch.setattr(fitz, "open", lambda _path: Doc())

    ex = extract_vector_km("dummy.pdf")
    assert ex is not None
    assert ex["orientation_from_legend"] is True
    roles = sorted(a["role"] for a in ex["arms"])
    assert roles == [0, 1]
