"""Real-world robustness regression: the extractors must stay world-class on figures with the
features that break digitizers -- translucent CI confidence bands in the curve's own hue,
serif fonts, dashed line styles, and dense gridlines. Run: pytest benchmark/tests/test_hard_robustness.py
"""
import os, sys, tempfile
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import synth_km_generator as G
from km_metrics import km_from_ipd, plot_metrics
from vector_km_extractor import extract_vector_km
from faithful_guyot import reconstruct_arm_faithful

try:
    import easyocr  # noqa: F401
    from raster_km_extractor import extract_raster_km
    HAVE_OCR = True
except Exception:
    HAVE_OCR = False

COL = ["#1f77b4", "#d62728", "#2ca02c"]


def _render_hard(rows, meta, tmp, seed=0):
    """Render a hard figure: serif font + CI bands (curve hue, translucent) + dashed arms + grid."""
    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    H = meta["horizon"]
    for a in range(meta["n_arms"]):
        ar = [r for r in rows if r["arm"] == a]
        et, sv = km_from_ipd([r["time"] for r in ar], [r["status"] for r in ar])
        et = np.append(et, H); sv = np.append(sv, sv[-1])
        ax.step(et, sv, where="post", color=COL[a], lw=1.8,
                ls="--" if a == 1 else "-", label=meta["arms"][a]["label"])
        lo = np.clip(sv - 0.06, 0, 1); hi = np.clip(sv + 0.06, 0, 1)
        ax.fill_between(et, lo, hi, step="post", color=COL[a], alpha=0.18, lw=0)
    ax.set_xlabel("Time (months)"); ax.set_ylabel("Survival probability")
    ax.set_xlim(0, H); ax.set_ylim(0, 1); ax.legend(frameon=False); ax.grid(True, alpha=0.35)
    fig.tight_layout()
    pdf = os.path.join(tmp, "h.pdf"); png = os.path.join(tmp, "h.png")
    fig.savefig(pdf); fig.savefig(png, dpi=150); plt.close(fig)
    plt.rcParams["font.family"] = "sans-serif"
    return pdf, png


def _iae(ex, rows, H):
    if not ex or ex["n_arms"] < 2:
        return None
    recon = []
    for a in ex["arms"]:
        role = a["role"] if a["role"] is not None else 0
        N = len([r for r in rows if r["arm"] == role])
        ipd = reconstruct_arm_faithful(np.array(a["times"]), np.array(a["survivals"]), N, follow_up=H)
        for r in ipd:
            recon.append({"time": r["time"], "status": r["status"], "arm": role})
    return plot_metrics(rows, recon)["iae"]


def test_vector_robust_to_ci_bands_serif_dashed():
    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(5)
        rows, meta = G.simulate_trial(2, 300, "administrative", rng)
        pdf, png = _render_hard(rows, meta, tmp)
        iae = _iae(extract_vector_km(pdf), rows, meta["horizon"])
        assert iae is not None and iae < 0.01, f"vector IAE {iae} on hard figure"


@pytest.mark.skipif(not HAVE_OCR, reason="easyocr not installed")
def test_raster_robust_to_ci_bands_serif_dashed():
    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(6)
        rows, meta = G.simulate_trial(2, 300, "administrative", rng)
        pdf, png = _render_hard(rows, meta, tmp)
        iae = _iae(extract_raster_km(png), rows, meta["horizon"])
        # translucent CI band has low saturation -> excluded by the HSV mask; median tracer holds
        assert iae is not None and iae < 0.02, f"raster IAE {iae} on hard figure"
