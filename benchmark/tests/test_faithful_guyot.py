"""Tests for the faithful Guyot port (L3). Pins population conservation, monotonicity, and
anchor behaviour that the heuristic violated. Run: pytest benchmark/tests/test_faithful_guyot.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from faithful_guyot import reconstruct_arm_faithful, pava_decreasing
from km_metrics import km_from_ipd


def test_pava_enforces_non_increasing():
    y = pava_decreasing([1.0, 0.7, 0.8, 0.5])   # 0.8 violates
    for i in range(1, len(y)):
        assert y[i] <= y[i - 1] + 1e-9


def test_population_conserved_exactly_no_nar():
    grid = np.linspace(0, 24, 100)
    S = np.clip(1 - grid / 40.0, 0, 1)          # smooth decline to ~0.4
    for N in (25, 137, 1000):
        ipd = reconstruct_arm_faithful(grid, S, N, follow_up=24)
        assert len(ipd) == N, f"expected {N} rows, got {len(ipd)}"


def test_population_conserved_with_nar():
    grid = np.linspace(0, 50, 100)
    S = np.clip(1 - 0.01 * grid, 0, 1)          # 1.0 -> 0.5
    nar_t = [0, 10, 20, 30, 40, 50]
    nar_v = [200, 170, 140, 110, 85, 60]
    ipd = reconstruct_arm_faithful(grid, S, 200, nar_times=np.array(nar_t, float),
                                   nar_values=np.array(nar_v, float), follow_up=50)
    assert len(ipd) == 200


def test_reconstructed_km_is_monotone_non_increasing():
    grid = np.linspace(0, 30, 100)
    S = np.clip(np.exp(-grid / 20.0), 0, 1)
    ipd = reconstruct_arm_faithful(grid, S, 300, follow_up=30)
    ets, svs = km_from_ipd([r["time"] for r in ipd], [r["status"] for r in ipd])
    for i in range(1, len(svs)):
        assert svs[i] <= svs[i - 1] + 1e-9


def test_reconstructed_curve_tracks_target():
    # a reconstructed KM should sit close to the target survival at the midpoint
    grid = np.linspace(0, 20, 100)
    S = np.clip(1 - grid / 40.0, 0, 1)          # S(10) = 0.75
    ipd = reconstruct_arm_faithful(grid, S, 500, follow_up=20)
    ets, svs = km_from_ipd([r["time"] for r in ipd], [r["status"] for r in ipd])
    idx = np.searchsorted(ets, 10.0, side="right") - 1
    assert abs(svs[max(0, idx)] - 0.75) < 0.05
