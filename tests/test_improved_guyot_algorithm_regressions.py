"""Regression tests for improved_guyot_algorithm.

Covers the empty-input guard (previously raised IndexError on times[0]) and a
basic monotone reconstruction. The refinement loop depends on lifelines, so the
basic case runs with max_iterations=0 to keep this test dependency-light.
"""
import numpy as np

from improved_guyot_algorithm import (
    improved_guyot_reconstruction,
    estimate_events_in_interval,
)


def test_empty_input_returns_empty_result():
    """Empty time/survival arrays must not raise IndexError."""
    result = improved_guyot_reconstruction(np.array([]), np.array([]), 100)
    assert result.n_patients == 0
    assert result.n_events == 0
    assert result.convergence is False
    assert result.iterations == 0


def test_basic_reconstruction_produces_patients():
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    survival = np.array([1.0, 0.9, 0.8, 0.7, 0.6])
    result = improved_guyot_reconstruction(
        times, survival, n_total=100, max_iterations=0
    )
    # No refinement (max_iterations=0) so iterations is reported as 0.
    assert result.iterations == 0
    # Some patients should be reconstructed from a declining curve.
    assert result.n_patients > 0
    # Events cannot exceed reconstructed patients.
    assert 0 <= result.n_events <= result.n_patients


def test_events_in_interval_bounds():
    # No survival drop -> no events.
    d, c = estimate_events_in_interval(0.8, 0.8, 50, 50)
    assert d == 0
    # Survival drop -> some events, bounded by n_start.
    d, c = estimate_events_in_interval(0.8, 0.6, 50, 40)
    assert 0 <= d <= 50
    assert c >= 0


def test_population_conserved_with_nar_table_under_censoring():
    """2026-07 audit fix: the final-censoring fill must conserve N.

    Previously the tail padded round(survival[-1] * n_total) survivors ON TOP of the
    interval-loop bodies, double-counting under a real number-at-risk table (n_total=100
    inflated to ~130). The fill must be (n_total - already_generated).
    """
    times = np.array([0.0, 10, 20, 30, 40, 50])
    survival = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    result = improved_guyot_reconstruction(
        times, survival, n_total=100,
        n_risk_times=times, n_risk_values=np.array([100, 70, 50, 35, 25, 20]),
        max_iterations=0,
    )
    assert result.n_patients == 100


def test_refinement_never_appends_bodies_when_curve_reaches_zero():
    """2026-07 audit fix: the refinement loop must RELABEL, never append new bodies.

    On a curve that reaches S=0 (no final-censoring pad) the old else-branch appended
    brand-new event rows, inflating n_patients above n_total (8 -> 10). Population must
    stay bounded by n_total.
    """
    times = np.array([0.0, 1.0, 2.0, 3.0])
    survival = np.array([1.0, 0.4, 0.1, 0.0])
    result = improved_guyot_reconstruction(times, survival, n_total=8)
    assert result.n_patients <= 8
    assert result.n_events <= 8
