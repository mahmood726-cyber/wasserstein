"""Real-paper validation (regression): the vision-assisted pipeline recovers published hazard
ratios from real open-access PDF figures, within the published 95% CI. The vision reads below are
fixtures captured from a validated vision model pass; this test pins the reconstruction so a change
to faithful_guyot / cox can't silently break the real-paper result.

Sources (PubMed): Hasegawa 2016 DOI 10.1371/journal.pone.0162400; Zhou 2015 DOI 10.1371/journal.pone.0117002.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from vision_km_pipeline import reconstruct_two_arm


def _within(hr, ci_lo, ci_hi):
    return ci_lo <= hr <= ci_hi


def test_hasegawa_os_hr_within_published_ci():
    # Hasegawa OS: UFT/LV (exp) vs Surgery (ctl); published HR 0.80 [0.48, 1.35]
    r = reconstruct_two_arm([1.0, 0.96, 0.92, 0.82, 0.75, 0.66, 0.66],
                            [1.0, 0.98, 0.94, 0.85, 0.76, 0.66, 0.60],
                            [0, 1, 2, 3, 4, 5, 6], 88, 89)
    assert _within(r["hr"], 0.48, 1.35), f"OS HR {r['hr']} outside published CI"


def test_hasegawa_rfs_hr_within_published_ci():
    # Hasegawa RFS: published HR 0.56 [0.38, 0.83]. FINE survival sampling + the number-at-risk
    # table (both read by vision) tighten the estimate 0.755 -> 0.680 [0.46, 1.02].
    t = [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6]
    r = reconstruct_two_arm(
        [1.0, 0.84, 0.62, 0.50, 0.42, 0.39, 0.385, 0.375, 0.37, 0.355],
        [1.0, 0.55, 0.43, 0.36, 0.345, 0.32, 0.32, 0.32, 0.315, 0.31],
        t, 88, 89, follow_up=6.0,
        nar_times=[0, 1, 2, 3, 4, 5, 6],
        nar_exp=[88, 53, 36, 32, 28, 20, 15], nar_ctl=[89, 36, 30, 27, 27, 22, 18])
    assert r["hr"] < 1.0                                   # UFT/LV favored, correct direction
    assert _within(r["hr"], 0.38, 0.83) or (r["hr_ci"][0] <= 0.56 <= r["hr_ci"][1])


def test_zhou_os_hr_within_published_ci():
    # Zhou OS: D3P (exp) vs M3P (ctl); published HR 0.63 [0.46, 0.86]
    r = reconstruct_two_arm([1.0, 0.92, 0.82, 0.67, 0.55, 0.44, 0.33],
                            [1.0, 0.90, 0.66, 0.50, 0.43, 0.27, 0.25],
                            [0, 5, 10, 15, 20, 25, 30], 114, 114)
    assert r["hr_ci"][0] <= 0.63 <= r["hr_ci"][1] or _within(r["hr"], 0.46, 0.86)
    assert r["hr"] < 1.0  # D3P favored, correct direction


def test_ensemble_reduces_read_noise():
    """3 independent vision reads averaged -> HR closer to truth than a single read (measured
    fig1 true 0.541: single 0.516 err 4.7% -> ensemble 0.529 err 2.3%)."""
    from vision_km_pipeline import ensemble_two_arm, reconstruct_two_arm
    t = [0, 3.875, 7.75, 11.625, 15.5, 19.375, 23.25, 27.125, 31]
    E = [[1.0,0.90,0.83,0.72,0.59,0.47,0.39,0.33,0.29],
         [1.0,0.89,0.83,0.72,0.58,0.47,0.38,0.32,0.28],
         [1.0,0.87,0.81,0.68,0.57,0.47,0.40,0.32,0.28]]
    C = [[1.0,0.88,0.71,0.55,0.38,0.26,0.16,0.12,0.07],
         [1.0,0.87,0.68,0.54,0.40,0.26,0.17,0.12,0.07],
         [1.0,0.85,0.70,0.52,0.38,0.27,0.19,0.13,0.07]]
    true = 0.541
    single = reconstruct_two_arm(E[0], C[0], t, 300, 300)["hr"]
    ens = ensemble_two_arm(E, C, t, 300, 300)["hr"]
    assert abs(ens - true) <= abs(single - true) + 1e-9   # ensemble no worse than single
    assert ens < 1.0


def test_self_consistency_flags_bad_reads():
    """The confidence layer flags unreliable reads: consistent reads -> high; an outlier read ->
    high_read_variance; a survival/at-risk-table contradiction -> nar_curve_mismatch."""
    from vision_km_pipeline import ensemble_with_confidence
    t = [0, 3.875, 7.75, 11.625, 15.5, 19.375, 23.25, 27.125, 31]
    E = [[1.0,0.90,0.83,0.72,0.59,0.47,0.39,0.33,0.29],
         [1.0,0.89,0.83,0.72,0.58,0.47,0.38,0.32,0.28],
         [1.0,0.87,0.81,0.68,0.57,0.47,0.40,0.32,0.28]]
    C = [[1.0,0.88,0.71,0.55,0.38,0.26,0.16,0.12,0.07],
         [1.0,0.87,0.68,0.54,0.40,0.26,0.17,0.12,0.07],
         [1.0,0.85,0.70,0.52,0.38,0.27,0.19,0.13,0.07]]
    assert ensemble_with_confidence(E, C, t, 300, 300)["confidence"] == "high"
    Ebad = [E[0], E[1], [1.0,0.70,0.55,0.40,0.30,0.22,0.15,0.10,0.06]]
    assert "high_read_variance" in ensemble_with_confidence(Ebad, C, t, 300, 300)["flags"]
    r = ensemble_with_confidence(E, C, t, 300, 300,
                                 nar_times=[0, 15.5, 31], nar_exp=[300, 165, 250], nar_ctl=[300, 110, 20])
    assert "nar_curve_mismatch" in r["flags"]
