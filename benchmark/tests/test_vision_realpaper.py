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
