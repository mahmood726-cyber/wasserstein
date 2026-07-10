"""Unit tests for the honest benchmark metrics. These pin the measurement layer so a
future edit cannot silently change what 'accuracy' means. Run: pytest benchmark/tests/
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import km_metrics as M
from bench_contract import F_TIME, F_STATUS, F_ARM


def _ipd(rows, arm=0):
    return [{F_TIME: t, F_STATUS: s, F_ARM: arm} for t, s in rows]


def test_km_from_ipd_hand_example():
    # times 1(ev),2(ev),3(cens),4(ev): S=3/4 ; 3/4*2/3=1/2 ; censor ; event at 1-at-risk ->0
    ets, svs = M.km_from_ipd([1, 2, 3, 4], [1, 1, 0, 1])
    assert abs(svs[1] - 0.75) < 1e-9
    assert abs(svs[2] - 0.5) < 1e-9
    assert svs[-1] <= 1e-9


def test_eval_km_step_right_continuous():
    ets, svs = M.km_from_ipd([2, 4], [1, 1])   # S=0.5 after t=2, 0 after t=4 (n=2)
    assert abs(M.eval_km(ets, svs, [0.0]) - 1.0) < 1e-9
    assert abs(M.eval_km(ets, svs, [1.9]) - 1.0) < 1e-9
    assert abs(M.eval_km(ets, svs, [2.0]) - 0.5) < 1e-9
    assert abs(M.eval_km(ets, svs, [3.9]) - 0.5) < 1e-9


def test_rmst_and_median():
    # 2 patients, events at 2 and 4: S=1 on [0,2), 0.5 on [2,4), 0 after.
    ets, svs = M.km_from_ipd([2, 4], [1, 1])
    # RMST(4) = 1*2 + 0.5*2 = 3.0
    assert abs(M.rmst(ets, svs, 4.0) - 3.0) < 0.02
    assert abs(M.median_survival_time(ets, svs) - 2.0) < 1e-9


def test_perfect_reconstruction_is_zero_error():
    rng = np.random.default_rng(1)
    t = rng.exponential(10, 200)
    s = (rng.random(200) < 0.7).astype(int)
    ipd = _ipd(list(zip(t, s)))
    m = M.curve_metrics(ipd, [dict(r) for r in ipd])
    assert m["ae_median"] < 1e-9
    assert m["iae"] < 1e-9
    assert m["rmse"] < 1e-9
    assert m["ks"] < 1e-9
    assert m["event_count_rel_err"] < 1e-9


def test_shifted_curve_has_expected_error():
    # true: everyone events at t=10 (S=1 until 10). recon: half die at t=5 (S drops to 0.5),
    # half at t=10. The curves genuinely differ on [5,10): KS should be ~0.5.
    true = _ipd([(10, 1)] * 100)
    recon = _ipd([(5, 1)] * 50 + [(10, 1)] * 50)
    m = M.curve_metrics(true, recon)
    assert m["ks"] > 0.4
    assert m["iae"] > 0.1
    assert m["event_count_rel_err"] == 0.0   # both have 100 events; only timing differs


def test_event_count_error_tracks_missed_events():
    true = _ipd([(10, 1)] * 100)               # 100 events
    recon = _ipd([(5, 0)] * 50 + [(10, 1)] * 50)  # 50 censored + 50 events => 50 events
    m = M.curve_metrics(true, recon)
    assert m["event_count_rel_err"] == 0.5


def test_arm_missing_penalised():
    true = _ipd([(1, 1), (2, 1)], arm=0) + _ipd([(3, 1), (4, 1)], arm=1)
    recon = _ipd([(1, 1), (2, 1)], arm=0)     # arm 1 missing entirely
    pm = M.plot_metrics(true, recon)
    assert pm["any_arm_missing"] is True
    assert pm["ks"] >= 0.5


def test_hr_direction_and_coverage_on_true_ipd():
    rng = np.random.default_rng(3)
    # control hazard 0.1, experimental hazard 0.05 -> HR ~0.5, favours experimental (logHR<0)
    ctl = [(rng.exponential(1 / 0.1), 1) for _ in range(300)]
    exp = [(rng.exponential(1 / 0.05), 1) for _ in range(300)]
    ipd = _ipd(ctl, arm=0) + _ipd(exp, arm=1)
    hm = M.hr_metrics(ipd, [dict(r) for r in ipd])
    assert hm["direction_correct"] is True
    assert hm["hr_recon"] < 1.0            # protective
    assert hm["true_in_recon_ci"] is True  # recon==true so trivially inside
    assert hm["hr_pct_diff"] < 1e-6
