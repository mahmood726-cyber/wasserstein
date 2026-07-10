"""Honest KM->IPD reconstruction metrics (roadmap L1).

Every metric compares reconstructed output to TRUE per-patient IPD (or the true KM
curve derived from it) -- never to a published summary. Orientation is supplied by the
caller from known metadata; this module never selects the error-minimising orientation.

Grounded in the SOTA metric set:
  - AE (median absolute survival error), IAE (integrated abs error ~ Wasserstein-1): KM-GPT (arXiv:2509.18141)
  - per-arm RMSE at 100 timepoints: SurvdigitizeR (BMC 2024), IPDfromKM (BMC 2021)
  - KS distance: IPDfromKM
  - RMST(tau) difference: KM-GPT metric set / EMA non-PH review
  - HR %-difference: n=58 reconstruction study (PMC12409465)
"""
from __future__ import annotations
import numpy as np
from bench_contract import F_TIME, F_STATUS, F_ARM


# --------------------------------------------------------------- KM from IPD
def km_from_ipd(times, status):
    """Right-continuous Kaplan-Meier estimate. Returns (event_times, S_at_those_times).

    S is the survival AFTER processing each distinct time. Censorings reduce the risk
    set but do not drop S. Matches lifelines / registry-ipd kmFromIPD semantics.
    """
    times = np.asarray(times, float)
    status = np.asarray(status, int)
    if times.size == 0:
        return np.array([0.0]), np.array([1.0])
    order = np.argsort(times, kind="mergesort")
    times, status = times[order], status[order]
    uniq = np.unique(times)
    n_risk = times.size
    S = 1.0
    ets, svs = [0.0], [1.0]
    for t in uniq:
        at = times == t
        d = int(status[at].sum())
        c = int((~status[at].astype(bool)).sum())
        if n_risk > 0 and d > 0:
            S *= (1.0 - d / n_risk)
        ets.append(float(t))
        svs.append(S)
        n_risk -= (d + c)
    return np.asarray(ets), np.asarray(svs)


def eval_km(ets, svs, grid):
    """Evaluate a right-continuous KM step function at arbitrary grid points."""
    ets = np.asarray(ets, float)
    svs = np.asarray(svs, float)
    grid = np.asarray(grid, float)
    # step index: last event time <= t
    idx = np.searchsorted(ets, grid, side="right") - 1
    idx = np.clip(idx, 0, len(svs) - 1)
    return svs[idx]


def rmst(ets, svs, tau):
    """Restricted mean survival time = area under the KM step function on [0, tau]."""
    ets = np.asarray(ets, float)
    svs = np.asarray(svs, float)
    # build a fine trapezoid on the step function
    grid = np.linspace(0.0, tau, 2001)
    s = eval_km(ets, svs, grid)
    return float(np.trapezoid(s, grid))


def median_survival_time(ets, svs):
    """Smallest time with S <= 0.5 (None if never reached)."""
    ets = np.asarray(ets, float)
    svs = np.asarray(svs, float)
    below = np.where(svs <= 0.5 + 1e-12)[0]
    if below.size == 0:
        return None
    return float(ets[below[0]])


# --------------------------------------------------------------- curve-level metrics
def curve_metrics(true_ipd_arm, recon_ipd_arm, tau=None, grid_n=100):
    """Per-arm survival-curve fidelity vs true IPD.

    true_ipd_arm / recon_ipd_arm: list of {time,status} for ONE arm.
    Returns dict with ae_median, iae, rmse, ks, rmst_*, median_time_rel_err,
    event_count_rel_err, n_true, n_recon.
    """
    tt = np.array([r[F_TIME] for r in true_ipd_arm], float)
    ts = np.array([r[F_STATUS] for r in true_ipd_arm], int)
    rt = np.array([r[F_TIME] for r in recon_ipd_arm], float)
    rs = np.array([r[F_STATUS] for r in recon_ipd_arm], int)

    ets_t, svs_t = km_from_ipd(tt, ts)
    ets_r, svs_r = km_from_ipd(rt, rs)

    if tau is None:
        # compare over the region both curves observe: min of the two last-times,
        # so we never score reconstruction on extrapolated tail beyond either curve.
        tau = float(min(ets_t[-1], ets_r[-1])) if ets_t[-1] > 0 and ets_r[-1] > 0 else float(max(ets_t[-1], ets_r[-1]))
    if tau <= 0:
        tau = 1.0

    grid = np.linspace(0.0, tau, grid_n)
    s_t = eval_km(ets_t, svs_t, grid)
    s_r = eval_km(ets_r, svs_r, grid)
    dS = np.abs(s_r - s_t)

    # IAE = integral |dS| dt over [0,tau], normalized by tau -> comparable to a mean abs error
    iae = float(np.trapezoid(dS, grid) / tau)

    e_true = int(ts.sum())
    e_recon = int(rs.sum())
    med_t = median_survival_time(ets_t, svs_t)
    med_r = median_survival_time(ets_r, svs_r)
    if med_t is not None and med_t > 0 and med_r is not None:
        med_rel = abs(med_r - med_t) / med_t
    else:
        med_rel = float("nan")   # median not reached in true curve => undefined, not a free pass

    return {
        "ae_median": float(np.median(dS)),
        "iae": iae,
        "rmse": float(np.sqrt(np.mean(dS ** 2))),
        "ks": float(np.max(dS)),
        "rmst_abs_err": abs(rmst(ets_r, svs_r, tau) - rmst(ets_t, svs_t, tau)),
        "rmst_rel_err": (abs(rmst(ets_r, svs_r, tau) - rmst(ets_t, svs_t, tau)) /
                         max(rmst(ets_t, svs_t, tau), 1e-9)),
        "median_time_rel_err": med_rel,
        "event_count_rel_err": abs(e_recon - e_true) / max(e_true, 1),
        "n_true": int(tt.size),
        "n_recon": int(rt.size),
        "tau": tau,
    }


def plot_metrics(true_ipd, recon_ipd):
    """Aggregate curve metrics across all arms of a plot (mean over arms).

    true_ipd / recon_ipd: list of {time,status,arm}. Arms matched by arm index
    (orientation is the caller's responsibility, from known metadata).
    """
    arms = sorted({r[F_ARM] for r in true_ipd})
    per_arm = []
    for a in arms:
        t_a = [r for r in true_ipd if r[F_ARM] == a]
        r_a = [r for r in recon_ipd if r[F_ARM] == a]
        if not r_a:                        # arm entirely missing from reconstruction
            per_arm.append({"ae_median": 1.0, "iae": 1.0, "rmse": 1.0, "ks": 1.0,
                            "rmst_rel_err": 1.0, "median_time_rel_err": float("nan"),
                            "event_count_rel_err": 1.0, "n_true": len(t_a), "n_recon": 0,
                            "arm_missing": True})
            continue
        m = curve_metrics(t_a, r_a)
        m["arm_missing"] = False
        per_arm.append(m)

    def agg(key):
        vals = [m[key] for m in per_arm if key in m and not (isinstance(m[key], float) and np.isnan(m[key]))]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "ae_median": agg("ae_median"),
        "iae": agg("iae"),
        "rmse": agg("rmse"),
        "ks": agg("ks"),
        "rmst_rel_err": agg("rmst_rel_err"),
        "median_time_rel_err": agg("median_time_rel_err"),
        "event_count_rel_err": agg("event_count_rel_err"),
        "n_arms": len(arms),
        "any_arm_missing": any(m.get("arm_missing") for m in per_arm),
        "per_arm": per_arm,
    }


# --------------------------------------------------------------- HR metrics
def cox_loghr_2arm(ipd):
    """Cox logHR for a two-arm comparison (arm 1 exp vs arm 0 ref) via lifelines.

    Returns (loghr, se) or (nan, nan) if not estimable. Uses the experimental=arm1,
    reference=arm0 orientation as given -- NEVER inverts.
    """
    import pandas as pd
    from lifelines import CoxPHFitter
    rows = [r for r in ipd if r[F_ARM] in (0, 1)]
    if not rows:
        return float("nan"), float("nan")
    df = pd.DataFrame({
        "T": [r[F_TIME] for r in rows],
        "E": [r[F_STATUS] for r in rows],
        "x": [1 if r[F_ARM] == 1 else 0 for r in rows],
    })
    if df["E"].sum() < 2 or df["x"].nunique() < 2:
        return float("nan"), float("nan")
    try:
        cph = CoxPHFitter(penalizer=0.0)
        cph.fit(df, duration_col="T", event_col="E", show_progress=False)
        return float(cph.params_["x"]), float(cph.standard_errors_["x"])
    except Exception:
        try:
            cph = CoxPHFitter(penalizer=0.1)   # ridge retry on separation
            cph.fit(df, duration_col="T", event_col="E", show_progress=False)
            return float(cph.params_["x"]), float(cph.standard_errors_["x"])
        except Exception:
            return float("nan"), float("nan")


def hr_metrics(true_ipd, recon_ipd):
    """HR fidelity (2-arm exp-vs-ref) vs the true-IPD Cox HR. Orientation fixed."""
    lt, _ = cox_loghr_2arm(true_ipd)
    lr, sr = cox_loghr_2arm(recon_ipd)
    if not np.isfinite(lt) or not np.isfinite(lr):
        return {"loghr_abs_err": float("nan"), "hr_pct_diff": float("nan"),
                "direction_correct": None, "true_in_recon_ci": None,
                "hr_true": float(np.exp(lt)) if np.isfinite(lt) else None,
                "hr_recon": float(np.exp(lr)) if np.isfinite(lr) else None}
    lo, hi = lr - 1.959964 * sr, lr + 1.959964 * sr
    return {
        "loghr_abs_err": abs(lr - lt),
        "hr_pct_diff": 100.0 * abs(np.exp(lr) - np.exp(lt)) / np.exp(lt),
        # direction: only meaningful if the true effect is non-trivial
        "direction_correct": bool((lr < 0) == (lt < 0)) if abs(lt) > np.log(1.05) else None,
        "true_in_recon_ci": bool(lo - 1e-9 <= lt <= hi + 1e-9),
        "hr_true": float(np.exp(lt)),
        "hr_recon": float(np.exp(lr)),
    }
