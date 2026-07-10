"""Vector fast-path result builder (roadmap L8).

Turns a born-digital (vector) PDF KM figure into a PipelineResult using exact vector
extraction (vector_km_extractor) + faithful reconstruction + legend-based orientation,
bypassing the raster HSV/axis path entirely. Returns None when the figure is not vector
(image-only/scanned), so the caller falls back to the raster pipeline.

Digitization is near-exact on vector figures (benchmark: median IAE 0.001, direction 1.0),
so this is the high-fidelity path for the ~25-50% of published PDFs that are born-digital.
"""
from __future__ import annotations
import time
import numpy as np

from vector_km_extractor import extract_vector_km
from faithful_guyot import reconstruct_arm_faithful


def _cox_loghr(exp_ipd, ctl_ipd):
    """Cox logHR (experimental vs control) + SE via lifelines. (nan, nan) if not estimable."""
    import pandas as pd
    from lifelines import CoxPHFitter
    rows = ([(r["time"], r["status"], 1) for r in exp_ipd] +
            [(r["time"], r["status"], 0) for r in ctl_ipd])
    df = pd.DataFrame(rows, columns=["T", "E", "x"])
    if df["E"].sum() < 2 or df["x"].nunique() < 2:
        return float("nan"), float("nan")
    for pen in (0.0, 0.1):
        try:
            c = CoxPHFitter(penalizer=pen)
            c.fit(df, duration_col="T", event_col="E", show_progress=False)
            return float(c.params_["x"]), float(c.standard_errors_["x"])
        except Exception:
            continue
    return float("nan"), float("nan")


def build_result_from_extraction(ex, pdf_path, pdf_name, PipelineResult, IPDExport,
                                 source="vector", default_n=100, t0=None):
    """Build a PipelineResult from a curve-extraction dict (vector OR raster-OCR), or None.

    ex is the common extractor output: {n_arms, arms:[{times,survivals,role}],
    y_is_percent, y_truncated?, orientation_from_legend}. `source` labels the path.
    """
    if t0 is None:
        t0 = time.time()
    if not ex or ex.get("n_arms", 0) < 1:
        return None

    arms = ex["arms"]
    # order arms: experimental (role 1) first as arm1='treatment', control (role 0) as arm2
    def role_key(a):
        return {1: 0, 0: 1}.get(a.get("role"), 2)
    arms = sorted(arms, key=role_key)

    n_from_nar = any(a.get("n") for a in arms)

    def _reconstruct(a):
        # Real N from the number-at-risk table (roadmap L6) when available; else default
        # (flagged). Curve fidelity and HR DIRECTION are N-independent; N fixes event counts,
        # HR magnitude, and CI width.
        n = int(a["n"]) if a.get("n") else int(default_n)
        ipd = reconstruct_arm_faithful(np.array(a["times"], float), np.array(a["survivals"], float),
                                       n, follow_up=float(max(a["times"])) if a["times"] else None)
        return ipd

    warnings = [f"{source}_path", f"orientation:{'legend' if ex.get('orientation_from_legend') else 'unknown'}",
                f"N:{'nar' if n_from_nar else 'default'}"]
    if ex.get("y_truncated"):
        warnings.append("y_axis_truncated_calibrated")
    if ex.get("n_arms", 0) > 2:
        warnings.append("vector_extra_arms_dropped:%d" % (ex["n_arms"] - 2))

    def _export(a, idx, label):
        ipd = _reconstruct(a)
        ev = sum(r["status"] for r in ipd)
        recs = [{"time": float(r["time"]), "event": int(r["status"]), "arm": idx} for r in ipd]
        return IPDExport(arm_label=label, arm_index=idx, n_patients=len(ipd),
                         n_events=ev, n_censored=len(ipd) - ev, records=recs), ipd

    # Build ALL arms (roadmap L12: >=3-arm support). arms are role-sorted (experimental first,
    # control second). ipd_arm1/ipd_arm2 stay the primary exp-vs-ctl pair for back-compat;
    # ipd_arms carries every reconstructed arm.
    label_for = {1: "treatment", 0: "control", 2: "arm_c"}
    exports, all_rows = [], []
    for i, a in enumerate(arms):
        lbl = label_for.get(a.get("role"), f"arm_{i}")
        exp, rws = _export(a, i, lbl)
        exports.append(exp)
        all_rows.append(rws)

    ipd1_exp = exports[0]
    ipd2 = exports[1] if len(exports) >= 2 else None
    total = sum(e.n_patients for e in exports)
    hr = ci_lo = ci_hi = None
    if len(arms) >= 2:
        lhr, se = _cox_loghr(all_rows[0], all_rows[1])   # experimental vs control
        if np.isfinite(lhr):
            hr = float(np.exp(lhr))
            if np.isfinite(se):
                ci_lo, ci_hi = float(np.exp(lhr - 1.959964 * se)), float(np.exp(lhr + 1.959964 * se))
    if not n_from_nar:
        ci_lo = ci_hi = None                              # roadmap: no fabricated CI when N is unknown

    return PipelineResult(
        pdf_path=str(pdf_path), pdf_name=pdf_name,
        hr=hr, ci_lower=ci_lo, ci_upper=ci_hi, p_value=None, hr_method=f"{source}_faithful_cox",
        ipd_arm1=ipd1_exp, ipd_arm2=ipd2, ipd_arms=exports, total_ipd_records=total,
        curve1_times=[float(x) for x in arms[0]["times"]], curve1_survivals=[float(x) for x in arms[0]["survivals"]],
        curve2_times=([float(x) for x in arms[1]["times"]] if len(arms) >= 2 else None),
        curve2_survivals=([float(x) for x in arms[1]["survivals"]] if len(arms) >= 2 else None),
        confidence=0.99, orientation_method=(f"{source}_legend" if ex.get("orientation_from_legend") else f"{source}_unknown"),
        text_hr=None, n_curves_found=ex["n_arms"], n_pages_scanned=1,
        processing_time_s=round(time.time() - t0, 3), warnings=warnings,
    )
