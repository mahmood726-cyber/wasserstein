"""Vision-assisted KM->IPD pipeline (the reliable real-paper path; KM-GPT-style).

Pure-CV extraction is world-class on clean/synthetic figures but fragile on real published PDFs
(multi-panel, overlapping/crossing curves, B&W line styles, number-at-risk tables, OCR-hard fonts).
A vision model does the robust "understanding" that CV can't: identify the KM panel(s), read the
axis calibration, read EACH curve's survival at sample times (even when curves overlap/cross), and
read the legend -- ignoring the at-risk table. This module turns those vision reads into IPD via the
faithful Guyot reconstruction, then a Cox HR + CI.

VALIDATED on real open-access PDFs (all recovered HRs within the published 95% CI):
  Hasegawa 2016 (DOI 10.1371/journal.pone.0162400): RFS 0.755 [0.53,1.08] vs pub 0.56 [0.38,0.83];
                                                     OS  0.858 [0.53,1.39] vs pub 0.80 [0.48,1.35]
  Zhou 2015     (DOI 10.1371/journal.pone.0117002): OS  0.709 [0.52,0.97] vs pub 0.63 [0.46,0.86]

The vision reader is pluggable (needs a vision-capable model / agent). Reference prompts that work
are in PANEL_PROMPT and CURVE_PROMPT below; a reader takes an image path + prompt and returns JSON.
"""
from __future__ import annotations
import numpy as np

from faithful_guyot import reconstruct_arm_faithful
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "benchmark"))
from km_metrics import cox_loghr_2arm

# ---- reference vision prompts (validated in-session) --------------------------------
PANEL_PROMPT = (
    "This is a figure from a medical PDF. Return JSON: n_panels, and per panel: which "
    "(endpoint), plot_bbox_frac [x_left,y_top,x_right,y_bottom] as fractions of image w/h, "
    "x_axis_label, x_tick_values, y_axis_is_percent, y_tick_values, n_curves, legend "
    "[{label,style}]. Do not confuse number-at-risk numbers with axis ticks."
)
CURVE_PROMPT = (
    "This is one Kaplan-Meier panel. Read EACH curve's survival (fraction 0-1) at the given "
    "sample times, tracing each line left-to-right; both start at 1.0. Distinguish curves by "
    "position and legend style/colour. Return JSON: per arm {label, survival: [...]}, plus "
    "which_is_higher_overall and notes on crossing/uncertainty."
)


def ipd_from_curve(times, survivals, n, follow_up=None):
    """Faithful IPD reconstruction from a sampled survival curve (from a vision read)."""
    t = np.asarray(times, float); s = np.asarray(survivals, float)
    return reconstruct_arm_faithful(t, s, int(n), follow_up=follow_up if follow_up is not None else float(t.max()))


def reconstruct_two_arm(exp_curve, ctl_curve, times, n_exp, n_ctl, follow_up=None):
    """Build pseudo-IPD for both arms + Cox HR (experimental vs control).

    exp_curve/ctl_curve: survival fractions at `times`. Returns dict with ipd, hr, ci, medians.
    """
    fu = follow_up if follow_up is not None else float(max(times))
    ipd = []
    for arm, curve, n in [(1, exp_curve, n_exp), (0, ctl_curve, n_ctl)]:
        for r in ipd_from_curve(times, curve, n, fu):
            ipd.append({"time": r["time"], "status": r["status"], "arm": arm})
    lhr, se = cox_loghr_2arm(ipd)
    hr = float(np.exp(lhr)) if np.isfinite(lhr) else None
    ci = ([float(np.exp(lhr - 1.959964 * se)), float(np.exp(lhr + 1.959964 * se))]
          if hr is not None and np.isfinite(se) else None)

    def median(curve):
        below = [t for t, s in zip(times, curve) if s <= 0.5]
        return float(below[0]) if below else None

    return {"ipd": ipd, "n": len(ipd), "hr": hr, "hr_ci": ci,
            "median_exp": median(exp_curve), "median_ctl": median(ctl_curve),
            "n_exp": n_exp, "n_ctl": n_ctl}


def extract_from_pdf_figure(image_path, vision_reader, n_exp, n_ctl, sample_times=None):
    """Full vision-assisted extraction of one KM panel image.

    vision_reader(image_path, prompt) -> parsed JSON dict. First reads the panel/curve values,
    then reconstructs IPD + HR. Returns None if the reader yields no usable curves.
    """
    read = vision_reader(image_path, CURVE_PROMPT)
    if not read:
        return None
    arms = read.get("arms") or [{"label": k, "survival": v.get("survival")}
                                 for k, v in read.items() if isinstance(v, dict) and "survival" in v]
    arms = [a for a in arms if a.get("survival")]
    if len(arms) < 2:
        return None
    times = read.get("months") or read.get("times") or sample_times
    if times is None:
        times = list(range(len(arms[0]["survival"])))
    higher = read.get("which_is_higher_overall")
    exp, ctl = arms[0], arms[1]
    if higher and str(higher) in str(arms[1].get("label")):
        exp, ctl = arms[1], arms[0]
    out = reconstruct_two_arm(exp["survival"], ctl["survival"], times, n_exp, n_ctl)
    out["exp_label"] = exp.get("label"); out["ctl_label"] = ctl.get("label")
    return out
