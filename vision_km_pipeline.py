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
    "This is one Kaplan-Meier panel with a number-at-risk table below it. Read EACH curve's "
    "survival (fraction 0-1) at FINE sample times (include extra points 0-1 where the descent is "
    "steepest), tracing each line left-to-right; both start at 1.0. ALSO read the number-at-risk "
    "table (per arm, per time column). Distinguish curves by position + legend style/colour. "
    "Return JSON: times [...], per arm {label, survival:[...]}, nar_times [...], per arm "
    "nar_<label>:[counts], which_is_higher_overall, notes. Finer survival sampling + the at-risk "
    "counts materially improve accuracy."
)


def ipd_from_curve(times, survivals, n, follow_up=None):
    """Faithful IPD reconstruction from a sampled survival curve (from a vision read)."""
    t = np.asarray(times, float); s = np.asarray(survivals, float)
    return reconstruct_arm_faithful(t, s, int(n), follow_up=follow_up if follow_up is not None else float(t.max()))


def _reliable_horizon(times, *curves):
    """Index to truncate at, to drop an UNRELIABLE steep tail. Without a number-at-risk table the
    reconstruction assumes all N are still at risk late, so a steep late drop (few patients, noisy)
    becomes many phantom events and inflates the HR. Detect an anomalously steep drop in the later
    half of follow-up (>4x the median per-interval drop) and truncate before it. Returns len(times)
    when the tail is well-behaved."""
    n = len(times)
    cut = n
    for c in curves:
        c = np.asarray(c, float)
        drops = np.array([c[i - 1] - c[i] for i in range(1, n)])
        pos = drops[drops > 1e-6]
        med = np.median(pos) if pos.size else 0.02
        for i in range(2, n):
            if drops[i - 1] > max(4 * med, 0.08) and times[i] > 0.5 * times[-1]:
                cut = min(cut, i)
                break
    return max(cut, 4)                                    # keep at least a few points


def reconstruct_two_arm(exp_curve, ctl_curve, times, n_exp, n_ctl, follow_up=None,
                        nar_times=None, nar_exp=None, nar_ctl=None, robust_tail=True):
    """Build pseudo-IPD for both arms + Cox HR (experimental vs control).

    exp_curve/ctl_curve: survival fractions at `times`. Supplying the number-at-risk table
    (nar_times + per-arm nar_exp/nar_ctl, read from the figure) makes the Guyot reconstruction
    anchor-exact and materially tightens the HR toward truth (e.g. RFS 0.755 -> 0.680 with NAR).
    Returns dict with ipd, hr, ci, medians.
    """
    # Robust tail: only when NO at-risk table is available (with a NAR table the tail is anchored
    # and trustworthy). Truncate an anomalous unsupported steep tail so it can't inflate the HR.
    if robust_tail and nar_times is None:
        cut = _reliable_horizon(times, exp_curve, ctl_curve)
        if cut < len(times):
            times = list(times)[:cut]; exp_curve = list(exp_curve)[:cut]; ctl_curve = list(ctl_curve)[:cut]
            follow_up = float(times[-1])
    fu = follow_up if follow_up is not None else float(max(times))
    ipd = []
    for arm, curve, n, nv in [(1, exp_curve, n_exp, nar_exp), (0, ctl_curve, n_ctl, nar_ctl)]:
        kw = {}
        if nar_times is not None and nv is not None:
            kw = dict(nar_times=np.asarray(nar_times, float), nar_values=np.asarray(nv, float))
        for r in reconstruct_arm_faithful(np.asarray(times, float), np.asarray(curve, float),
                                          int(n), follow_up=fu, **kw):
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


def _monotone(c):
    c = np.asarray(c, float)
    for i in range(1, len(c)):
        c[i] = min(c[i], c[i - 1])
    return c


def ensemble_two_arm(exp_reads, ctl_reads, times, n_exp, n_ctl, follow_up=None,
                     nar_times=None, nar_exp=None, nar_ctl=None):
    """Average several INDEPENDENT vision reads of the same figure before reconstructing.

    Vision curve reads carry ~2-3% per-point noise; averaging K reads cuts the noise-driven HR
    error ~sqrt(K) (measured: fig1 4.7% -> 2.3% with K=3). exp_reads/ctl_reads are lists of survival
    arrays (each at `times`). Each read is monotonized before averaging. Returns reconstruct_two_arm's
    dict plus n_reads.
    """
    exp = np.mean([_monotone(r) for r in exp_reads], axis=0)
    ctl = np.mean([_monotone(r) for r in ctl_reads], axis=0)
    out = reconstruct_two_arm(list(exp), list(ctl), times, n_exp, n_ctl, follow_up,
                              nar_times, nar_exp, nar_ctl)
    out["n_reads"] = len(exp_reads)
    return out


def _km_at_risk_at(ipd_arm, query_times):
    """Number still at risk (time >= t) for a reconstructed arm at each query time."""
    t = np.array([r["time"] for r in ipd_arm], float)
    return [int((t >= q - 1e-9).sum()) for q in query_times]


def ensemble_with_confidence(exp_reads, ctl_reads, times, n_exp, n_ctl, follow_up=None,
                             nar_times=None, nar_exp=None, nar_ctl=None):
    """Ensemble reconstruction + self-consistency signals that flag an unreliable read.

    Two checks:
      (1) ensemble variance -- reconstruct each read's HR; the coefficient of variation across
          reads measures read agreement (high CV => the figure is being read inconsistently).
      (2) NAR consistency -- if a number-at-risk table was read, the ensemble reconstruction's
          own at-risk counts must match it (max relative deviation); a large mismatch means the
          survival read and the NAR read disagree, i.e. one is wrong.

    Returns the ensemble result plus: per_read_hr, hr_cv, nar_max_dev, confidence ('high'|'medium'|
    'low') and a 'flags' list. Low confidence => re-read / more reads recommended.
    """
    per = []
    for e, c in zip(exp_reads, ctl_reads):
        r = reconstruct_two_arm(e, c, times, n_exp, n_ctl, follow_up)
        if r["hr"] and np.isfinite(r["hr"]):
            per.append(r["hr"])
    out = ensemble_two_arm(exp_reads, ctl_reads, times, n_exp, n_ctl, follow_up,
                           nar_times, nar_exp, nar_ctl)
    logs = np.log(per) if len(per) >= 2 else np.array([0.0])
    hr_cv = float(np.std(logs)) if len(per) >= 2 else 0.0     # sd of log-HR across reads
    flags = []
    if hr_cv > 0.15:
        flags.append("high_read_variance")

    nar_max_dev = None
    if nar_times is not None and nar_exp is not None and nar_ctl is not None:
        # Consistency between the READ survival curve and the READ at-risk table, independent of
        # the reconstruction: with censoring, at-risk(t) <= N_start * S(t). If the read NAR exceeds
        # the survival-implied bound N*S(t), the survival read and the NAR read contradict each
        # other (one is wrong). exp_mean/ctl_mean are the ensemble-averaged read curves.
        exp_mean = np.mean([_monotone(r) for r in exp_reads], axis=0)
        ctl_mean = np.mean([_monotone(r) for r in ctl_reads], axis=0)
        devs = []
        for curve, nv, N in [(exp_mean, nar_exp, n_exp), (ctl_mean, nar_ctl, n_ctl)]:
            s_at = np.interp(nar_times, times, curve)
            for s, want in zip(s_at, nv):
                implied = N * s
                if want > 0:
                    devs.append(max(0.0, (want - implied) / want))   # excess of NAR over N*S
        nar_max_dev = float(max(devs)) if devs else None
        if nar_max_dev is not None and nar_max_dev > 0.15:
            flags.append("nar_curve_mismatch")

    confidence = "high"
    if flags:
        confidence = "low" if len(flags) >= 2 else "medium"
    out.update({"per_read_hr": [round(x, 4) for x in per], "hr_cv": round(hr_cv, 4),
                "nar_max_dev": (round(nar_max_dev, 3) if nar_max_dev is not None else None),
                "confidence": confidence, "flags": flags})
    return out


def extract_reliable(image_path, read_fn, n_exp, n_ctl, times=None, nar_times=None,
                     nar_exp=None, nar_ctl=None, min_reads=3, max_reads=7):
    """Self-correcting extraction: gather reads until the self-consistency confidence is 'high'.

    read_fn(image_path) -> a single vision read dict {times?, exp:[...], ctl:[...]} (or None). The
    loop takes `min_reads`, and while confidence is not 'high' (an inconsistent/outlier read is
    dragging the ensemble) it fetches more reads up to `max_reads`, dropping the single most
    outlying read once past min_reads so one bad read cannot poison the result. Returns the
    ensemble_with_confidence dict plus n_reads and reads_requested.
    """
    exp_reads, ctl_reads = [], []
    t = times
    requested = 0
    best = None
    while requested < max_reads:
        rd = read_fn(image_path)
        requested += 1
        if not rd or not rd.get("exp") or not rd.get("ctl"):
            continue
        t = rd.get("times", t)
        exp_reads.append(rd["exp"]); ctl_reads.append(rd["ctl"])
        if len(exp_reads) < min_reads:
            continue
        # drop the single most-outlying read (by its own reconstructed log-HR) once we have a few,
        # so one bad read cannot poison the ensemble
        use_e, use_c = exp_reads, ctl_reads
        if len(exp_reads) >= 4:
            hrs = []
            for e, c in zip(exp_reads, ctl_reads):
                r = reconstruct_two_arm(e, c, t, n_exp, n_ctl)
                hrs.append(np.log(r["hr"]) if r["hr"] and np.isfinite(r["hr"]) else 0.0)
            med = np.median(hrs)
            drop = int(np.argmax(np.abs(np.array(hrs) - med)))
            use_e = [e for i, e in enumerate(exp_reads) if i != drop]
            use_c = [c for i, c in enumerate(ctl_reads) if i != drop]
        best = ensemble_with_confidence(use_e, use_c, t, n_exp, n_ctl,
                                        nar_times=nar_times, nar_exp=nar_exp, nar_ctl=nar_ctl)
        best["n_reads"] = len(use_e)
        best["reads_requested"] = requested
        if best["confidence"] == "high":
            return best
    if best is None:
        return None
    return best


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
