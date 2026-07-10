"""Shared contract for the wasserstein KM->IPD honest benchmark (roadmap L1).

Single source of truth for corpus layout, strata, field names, and metric names so
the generator, metric module, and runners never drift. Defined FIRST, before any
component, per the integration-contract lesson.

Corpus layout (under validation_ground_truth/kmdata/):
  kmdata_validation_summary.json      -> {"datasets": [ {plot meta}, ... ]}
  <plot_id>/<plot_id>_ipd.json        -> [ {"time": float, "status": 0|1, "arm": int}, ... ]  (TRUE IPD)
  <plot_id>/<plot_id>.pdf             -> rendered KM figure (vector, for end-to-end pipeline)
  <plot_id>/<plot_id>.png             -> rendered raster at the plot's degradation tier
  <plot_id>/<plot_id>_meta.json       -> full generation metadata (strata, per-arm truth, orientation)

Orientation is ALWAYS known from meta (arm labels + which arm is experimental), so the
benchmark scores with a FIXED orientation. No reciprocal selection, no text-HR flip, no
post-hoc exclusions -- those three cheats are exactly what this benchmark exists to kill.
"""
from __future__ import annotations
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent          # wasserstein/
KMDATA_DIR = BASE_DIR / "validation_ground_truth" / "kmdata"
SUMMARY_PATH = KMDATA_DIR / "kmdata_validation_summary.json"
RESULTS_DIR = BASE_DIR / "benchmark" / "results"

# ---- strata (the corpus spans these; a plot draws one value from each) --------------
ARM_COUNTS = [1, 2, 3]
N_PER_ARM = [25, 50, 150, 250, 1000]
# censoring regimes: how censoring times are generated relative to events
CENSORING = ["light", "administrative", "heavy"]   # light~10%, admin block at cutoff, heavy~40%
# figure degradation tiers (render/encode quality)
DEGRADATION = ["clean_600dpi", "raster_300dpi", "raster_150dpi", "jpeg_q40_150dpi"]
# y-axis presentation (tests the truncation-silently-read-as-full-range bug)
YAXIS = ["full_0_1", "full_0_100pct", "truncated"]   # truncated => e.g. 0.4..1.0

# fraction of plots that carry a number-at-risk table (drives Guyot fidelity)
NAR_PRESENT_FRACTION = 0.6

DEFAULT_N_PLOTS = 500
GLOBAL_SEED = 20260710          # fixed; Date.now()/random are NOT used anywhere

# ---- IPD row field names (TRUE and reconstructed share these) -----------------------
F_TIME = "time"
F_STATUS = "status"     # 1 = event, 0 = censored
F_ARM = "arm"           # 0 = control/reference, 1 = experimental, 2 = third arm

# ---- metric names (km_metrics returns exactly these keys) ---------------------------
METRICS = [
    "ae_median",        # median_t |S_recon(t) - S_true(t)|            (KM-GPT: <=0.005)
    "iae",              # integral |S_recon - S_true| dt over [0,tau], normalized by tau  (~Wasserstein-1)
    "rmse",             # sqrt(mean_t (dS)^2) at 100 grid points        (SurvdigitizeR: <=0.012)
    "ks",               # max_t |dS|
    "rmst_abs_err",     # |RMST_recon(tau) - RMST_true(tau)|
    "rmst_rel_err",     # rmst_abs_err / RMST_true
    "median_time_rel_err",  # |med_recon - med_true| / med_true  (survival-TIME median)
    "event_count_rel_err",  # |E_recon - E_true| / max(E_true,1)
    "n_recon", "n_true",
]
# HR-level metrics (2+ arm plots only)
HR_METRICS = [
    "loghr_abs_err",    # |log(HR_recon) - log(HR_true)|
    "hr_pct_diff",      # 100*|HR_recon - HR_true| / HR_true   (benchmark target mean<=2.85%, median<=2.14%)
    "direction_correct",# sign(logHR_recon) == sign(logHR_true)
    "true_in_recon_ci", # true logHR inside reconstructed 95% CI (coverage; target 93-97%)
]

# strata target thresholds (grounded SOTA anchors, for pass/fail flags in reports)
TARGETS = {
    "ae_median": 0.005,     # KM-GPT
    "iae": 0.018,           # KM-GPT (Wasserstein-1)
    "rmse": 0.012,          # SurvdigitizeR raster
    "hr_pct_diff_mean": 2.85,   # PMC12409465 n=58
    "hr_pct_diff_median": 2.14,
    "success_rate": 0.996,  # KM-GPT end-to-end
}
