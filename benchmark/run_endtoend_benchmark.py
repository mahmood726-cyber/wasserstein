"""Tier-2 END-TO-END benchmark (roadmap L1): the real headline.

figure PDF -> full wasserstein pipeline (rasterize, classify, axis, curve-extract,
multi-arm, NAR, Guyot, HR) -> reconstructed IPD -> honest metrics vs TRUE IPD.

Honesty properties, by construction of the synthetic corpus:
  - the figures carry NO printed HR, so the pipeline's text-HR orientation shortcut has
    nothing to peek at (text_hr is None) -> direction accuracy is a REAL measurement.
  - reconstructed arms are matched to true arms GEOMETRICALLY (nearest survival curve,
    Hungarian assignment) for curve metrics -- this is curve-identity assignment, NOT the
    effect-orientation cheat; the HR direction is scored separately and never flipped.
  - success rate is reported on the FULL corpus (a pipeline failure counts as a failure).

Usage: python run_endtoend_benchmark.py [--limit N] [--dpi 150] [--tag e2e]
"""
from __future__ import annotations
import argparse, json, time, warnings
import numpy as np
from scipy.optimize import linear_sum_assignment

from bench_contract import KMDATA_DIR, SUMMARY_PATH, RESULTS_DIR, F_TIME, F_STATUS, F_ARM, TARGETS
from km_metrics import curve_metrics, km_from_ipd, eval_km, cox_loghr_2arm
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
warnings.filterwarnings("ignore")


def _arm_ipd(export):
    if export is None or not getattr(export, "records", None):
        return []
    out = []
    for r in export.records:
        out.append({F_TIME: float(r["time"]), F_STATUS: int(r.get("event", r.get("status", 0))),
                    F_ARM: int(r.get("arm", export.arm_index))})
    return out


def _curve_iae(true_arm, recon_arm, tau):
    et, sv = km_from_ipd([r[F_TIME] for r in true_arm], [r[F_STATUS] for r in true_arm])
    er, sr = km_from_ipd([r[F_TIME] for r in recon_arm], [r[F_STATUS] for r in recon_arm])
    g = np.linspace(0, tau, 100)
    return float(np.trapezoid(np.abs(eval_km(er, sr, g) - eval_km(et, sv, g)), g) / tau)


def score_plot(true_ipd, recon_arms, meta):
    """recon_arms: list of arm-IPD lists (1..2 arms). Geometric match to true arms."""
    horizon = meta["horizon"]
    true_arm_ids = sorted({r[F_ARM] for r in true_ipd})
    true_by_arm = {a: [r for r in true_ipd if r[F_ARM] == a] for a in true_arm_ids}
    recon_arms = [ra for ra in recon_arms if ra]
    if not recon_arms:
        return {"ok": False, "reason": "no_recon_arms"}

    # cost matrix: IAE between each recon arm and each true arm
    cost = np.zeros((len(recon_arms), len(true_arm_ids)))
    for i, ra in enumerate(recon_arms):
        for j, a in enumerate(true_arm_ids):
            cost[i, j] = _curve_iae(true_by_arm[a], ra, horizon)
    ri, cj = linear_sum_assignment(cost)
    mapping = {int(i): int(true_arm_ids[j]) for i, j in zip(ri, cj)}

    per_arm = []
    matched_true = set()
    for i, ra in enumerate(recon_arms):
        if i not in mapping:
            continue
        ta = mapping[i]
        matched_true.add(ta)
        m = curve_metrics(true_by_arm[ta], ra, tau=horizon)
        per_arm.append(m)
    n_missing = len(true_arm_ids) - len(matched_true)

    def agg(k):
        vals = [m[k] for m in per_arm if k in m and np.isfinite(m[k])]
        return float(np.mean(vals)) if vals else float("nan")

    out = {"ok": True, "n_true_arms": len(true_arm_ids), "n_recon_arms": len(recon_arms),
           "n_arms_missing": n_missing,
           "ae_median": agg("ae_median"), "iae": agg("iae"), "rmse": agg("rmse"),
           "ks": agg("ks"), "event_count_rel_err": agg("event_count_rel_err"),
           "mapping": mapping}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--max-pages", type=int, default=2)
    ap.add_argument("--tag", default="e2e")
    args = ap.parse_args()

    from km_pipeline import KMPipeline
    summary = json.load(open(SUMMARY_PATH, encoding="utf-8"))
    datasets = summary["datasets"][args.offset:]
    if args.limit:
        datasets = datasets[:args.limit]

    pipe = KMPipeline(dpi=args.dpi, max_pages=args.max_pages)
    per_plot = []
    t0 = time.time()
    for i, ds in enumerate(datasets):
        pid = ds["name"]
        d = KMDATA_DIR / pid
        pdf = d / f"{pid}.pdf"
        rows = json.load(open(d / f"{pid}_ipd.json", encoding="utf-8"))
        meta = json.load(open(d / f"{pid}_meta.json", encoding="utf-8"))
        rec = {"pid": pid, "arms": ds["n_arms"], "n": ds["n_per_arm"], "cens": ds["censoring"],
               "deg": ds["degradation"], "yaxis": ds.get("yaxis"), "nar": ds["nar_present"],
               "hr_true": meta.get("hr_true")}
        try:
            res = pipe.extract(str(pdf))
            if getattr(res, "ipd_arms", None):                 # L12: full N-arm result
                recon_arms = [_arm_ipd(a) for a in res.ipd_arms]
            else:
                recon_arms = [_arm_ipd(res.ipd_arm1), _arm_ipd(res.ipd_arm2)]
            sc = score_plot(rows, recon_arms, meta)
            rec.update(sc)
            rec["pipeline_succeeded"] = bool(res.succeeded)
            rec["hr_recon"] = res.hr
            rec["n_curves_found"] = res.n_curves_found
            # Score the reconstructed HR against the REALIZED true-IPD Cox HR (the actual
            # effect in the data a perfect extractor recovers) -- NOT the nominal simulation
            # parameter meta['hr_true'], which differs from the realized Cox HR by a median
            # ~9.5% (finite-sample + censoring) that is unobservable from the figure and would
            # unfairly inflate the extractor's HR error.
            if res.hr:
                lt, _ = cox_loghr_2arm(rows)
                if np.isfinite(lt):
                    true_hr = float(np.exp(lt))
                    rec["hr_true_ipd"] = true_hr
                    rec["hr_abs_logdiff"] = abs(np.log(res.hr) - lt)
                    rec["hr_pct_diff"] = 100 * abs(res.hr - true_hr) / true_hr
                    if abs(lt) > np.log(1.05):
                        rec["direction_correct"] = bool((res.hr < 1) == (true_hr < 1))
        except Exception as e:
            rec.update({"ok": False, "pipeline_succeeded": False, "error": str(e)[:200]})
        per_plot.append(rec)
        if (i + 1) % 20 == 0:
            ok = sum(1 for r in per_plot if r.get("ok"))
            print(f"  {i+1}/{len(datasets)}  extracted_ok={ok}  ({time.time()-t0:.0f}s)")

    n = len(per_plot)
    got_curves = [r for r in per_plot if r.get("ok")]          # produced >=1 usable arm
    succeeded = [r for r in per_plot if r.get("pipeline_succeeded")]

    def col(key, rows):
        return [r[key] for r in rows if key in r and r[key] is not None and (not isinstance(r[key], float) or np.isfinite(r[key]))]

    def med_ci(vals, stat=np.median, seed=0):
        vals = np.asarray([v for v in vals if np.isfinite(v)], float)
        if not vals.size:
            return (float("nan"),) * 3
        rng = np.random.default_rng(seed)
        b = [stat(rng.choice(vals, vals.size, True)) for _ in range(1000)]
        return float(stat(vals)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

    hr_rows = [r for r in got_curves if "hr_pct_diff" in r]
    dir_rows = [r for r in got_curves if r.get("direction_correct") is not None]
    report = {
        "tag": args.tag, "dpi": args.dpi, "n_plots": n,
        "success_rate_hr": len(succeeded) / max(n, 1),          # pipeline returned an HR
        "success_rate_curves": len(got_curves) / max(n, 1),     # produced >=1 usable arm
        "headline_on_extracted": {
            "ae_median": med_ci(col("ae_median", got_curves)),
            "iae": med_ci(col("iae", got_curves)),
            "rmse": med_ci(col("rmse", got_curves)),
            "ks": med_ci(col("ks", got_curves)),
            "event_count_rel_err": med_ci(col("event_count_rel_err", got_curves)),
            "hr_pct_diff_mean": med_ci(col("hr_pct_diff", hr_rows), stat=np.mean),
            "hr_pct_diff_median": med_ci(col("hr_pct_diff", hr_rows)),
            "direction_accuracy": float(np.mean([r["direction_correct"] for r in dir_rows])) if dir_rows else float("nan"),
        },
        "arm_recovery": {
            "3arm_success": (sum(1 for r in got_curves if r.get("arms") == 3 and r.get("n_arms_missing", 9) == 0) /
                             max(sum(1 for r in per_plot if r.get("arms") == 3), 1)),
            "mean_arms_missing": float(np.mean(col("n_arms_missing", got_curves))) if col("n_arms_missing", got_curves) else None,
        },
        "targets": TARGETS, "per_plot": per_plot,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"endtoend_{args.tag}.json"
    json.dump(report, open(out, "w", encoding="utf-8"), indent=1)

    h = report["headline_on_extracted"]
    print("\n" + "=" * 66)
    print(f"TIER-2 END-TO-END BASELINE  (dpi={args.dpi}, n={n})")
    print("=" * 66)
    print(f"  pipeline success (HR returned) : {report['success_rate_hr']:.3f}   (KM-GPT target 0.996)")
    print(f"  produced >=1 usable curve      : {report['success_rate_curves']:.3f}")
    def line(nm, k, tgt=None):
        v, lo, hi = h[k]
        f = ("" if tgt is None or not np.isfinite(v) else ("  PASS" if v <= tgt else f"  MISS(t={tgt})"))
        print(f"  {nm:26s} {v:8.4f}  [{lo:.4f},{hi:.4f}]{f}")
    line("AE median (S)", "ae_median", TARGETS["ae_median"])
    line("IAE / Wasserstein-1", "iae", TARGETS["iae"])
    line("per-arm RMSE", "rmse", TARGETS["rmse"])
    line("event-count rel err", "event_count_rel_err")
    line("HR %-diff mean", "hr_pct_diff_mean", TARGETS["hr_pct_diff_mean"])
    line("HR %-diff median", "hr_pct_diff_median", TARGETS["hr_pct_diff_median"])
    print(f"  {'direction accuracy':26s} {h['direction_accuracy']:8.3f}")
    print(f"  3-arm full recovery: {report['arm_recovery']['3arm_success']:.3f}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
