"""Tier-1 reconstruction benchmark (roadmap L1): isolate the reconstruction stage.

For each synthetic plot: take the TRUE per-arm KM curve (what a perfect digitizer would
hand off -- exact survival at a grid of timepoints, plus the true NAR table and N), run
the reconstruction algorithm, and score the reconstructed IPD against the TRUE IPD with
the honest metric set. Orientation is fixed from metadata (experimental = arm 1). No
reciprocal selection, no exclusions.

This measures the CEILING of the current reconstruction algorithm (image errors excluded).
The gap it exposes is the L3 lever (faithful Guyot). Run the end-to-end benchmark
separately to fold in axis/curve/NAR extraction error.

Usage: python run_reconstruction_benchmark.py [--limit N] [--noise 0.0] [--max-iter 100]
"""
from __future__ import annotations
import argparse, json, time
import numpy as np

from bench_contract import KMDATA_DIR, SUMMARY_PATH, RESULTS_DIR, F_TIME, F_STATUS, F_ARM, TARGETS
from km_metrics import km_from_ipd, eval_km, plot_metrics, hr_metrics
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from faithful_guyot import reconstruct_arm_faithful
from improved_guyot_algorithm import improved_guyot_reconstruction


def digitize_curve(arm_rows, horizon, n_points=100, noise=0.0, rng=None):
    """Emulate a digitized survival curve: exact S(t) sampled at n_points over [0,horizon],
    optionally with small additive pixel-style noise (clipped, re-monotonized)."""
    ets, svs = km_from_ipd([r[F_TIME] for r in arm_rows], [r[F_STATUS] for r in arm_rows])
    grid = np.linspace(0.0, horizon, n_points)
    s = eval_km(ets, svs, grid)
    if noise > 0 and rng is not None:
        s = np.clip(s + rng.normal(0, noise, s.size), 0, 1)
        for i in range(1, s.size):
            s[i] = min(s[i], s[i - 1])
    return grid, s


def reconstruct_plot(rows, meta, noise=0.0, max_iter=100, rng=None, engine="heuristic"):
    """Reconstruct each arm from its (digitized) curve + true N (+ NAR if present).

    engine='heuristic' -> wasserstein's improved_guyot_reconstruction (single-swap loop).
    engine='faithful'  -> the registry-ipd guyotCore + normalizeAndExpand Python port (L3).
    """
    recon = []
    horizon = meta["horizon"]
    nar_times = meta.get("nar_tick_times")
    nar_table = meta.get("nar_table")
    for a in range(meta["n_arms"]):
        arm_rows = [r for r in rows if r[F_ARM] == a]
        n_total = len(arm_rows)
        grid, s = digitize_curve(arm_rows, horizon, noise=noise, rng=rng)
        nrt = np.array(nar_times, float) if nar_times else None
        nrv = np.array(nar_table[a], float) if nar_table else None
        if engine == "faithful":
            ipd = reconstruct_arm_faithful(grid, s, n_total, nar_times=nrt, nar_values=nrv,
                                           follow_up=horizon)
            for r in ipd:
                recon.append({F_TIME: r["time"], F_STATUS: r["status"], F_ARM: a})
        else:
            res = improved_guyot_reconstruction(grid, s, n_total,
                                                n_risk_times=nrt, n_risk_values=nrv,
                                                max_iterations=max_iter)
            for t, e in zip(res.times, res.events):
                recon.append({F_TIME: float(t), F_STATUS: int(e), F_ARM: a})
    return recon


def bootstrap_ci(vals, stat=np.median, n_boot=1000, seed=0):
    vals = np.asarray([v for v in vals if v is not None and np.isfinite(v)], float)
    if vals.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boots = [stat(rng.choice(vals, vals.size, replace=True)) for _ in range(n_boot)]
    return float(stat(vals)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--engine", choices=["heuristic", "faithful"], default="heuristic")
    ap.add_argument("--tag", default="recon")
    args = ap.parse_args()

    if not SUMMARY_PATH.exists():
        raise SystemExit(f"No corpus. Run synth_km_generator.py first. Missing {SUMMARY_PATH}")
    summary = json.load(open(SUMMARY_PATH, encoding="utf-8"))
    datasets = summary["datasets"]
    if args.limit:
        datasets = datasets[:args.limit]

    per_plot = []
    t0 = time.time()
    for i, ds in enumerate(datasets):
        pid = ds["name"]
        d = KMDATA_DIR / pid
        rows = json.load(open(d / f"{pid}_ipd.json", encoding="utf-8"))
        meta = json.load(open(d / f"{pid}_meta.json", encoding="utf-8"))
        rng = np.random.default_rng(1000 + i)
        try:
            recon = reconstruct_plot(rows, meta, noise=args.noise, max_iter=args.max_iter, rng=rng, engine=args.engine)
            pm = plot_metrics(rows, recon)
            rec = {"pid": pid, "arms": ds["n_arms"], "n": ds["n_per_arm"], "cens": ds["censoring"],
                   "deg": ds["degradation"], "nar": ds["nar_present"], "ok": True,
                   "ae_median": pm["ae_median"], "iae": pm["iae"], "rmse": pm["rmse"], "ks": pm["ks"],
                   "rmst_rel_err": pm["rmst_rel_err"], "event_count_rel_err": pm["event_count_rel_err"],
                   "any_arm_missing": pm["any_arm_missing"]}
            if ds["n_arms"] >= 2:
                hm = hr_metrics(rows, recon)
                rec.update({"hr_pct_diff": hm["hr_pct_diff"], "loghr_abs_err": hm["loghr_abs_err"],
                            "direction_correct": hm["direction_correct"], "true_in_recon_ci": hm["true_in_recon_ci"],
                            "hr_true": hm["hr_true"], "hr_recon": hm["hr_recon"]})
        except Exception as e:
            rec = {"pid": pid, "arms": ds["n_arms"], "ok": False, "error": str(e)}
        per_plot.append(rec)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(datasets)}  ({time.time()-t0:.0f}s)")

    ok = [r for r in per_plot if r.get("ok")]
    def col(key, rows=ok):
        return [r[key] for r in rows if key in r and r[key] is not None and (not isinstance(r[key], float) or np.isfinite(r[key]))]

    hr_rows = [r for r in ok if "hr_pct_diff" in r]
    dir_rows = [r for r in ok if r.get("direction_correct") is not None]
    report = {
        "tag": args.tag, "noise": args.noise, "max_iter": args.max_iter,
        "n_plots": len(per_plot), "n_ok": len(ok),
        "success_rate": len(ok) / max(len(per_plot), 1),
        "headline": {
            "ae_median": bootstrap_ci(col("ae_median")),
            "iae": bootstrap_ci(col("iae")),
            "rmse": bootstrap_ci(col("rmse")),
            "ks": bootstrap_ci(col("ks")),
            "event_count_rel_err": bootstrap_ci(col("event_count_rel_err")),
            "hr_pct_diff_mean": bootstrap_ci(col("hr_pct_diff", hr_rows), stat=np.mean),
            "hr_pct_diff_median": bootstrap_ci(col("hr_pct_diff", hr_rows)),
            "direction_accuracy": (np.mean([r["direction_correct"] for r in dir_rows]) if dir_rows else float("nan")),
            "hr_ci_coverage": (np.mean([r["true_in_recon_ci"] for r in ok if r.get("true_in_recon_ci") is not None])
                               if any(r.get("true_in_recon_ci") is not None for r in ok) else float("nan")),
        },
        "targets": TARGETS,
        "by_arms": {}, "by_nar": {}, "per_plot": per_plot,
    }
    for a in sorted({r.get("arms") for r in ok if r.get("arms")}):
        sub = [r for r in ok if r.get("arms") == a]
        report["by_arms"][str(a)] = {"n": len(sub),
            "iae_median": float(np.median(col("iae", sub))) if col("iae", sub) else None,
            "rmse_median": float(np.median(col("rmse", sub))) if col("rmse", sub) else None,
            "any_arm_missing_rate": float(np.mean([r.get("any_arm_missing", False) for r in sub]))}
    for flag in (True, False):
        sub = [r for r in ok if r.get("nar") == flag]
        report["by_nar"][str(flag)] = {"n": len(sub),
            "iae_median": float(np.median(col("iae", sub))) if col("iae", sub) else None,
            "hr_pct_diff_median": float(np.median(col("hr_pct_diff", [r for r in sub if "hr_pct_diff" in r])))
                                  if [r for r in sub if "hr_pct_diff" in r] else None}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"reconstruction_{args.tag}.json"
    json.dump(report, open(out, "w", encoding="utf-8"), indent=1)

    h = report["headline"]
    print("\n" + "=" * 66)
    print(f"TIER-1 RECONSTRUCTION BASELINE  (noise={args.noise}, {len(ok)}/{len(per_plot)} ok)")
    print("=" * 66)
    def line(name, key, tgt=None):
        v, lo, hi = h[key]
        flag = ""
        if tgt is not None and np.isfinite(v):
            flag = "  PASS" if v <= tgt else f"  MISS (target {tgt})"
        print(f"  {name:26s} {v:8.4f}  [{lo:.4f}, {hi:.4f}]{flag}")
    line("AE median (S)", "ae_median", TARGETS["ae_median"])
    line("IAE / Wasserstein-1", "iae", TARGETS["iae"])
    line("per-arm RMSE", "rmse", TARGETS["rmse"])
    line("KS distance", "ks")
    line("event-count rel err", "event_count_rel_err")
    line("HR %-diff (mean)", "hr_pct_diff_mean", TARGETS["hr_pct_diff_mean"])
    line("HR %-diff (median)", "hr_pct_diff_median", TARGETS["hr_pct_diff_median"])
    print(f"  {'direction accuracy':26s} {h['direction_accuracy']:8.3f}")
    print(f"  {'HR 95% CI coverage':26s} {h['hr_ci_coverage']:8.3f}   (target 0.93-0.97)")
    print(f"\n  by arm-count IAE median: " +
          ", ".join(f"{k}-arm={v['iae_median']:.4f}" for k, v in report["by_arms"].items() if v['iae_median'] is not None))
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
