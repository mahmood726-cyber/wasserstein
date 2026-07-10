"""Synthetic true-IPD KM benchmark generator (roadmap L1).

Simulates known per-patient event/censor times, renders a realistic KM figure (vector
PDF + raster tier + optional number-at-risk table), and writes the corpus in the
bench_contract layout so BOTH the reconstruction benchmark (true IPD -> exact KM ->
reconstruct) and the end-to-end benchmark (figure PDF -> full pipeline -> reconstruct)
can score against known ground truth.

This is the same ground-truth construction used by KM-GPT (540 synthetic), SurvdigitizeR,
and RESOLVE-IPD: simulate -> plot -> reconstruct -> compare.

Deterministic: one seeded numpy Generator per plot (GLOBAL_SEED + index). No global
random state, no Date.now(). Reproducible corpus.
"""
from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bench_contract import (
    KMDATA_DIR, SUMMARY_PATH, ARM_COUNTS, N_PER_ARM, CENSORING, DEGRADATION, YAXIS,
    NAR_PRESENT_FRACTION, DEFAULT_N_PLOTS, GLOBAL_SEED, F_TIME, F_STATUS, F_ARM,
)
from km_metrics import km_from_ipd

ARM_LABELS = ["Control", "Experimental", "Arm C"]
ARM_COLORS = ["#1f77b4", "#d62728", "#2ca02c"]   # blue, red, green -- distinct hues for HSV separation


# --------------------------------------------------------------- simulation
def simulate_trial(n_arms, n_per_arm, censoring, rng):
    """Simulate true IPD. Returns (ipd_rows, meta) with known per-arm hazards + true HR."""
    shape = float(rng.choice([0.8, 1.0, 1.0, 1.3]))          # Weibull shape (1=exponential)
    ctrl_median = float(rng.uniform(8.0, 30.0))              # months
    ctrl_scale = ctrl_median / (np.log(2) ** (1.0 / shape))  # Weibull scale from median

    # per-arm log-HR vs control (arm 0 ref). Clip to HR in [0.40, 1.60].
    loghrs = [0.0]
    for a in range(1, n_arms):
        lh = float(np.clip(rng.normal(0.0, 0.45), np.log(0.40), np.log(1.60)))
        loghrs.append(lh)

    # administrative follow-up horizon (all arms share it)
    horizon = ctrl_median * float(rng.uniform(1.6, 3.0))

    rows = []
    per_arm_meta = []
    for a in range(n_arms):
        # PH: scale_a = ctrl_scale * exp(-logHR/shape)  (larger HR => shorter survival)
        scale_a = ctrl_scale * np.exp(-loghrs[a] / shape)
        T = scale_a * rng.weibull(shape, n_per_arm)          # true event times
        if censoring == "administrative":
            C = np.full(n_per_arm, horizon)                  # block admin censor at horizon
        elif censoring == "light":
            C = np.minimum(rng.exponential(horizon * 3.0, n_per_arm), horizon)  # ~light random + admin
        else:  # heavy
            C = np.minimum(rng.exponential(horizon * 0.9, n_per_arm), horizon)  # ~heavy random dropout
        obs = np.minimum(T, C)
        status = (T <= C).astype(int)
        for t, s in zip(obs, status):
            rows.append({F_TIME: round(float(t), 4), F_STATUS: int(s), F_ARM: a})
        per_arm_meta.append({
            "arm": a, "label": ARM_LABELS[a], "n": n_per_arm,
            "events": int(status.sum()), "median_true": round(float(np.median(np.sort(obs))), 3),
            "loghr_true": round(loghrs[a], 4), "hr_true": round(float(np.exp(loghrs[a])), 4),
        })

    meta = {
        "n_arms": n_arms, "n_per_arm": n_per_arm, "censoring": censoring,
        "weibull_shape": shape, "ctrl_median": round(ctrl_median, 3),
        "horizon": round(horizon, 3),
        "experimental_arm": 1 if n_arms >= 2 else None,     # ORIENTATION is known here
        "reference_arm": 0,
        "hr_true": round(float(np.exp(loghrs[1])), 4) if n_arms >= 2 else None,
        "arms": per_arm_meta,
    }
    return rows, meta


# --------------------------------------------------------------- rendering
def _nar_table_values(rows, n_arms, tick_times):
    """Number at risk per arm at each tick time (n with observed time >= t)."""
    table = []
    for a in range(n_arms):
        ts = np.array([r[F_TIME] for r in rows if r[F_ARM] == a])
        table.append([int((ts >= t).sum()) for t in tick_times])
    return table


def render_figure(rows, meta, out_pdf, out_png, degradation, yaxis, nar):
    """Render the KM figure. Always writes a vector PDF; PNG at the degradation tier."""
    n_arms = meta["n_arms"]
    horizon = meta["horizon"]
    as_pct = (yaxis == "full_0_100pct")
    scale = 100.0 if as_pct else 1.0

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    min_s = 1.0
    for a in range(n_arms):
        arm_rows = [r for r in rows if r[F_ARM] == a]
        ets, svs = km_from_ipd([r[F_TIME] for r in arm_rows], [r[F_STATUS] for r in arm_rows])
        ets = np.append(ets, horizon)            # extend the last step to the horizon
        svs = np.append(svs, svs[-1])
        ax.step(ets, svs * scale, where="post", color=ARM_COLORS[a], lw=1.8,
                label=meta["arms"][a]["label"])
        # censoring tick marks (small vertical ticks on the curve at censor times)
        cens_t = [r[F_TIME] for r in arm_rows if r[F_STATUS] == 0]
        if cens_t:
            from km_metrics import eval_km
            cy = eval_km(ets, svs, cens_t) * scale
            ax.plot(cens_t, cy, "|", color=ARM_COLORS[a], markersize=6, mew=1.2)
        min_s = min(min_s, float(svs.min()))

    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Survival probability" + (" (%)" if as_pct else ""))
    ax.set_xlim(0, horizon)
    if yaxis == "truncated":
        lo = max(0.0, min_s - 0.1)
        ax.set_ylim(lo * scale, 1.0 * scale)      # truncated y-axis -- the silent-bias trap
    else:
        ax.set_ylim(0, 1.0 * scale)
    ax.legend(loc="best", frameon=False)
    ax.grid(True, alpha=0.25)

    tick_times = [round(horizon * f, 1) for f in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]
    if nar and n_arms >= 1:
        tbl = _nar_table_values(rows, n_arms, tick_times)
        lines = ["No. at risk"]
        for a in range(n_arms):
            lines.append(f"{meta['arms'][a]['label']:>12}: " + "  ".join(f"{v:4d}" for v in tbl[a]))
        fig.text(0.12, -0.02 - 0.0, "\n".join(lines), fontsize=7, family="monospace", va="top")
        meta["nar_tick_times"] = tick_times
        meta["nar_table"] = tbl
    meta["nar_present"] = bool(nar)

    fig.tight_layout()
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")     # vector, exact
    dpi = {"clean_600dpi": 600, "raster_300dpi": 300,
           "raster_150dpi": 150, "jpeg_q40_150dpi": 150}[degradation]
    fig.savefig(out_png, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    if degradation == "jpeg_q40_150dpi":
        from PIL import Image
        im = Image.open(out_png).convert("RGB")
        jpg = str(out_png).rsplit(".", 1)[0] + ".jpg"
        im.save(jpg, "JPEG", quality=40)
        meta["compressed_jpg"] = True


# --------------------------------------------------------------- corpus plan
def stratified_plan(n_plots):
    """Deterministic stratified assignment of strata to each plot index."""
    plan = []
    combos = [(a, n, c) for a in ARM_COUNTS for n in N_PER_ARM for c in CENSORING]  # 45 combos
    i = 0
    while len(plan) < n_plots:
        a, n, c = combos[i % len(combos)]
        deg = DEGRADATION[i % len(DEGRADATION)]
        yx = YAXIS[i % len(YAXIS)]
        nar = ((i * 7) % 10) < int(NAR_PRESENT_FRACTION * 10)
        plan.append({"arms": a, "n": n, "cens": c, "deg": deg, "yaxis": yx, "nar": nar})
        i += 1
    return plan


def generate_corpus(n_plots=DEFAULT_N_PLOTS, verbose=True):
    KMDATA_DIR.mkdir(parents=True, exist_ok=True)
    plan = stratified_plan(n_plots)
    datasets = []
    for idx, spec in enumerate(plan):
        rng = np.random.default_rng(GLOBAL_SEED + idx)       # deterministic per plot
        rows, meta = simulate_trial(spec["arms"], spec["n"], spec["cens"], rng)
        pid = f"synt_{idx:04d}_a{spec['arms']}_n{spec['n']}_{spec['cens']}_{spec['deg']}"
        d = KMDATA_DIR / pid
        d.mkdir(parents=True, exist_ok=True)
        meta.update({"plot_id": pid, "index": idx, "degradation": spec["deg"], "yaxis": spec["yaxis"]})
        # true IPD (contract format)
        with open(d / f"{pid}_ipd.json", "w", encoding="utf-8") as f:
            json.dump(rows, f)
        try:
            render_figure(rows, meta, d / f"{pid}.pdf", d / f"{pid}.png",
                          spec["deg"], spec["yaxis"], spec["nar"])
            meta["render_ok"] = True
        except Exception as e:
            meta["render_ok"] = False
            meta["render_error"] = str(e)
        with open(d / f"{pid}_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)
        datasets.append({"name": pid, "n_arms": spec["arms"], "n_per_arm": spec["n"],
                         "censoring": spec["cens"], "degradation": spec["deg"],
                         "yaxis": spec["yaxis"], "nar_present": bool(spec["nar"]),
                         "hr_true": meta.get("hr_true"), "render_ok": meta.get("render_ok", False)})
        if verbose and (idx + 1) % 50 == 0:
            print(f"  generated {idx + 1}/{n_plots}")
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump({"n_plots": len(datasets), "seed": GLOBAL_SEED, "datasets": datasets}, f, indent=1)
    if verbose:
        ok = sum(d["render_ok"] for d in datasets)
        print(f"corpus written: {len(datasets)} plots ({ok} rendered) -> {KMDATA_DIR}")
    return datasets


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_N_PLOTS
    generate_corpus(n)
