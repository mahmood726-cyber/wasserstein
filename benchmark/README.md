# Honest KM→IPD Benchmark (roadmap L1 + L2)

The ungameable measurement layer for the wasserstein PDF→KM-curve→IPD pipeline. Built
2026-07-10 because the legacy validation compared only to *published* HR/median on 13
single-specialty trials and was inflated by three evaluation cheats. **You cannot claim
"best in the world" without a benchmark that would convince a skeptic — this is it.**

## Why this exists

The prior headline ("100% extraction, 1.7% median error") is not evidence of accuracy:
1. it never compared reconstructed output to **true per-patient IPD** (only to the paper's own published HR/median);
2. it picked whichever HR **orientation (or reciprocal)** was closest to ground truth — circular;
3. it **post-hoc excluded** the trials the pipeline is known to fail.

This benchmark scores against a **synthetic true-IPD corpus** — the same construction used
by KM-GPT (540 plots), SurvdigitizeR, and RESOLVE-IPD: simulate known per-patient
event/censor times → render a KM figure → reconstruct → compare.

## Components

| File | Role |
|---|---|
| `bench_contract.py` | Single source of truth: corpus layout, strata, field names, metric names, SOTA targets. |
| `synth_km_generator.py` | Simulate true IPD (arms 1–3, n 25–1000, censoring/DPI/JPEG/y-truncation tiers) → render vector PDF + raster + optional number-at-risk table. Deterministic (seeded). |
| `km_metrics.py` | Honest metrics: AE, IAE (Wasserstein-1), per-arm RMSE, KS, RMST(τ), event-count error, HR %-diff, direction accuracy, CI coverage. Never selects the error-minimising orientation. |
| `run_reconstruction_benchmark.py` | **Tier 1**: true curve → reconstruct → score. Isolates the reconstruction stage (image errors excluded) — the ceiling of the current Guyot algorithm. |
| `run_endtoend_benchmark.py` | **Tier 2**: figure PDF → full pipeline → score. The real headline. Figures carry **no printed HR**, so the pipeline can't peek to cheat orientation; arms matched geometrically (not by effect). |
| `tests/test_km_metrics.py` | Pins the metric definitions (8 tests). |

## Metric targets (grounded SOTA)

| Metric | Target | Anchor |
|---|---|---|
| End-to-end success (≥500 plots) | ≥ 99.6% | KM-GPT; beat SurvdigitizeR 67% |
| Survival AE (median) | ≤ 0.005 | KM-GPT arXiv:2509.18141 |
| IAE / Wasserstein-1 | ≤ 0.018 | KM-GPT |
| per-arm RMSE | ≤ 0.012 (raster) / ≤ 0.002 (vector) | SurvdigitizeR BMC 2024 / RESOLVE arXiv:2511.01785 |
| HR mean / median %-diff | ≤ 2.85% / ≤ 2.14% | n=58 study PMC12409465 |
| HR 95% CI coverage | 93–97% | Guyot/RESOLVE practice |
| 3-arm separation | ≥ 90% | SurvdigitizeR |

## Run

```bash
cd benchmark
python -m pytest tests/                       # lock the metrics (8 tests)
python synth_km_generator.py 500              # build corpus -> validation_ground_truth/kmdata/
python run_reconstruction_benchmark.py        # Tier 1 baseline
python run_endtoend_benchmark.py --limit 120  # Tier 2 baseline (stratified subset; full run is slow)
```

Results land in `benchmark/results/`. The **first run of these is the true baseline the
entire roadmap is measured against** — expect it to be far below the legacy headline.
That is the point: an honest, lower number you can actually improve, not an inflated one
you can't trust.
