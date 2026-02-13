#!/usr/bin/env python3
"""
Improved Guyot IPD Reconstruction Algorithm
============================================
Implements the full Guyot et al. (2012) algorithm for reconstructing
individual patient data from digitized Kaplan-Meier curves.

Key improvements over simplified version:
1. Proper hazard estimation from survival curve slope
2. Correct handling of number at risk information
3. Iterative refinement for better event distribution
4. Support for censoring pattern estimation

Reference:
Guyot, P., et al. (2012). "Enhanced secondary analysis of survival data:
reconstructing the data from published Kaplan-Meier survival curves."
BMC Medical Research Methodology, 12:9.
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
import warnings


@dataclass
class ReconstructionResult:
    """Results from IPD reconstruction."""
    times: np.ndarray
    events: np.ndarray
    n_patients: int
    n_events: int
    n_censored: int
    convergence: bool
    iterations: int


def estimate_number_at_risk(times: np.ndarray, survival: np.ndarray,
                            n_total: int,
                            n_risk_times: Optional[np.ndarray] = None,
                            n_risk_values: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Estimate number at risk at each time point.

    If n_risk table is provided, interpolate to get values at curve times.
    Otherwise, estimate from survival probability and total N.
    """
    if n_risk_times is not None and n_risk_values is not None:
        # Interpolate from provided table
        from scipy.interpolate import interp1d
        interp = interp1d(n_risk_times, n_risk_values,
                         kind='previous', fill_value='extrapolate')
        return np.maximum(1, np.round(interp(times)).astype(int))
    else:
        # Estimate from survival curve
        # n_risk(t) = n_total * S(t) + adjustment for accumulated events
        n_risk = np.round(n_total * survival).astype(int)
        return np.maximum(1, n_risk)


def estimate_events_in_interval(s_start: float, s_end: float,
                                n_start: int, n_end: int) -> Tuple[int, int]:
    """
    Estimate events and censorings in an interval.

    Uses the Nelson-Aalen estimator relationship:
    S(t) = exp(-H(t)) where H(t) is cumulative hazard

    The number of events d is estimated from:
    S(t+dt)/S(t) = (n-d)/n (approximately)
    """
    if s_start <= 0 or s_end < 0:
        return 0, 0

    if s_start < s_end:
        s_end = s_start  # Can't increase

    # Estimate cumulative hazard difference
    # log(S(t2)/S(t1)) = -deltaH
    if s_end > 0:
        delta_h = -np.log(s_end / s_start)
    else:
        delta_h = 4.0  # Large hazard for S -> 0

    # Events from hazard: d ≈ n * (1 - exp(-h))
    d = n_start * (1 - np.exp(-delta_h))
    d = int(round(d))
    d = max(0, min(d, n_start))

    # Censoring is the remaining decrease in n_risk
    c = max(0, n_start - n_end - d)

    return d, c


def improved_guyot_reconstruction(
    times: np.ndarray,
    survival: np.ndarray,
    n_total: int,
    n_risk_times: Optional[np.ndarray] = None,
    n_risk_values: Optional[np.ndarray] = None,
    max_iterations: int = 100,
    tolerance: float = 0.01
) -> ReconstructionResult:
    """
    Reconstruct IPD using improved Guyot algorithm.

    Parameters:
    -----------
    times : array
        Time points from digitized curve
    survival : array
        Survival probabilities at each time point
    n_total : int
        Total number of patients
    n_risk_times : array, optional
        Time points where n-at-risk is known
    n_risk_values : array, optional
        Number at risk at specified times
    max_iterations : int
        Maximum iterations for refinement
    tolerance : float
        Convergence tolerance

    Returns:
    --------
    ReconstructionResult with times, events, and metadata
    """
    # Ensure starts at time 0 with S=1
    if times[0] > 0:
        times = np.concatenate([[0], times])
        survival = np.concatenate([[1.0], survival])

    # Ensure monotonically decreasing
    for i in range(1, len(survival)):
        if survival[i] > survival[i-1]:
            survival[i] = survival[i-1]

    # Estimate n_risk at each time point
    n_risk = estimate_number_at_risk(times, survival, n_total,
                                     n_risk_times, n_risk_values)

    # Initial reconstruction
    recon_times = []
    recon_events = []

    for i in range(1, len(times)):
        dt = times[i] - times[i-1]
        if dt <= 0:
            continue

        n_start = n_risk[i-1]
        n_end = n_risk[i] if i < len(n_risk) else n_risk[-1]
        s_start = survival[i-1]
        s_end = survival[i]

        d, c = estimate_events_in_interval(s_start, s_end, n_start, n_end)

        # Distribute events uniformly in interval
        if d > 0:
            event_times = np.linspace(times[i-1] + dt*0.1, times[i] - dt*0.1, d)
            recon_times.extend(event_times)
            recon_events.extend([1] * d)

        # Distribute censoring uniformly in interval
        if c > 0:
            censor_times = np.linspace(times[i-1] + dt*0.05, times[i] - dt*0.05, c)
            recon_times.extend(censor_times)
            recon_events.extend([0] * c)

    # Add final censoring for remaining patients
    if len(survival) > 0 and survival[-1] > 0:
        n_remaining = int(round(survival[-1] * n_total))
        if n_remaining > 0:
            recon_times.extend([times[-1]] * n_remaining)
            recon_events.extend([0] * n_remaining)

    recon_times = np.array(recon_times)
    recon_events = np.array(recon_events)

    # Sort by time
    sort_idx = np.argsort(recon_times)
    recon_times = recon_times[sort_idx]
    recon_events = recon_events[sort_idx]

    # Iterative refinement to match target survival curve
    converged = False
    iteration = 0

    for iteration in range(max_iterations):
        # Check if we match target
        if len(recon_times) == 0:
            break

        # Compute current survival curve
        from lifelines import KaplanMeierFitter
        kmf = KaplanMeierFitter()
        kmf.fit(recon_times, event_observed=recon_events)
        current_surv = kmf.survival_function_

        # Compare at original time points and identify discrepancies
        errors = []
        signed_errors = []  # positive = reconstructed too high (need more events)
        for t, s in zip(times, survival):
            idx = (np.abs(current_surv.index.values - t)).argmin()
            current_s = current_surv.iloc[idx, 0]
            errors.append(abs(current_s - s))
            signed_errors.append(current_s - s)

        max_error = max(errors) if errors else 0

        if max_error < tolerance:
            converged = True
            break

        # Iterative adjustment: modify events based on discrepancies
        # Find the interval with largest error
        if len(signed_errors) > 1:
            max_err_idx = np.argmax(np.abs(signed_errors))
            error_sign = signed_errors[max_err_idx]
            error_time = times[max_err_idx]

            # Find events/censorings near this time point
            time_window = (times[-1] - times[0]) / len(times) if len(times) > 1 else 1.0
            near_mask = np.abs(recon_times - error_time) < time_window

            if error_sign > tolerance:
                # Reconstructed survival too high - need more events
                # Convert a censoring to an event near this time
                censor_mask = near_mask & (recon_events == 0)
                censor_indices = np.where(censor_mask)[0]

                if len(censor_indices) > 0:
                    # Convert one censoring to event
                    idx_to_change = censor_indices[0]
                    recon_events[idx_to_change] = 1
                elif np.sum(recon_events == 0) > 0:
                    # Find closest censoring
                    censor_times = recon_times[recon_events == 0]
                    closest_idx = np.argmin(np.abs(censor_times - error_time))
                    global_idx = np.where(recon_events == 0)[0][closest_idx]
                    recon_events[global_idx] = 1
                else:
                    # Add a new event at this time
                    recon_times = np.append(recon_times, error_time)
                    recon_events = np.append(recon_events, 1)
                    sort_idx = np.argsort(recon_times)
                    recon_times = recon_times[sort_idx]
                    recon_events = recon_events[sort_idx]

            elif error_sign < -tolerance:
                # Reconstructed survival too low - too many events
                # Convert an event to censoring near this time
                event_mask = near_mask & (recon_events == 1)
                event_indices = np.where(event_mask)[0]

                if len(event_indices) > 0:
                    # Convert one event to censoring
                    idx_to_change = event_indices[0]
                    recon_events[idx_to_change] = 0
                elif np.sum(recon_events == 1) > 0:
                    # Find closest event
                    event_times = recon_times[recon_events == 1]
                    closest_idx = np.argmin(np.abs(event_times - error_time))
                    global_idx = np.where(recon_events == 1)[0][closest_idx]
                    recon_events[global_idx] = 0

    n_events = int(recon_events.sum())
    n_censored = len(recon_events) - n_events

    return ReconstructionResult(
        times=recon_times,
        events=recon_events,
        n_patients=len(recon_times),
        n_events=n_events,
        n_censored=n_censored,
        convergence=converged,
        iterations=iteration + 1 if 'iteration' in dir() else 1
    )


def run_improved_validation():
    """Test improved algorithm on validation datasets."""
    import json
    from pathlib import Path
    from lifelines import KaplanMeierFitter

    BASE_DIR = Path(__file__).parent
    KMDATA_DIR = BASE_DIR / "validation_ground_truth" / "kmdata"
    OUTPUT_DIR = BASE_DIR / "accuracy_results"

    # Load summary
    summary_path = KMDATA_DIR / "kmdata_validation_summary.json"
    if not summary_path.exists():
        print("No kmdata summary found")
        return

    with open(summary_path) as f:
        data = json.load(f)

    datasets = data.get('datasets', [])[:50]  # Test on first 50

    results = []
    grades = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}

    print(f"Testing improved algorithm on {len(datasets)} datasets...")

    for ds in datasets:
        name = ds['name']
        ipd_path = KMDATA_DIR / name / f"{name}_ipd.json"

        if not ipd_path.exists():
            continue

        try:
            with open(ipd_path) as f:
                ipd = json.load(f)

            times_true = np.array([p['time'] for p in ipd])
            events_true = np.array([p['status'] for p in ipd])

            n_true = len(times_true)
            e_true = int(events_true.sum())

            # Compute KM curve
            kmf = KaplanMeierFitter()
            kmf.fit(times_true, event_observed=events_true)
            km_times = kmf.survival_function_.index.values
            km_surv = kmf.survival_function_.values.flatten()

            # Add noise
            noise = 0.02
            km_surv = km_surv + np.random.normal(0, noise, len(km_surv))
            km_surv = np.clip(km_surv, 0, 1)
            for i in range(1, len(km_surv)):
                if km_surv[i] > km_surv[i-1]:
                    km_surv[i] = km_surv[i-1]

            # Reconstruct
            result = improved_guyot_reconstruction(km_times, km_surv, n_true)

            n_recon = result.n_patients
            e_recon = result.n_events

            n_err = abs(n_recon - n_true) / n_true * 100
            e_err = abs(e_recon - e_true) / max(e_true, 1) * 100

            avg_err = (n_err + e_err) / 2

            if avg_err < 2:
                grade = 'A'
            elif avg_err < 5:
                grade = 'B'
            elif avg_err < 10:
                grade = 'C'
            elif avg_err < 20:
                grade = 'D'
            else:
                grade = 'F'

            grades[grade] += 1
            results.append({
                'name': name,
                'n_true': n_true,
                'n_recon': n_recon,
                'n_err': n_err,
                'e_true': e_true,
                'e_recon': e_recon,
                'e_err': e_err,
                'grade': grade
            })

        except Exception as e:
            print(f"  Error with {name}: {e}")

    # Print results
    print("\n" + "=" * 60)
    print("IMPROVED ALGORITHM RESULTS")
    print("=" * 60)
    print(f"Datasets tested: {len(results)}")
    print(f"\nGrade distribution:")
    total = len(results)
    for g in 'ABCDF':
        pct = grades[g] / total * 100 if total > 0 else 0
        print(f"  {g}: {grades[g]} ({pct:.1f}%)")

    n_errors = [r['n_err'] for r in results]
    e_errors = [r['e_err'] for r in results]

    print(f"\nN error: mean={np.mean(n_errors):.2f}%, median={np.median(n_errors):.2f}%")
    print(f"Events error: mean={np.mean(e_errors):.2f}%, median={np.median(e_errors):.2f}%")

    # Save results
    output_path = OUTPUT_DIR / "improved_algorithm_results.json"
    with open(output_path, 'w') as f:
        json.dump({
            'grades': grades,
            'n_error_mean': np.mean(n_errors),
            'e_error_mean': np.mean(e_errors),
            'results': results
        }, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_improved_validation()
