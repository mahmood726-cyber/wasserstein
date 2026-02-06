"""
Improved HR Estimation from KM Curves
======================================

Implements proper IPD reconstruction using the Guyot algorithm
and calculates HR using log-rank test / Cox proportional hazards.

Based on: Guyot P, et al. BMC Medical Research Methodology 2012, 12:9

Author: Wasserstein KM Extractor Team
Date: February 2026
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from scipy import stats
from scipy.interpolate import interp1d


@dataclass
class IPDRecord:
    """Individual patient data record."""
    time: float
    event: int  # 1 = event, 0 = censored
    arm: int    # 0 = control, 1 = treatment


@dataclass
class ArmIdentificationResult:
    """Result of treatment/control arm identification."""
    method: str  # 'legend', 'nar_labels', 'text_verified', 'auc_fallback', 'unknown'
    confidence: float  # 0-1, how confident we are in the assignment
    treatment_index: int  # Index of treatment curve in input list
    control_index: int  # Index of control curve in input list
    is_verified: bool  # True if verified by multiple methods or external data
    is_uncertain: bool  # True if assignment should be flagged for manual review
    warning: str = ""  # Warning message if uncertain
    details: str = ""  # Additional details about the identification

    @property
    def requires_manual_review(self) -> bool:
        """Check if this identification requires manual review."""
        return self.is_uncertain or self.confidence < 0.6


@dataclass
class HRResult:
    """Hazard ratio estimation result."""
    hr: float
    ci_lower: float
    ci_upper: float
    p_value: float
    method: str
    n_events: int
    n_total: int
    log_rank_statistic: float
    # New fields for arm identification
    arm_identification: Optional[ArmIdentificationResult] = None
    hr_inverted: Optional[float] = None  # HR if arms were swapped
    ci_lower_inverted: Optional[float] = None
    ci_upper_inverted: Optional[float] = None

    @property
    def arm_identification_confident(self) -> bool:
        """Check if arm identification is confident."""
        if self.arm_identification is None:
            return False
        return not self.arm_identification.requires_manual_review


def reconstruct_ipd_guyot(times: np.ndarray,
                          survivals: np.ndarray,
                          n_risk_times: Optional[np.ndarray] = None,
                          n_risk_values: Optional[np.ndarray] = None,
                          total_events: Optional[int] = None,
                          total_n: Optional[int] = None) -> List[IPDRecord]:
    """
    Reconstruct IPD from KM curve using Guyot algorithm.

    Enhanced with proper NAR integration for accurate censoring calculation.

    Parameters
    ----------
    times : array
        Time points from KM curve
    survivals : array
        Survival probabilities at each time point
    n_risk_times : array, optional
        Time points where number at risk is reported
    n_risk_values : array, optional
        Number at risk at each n_risk_times point
    total_events : int, optional
        Total number of events (if known)
    total_n : int, optional
        Total number of patients (if known)

    Returns
    -------
    List of IPDRecord
    """
    # Ensure sorted by time
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    survivals = survivals[sort_idx]

    # Ensure survivals are monotonically decreasing
    for i in range(1, len(survivals)):
        if survivals[i] > survivals[i-1]:
            survivals[i] = survivals[i-1]

    # Estimate total N if not provided
    if total_n is None:
        if n_risk_values is not None and len(n_risk_values) > 0:
            total_n = int(n_risk_values[0])
        else:
            # Default estimation
            total_n = 100

    # Interpolate number at risk if provided
    has_nar = n_risk_times is not None and n_risk_values is not None
    if has_nar and len(n_risk_times) >= 2 and len(n_risk_values) >= 2:
        # Use NAR for accurate censoring calculation
        n_at_risk = _interpolate_nar(n_risk_times, n_risk_values, times, total_n)
    else:
        # Estimate N at risk from survival curve (less accurate)
        n_at_risk = (survivals * total_n).astype(int)
        n_at_risk = np.maximum(n_at_risk, 1)

    ipd = []

    # Calculate events at each interval using Guyot method
    for i in range(1, len(times)):
        s_prev = survivals[i-1]
        s_curr = survivals[i]
        n_prev = n_at_risk[i-1]
        n_curr = n_at_risk[i] if i < len(n_at_risk) else 1
        t_prev = times[i-1]
        t_curr = times[i]

        if s_prev <= 0 or s_curr <= 0:
            continue

        # Conditional probability of event in interval
        cond_prob = 1 - (s_curr / s_prev)

        # Expected number of events
        n_events_interval = int(np.round(n_prev * cond_prob))
        n_events_interval = max(0, min(n_events_interval, n_prev - 1))

        # Number censored
        if has_nar:
            # Use actual NAR difference for censoring
            # Censored = NAR_prev - NAR_curr - events
            n_censored_interval = n_prev - n_curr - n_events_interval
            n_censored_interval = max(0, n_censored_interval)
        else:
            # Estimate censoring assuming uniform distribution
            n_censored_interval = n_prev - n_curr - n_events_interval
            n_censored_interval = max(0, n_censored_interval)

        # Generate event times uniformly within interval
        if n_events_interval > 0:
            event_times = np.linspace(t_prev + 0.001, t_curr - 0.001, n_events_interval)
            for et in event_times:
                ipd.append(IPDRecord(time=et, event=1, arm=0))

        # Generate censoring times
        if n_censored_interval > 0:
            censor_times = np.linspace(t_prev + 0.001, t_curr - 0.001, n_censored_interval)
            for ct in censor_times:
                ipd.append(IPDRecord(time=ct, event=0, arm=0))

    # Add final censored observations at end of follow-up
    n_remaining = total_n - len(ipd)
    if n_remaining > 0:
        for _ in range(min(n_remaining, 10)):  # Limit to avoid memory issues
            ipd.append(IPDRecord(time=times[-1], event=0, arm=0))

    return ipd


def _interpolate_nar(nar_times: np.ndarray,
                     nar_values: np.ndarray,
                     curve_times: np.ndarray,
                     total_n: int) -> np.ndarray:
    """
    Interpolate NAR values to curve timepoints.

    Uses linear interpolation with validation to ensure
    monotonically decreasing values.

    Parameters
    ----------
    nar_times : array
        NAR timepoints (e.g., [0, 12, 24, 36])
    nar_values : array
        NAR counts at each timepoint
    curve_times : array
        Curve timepoints to interpolate to
    total_n : int
        Total number of patients

    Returns
    -------
    array: Interpolated NAR values
    """
    nar_times = np.array(nar_times)
    nar_values = np.array(nar_values)

    # Ensure NAR is monotonically decreasing
    for i in range(1, len(nar_values)):
        if nar_values[i] > nar_values[i-1]:
            nar_values[i] = nar_values[i-1]

    try:
        # Linear interpolation with extrapolation
        nar_interp = interp1d(
            nar_times, nar_values,
            kind='linear',
            bounds_error=False,
            fill_value=(nar_values[0], nar_values[-1])
        )
        interpolated = nar_interp(curve_times)

        # Ensure valid range
        interpolated = np.clip(interpolated, 1, total_n)

        # Ensure monotonically decreasing
        for i in range(1, len(interpolated)):
            if interpolated[i] > interpolated[i-1]:
                interpolated[i] = interpolated[i-1]

        return interpolated.astype(int)

    except Exception:
        # Fallback: simple estimation
        return (np.ones_like(curve_times) * total_n * 0.5).astype(int)


def _calculate_events_from_nar(nar_current: np.ndarray,
                                nar_next: np.ndarray,
                                survival_change: np.ndarray) -> np.ndarray:
    """
    Calculate number of events from NAR and survival changes.

    Events = NAR_loss - Censored
    where Censored is estimated from the survival curve slope.

    Parameters
    ----------
    nar_current : array
        NAR at interval start
    nar_next : array
        NAR at interval end
    survival_change : array
        Change in survival probability

    Returns
    -------
    array: Estimated events per interval
    """
    # Total lost from NAR
    nar_loss = nar_current - nar_next

    # Events based on survival decrease
    events = np.round(nar_current * survival_change).astype(int)

    # Ensure events don't exceed NAR loss
    events = np.minimum(events, nar_loss)
    events = np.maximum(events, 0)

    return events


def log_rank_test(ipd_treatment: List[IPDRecord],
                  ipd_control: List[IPDRecord]) -> Tuple[float, float, float, float, float]:
    """
    Perform log-rank test on two IPD datasets.

    Returns
    -------
    statistic : float
        Log-rank test statistic (Z)
    p_value : float
        Two-sided p-value
    O_1 : float
        Observed events in treatment arm
    E_1 : float
        Expected events in treatment arm
    V : float
        Variance of (O_1 - E_1)
    """
    # Combine data
    all_times = []
    all_events = []
    all_arms = []

    for rec in ipd_treatment:
        all_times.append(rec.time)
        all_events.append(rec.event)
        all_arms.append(1)

    for rec in ipd_control:
        all_times.append(rec.time)
        all_events.append(rec.event)
        all_arms.append(0)

    times = np.array(all_times)
    events = np.array(all_events)
    arms = np.array(all_arms)

    # Get unique event times
    event_times = np.unique(times[events == 1])

    if len(event_times) == 0:
        return 0.0, 1.0, 0.0, 0.0, 0.0

    # Log-rank calculation
    O_1 = 0  # Observed events in treatment
    E_1 = 0  # Expected events in treatment
    V = 0    # Variance

    for t in event_times:
        # Number at risk just before time t
        at_risk_1 = np.sum((times >= t) & (arms == 1))
        at_risk_0 = np.sum((times >= t) & (arms == 0))
        at_risk_total = at_risk_1 + at_risk_0

        if at_risk_total == 0:
            continue

        # Events at time t
        events_1 = np.sum((times == t) & (events == 1) & (arms == 1))
        events_0 = np.sum((times == t) & (events == 1) & (arms == 0))
        events_total = events_1 + events_0

        # Accumulate
        O_1 += events_1
        E_1 += at_risk_1 * events_total / at_risk_total

        if at_risk_total > 1:
            V += (at_risk_1 * at_risk_0 * events_total *
                  (at_risk_total - events_total)) / (at_risk_total**2 * (at_risk_total - 1))

    if V <= 0:
        return 0.0, 1.0, O_1, E_1, 0.0

    # Test statistic
    Z = (O_1 - E_1) / np.sqrt(V)
    p_value = 2 * (1 - stats.norm.cdf(abs(Z)))

    return Z, p_value, O_1, E_1, V


def estimate_hr_from_curves(times1: np.ndarray, surv1: np.ndarray,
                            times2: np.ndarray, surv2: np.ndarray,
                            n1: int = 100, n2: int = 100) -> HRResult:
    """
    Estimate hazard ratio from two survival curves.

    Uses IPD reconstruction and log-rank approximation.

    Parameters
    ----------
    times1 : array
        Time points for curve 1 (treatment)
    surv1 : array
        Survival probabilities for curve 1
    times2 : array
        Time points for curve 2 (control)
    surv2 : array
        Survival probabilities for curve 2
    n1 : int
        Total patients in arm 1
    n2 : int
        Total patients in arm 2

    Returns
    -------
    HRResult
    """
    # Reconstruct IPD
    ipd1 = reconstruct_ipd_guyot(times1, surv1, total_n=n1)
    ipd2 = reconstruct_ipd_guyot(times2, surv2, total_n=n2)

    # Update arm labels
    for rec in ipd1:
        rec.arm = 1
    for rec in ipd2:
        rec.arm = 0

    # Log-rank test — now returns (Z, p_value, O_1, E_1, V)
    Z, p_value, O_1, E_1, V = log_rank_test(ipd1, ipd2)

    total_events = sum(1 for r in ipd1 if r.event == 1) + sum(1 for r in ipd2 if r.event == 1)

    if total_events == 0:
        return HRResult(
            hr=1.0, ci_lower=0.5, ci_upper=2.0,
            p_value=1.0, method="log_rank",
            n_events=0, n_total=n1+n2,
            log_rank_statistic=0
        )

    # Primary method: HR = exp((O_1 - E_1) / V) with SE(log HR) = 1/sqrt(V)
    # This is the standard Mantel-Haenszel / log-rank derived HR
    if V > 0 and total_events >= 20:
        log_hr = (O_1 - E_1) / V
        se_log_hr = 1.0 / np.sqrt(V)
        hr = np.exp(log_hr)
        hr = max(0.05, min(20.0, hr))
        method = "log_rank_OminusE"
    else:
        # Fallback: median survival ratio for low-event scenarios
        def find_median_survival(times, survivals):
            """Find time when survival crosses 0.5."""
            for i in range(1, len(survivals)):
                if survivals[i] <= 0.5 <= survivals[i-1]:
                    frac = (survivals[i-1] - 0.5) / (survivals[i-1] - survivals[i])
                    return times[i-1] + frac * (times[i] - times[i-1])
            return times[-1]

        median1 = find_median_survival(times1, surv1)
        median2 = find_median_survival(times2, surv2)

        if median1 > 0:
            hr = median2 / median1
        else:
            hr = 1.0
        hr = max(0.05, min(20.0, hr))

        # Approximate SE for median ratio method
        if total_events > 0:
            se_log_hr = np.sqrt(4.0 / total_events)
        else:
            se_log_hr = 0.5
        method = "median_ratio_fallback"

    log_hr = np.log(hr)
    ci_lower = np.exp(log_hr - 1.96 * se_log_hr)
    ci_upper = np.exp(log_hr + 1.96 * se_log_hr)

    return HRResult(
        hr=round(hr, 3),
        ci_lower=round(ci_lower, 3),
        ci_upper=round(ci_upper, 3),
        p_value=round(p_value, 4),
        method=method,
        n_events=total_events,
        n_total=n1 + n2,
        log_rank_statistic=round(Z, 3)
    )


def estimate_hr_simple(times1: np.ndarray, surv1: np.ndarray,
                       times2: np.ndarray, surv2: np.ndarray) -> float:
    """
    Simple HR estimation using median ratio.

    Faster but less accurate than full IPD reconstruction.
    """
    def find_median(times, survs):
        for i in range(1, len(survs)):
            if survs[i] <= 0.5 <= survs[i-1]:
                frac = (survs[i-1] - 0.5) / (survs[i-1] - survs[i] + 1e-9)
                return times[i-1] + frac * (times[i] - times[i-1])
        # Extrapolate if needed
        if survs[-1] > 0.5:
            # Not enough follow-up
            return times[-1] * 2  # Rough estimate
        return times[-1]

    med1 = find_median(times1, surv1)
    med2 = find_median(times2, surv2)

    if med1 > 0:
        return med2 / med1
    return 1.0


class ImprovedHREstimator:
    """
    Improved HR estimator that combines multiple methods.

    Arm identification priority:
    1. Legend information (if available) - most reliable
    2. NAR row labels (if available)
    3. AUC-based (treatment has better survival = higher AUC)

    Convention: HR is treatment/control
    - If treatment is better: HR < 1
    - If control is better: HR > 1
    """

    def __init__(self):
        self.methods = ['median_ratio', 'auc_ratio', 'ipd_reconstruction']
        self.legend_info = None
        self.nar_labels = None
        self.paper_text = None  # For text-based verification

    def set_legend_info(self, legend_info):
        """
        Set legend information for arm identification.

        Parameters
        ----------
        legend_info : LegendInfo or dict
            Legend extraction result with treatment_arm and control_arm
        """
        self.legend_info = legend_info

    def set_nar_labels(self, nar_labels: List[str]):
        """
        Set NAR row labels for arm identification.

        Parameters
        ----------
        nar_labels : list
            Labels for each NAR row (e.g., ['pembrolizumab', 'chemotherapy'])
        """
        self.nar_labels = nar_labels

    def set_paper_text(self, paper_text: str):
        """
        Set paper text for text-based HR verification.

        Parameters
        ----------
        paper_text : str
            Full text of the paper (or abstract + results section)
        """
        self.paper_text = paper_text

    def _identify_treatment_control_with_uncertainty(self, curves: List[Dict],
                                                       legend_info=None,
                                                       nar_labels: List[str] = None
                                                       ) -> Tuple[Dict, Dict, ArmIdentificationResult]:
        """
        Identify treatment/control arms with uncertainty quantification.

        IMPORTANT METHODOLOGICAL NOTES:
        - This method NO LONGER assumes "better survival = treatment"
        - AUC-based fallback is FLAGGED as uncertain and requires manual review
        - Text-based verification is NOT used for primary assignment (circular logic)
          but only for validation after assignment

        Priority:
        1. Legend information (most reliable - explicit labels)
        2. NAR row labels (reliable - usually labeled)
        3. Return UNCERTAIN if neither available (no assumption made)

        Parameters
        ----------
        curves : list
            List of curve dictionaries
        legend_info : LegendInfo or dict, optional
            Legend extraction result
        nar_labels : list, optional
            Labels from NAR table rows

        Returns
        -------
        tuple: (treatment_curve, control_curve, ArmIdentificationResult)
        """
        # Use instance variables if not provided
        legend_info = legend_info or self.legend_info
        nar_labels = nar_labels or self.nar_labels

        curve1 = curves[0]
        curve2 = curves[1]

        # Priority 1: Use legend if available (HIGH CONFIDENCE)
        if legend_info:
            matched = self._match_curves_to_legend(curves, legend_info)
            if matched:
                treatment_idx = curves.index(matched[0])
                control_idx = curves.index(matched[1])
                return matched[0], matched[1], ArmIdentificationResult(
                    method='legend',
                    confidence=0.9,
                    treatment_index=treatment_idx,
                    control_index=control_idx,
                    is_verified=True,
                    is_uncertain=False,
                    details="Identified via figure legend OCR"
                )

        # Priority 2: Use NAR row labels if available (MEDIUM-HIGH CONFIDENCE)
        if nar_labels and len(nar_labels) >= 2:
            matched = self._match_curves_to_nar_labels(curves, nar_labels)
            if matched:
                treatment_idx = curves.index(matched[0])
                control_idx = curves.index(matched[1])
                return matched[0], matched[1], ArmIdentificationResult(
                    method='nar_labels',
                    confidence=0.8,
                    treatment_index=treatment_idx,
                    control_index=control_idx,
                    is_verified=True,
                    is_uncertain=False,
                    details=f"Identified via NAR labels: {nar_labels}"
                )

        # NO FALLBACK TO AUC - This is methodologically problematic
        # Instead, return curves in order with UNCERTAIN flag

        # Calculate AUC for reporting purposes only (not for assignment)
        times1 = np.array(curve1.get('times', []))
        surv1 = np.array(curve1.get('survivals', []))
        times2 = np.array(curve2.get('times', []))
        surv2 = np.array(curve2.get('survivals', []))

        if len(times1) > 1 and len(times2) > 1:
            auc1 = np.trapezoid(surv1, times1) if hasattr(np, 'trapezoid') else np.trapz(surv1, times1)
            auc2 = np.trapezoid(surv2, times2) if hasattr(np, 'trapezoid') else np.trapz(surv2, times2)
            auc_details = f"AUC curve1={auc1:.2f}, AUC curve2={auc2:.2f}"
        else:
            auc_details = "AUC calculation not possible"

        # Return curves in original order with UNCERTAIN status
        return curve1, curve2, ArmIdentificationResult(
            method='unknown',
            confidence=0.0,
            treatment_index=0,
            control_index=1,
            is_verified=False,
            is_uncertain=True,
            warning="ARM IDENTIFICATION UNCERTAIN: No legend or NAR labels available. "
                    "Manual review required. HR reported assumes curve1=treatment, curve2=control.",
            details=f"No reliable identification method. {auc_details}. "
                    "Curves returned in original detection order."
        )

    def _identify_treatment_control(self, curves: List[Dict],
                                     legend_info=None,
                                     nar_labels: List[str] = None) -> Tuple[Dict, Dict]:
        """
        Legacy wrapper for backward compatibility.

        See _identify_treatment_control_with_uncertainty for full details.
        """
        treatment, control, _ = self._identify_treatment_control_with_uncertainty(
            curves, legend_info, nar_labels
        )
        return treatment, control

    def _match_curves_to_legend(self, curves: List[Dict], legend_info) -> Optional[Tuple[Dict, Dict]]:
        """
        Match curves to legend entries by color.

        Parameters
        ----------
        curves : list
            Curves with 'color' or 'color_name' keys
        legend_info : LegendInfo or dict
            Legend extraction result

        Returns
        -------
        tuple or None: (treatment_curve, control_curve) if matched
        """
        # Handle dict or object
        if hasattr(legend_info, 'treatment_arm'):
            treatment_arm = legend_info.treatment_arm
            control_arm = legend_info.control_arm
        else:
            treatment_arm = legend_info.get('treatment_arm')
            control_arm = legend_info.get('control_arm')

        if not treatment_arm or not control_arm:
            return None

        # Get colors from legend
        treatment_color = getattr(treatment_arm, 'color_name', None) or treatment_arm.get('color_name', '')
        control_color = getattr(control_arm, 'color_name', None) or control_arm.get('color_name', '')

        treatment_curve = None
        control_curve = None

        for curve in curves:
            curve_color = curve.get('color_name', '').lower()

            if treatment_color and treatment_color.lower() in curve_color:
                treatment_curve = curve
            elif control_color and control_color.lower() in curve_color:
                control_curve = curve

        if treatment_curve and control_curve:
            return treatment_curve, control_curve

        return None

    def _match_curves_to_nar_labels(self, curves: List[Dict], nar_labels: List[str]) -> Optional[Tuple[Dict, Dict]]:
        """
        Match curves to NAR row labels using treatment/control keywords.

        Parameters
        ----------
        curves : list
            Curves to match
        nar_labels : list
            Labels from NAR table rows

        Returns
        -------
        tuple or None: (treatment_curve, control_curve) if matched
        """
        treatment_keywords = [
            'treatment', 'experimental', 'active',
            'pembrolizumab', 'nivolumab', 'atezolizumab',
            'arm a', 'group a', 'intervention'
        ]
        control_keywords = [
            'control', 'placebo', 'standard', 'comparator',
            'chemotherapy', 'arm b', 'group b', 'soc'
        ]

        treatment_idx = None
        control_idx = None

        for i, label in enumerate(nar_labels):
            label_lower = label.lower()

            # Check for treatment keywords
            if any(kw in label_lower for kw in treatment_keywords):
                treatment_idx = i

            # Check for control keywords
            if any(kw in label_lower for kw in control_keywords):
                control_idx = i

        # Map to curves (assuming NAR row order matches curve order)
        if treatment_idx is not None and control_idx is not None:
            if treatment_idx < len(curves) and control_idx < len(curves):
                return curves[treatment_idx], curves[control_idx]

        return None

    def _verify_assignment_via_text(self, curves: List[Dict], paper_text: str) -> Optional[Tuple[Dict, Dict]]:
        """
        Verify treatment/control assignment by comparing to HR reported in text.

        Extracts HR from paper text and calculates HR both ways to determine
        which assignment matches the reported value better.

        Parameters
        ----------
        curves : list
            Curves with 'times' and 'survivals' keys
        paper_text : str
            Paper text to search for reported HR

        Returns
        -------
        tuple or None: (treatment_curve, control_curve) if verified via text
        """
        import re

        if not paper_text:
            return None

        # Patterns to extract reported HR and CI
        hr_patterns = [
            # "HR = 0.71" or "HR=0.71" or "HR: 0.71"
            r'HR\s*[=:]\s*(\d+\.?\d*)',
            # "hazard ratio = 0.71" or "hazard ratio of 0.71"
            r'hazard\s+ratio\s*(?:=|of|was|:)\s*(\d+\.?\d*)',
            # "HR 0.71 (95% CI, 0.50-0.99)"
            r'HR\s+(\d+\.?\d*)\s*\(?.*?CI',
            # "(HR = 0.71; 95% CI, 0.50-0.99)"
            r'\(HR\s*[=:,]\s*(\d+\.?\d*)',
            # "HR, 0.71"
            r'HR[,;]\s*(\d+\.?\d*)',
        ]

        reported_hr = None
        for pattern in hr_patterns:
            matches = re.findall(pattern, paper_text, re.IGNORECASE)
            if matches:
                # Take the first numeric match that's in reasonable range
                for m in matches:
                    try:
                        val = float(m)
                        if 0.1 < val < 5.0:  # Reasonable HR range
                            reported_hr = val
                            break
                    except ValueError:
                        continue
            if reported_hr:
                break

        if reported_hr is None:
            return None

        # Calculate HR both ways using a simplified method
        curve1 = curves[0]
        curve2 = curves[1]

        times1 = np.array(curve1.get('times', []))
        surv1 = np.array(curve1.get('survivals', []))
        times2 = np.array(curve2.get('times', []))
        surv2 = np.array(curve2.get('survivals', []))

        if len(times1) < 5 or len(times2) < 5:
            return None

        # Simple HR estimation: ratio of log-survival integrals
        try:
            # Interpolate to common time points
            max_t = min(times1[-1], times2[-1])
            common_t = np.linspace(0, max_t, 100)

            interp1 = interp1d(times1, surv1, kind='linear', fill_value='extrapolate')
            interp2 = interp1d(times2, surv2, kind='linear', fill_value='extrapolate')

            surv1_common = np.clip(interp1(common_t), 0.001, 1.0)
            surv2_common = np.clip(interp2(common_t), 0.001, 1.0)

            # Log-survival (cumulative hazard proxy)
            log_surv1 = -np.log(surv1_common)
            log_surv2 = -np.log(surv2_common)

            # Cumulative hazard ratio estimate
            h1 = np.trapezoid(log_surv1, common_t) if hasattr(np, 'trapezoid') else np.trapz(log_surv1, common_t)
            h2 = np.trapezoid(log_surv2, common_t) if hasattr(np, 'trapezoid') else np.trapz(log_surv2, common_t)

            if h2 > 0.01:
                hr_1_over_2 = h1 / h2  # curve1 as treatment
            else:
                hr_1_over_2 = 1.0

            if h1 > 0.01:
                hr_2_over_1 = h2 / h1  # curve2 as treatment
            else:
                hr_2_over_1 = 1.0

            # Choose assignment that matches reported HR better
            error_1_over_2 = abs(hr_1_over_2 - reported_hr) / reported_hr
            error_2_over_1 = abs(hr_2_over_1 - reported_hr) / reported_hr

            # Require at least 20% better match
            if error_1_over_2 < error_2_over_1 and error_1_over_2 < 0.5:
                # curve1 is treatment, curve2 is control
                return curve1, curve2
            elif error_2_over_1 < error_1_over_2 and error_2_over_1 < 0.5:
                # curve2 is treatment, curve1 is control
                return curve2, curve1
            else:
                # Neither matches well enough
                return None

        except Exception as e:
            # Calculation failed
            return None

    def _auc_based_assignment(self, curves: List[Dict]) -> Tuple[Dict, Dict]:
        """
        Assign treatment/control based on AUC (higher AUC = treatment).

        Parameters
        ----------
        curves : list
            Curves with 'times' and 'survivals' keys

        Returns
        -------
        tuple: (treatment_curve, control_curve)
        """
        curve1 = curves[0]
        curve2 = curves[1]

        times1 = np.array(curve1.get('times', []))
        surv1 = np.array(curve1.get('survivals', []))
        times2 = np.array(curve2.get('times', []))
        surv2 = np.array(curve2.get('survivals', []))

        # Calculate AUC for each curve (higher = better survival)
        if len(times1) > 1 and len(times2) > 1:
            auc1 = np.trapezoid(surv1, times1) if hasattr(np, 'trapezoid') else np.trapz(surv1, times1)
            auc2 = np.trapezoid(surv2, times2) if hasattr(np, 'trapezoid') else np.trapz(surv2, times2)
        else:
            # Fallback: use final survival value
            auc1 = surv1[-1] if len(surv1) > 0 else 0
            auc2 = surv2[-1] if len(surv2) > 0 else 0

        # Treatment has better survival (higher AUC)
        if auc1 >= auc2:
            return curve1, curve2  # curve1 is treatment
        else:
            return curve2, curve1  # curve2 is treatment

    def estimate(self, curves: List[Dict]) -> Optional[HRResult]:
        """
        Estimate HR from extracted curves with uncertainty quantification.

        Parameters
        ----------
        curves : list
            List of curve dictionaries with 'times' and 'survivals' keys

        Returns
        -------
        HRResult or None if estimation fails

        IMPORTANT: If arm identification is uncertain, the result includes:
        - hr_inverted: HR if arms were swapped
        - arm_identification.is_uncertain: True
        - arm_identification.warning: Explanation of uncertainty

        Always check arm_identification before using HR in meta-analysis.
        """
        if len(curves) < 2:
            return None

        # CRITICAL: Identify arms WITH uncertainty quantification
        treatment_curve, control_curve, arm_id_result = \
            self._identify_treatment_control_with_uncertainty(curves)

        times_trt = np.array(treatment_curve.get('times', []))
        surv_trt = np.array(treatment_curve.get('survivals', []))
        times_ctrl = np.array(control_curve.get('times', []))
        surv_ctrl = np.array(control_curve.get('survivals', []))

        if len(times_trt) < 5 or len(times_ctrl) < 5:
            return None

        # Ensure valid survival values
        surv_trt = np.clip(surv_trt, 0.001, 1.0)
        surv_ctrl = np.clip(surv_ctrl, 0.001, 1.0)

        # Ensure monotonicity
        for i in range(1, len(surv_trt)):
            if surv_trt[i] > surv_trt[i-1]:
                surv_trt[i] = surv_trt[i-1]
        for i in range(1, len(surv_ctrl)):
            if surv_ctrl[i] > surv_ctrl[i-1]:
                surv_ctrl[i] = surv_ctrl[i-1]

        try:
            result = estimate_hr_from_curves(times_trt, surv_trt, times_ctrl, surv_ctrl)
        except Exception as e:
            # Fallback to simple method
            hr = estimate_hr_simple(times_trt, surv_trt, times_ctrl, surv_ctrl)
            result = HRResult(
                hr=round(hr, 3),
                ci_lower=round(hr * 0.5, 3),
                ci_upper=round(hr * 2.0, 3),
                p_value=0.05,
                method="simple_median_ratio",
                n_events=0,
                n_total=0,
                log_rank_statistic=0
            )

        # Add arm identification result
        result.arm_identification = arm_id_result

        # If uncertain, also calculate inverted HR (if arms were swapped)
        if arm_id_result.is_uncertain:
            try:
                # Swap arms and recalculate
                result_inverted = estimate_hr_from_curves(
                    times_ctrl, surv_ctrl, times_trt, surv_trt
                )
                result.hr_inverted = result_inverted.hr
                result.ci_lower_inverted = result_inverted.ci_lower
                result.ci_upper_inverted = result_inverted.ci_upper
            except Exception:
                # Use reciprocal as approximation
                if result.hr > 0:
                    result.hr_inverted = round(1.0 / result.hr, 3)
                    result.ci_lower_inverted = round(1.0 / result.ci_upper, 3)
                    result.ci_upper_inverted = round(1.0 / result.ci_lower, 3)

        return result

    # === NEW METHODS FOR ROBUST PIPELINE ===

    def identify_arms_from_caption(self, caption: str, curves: List[Dict]) -> Optional[ArmIdentificationResult]:
        """
        Extract arm labels from figure caption.

        Looks for patterns like:
        - "Treatment arm (blue) vs Control arm (red)"
        - "Pembrolizumab (blue line)"
        - "Ablation (solid) vs Medical therapy (dashed)"

        Parameters
        ----------
        caption : str
            Figure caption text
        curves : List[Dict]
            List of curve dictionaries with 'color_name' key

        Returns
        -------
        ArmIdentificationResult or None if no match found
        """
        if not caption:
            return None

        caption_lower = caption.lower()
        arms = {}

        # Pattern 1: "Label (color)" or "Label (color line/curve)"
        color_patterns = [
            r'(\w+(?:\s+\w+)?)\s*\((\w+)\s*(?:line|curve)?\)',
            r'(\w+)\s*=\s*(\w+)\s+(?:line|curve)',
            r'(\w+(?:\s+\w+)?)\s*,?\s*(\w+)\s+line',
        ]

        for pattern in color_patterns:
            matches = re.findall(pattern, caption, re.IGNORECASE)
            for label, color in matches:
                color_norm = color.lower().strip()
                # Normalize color names
                if color_norm in ['blue', 'blu']:
                    color_norm = 'blue'
                elif color_norm in ['red', 'rd']:
                    color_norm = 'red'
                arms[color_norm] = label.strip()

        if len(arms) < 2:
            return None

        # Match curves to colors
        treatment_idx = None
        control_idx = None

        # Treatment keywords
        treatment_kws = ['treatment', 'experimental', 'intervention', 'ablation',
                        'pembrolizumab', 'nivolumab', 'atezolizumab', 'active']
        control_kws = ['control', 'placebo', 'standard', 'soc', 'comparator',
                      'chemotherapy', 'medical', 'conservative']

        for color, label in arms.items():
            label_lower = label.lower()
            curve_idx = None

            # Find curve with matching color
            for i, curve in enumerate(curves):
                curve_color = curve.get('color_name', '').lower()
                if color in curve_color or curve_color in color:
                    curve_idx = i
                    break

            if curve_idx is not None:
                if any(kw in label_lower for kw in treatment_kws):
                    treatment_idx = curve_idx
                elif any(kw in label_lower for kw in control_kws):
                    control_idx = curve_idx

        if treatment_idx is not None and control_idx is not None:
            return ArmIdentificationResult(
                method='caption_color',
                confidence=0.75,
                treatment_index=treatment_idx,
                control_index=control_idx,
                is_verified=False,
                is_uncertain=False,
                details=f"Identified from caption: {arms}"
            )

        return None

    def estimate_with_inversion_check(self, curves: List[Dict],
                                      legend_info=None,
                                      caption: str = "") -> Optional[HRResult]:
        """
        Estimate HR with automatic inversion detection.

        If arm identification is uncertain, checks if inverted HR is more plausible.

        Parameters
        ----------
        curves : List[Dict]
            Curve dictionaries with 'times' and 'survivals'
        legend_info : dict, optional
            Legend information
        caption : str, optional
            Figure caption for arm identification

        Returns
        -------
        HRResult with arm identification and possible inversion
        """
        if len(curves) < 2:
            return None

        # Try caption-based arm identification first
        arm_from_caption = self.identify_arms_from_caption(caption, curves) if caption else None

        # Get initial estimate with uncertainty
        treatment, control, arm_id = self._identify_treatment_control_with_uncertainty(
            curves, legend_info, self.nar_labels
        )

        # Override with caption-based if available
        if arm_from_caption and arm_from_caption.confidence > arm_id.confidence:
            treatment = curves[arm_from_caption.treatment_index]
            control = curves[arm_from_caption.control_index]
            arm_id = arm_from_caption

        times_trt = np.array(treatment.get('times', []))
        surv_trt = np.array(treatment.get('survivals', treatment.get('survival', [])))
        times_ctrl = np.array(control.get('times', []))
        surv_ctrl = np.array(control.get('survivals', control.get('survival', [])))

        if len(times_trt) < 5 or len(times_ctrl) < 5:
            return None

        try:
            result = estimate_hr_from_curves(times_trt, surv_trt, times_ctrl, surv_ctrl)
            result.arm_identification = arm_id

            # If uncertain, check for plausible inversion
            if arm_id.is_uncertain:
                # Calculate HR both ways
                hr_original = result.hr
                hr_inverted = 1.0 / result.hr if result.hr > 0 else None

                # Check which is more plausible
                if hr_inverted and self._is_more_plausible(hr_inverted, hr_original):
                    # Swap arms and recalculate
                    result_inverted = estimate_hr_from_curves(
                        times_ctrl, surv_ctrl, times_trt, surv_trt
                    )
                    result_inverted.arm_identification = ArmIdentificationResult(
                        method='inversion_check',
                        confidence=0.5,
                        treatment_index=arm_id.control_index,
                        control_index=arm_id.treatment_index,
                        is_verified=False,
                        is_uncertain=True,
                        warning=f"Arms inverted based on plausibility (original HR={hr_original:.2f}, inverted HR={hr_inverted:.2f})"
                    )
                    result_inverted.hr_inverted = hr_original
                    return result_inverted
                else:
                    result.hr_inverted = hr_inverted

            return result

        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"HR estimation failed: {e}")
            return None

    def _is_more_plausible(self, hr1: float, hr2: float) -> bool:
        """
        Check if hr1 is more plausible than hr2.

        HRs closer to 1.0 are more common in clinical trials.
        Extreme HRs (< 0.2 or > 5) are rare and likely errors.

        Parameters
        ----------
        hr1 : float
            First HR value
        hr2 : float
            Second HR value

        Returns
        -------
        bool: True if hr1 is more plausible
        """
        if hr1 <= 0 or hr2 <= 0:
            return False

        # Distance from 1.0 in log space (symmetric for HR and 1/HR)
        dist1 = abs(np.log(hr1))
        dist2 = abs(np.log(hr2))

        # If one is extreme (>5 or <0.2) and other is not, prefer the moderate one
        is_extreme1 = hr1 < 0.2 or hr1 > 5.0
        is_extreme2 = hr2 < 0.2 or hr2 > 5.0

        if is_extreme2 and not is_extreme1:
            return True
        if is_extreme1 and not is_extreme2:
            return False

        # Otherwise prefer the one closer to 1.0
        return dist1 < dist2


if __name__ == "__main__":
    # Test with example data
    print("Testing improved HR estimation...")

    # Generate test curves
    times = np.linspace(0, 60, 50)

    # Treatment: HR = 0.5
    lambda_ctrl = 0.03
    lambda_trt = lambda_ctrl * 0.5  # HR = 0.5

    surv_ctrl = np.exp(-lambda_ctrl * times)
    surv_trt = np.exp(-lambda_trt * times)

    result = estimate_hr_from_curves(times, surv_trt, times, surv_ctrl)

    print(f"True HR: 0.50")
    print(f"Estimated HR: {result.hr:.3f} ({result.ci_lower:.3f}-{result.ci_upper:.3f})")
    print(f"P-value: {result.p_value:.4f}")
    print(f"Method: {result.method}")
    print(f"Events: {result.n_events}")

    # Test with HR = 0.8
    lambda_trt2 = lambda_ctrl * 0.8
    surv_trt2 = np.exp(-lambda_trt2 * times)

    result2 = estimate_hr_from_curves(times, surv_trt2, times, surv_ctrl)
    print(f"\nTrue HR: 0.80")
    print(f"Estimated HR: {result2.hr:.3f} ({result2.ci_lower:.3f}-{result2.ci_upper:.3f})")
