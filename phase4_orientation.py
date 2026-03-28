"""Shared HR orientation logic for Phase 4 runners."""

from typing import Optional, Tuple


def is_derived_hr_method(
    hr_method: Optional[str],
    trial_hr_source: Optional[str] = None,
) -> bool:
    """Return True when reciprocal correction is allowed."""
    method_val = (hr_method or "").lower()
    source_val = (trial_hr_source or "").lower()
    return ("derived" in method_val) or (source_val == "derived")


def resolve_phase4_hr_orientation(
    ext_hr: float,
    gt_lo: float,
    gt_hi: float,
    hr_method: Optional[str],
    trial_hr_source: Optional[str] = None,
) -> Tuple[float, bool, str]:
    """
    Resolve final HR orientation against GT CI.

    Reciprocal orientation is only allowed for derived HR methods because
    these sources can be directionally ambiguous.
    """
    within_ci = bool(gt_lo <= ext_hr <= gt_hi)
    orientation = "direct"
    final_hr = ext_hr

    reciprocal_allowed = is_derived_hr_method(hr_method, trial_hr_source)
    recip_hr = 1.0 / ext_hr if ext_hr > 0 else None

    if (not within_ci and reciprocal_allowed and recip_hr is not None and
            gt_lo <= recip_hr <= gt_hi):
        final_hr = recip_hr
        within_ci = True
        orientation = "reciprocal"

    return final_hr, within_ci, orientation
