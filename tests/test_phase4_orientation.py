from phase4_orientation import is_derived_hr_method, resolve_phase4_hr_orientation


def test_resolve_orientation_allows_reciprocal_for_derived_methods():
    final_hr, within_ci, orientation = resolve_phase4_hr_orientation(
        ext_hr=2.0,
        gt_lo=0.45,
        gt_hi=0.55,
        hr_method="derived_hr_match",
    )

    assert final_hr == 0.5
    assert within_ci is True
    assert orientation == "reciprocal"


def test_resolve_orientation_disallows_reciprocal_for_non_derived_methods():
    final_hr, within_ci, orientation = resolve_phase4_hr_orientation(
        ext_hr=2.0,
        gt_lo=0.45,
        gt_hi=0.55,
        hr_method="text_hr_match",
    )

    assert final_hr == 2.0
    assert within_ci is False
    assert orientation == "direct"


def test_is_derived_hr_method_handles_text_derived_only_and_none():
    assert is_derived_hr_method("text_derived_only") is True
    assert is_derived_hr_method(None) is False


def test_resolve_orientation_allows_reciprocal_when_trial_source_is_derived():
    final_hr, within_ci, orientation = resolve_phase4_hr_orientation(
        ext_hr=2.0,
        gt_lo=0.45,
        gt_hi=0.55,
        hr_method="text_hr_match",
        trial_hr_source="derived",
    )

    assert final_hr == 0.5
    assert within_ci is True
    assert orientation == "reciprocal"
