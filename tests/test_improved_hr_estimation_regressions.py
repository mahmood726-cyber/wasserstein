import improved_hr_estimation as hr_mod


def test_uncertain_inversion_fallback_handles_missing_ci(monkeypatch):
    estimator = hr_mod.ImprovedHREstimator()

    curve_t = {
        "times": [0, 1, 2, 3, 4, 5],
        "survivals": [1.0, 0.95, 0.9, 0.85, 0.8, 0.75],
    }
    curve_c = {
        "times": [0, 1, 2, 3, 4, 5],
        "survivals": [1.0, 0.92, 0.84, 0.76, 0.68, 0.6],
    }

    arm_id = hr_mod.ArmIdentificationResult(
        method="auc_fallback",
        confidence=0.5,
        treatment_index=0,
        control_index=1,
        is_verified=False,
        is_uncertain=True,
    )

    monkeypatch.setattr(
        estimator,
        "_identify_treatment_control_with_uncertainty",
        lambda curves: (curve_t, curve_c, arm_id),
    )

    def _raise(*_args, **_kwargs):
        raise RuntimeError("forced failure")

    monkeypatch.setattr(hr_mod, "estimate_hr_from_curves", _raise)
    monkeypatch.setattr(hr_mod, "estimate_hr_simple", lambda *_a, **_k: 2.0)

    result = estimator.estimate([curve_t, curve_c])

    assert result is not None
    assert result.hr == 2.0
    assert result.hr_inverted == 0.5
    assert result.ci_lower_inverted is None
    assert result.ci_upper_inverted is None
