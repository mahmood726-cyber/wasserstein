import time

import robust_km_pipeline as rkp


class _DummyRes:
    def __init__(self, ci_lower, ci_upper, method="guyot"):
        self.ci_lower = ci_lower
        self.ci_upper = ci_upper
        self.method = method


def _build_best(hr, res):
    return {
        "hr": hr,
        "res": res,
        "rank": 1,
        "score": 120.0,
        "desc": "blue(p1) vs red(p1)",
        "orient": "forward",
        "hr_fwd": hr,
        "hr_inv": None,
    }


def test_build_result_records_consistent_verification():
    pipeline = rkp.RobustKMPipeline()
    best = _build_best(0.8, _DummyRes(0.7, 0.91))

    result = pipeline._build_result(
        best=best,
        text_hr=0.82,
        text_ctx="primary endpoint",
        method="text_hr_match",
        all_curves=[{}, {}],
        pdf_path="paper.pdf",
        pdf_hash="abc123",
        n_pages=1,
        warnings=[],
        t0=time.time() - 0.1,
    )

    assert result.provenance.verification_level == "consistent"
    assert "CI_ORDERED" in result.provenance.verification_checks_passed
    assert "CI_CONTAINS_POINT" in result.provenance.verification_checks_passed
    assert result.provenance.verification_checks_failed == []
    assert result.provenance.verification_hash


def test_build_result_flags_ci_containment_failure_as_critical():
    pipeline = rkp.RobustKMPipeline()
    best = _build_best(0.8, _DummyRes(0.9, 1.1))

    result = pipeline._build_result(
        best=best,
        text_hr=0.8,
        text_ctx="primary endpoint",
        method="text_hr_match",
        all_curves=[{}, {}],
        pdf_path="paper.pdf",
        pdf_hash="abc123",
        n_pages=1,
        warnings=[],
        t0=time.time() - 0.1,
    )

    assert result.provenance.verification_level == "violated"
    assert "CI_CONTAINS_POINT" in result.provenance.verification_checks_failed
    assert any(
        "Critical consistency checks failed" in warning
        for warning in result.warnings
    )
    assert result.confidence <= 0.75


def test_build_result_keeps_ci_checks_non_blocking_when_ci_missing():
    pipeline = rkp.RobustKMPipeline()
    best = _build_best(1.2, None)

    result = pipeline._build_result(
        best=best,
        text_hr=None,
        text_ctx=None,
        method="derived_hr_fallback",
        all_curves=[{}, {}],
        pdf_path="paper.pdf",
        pdf_hash="abc123",
        n_pages=1,
        warnings=[],
        t0=time.time() - 0.1,
    )

    assert result.provenance.verification_level == "consistent"
    assert "CI_ORDERED" in result.provenance.verification_checks_passed
    assert "CI_CONTAINS_POINT" in result.provenance.verification_checks_passed
    assert result.provenance.verification_checks_failed == []
