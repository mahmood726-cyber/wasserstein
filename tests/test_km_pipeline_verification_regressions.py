from km_pipeline import PipelineResult


def test_pipeline_result_to_dict_preserves_verification_fields():
    result = PipelineResult(
        pdf_path="paper.pdf",
        pdf_name="paper.pdf",
        hr=0.91,
        ci_lower=0.82,
        ci_upper=1.01,
        p_value=None,
        hr_method="text_hr_match",
        ipd_arm1=None,
        ipd_arm2=None,
        total_ipd_records=0,
        curve1_times=None,
        curve1_survivals=None,
        curve2_times=None,
        curve2_survivals=None,
        confidence=0.85,
        orientation_method="text_hr_match",
        text_hr=0.9,
        n_curves_found=2,
        n_pages_scanned=1,
        processing_time_s=1.2,
        warnings=[],
        verification_level="consistent",
        verification_checks_passed=["CI_ORDERED", "CI_CONTAINS_POINT"],
        verification_checks_failed=[],
        verification_hash="abc123def4567890",
    )

    payload = result.to_dict()

    assert payload["verification_level"] == "consistent"
    assert payload["verification_checks_passed"] == [
        "CI_ORDERED", "CI_CONTAINS_POINT"
    ]
    assert payload["verification_checks_failed"] == []
    assert payload["verification_hash"] == "abc123def4567890"
