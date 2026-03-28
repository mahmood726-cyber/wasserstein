from dataclasses import dataclass

import km_pipeline as kp
from robust_km_pipeline import ExtractionProvenance, HRExtractionResult


@dataclass
class _DummyIPD:
    time: float
    event: int
    arm: int = 0


def _make_hr_result(pair_desc: str, warnings=None) -> HRExtractionResult:
    if warnings is None:
        warnings = []
    prov = ExtractionProvenance(
        pdf_path="paper.pdf",
        pdf_sha256="abc123",
        pages_scanned=1,
        total_curves_extracted=2,
        pair_description=pair_desc,
        pair_quality_score=120.0,
        pair_rank=1,
        orientation="forward",
        orientation_method="text_hr_match",
    )
    return HRExtractionResult(
        hr=1.0,
        ci_lower=0.8,
        ci_upper=1.2,
        confidence=0.8,
        provenance=prov,
        warnings=warnings,
    )


def test_reconstruct_ipd_transforms_cif_like_curves_before_guyot(monkeypatch):
    pipeline = kp.KMPipeline()

    c1 = {
        "times": [0, 1, 2, 3, 4],
        "survivals": [0.02, 0.08, 0.15, 0.25, 0.35],  # CIF-like (increasing)
        "color_name": "blue",
        "page": 1,
    }
    c2 = {
        "times": [0, 1, 2, 3, 4],
        "survivals": [0.01, 0.06, 0.12, 0.20, 0.30],  # CIF-like (increasing)
        "color_name": "red",
        "page": 1,
    }
    pair_desc = "blue(p1) vs red(p1)"
    pipeline._cached_all_curves = [c1, c2]

    monkeypatch.setattr(
        pipeline,
        "_select_top_pairs",
        lambda _curves, _top_k: [(c1, c2, 120.0)],
    )
    monkeypatch.setattr(
        pipeline,
        "_extract_nar_for_curve",
        lambda _curve, arm_index=0: (100, None, None),
    )

    captured_survs = []

    def _fake_reconstruct(times, survs, **_kwargs):
        captured_survs.append(list(survs))
        return [_DummyIPD(time=float(times[0]), event=1)]

    monkeypatch.setattr(kp, "reconstruct_ipd_guyot", _fake_reconstruct)

    hr_result = _make_hr_result(pair_desc)
    arm1, arm2, c1_data, c2_data = pipeline._reconstruct_ipd(
        "paper.pdf", hr_result
    )

    assert arm1 is not None and arm2 is not None
    assert captured_survs[0][0] > captured_survs[0][-1]
    assert captured_survs[1][0] > captured_survs[1][-1]
    assert c1_data[1][0] > c1_data[1][-1]
    assert c2_data[1][0] > c2_data[1][-1]
    assert any(
        "cumulative-incidence-like curves" in w for w in hr_result.warnings
    )


def test_reconstruct_ipd_rejects_mixed_curve_representations(monkeypatch):
    pipeline = kp.KMPipeline()

    c1 = {
        "times": [0, 1, 2, 3, 4],
        "survivals": [1.00, 0.94, 0.86, 0.79, 0.70],  # KM-like (decreasing)
        "color_name": "blue",
        "page": 1,
    }
    c2 = {
        "times": [0, 1, 2, 3, 4],
        "survivals": [0.01, 0.05, 0.11, 0.19, 0.28],  # CIF-like (increasing)
        "color_name": "red",
        "page": 1,
    }
    pair_desc = "blue(p1) vs red(p1)"
    pipeline._cached_all_curves = [c1, c2]

    monkeypatch.setattr(
        pipeline,
        "_select_top_pairs",
        lambda _curves, _top_k: [(c1, c2, 120.0)],
    )
    monkeypatch.setattr(
        pipeline,
        "_extract_nar_for_curve",
        lambda _curve, arm_index=0: (100, None, None),
    )

    calls = {"n": 0}

    def _fake_reconstruct(*_args, **_kwargs):
        calls["n"] += 1
        return [_DummyIPD(time=1.0, event=1)]

    monkeypatch.setattr(kp, "reconstruct_ipd_guyot", _fake_reconstruct)

    hr_result = _make_hr_result(pair_desc)
    arm1, arm2, c1_data, c2_data = pipeline._reconstruct_ipd(
        "paper.pdf", hr_result
    )

    assert arm1 is None and arm2 is None
    assert c1_data is None and c2_data is None
    assert calls["n"] == 0
    assert any(
        "failed time-to-event normalization for IPD" in w
        for w in hr_result.warnings
    )
