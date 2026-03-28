from ground_truth_database import GroundTruthDatabase, GroundTruthEntry, RCTGroundTruth


def test_db_hash_updates_after_add_entry(tmp_path):
    db = GroundTruthDatabase(db_path=str(tmp_path))
    before = db.get_db_hash()

    entry = GroundTruthEntry(
        entry_id="",
        pdf_id="pdf_1",
        hr=0.82,
        ci_lower=0.7,
        ci_upper=0.95,
    )
    db.add_entry(entry)

    assert db.get_db_hash() != before


def test_db_hash_updates_after_add_rct_ground_truth(tmp_path):
    db = GroundTruthDatabase(db_path=str(tmp_path))
    before = db.get_db_hash()

    db.add_rct_ground_truth(
        RCTGroundTruth(
            paper_id="trial_a",
            pdf_path="trial_a.pdf",
            hr_reported=0.8,
            ci_lower=0.7,
            ci_upper=0.9,
            therapeutic_area="oncology",
        )
    )

    assert db.get_db_hash() != before


def test_query_rct_ground_truths_filters_by_therapeutic_area(tmp_path):
    db = GroundTruthDatabase(db_path=str(tmp_path))

    db.add_rct_ground_truth(
        RCTGroundTruth(
            paper_id="onc_a",
            pdf_path="onc.pdf",
            hr_reported=0.8,
            ci_lower=0.7,
            ci_upper=0.9,
            therapeutic_area="Oncology",
            outcome_type="OS",
        )
    )
    db.add_rct_ground_truth(
        RCTGroundTruth(
            paper_id="cardio_b",
            pdf_path="cardio.pdf",
            hr_reported=1.1,
            ci_lower=0.95,
            ci_upper=1.3,
            therapeutic_area="Cardiology",
            outcome_type="MACE",
        )
    )

    oncology = db.query_rct_ground_truths(therapeutic_area="oncology")

    assert len(oncology) == 1
    assert oncology[0].paper_id == "onc_a"
