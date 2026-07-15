import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from realbatch.analyze_batch import _select_candidate


def test_select_candidate_prefers_published_point_over_extreme_ci_edge():
    candidates = [
        {"hr": 1.5099, "method": "bw", "n_arms": 2, "figure": "f0"},
        {"hr": 14.144, "method": "bw", "n_arms": 2, "figure": "f2"},
        {"hr": 2.8514, "method": "bw", "n_arms": 2, "figure": "f3"},
    ]

    selected = _select_candidate(candidates, published_hrs=[(1.43, 0.14, 14.71)])

    assert selected["figure"] == "f0"
    assert selected["hr"] == 1.5099


def test_select_candidate_fails_closed_without_matching_context():
    candidates = [
        {"hr": 1.5099, "method": "bw", "n_arms": 2, "figure": "f0"},
        {"hr": 14.144, "method": "bw", "n_arms": 2, "figure": "f2"},
    ]

    assert _select_candidate(candidates) is None


def test_select_candidate_keeps_single_candidate_without_matching_context():
    candidate = {"hr": 0.73, "method": "color", "n_arms": 2, "figure": "f0"}

    assert _select_candidate([candidate]) == candidate
