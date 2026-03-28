"""
recover_single_trial.py
-----------------------
Recover one trial summary in isolation.

Usage:
  python recover_single_trial.py --stem PMC12024468
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import expanded_gold_trials as e

OUTPUT_DIR = e.OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PLACEHOLDER_METHODS = {"resume_placeholder_fail", "recovery_crash_placeholder"}

TIERS = [
    {"label": "t1", "dpi": 300, "max_pages": 12, "n_per_arm": 100, "timeout": 1800},
    {"label": "t2", "dpi": 260, "max_pages": 10, "n_per_arm": 90, "timeout": 1500},
    {"label": "t3", "dpi": 220, "max_pages": 8, "n_per_arm": 80, "timeout": 1200},
    {"label": "t4", "dpi": 200, "max_pages": 6, "n_per_arm": 70, "timeout": 900},
]


def summary_path_for_trial(trial):
    stem = e._safe_stem(trial["pdf"])
    return OUTPUT_DIR / f"{stem}_summary.json"


def load_trials_by_stem():
    trials = e.ALL_TRIALS + e.load_phase4_trials()
    return {e._safe_stem(t["pdf"]): t for t in trials}


def is_placeholder(path: Path) -> bool:
    if not path.exists():
        return True
    try:
        j = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return True
    return j.get("hr_method") in PLACEHOLDER_METHODS


def write_crash_placeholder(trial, notes):
    path = summary_path_for_trial(trial)
    payload = {
        "pdf_path": trial["pdf"],
        "pdf_name": trial["pdf"],
        "hr": None,
        "ci_lower": None,
        "ci_upper": None,
        "hr_method": "recovery_crash_placeholder",
        "total_ipd_records": 0,
        "n_curves_found": 0,
        "processing_time_s": 0.0,
        "warnings": notes,
        "pipeline_version": "recovery-single",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


CHILD_CODE = r"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import expanded_gold_trials as e
from km_pipeline import KMPipeline, write_summary_json

trial = json.loads(os.environ["TRIAL_JSON"])
tier = json.loads(os.environ["TIER_JSON"])
out_path = Path(os.environ["OUT_JSON"])

pdf_path = e.get_pdf_path(trial)
target = trial.get("outcome_type")

try:
    pipe = KMPipeline(
        dpi=tier["dpi"],
        max_pages=tier["max_pages"],
        n_per_arm=tier["n_per_arm"],
    )
    result = pipe.extract(str(pdf_path), target_endpoint=target)
    write_summary_json(result, str(out_path))
except Exception as ex:
    payload = {
        "pdf_path": trial["pdf"],
        "pdf_name": trial["pdf"],
        "hr": None,
        "ci_lower": None,
        "ci_upper": None,
        "hr_method": "child_exception",
        "total_ipd_records": 0,
        "n_curves_found": 0,
        "processing_time_s": 0.0,
        "warnings": [f"child_exception: {ex}"],
        "pipeline_version": "recovery-child",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
"""


def recover_trial(trial):
    out_path = summary_path_for_trial(trial)
    notes = []

    for tier in TIERS:
        env = os.environ.copy()
        env["TRIAL_JSON"] = json.dumps(trial)
        env["TIER_JSON"] = json.dumps(tier)
        env["OUT_JSON"] = str(out_path)

        try:
            proc = subprocess.run(
                [sys.executable, "-c", CHILD_CODE],
                cwd=str(Path(__file__).resolve().parent),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=tier["timeout"],
            )
            notes.append(f"{tier['label']}: rc={proc.returncode}")
        except subprocess.TimeoutExpired:
            notes.append(f"{tier['label']}: timeout_{tier['timeout']}s")

        if out_path.exists() and not is_placeholder(out_path):
            return True, notes

    write_crash_placeholder(trial, notes)
    return False, notes


def main():
    parser = argparse.ArgumentParser(description="Recover one trial summary by stem")
    parser.add_argument("--stem", required=True, help="Safe stem (e.g., PMC12024468)")
    parser.add_argument("--force", action="store_true", help="Recover even if non-placeholder")
    args = parser.parse_args()

    trials_by_stem = load_trials_by_stem()
    trial = trials_by_stem.get(args.stem)
    if trial is None:
        print(f"ERROR: stem not found: {args.stem}")
        sys.exit(2)

    out_path = summary_path_for_trial(trial)
    if out_path.exists() and (not is_placeholder(out_path)) and (not args.force):
        print(f"already_non_placeholder: {out_path.name}")
        return

    print(f"recovering: {trial['name']} ({trial.get('area', 'unknown')})")
    print(f"summary: {out_path}")
    ok, notes = recover_trial(trial)
    print(f"recovered={ok}")
    for n in notes:
        print(f"note: {n}")


if __name__ == "__main__":
    main()
