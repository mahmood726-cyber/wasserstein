"""
RCT Ground Truth Database
=========================

Single source of truth for RCT validation data.
Each entry has: trial name, PDF path, reported HR with 95% CI.

HR source types:
  "reported" — HR with CI explicitly stated in the paper text
  "derived"  — HR estimated from event rates using exponential
               approximation; CIs are wide estimates. NOT reported
               in the paper.

IMPORTANT: Ground truth is used ONLY for evaluation metrics.
The pipeline never sees these values during extraction.
"""

import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RCTGroundTruth:
    """Ground truth entry for an RCT."""
    name: str
    pdf_path: str
    hr: float
    ci_lower: float
    ci_upper: float
    endpoint: str = ""
    notes: str = ""
    hr_source: str = "reported"  # "reported" | "derived"

    @property
    def available(self) -> bool:
        return os.path.exists(self.pdf_path)


ABLATION_DIR = r"C:\Users\user\Downloads\Ablation"


GROUND_TRUTH: List[RCTGroundTruth] = [
    # -------------------------------------------------------------------
    # Papers with REPORTED HRs (verified from paper text)
    # -------------------------------------------------------------------
    RCTGroundTruth(
        name="FIRE AND ICE",
        pdf_path=os.path.join(ABLATION_DIR, "NEJMoa1602014.pdf"),
        hr=0.96, ci_lower=0.76, ci_upper=1.22,
        endpoint="Clinical failure (recurrence of AF/AFL/AT, AADs, re-ablation)",
        notes="Cryo vs RF, paroxysmal AF. Kuck et al, NEJM 2016. "
              "Primary efficacy, modified ITT. P<0.001 for noninferiority.",
        hr_source="reported",
    ),
    RCTGroundTruth(
        name="CIRCA-DOSE",
        pdf_path=os.path.join(ABLATION_DIR,
            "andrade-et-al-cryoballoon-or-radiofrequency-ablation-for-atrial-"
            "fibrillation-assessed-by-continuous-monitoring.pdf"),
        hr=1.08, ci_lower=0.78, ci_upper=1.50,
        endpoint="AF/AFL/AT recurrence (combined cryo vs CF-RF)",
        notes="3-arm RCT: CF-RF vs Cryo-4min vs Cryo-2min. Andrade et al, "
              "Circulation 2019. HR 1.08 is pooled cryo vs CF-RF (secondary). "
              "Previously mislabeled as EARLY-AF (a different NEJM 2021 trial).",
        hr_source="reported",
    ),
    RCTGroundTruth(
        name="CRRF-PeAF",
        pdf_path=os.path.join(ABLATION_DIR, "ehaf451.pdf"),
        hr=0.99, ci_lower=0.69, ci_upper=1.43,
        endpoint="Atrial tachyarrhythmia recurrence at 12 months",
        notes="Cryo vs RF for persistent AF. Miyamoto et al, EHJ 2025. "
              "Noninferiority demonstrated (margin 1.5). P=0.96 log-rank.",
        hr_source="reported",
    ),
    # -------------------------------------------------------------------
    # Papers WITHOUT reported HRs (derived from event rates)
    # These use exponential approximation: HR = log(S_trt) / log(S_ref)
    # CIs are wide estimates reflecting the uncertainty.
    # -------------------------------------------------------------------
    RCTGroundTruth(
        name="ADVENT",
        pdf_path=os.path.join(ABLATION_DIR,
            "reddy-et-al-pulsed-field-vs-conventional-thermal-ablation-for-"
            "paroxysmal-atrial-fibrillation.pdf"),
        hr=0.92, ci_lower=0.55, ci_upper=1.50,
        endpoint="Treatment failure at 1 year (Bayesian)",
        notes="PFA vs thermal ablation. Reddy et al, JACC 2024. "
              "No HR reported; Bayesian framework with 73.3% vs 71.3% success. "
              "HR derived from event rates. Noninferiority met.",
        hr_source="derived",
    ),
    RCTGroundTruth(
        name="FROZEN AF",
        pdf_path=os.path.join(ABLATION_DIR,
            "chun-et-al-cryoballoon-versus-laserballoon.pdf"),
        hr=0.90, ci_lower=0.50, ci_upper=1.60,
        endpoint="Freedom from AF/AFL/AT recurrence off AADs (90-365 d)",
        notes="Cryo vs laser balloon. Chun et al, Circ Arrhythm 2021. "
              "No HR reported; proportions 80.0% vs 78.0%, P=ns. "
              "HR derived from event rates.",
        hr_source="derived",
    ),
    RCTGroundTruth(
        name="CRAVE",
        pdf_path=os.path.join(ABLATION_DIR,
            "pak-et-al-cryoballoon-versus-high-power-short-duration-"
            "radiofrequency-ablation-for-pulmonary-vein-isolation-in-"
            "patients.pdf"),
        hr=0.91, ci_lower=0.45, ci_upper=1.85,
        endpoint="AF recurrence at 12 months",
        notes="Cryo vs HPSD RF. Pak et al, Circ Arrhythm 2021. "
              "No treatment-comparison HR reported; log-rank P=0.840. "
              "Recurrence 11.1% vs 12.1%. Paper reports only predictor HRs "
              "(multivariate: extra-PV trigger HR=2.88, AAD HR=5.77). "
              "HR derived from event rates.",
        hr_source="derived",
    ),
    RCTGroundTruth(
        name="LBRF-PERSISTENT",
        pdf_path=os.path.join(ABLATION_DIR,
            "schmidt-et-al-laser-balloon-or-wide-area-circumferential-"
            "irrigated-radiofrequency-ablation-for-persistent-atrial.pdf"),
        hr=0.93, ci_lower=0.50, ci_upper=1.70,
        endpoint="Freedom from AF/AFL/AT recurrence (90-365 d)",
        notes="Laser vs RF for persistent AF. Schmidt et al, Circ Arrhythm "
              "2017. No HR reported; proportions 71.2% vs 69.3%, P=0.40. "
              "HR derived from event rates.",
        hr_source="derived",
    ),
    RCTGroundTruth(
        name="HIPAF",
        pdf_path=os.path.join(ABLATION_DIR, "euaf066.pdf"),
        hr=1.47, ci_lower=0.70, ci_upper=2.50,
        endpoint="Composite arrhythmia recurrence/AADs/re-ablation at 1 year",
        notes="vHPSD RF vs CBA. Sultan et al, Europace 2025. "
              "No HR reported; event-free 73.8% vs 81.4%. "
              "Non-inferiority NOT met (diff -7.6%, 95% CI -20.1 to 4.9). "
              "Previously mislabeled as CABANA (a different trial). "
              "HR derived from event rates.",
        hr_source="derived",
    ),
]


def get_available() -> List[RCTGroundTruth]:
    """Return only entries with existing PDF files."""
    return [g for g in GROUND_TRUTH if g.available]


def get_reported_only() -> List[RCTGroundTruth]:
    """Return only entries with reported (not derived) HRs."""
    return [g for g in GROUND_TRUTH if g.available
            and g.hr_source == "reported"]


def get_by_name(name: str) -> Optional[RCTGroundTruth]:
    """Look up ground truth by trial name (case-insensitive)."""
    name_lower = name.lower()
    for g in GROUND_TRUTH:
        if g.name.lower() == name_lower:
            return g
    return None


if __name__ == "__main__":
    available = get_available()
    reported = get_reported_only()
    print(f"Ground truth: {len(GROUND_TRUTH)} trials, "
          f"{len(available)} with PDFs available, "
          f"{len(reported)} with reported HRs")
    print()
    for g in GROUND_TRUTH:
        status = "OK" if g.available else "MISSING"
        src = "R" if g.hr_source == "reported" else "D"
        print(f"  [{status:7s}] [{src}] {g.name:20s}  HR={g.hr:.2f} "
              f"({g.ci_lower:.2f}-{g.ci_upper:.2f})")
