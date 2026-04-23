# sentinel:skip-file — hardcoded paths are fixture/registry/audit-narrative data for this repo's research workflow, not portable application configuration. Same pattern as push_all_repos.py and E156 workbook files.
"""
select_diverse_trials.py — Phase 4 Trial Selection for v1.5
============================================================
Scans ground_truth_300.json for gold/silver trials with CI and existing PDFs,
checks each for KM figure presence, and selects stratified candidates across
7 specialties with HR range diversity.

v1.5 targets: 95 new trials (135 total with existing 40).

Selection criteria:
  1. gold/silver tier, has_ci=True, HR in [0.1, 5.0], PDF exists
  2. PDF contains KM-related text in first 8 pages (>=2 matches)
  3. Context suggests individual RCT (not meta-analysis/review)
  4. CI width >= 0.02 (reject impossibly precise pooled estimates)
  5. Stratified by HR range bins for diversity
  6. Prefer: named trials > gold > reported HR > RCT context > narrow CI
"""

import json
import os
import sys
import io
import re
from pathlib import Path

# UTF-8 stdout for Windows
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                  errors='replace')

WASSERSTEIN_DIR = Path(__file__).parent
GT_PATH = WASSERSTEIN_DIR / "ground_truth_300.json"

# PDFs already in the v1.4 40-trial set — EXCLUDE from new selection
EXISTING_PDFS = {
    # 13 original AF-ablation
    "Cardiovasc electrophysiol - 2015 - HUNTER - Point\u2010by\u2010Point Radiofrequency Ablation Versus the Cryoballoon or a Novel.pdf",
    "Cardiovasc electrophysiol - 2023 - Mililis - Radiofrequency versus cryoballoon catheter ablation in patients with.pdf",
    "NEJMoa1602014.pdf",
    "andrade-et-al-cryoballoon-or-radiofrequency-ablation-for-atrial-fibrillation-assessed-by-continuous-monitoring.pdf",
    "chun-et-al-cryoballoon-versus-laserballoon.pdf",
    "ehaf451.pdf",
    "euaf066.pdf",
    "eut398.pdf",
    "euu064.pdf",
    "pak-et-al-cryoballoon-versus-high-power-short-duration-radiofrequency-ablation-for-pulmonary-vein-isolation-in-patients.pdf",
    "reddy-et-al-pulsed-field-vs-conventional-thermal-ablation-for-paroxysmal-atrial-fibrillation.pdf",
    "s12872-017-0566-6.pdf",
    "schmidt-et-al-laser-balloon-or-wide-area-circumferential-irrigated-radiofrequency-ablation-for-persistent-atrial.pdf",
    # 12 cardio/metabolic
    "NEJMoa2107038.pdf", "NEJMoa2206286.pdf", "NEJMoa2307563.pdf",
    "NEJMoa1904143.pdf", "Partner3.pdf", "Augustus.pdf",
    "IvabradineandoutcomesinchronicheartfailureSHIFT-arandomisedplacebo-controlledstudy.pdf",
    "EffectofmetoprololCR-XLinchronicheartfailure.pdf",
    "entrust.pdf", "Pioneer AF.pdf", "NEJMoa0802987.pdf",
    "TheLancet1999Investigators.pdf",
    # 15 Phase 2 diverse
    "PMC10115555.pdf", "PMC10427418.pdf", "PMC10553121.pdf",
    "PMC11943310.pdf",
    "40036884_Effects of SGLT2 inhibition on insulin u.pdf",
    "40978864_Sodium-Glucose Cotransporter 2 SGLT2 Inh.pdf",
    "PMC10323201.pdf", "PMC10952210.pdf", "PMC10990610.pdf",
    "PMC10805160.pdf", "PMC11296275.pdf", "PMC11448330.pdf",
    "PMC10052578.pdf", "PMC10105623.pdf", "PMC11103724.pdf",
}

# Per-specialty target counts
SPECIALTY_TARGETS = {
    'oncology': 25,
    'cardiology': 20,
    'diabetes': 10,
    'rheumatology': 10,
    'infectious': 10,
    'respiratory': 10,
    'neurology': 10,
}

# HR range bins for stratified selection (ensure diversity)
HR_BINS = [
    (0.1, 0.5, "protective-strong"),
    (0.5, 0.8, "protective-moderate"),
    (0.8, 1.2, "neutral"),
    (1.2, 5.0, "harmful"),
]

# Meta-analysis indicators in context
META_INDICATORS = re.compile(
    r'\bI[2\u00b2]\s*=|\bmeta-analy|\bpooled\s+(analysis|data|estimate)|'
    r'\bheterogeneity\b|\bsystematic\s+review\b|\bforest\s+plot\b|'
    r'\bnetwork\s+meta\b|\brandom.effects\b|\bfixed.effects\b',
    re.IGNORECASE
)

# Review paper indicators
REVIEW_INDICATORS = re.compile(
    r'\bin\s+another\s+study\b|\breview\s+found\b|\bsystematic\s+review\b|'
    r'\bpooled\s+analysis\b|\bcombined\s+analysis\b',
    re.IGNORECASE
)

# RCT indicators in context
RCT_INDICATORS = re.compile(
    r'\b(randomiz|RCT|placebo|double.blind|open.label|phase\s*[1234IViv]|'
    r'trial\b|versus\b|treated\s+with\b|intention.to.treat)',
    re.IGNORECASE
)

# KM figure keywords to search in PDF text
KM_KEYWORDS = re.compile(
    r'Kaplan|Meier|survival\s+(curve|analysis|probability|rate|function|estimate)|'
    r'time.to.event|hazard\s+ratio|log.rank|cumulative\s+(incidence|hazard|event)|'
    r'event.free\s+survival|progression.free|overall\s+survival|'
    r'disease.free|recurrence.free|time\s+to\s+(first|primary)',
    re.IGNORECASE
)


def scan_pdf_for_km(pdf_path, max_pages=8):
    """Check if PDF has KM-related text in first N pages."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return True  # Can't check, assume yes

    try:
        doc = fitz.open(str(pdf_path))
        text = ""
        for i in range(min(max_pages, len(doc))):
            text += doc[i].get_text()
        doc.close()

        matches = KM_KEYWORDS.findall(text)
        return len(matches) >= 2  # At least 2 KM-related mentions
    except Exception:
        return False


def score_candidate(entry):
    """Score a candidate trial for selection priority.
    Higher = better candidate."""
    score = 0

    # Named trials are much better
    if entry.get('trial_name'):
        score += 100

    # Gold > silver
    if entry['tier'] == 'gold':
        score += 50
    elif entry['tier'] == 'silver':
        score += 10

    # Reported HR source is more reliable
    if entry.get('hr_source') == 'reported':
        score += 20

    # Narrow CI = more precise (penalize wide CI)
    ci_width = entry['ci_upper'] - entry['ci_lower']
    if ci_width < 0.3:
        score += 15
    elif ci_width < 0.5:
        score += 10
    elif ci_width > 1.5:
        score -= 15

    # RCT context indicators
    ctx = entry.get('context', '')
    if RCT_INDICATORS.search(ctx):
        score += 30

    # Meta-analysis / review context (strong negative)
    if META_INDICATORS.search(ctx):
        score -= 80
    if REVIEW_INDICATORS.search(ctx):
        score -= 60

    # Prefer HRs not too close to 1.0 (clearer signal)
    hr = entry['hr']
    if abs(hr - 1.0) > 0.15:
        score += 5

    return score


def get_hr_bin(hr):
    """Return the HR range bin for a given HR."""
    for lo, hi, label in HR_BINS:
        if lo <= hr < hi:
            return label
    return "harmful"  # HR >= 5.0


def select_stratified(scored_candidates, target_n, spec_name):
    """Select trials with HR range diversity."""
    # Group by HR bin
    bins = {}
    for score, entry in scored_candidates:
        b = get_hr_bin(entry['hr'])
        bins.setdefault(b, []).append((score, entry))

    # Allocate per-bin: proportional to available, min 1 per bin if available
    total_available = sum(len(v) for v in bins.values())
    if total_available == 0:
        return []

    selected = []
    remaining_target = target_n

    # First pass: ensure at least 1 per non-empty bin
    for label in [b[2] for b in HR_BINS]:
        if label in bins and bins[label] and remaining_target > 0:
            # Take top candidate from this bin
            entry_pair = bins[label][0]
            selected.append(entry_pair)
            bins[label] = bins[label][1:]
            remaining_target -= 1

    # Second pass: fill remaining proportionally from largest bins
    all_remaining = []
    for label, entries in bins.items():
        all_remaining.extend(entries)
    # Sort by score descending
    all_remaining.sort(key=lambda x: -x[0])

    for score, entry in all_remaining:
        if remaining_target <= 0:
            break
        # Check not already selected (by pdf_path)
        if any(s[1]['pdf_path'] == entry['pdf_path'] for s in selected):
            continue
        selected.append((score, entry))
        remaining_target -= 1

    return selected


def _pdf_dedupe_key(pdf_path):
    """Canonical key for path-level deduplication."""
    try:
        return str(Path(pdf_path).resolve()).lower()
    except Exception:
        return str(Path(pdf_path)).lower()


def _candidate_dedupe_key(entry):
    """Stable key for removing repeated trials across specialty labels.

    Prefer PDF basename so identical PDFs copied into multiple specialty
    directories are treated as one candidate.
    """
    pdf_name = Path(entry.get('pdf_path', '')).name.lower()
    if pdf_name:
        return pdf_name
    return _pdf_dedupe_key(entry.get('pdf_path', ''))


def dedupe_candidates_by_pdf(scored_candidates):
    """Keep one best-scored entry per PDF across specialties."""
    best_by_pdf = {}
    for score, entry in scored_candidates:
        key = _candidate_dedupe_key(entry)
        tier_score = 1 if entry.get('tier') == 'gold' else 0
        src_score = 1 if entry.get('hr_source') == 'reported' else 0
        named_score = 1 if entry.get('trial_name') else 0
        name = (entry.get('trial_name')
                or Path(entry.get('pdf_path', '')).stem).lower()
        rank = (score, tier_score, src_score, named_score, name)

        existing = best_by_pdf.get(key)
        if existing is None or rank > existing['rank']:
            best_by_pdf[key] = {'rank': rank, 'score': score, 'entry': entry}

    deduped = [(v['score'], v['entry']) for v in best_by_pdf.values()]
    deduped.sort(key=lambda x: -x[0])
    removed = len(scored_candidates) - len(deduped)
    return deduped, removed


def compute_specialty_targets(base_targets, requested_total, available_counts):
    """Compute specialty targets that sum to requested_total when possible."""
    specs = sorted(base_targets.keys())
    capacities = {s: max(0, int(available_counts.get(s, 0))) for s in specs}
    max_total = sum(capacities.values())

    requested = max(0, int(requested_total))
    requested = min(requested, max_total)

    targets = {s: 0 for s in specs}
    if requested == 0:
        return targets

    nonempty = [s for s in specs if capacities[s] > 0]
    if requested >= len(nonempty):
        for s in nonempty:
            targets[s] = 1

    remaining = requested - sum(targets.values())
    if remaining <= 0:
        return targets

    weight_sum = sum(base_targets.get(s, 0) for s in nonempty)
    if weight_sum <= 0:
        weights = {s: 1.0 / len(nonempty) for s in nonempty}
    else:
        weights = {s: base_targets.get(s, 0) / weight_sum for s in nonempty}
    raw_quota = {s: requested * weights[s] for s in nonempty}

    while remaining > 0:
        candidates = [s for s in nonempty if targets[s] < capacities[s]]
        if not candidates:
            break
        best_spec = max(
            candidates,
            key=lambda s: (
                raw_quota[s] - targets[s],
                weights[s],
                capacities[s] - targets[s],
            ),
        )
        targets[best_spec] += 1
        remaining -= 1

    return targets


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 4 trial selection")
    parser.add_argument("--skip-km-scan", action="store_true",
                        help="Skip KM keyword scan (faster, less precise)")
    parser.add_argument("--target-total", type=int, default=95,
                        help="Total new trials to select (default 95)")
    args = parser.parse_args()
    if args.target_total <= 0:
        parser.error("--target-total must be > 0")

    with open(GT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    entries = data['entries']
    print(f"Total ground truth entries: {len(entries)}")

    # Phase 1: Basic filtering
    candidates = []
    skipped = {'tier': 0, 'ci': 0, 'hr_range': 0, 'pdf': 0,
               'existing': 0, 'ci_width': 0}
    for e in entries:
        if e['tier'] not in ('gold', 'silver'):
            skipped['tier'] += 1
            continue
        if not e.get('has_ci'):
            skipped['ci'] += 1
            continue
        hr = e.get('hr')
        if hr is None or hr < 0.1 or hr > 5.0:
            skipped['hr_range'] += 1
            continue
        # Reject impossibly precise CI (likely pooled data)
        ci_width = e['ci_upper'] - e['ci_lower']
        if ci_width < 0.02:
            skipped['ci_width'] += 1
            continue
        pdf_path = e.get('pdf_path', '')
        if not os.path.exists(pdf_path):
            skipped['pdf'] += 1
            continue
        # Skip existing trials
        pdf_name = Path(pdf_path).name
        if pdf_name in EXISTING_PDFS:
            skipped['existing'] += 1
            continue
        candidates.append(e)

    print(f"After basic filter: {len(candidates)}")
    print(f"  Skipped: {skipped}")

    # Phase 2: Score all candidates
    scored_all = [(score_candidate(e), e) for e in candidates]
    scored_all, dedup_removed = dedupe_candidates_by_pdf(scored_all)
    print(f"After PDF dedupe: {len(scored_all)} "
          f"(removed {dedup_removed} duplicates)")

    # Phase 3: KM scan (optional, slow)
    if not args.skip_km_scan:
        print("\nScanning PDFs for KM keywords (this may take a while)...")
        km_passed = []
        km_failed = 0
        for i, (score, entry) in enumerate(scored_all):
            if i % 50 == 0 and i > 0:
                print(f"  Scanned {i}/{len(scored_all)}, "
                      f"{len(km_passed)} passed, {km_failed} failed")
            if scan_pdf_for_km(entry['pdf_path']):
                km_passed.append((score, entry))
            else:
                km_failed += 1
        print(f"  KM scan: {len(km_passed)} passed, {km_failed} failed")
        scored_all = km_passed
    else:
        print("Skipping KM scan (--skip-km-scan)")

    # Phase 4: Select per specialty with stratification
    all_specs = sorted(SPECIALTY_TARGETS.keys())
    results = {}
    spec_pools = {}
    for spec in all_specs:
        spec_scored = [(s, e) for s, e in scored_all
                       if e.get('specialty') == spec]
        # Remove meta-analysis entries
        spec_scored = [(s, e) for s, e in spec_scored
                       if not (META_INDICATORS.search(e.get('context', ''))
                               and not e.get('trial_name'))]
        spec_scored.sort(key=lambda x: -x[0])
        spec_pools[spec] = spec_scored

    available_by_spec = {spec: len(spec_pools[spec]) for spec in all_specs}
    spec_targets = compute_specialty_targets(
        SPECIALTY_TARGETS, args.target_total, available_by_spec)
    total_target = sum(spec_targets.values())
    if total_target < args.target_total:
        print(f"NOTE: requested {args.target_total} trials but only "
              f"{total_target} unique candidates available after filtering.")

    total_selected = 0
    for spec in all_specs:
        target = spec_targets[spec]
        spec_scored = spec_pools[spec]
        selected = select_stratified(spec_scored, target, spec)

        results[spec] = selected
        total_selected += len(selected)

        # Print summary
        hr_vals = [e['hr'] for _, e in selected]
        bins_used = set(get_hr_bin(h) for h in hr_vals) if hr_vals else set()
        print(f"\n{spec.upper()}: {len(selected)}/{target} selected "
              f"(from {len(spec_scored)} candidates)")
        print(f"  HR bins: {sorted(bins_used)}")
        if hr_vals:
            print(f"  HR range: [{min(hr_vals):.2f}, {max(hr_vals):.2f}]")
        for score, entry in selected[:3]:  # Show top 3
            tn = entry.get('trial_name', Path(entry['pdf_path']).stem[:30])
            print(f"  [{score:+d}] {tn}: HR={entry['hr']} "
                  f"[{entry['ci_lower']}, {entry['ci_upper']}]")
        if len(selected) > 3:
            print(f"  ... and {len(selected) - 3} more")

    # Phase 5: Output
    print(f"\n{'='*60}")
    print(f"PHASE 4 SELECTION SUMMARY: {total_selected}/{total_target} trials")
    print(f"{'='*60}")

    # Map pdf_dir paths to short keys
    DIR_MAP = {
        r"C:\Users\user\Downloads\Ablation": "ablation",
        r"C:\Users\user\Downloads": "downloads",
        r"C:\Users\user\oncology_rcts": "oncology",
        r"C:\Users\user\diabetes_rcts": "diabetes",
        r"C:\Users\user\respiratory_rcts": "respiratory",
        r"C:\Users\user\neurology_rcts": "neurology",
        r"C:\Users\user\infectious_rcts": "infectious",
        r"C:\Users\user\cardiology_rcts": "cardiology",
        r"C:\Users\user\rheumatology_rcts": "rheumatology",
    }

    all_selected = []
    for spec in all_specs:
        for score, e in results[spec]:
            pdf_path = e['pdf_path']
            pdf_name = Path(pdf_path).name
            pdf_dir_full = str(Path(pdf_path).parent)
            pdf_dir = DIR_MAP.get(pdf_dir_full, pdf_dir_full)

            trial_dict = {
                "name": e.get('trial_name') or pdf_name.replace('.pdf', ''),
                "pdf": pdf_name,
                "pdf_dir": pdf_dir,
                "gt_hr": e['hr'],
                "gt_ci_lower": e['ci_lower'],
                "gt_ci_upper": e['ci_upper'],
                "hr_source": e.get('hr_source', 'unknown'),
                "area": spec,
                "tier": e['tier'],
                "score": score,
            }
            all_selected.append(trial_dict)

    # Per-specialty summary table
    print(f"\n{'Specialty':<15} {'Selected':>8} {'Target':>7} "
          f"{'HR range':>12}")
    print("-" * 50)
    for spec in all_specs:
        sel = [t for t in all_selected if t['area'] == spec]
        if sel:
            hrs = [t['gt_hr'] for t in sel]
            print(f"{spec:<15} {len(sel):>8} {spec_targets[spec]:>7} "
                  f"{min(hrs):.2f}-{max(hrs):.2f}")
        else:
            print(f"{spec:<15} {'0':>8} {spec_targets[spec]:>7}")
    print(f"{'TOTAL':<15} {len(all_selected):>8} "
          f"{total_target:>7}")

    # Save to JSON
    out_path = WASSERSTEIN_DIR / "selected_phase4_trials.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_selected, f, indent=2, default=str, ensure_ascii=False)
    print(f"\nSaved to: {out_path}")

    # Generate Python code snippet for expanded_gold_trials.py
    snippet_path = WASSERSTEIN_DIR / "phase4_snippet.py"
    with open(snippet_path, 'w', encoding='utf-8') as f:
        f.write("# Auto-generated Phase 4 trials for expanded_gold_trials.py\n")
        f.write(f"# {len(all_selected)} trials across "
                f"{len(set(t['area'] for t in all_selected))} specialties\n\n")
        f.write("PHASE4_TRIALS = [\n")
        for t in all_selected:
            f.write(f"    {{\n")
            f.write(f'        "name": {json.dumps(t["name"])},\n')
            f.write(f'        "pdf": {json.dumps(t["pdf"])},\n')
            f.write(f'        "pdf_dir": {json.dumps(t["pdf_dir"])},\n')
            f.write(f'        "gt_hr": {t["gt_hr"]}, '
                    f'"gt_ci_lower": {t["gt_ci_lower"]}, '
                    f'"gt_ci_upper": {t["gt_ci_upper"]},\n')
            f.write(f'        "hr_source": "{t["hr_source"]}", '
                    f'"area": "{t["area"]}",\n')
            f.write(f"    }},\n")
        f.write("]\n")
    print(f"Python snippet: {snippet_path}")


if __name__ == '__main__':
    main()
