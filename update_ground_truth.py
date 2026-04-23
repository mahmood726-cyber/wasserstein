# sentinel:skip-file — hardcoded paths are fixture/registry/audit-narrative data for this repo's research workflow, not portable application configuration. Same pattern as push_all_repos.py and E156 workbook files.
"""Add 12 new gold entries to ground_truth_300.json"""
import json
import sys
import io
from datetime import datetime, timezone

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

with open('ground_truth_300.json', 'r', encoding='utf-8') as f:
    db = json.load(f)

new_entries = [
    {
        "pdf_path": r"C:\Users\user\Downloads\NEJMoa2107038.pdf",
        "specialty": "cardiology", "hr": 0.79, "ci_lower": 0.69, "ci_upper": 0.90,
        "has_ci": True, "tier": "gold",
        "context": "EMPEROR-Preserved. Empagliflozin in HFpEF. Primary: CV death or HF hospitalization. Anker et al, NEJM 2021. HR 0.79 (0.69-0.90). N=5988.",
        "pattern_name": "expanded_v12_gold", "pipeline_validated": False,
        "trial_name": "EMPEROR-Preserved", "hr_source": "reported",
    },
    {
        "pdf_path": r"C:\Users\user\Downloads\NEJMoa2206286.pdf",
        "specialty": "cardiology", "hr": 0.82, "ci_lower": 0.73, "ci_upper": 0.92,
        "has_ci": True, "tier": "gold",
        "context": "DELIVER. Dapagliflozin in HFmrEF/HFpEF. Primary: worsening HF or CV death. Solomon et al, NEJM 2022. HR 0.82 (0.73-0.92). N=6263.",
        "pattern_name": "expanded_v12_gold", "pipeline_validated": True,
        "trial_name": "DELIVER", "hr_source": "reported",
    },
    {
        "pdf_path": r"C:\Users\user\Downloads\NEJMoa2307563.pdf",
        "specialty": "cardiology", "hr": 0.80, "ci_lower": 0.72, "ci_upper": 0.90,
        "has_ci": True, "tier": "gold",
        "context": "SELECT. Semaglutide in overweight/obese with CV disease. Primary: CV death, MI, or stroke. Lincoff et al, NEJM 2023. HR 0.80 (0.72-0.90). N=17604.",
        "pattern_name": "expanded_v12_gold", "pipeline_validated": True,
        "trial_name": "SELECT", "hr_source": "reported",
    },
    {
        "pdf_path": r"C:\Users\user\Downloads\NEJMoa1904143.pdf",
        "specialty": "cardiology", "hr": 0.72, "ci_lower": 0.55, "ci_upper": 0.95,
        "has_ci": True, "tier": "gold",
        "context": "AFIRE. Rivaroxaban mono vs combo in AF+stable CAD. Primary efficacy: stroke/embolism/MI/death. Yasuda et al, NEJM 2019. HR 0.72 (0.55-0.95). N=2236.",
        "pattern_name": "expanded_v12_gold", "pipeline_validated": False,
        "trial_name": "AFIRE", "hr_source": "reported",
    },
    {
        "pdf_path": r"C:\Users\user\Downloads\Partner3.pdf",
        "specialty": "cardiology", "hr": 0.54, "ci_lower": 0.37, "ci_upper": 0.79,
        "has_ci": True, "tier": "gold",
        "context": "PARTNER 3. TAVR vs surgery in low-risk aortic stenosis. Primary: death/stroke/rehospitalization at 1yr. Mack et al, NEJM 2019. HR 0.54 (0.37-0.79). N=1000.",
        "pattern_name": "expanded_v12_gold", "pipeline_validated": False,
        "trial_name": "PARTNER 3", "hr_source": "reported",
    },
    {
        "pdf_path": r"C:\Users\user\Downloads\Augustus.pdf",
        "specialty": "cardiology", "hr": 0.69, "ci_lower": 0.58, "ci_upper": 0.81,
        "has_ci": True, "tier": "gold",
        "context": "AUGUSTUS. 2x2 factorial: apixaban vs VKA in AF+ACS/PCI. Primary: bleeding. Lopes et al, NEJM 2019. HR 0.69 (0.58-0.81). N=4614.",
        "pattern_name": "expanded_v12_gold", "pipeline_validated": False,
        "trial_name": "AUGUSTUS", "hr_source": "reported",
    },
    {
        "pdf_path": r"C:\Users\user\Downloads\IvabradineandoutcomesinchronicheartfailureSHIFT-arandomisedplacebo-controlledstudy.pdf",
        "specialty": "cardiology", "hr": 0.82, "ci_lower": 0.75, "ci_upper": 0.90,
        "has_ci": True, "tier": "gold",
        "context": "SHIFT. Ivabradine in chronic HF. Primary: CV death or HF hospitalization. Swedberg et al, Lancet 2010. HR 0.82 (0.75-0.90). N=6558.",
        "pattern_name": "expanded_v12_gold", "pipeline_validated": True,
        "trial_name": "SHIFT", "hr_source": "reported",
    },
    {
        "pdf_path": r"C:\Users\user\Downloads\EffectofmetoprololCR-XLinchronicheartfailure.pdf",
        "specialty": "cardiology", "hr": 0.66, "ci_lower": 0.53, "ci_upper": 0.81,
        "has_ci": True, "tier": "gold",
        "context": "MERIT-HF. Metoprolol CR/XL in chronic HF. Primary: all-cause mortality. MERIT-HF Study Group, Lancet 1999. RR 0.66 (0.53-0.81). N=3991.",
        "pattern_name": "expanded_v12_gold", "pipeline_validated": True,
        "trial_name": "MERIT-HF", "hr_source": "reported",
    },
    {
        "pdf_path": r"C:\Users\user\Downloads\entrust.pdf",
        "specialty": "cardiology", "hr": 0.83, "ci_lower": 0.65, "ci_upper": 1.05,
        "has_ci": True, "tier": "gold",
        "context": "ENTRUST-AF PCI. Edoxaban vs VKA in AF+PCI. Primary: bleeding. Vranckx et al, Lancet 2019. HR 0.83 (0.65-1.05). N=1506.",
        "pattern_name": "expanded_v12_gold", "pipeline_validated": True,
        "trial_name": "ENTRUST-AF PCI", "hr_source": "reported",
    },
    {
        "pdf_path": r"C:\Users\user\Downloads\Pioneer AF.pdf",
        "specialty": "cardiology", "hr": 0.59, "ci_lower": 0.47, "ci_upper": 0.76,
        "has_ci": True, "tier": "gold",
        "context": "PIONEER AF-PCI. 3-arm trial. Rivaroxaban low-dose vs VKA+DAPT. Primary safety: bleeding (group 1 vs 3). Gibson et al, NEJM 2016. HR 0.59 (0.47-0.76). N=2124.",
        "pattern_name": "expanded_v12_gold", "pipeline_validated": False,
        "trial_name": "PIONEER AF-PCI", "hr_source": "reported",
    },
    {
        "pdf_path": r"C:\Users\user\Downloads\NEJMoa0802987.pdf",
        "specialty": "diabetes", "hr": 0.90, "ci_lower": 0.82, "ci_upper": 0.98,
        "has_ci": True, "tier": "gold",
        "context": "ADVANCE. Intensive glucose control in T2DM. Primary: combined major macro+microvascular events. ADVANCE Group, NEJM 2008. HR 0.90 (0.82-0.98). N=11140.",
        "pattern_name": "expanded_v12_gold", "pipeline_validated": True,
        "trial_name": "ADVANCE", "hr_source": "reported",
    },
    {
        "pdf_path": r"C:\Users\user\Downloads\TheLancet1999Investigators.pdf",
        "specialty": "cardiology", "hr": 0.66, "ci_lower": 0.54, "ci_upper": 0.81,
        "has_ci": True, "tier": "gold",
        "context": "CIBIS-II. Bisoprolol in chronic HF. Primary: all-cause mortality. CIBIS-II Investigators, Lancet 1999. HR 0.66 (0.54-0.81). N=2647.",
        "pattern_name": "expanded_v12_gold", "pipeline_validated": True,
        "trial_name": "CIBIS-II", "hr_source": "reported",
    },
]

db['entries'].extend(new_entries)

# Update summary
db['summary']['total_entries'] = len(db['entries'])
gold = sum(1 for e in db['entries'] if e.get('tier') == 'gold')
silver = sum(1 for e in db['entries'] if e.get('tier') == 'silver')
bronze = sum(1 for e in db['entries'] if e.get('tier') == 'bronze')
db['summary']['tier_counts'] = {'gold': gold, 'silver': silver, 'bronze': bronze}
db['summary']['with_ci'] = sum(1 for e in db['entries'] if e.get('has_ci'))
db['summary']['pipeline_validated'] = sum(1 for e in db['entries'] if e.get('pipeline_validated'))

spec_counts = {}
for e in db['entries']:
    s = e.get('specialty', 'other')
    spec_counts[s] = spec_counts.get(s, 0) + 1
db['summary']['specialty_counts'] = spec_counts
db['generated'] = datetime.now(timezone.utc).isoformat()
db['version'] = '1.1'

with open('ground_truth_300.json', 'w', encoding='utf-8') as f:
    json.dump(db, f, indent=2, ensure_ascii=False)

print(f"Updated ground_truth_300.json:")
print(f"  Total entries: {db['summary']['total_entries']}")
print(f"  Gold: {gold}, Silver: {silver}, Bronze: {bronze}")
print(f"  Pipeline validated: {db['summary']['pipeline_validated']}")
