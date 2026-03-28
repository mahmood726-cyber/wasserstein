"""Debug script to see all HR candidates from failing PDFs."""
import sys
import io
import re
import platform

if sys.platform == 'win32':
    def _safe_wmi_query(*args, **kwargs):
        raise OSError("WMI bypassed")
    platform._wmi_query = _safe_wmi_query

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                  errors='replace')

import fitz
from pathlib import Path

FAILING_TRIALS = [
    ("PMC10553121", r"C:\Users\user\oncology_rcts\PMC10553121.pdf",
     0.73, "OS"),
    ("PMC10990610", r"C:\Users\user\respiratory_rcts\PMC10990610.pdf",
     0.66, "PFS"),
    ("PMC11296275", r"C:\Users\user\neurology_rcts\PMC11296275.pdf",
     0.69, "PFS"),
    ("PMC10052578", r"C:\Users\user\infectious_rcts\PMC10052578.pdf",
     0.69, "PFS"),
]

# Patterns from robust_km_pipeline.py
PATTERNS = [
    r'hazard\s+ratio[,;:\s]+(\d+\.?\d*)\s*\(?\s*95\s*%?\s*(?:CI|confidence(?:\s+interval)?(?:\s*\[CI\])?)[,;:\s]+(\d+\.?\d*)\s*(?:to|-|\u2013)\s*(\d+\.?\d*)',
    r'hazard\s+ratio(?:\s+for\s+[^,;]{1,60})?[,;:\s]+(\d+\.?\d*)\s*[;,]\s*95\s*%?\s*confidence\s+interval\s*(?:\[CI\])?\s*[,;:\s]+(\d+\.?\d*)\s*(?:to|-|\u2013)\s*(\d+\.?\d*)',
    r'HR[,;:\s=]+(\d+\.?\d*)\s*\(?\s*95\s*%?\s*(?:CI|confidence(?:\s+interval)?(?:\s*\[CI\])?)[,;:\s]+(\d+\.?\d*)\s*(?:to|-|\u2013)\s*(\d+\.?\d*)',
    r'hazard\s+ratio[,;:\s]+(\d+\.?\d*)\s*\(\s*(\d+\.?\d*)\s*(?:to|-|\u2013)\s*(\d+\.?\d*)\s*\)',
    r'relative\s+risk[,;:\s]+(?:of\s+)?(\d+\.?\d*)\s*[\(\[]?\s*95\s*%?\s*(?:CI|confidence(?:\s+interval)?(?:\s*\[CI\])?)[,;:\s]+(\d+\.?\d*)\s*(?:to|-|\u2013)\s*(\d+\.?\d*)',
    r'HR[,;:\s=]+(\d+\.?\d*)\s*\[\s*(\d+\.?\d*)\s*(?:to|-|\u2013)\s*(\d+\.?\d*)\s*\]',
    r'HR[,;:\s=]+(\d+\.?\d*)\s*[;,]\s*95\s*%?\s*CI[,;:\s]+(\d+\.?\d*)\s*(?:to|-|\u2013)\s*(\d+\.?\d*)',
    r'hazard\s+ratio\s*\[HR\][,;:\s]+(\d+\.?\d*)',
    r'HR\s*\(95\s*%?\s*CI\)\s*[,;:\s]*(\d+\.?\d*)\s*\(\s*(\d+\.?\d*)\s*(?:to|-|\u2013)\s*(\d+\.?\d*)\s*\)',
    r'(?:hazard\s+ratio|HR)[,;:\s=]+(\d+\.?\d*)',
]

_NEGATIVE = re.compile(
    r'multivariat|multivariable|adjust|model\s*\d|predictor|'
    r'covariat|regression|independent|subgroup|stratif|'
    r'per.protocol|sensitivity|landmark|competing|'
    r'secondary|pooled|combined|interaction|'
    r'biomarker|prognostic\s+factor|monocyte|neutrophil|'
    r'associated\s+with\s+(?:poor|worse|better)',
    re.IGNORECASE,
)

_CITED_STUDY = re.compile(
    r'\b\w+\s+et\s+al\.?\s*[\[\d(,;]', re.IGNORECASE,
)

_CITED_TRIAL_PRE = re.compile(
    r'(?:[A-Z]{2,}[-\s]?\d*\s+(?:trial|study)|'
    r'\b\w+\s+et\s+al\.?\s*(?:\d{4}|\(\d{4}\)))',
    re.IGNORECASE,
)

_PRIMARY_KW = re.compile(
    r'primary|main\s+(?:end|out)|principal|overall\s+survival'
    r'|progression.free|disease.free|event.free'
    r'|composite\s+(?:end|out)|MACE',
    re.IGNORECASE,
)

_EP_MAP = {
    'OS': r'overall\s+survival|(?<!\w)OS\b',
    'PFS': r'progression.free\s+survival|(?<!\w)PFS\b',
    'DFS': r'disease.free\s+survival|(?<!\w)DFS\b',
    'EFS': r'event.free\s+survival|(?<!\w)EFS\b',
}


for name, pdf_path, gt_hr, target_ep in FAILING_TRIALS:
    print(f"\n{'='*70}")
    print(f"{name}: GT HR = {gt_hr}, target = {target_ep}")
    print(f"{'='*70}")

    doc = fitz.open(pdf_path)
    full_text = ""
    for i in range(len(doc)):
        full_text += doc[i].get_text() + "\n"
    doc.close()

    # Normalize
    full_text = full_text.replace('\u037e', ';')
    full_text = full_text.replace('\u2013', '-')
    full_text = full_text.replace('\u2014', '-')
    full_text = re.sub(r'(\d)\u00b7(\d)', r'\1.\2', full_text)
    full_text = re.sub(r'([a-zA-Z])-\n([a-zA-Z])', r'\1\2', full_text)
    full_text = re.sub(r'(?<=[a-z,;%])\n(?=[a-zA-Z(0-9])', ' ', full_text)
    full_text = re.sub(r'(\d),(\d)', r'\1.\2', full_text)
    full_text = re.sub(r'(\d%)\s*([A-Za-z])', r'\1 \2', full_text)
    full_text = re.sub(r'([a-z])(\d+%)', r'\1 \2', full_text)
    # Fix OCR artifact: = rendered as 5
    full_text = re.sub(
        r'((?:HR|hazard\s+ratio)\s*)[=5]\s+(\d+\.\d)',
        r'\1\2', full_text, flags=re.IGNORECASE)

    # Review detection
    n_trial_refs = len(re.findall(
        r'\b[A-Z]{2,}[-\s]?\d*\s+(?:trial|study)\b', full_text, re.I))
    n_et_al = len(set(re.findall(
        r'(\w+)\s+et\s+al\.?', full_text, re.I)))
    is_review = n_trial_refs > 5 or n_et_al > 10
    print(f"  trial_refs={n_trial_refs}, et_al_unique={n_et_al}, "
          f"is_review={is_review}")
    print(f"  text length: {len(full_text)} chars")

    candidates = []
    for pat_idx, pattern in enumerate(PATTERNS):
        has_ci = pat_idx < len(PATTERNS) - 2
        for m in re.finditer(pattern, full_text, re.IGNORECASE):
            try:
                hr = float(m.group(1))
                if not (0.05 <= hr <= 20.0):
                    continue

                pre_ctx = full_text[max(0, m.start() - 120):m.start()]
                neg_match = _NEGATIVE.search(pre_ctx)
                if neg_match:
                    print(f"  SKIP neg [{hr:.3f}] pat={pat_idx} "
                          f"neg='{neg_match.group()}'")
                    continue

                near_post = full_text[m.end():min(len(full_text), m.end()+120)]
                if _CITED_STUDY.search(near_post):
                    print(f"  SKIP cited_post [{hr:.3f}] pat={pat_idx}")
                    continue

                # Post-context trial name filter
                if is_review and re.search(
                        r'[A-Z]{2,}[-\s]?\d*\s+(?:trial|study)',
                        near_post, re.I):
                    print(f"  SKIP trial_post [{hr:.3f}] pat={pat_idx}")
                    continue

                pre_cite = full_text[max(0, m.start()-200):m.start()]
                if (_CITED_TRIAL_PRE.search(pre_cite) and
                        re.search(r'\(\d{4}\)|\b20[012]\d\b', pre_cite)):
                    print(f"  SKIP cited_pre [{hr:.3f}] pat={pat_idx}")
                    continue

                priority = 0
                pos = m.start()
                in_abstract = pos < 3000
                if in_abstract:
                    priority += 10
                if has_ci:
                    priority += 5
                nearby = full_text[max(0, pos - 300):pos + 100]
                if _PRIMARY_KW.search(nearby):
                    priority += 3
                priority -= pos / len(full_text) * 2

                if is_review and not in_abstract:
                    if n_trial_refs > 20:
                        priority -= 25
                    else:
                        priority -= 15
                if target_ep:
                    ep_window = full_text[max(0, pos-150):
                                          min(len(full_text), m.end()+150)]
                    target_pat = _EP_MAP.get(target_ep, '')
                    if target_pat and re.search(target_pat, ep_window, re.I):
                        priority += 10
                    else:
                        for ep, ep_re in _EP_MAP.items():
                            if ep != target_ep and re.search(ep_re, ep_window,
                                                              re.I):
                                priority -= 8
                                break

                if hr > 3.0 or hr < 0.1:
                    priority -= 20

                ctx = full_text[max(0, m.start()-60):m.end()+60].strip()
                ctx = re.sub(r'\s+', ' ', ctx)[:120]
                candidates.append((hr, priority, pos, pat_idx, ctx))
            except (ValueError, IndexError):
                continue

    candidates.sort(key=lambda x: (-x[1], x[2]))

    print(f"\n  ALL CANDIDATES ({len(candidates)}):")
    for i, (hr, pri, pos, pat, ctx) in enumerate(candidates[:15]):
        marker = " <-- WINNER" if i == 0 else ""
        gt_match = " [CORRECT]" if abs(hr - gt_hr) / gt_hr < 0.05 else ""
        print(f"    [{i+1}] HR={hr:.3f} pri={pri:+.1f} pos={pos} "
              f"pat={pat}{gt_match}{marker}")
        print(f"        ctx: {ctx}")
