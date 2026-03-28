import json

d = json.load(open('expanded_40_results/expanded_report.json', encoding='utf-8'))
for r in d['trials']:
    if r.get('within_ci'):
        continue
    ext = r.get('final_hr') if r.get('final_hr') is not None else r.get('ext_hr')
    lo, hi = r.get('gt_ci', [None, None])
    rec = None
    rec_ok = False
    if ext and ext > 0:
        rec = 1.0 / ext
        rec_ok = lo is not None and hi is not None and lo <= rec <= hi
    print(f"{r['name']}|method={r.get('hr_method')}|gt_ci=[{lo},{hi}]|ext={ext}|rec={rec}|rec_ok={rec_ok}|err={r.get('rel_error_pct')}")
