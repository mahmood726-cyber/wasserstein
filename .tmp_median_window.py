import json
from pathlib import Path
r=json.loads(Path(r"C:/Users/user/Downloads/wasserstein/expanded_40_results/expanded_report.json").read_text(encoding="utf-8"))
rows=[]
for t in r['trials']:
    e=t.get('rel_error_pct')
    if e is not None:
        rows.append((float(e), t['name'], t.get('area'), t.get('hr_method'), t.get('status'), t.get('ext_hr'), t.get('final_hr'), t.get('gt_hr')))
rows.sort(key=lambda x:x[0])
n=len(rows)
mid=(n//2)
print('n_errors',n,'median_index0',mid,'median_entry',rows[mid])
lt5=sum(1 for x in rows if x[0] < 5)
le5=sum(1 for x in rows if x[0] <= 5)
print('count_<5',lt5,'count_<=5',le5)
print('entries_around_median:')
for i in range(max(0,mid-8), min(n,mid+9)):
    print(i, rows[i])
