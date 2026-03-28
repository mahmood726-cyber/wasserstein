import json
from pathlib import Path
stems=['40337145_Effects of Intensive Systolic Blood Pres','PMC10062393','PMC12236988','PMC10672715','PMC10872694','PMC11296275']
for s in stems:
    p=Path('expanded_40_results')/f'{s}_summary.json'
    if not p.exists():
        c=list(Path('expanded_40_results').glob(f"{s}*_summary.json"))
        p=c[0] if c else None
    if not p:
        print('missing',s); continue
    j=json.load(open(p,encoding='utf-8'))
    print('\n---',s,'file',p.name)
    print('hr',j.get('hr'),'ci',j.get('ci_lower'),j.get('ci_upper'),'method',j.get('hr_method'),'conf',j.get('confidence'))
    print('text_hr',j.get('text_hr'))
    print('text_ctx', (j.get('text_hr_context') or '')[:280].replace('\n',' '))
    print('pair',j.get('pair_description'),j.get('pair_rank'),j.get('pair_quality_score'),j.get('pair_orientation'))
    print('warn', '; '.join((j.get('warnings') or [])[:3]))
