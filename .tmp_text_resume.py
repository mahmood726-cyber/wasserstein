import json, subprocess, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import expanded_gold_trials as e

report=json.loads(Path('expanded_40_results/expanded_report.json').read_text(encoding='utf-8'))
trials={t['name']:t for t in (e.ALL_TRIALS+e.load_phase4_trials())}
stems=[]
for t in report['trials']:
    tr=trials[t['name']]
    stem=e._safe_stem(tr['pdf'])
    sp=Path('expanded_40_results')/f'{stem}_summary.json'
    if not sp.exists():
        stems.append(stem)
        continue
    s=json.loads(sp.read_text(encoding='utf-8'))
    if s.get('hr_method')=='text_derived_only':
        stems.append(stem)
stems=sorted(set(stems))
print('remaining_text_only_stems',len(stems), flush=True)

logp=Path('recover_logs')/'text_only_batch_recover_resume.log'
logp.parent.mkdir(parents=True, exist_ok=True)
start=time.time()


def run_one(stem):
    t0=time.time()
    try:
        p=subprocess.run([sys.executable,'recover_single_trial.py','--stem',stem,'--force'],cwd=str(Path('.').resolve()),capture_output=True,text=True)
        out=(p.stdout or '').strip().replace('\n',' | ')
        err=(p.stderr or '').strip().replace('\n',' | ')
        dt=time.time()-t0
        return stem,p.returncode,dt,out,err
    except Exception as ex:
        dt=time.time()-t0
        return stem,-999,dt,'',f'exception:{ex}'

with logp.open('w',encoding='utf-8') as log:
    log.write(f'start {time.time()} n={len(stems)}\\n')
    log.flush()
    done=0
    ok=0
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs=[ex.submit(run_one,s) for s in stems]
        for fut in as_completed(futs):
            stem,rc,dt,out,err=fut.result()
            done+=1
            if rc==0:
                ok+=1
            line=f'[{done}/{len(stems)}] stem={stem} rc={rc} sec={dt:.1f} out={out} err={err}\\n'
            print(line.strip(), flush=True)
            log.write(line)
            log.flush()
    log.write(f'complete ok={ok}/{len(stems)} total_sec={time.time()-start:.1f}\\n')
    log.flush()
print('done',ok,len(stems),'elapsed',round(time.time()-start,1), flush=True)
