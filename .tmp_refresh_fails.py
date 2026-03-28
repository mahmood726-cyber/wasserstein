import json, subprocess, sys, time
import expanded_gold_trials as e

rep = json.load(open('expanded_40_results/expanded_report.json', encoding='utf-8'))
trials = e.ALL_TRIALS + e.load_phase4_trials()
by_name = {t['name']: t for t in trials}

fails = [r for r in rep['trials'] if not r.get('within_ci')]
print('n_fails', len(fails))
for i, r in enumerate(fails, 1):
    t = by_name[r['name']]
    stem = e._safe_stem(t['pdf'])
    t0 = time.time()
    p = subprocess.run([sys.executable, 'recover_single_trial.py', '--stem', stem, '--force'], capture_output=True, text=True)
    dt = time.time() - t0
    out = (p.stdout or '').strip().replace('\n', ' | ')
    err = (p.stderr or '').strip().replace('\n', ' | ')
    print(f"[{i}/{len(fails)}] {r['name']} stem={stem} rc={p.returncode} sec={dt:.1f}")
    if out:
        print('  out:', out[:500])
    if err:
        print('  err:', err[:500])
