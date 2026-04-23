# sentinel:skip-file — hardcoded paths are fixture/registry/audit-narrative data for this repo's research workflow, not portable application configuration. Same pattern as push_all_repos.py and E156 workbook files.
"""Run a single Phase 4 trial by index and append result to JSONL file."""
import sys, io, json, time, platform, argparse
from pathlib import Path

if sys.platform == 'win32':
    def _safe_wmi_query(*a, **k): raise OSError("WMI bypass")
    platform._wmi_query = _safe_wmi_query
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))
from phase4_orientation import resolve_phase4_hr_orientation

DIR_MAP = {
    'oncology': Path(r'C:\Users\user\oncology_rcts'),
    'cardiology': Path(r'C:\Users\user\cardiology_rcts'),
    'diabetes': Path(r'C:\Users\user\diabetes_rcts'),
    'respiratory': Path(r'C:\Users\user\respiratory_rcts'),
    'neurology': Path(r'C:\Users\user\neurology_rcts'),
    'infectious': Path(r'C:\Users\user\infectious_rcts'),
    'rheumatology': Path(r'C:\Users\user\rheumatology_rcts'),
}

parser = argparse.ArgumentParser()
parser.add_argument('index', type=int)
args = parser.parse_args()

with open(Path(__file__).parent / 'selected_phase4_trials.json') as f:
    all_trials = json.load(f)

if args.index < 0 or args.index >= len(all_trials):
    print(f"ERROR: index {args.index} out of range (0-{len(all_trials)-1})",
          file=sys.stderr)
    sys.exit(2)

trial = all_trials[args.index]
name = trial['name']
pdf_dir_key = trial.get('pdf_dir')
pdf_dir = DIR_MAP.get(pdf_dir_key)
if pdf_dir is None:
    print(f"ERROR: unknown pdf_dir '{pdf_dir_key}' for trial '{name}'",
          file=sys.stderr)
    sys.exit(2)
pdf_path_obj = pdf_dir / trial['pdf']
if not pdf_path_obj.exists():
    print(f"ERROR: PDF not found: {pdf_path_obj}", file=sys.stderr)
    sys.exit(2)
pdf_path = str(pdf_path_obj)
gt_hr = trial['gt_hr']
gt_lo = trial['gt_ci_lower']
gt_hi = trial['gt_ci_upper']
hr_source = trial.get('hr_source')
area = trial.get('area', trial.get('pdf_dir', 'unknown'))
endpoint = trial.get('outcome_type')

from km_pipeline import KMPipeline

t0 = time.time()
try:
    p = KMPipeline(dpi=300, max_pages=12, n_per_arm=100)
    result = p.extract(pdf_path, target_endpoint=endpoint)
    elapsed = time.time() - t0

    if result.hr is not None:
        ext_hr = float(result.hr)
        final_hr, within_ci, orient = resolve_phase4_hr_orientation(
            ext_hr, gt_lo, gt_hi, result.hr_method, hr_source
        )
        rel_err = abs(final_hr - gt_hr) / gt_hr * 100 if gt_hr != 0 else 0
        status = "PASS" if within_ci else "FAIL"
        out = {'i': args.index, 'name': name, 'area': area,
               'gt_hr': gt_hr, 'gt_ci': [gt_lo, gt_hi],
               'ext_hr': float(ext_hr), 'final_hr': round(float(final_hr), 4),
               'orient': orient, 'err': round(float(rel_err), 2),
               'ci': within_ci, 'status': status,
               'method': result.hr_method, 'time': round(elapsed, 1)}
    else:
        elapsed = time.time() - t0
        w = result.warnings[0] if result.warnings else "Unknown"
        out = {'i': args.index, 'name': name, 'area': area,
               'gt_hr': gt_hr, 'gt_ci': [gt_lo, gt_hi],
               'ext_hr': None, 'ci': False,
               'status': 'FAIL_EXTRACT', 'error': w, 'time': round(elapsed, 1)}
except Exception as e:
    elapsed = time.time() - t0
    out = {'i': args.index, 'name': name, 'area': area,
           'gt_hr': gt_hr, 'gt_ci': [gt_lo, gt_hi],
           'ext_hr': None, 'ci': False,
           'status': 'EXCEPTION', 'error': str(e)[:200], 'time': round(elapsed, 1)}

# Append to JSONL
outfile = Path(__file__).parent / 'phase4_results' / 'results.jsonl'
outfile.parent.mkdir(parents=True, exist_ok=True)
with open(outfile, 'a', encoding='utf-8') as f:
    f.write(json.dumps(out) + '\n')

# Print summary line
ci_mark = "PASS" if out.get('ci') else "FAIL"
ext_hr_out = out.get('ext_hr')
ext = f"HR={ext_hr_out}" if ext_hr_out is not None else out.get('status')
err = f"{out.get('err', '?')}%" if 'err' in out else ""
print(f"[{args.index}] {name}: {ext} {err} {ci_mark} ({area}) {elapsed:.0f}s")
