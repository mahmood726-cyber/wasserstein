# sentinel:skip-file — hardcoded paths are fixture/registry/audit-narrative data for this repo's research workflow, not portable application configuration. Same pattern as push_all_repos.py and E156 workbook files.
"""Quick test of the 4 previously-failing trials."""
import sys
import io
import platform
import time
from pathlib import Path

if sys.platform == 'win32':
    def _safe_wmi_query(*a, **k):
        raise OSError("WMI bypass")
    platform._wmi_query = _safe_wmi_query
if 'pytest' not in sys.modules and hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                  errors='replace')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from km_pipeline import KMPipeline

TRIALS = [
    ("PMC10553121", r"C:\Users\user\oncology_rcts\PMC10553121.pdf",
     0.73, 0.62, 0.86, "OS"),
    ("PMC10990610", r"C:\Users\user\respiratory_rcts\PMC10990610.pdf",
     0.66, 0.53, 0.83, "PFS"),
    ("PMC11296275", r"C:\Users\user\neurology_rcts\PMC11296275.pdf",
     0.69, 0.57, 0.83, "PFS"),
    ("PMC10052578", r"C:\Users\user\infectious_rcts\PMC10052578.pdf",
     0.69, 0.56, 0.85, "PFS"),
]

def main():
    for name, pdf, gt_hr, gt_lo, gt_hi, endpoint in TRIALS:
        print(f"\n{'='*60}")
        print(f"{name}: GT={gt_hr} [{gt_lo}, {gt_hi}], endpoint={endpoint}")
        t0 = time.time()
        pipeline = KMPipeline(dpi=300, max_pages=12, n_per_arm=100)
        result = pipeline.extract(pdf, target_endpoint=endpoint)
        elapsed = time.time() - t0
        if result.hr is not None:
            within_ci = gt_lo <= result.hr <= gt_hi
            err = abs(result.hr - gt_hr) / gt_hr * 100
            print(f"  Extracted: HR={result.hr:.3f}")
            print(f"  Error: {err:.1f}%")
            print(f"  Within CI: {within_ci}")
            print(f"  Method: {result.hr_method}")
            print(f"  Status: {'PASS' if within_ci else 'FAIL'}")
        else:
            print(f"  FAILED: {result.warnings}")
        print(f"  Time: {elapsed:.0f}s")


if __name__ == '__main__':
    main()
