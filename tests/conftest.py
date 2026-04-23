import sys
import importlib.util
from pathlib import Path

if sys.platform == "win32":
    def _safe_wmi_query(*_args, **_kwargs):
        raise OSError("WMI bypassed")
    import platform
    platform._wmi_query = _safe_wmi_query


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


if importlib.util.find_spec("cv2") is None:
    collect_ignore = [
        "test_km_pipeline_time_to_event_regressions.py",
        "test_km_pipeline_verification_regressions.py",
        "test_pdf_extractor_regressions.py",
        "test_text_hr_selector_regressions.py",
        "test_verification_regressions.py",
    ]
