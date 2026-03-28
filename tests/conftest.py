import sys
from pathlib import Path

if sys.platform == "win32":
    def _safe_wmi_query(*_args, **_kwargs):
        raise OSError("WMI bypassed")
    import platform
    platform._wmi_query = _safe_wmi_query


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
