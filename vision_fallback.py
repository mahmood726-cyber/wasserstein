"""Vision-LLM fallback for the raster path (the 4th raster track: hybrid OCR + vision).

When local OCR calibration fails (degraded scan, unusual font, rotated axis), the raster
extractor can escalate to a vision model — the KM-GPT approach. This module is the pluggable
seam: register a vision reader (a function that sends the figure image to a vision model and
returns structured axis/legend info) and the raster extractor will call it as a fallback.

The vision model call itself is intentionally NOT hard-wired (it needs the caller's API key /
agent). Register one with register_vision_reader(fn). Contract for fn(png_path) -> dict or None:

    {
      "x_ticks":  [[value, pixel_x], ...],   # >=2 numeric x-axis ticks + their pixel x
      "y_ticks":  [[value, pixel_y], ...],   # >=2 numeric y-axis ticks + their pixel y
      "y_is_percent": true|false,
      "legend":   [{"label": "Control", "hue": "blue"}, {"label": "Experimental", "hue": "red"}]
    }

A reference prompt for the vision model is in REFERENCE_PROMPT below; validated in-session that
a Claude vision read returns accurate axis ticks + legend on these figures.
"""
from __future__ import annotations

_VISION_READER = None

REFERENCE_PROMPT = (
    "This is a Kaplan-Meier survival plot. Read ONLY what is printed. Return JSON with: "
    "x_ticks (each [value, pixel_x] for the numeric x-axis tick labels), y_ticks (each "
    "[value, pixel_y]), y_is_percent (true if the y-axis is 0-100%), and legend (each "
    "{label, hue} mapping a curve's printed label to its line color). Do not guess."
)


def register_vision_reader(fn):
    """Register a callable fn(png_path)->dict (see module docstring) as the vision backend."""
    global _VISION_READER
    _VISION_READER = fn


def has_vision():
    return _VISION_READER is not None


def vision_read(png_path):
    """Call the registered vision reader; None if unregistered or it fails."""
    if _VISION_READER is None:
        return None
    try:
        out = _VISION_READER(str(png_path))
        if out and out.get("x_ticks") and out.get("y_ticks"):
            return out
    except Exception:
        pass
    return None
