"""Raster KM extractor (roadmap: raster/scanned path) — clean-room, mirrors the vector path.

For a scanned/image KM figure (no PDF text/vector layer), recover:
  - axis calibration + tick VALUES via easyocr (text + pixel bboxes -> per-row/col linearity fit,
    robust to the number-at-risk table exactly like the vector extractor), and
  - per-arm survival curves via a column-wise sub-pixel HSV tracer (median masked row per x-column),
  - orientation from the legend (label text near a colored swatch -> curve color match).

This replaces the tangled legacy raster path for the images it can read. Returns None (fall back)
when OCR/curve detection is too weak. easyocr is heavy (torch); the reader is cached module-wide.
"""
from __future__ import annotations
import numpy as np

# arm colors used by the generator / common in papers (BGR). role by legend label handled separately.
_ARM_HUES = {  # name -> (H center in OpenCV 0-179, tolerance)
    "blue": (110, 20), "red": (0, 12), "green": (60, 25),
}
ROLE_BY_LABEL = [
    (("control", "placebo", "standard", "comparator"), 0),
    (("experimental", "treatment", "intervention"), 1),
    (("arm c", "third"), 2),
]

_READER = None


def _reader():
    global _READER
    if _READER is None:
        import easyocr
        _READER = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _READER


def _is_num(s):
    try:
        float(str(s).replace("%", "").replace(",", "").replace("O", "0").strip())
        return True
    except Exception:
        return False


def _num(s):
    return float(str(s).replace("%", "").replace(",", "").replace("O", "0").strip())


def _best_line(groups, want_positive):
    best = None
    for items in groups.values():
        if len(items) < 3:
            continue
        vals = np.array([v for v, p in items], float)
        pos = np.array([p for v, p in items], float)
        if len(np.unique(vals)) < 3 or vals.min() < -1e-6:
            continue
        m, b = np.polyfit(pos, vals, 1)
        ss = 1 - np.sum((vals - (m * pos + b)) ** 2) / max(np.sum((vals - vals.mean()) ** 2), 1e-9)
        if ss < 0.98:
            continue
        if want_positive and m <= 0:
            continue
        if not want_positive and m >= 0:
            continue
        score = ss * (vals.max() - vals.min())
        if best is None or score > best[0]:
            best = (score, m, b, vals)
    return best


def extract_raster_km(png_path):
    """Extract KM curves from a raster image. Returns dict or None."""
    import cv2
    img = cv2.imread(str(png_path))
    if img is None:
        return None
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ---- OCR text + boxes -----------------------------------------------------------
    try:
        results = _reader().readtext(str(png_path))
    except Exception:
        return None
    # results: [ [box(4 pts), text, conf], ... ]
    toks = []
    for box, text, conf in results:
        xs = [p[0] for p in box]; ys = [p[1] for p in box]
        toks.append({"text": text, "cx": float(np.mean(xs)), "cy": float(np.mean(ys)),
                     "x0": min(xs), "y0": min(ys), "conf": conf})
    nums = [(_num(t["text"]), t["cx"], t["cy"]) for t in toks if _is_num(t["text"])]
    if len(nums) < 4:
        return None
    # x-axis: rows (group by cy), fit value~cx, positive slope
    xrows = {}
    for v, cx, cy in nums:
        xrows.setdefault(round(cy / 10), []).append((v, cx))
    ycols = {}
    for v, cx, cy in nums:
        ycols.setdefault(round(cx / 10), []).append((v, cy))
    xf = _best_line(xrows, want_positive=True)
    yf = _best_line(ycols, want_positive=False)
    if not xf or not yf:
        return None
    _, xm, xb, xvals = xf
    _, ym, yb, yvals = yf
    y_is_percent = bool(max(yvals) > 1.5)
    y_scale = 100.0 if y_is_percent else 1.0

    def px2t(px):
        return xm * px + xb

    def py2s(py):
        return (ym * py + yb) / y_scale

    # ---- plot region (where curves live): between axes, above x-tick row ------------
    # rough box: x in [min curve-colored col .. max], y above the x-axis tick row
    # detect saturated colored pixels
    sat = hsv[:, :, 1] > 60
    val = hsv[:, :, 2] > 40

    # ---- per-arm column-wise sub-pixel tracer --------------------------------------
    arms = []
    for name, (hc, tol) in _ARM_HUES.items():
        hue = hsv[:, :, 0].astype(int)
        if name == "red":
            hmask = (hue <= tol) | (hue >= 179 - tol)
        else:
            hmask = np.abs(hue - hc) <= tol
        mask = hmask & sat & val
        if mask.sum() < 200:               # not enough of this color -> arm absent
            continue
        cols = np.where(mask.any(axis=0))[0]
        if cols.size < 20:
            continue
        cx0, cx1 = cols.min(), cols.max()
        ts, ss = [], []
        for c in range(cx0, cx1 + 1):
            rows = np.where(mask[:, c])[0]
            if rows.size == 0:
                continue
            ts.append(px2t(c))
            ss.append(py2s(float(np.median(rows))))   # median row = sub-pixel curve position
        if len(ts) < 10:
            continue
        t = np.array(ts); s = np.clip(np.array(ss), 0, 1)
        order = np.argsort(t); t, s = t[order], s[order]
        for i in range(1, len(s)):
            s[i] = min(s[i], s[i - 1])
        arms.append({"hue": name, "cx0": int(cx0), "cx1": int(cx1),
                     "times": t.tolist(), "survivals": s.tolist(),
                     "swatch_y": None})

    if not arms:
        return None

    # ---- legend: label text -> role, matched to arm by swatch color near the label --
    # find each arm's legend swatch: a short run of the arm color near a label token.
    for a in arms:
        a["role"] = None
    labels = [t for t in toks if not _is_num(t["text"]) and len(t["text"]) >= 3]
    for lab in labels:
        ll = lab["text"].lower()
        role = None
        for subs, r in ROLE_BY_LABEL:
            if any(x in ll for x in subs):
                role = r
                break
        if role is None:
            continue
        # swatch is just LEFT of the label at the same y; sample hue there
        ly = int(lab["cy"]); lx0 = int(lab["x0"])
        x_lo = max(0, lx0 - 40)
        patch = hsv[max(0, ly - 6):ly + 6, x_lo:lx0]
        if patch.size == 0:
            continue
        pm = (patch[:, :, 1] > 60) & (patch[:, :, 2] > 40)
        if pm.sum() < 3:
            continue
        hh = int(np.median(patch[:, :, 0][pm]))
        # match to arm hue
        best = None
        for a in arms:
            hc, tol = _ARM_HUES[a["hue"]]
            d = min(abs(hh - hc), 179 - abs(hh - hc)) if a["hue"] == "red" else abs(hh - hc)
            if best is None or d < best[1]:
                best = (a, d)
        if best and best[1] <= 25:
            best[0]["role"] = role

    arms.sort(key=lambda a: (a["role"] if a["role"] is not None else 9))
    return {
        "n_arms": len(arms), "arms": arms,
        "x_range": [float(min(xvals)), float(max(xvals))],
        "y_is_percent": y_is_percent,
        "orientation_from_legend": all(a["role"] is not None for a in arms) and len(arms) >= 2,
    }
