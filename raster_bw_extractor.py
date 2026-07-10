"""Black-&-white / grayscale KM extractor (real-paper frontier).

Many published KM figures distinguish arms by line STYLE (solid vs dashed) in black, not colour,
so the HSV colour tracer finds nothing. For two MONOTONE, NON-CROSSING survival curves the arms
can be recovered without solving solid-vs-dashed tracking: after removing gridlines/axes/text,
take the TOP and BOTTOM dark-pixel envelope per x-column (upper envelope = higher-survival arm,
lower = lower-survival arm). Axes are calibrated from OCR'd tick labels (normalised to ~1100px).

Returns the same {n_arms, arms:[{times,survivals,role,n}], ...} contract as the colour extractor,
so it plugs into build_result_from_extraction. Roles stay None (drug-name legends aren't matched);
HR MAGNITUDE is the recoverable quantity. Falls back to None when calibration/curves are too weak.
"""
from __future__ import annotations
import numpy as np

_READER = None


def _reader():
    global _READER
    if _READER is None:
        import easyocr
        _READER = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _READER


def _is_num(s):
    try:
        float(str(s).replace("%", "").replace(",", "").replace("O", "0").replace("o", "0").strip())
        return True
    except Exception:
        return False


def _num(s):
    return float(str(s).replace("%", "").replace(",", "").replace("O", "0").replace("o", "0").strip())


def _best_line(groups, want_positive):
    """Strict grouped linear fit (accurate on clean figures): within a row/column bucket, require
    a tight linear value~position relation with the correct slope sign."""
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
        if ss < 0.985:
            continue
        if (want_positive and m <= 0) or (not want_positive and m >= 0):
            continue
        score = ss * (vals.max() - vals.min())
        if best is None or score > best[0]:
            best = (score, m, b, vals, pos)
    return best


def _ransac_axis(nums, axis):
    """Robustly find an axis from OCR'd numeric tokens, tolerant of misreads (RANSAC).

    nums: list of (value, cx, cy). axis 'x' fits value~cx (positive slope, tokens share a row);
    axis 'y' fits value~cy (negative slope, tokens share a column). Real axis fonts are misread
    (40->20, 100->00), so a strict all-points linear fit fails; RANSAC keeps the dominant colinear,
    linearly-spaced tick set and rejects the outliers. Returns (m, b, inlier_vals, inlier_pos) or None.
    """
    pos_i = 1 if axis == "x" else 2
    cross_i = 2 if axis == "x" else 1
    pts = [(float(v), float(n[pos_i]), float(n[cross_i])) for v, n in [(n[0], n) for n in nums]]
    vals = [p[0] for p in pts]
    vr = (max(vals) - min(vals)) if vals else 0
    if len(pts) < 3 or vr <= 0:
        return None
    tol = max(0.05 * vr, 1e-6)
    cross_tol = 22.0
    best = None
    for i in range(len(pts)):
        for j in range(len(pts)):
            if i == j:
                continue
            v0, p0, c0 = pts[i]; v1, p1, c1 = pts[j]
            if abs(p1 - p0) < 3 or v0 == v1:
                continue
            m = (v1 - v0) / (p1 - p0); b = v0 - m * p0
            if (axis == "x" and m <= 0) or (axis == "y" and m >= 0):
                continue
            cref = (c0 + c1) / 2.0
            inl = [p for p in pts if abs(p[0] - (m * p[1] + b)) < tol and abs(p[2] - cref) < cross_tol]
            # dedupe near-identical positions (repeated OCR of same tick)
            if len(inl) >= 3 and (best is None or len(inl) > len(best[0])):
                best = (inl, m, b)
    if not best or len(best[0]) < 3:
        return None
    inl = best[0]
    vv = np.array([p[0] for p in inl]); pp = np.array([p[1] for p in inl])
    m, b = np.polyfit(pp, vv, 1)
    return m, b, vv, pp


def _ocr_tokens(img, png_path):
    import cv2
    H, W = img.shape[:2]
    s = min(max(1100.0 / W, 0.5), 4.0) if W > 0 else 1.0
    ocr_in = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC) if abs(s - 1) > 0.05 else str(png_path)
    res = _reader().readtext(ocr_in)
    toks = []
    for box, text, conf in res:
        xs = [p[0] / s for p in box]; ys = [p[1] / s for p in box]
        toks.append({"text": text, "cx": float(np.mean(xs)), "cy": float(np.mean(ys)),
                     "x0": min(xs), "x1": max(xs), "y0": min(ys), "y1": max(ys), "conf": conf})
    return toks


def extract_bw_km(png_path):
    import cv2
    img = cv2.imread(str(png_path))
    if img is None:
        return None
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    toks = _ocr_tokens(img, png_path)
    nums = [(_num(t["text"]), t["cx"], t["cy"]) for t in toks if _is_num(t["text"]) and t["conf"] > 0.25]
    if len(nums) < 4:
        return None
    # 1) Strict grouped fit first -- accurate on clean figures (synthetic + tidy real).
    xrows, ycols = {}, {}
    for v, cx, cy in nums:
        xrows.setdefault(round(cy / 12), []).append((v, cx))
        ycols.setdefault(round(cx / 12), []).append((v, cy))
    gx, gy = _best_line(xrows, True), _best_line(ycols, False)
    if gx and gy:
        _, xm, xb, xvals, xpos = gx
        _, ym, yb, yvals, ypos = gy
    else:
        # 2) Robust fallback for messy real figures (OCR misreads, NAR-table contamination):
        # RANSAC the y-axis, use it to locate the plot floor, then find x-ticks ONLY in the thin
        # band just below the floor (above the number-at-risk table).
        yf = _ransac_axis([n for n in nums if n[1] < W * 0.4] or nums, "y")
        if not yf:
            return None
        ym, yb, yvals, ypos = yf
        yscale_tmp = 100.0 if max(yvals) > 1.5 else 1.0
        plot_bottom = max((0.0 - yb) / ym, (yscale_tmp - yb) / ym)
        xband = [n for n in nums if plot_bottom - 25 < n[2] < plot_bottom + 90]
        xf = _ransac_axis(xband, "x") if len(xband) >= 3 else None
        if not xf:
            xf = _ransac_axis(nums, "x")
        if not xf:
            return None
        xm, xb, xvals, xpos = xf
    y_is_percent = bool(max(yvals) > 1.5)
    y_scale = 100.0 if y_is_percent else 1.0

    def px2t(px):
        return xm * px + xb

    def py2s(py):
        return (ym * py + yb) / y_scale

    # Plot box from CALIBRATION, not the tick-label span: the axes extend beyond the first/last
    # tick (e.g. ticks 10,20,30 but the plot starts at t=0 where S=1). Compute the pixel of t=0
    # (left edge), S=1 (top) and S=0 (bottom); the curves live between these.
    if abs(xm) < 1e-9 or abs(ym) < 1e-9:
        return None
    px_t0 = (0.0 - xb) / xm                       # pixel x at t=0
    py_s1 = (1.0 * y_scale - yb) / ym             # pixel y at S=1 (top)
    py_s0 = (0.0 - yb) / ym                       # pixel y at S=0 (bottom)
    x_left = int(max(0, min(px_t0, min(xpos))))
    y_top = int(max(0, min(py_s1, py_s0)))
    y_bot = int(min(H, max(py_s1, py_s0)))
    if (max(xpos) - x_left) < 40 or (y_bot - y_top) < 40:
        return None

    thr = max(90, int(np.percentile(gray, 25)))
    dark = gray < thr
    x0, x1 = x_left + 3, W                          # scan to the right edge; curve extent bounds it
    y0, y1 = y_top - 2, y_bot - 3                   # exclude the bottom axis line
    box = np.zeros_like(dark)
    box[max(0, y0):max(0, y1), max(0, x0):x1] = True
    dark = dark & box

    # Remove ONLY near-full-width horizontal lines (axis/gridlines) and near-full-height vertical
    # lines (y-axis). KM step functions are MOSTLY horizontal, so the kernel must be wide enough
    # that the flat curve segments survive -- kill lines spanning >=70% of the plot, not curve steps.
    kx = max(25, int((x1 - x0) * 0.7))
    ky = max(25, int((y1 - y0) * 0.7))
    horiz = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1)))
    vert = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (1, ky)))
    curves = dark & ~(horiz.astype(bool)) & ~(vert.astype(bool))

    # mask legend: text tokens inside the box PLUS the colored/black line-sample to their left
    for t in toks:
        if _is_num(t["text"]):
            continue
        tx0, tx1 = int(t["x0"]) - 55, int(t["x1"]) + 3   # extend left for the legend line sample
        ty0, ty1 = int(t["y0"]) - 3, int(t["y1"]) + 3
        if tx1 > x0 and tx0 < x1 and ty1 > y0 and ty0 < y1:
            curves[max(0, ty0):min(H, ty1), max(0, tx0):min(W, tx1)] = False

    # per-column top/bottom envelope -> two arms (upper = higher survival). Use the min/max dark
    # row per column; median-filter over columns to reject censoring-tick spikes.
    cols = list(range(x0, x1))
    top_px = np.full(len(cols), np.nan); bot_px = np.full(len(cols), np.nan)
    for i, c in enumerate(cols):
        rows = np.where(curves[:, c])[0]
        if rows.size:
            top_px[i] = rows.min(); bot_px[i] = rows.max()

    def _medfilt(a, k=7):
        out = a.copy()
        for i in range(len(a)):
            lo = max(0, i - k); hi = min(len(a), i + k + 1)
            w = a[lo:hi]; w = w[~np.isnan(w)]
            if w.size:
                out[i] = np.median(w)
        return out
    top_px = _medfilt(top_px); bot_px = _medfilt(bot_px)

    top_t, top_s, bot_t, bot_s = [], [], [], []
    for i, c in enumerate(cols):
        if not np.isnan(top_px[i]):
            top_t.append(px2t(c)); top_s.append(py2s(float(top_px[i])))
        if not np.isnan(bot_px[i]):
            bot_t.append(px2t(c)); bot_s.append(py2s(float(bot_px[i])))

    def _finish(ts, ss):
        if len(ts) < 10:
            return None
        t = np.array(ts, float); s = np.clip(np.array(ss, float), 0, 1)
        o = np.argsort(t); t, s = t[o], s[o]
        for i in range(1, len(s)):
            s[i] = min(s[i], s[i - 1])
        return {"times": t.tolist(), "survivals": s.tolist(), "role": None, "n": None}

    a_top = _finish(top_t, top_s)
    a_bot = _finish(bot_t, bot_s)
    arms = [a for a in (a_top, a_bot) if a is not None]
    if len(arms) < 2:
        return None
    # if the two envelopes are essentially identical (single curve / arms never separate), reject
    st = np.array(arms[0]["survivals"]); sb = np.array(arms[1]["survivals"])
    n = min(len(st), len(sb))
    if n >= 10 and np.mean(np.abs(st[:n] - sb[:n])) < 0.02:
        return None

    return {"n_arms": len(arms), "arms": arms,
            "x_range": [float(min(xvals)), float(max(xvals))],
            "y_is_percent": y_is_percent, "orientation_from_legend": False,
            "method": "bw_envelope"}
