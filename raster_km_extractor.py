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
    (("control", "placebo", "standard", "usual care", "routine care", "comparator"), 0),
    (("experimental", "treatment", "intervention", "active"), 1),
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


def _role_from_label(label):
    if not label:
        return None
    ll = str(label).lower()
    for subs, role in ROLE_BY_LABEL:
        if any(x in ll for x in subs):
            return role
    return None


def _label_rows(labels, row_tol=10, max_gap=35):
    """Group OCR word tokens into same-row legend phrases."""
    rows = []
    for token in sorted(labels, key=lambda t: (t["cy"], t["x0"])):
        placed = False
        for row in rows:
            if abs(token["cy"] - row["cy"]) <= row_tol:
                row["tokens"].append(token)
                n = len(row["tokens"])
                row["cy"] = ((row["cy"] * (n - 1)) + token["cy"]) / n
                row["x0"] = min(row["x0"], token["x0"])
                placed = True
                break
        if not placed:
            rows.append({"cy": token["cy"], "x0": token["x0"], "tokens": [token]})

    grouped = []
    for row in rows:
        tokens = sorted(row["tokens"], key=lambda t: t["x0"])
        phrase = []
        current = []
        last_x = None
        for token in tokens:
            if current and last_x is not None and token["x0"] - last_x > max_gap:
                phrase.append(current)
                current = []
            current.append(token)
            # Prefer right edge if present; otherwise center is still stable.
            last_x = float(token.get("x1", token["cx"]))
        if current:
            phrase.append(current)
        for chunk in phrase:
            grouped.append({
                "text": " ".join(str(t["text"]) for t in chunk),
                "x0": min(t["x0"] for t in chunk),
                "cy": sum(t["cy"] for t in chunk) / len(chunk),
            })
    return grouped


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
    # Normalize the figure to a target width before OCR (~1100 px): real figures come at wildly
    # varying resolutions (embedded 500px thumbnails or 2200px scans); easyocr misses small axis
    # digits on both extremes. OCR on the resized image, then scale bboxes back to ORIGINAL pixel
    # coords so they line up with the HSV curve tracer (which runs on the original image).
    ocr_scale = 1.0
    ocr_img = str(png_path)
    if W > 0:
        s = 1100.0 / W
        s = min(max(s, 0.5), 4.0)
        if abs(s - 1.0) > 0.05:
            ocr_scale = s
            ocr_img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    try:
        results = _reader().readtext(ocr_img)
    except Exception:
        return None
    # results: [ [box(4 pts), text, conf], ... ]
    toks = []
    for box, text, conf in results:
        xs = [p[0] / ocr_scale for p in box]; ys = [p[1] / ocr_scale for p in box]
        toks.append({"text": text, "cx": float(np.mean(xs)), "cy": float(np.mean(ys)),
                     "x0": min(xs), "x1": max(xs), "y0": min(ys), "conf": conf})
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
    if xf and yf:
        _, xm, xb, xvals = xf
        _, ym, yb, yvals = yf
        y_is_percent = bool(max(yvals) > 1.5)
    else:
        # 4th raster track (hybrid): escalate to a registered vision model when local OCR
        # calibration fails on a degraded scan. No-op if no vision backend is registered.
        from vision_fallback import vision_read
        vr = vision_read(png_path)
        if not vr:
            return None
        xt = np.array(vr["x_ticks"], float)
        yt = np.array(vr["y_ticks"], float)
        if len(xt) < 2 or len(yt) < 2:
            return None
        xm, xb = np.polyfit(xt[:, 1], xt[:, 0], 1)
        ym, yb = np.polyfit(yt[:, 1], yt[:, 0], 1)
        xvals, yvals = xt[:, 0], yt[:, 0]
        y_is_percent = bool(vr.get("y_is_percent", max(yvals) > 1.5))
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
    for lab in _label_rows(labels):
        role = _role_from_label(lab["text"])
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

    # ---- N per arm from the number-at-risk table (roadmap L6) -----------------------
    for a in arms:
        a["n"] = None
    risk_y = None
    for t in toks:
        if "risk" in str(t["text"]).lower():
            risk_y = t["cy"]
            break
    if risk_y is not None:
        # L16: re-OCR the number-at-risk band on an UPSCALED crop with a digit allowlist —
        # the small NAR font is misread at 150 dpi in the whole-image pass. Fall back to the
        # first-pass tokens if the focused pass yields too little.
        y0 = max(0, int(risk_y) - 4)
        crop = img[y0:, :]
        nar = []
        if crop.size and crop.shape[0] > 8:
            up = cv2.resize(crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
            try:
                nres = _reader().readtext(up, allowlist="0123456789")
                for box, text, conf in nres:
                    if _is_num(text) and conf > 0.3:
                        cx = float(np.mean([p[0] for p in box])) / 3.0
                        cy = y0 + float(np.mean([p[1] for p in box])) / 3.0
                        nar.append((_num(text), cx, cy))
            except Exception:
                nar = []
        if len(nar) < 2 * max(1, len(arms)):     # focused pass too sparse -> first-pass tokens
            nar = [(_num(t["text"]), t["cx"], t["cy"]) for t in toks
                   if _is_num(t["text"]) and t["cy"] > risk_y - 5]
        rows_nar = {}
        for v, cx, cy in nar:
            rows_nar.setdefault(round(cy / 12), []).append((v, cx, cy))
        row_items = sorted(rows_nar.values(), key=lambda it: np.mean([y for _, _, y in it]))
        for ri, items in enumerate(row_items):
            if len(items) < 2:
                continue
            n_est = int(max(v for v, _, _ in items))
            ry = np.mean([y for _, _, y in items])
            role = None
            for t in toks:
                if _is_num(t["text"]):
                    continue
                if abs(t["cy"] - ry) < 10:
                    role = _role_from_label(t["text"])
                if role is not None:
                    break
            target = None
            if role is not None:
                target = next((a for a in arms if a["role"] == role), None)
            if target is None and ri < len(arms):
                target = arms[ri]
            if target is not None and target.get("n") is None:
                target["n"] = n_est

    arms.sort(key=lambda a: (a["role"] if a["role"] is not None else 9))
    return {
        "n_arms": len(arms), "arms": arms,
        "x_range": [float(min(xvals)), float(max(xvals))],
        "y_is_percent": y_is_percent,
        "orientation_from_legend": all(a["role"] is not None for a in arms) and len(arms) >= 2,
    }
