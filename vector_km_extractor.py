"""Vector-path KM extractor (roadmap L8) — exact digitization of born-digital PDF figures.

For a vector (born-digital) KM figure, the step-polyline vertices, axis tick labels, and
legend are all exact objects in the PDF content stream. Reading them via PyMuPDF
(get_drawings + get_text) gives near-zero digitization error and — because it calibrates
from the PRINTED tick labels — inherently fixes the axis-truncation bug and reads
orientation from the legend (never from a reported HR). Falls back to None when the figure
is raster (scanned), so the caller uses the pixel path.

Returns per-arm survival curves + a legend->role orientation map. Feed to faithful_guyot.
"""
from __future__ import annotations
import numpy as np

# arm role by legend label (lowercased substring match)
ROLE_BY_LABEL = [
    (("control", "placebo", "standard", "comparator"), 0),
    (("experimental", "treatment", "intervention"), 1),
    (("arm c", "third"), 2),
]


def _is_num(s):
    try:
        float(str(s).replace("%", "").replace(",", "").strip())
        return True
    except Exception:
        return False


def _num(s):
    return float(str(s).replace("%", "").replace(",", "").strip())


def _path_points(d):
    """Flatten a get_drawings path's items into an ordered list of (x, y) vertices."""
    pts = []
    for it in d["items"]:
        op = it[0]
        for p in it[1:]:
            if hasattr(p, "x"):
                pts.append((float(p.x), float(p.y)))
    return pts


def _color_key(c):
    if not c:
        return None
    return tuple(round(float(x), 2) for x in c)


def _saturation(c):
    if not c:
        return 0.0
    r, g, b = c[:3]
    mx, mn = max(r, g, b), min(r, g, b)
    return mx - mn                      # chroma; grey/black ~ 0


def extract_vector_km(pdf_path, page_index=0):
    """Extract KM curves from a vector PDF. Returns dict or None if not vector/failed."""
    import fitz
    doc = fitz.open(str(pdf_path))
    try:
        pg = doc[page_index]
        drawings = pg.get_drawings()
        if not drawings:
            return None
        words = pg.get_text("words")    # (x0,y0,x1,y1,word,block,line,wno)

        # ---- axis calibration from printed tick labels -------------------------------
        # Robust to the number-at-risk table (a grid of numbers below the plot) by finding
        # the ROW (x-axis) / COLUMN (y-axis) of numbers whose value is near-perfectly LINEAR
        # in pixel position with the correct slope sign. NAR counts do not satisfy this
        # jointly with the x/y-axis geometry, so they are excluded.
        num_words = [(_num(w[4]), (w[0] + w[2]) / 2, (w[1] + w[3]) / 2) for w in words if _is_num(w[4])]
        if len(num_words) < 4:
            return None

        def _best_line(groups, pos_index, want_positive):
            """groups: dict key->list of (value, pos_for_fit, other_pos). Return (m,b,vals,other_mean)."""
            best = None
            for items in groups.values():
                if len(items) < 3:
                    continue
                vals = np.array([v for v, p, o in items], float)
                pos = np.array([p for v, p, o in items], float)
                if len(np.unique(vals)) < 3 or vals.min() < -1e-6:
                    continue
                m, b = np.polyfit(pos, vals, 1)
                pred = m * pos + b
                ss = 1 - np.sum((vals - pred) ** 2) / max(np.sum((vals - vals.mean()) ** 2), 1e-9)
                if ss < 0.985:
                    continue
                if want_positive and m <= 0:      # x-axis: value increases left->right
                    continue
                if not want_positive and m >= 0:  # y-axis (y-down): value increases upward => neg slope
                    continue
                score = ss * (vals.max() - vals.min())        # prefer the widest true axis
                if best is None or score > best[0]:
                    best = (score, m, b, vals, np.mean([o for v, p, o in items]))
            return best

        # x-axis: group by row (rounded y), fit value ~ x
        xrows = {}
        for v, wx, wy in num_words:
            xrows.setdefault(round(wy / 5), []).append((v, wx, wy))
        # y-axis: group by column (rounded x), fit value ~ y
        ycols = {}
        for v, wx, wy in num_words:
            ycols.setdefault(round(wx / 5), []).append((v, wy, wx))
        xfit = _best_line(xrows, 1, want_positive=True)
        yfit = _best_line(ycols, 1, want_positive=False)
        if xfit is None or yfit is None:
            return None
        _, xm, xb, xvals, _ = xfit
        _, ym, yb, yvals, _ = yfit
        y_is_percent = bool(max(yvals) > 1.5)
        y_scale = 100.0 if y_is_percent else 1.0
        y_truncated = bool((min(yvals) / y_scale) > 0.05)     # bottom tick well above 0 => truncated axis

        def px2t(px):
            return xm * px + xb

        def py2s(py):
            return (ym * py + yb) / y_scale

        # ---- curve polylines by saturated color -------------------------------------
        colored = {}
        for d in drawings:
            c = d.get("color")
            if _saturation(c) < 0.15:                 # skip black axes / grey gridlines
                continue
            pts = _path_points(d)
            if len(pts) < 3:
                continue
            key = _color_key(c)
            span = max(p[0] for p in pts) - min(p[0] for p in pts)
            prev = colored.get(key)
            if prev is None or span > prev["span"]:
                colored[key] = {"pts": pts, "span": span, "color": c}

        if not colored:
            return None

        # ---- legend: short colored swatch + nearest label word ----------------------
        # legend swatches are short saturated paths; map their color to the nearest word.
        label_by_color = {}
        for d in drawings:
            c = d.get("color")
            if _saturation(c) < 0.15:
                continue
            pts = _path_points(d)
            if not pts:
                continue
            span = max(p[0] for p in pts) - min(p[0] for p in pts)
            if span > 40 or len(pts) > 6:             # not a swatch (that's the curve)
                continue
            sx = np.mean([p[0] for p in pts]); sy = np.mean([p[1] for p in pts])
            best, bestd = None, 1e9
            for w in words:
                if _is_num(w[4]):
                    continue
                wx, wy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
                dist = abs(wy - sy) + max(0, w[0] - sx) * 0.2   # prefer word to the right, same row
                if wy > sy - 8 and wy < sy + 8 and dist < bestd:
                    bestd, best = dist, w[4]
            if best:
                label_by_color[_color_key(c)] = best

        # ---- assemble per-arm curves ------------------------------------------------
        arms = []
        for key, info in colored.items():
            pts = sorted(info["pts"], key=lambda p: p[0])
            t = np.array([px2t(px) for px, py in pts])
            s = np.array([py2s(py) for px, py in pts])
            # clip + enforce non-increasing survival (PAVA-lite: cumulative min)
            s = np.clip(s, 0, 1)
            order = np.argsort(t)
            t, s = t[order], s[order]
            for i in range(1, len(s)):
                s[i] = min(s[i], s[i - 1])
            label = label_by_color.get(key)
            role = None
            if label:
                ll = label.lower()
                for subs, r in ROLE_BY_LABEL:
                    if any(x in ll for x in subs):
                        role = r
                        break
            arms.append({"color": key, "label": label, "role": role,
                         "times": t.tolist(), "survivals": s.tolist()})

        # order arms by resolved role (Control=0, Experimental=1, ...) then by curve position
        arms.sort(key=lambda a: (a["role"] if a["role"] is not None else 9))
        return {
            "n_arms": len(arms), "arms": arms,
            "x_range": [float(min(xvals)), float(max(xvals))],
            "y_is_percent": y_is_percent, "y_truncated": bool(y_truncated),
            "orientation_from_legend": all(a["role"] is not None for a in arms),
        }
    finally:
        doc.close()
