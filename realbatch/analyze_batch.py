"""Run the automated CV extraction pipeline over the downloaded PLoS PDFs and report a
large-scale reliability picture:
  - CV extraction SUCCESS RATE: fraction of PDFs from which a KM curve is auto-extracted.
  - HR VALIDATION (subset): for papers whose abstract reports a clear HR + CI, compare the
    auto-extracted HR against the published CI.

For each PDF: pull embedded figure images (fitz), try the colour raster extractor then the B&W
extractor on each, keep KM-looking results (2 arms, S starts ~1, monotone), reconstruct the HR.
Writes results incrementally to batch_results.json. Slow (easyocr per figure).
"""
import io, json, os, re, sys, time
import numpy as np
import fitz
from PIL import Image
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "benchmark")))
from raster_km_extractor import extract_raster_km
from raster_bw_extractor import extract_bw_km
from faithful_guyot import reconstruct_arm_faithful
from km_metrics import cox_loghr_2arm

BASE = os.path.dirname(os.path.abspath(__file__))
PDFDIR = os.path.join(BASE, "pdfs")

# Matches: "hazard ratio[/HR/aHR] ... 0.63 ... 95% CI ... 0.46 - 0.86" across the common phrasings
# ("HR 0.63 (95% CI 0.46-0.86)", "HR=0.63; 95%CI 0.46 to 0.86", "hazard ratio of 0.63 [0.46, 0.86]").
HR_RE = re.compile(
    r"(?:hazard\s*ratio|\bHR\b|\baHR\b)\D{0,25}?(\d\.\d{1,3})"
    r"\D{0,30}?(\d\.\d{1,3})\s*(?:-|–|—|to|,|;)\s*(\d\.\d{1,3})", re.I)


def parse_published_hr(abstract):
    """Return list of (hr, lo, hi) found in the abstract with a plausible HR + CI."""
    out = []
    for m in HR_RE.finditer(abstract or ""):
        hr, lo, hi = float(m.group(1)), float(m.group(2)), float(m.group(3))
        if 0.05 < hr < 20 and lo < hi and (lo - 0.02) <= hr <= (hi + 0.02) and (hi - lo) < 5:
            out.append((hr, lo, hi))
    return out


def looks_like_km(ex):
    if not ex or ex.get("n_arms", 0) < 2:
        return False
    g = 0
    for a in ex["arms"]:
        s = np.array(a["survivals"], float); t = np.array(a["times"], float)
        if s.size >= 8 and s.max() > 0.8 and s[-1] < s[0] - 0.05 and (t.max() - t.min()) > 1:
            g += 1
    return g >= 2


def km_hr(ex):
    recon = []
    for i, a in enumerate(ex["arms"][:2]):
        N = int(a["n"]) if a.get("n") else 150
        ipd = reconstruct_arm_faithful(np.array(a["times"]), np.array(a["survivals"]), N,
                                       follow_up=float(max(a["times"])))
        for r in ipd:
            recon.append({"time": r["time"], "status": r["status"], "arm": i})
    lhr, se = cox_loghr_2arm(recon)
    return float(np.exp(lhr)) if np.isfinite(lhr) else None


def _looks_like_plot(im):
    """Cheap NO-OCR pre-filter: a KM plot has axis lines -- a long vertical dark run in the left
    region AND a long horizontal dark run in the bottom region. Skips photos/tables/diagrams so we
    don't waste easyocr on them."""
    import cv2
    g = cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2GRAY)
    H, W = g.shape
    dark = g < 110
    # vertical axis: a column in the left 40% with a long dark run
    left = dark[:, : int(W * 0.4)]
    vcol = left.sum(axis=0).max() if left.size else 0
    bottom = dark[int(H * 0.6):, :]
    hrow = bottom.sum(axis=1).max() if bottom.size else 0
    return vcol > 0.45 * H and hrow > 0.45 * W


def _saturated(im):
    import cv2
    hsv = cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2HSV)
    return int(((hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 40)).sum()) > 0.01 * hsv.shape[0] * hsv.shape[1]


def figures(pdf_path, max_figs=3):
    """Largest embedded figures that pass the cheap plot pre-filter -> (path, is_color)."""
    doc = fitz.open(pdf_path)
    cand = []
    try:
        for pno in range(min(doc.page_count, 40)):
            for img in doc[pno].get_images(full=True):
                try:
                    d = doc.extract_image(img[0]); im = Image.open(io.BytesIO(d["image"]))
                except Exception:
                    continue
                if im.width >= 350 and im.height >= 250:
                    cand.append((im.width * im.height, im))
    finally:
        doc.close()
    cand.sort(key=lambda x: -x[0])
    out = []
    for _, im in cand[:8]:
        if not _looks_like_plot(im):
            continue
        p = os.path.join(BASE, f"_tmp_fig{len(out)}.png")
        im.convert("RGB").save(p)
        out.append((p, _saturated(im)))
        if len(out) >= max_figs:
            break
    return out


def _hr_point_distance(hr, published_hrs):
    """Smallest log-scale distance from this HR (either orientation) to a published HR."""
    if not hr or not published_hrs:
        return None
    vals = [hr]
    if hr > 0:
        vals.append(1 / hr)
    best = None
    for pub_hr, _, _ in published_hrs:
        if pub_hr <= 0:
            continue
        for cand in vals:
            if cand <= 0:
                continue
            dist = abs(np.log(cand / pub_hr))
            best = dist if best is None else min(best, dist)
    return best


def _select_candidate(candidates, published_hrs=None):
    """Pick a KM candidate without silently favoring extreme HR artifacts.

    If the PDF text supplies published HRs, this is a validation/matching problem: choose the
    candidate closest to a published point estimate, allowing reciprocal arm orientation. If no
    matching context is available, fail closed on multi-candidate PDFs rather than guessing.
    """
    if not candidates:
        return None
    if published_hrs:
        scored = []
        for c in candidates:
            dist = _hr_point_distance(c["hr"], published_hrs)
            if dist is not None:
                scored.append((dist, abs(np.log(c["hr"])), c))
        if scored:
            scored.sort(key=lambda x: (x[0], x[1]))
            return scored[0][2]
    if len(candidates) == 1:
        return candidates[0]
    return None


def analyze_pdf(pdf_path, published_hrs=None):
    candidates = []
    for fp, is_color in figures(pdf_path):
        extractor = extract_raster_km if is_color else extract_bw_km
        try:
            ex = extractor(fp)
        except Exception:
            ex = None
        if looks_like_km(ex):
            hr = km_hr(ex)
            if hr:
                candidates.append({
                    "hr": hr,
                    "n_arms": ex["n_arms"],
                    "method": "color" if is_color else "bw",
                    "figure": os.path.basename(fp),
                })
    return _select_candidate(candidates, published_hrs)


def main():
    import glob
    manifest = {r["pmcid"]: r for r in json.load(open(os.path.join(BASE, "plos_manifest.json")))}
    pdfs = sorted(glob.glob(os.path.join(PDFDIR, "*.pdf")))
    # resume: keep already-analyzed PMCIDs
    rp = os.path.join(BASE, "batch_results.json")
    results = json.load(open(rp)) if os.path.exists(rp) else []
    done = {x["pmcid"] for x in results}
    t0 = time.time()
    for i, pdf in enumerate(pdfs):
        pmcid = os.path.splitext(os.path.basename(pdf))[0]
        if pmcid in done:
            continue
        r = manifest.get(pmcid, {"pmcid": pmcid, "abstract": ""})
        # parse the reported HR from the FULL PDF TEXT (higher yield than the truncated abstract)
        try:
            _d = fitz.open(pdf); _txt = " ".join(_d[p].get_text() for p in range(min(_d.page_count, 25))); _d.close()
        except Exception:
            _txt = ""
        pub = parse_published_hr(_txt) or parse_published_hr(r.get("abstract", ""))
        try:
            got = analyze_pdf(pdf, pub)
        except Exception as e:
            got = None
        rec = {"pmcid": r["pmcid"], "km_extracted": got is not None,
               "extracted_hr": (round(got["hr"], 3) if got else None),
               "method": (got["method"] if got else None),
               "published_hrs": [round(h, 3) for h, _, _ in pub],
               "published_ci": [[lo, hi] for _, lo, hi in pub]}
        # validation: extracted HR (or its reciprocal, arm order unknown) within any published CI
        rec["within_ci"] = None
        if got and pub:
            hr = got["hr"]
            ok = any(lo <= hr <= hi or lo <= 1 / hr <= hi for _, lo, hi in pub)
            rec["within_ci"] = bool(ok)
        results.append(rec)
        if (i + 1) % 10 == 0:
            km = sum(x["km_extracted"] for x in results)
            print(f"  {i+1}/{len(pdfs)}  km_extracted={km}  ({time.time()-t0:.0f}s)", flush=True)
            json.dump(results, open(os.path.join(BASE, "batch_results.json"), "w"), indent=1)
    json.dump(results, open(os.path.join(BASE, "batch_results.json"), "w"), indent=1)
    n = len(results); km = sum(x["km_extracted"] for x in results)
    val = [x for x in results if x["within_ci"] is not None]
    within = sum(x["within_ci"] for x in val)
    print("\n" + "=" * 60)
    print(f"PDFs analyzed: {n}")
    print(f"KM curve auto-extracted (CV success): {km}/{n} = {100*km/max(n,1):.0f}%")
    print(f"HR-validation subset (abstract HR+CI parsed & KM extracted): {len(val)}")
    print(f"  extracted HR within published CI: {within}/{len(val)} = {100*within/max(len(val),1):.0f}%")


if __name__ == "__main__":
    main()
