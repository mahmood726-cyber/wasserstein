"""Real-world validation: run the extractor on figures from real open-access PDFs.

Extracts every embedded image from each PDF page (isolating figures from body text), runs
the raster KM extractor, keeps the ones that look like survival curves (>=2 arms, S starts
high, decreasing, reasonable length), reconstructs IPD, and reports the Cox HR magnitude to
compare against the paper's published HR. Real trials use drug names in the legend (not
Control/Experimental), so orientation-role matching does not fire -- HR MAGNITUDE (|log HR|)
is the fair comparison; direction needs a legend parser tuned to arm names.
"""
import sys, os, io
import numpy as np
import fitz
from PIL import Image
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "benchmark")))
from raster_km_extractor import extract_raster_km
from faithful_guyot import reconstruct_arm_faithful
from km_metrics import cox_loghr_2arm


def looks_like_km(ex):
    if not ex or ex.get("n_arms", 0) < 2:
        return False
    good = 0
    for a in ex["arms"]:
        s = np.array(a["survivals"], float)
        t = np.array(a["times"], float)
        if s.size < 8:
            continue
        if s.max() > 0.8 and s[-1] < s[0] - 0.05 and (t.max() - t.min()) > 1:
            good += 1
    return good >= 2


def extract_images(pdf_path, out_dir):
    doc = fitz.open(pdf_path)
    saved = []
    for pno in range(len(doc)):
        for img in doc[pno].get_images(full=True):
            xref = img[0]
            try:
                d = doc.extract_image(xref)
            except Exception:
                continue
            im = Image.open(io.BytesIO(d["image"]))
            if im.width < 300 or im.height < 200:      # skip logos/tiny
                continue
            p = os.path.join(out_dir, f"p{pno}_x{xref}.png")
            im.convert("RGB").save(p)
            saved.append((pno, xref, p, im.width, im.height))
    doc.close()
    return saved


def run(pdf_path, published_hr, name):
    out_dir = os.path.join(os.path.dirname(pdf_path), name + "_imgs")
    os.makedirs(out_dir, exist_ok=True)
    imgs = extract_images(pdf_path, out_dir)
    print(f"\n=== {name}: {len(imgs)} embedded images (>=300x200) ===")
    hits = []
    for pno, xref, p, w, h in imgs:
        try:
            ex = extract_raster_km(p)
        except Exception as e:
            ex = None
        if looks_like_km(ex):
            # reconstruct + Cox HR (arm order unknown -> report magnitude both ways)
            recon = []
            for i, a in enumerate(ex["arms"][:2]):
                N = int(a["n"]) if a.get("n") else 200
                ipd = reconstruct_arm_faithful(np.array(a["times"]), np.array(a["survivals"]), N,
                                               follow_up=float(max(a["times"])))
                for r in ipd:
                    recon.append({"time": r["time"], "status": r["status"], "arm": i})
            lhr, se = cox_loghr_2arm(recon)
            hr = float(np.exp(lhr)) if np.isfinite(lhr) else None
            hits.append((p, w, h, ex, hr))
            ns = [a.get("n") for a in ex["arms"]]
            print(f"  KM figure: {os.path.basename(p)} ({w}x{h}) arms={ex['n_arms']} "
                  f"x_range={[round(x,1) for x in ex['x_range']]} N={ns} -> reconstructed |HR|="
                  f"{(min(hr,1/hr) if hr else None) and round(min(hr,1/hr),3)}")
    if not hits:
        print("  no KM figure recognized")
    print(f"  PUBLISHED HR(s): {published_hr}")
    return hits


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    run(os.path.join(base, "hasegawa2016.pdf"), "RFS 0.56 (0.38-0.83), OS 0.80 (0.48-1.35)", "hasegawa2016")
    run(os.path.join(base, "zhou2015.pdf"), "OS 0.63 (0.46-0.86)", "zhou2015")
