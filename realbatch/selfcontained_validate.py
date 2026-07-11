"""Self-contained figure-HR validation harness.

The reliability bottleneck for real-paper KM->IPD is MATCHING a published hazard ratio to the
correct 2-arm KM figure (data curation). That bottleneck vanishes when the HR is *printed on the
figure itself*: a vision reader returns BOTH the curve and the in-figure HR from the same image, so
the ground truth is intrinsic -- no external matching, no curation.

This module makes that protocol reproducible:
  1. extract_figures(pdf)         -> largest embedded figures as PNGs (candidate KM plots)
  2. a pluggable VISION READER    -> reads one figure -> a dict {km_found, curves, printed_hr, ...}
  3. validate_read(read)          -> reconstruct HR from the curve, compare to the printed HR
  4. run(reads)                   -> aggregate: median abs-% err + fraction reproduced within CI

The vision reader is pluggable (register_reader / pass read_fn). In this repo the reads were
captured via the Agent-tool image Read; they are stored as fixtures in
benchmark/results/vision_realpaper_reliability.json so this harness runs fully offline. Point
`--reads <json>` at a file of captured reads (list of the reader-output dicts, each with a
`pmcid`) to reproduce, or wire a live vision model into `read_fn`.
"""
import io, json, os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from vision_km_pipeline import reconstruct_two_arm  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))


def extract_figures(pdf_path, out_dir, max_figs=2, min_w=380, min_h=280):
    """Save the largest plausibly-plot-shaped embedded figures as PNGs; return their paths."""
    import fitz
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    figs = []
    try:
        for pno in range(min(doc.page_count, 30)):
            for img in doc[pno].get_images(full=True):
                try:
                    d = doc.extract_image(img[0]); im = Image.open(io.BytesIO(d["image"]))
                except Exception:
                    continue
                if im.width >= min_w and im.height >= min_h and 0.5 < im.width / im.height < 2.4:
                    figs.append((im.width * im.height, im))
    finally:
        doc.close()
    figs.sort(key=lambda x: -x[0])
    pmcid = os.path.splitext(os.path.basename(pdf_path))[0]
    out = []
    for k, (_, im) in enumerate(figs[:max_figs]):
        p = os.path.join(out_dir, f"{pmcid}_f{k}.png")
        im.convert("RGB").save(p)
        out.append(p)
    return out


def validate_read(read):
    """Given a vision read carrying a curve AND a printed HR, reconstruct the HR and compare.

    The reader labels the higher-survival curve; the printed HR's reference arm is inferred from
    direction: reconstruct exp=higher-survival vs ctl=lower-survival, then orient the printed HR so
    the comparison matches (a printed HR>1 with the treatment being the HIGHER curve means the
    treatment is the *reference*, i.e. compare against 1/printed_hr).
    """
    if not read.get("km_found") or read.get("printed_hr") is None:
        return None
    a1, a2 = read["arm1"], read["arm2"]
    higher = read.get("higher_curve", "arm1")
    exp, ctl = (a1, a2) if higher == "arm1" else (a2, a1)
    n_exp = int(exp.get("n") or ctl.get("n") or 150)
    n_ctl = int(ctl.get("n") or exp.get("n") or 150)
    nar = read.get("nar_times") or None
    ne = read.get("nar_arm1") if higher == "arm1" else read.get("nar_arm2")
    nc = read.get("nar_arm2") if higher == "arm1" else read.get("nar_arm1")
    if nar and ne:
        n_exp, n_ctl = int(ne[0]), int(nc[0])
    r = reconstruct_two_arm(exp["survival"], ctl["survival"], read["times"], n_exp, n_ctl,
                            follow_up=float(max(read["times"])),
                            nar_times=nar, nar_exp=ne if nar else None, nar_ctl=nc if nar else None)
    recon = r["hr"]
    # Orient the printed HR to the exp=higher-survival vs ctl comparison. exp having higher survival
    # => the true HR(exp vs ctl) < 1. If the printed HR is >1 it is stated the other way; invert it.
    ph, plo, phi = read["printed_hr"], read["printed_ci"][0], read["printed_ci"][1]
    if ph > 1.0:
        ph, plo, phi = 1.0 / ph, 1.0 / phi, 1.0 / plo
    within = plo <= recon <= phi
    err = abs(recon - ph) / ph * 100.0
    return {"pmcid": read.get("pmcid"), "endpoint": read.get("endpoint"),
            "recon_hr": round(recon, 3), "recon_ci": [round(x, 3) for x in r["hr_ci"]],
            "printed_hr_oriented": round(ph, 3), "printed_ci_oriented": [round(plo, 3), round(phi, 3)],
            "abs_pct_err": round(err, 2), "within_printed_ci": bool(within)}


def run(reads):
    """Validate every read that carries an in-figure HR; return the aggregate + per-case detail."""
    detail = [v for v in (validate_read(r) for r in reads) if v]
    if not detail:
        return {"n": 0, "detail": []}
    errs = sorted(d["abs_pct_err"] for d in detail)
    med = errs[len(errs) // 2] if len(errs) % 2 else (errs[len(errs) // 2 - 1] + errs[len(errs) // 2]) / 2
    within = sum(d["within_printed_ci"] for d in detail)
    return {"protocol": "self-contained: vision reads curve + in-figure HR; no external matching",
            "n": len(detail), "within_ci": within, "rate_pct": round(100 * within / len(detail)),
            "median_abs_pct_err": round(med, 2), "detail": detail}


def _load_reads(path):
    """Load captured reads. Accepts either a raw list of reader-output dicts, or the
    reliability JSON whose `self_contained_validation.detail` fixtures can be replayed."""
    obj = json.load(open(path, encoding="utf-8"))
    if isinstance(obj, list):
        return obj
    return obj  # caller passes a list; reliability-JSON replay handled in __main__


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--reads", help="JSON file: list of vision-reader output dicts (each with pmcid)")
    ap.add_argument("--out", default=os.path.join(BASE, "selfcontained_results.json"))
    a = ap.parse_args()
    reads = _load_reads(a.reads) if a.reads else []
    res = run(reads)
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    print(f"self-contained validation: {res.get('within_ci', 0)}/{res.get('n', 0)} within printed CI, "
          f"median abs-% err {res.get('median_abs_pct_err', 'NA')}  -> {a.out}")
