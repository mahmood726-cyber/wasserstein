"""Download ~300 open-access RCT figure sets from the NCBI PMC Open-Access subset.

Europe PMC search -> PMCIDs (survival + hazard ratio + randomized, open access) -> NCBI OA
service -> tgz package (via the https mirror) -> extract the figure JPEGs (gNNN.jpg). Figures land
in realbatch/figs/<PMCID>/. Polite: small delays, skip failures, resumable (skips already-done).
"""
import io, json, os, sys, time, tarfile, urllib.request, urllib.parse

BASE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(BASE, "figs")
os.makedirs(FIGDIR, exist_ok=True)
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 300
UA = {"User-Agent": "Mozilla/5.0 (research; KM-digitization batch)"}


def _get(url, timeout=40):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def collect_pmcids(n):
    """Page Europe PMC for open-access survival RCTs -> list of PMCIDs."""
    q = urllib.parse.quote('(kaplan-meier AND "hazard ratio" AND randomized) AND OPEN_ACCESS:Y AND IN_EPMC:Y')
    ids, cursor = [], "*"
    while len(ids) < n:
        url = (f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={q}"
               f"&format=json&pageSize=100&cursorMark={urllib.parse.quote(cursor)}")
        d = json.loads(_get(url))
        res = d.get("resultList", {}).get("result", [])
        if not res:
            break
        for r in res:
            pmcid = r.get("pmcid")
            if pmcid and pmcid not in ids:
                ids.append(pmcid)
        nc = d.get("nextCursorMark")
        if not nc or nc == cursor:
            break
        cursor = nc
        time.sleep(0.34)
    return ids[:n]


def oa_tgz_url(pmcid):
    xml = _get(f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmcid}").decode("utf-8", "replace")
    i = xml.find('format="tgz"')
    if i < 0:
        return None
    h = xml.find('href="', i)
    if h < 0:
        return None
    href = xml[h + 6: xml.find('"', h + 6)]
    return href.replace("ftp://ftp.ncbi.nlm.nih.gov/", "https://ftp.ncbi.nlm.nih.gov/")


def fetch_figs(pmcid):
    outdir = os.path.join(FIGDIR, pmcid)
    if os.path.isdir(outdir) and any(f.endswith((".jpg", ".png")) for f in os.listdir(outdir)):
        return "skip"
    url = oa_tgz_url(pmcid)
    if not url:
        return "no_oa"
    try:
        blob = _get(url, timeout=90)
        tf = tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz")
    except Exception as e:
        return f"dl_err:{str(e)[:30]}"
    os.makedirs(outdir, exist_ok=True)
    n = 0
    for m in tf.getmembers():
        base = os.path.basename(m.name).lower()
        # figure graphics only (skip tables t###, thumbnails); KM figures are gNNN.jpg/png
        if m.isfile() and base.endswith((".jpg", ".png")) and (".g0" in base or base.startswith("fig") or "figure" in base):
            try:
                data = tf.extractfile(m).read()
                if len(data) > 8000:                      # skip tiny thumbnails
                    open(os.path.join(outdir, base), "wb").write(data)
                    n += 1
            except Exception:
                pass
    if n == 0:
        # fallback: keep any reasonably-large image
        for m in tf.getmembers():
            base = os.path.basename(m.name).lower()
            if m.isfile() and base.endswith((".jpg", ".png")):
                data = tf.extractfile(m).read()
                if len(data) > 30000:
                    open(os.path.join(outdir, base), "wb").write(data); n += 1
    return f"ok:{n}" if n else "no_fig"


def main():
    print(f"collecting up to {TARGET} PMCIDs ...", flush=True)
    ids = collect_pmcids(TARGET)
    print(f"got {len(ids)} PMCIDs; downloading figure sets ...", flush=True)
    counts = {"ok": 0, "skip": 0, "fail": 0}
    manifest = []
    for i, pmcid in enumerate(ids):
        try:
            r = fetch_figs(pmcid)
        except Exception as e:
            r = f"err:{str(e)[:30]}"
        if r.startswith("ok") or r == "skip":
            counts["ok" if r.startswith("ok") else "skip"] += 1
        else:
            counts["fail"] += 1
        manifest.append({"pmcid": pmcid, "result": r})
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(ids)}  ok={counts['ok']} skip={counts['skip']} fail={counts['fail']}", flush=True)
        time.sleep(0.25)
    json.dump({"target": TARGET, "n_ids": len(ids), "counts": counts, "manifest": manifest},
              open(os.path.join(BASE, "download_manifest.json"), "w"), indent=1)
    fig_sets = sum(1 for d in os.listdir(FIGDIR) if os.path.isdir(os.path.join(FIGDIR, d))
                   and any(f.endswith((".jpg", ".png")) for f in os.listdir(os.path.join(FIGDIR, d))))
    print(f"DONE: {fig_sets} figure sets on disk; ok={counts['ok']} skip={counts['skip']} fail={counts['fail']}", flush=True)


if __name__ == "__main__":
    main()
