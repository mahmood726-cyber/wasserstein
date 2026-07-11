"""Download ~300 open-access PLoS RCT PDFs (survival + hazard ratio + randomized).

Europe PMC -> PLoS DOIs -> publisher PDF (journals.plos.org, which allows direct download).
Stores PDFs in realbatch/pdfs/ and a manifest with DOI/PMCID/title/abstract (abstract kept so
the published HR can be parsed later for validation). Resumable, polite, skips failures.
"""
import json, os, sys, time, urllib.request, urllib.parse

BASE = os.path.dirname(os.path.abspath(__file__))
PDFDIR = os.path.join(BASE, "pdfs")
os.makedirs(PDFDIR, exist_ok=True)
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 300
UA = {"User-Agent": "Mozilla/5.0 (research; KM digitization)"}
JMAP = {"pone": "plosone", "pmed": "plosmedicine", "pbio": "plosbiology",
        "pgen": "plosgenetics", "ppat": "plospathogens", "pcbi": "ploscompbiol", "pntd": "plosntds"}


def _get(url, timeout=60):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()


def collect(n):
    q = urllib.parse.quote('(kaplan-meier AND "hazard ratio" AND randomized) AND OPEN_ACCESS:Y '
                           'AND (JOURNAL:"PLoS One" OR JOURNAL:"PLoS Medicine")')
    out, cursor = [], "*"
    while len(out) < n:
        url = (f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={q}"
               f"&format=json&pageSize=100&resultType=core&cursorMark={urllib.parse.quote(cursor)}")
        d = json.loads(_get(url, 40))
        res = d.get("resultList", {}).get("result", [])
        if not res:
            break
        for r in res:
            doi = r.get("doi")
            if doi and doi.startswith("10.1371/journal.") and r.get("pmcid"):
                out.append({"doi": doi, "pmcid": r.get("pmcid"), "title": (r.get("title") or "")[:120],
                            "abstract": (r.get("abstractText") or "")[:1500]})
        nc = d.get("nextCursorMark")
        if not nc or nc == cursor:
            break
        cursor = nc
        time.sleep(0.34)
    # dedupe by doi
    seen, uniq = set(), []
    for r in out:
        if r["doi"] not in seen:
            seen.add(r["doi"]); uniq.append(r)
    return uniq[:n]


def pdf_url(doi):
    code = doi.split("journal.")[1].split(".")[0]
    j = JMAP.get(code)
    if not j:
        return None
    return f"https://journals.plos.org/{j}/article/file?id={doi}&type=printable"


def main():
    print(f"collecting up to {TARGET} PLoS DOIs ...", flush=True)
    recs = collect(TARGET)
    json.dump(recs, open(os.path.join(BASE, "plos_manifest.json"), "w"), indent=1)  # save early
    print(f"got {len(recs)} DOIs; downloading PDFs ...", flush=True)
    ok = fail = skip = 0
    for i, r in enumerate(recs):
        path = os.path.join(PDFDIR, f"{r['pmcid']}.pdf")
        if os.path.exists(path) and os.path.getsize(path) > 20000:
            r["pdf"] = os.path.basename(path); skip += 1
        else:
            u = pdf_url(r["doi"])
            try:
                blob = _get(u, 70)
                if blob[:5] == b"%PDF-" and len(blob) > 20000:
                    open(path, "wb").write(blob); r["pdf"] = os.path.basename(path); ok += 1
                else:
                    r["pdf"] = None; fail += 1
            except Exception as e:
                r["pdf"] = None; fail += 1
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(recs)}  ok={ok} skip={skip} fail={fail}", flush=True)
            json.dump(recs, open(os.path.join(BASE, "plos_manifest.json"), "w"), indent=1)
        time.sleep(0.3)
    json.dump(recs, open(os.path.join(BASE, "plos_manifest.json"), "w"), indent=1)
    have = sum(1 for f in os.listdir(PDFDIR) if f.endswith(".pdf") and os.path.getsize(os.path.join(PDFDIR, f)) > 20000)
    print(f"DONE: {have} PDFs on disk (ok={ok} skip={skip} fail={fail})", flush=True)


if __name__ == "__main__":
    main()
