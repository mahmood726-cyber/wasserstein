"""Faithful Guyot reconstruction (roadmap L3) — Python port of the registry-ipd engine.

Ports guyotCore + normalizeAndExpand + buildRiskIndices + pavaDecreasing from the audited,
R-/test-verified JS engine at C:\\Projects\\registry-ipd\\src\\engine.js (functions of the
same name). This REPLACES wasserstein's heuristic single-swap reconstruction
(improved_guyot_algorithm.py) with the faithful anchor-matching algorithm:
  - iterates censoring so the reconstructed at-risk matches the number-at-risk anchors,
  - reconciles to total_events by SWAPPING censor<->event (never adds/drops bodies),
  - guarantees EXACTLY N rows (population conservation) by construction.

Parity target: the JS engine's Tier-A integer round-trip recovers the exact death schedule
(engine.spec.js 'Guyot parity'). This port aims to match that behaviour in Python so the
benchmark can measure the reconstruction-fidelity gain vs the heuristic.
"""
from __future__ import annotations
import numpy as np


# --------------------------------------------------------------- PAVA (decreasing)
def pava_decreasing(s):
    """Pool-adjacent-violators enforcing a NON-INCREASING sequence. Port of pavaDecreasing."""
    s = list(map(float, s))
    n = len(s)
    if n == 0:
        return np.array([])
    v = [-x for x in s]                      # negate to reuse increasing-PAVA
    block_val, block_wt, block_start = [], [], []
    for i in range(n):
        cv, cw, cs = v[i], 1.0, i
        while block_val and block_val[-1] >= cv:
            pv = block_val.pop(); pw = block_wt.pop(); cs = block_start.pop()
            cv = (pv * pw + cv * cw) / (pw + cw); cw = pw + cw
        block_val.append(cv); block_wt.append(cw); block_start.append(cs)
    out = [0.0] * n
    idx = n
    for b in range(len(block_val) - 1, -1, -1):
        start = block_start[b]
        for k in range(start, idx):
            out[k] = -block_val[b]
        idx = start
    return np.array(out)


# --------------------------------------------------------------- risk indices
def build_risk_indices(tS, tRisk):
    nt = len(tS)
    lower = [0] * len(tRisk)
    upper = [0] * len(tRisk)
    for i in range(len(tRisk)):
        k = 0
        while k < nt and tS[k] < tRisk[i] - 1e-9:
            k += 1
        lower[i] = min(k, nt - 1)
    for i in range(len(tRisk)):
        upper[i] = max(lower[i + 1] - 1, lower[i]) if i < len(tRisk) - 1 else nt - 1
    return lower, upper


# --------------------------------------------------------------- Guyot core
def guyot_core(tS, S, tRisk, nRisk, tot_events=None):
    """Faithful port of guyotCore. Returns (d, cen) arrays over clicked points."""
    tS = list(map(float, tS)); S = list(map(float, S))
    tRisk = list(map(float, tRisk)); nRisk = [int(round(x)) for x in nRisk]
    lower, upper = build_risk_indices(tS, tRisk)
    n_int = len(nRisk); nt = len(tS)
    n_censor = [0] * n_int
    nhat = [nRisk[0] + 1] * (nt + 1)
    cen = [0] * nt
    d = [0] * nt
    KMhat = [1.0] * nt
    lastI = [0] * n_int

    def distribute_censor(i, m):
        for k in range(lower[i], upper[i] + 1):
            cen[k] = 0
        if m <= 0:
            return
        a = tS[lower[i]]; b = tS[min(lower[i + 1], nt - 1)]
        span = (b - a) or 1.0
        for j in range(m):
            ct = a + span * (j + 0.5) / m
            kk = lower[i]
            while kk < upper[i] and tS[kk + 1] <= ct + 1e-12:
                kk += 1
            cen[kk] += 1

    for i in range(n_int - 1):
        sLo = S[lower[i]] or 1e-12
        n_censor[i] = int(round(nRisk[i] * (S[lower[i + 1]] / sLo) - nRisk[i + 1]))
        guard = 0
        while (nhat[lower[i + 1]] > nRisk[i + 1]) or \
              (nhat[lower[i + 1]] < nRisk[i + 1] and n_censor[i] > 0):
            guard += 1
            if guard > 5000:
                break
            if n_censor[i] <= 0:
                for k in range(lower[i], upper[i] + 1):
                    cen[k] = 0
                n_censor[i] = 0
            else:
                distribute_censor(i, n_censor[i])
            nhat[lower[i]] = nRisk[i]
            last = lastI[i]
            for k in range(lower[i], upper[i] + 1):
                if i == 0 and k == lower[i]:
                    d[k] = 0; KMhat[k] = 1.0
                else:
                    ref = KMhat[last] or 1e-12
                    d[k] = int(round(nhat[k] * (1 - S[k] / ref)))
                    if d[k] < 0:
                        d[k] = 0
                    if d[k] > nhat[k]:
                        d[k] = nhat[k]
                KMhat[k] = (KMhat[last] or 1.0) * (1 - d[k] / (nhat[k] or 1))
                nhat[k + 1] = nhat[k] - d[k] - cen[k]
                if nhat[k + 1] < 0:
                    nhat[k + 1] = 0
                if d[k] != 0:
                    last = k
            n_censor[i] = n_censor[i] + (nhat[lower[i + 1]] - nRisk[i + 1])
            lastI[i + 1] = last

    # final interval
    i = n_int - 1
    if nt - 1 >= lower[i]:
        nhat[lower[i]] = nRisk[i]
        last = lastI[i]
        for k in range(lower[i], nt):
            if i == 0 and k == lower[i]:
                d[k] = 0; KMhat[k] = 1.0; nhat[k + 1] = nhat[k] - cen[k]
                continue
            ref = KMhat[last] or 1e-12
            d[k] = int(round(nhat[k] * (1 - S[k] / ref)))
            if d[k] < 0:
                d[k] = 0
            if d[k] > nhat[k]:
                d[k] = nhat[k]
            KMhat[k] = ref * (1 - d[k] / (nhat[k] or 1))
            cen[k] = max(0, nhat[k] - d[k]) if k == nt - 1 else 0
            nhat[k + 1] = nhat[k] - d[k] - cen[k]
            if nhat[k + 1] < 0:
                nhat[k + 1] = 0
            if d[k] != 0:
                last = k
    return d, cen


# --------------------------------------------------------------- normalize + expand
def normalize_and_expand(tS, d, cen, N, tot_events=None, follow_up=None):
    """Port of normalizeAndExpand. Returns list of {time,status} with EXACTLY N rows."""
    nt = len(tS)
    D = list(d); C = list(cen)
    n = N
    for k in range(nt):
        if D[k] < 0:
            D[k] = 0
        if D[k] > n:
            D[k] = n
        if C[k] < 0:
            C[k] = 0
        if C[k] > n - D[k]:
            C[k] = n - D[k]
        n = n - D[k] - C[k]
    tailC = max(0, n)
    tailT = follow_up if follow_up is not None else tS[nt - 1]
    if tot_events is not None:
        delta = tot_events - sum(D)
        if delta > 0:
            guard = 0
            while delta > 0 and guard < 1000:
                guard += 1
                wsum = sum(D[k] for k in range(1, nt) if C[k] > 0)
                if wsum <= 0:
                    break
                moved = 0
                for k in range(1, nt):
                    if delta <= 0:
                        break
                    if C[k] <= 0:
                        continue
                    want = max(0, int(round(delta * D[k] / wsum)))
                    take = min(want, C[k], delta)
                    C[k] -= take; D[k] += take; delta -= take; moved += take
                if moved == 0:
                    break
            for k in range(nt - 1, 0, -1):
                if delta <= 0:
                    break
                take = min(C[k], delta); C[k] -= take; D[k] += take; delta -= take
            while delta > 0 and tailC > 0:
                tailC -= 1; D[nt - 1] += 1; delta -= 1
        elif delta < 0:
            need = -delta
            dsum = sum(D[k] for k in range(1, nt))
            for k in range(nt - 1, 0, -1):
                if need <= 0:
                    break
                want = min(D[k], int(round((-delta) * D[k] / dsum))) if dsum > 0 else D[k]
                take = min(want, D[k], need)
                D[k] -= take; C[k] += take; need -= take
            for k in range(nt - 1, 0, -1):
                if need <= 0:
                    break
                take = min(D[k], need); D[k] -= take; C[k] += take; need -= take
    ipd = []
    for k in range(nt):
        for _ in range(D[k]):
            ipd.append({"time": tS[k], "status": 1})
        for _ in range(C[k]):
            ipd.append({"time": tS[k], "status": 0})
    for _ in range(tailC):
        ipd.append({"time": tailT, "status": 0})
    return ipd


# --------------------------------------------------------------- arm-level entry
def reconstruct_arm_faithful(times, survival, n_total, nar_times=None, nar_values=None,
                             total_events=None, follow_up=None):
    """Reconstruct one arm's IPD from a digitized curve + (optional) number-at-risk anchors.

    Mirrors reconstructArmGuyot in engine.js: ensure origin, PAVA-monotone S, risk anchors.
    Returns list of {time,status} with exactly n_total rows.
    """
    pts = [(float(t), float(s)) for t, s in zip(times, survival) if np.isfinite(t) and np.isfinite(s)]
    pts.sort(key=lambda p: p[0])
    if not pts or pts[0][0] > 1e-9 or pts[0][1] < 1 - 1e-9:
        pts = [(0.0, 1.0)] + pts
    tS = [p[0] for p in pts]
    Svec = pava_decreasing([min(1.0, max(0.0, p[1])) for p in pts])

    if nar_times is not None and nar_values is not None and len(nar_times):
        order = np.argsort(nar_times)
        tRisk = [float(nar_times[i]) for i in order]
        nRisk = [int(round(nar_values[i])) for i in order]
    else:
        tRisk, nRisk = [], []
    if not tRisk or tRisk[0] > tS[0] + 1e-9:
        tRisk = [tS[0]] + tRisk
        nRisk = [int(n_total if n_total else (nRisk[0] if nRisk else 1))] + nRisk

    d, cen = guyot_core(tS, Svec, tRisk, nRisk, total_events)
    ipd = normalize_and_expand(tS, d, cen, int(n_total), total_events, follow_up)
    return ipd
