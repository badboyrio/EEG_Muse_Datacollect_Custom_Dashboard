"""Compare subjects: quality, blink rate, band-power effects (before / after blink removal),
within-subject and cross-subject classification.
Usage: python compare.py <sub01_folder> <sub02_folder> [...] --out out
"""
import argparse, json, os, warnings
import numpy as np, pandas as pd
from scipy import signal, stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
FS = 256
ALL = ["TP9", "AF7", "AF8", "TP10"]
BANDS = {"Delta": (1, 4), "Theta": (4, 8), "Alpha": (8, 13), "Beta": (13, 30)}
COMMON = ["TP9", "AF7"]          # channels usable in every subject so far


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def load(folder):
    d = pd.read_csv(os.path.join(folder, "eeg.csv")); ev = pd.read_csv(os.path.join(folder, "events.csv"))
    b, a = signal.butter(4, [1, 40], btype="band", fs=FS)
    raw = d[ALL].values - d[ALL].values.mean(0)
    X = signal.filtfilt(b, a, raw, axis=0)
    return d, ev, raw, X


def quality(d, raw, X, ev):
    t = d.t_rel.values; q = {}
    task = np.zeros(len(t), bool)
    for _, e in ev[ev.trial_type.isin(["rest", "arithmetic"])].iterrows():
        task |= (t >= e.onset) & (t < e.onset + e.duration)
    for i, c in enumerate(ALL):
        f, p = signal.welch(raw[:, i], FS, nperseg=FS * 4)
        db = lambda lo, hi: 10 * np.log10(p[(f >= lo) & (f < hi)].mean())
        x = X[task, i]
        # share of 1-s task windows exceeding 100 uV
        nwin = len(x) // FS
        bad = np.mean([np.abs(x[k * FS:(k + 1) * FS]).max() > 100 for k in range(nwin)])
        slope = db(1, 4) - db(30, 40)
        status = "dead (no contact)" if slope < 10 else ("noisy" if bad > .25 and c.startswith("AF") else "good")
        q[c] = {"slope_db": round(float(slope), 1), "median_abs_uV": round(float(np.median(np.abs(x))), 1),
                "pct_windows_over_100uV": round(100 * float(bad), 1), "status": status}
    return q


def blinks(X):
    pk, _ = signal.find_peaks(-X[:, 0], height=60, distance=int(.4 * FS))
    return pk


def analyse(folder):
    d, ev, raw, X = load(folder); t = d.t_rel.values
    Q = quality(d, raw, X, ev)
    pk = blinks(X); tpk = t[pk]
    mask = np.zeros(len(X), bool)
    for p in pk: mask[max(0, p - FS // 5):p + FS // 5] = True
    ci = {c: ALL.index(c) for c in ALL}

    def rate(lab):
        e = ev[ev.trial_type == lab]
        n = sum(((tpk >= r.onset) & (tpk < r.onset + r.duration)).sum() for r in e.itertuples())
        return n / (e.duration.sum() / 60)

    blocks = ev[ev.trial_type.isin(["rest", "arithmetic"])]
    rows, wins, cwins = [], [], []
    for e in blocks.itertuples():
        i0 = np.searchsorted(t, e.onset + 2); i1 = np.searchsorted(t, e.onset + e.duration)
        rec = {"cls": e.trial_type, "block": int(e.block),
               "blink_rate": ((tpk >= e.onset + 2) & (tpk < e.onset + e.duration)).sum() / ((e.duration - 2) / 60)}
        # all-window PSD (4-s windows, amplitude rejection only)
        Ps = []
        for i in range(i0, i1 - 4 * FS + 1, 2 * FS):
            seg = X[i:i + 4 * FS]
            if np.abs(seg[:, ci["AF7"]]).max() > 100 or np.abs(seg[:, ci["TP9"]]).max() > 250: continue
            f, P = signal.welch(seg, FS, nperseg=2 * FS, noverlap=FS, axis=0); Ps.append(P)
            w = {"cls": e.trial_type, "block": int(e.block), "nblink": int(((pk >= i) & (pk < i + 4 * FS)).sum())}
            for c in COMMON:
                tot = P[(f >= 1) & (f <= 40), ci[c]].sum()
                for bn, (lo, hi) in BANDS.items():
                    v = P[(f >= lo) & (f < hi), ci[c]].sum()
                    w[f"{c} {bn} log"] = np.log10(v); w[f"{c} {bn} rel"] = v / tot
            wins.append(w)
        P = np.mean(Ps, 0)
        for c in COMMON:
            tot = P[(f >= 1) & (f <= 40), ci[c]].sum()
            for bn, (lo, hi) in BANDS.items():
                v = P[(f >= lo) & (f < hi), ci[c]].sum()
                rec[f"{c} {bn} (log abs)"] = np.log10(v); rec[f"{c} {bn} (rel %)"] = 100 * v / tot
        # blink-free 1-s windows
        Pc = []
        for i in range(i0, i1 - FS, FS // 2):
            seg = X[i:i + FS]
            if mask[i:i + FS].any() or np.abs(seg[:, ci["AF7"]]).max() > 100: continue
            f1, P1 = signal.periodogram(seg * np.hanning(FS)[:, None], FS, axis=0); Pc.append(P1)
            w = {"cls": e.trial_type, "block": int(e.block)}
            for c in COMMON:
                tot = P1[(f1 >= 1) & (f1 <= 40), ci[c]].sum()
                for bn, (lo, hi) in BANDS.items():
                    v = P1[(f1 >= lo) & (f1 < hi), ci[c]].sum()
                    w[f"{c} {bn} log"] = np.log10(v); w[f"{c} {bn} rel"] = v / tot
            cwins.append(w)
        rec["n_clean"] = len(Pc)
        if len(Pc) >= 3:
            P1m = np.mean(Pc, 0)
            for c in COMMON:
                tot = P1m[(f1 >= 1) & (f1 <= 40), ci[c]].sum()
                for bn, (lo, hi) in BANDS.items():
                    v = P1m[(f1 >= lo) & (f1 < hi), ci[c]].sum()
                    rec[f"CLEAN {c} {bn} (log abs)"] = np.log10(v); rec[f"CLEAN {c} {bn} (rel %)"] = 100 * v / tot
        rows.append(rec)
    B = pd.DataFrame(rows)
    R = B[B.cls == "rest"].set_index("block").sort_index(); A = B[B.cls == "arithmetic"].set_index("block").sort_index()
    st = []
    for col in [c for c in B.columns if c not in ("cls", "block", "n_clean")]:
        ok = R[col].notna() & A[col].notna()
        r, ar = R.loc[ok, col].values, A.loc[ok, col].values
        if len(r) < 5: continue
        dd = ar - r; tt = stats.ttest_rel(ar, r)
        st.append({"feature": col, "rest": r.mean(), "arith": ar.mean(), "dz": dd.mean() / dd.std(ddof=1),
                   "p": tt.pvalue, "n_pairs": len(r), "arith_gt": int((dd > 0).sum())})
    S = pd.DataFrame(st)
    for grp in [S.feature.str.startswith("CLEAN"), ~S.feature.str.startswith("CLEAN") & (S.feature != "blink_rate")]:
        S.loc[grp, "q"] = bh(S.loc[grp, "p"])
    S.loc[S.feature == "blink_rate", "q"] = S.loc[S.feature == "blink_rate", "p"]
    return {"quality": Q, "blink": {k: rate(k) for k in ["baseline_eyes_closed", "baseline_eyes_open", "rest", "arithmetic"]},
            "blocks": B, "stats": S, "wins": pd.DataFrame(wins), "cwins": pd.DataFrame(cwins)}


def lobo(W, cols):
    y = (W.cls == "arithmetic").astype(int).values; X = W[cols].values; g = W.block.values; ok = 0
    for b in np.unique(g):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
        m.fit(X[g != b], y[g != b]); ok += (m.predict(X[g == b]) == y[g == b]).sum()
    return ok / len(y)


def cross(Wtr, Wte, cols):
    z = lambda W: (W[cols] - W[cols].mean()) / W[cols].std()       # per-subject normalisation
    m = LogisticRegression(C=0.1, max_iter=2000).fit(z(Wtr).values, (Wtr.cls == "arithmetic").astype(int))
    return (m.predict(z(Wte).values) == (Wte.cls == "arithmetic").astype(int)).mean()


def perm_p(fn, W, cols, acc, n=500, seed=0):
    rng = np.random.default_rng(seed); null = []
    for _ in range(n):
        Wp = W.copy()
        flip = {b: rng.random() < .5 for b in W.block.unique()}
        Wp["cls"] = [("rest" if c == "arithmetic" else "arithmetic") if flip[b] else c for c, b in zip(W.cls, W.block)]
        null.append(fn(Wp, cols))
    null = np.array(null)
    return (1 + (null >= acc).sum()) / (1 + n)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("folders", nargs="+"); ap.add_argument("--out", default="out")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    R = {f"sub-{i + 1:02d}": analyse(f) for i, f in enumerate(a.folders)}
    feat = [c for c in R["sub-01"]["wins"].columns if c not in ("cls", "block", "nblink")]
    summ = {}
    for s, r in R.items():
        r["blocks"].to_csv(os.path.join(a.out, f"{s}_blocks.csv"), index=False)
        r["stats"].to_csv(os.path.join(a.out, f"{s}_stats.csv"), index=False)
        acc_all = lobo(r["wins"], feat); acc_blink = lobo(r["wins"], ["nblink"])
        acc_clean = lobo(r["cwins"], feat)
        summ[s] = {"quality": r["quality"], "blink_per_min": r["blink"],
                   "clf_within_all": acc_all, "p_within_all": perm_p(lobo, r["wins"], feat, acc_all),
                   "clf_within_blinkcount_only": acc_blink,
                   "clf_within_blinkfree": acc_clean, "p_within_blinkfree": perm_p(lobo, r["cwins"], feat, acc_clean),
                   "n_windows": len(r["wins"]), "n_blinkfree_windows": r["cwins"].groupby("cls").size().to_dict()}
    subs = list(R)
    cr = {}
    for tr in subs:
        for te in subs:
            if tr == te: continue
            cr[f"{tr}->{te}"] = {"all_windows": cross(R[tr]["wins"], R[te]["wins"], feat),
                                 "blink_free": cross(R[tr]["cwins"], R[te]["cwins"], feat),
                                 "blink_count_only": cross(R[tr]["wins"], R[te]["wins"], ["nblink"])}
    summ["cross_subject"] = cr
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float)
    print(json.dumps(summ, indent=1, default=float))
    for s, r in R.items():
        print("\n", s); print(r["stats"].sort_values("p").round(4).to_string())


if __name__ == "__main__":
    main()
