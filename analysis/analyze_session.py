"""
Muse EEG — rest vs mental arithmetic: statistics + figures for one session.

Usage:
    python analyze_session.py <session_folder> [--out results]
Session folder must contain eeg.csv and events.csv from record_session.py.

Pipeline
  1. Channel QC (flat / broadband-noise channels excluded; TP10 in sub-01 ses-01)
  2. 1-40 Hz zero-phase band-pass (4th-order Butterworth)
  3. Each 40-s block -> drop first 2 s -> 4-s windows, 50 % overlap
  4. Artifact rejection per window (|x| > 100 uV frontal, > 250 uV temporal)
  5. Welch PSD (2-s segments, 0.5 Hz resolution), averaged per block
  6. Band power features: log10 absolute, relative (% of 1-40 Hz), ratios, FAA
  7. Statistics at BLOCK level (n = 10 rest, 10 arithmetic), paired by block
     number: paired t-test + Wilcoxon, Cohen's dz, Benjamini-Hochberg FDR
  8. Per-frequency paired tests (FDR) for the PSD figure
  9. EMG confound check on raw 55-100 Hz power
 10. Classification: logistic regression, leave-one-block-pair-out CV,
     block-level label-permutation test
"""
import argparse, json, os, warnings
import numpy as np, pandas as pd
from scipy import signal, stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
FS = 256
ALL_CH = ["TP9", "AF7", "AF8", "TP10"]
BANDS = {"Delta": (1, 4), "Theta": (4, 8), "Alpha": (8, 13), "Beta": (13, 30), "Gamma": (30, 40)}
THRESH = {"TP9": 250, "TP10": 250, "AF7": 100, "AF8": 100}
WIN, STEP, SKIP = 4.0, 2.0, 2.0

# palette (validated reference palette: slot 1 blue, slot 2 orange; diverging blue<->red)
C_REST, C_ARI = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8983", "#e6e5e0"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlecolor": INK,
    "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 200,
    "savefig.bbox": "tight", "figure.facecolor": "white",
})


def bh_fdr(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def stars(q):
    return "***" if q < .001 else "**" if q < .01 else "*" if q < .05 else "n.s."


def channel_qc(d):
    """Flag a channel as bad if its spectrum is flat (white-noise-like): 8-30 Hz
    power within 6 dB of 1-4 Hz power while 30-40 Hz is within 8 dB."""
    rep = {}
    for c in ALL_CH:
        x = d[c].values - d[c].mean()
        f, p = signal.welch(x, FS, nperseg=FS * 4)
        db = lambda lo, hi: 10 * np.log10(p[(f >= lo) & (f < hi)].mean())
        slope = db(1, 4) - db(30, 40)
        rep[c] = {"slope_db_1to4_vs_30to40": round(float(slope), 1),
                  "std_uV": round(float(x.std()), 1),
                  "bad": bool(slope < 10 or x.std() < 1)}
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--out", default="results")
    ap.add_argument("--perm", type=int, default=1000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0)

    d = pd.read_csv(os.path.join(a.session, "eeg.csv"))
    ev = pd.read_csv(os.path.join(a.session, "events.csv"))
    t = d.t_rel.values

    qc = channel_qc(d)
    CH = [c for c in ALL_CH if not qc[c]["bad"]]
    print("Channel QC:", qc, "\nUsing:", CH)

    b, aa = signal.butter(4, [1, 40], btype="band", fs=FS)
    Xf = signal.filtfilt(b, aa, d[ALL_CH].values - d[ALL_CH].values.mean(0), axis=0)
    Xraw = d[ALL_CH].values - d[ALL_CH].values.mean(0)
    ci = {c: ALL_CH.index(c) for c in ALL_CH}

    # ---------- windows ----------
    rows, psd_rows = [], []
    blocks = ev[ev.trial_type.isin(["rest", "arithmetic"])].reset_index(drop=True)
    for _, e in blocks.iterrows():
        s0 = e.onset + SKIP
        while s0 + WIN <= e.onset + e.duration + 1e-6:
            i0 = np.searchsorted(t, s0); i1 = i0 + int(WIN * FS)
            seg = Xf[i0:i1]
            bad = any(np.abs(seg[:, ci[c]]).max() > THRESH[c] for c in CH)
            f, P = signal.welch(seg, FS, nperseg=FS * 2, noverlap=FS, axis=0)
            fr, Pr = signal.welch(Xraw[i0:i1], FS, nperseg=FS * 2, noverlap=FS, axis=0)
            emg = {c: float(np.log10(Pr[(fr >= 55) & (fr <= 100), ci[c]].mean())) for c in CH}
            rows.append({"cls": e.trial_type, "block": int(e.block), "t": s0, "bad": bad,
                         **{f"EMG_{c}": emg[c] for c in CH}})
            psd_rows.append(P)
            s0 += STEP
    W = pd.DataFrame(rows); PSD = np.array(psd_rows)  # (n_win, n_freq, 4)
    fmask = (f >= 1) & (f <= 40); F = f[fmask]
    rej = W.groupby("cls").bad.agg(["sum", "count"]).to_dict("index")
    print("Rejected windows:", rej)

    def feats_from_psd(P):
        out = {}
        tot = {c: P[(f >= 1) & (f <= 40), ci[c]].sum() for c in CH}
        bp = {}
        for c in CH:
            for bn, (lo, hi) in BANDS.items():
                v = P[(f >= lo) & (f < hi), ci[c]].sum() * (f[1] - f[0])
                bp[(c, bn)] = v
                out[f"{c} {bn} (log abs)"] = np.log10(v)
                out[f"{c} {bn} (rel %)"] = 100 * P[(f >= lo) & (f < hi), ci[c]].sum() / tot[c]
            out[f"{c} Theta/Alpha"] = bp[(c, "Theta")] / bp[(c, "Alpha")]
            out[f"{c} Theta/Beta"] = bp[(c, "Theta")] / bp[(c, "Beta")]
            out[f"{c} Engagement β/(α+θ)"] = bp[(c, "Beta")] / (bp[(c, "Alpha")] + bp[(c, "Theta")])
        if "AF7" in CH and "AF8" in CH:
            out["Frontal alpha asymmetry (AF8−AF7)"] = np.log(bp[("AF8", "Alpha")]) - np.log(bp[("AF7", "Alpha")])
        return out

    good = ~W.bad.values
    # window-level features (for timeline + classifier)
    WF = pd.DataFrame([feats_from_psd(PSD[i]) for i in range(len(W))])
    WF = pd.concat([W, WF], axis=1)

    # block-level: average PSD over clean windows then features
    B = []
    blockpsd = {}
    for (cls, blk), g in W[good].groupby(["cls", "block"]):
        P = PSD[g.index].mean(0)
        blockpsd[(cls, blk)] = P
        B.append({"cls": cls, "block": blk, "n_win": len(g),
                  **{f"EMG_{c}": g[f"EMG_{c}"].mean() for c in CH}, **feats_from_psd(P)})
    B = pd.DataFrame(B).sort_values(["block", "cls"])
    B.to_csv(os.path.join(a.out, "block_features.csv"), index=False)

    # ---------- statistics ----------
    featcols = [c for c in B.columns if c not in ("cls", "block", "n_win")]
    R = B[B.cls == "rest"].set_index("block"); A = B[B.cls == "arithmetic"].set_index("block")
    pairs = sorted(set(R.index) & set(A.index))
    st = []
    for fc in featcols:
        r, ar = R.loc[pairs, fc].values, A.loc[pairs, fc].values
        diff = ar - r
        tt = stats.ttest_rel(ar, r); wx = stats.wilcoxon(ar, r)
        wt = stats.ttest_ind(ar, r, equal_var=False)
        st.append({"feature": fc, "rest_mean": r.mean(), "rest_sd": r.std(ddof=1),
                   "arith_mean": ar.mean(), "arith_sd": ar.std(ddof=1),
                   "mean_diff": diff.mean(), "cohen_dz": diff.mean() / diff.std(ddof=1),
                   "t_paired": tt.statistic, "p_paired_t": tt.pvalue, "p_wilcoxon": wx.pvalue,
                   "p_welch_unpaired": wt.pvalue, "n_pairs": len(pairs),
                   "arith_gt_rest_pairs": int((diff > 0).sum())})
    S = pd.DataFrame(st)
    main_mask = ~S.feature.str.startswith("EMG_")
    S["q_fdr"] = np.nan
    S.loc[main_mask, "q_fdr"] = bh_fdr(S.loc[main_mask, "p_paired_t"])
    S.loc[~main_mask, "q_fdr"] = bh_fdr(S.loc[~main_mask, "p_paired_t"])
    S["sig"] = S.q_fdr.apply(stars)
    S = S.sort_values("p_paired_t")
    S.to_csv(os.path.join(a.out, "statistics.csv"), index=False)

    # per-frequency paired t (FDR within channel)
    perfreq = {}
    for c in CH:
        r = np.array([np.log10(blockpsd[("rest", k)][fmask, ci[c]]) for k in pairs])
        ar = np.array([np.log10(blockpsd[("arithmetic", k)][fmask, ci[c]]) for k in pairs])
        p = stats.ttest_rel(ar, r, axis=0).pvalue
        perfreq[c] = {"rest": r, "ari": ar, "q": bh_fdr(p)}

    # ---------- classification ----------
    def cv_acc(cols, labels_by_block=None):
        D = WF[good].copy()
        y = (D.cls == "arithmetic").astype(int).values
        if labels_by_block is not None:
            y = np.array([labels_by_block[(c, b_)] for c, b_ in zip(D.cls, D.block)])
        X = D[cols].values; groups = D.block.values
        correct = 0
        for g in np.unique(groups):
            tr, te = groups != g, groups == g
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
            m.fit(X[tr], y[tr]); correct += (m.predict(X[te]) == y[te]).sum()
        return correct / len(y)

    base = [f"{c} {bn} (log abs)" for c in CH for bn in BANDS] + [f"{c} {bn} (rel %)" for c in CH for bn in BANDS]
    sets = {"All channels": base,
            "Frontal only (AF7, AF8)": [x for x in base if x.split()[0] in ("AF7", "AF8")],
            "Temporal only (TP9)": [x for x in base if x.split()[0] == "TP9"]}
    sets = {k: v for k, v in sets.items() if v}
    clf = {}
    for name, cols in sets.items():
        acc = cv_acc(cols)
        null = []
        for _ in range(a.perm):
            flip = {k: rng.random() < .5 for k in pairs}
            lab = {}
            for k in pairs:
                lab[("rest", k)] = int(flip[k]); lab[("arithmetic", k)] = int(not flip[k])
            null.append(cv_acc(cols, lab))
        null = np.array(null)
        clf[name] = {"acc": acc, "p_perm": (1 + (null >= acc).sum()) / (1 + len(null)),
                     "null_mean": null.mean(), "null_95": np.percentile(null, 95), "null": null}
        print(f"{name}: acc={acc:.3f} p={clf[name]['p_perm']:.4f}")

    # ---------- figures ----------
    figs = {}

    # F0 channel QC
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    qc_cols = ["#2a78d6", "#1baf7a", "#4a3aa7", "#e34948"]
    for c, col in zip(ALL_CH, qc_cols):
        x = d[c].values - d[c].mean()
        ff, pp = signal.welch(x, FS, nperseg=FS * 4)
        m = (ff >= 1) & (ff <= 100)
        ax.semilogy(ff[m], pp[m], color=col, lw=1.6, label=f"{c}{'  (excluded)' if qc[c]['bad'] else ''}")
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD (µV²/Hz)")
    ax.set_title("Channel quality: whole-session spectrum")
    ax.legend(ncol=4, loc="upper right", fontsize=9)
    figs["fig0_channel_qc.png"] = fig

    # F1 PSD
    fig, axs = plt.subplots(1, len(CH), figsize=(4.0 * len(CH), 3.6), sharey=False)
    axs = np.atleast_1d(axs)
    for ax, c in zip(axs, CH):
        pf = perfreq[c]
        for arr, col, lab in [(pf["rest"], C_REST, "Rest"), (pf["ari"], C_ARI, "Arithmetic")]:
            mu = arr.mean(0); se = arr.std(0, ddof=1) / np.sqrt(len(arr))
            ax.fill_between(F, 10 ** (mu - se), 10 ** (mu + se), color=col, alpha=.18, lw=0)
            ax.semilogy(F, 10 ** mu, color=col, lw=2, label=lab)
        for bn, (lo, hi) in BANDS.items():
            ax.axvline(lo, color=GRID, lw=.8, zorder=0)
        yl = ax.get_ylim()
        sig = pf["q"] < .05
        if sig.any():
            ax.scatter(F[sig], np.full(sig.sum(), yl[0] * 1.25), marker="s", s=10, color=INK2, zorder=5)
        ax.set_ylim(yl)
        for bn, (lo, hi) in BANDS.items():
            ax.text((lo + hi) / 2, yl[1], bn[0] if bn != "Gamma" else "γ", ha="center", va="top",
                    fontsize=8, color=MUTED)
        ax.set_title(c); ax.set_xlabel("Frequency (Hz)"); ax.set_xlim(1, 40)
    axs[0].set_ylabel("PSD (µV²/Hz), mean ± SEM over blocks")
    handles = [Line2D([], [], color=C_REST, lw=2, label="Rest"), Line2D([], [], color=C_ARI, lw=2, label="Arithmetic"),
               Line2D([], [], marker="s", ls="", color=INK2, ms=4, label="FDR q < .05 (paired, per frequency)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(.5, 1.06))
    fig.tight_layout()
    figs["fig1_psd.png"] = fig

    # F2 paired block plot
    fig, axs = plt.subplots(len(CH), len(BANDS), figsize=(2.4 * len(BANDS), 2.4 * len(CH)))
    axs = np.atleast_2d(axs)
    for i, c in enumerate(CH):
        for j, bn in enumerate(BANDS):
            ax = axs[i, j]; fc = f"{c} {bn} (log abs)"
            r, ar = R.loc[pairs, fc].values, A.loc[pairs, fc].values
            for k in range(len(pairs)):
                ax.plot([0, 1], [r[k], ar[k]], color=GRID if ar[k] > r[k] else "#d3d2cc", lw=1, zorder=1)
            ax.scatter(np.zeros(len(r)), r, color=C_REST, s=28, zorder=3, edgecolor="white", lw=1)
            ax.scatter(np.ones(len(ar)), ar, color=C_ARI, s=28, zorder=3, edgecolor="white", lw=1)
            ax.plot([-.18, .18], [r.mean()] * 2, color=INK, lw=2)
            ax.plot([.82, 1.18], [ar.mean()] * 2, color=INK, lw=2)
            row = S.set_index("feature").loc[fc]
            ax.set_title(f"{bn}\n{stars(row.q_fdr)}  dz={row.cohen_dz:+.2f}", fontsize=9,
                         color=INK if row.q_fdr < .05 else MUTED)
            ax.set_xticks([0, 1]); ax.set_xticklabels(["Rest", "Arith."] if i == len(CH) - 1 else ["", ""])
            ax.set_xlim(-.5, 1.5); ax.grid(axis="x", visible=False)
            if j == 0:
                ax.set_ylabel(f"{c}\nlog₁₀ power (µV²)")
    fig.suptitle("Band power per block (lines join rest/arithmetic of the same block pair)", y=1.01, fontsize=11,
                 fontweight="bold", color=INK)
    fig.tight_layout()
    figs["fig2_paired_bandpower.png"] = fig

    # F3 effect size heatmap
    rowsH = [f"{bn} (log abs)" for bn in BANDS] + [f"{bn} (rel %)" for bn in BANDS] + ["Theta/Alpha", "Theta/Beta", "Engagement β/(α+θ)"]
    M = np.full((len(rowsH), len(CH)), np.nan); Q = np.ones_like(M)
    Si = S.set_index("feature")
    for i, rn in enumerate(rowsH):
        for j, c in enumerate(CH):
            k = f"{c} {rn}"
            if k in Si.index:
                M[i, j] = Si.loc[k, "cohen_dz"]; Q[i, j] = Si.loc[k, "q_fdr"]
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#ef8f8e", "#b92f2e"])
    lim = max(1.5, np.nanmax(np.abs(M)))
    fig, ax = plt.subplots(figsize=(1.6 * len(CH) + 2.6, 6.0))
    im = ax.imshow(M, cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            s = stars(Q[i, j]); s = "" if s == "n.s." else s
            ax.text(j, i, f"{M[i, j]:+.2f}\n{s}", ha="center", va="center", fontsize=8.5,
                    color="white" if abs(M[i, j]) > lim * .6 else INK)
    ax.set_xticks(range(len(CH))); ax.set_xticklabels(CH)
    ax.set_yticks(range(len(rowsH))); ax.set_yticklabels(rowsH)
    ax.grid(False)
    for y in (4.5, 9.5):
        ax.axhline(y, color="white", lw=3)
    cb = fig.colorbar(im, ax=ax, shrink=.8); cb.set_label("Cohen's dz  (arithmetic − rest)", color=INK2)
    cb.outline.set_visible(False)
    ax.set_title("Effect size by channel and feature\n(red = higher in arithmetic; * FDR q<.05)")
    figs["fig3_effect_heatmap.png"] = fig

    # F4 session timeline of top features
    top = [x for x in S.feature if not x.startswith("EMG_")][:3]
    fig, axs = plt.subplots(len(top), 1, figsize=(11, 2.3 * len(top)), sharex=True)
    axs = np.atleast_1d(axs)
    for ax, fc in zip(axs, top):
        for _, e in blocks.iterrows():
            ax.axvspan(e.onset / 60, (e.onset + e.duration) / 60,
                       color=C_ARI if e.trial_type == "arithmetic" else C_REST, alpha=.10, lw=0)
        g = WF[good]
        ax.plot(g.t / 60 + WIN / 120, g[fc], ".", color=MUTED, ms=3)
        bm = B.copy(); bm["tm"] = [blocks[(blocks.trial_type == cl) & (blocks.block == bk)].onset.iloc[0] / 60 + .33
                                   for cl, bk in zip(bm.cls, bm.block)]
        for cl, col in [("rest", C_REST), ("arithmetic", C_ARI)]:
            s_ = bm[bm.cls == cl]
            ax.plot(s_.tm, s_[fc], "o", color=col, ms=8, mec="white", mew=1.5)
        q = Si.loc[fc, "q_fdr"]
        ax.set_title(f"{fc}   (dz = {Si.loc[fc, 'cohen_dz']:+.2f}, q = {q:.3g})", loc="left", fontsize=10)
    axs[-1].set_xlabel("Session time (min)")
    fig.legend(handles=[Patch(color=C_REST, alpha=.35, label="Rest block"), Patch(color=C_ARI, alpha=.35, label="Arithmetic block"),
                        Line2D([], [], marker="o", ls="", color=INK2, mec="white", label="Block mean"),
                        Line2D([], [], marker=".", ls="", color=MUTED, label="4-s window")],
               loc="upper center", ncol=4, bbox_to_anchor=(.5, 1.04))
    fig.tight_layout()
    figs["fig4_timeline.png"] = fig

    # F5 classification
    fig, axs = plt.subplots(1, len(clf), figsize=(3.8 * len(clf), 3.2), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, (name, r) in zip(axs, clf.items()):
        ax.hist(r["null"] * 100, bins=np.arange(20, 101, 2.5), color="#c9c8c1", edgecolor="white")
        ax.axvline(r["acc"] * 100, color=C_ARI, lw=2.5)
        ax.text(r["acc"] * 100, ax.get_ylim()[1] * .92, f" {r['acc'] * 100:.1f}%\n p = {r['p_perm']:.3f}",
                color=INK, fontsize=9, va="top", ha="left" if r["acc"] < .8 else "right")
        ax.set_title(name, fontsize=10); ax.set_xlabel("Accuracy (%)")
    axs[0].set_ylabel("Permutations")
    fig.suptitle("Leave-one-block-pair-out classification vs label-permutation null", y=1.03, fontsize=11,
                 fontweight="bold", color=INK)
    fig.tight_layout()
    figs["fig5_classification.png"] = fig

    for name, fg in figs.items():
        fg.savefig(os.path.join(a.out, name)); plt.close(fg)

    summary = {"channels_used": CH, "channel_qc": qc, "rejected_windows": rej,
               "n_pairs": len(pairs),
               "classification": {k: {kk: float(vv) for kk, vv in v.items() if kk != "null"} for k, v in clf.items()}}
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    print("Done ->", a.out)


if __name__ == "__main__":
    main()
