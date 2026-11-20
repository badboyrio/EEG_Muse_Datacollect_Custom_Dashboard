"""Annotated figures (circles + arrow labels) marking Normal (rest) vs Arithmetic.
Run after analyze_session.py:  python annotated_figures.py <session_folder> --res results --out annotated
"""
import argparse, json, os
import numpy as np, pandas as pd
from scipy import signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

FS = 256
CH = ["TP9", "AF7", "AF8"]
ALL = ["TP9", "AF7", "AF8", "TP10"]
BANDS = {"Delta": (1, 4), "Theta": (4, 8), "Alpha": (8, 13), "Beta": (13, 30), "Gamma": (30, 40)}
THRESH = {"TP9": 250, "AF7": 100, "AF8": 100}
C_REST, C_ARI = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8983", "#e6e5e0"
HL = "#c0392b"  # highlight ring colour (circles / boxes)
NORMAL, ARITH = "Normal (rest)", "Arithmetic"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "axes.axisbelow": True,
    "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlecolor": INK, "legend.frameon": False,
    "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white"})
ARW = dict(arrowstyle="-|>", lw=1.6, mutation_scale=14)


def callout(ax, text, xy, xytext, color, coords="data", tcoords="data", **kw):
    ax.annotate(text, xy=xy, xytext=xytext, xycoords=coords, textcoords=tcoords,
                fontsize=kw.pop("fs", 9.5), fontweight="bold", color=color, ha=kw.pop("ha", "center"),
                va=kw.pop("va", "center"),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1.2),
                arrowprops=dict(color=color, **ARW), zorder=10, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session"); ap.add_argument("--res", default="results"); ap.add_argument("--out", default="annotated")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    d = pd.read_csv(os.path.join(a.session, "eeg.csv")); ev = pd.read_csv(os.path.join(a.session, "events.csv"))
    S = pd.read_csv(os.path.join(a.res, "statistics.csv")).set_index("feature")
    B = pd.read_csv(os.path.join(a.res, "block_features.csv"))
    J = json.load(open(os.path.join(a.res, "summary.json")))
    t = d.t_rel.values
    b, aa = signal.butter(4, [1, 40], btype="band", fs=FS)
    X = signal.filtfilt(b, aa, d[ALL].values - d[ALL].values.mean(0), axis=0)
    ci = {c: ALL.index(c) for c in ALL}
    blocks = ev[ev.trial_type.isin(["rest", "arithmetic"])].reset_index(drop=True)

    # block PSDs (same method as analyze_session.py)
    bpsd, wins = {}, []
    for _, e in blocks.iterrows():
        Ps, s0 = [], e.onset + 2
        while s0 + 4 <= e.onset + e.duration + 1e-6:
            i0 = np.searchsorted(t, s0); seg = X[i0:i0 + 4 * FS]
            if not any(np.abs(seg[:, ci[c]]).max() > THRESH[c] for c in CH):
                f, P = signal.welch(seg, FS, nperseg=2 * FS, noverlap=FS, axis=0)
                Ps.append(P); wins.append((e.trial_type, int(e.block), i0, np.log10(P[(f >= 1) & (f < 4), ci["TP9"]].sum())))
            s0 += 2
        bpsd[(e.trial_type, int(e.block))] = np.mean(Ps, 0)
    fm = (f >= 1) & (f <= 40); F = f[fm]; pairs = sorted(B.block.unique())
    out = {}

    # ---------- A. raw waveform examples ----------
    W = pd.DataFrame(wins, columns=["cls", "block", "i0", "delta"])
    pick = {}
    tp = ci["TP9"]
    W["peak"] = [np.abs(X[i:i + 4 * FS, tp]).max() / X[i:i + 4 * FS, tp].std() for i in W.i0]
    for cl in ["rest", "arithmetic"]:
        w = W[(W.cls == cl) & (W.peak < 3.2)]
        pick[cl] = int(w.iloc[(w.delta - W[W.cls == cl].delta.median()).abs().argsort().iloc[0]].i0)
    bl, al = signal.butter(4, 6, fs=FS)
    fig, axs = plt.subplots(2, 2, figsize=(12, 5.6), sharex=True)
    tt = np.arange(4 * FS) / FS
    for r, c in enumerate(["TP9", "AF8"]):
        lim = 1.15 * max(np.abs(X[pick[k]:pick[k] + 4 * FS, ci[c]]).max() for k in pick)
        for k, (cl, col, name) in enumerate([("rest", C_REST, NORMAL), ("arithmetic", C_ARI, ARITH)]):
            ax = axs[r, k]; y = X[pick[cl]:pick[cl] + 4 * FS, ci[c]]
            ax.plot(tt, y, color=col, lw=1.0, alpha=.55)
            ax.plot(tt, signal.filtfilt(bl, al, y), color=col, lw=2.4)
            ax.set_ylim(-lim, lim)
            if r == 0: ax.set_title(name, color=col, fontsize=13)
            if k == 0: ax.set_ylabel(f"{c}  (µV)")
            ax.text(.01, .96, f"SD = {y.std():.1f} µV", transform=ax.transAxes, va="top", fontsize=9, color=INK2)
        # annotate the TP9 pair
    sa = signal.filtfilt(bl, al, X[pick["arithmetic"]:pick["arithmetic"] + 4 * FS, tp])
    sr = signal.filtfilt(bl, al, X[pick["rest"]:pick["rest"] + 4 * FS, tp])
    ia = int(np.argmax(np.abs(sa[FS // 2:-FS // 2]))) + FS // 2
    lim0 = axs[0, 1].get_ylim()[1]
    axs[0, 1].add_patch(Ellipse((tt[ia], sa[ia]), .8, abs(sa[ia]) * 1.3 + .15 * lim0, fill=False, ec=HL, lw=2.4, zorder=6))
    callout(axs[0, 1], "Large slow (delta/theta) wave\nduring calculation", (tt[ia] + .4 * (1 if tt[ia] < 2.5 else -1), sa[ia]),
            (2.0, -.78 * lim0 if sa[ia] > 0 else .78 * lim0), HL)
    ir = int(np.argmax(np.abs(sr)))
    callout(axs[0, 0], "Smaller slow waves\nat rest",(tt[ir], sr[ir]), (2.0, -.78 * lim0 if sr[ir] > 0 else .78 * lim0), C_REST)
    axs[0, 0].legend(handles=[Line2D([], [], color=INK2, lw=1, alpha=.55, label="EEG 1–40 Hz"),
                              Line2D([], [], color=INK2, lw=2.4, label="slow part (< 6 Hz)")], loc="lower left", ncol=2, fontsize=8.5)
    for ax in axs[1]: ax.set_xlabel("Time (s)")
    fig.suptitle("Example 4-s EEG: Normal (rest) vs Arithmetic (typical windows, 1–40 Hz)", fontweight="bold", y=1.0)
    fig.tight_layout(); out["figA_raw_waveforms.png"] = fig

    # ---------- B. PSD with markers ----------
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, c in zip(axs, CH):
        curves = {}
        for cl, col in [("rest", C_REST), ("arithmetic", C_ARI)]:
            arr = np.array([np.log10(bpsd[(cl, k)][fm, ci[c]]) for k in pairs])
            mu, se = arr.mean(0), arr.std(0, ddof=1) / np.sqrt(len(arr)); curves[cl] = mu
            ax.fill_between(F, 10 ** (mu - se), 10 ** (mu + se), color=col, alpha=.18, lw=0)
            ax.semilogy(F, 10 ** mu, color=col, lw=2.2)
        for lo, hi in BANDS.values(): ax.axvline(lo, color=GRID, lw=.8, zorder=0)
        ax.set_xlim(1, 40); ax.set_title(c, fontsize=13); ax.set_xlabel("Frequency (Hz)")
        # curve labels with arrows
        fx = 17; i = np.argmin(abs(F - fx))
        off = {"TP9": (27, .9), "AF7": (31, .55), "AF8": (35, 1.25)}[c]
        callout(ax, NORMAL, (fx if c != "AF8" else 30, 10 ** curves["rest"][i if c != "AF8" else np.argmin(abs(F - 30))]),
                (off[0], 10 ** (curves["rest"][i] + off[1])), C_REST)
        fx2 = 22 if c == "TP9" else 11
        i2 = np.argmin(abs(F - fx2))
        callout(ax, ARITH, (fx2, 10 ** curves["arithmetic"][i2]), (30 if c == "TP9" else 7, 10 ** (curves["arithmetic"][i2] - .8)), C_ARI)
    axs[0].set_ylabel("Power (µV²/Hz), mean ± SEM")
    # TP9 significant zone
    ax = axs[0]
    ax.add_patch(Ellipse((.14, .70), .27, .42, transform=ax.transAxes, fill=False, ec=HL, lw=2.4, zorder=6))
    callout(ax, "Arithmetic > Normal\n1–9 Hz (delta/theta)\nFDR q < .05", (.27, .72), (.55, .88), HL,
            coords="axes fraction", tcoords="axes fraction")
    # AF7 theta bump
    ax = axs[1]
    ax.add_patch(Ellipse((.115, .80), .13, .22, transform=ax.transAxes, fill=False, ec=HL, lw=2.2, ls="--", zorder=6))
    callout(ax, "Theta bump in\nArithmetic (trend)", (.18, .80), (.45, .9), HL, coords="axes fraction", tcoords="axes fraction")
    # AF8
    ax = axs[2]
    ax.add_patch(Ellipse((.03, .86), .09, .2, transform=ax.transAxes, fill=False, ec=HL, lw=2.2, zorder=6))
    ax.add_patch(Ellipse((.72, .33), .5, .16, transform=ax.transAxes, fill=False, ec=HL, lw=2.2, zorder=6))
    callout(ax, "Arithmetic lower in delta & beta\n→ theta share rises\n(relative theta q = .02)", (.47, .36), (.42, .6), HL,
            coords="axes fraction", tcoords="axes fraction")
    fig.suptitle("Power spectrum: Normal (rest) vs Arithmetic — circled regions show where the curves separate",
                 fontweight="bold", y=1.02)
    fig.tight_layout(); out["figB_psd_marked.png"] = fig

    # ---------- C. band power bars with markers ----------
    feats = [("TP9", "Delta"), ("TP9", "Theta"), ("TP9", "Alpha"), ("AF7", "Theta"), ("AF8", "Theta")]
    fig, axs = plt.subplots(1, len(feats), figsize=(14, 4.2))
    for ax, (c, bn) in zip(axs, feats):
        key = f"{c} {bn} (log abs)" if c == "TP9" else f"{c} {bn} (rel %)"
        r = B[B.cls == "rest"].set_index("block").loc[pairs, key].values
        ar = B[B.cls == "arithmetic"].set_index("block").loc[pairs, key].values
        for k in range(len(r)): ax.plot([0, 1], [r[k], ar[k]], color="#d3d2cc", lw=1, zorder=1)
        ax.bar([0, 1], [r.mean(), ar.mean()], color=[C_REST, C_ARI], width=.6, alpha=.25, zorder=0)
        ax.errorbar([0, 1], [r.mean(), ar.mean()], yerr=[r.std(ddof=1) / np.sqrt(10), ar.std(ddof=1) / np.sqrt(10)],
                    fmt="none", ecolor=INK, capsize=5, lw=1.5, zorder=4)
        ax.scatter(np.zeros(10), r, color=C_REST, s=26, zorder=3, ec="white"); ax.scatter(np.ones(10), ar, color=C_ARI, s=26, zorder=3, ec="white")
        lo = min(r.min(), ar.min()); hi = max(r.max(), ar.max()); pad = (hi - lo) * .12
        ax.set_ylim(lo - pad, hi + pad * 4.2)
        ax.set_xticks([0, 1]); ax.set_xticklabels([NORMAL, ARITH], fontsize=9); ax.grid(axis="x", visible=False)
        q = S.loc[key, "q_fdr"]; sig = q < .05
        stars = "***" if q < .001 else "**" if q < .01 else "*" if q < .05 else "n.s."
        y = hi + pad * 1.2
        ax.plot([0, 0, 1, 1], [y, y + pad * .6, y + pad * .6, y], color=INK, lw=1.2)
        ax.text(.5, y + pad * .9, f"{stars}  q = {q:.3f}", ha="center", va="bottom", fontsize=9.5,
                fontweight="bold", color=HL if sig else MUTED)
        if sig:
            ax.add_patch(FancyBboxPatch((.02, .02), .96, .96, transform=ax.transAxes, boxstyle="round,pad=0,rounding_size=.04",
                                        fill=False, ec=HL, lw=2, zorder=8, clip_on=False))
        ax.annotate("", xy=(1, ar.mean()), xytext=(0, r.mean()),
                    arrowprops=dict(arrowstyle="-|>", color=HL if sig else MUTED, lw=2.2, mutation_scale=16), zorder=9)
        ax.set_title(f"{c} {bn}\n({'log abs power' if c == 'TP9' else 'relative power %'})", fontsize=10)
    fig.suptitle("Normal (rest) vs Arithmetic — boxed panels are significant (FDR q < .05); arrow shows direction of change",
                 fontweight="bold", y=1.04)
    fig.tight_layout(); out["figC_bands_marked.png"] = fig

    # ---------- D. timeline TP9 delta ----------
    fc = "TP9 Delta (log abs)"
    fig, ax = plt.subplots(figsize=(13, 4.2))
    for _, e in blocks.iterrows():
        ax.axvspan(e.onset / 60, (e.onset + e.duration) / 60, color=C_ARI if e.trial_type == "arithmetic" else C_REST, alpha=.10, lw=0)
    bm = B.copy(); bm["tm"] = [blocks[(blocks.trial_type == cl) & (blocks.block == bk)].onset.iloc[0] / 60 + .33 for cl, bk in zip(bm.cls, bm.block)]
    for cl, col, name in [("rest", C_REST, NORMAL), ("arithmetic", C_ARI, ARITH)]:
        s_ = bm[bm.cls == cl].sort_values("tm")
        ax.plot(s_.tm, s_[fc], "-o", color=col, ms=9, mec="white", mew=1.5, lw=1.4, label=f"{name} block")
    ra = bm[(bm.cls == "rest") & (bm.block == 2)].iloc[0]; aa_ = bm[(bm.cls == "arithmetic") & (bm.block == 3)].iloc[0]
    callout(ax, "Normal (rest) block\nlower delta", (ra.tm, ra[fc]), (ra.tm + .6, ra[fc] - .35), C_REST)
    callout(ax, "Arithmetic block\nhigher delta", (aa_.tm, aa_[fc]), (aa_.tm - .9, aa_[fc] + .38), C_ARI)
    for k in [6, 7]:
        p_ = bm[bm.block == k]
        ax.add_patch(Ellipse(((p_.tm.min() + p_.tm.max()) / 2, p_[fc].mean()), abs(p_.tm.max() - p_.tm.min()) + .9,
                             abs(p_[fc].max() - p_[fc].min()) + .35, fill=False, ec=HL, lw=2, zorder=6))
    callout(ax, "Each pair: Arithmetic above Normal\n(10 of 10 pairs)", (12.25, 2.86), (8.3, 3.25), HL)
    ax.set_ylim(1.75, 3.45); ax.set_xlabel("Session time (min)"); ax.set_ylabel("TP9 delta, log₁₀ µV²")
    ax.legend(loc="lower right", ncol=2)
    ax.set_title("TP9 delta power, block by block across the session", loc="left")
    fig.tight_layout(); out["figD_timeline_marked.png"] = fig

    # ---------- E. heatmap with boxes ----------
    rowsH = [f"{bn} (log abs)" for bn in BANDS] + [f"{bn} (rel %)" for bn in BANDS] + ["Theta/Alpha", "Theta/Beta", "Engagement β/(α+θ)"]
    M = np.array([[S.loc[f"{c} {r}", "cohen_dz"] for c in CH] for r in rowsH])
    Q = np.array([[S.loc[f"{c} {r}", "q_fdr"] for c in CH] for r in rowsH])
    cmap = LinearSegmentedColormap.from_list("d", ["#184f95", "#6da7ec", "#f0efec", "#ef8f8e", "#b92f2e"])
    fig, ax = plt.subplots(figsize=(7.4, 6.4)); lim = np.abs(M).max()
    im = ax.imshow(M, cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=9, color="white" if abs(M[i, j]) > lim * .6 else INK)
            if Q[i, j] < .05:
                ax.add_patch(Rectangle((j - .47, i - .45), .94, .9, fill=False, ec="#111111", lw=2.6, zorder=5))
    ax.set_xticks(range(3)); ax.set_xticklabels(CH); ax.set_yticks(range(len(rowsH))); ax.set_yticklabels(rowsH); ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=.8); cb.set_label("Cohen's dz", color=INK2); cb.outline.set_visible(False)
    cb.ax.text(1.5, 1.03, "Arithmetic\nhigher", transform=cb.ax.transAxes, color="#b92f2e", fontsize=8.5, fontweight="bold", ha="center", va="bottom")
    cb.ax.text(1.5, -.03, "Normal (rest)\nhigher", transform=cb.ax.transAxes, color="#184f95", fontsize=8.5, fontweight="bold", ha="center", va="top")
    ax.set_title("Effect size map — black boxes = significant (FDR q < .05)")
    fig.tight_layout(); out["figE_heatmap_marked.png"] = fig

    # ---------- F. classification ----------
    clf = J["classification"]; names = list(clf)
    fig, ax = plt.subplots(figsize=(8.5, 4))
    acc = [clf[n]["acc"] * 100 for n in names]; n95 = [clf[n]["null_95"] * 100 for n in names]
    y = np.arange(len(names))
    ax.barh(y, acc, color=C_ARI, height=.5, zorder=3)
    ax.axvline(50, color=INK2, ls="--", lw=1.4)
    for i, (v, n9) in enumerate(zip(acc, n95)):
        ax.plot([n9, n9], [i - .32, i + .32], color=INK, lw=2, zorder=4)
        ax.text(v + 1, i, f"{v:.1f}%  (p = {clf[names[i]]['p_perm']:.3f})", va="center", fontsize=9.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(names); ax.set_xlim(0, 100); ax.invert_yaxis(); ax.set_xlabel("Accuracy (%)")
    callout(ax, "Chance = 50%", (50, 2.45), (30, 2.45), INK2)
    callout(ax, "95% of shuffled-label\nruns fall below this line", (n95[1], .72), (84, 1.5), INK)
    ax.set_title("Classifier separating Normal (rest) from Arithmetic, 4-s windows", loc="left")
    fig.tight_layout(); out["figF_classification_marked.png"] = fig

    for k, fg in out.items(): fg.savefig(os.path.join(a.out, k)); plt.close(fg)
    print("saved", list(out))


if __name__ == "__main__":
    main()
