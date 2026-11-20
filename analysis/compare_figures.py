import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

C_REST, C_ARI = "#2a78d6", "#eb6834"
C_S = {"sub-01": "#1baf7a", "sub-02": "#4a3aa7"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8983", "#e6e5e0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": .6, "axes.axisbelow": True, "axes.titleweight": "bold", "axes.titlesize": 11.5,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white"})
J = json.load(open("out/summary.json"))
subs = ["sub-01", "sub-02"]
B = {s: pd.read_csv(f"out/{s}_blocks.csv") for s in subs}
S = {s: pd.read_csv(f"out/{s}_stats.csv").set_index("feature") for s in subs}

# ---- F1 blink rate
fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True)
for ax, s in zip(axs, subs):
    b = B[s]; r = b[b.cls == "rest"].set_index("block").blink_rate.sort_index(); a = b[b.cls == "arithmetic"].set_index("block").blink_rate.sort_index()
    for k in r.index: ax.plot([0, 1], [r[k], a[k]], color="#d3d2cc", lw=1, zorder=1)
    ax.bar([0, 1], [r.mean(), a.mean()], color=[C_REST, C_ARI], alpha=.22, width=.6, zorder=0)
    ax.scatter(np.zeros(len(r)), r, color=C_REST, s=30, ec="white", zorder=3); ax.scatter(np.ones(len(a)), a, color=C_ARI, s=30, ec="white", zorder=3)
    st = S[s].loc["blink_rate"]
    ax.set_title(f"{s.replace('sub-', 'Subject ')}\n{r.mean():.0f} → {a.mean():.0f} per min,  p = {st.p:.2g},  {int(st.arith_gt)}/10 pairs higher",
                 fontsize=10.5, color=INK if st.p < .05 else INK2)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Rest", "Arithmetic"]); ax.set_xlim(-.6, 1.6); ax.grid(axis="x", visible=False)
axs[0].set_ylabel("Blink-type events per minute (TP9)")
fig.tight_layout(); fig.savefig("out/cmp1_blink.png"); plt.close(fig)

# ---- F2 effect sizes before / after
rows = [("TP9 Delta (log abs)", "TP9 delta power"), ("TP9 Theta (log abs)", "TP9 theta power"), ("TP9 Alpha (rel %)", "TP9 alpha share"),
        ("TP9 Beta (rel %)", "TP9 beta share"), ("AF7 Theta (rel %)", "AF7 theta share"), ("AF7 Alpha (rel %)", "AF7 alpha share"),
        ("AF7 Beta (rel %)", "AF7 beta share")]
fig, axs = plt.subplots(1, 2, figsize=(11.5, 5), sharey=True)
y = np.arange(len(rows))
for ax, pre, ttl in zip(axs, ["", "CLEAN "], ["All windows (blinks included)", "Blink-free windows only"]):
    for j, s in enumerate(subs):
        dz = [S[s].loc[pre + k, "dz"] for k, _ in rows]; q = [S[s].loc[pre + k, "q"] for k, _ in rows]
        off = (-.19 if j == 0 else .19)
        ax.barh(y + off, dz, height=.34, color=C_S[s], zorder=3)
        for i in range(len(rows)):
            ax.text(dz[i] + (.06 if dz[i] >= 0 else -.06), i + off, f"{dz[i]:+.2f}{' *' if q[i] < .05 else ''}", va="center",
                    ha="left" if dz[i] >= 0 else "right", fontsize=8.5, color=INK if q[i] < .05 else INK2,
                    fontweight="bold" if q[i] < .05 else "normal")
    ax.axvline(0, color=INK2, lw=1); ax.set_xlim(-2.4, 2.6); ax.set_title(ttl); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Cohen's dz  (positive = higher in arithmetic)")
axs[0].set_yticks(y); axs[0].set_yticklabels([n for _, n in rows]); axs[0].invert_yaxis()
fig.legend(handles=[Patch(color=C_S[s], label=s.replace("sub-", "Subject ")) for s in subs] +
           [Patch(color="none", label="* FDR q < .05")], loc="upper center", ncol=3, bbox_to_anchor=(.5, 1.04))
fig.tight_layout(); fig.savefig("out/cmp2_effects.png"); plt.close(fig)

# ---- F3 classification
cr = J["cross_subject"]
labels = ["Within\nSubject 01", "Within\nSubject 02", "Train 01 →\ntest 02", "Train 02 →\ntest 01"]
allw = [J["sub-01"]["clf_within_all"], J["sub-02"]["clf_within_all"], cr["sub-01->sub-02"]["all_windows"], cr["sub-02->sub-01"]["all_windows"]]
cln = [J["sub-01"]["clf_within_blinkfree"], J["sub-02"]["clf_within_blinkfree"], cr["sub-01->sub-02"]["blink_free"], cr["sub-02->sub-01"]["blink_free"]]
pa = [J["sub-01"]["p_within_all"], J["sub-02"]["p_within_all"], cr["sub-01->sub-02"]["p_all_windows"], cr["sub-02->sub-01"]["p_all_windows"]]
pc = [J["sub-01"]["p_within_blinkfree"], J["sub-02"]["p_within_blinkfree"], cr["sub-01->sub-02"]["p_blink_free"], cr["sub-02->sub-01"]["p_blink_free"]]
fig, ax = plt.subplots(figsize=(9.6, 4.4)); x = np.arange(4)
ax.bar(x - .2, np.array(allw) * 100, .38, color="#9a9990", label="All windows (blinks included)", zorder=3)
ax.bar(x + .2, np.array(cln) * 100, .38, color=C_ARI, label="Blink-free windows only", zorder=3)
for i in range(4):
    ax.text(i - .2, allw[i] * 100 + 1, f"{allw[i] * 100:.0f}%\np={pa[i]:.3f}", ha="center", fontsize=8.5, color=INK)
    ax.text(i + .2, cln[i] * 100 + 1, f"{cln[i] * 100:.0f}%\np={pc[i]:.2f}", ha="center", fontsize=8.5, color=INK2)
ax.axhline(50, color=INK2, ls="--", lw=1.2); ax.text(3.62, 51, "chance", fontsize=9, color=INK2, ha="right")
ax.set_ylim(0, 88); ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("Balanced accuracy (%)"); ax.grid(axis="x", visible=False)
ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(.5, 1.13))
fig.tight_layout(); fig.savefig("out/cmp3_classification.png"); plt.close(fig)
print("ok")
