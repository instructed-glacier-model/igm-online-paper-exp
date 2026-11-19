# Variant of fig_searches.py: the top row shows box plots of the uniform search (one box per
# backbone and input, all 50 trials overlaid) instead of empirical distribution functions.
import pandas as pd, numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from plotting import *
from pathlib import Path

HERE = Path(__file__).resolve().parent
t = pd.read_csv(HERE / "data/trials.csv")
t["median_only_qualifies"] = t["qualified"]
ylim_t = {"aletsch": (40, 200), "jakobshavn": (110, 400), "valais": (150, 2000)}
yt = {
    "aletsch": [50, 100, 200],
    "jakobshavn": [150, 200, 300, 400],
    "valais": [200, 500, 1000, 2000],
}
SHORT = {
    "MLP": "MLP\nraw",
    "CNN": "CNN\nraw",
    "DahuNet_cnn_5f": "CNN\nenriched",
    "FNO2": "FNO\nraw",
    "DahuNet_fno_5f": "FNO\nenriched",
}
rng = np.random.default_rng(0)
fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.7))
for j, (case, title) in enumerate(CASES):
    axb = axes[0, j]
    axp = axes[1, j]
    d = t[t.case == case]
    axb.set_title(title, loc="left")
    axb.set_yscale("log")
    axb.set_ylim(1, 330)
    axb.set_yticks([1, 5, 10, 100])
    axb.set_yticklabels(["1", "5", "10", "100"])
    axb.axhline(5, color="k", lw=0.6, ls=":")
    for i, a in enumerate(ORDER):
        s = STYLE[a]
        u = d[(d.architecture == a) & (d.sampler == "uniform")].worst_speed_median_percent.values
        finite = u[np.isfinite(u)]
        display = np.where(np.isfinite(u), u, 1e6)
        if len(u) == 0:
            continue
        filled = s["ls"] == "-" and a != "MLP"
        bp = axb.boxplot(
            [display],
            positions=[i],
            widths=0.55,
            whis=(5, 95),
            showfliers=False,
            patch_artist=True,
            medianprops=dict(color="k", lw=1.1),
            boxprops=dict(
                facecolor=s["color"] if filled else "white",
                edgecolor=s["color"],
                lw=1.0,
                alpha=0.55 if filled else 1.0,
            ),
            whiskerprops=dict(color=s["color"], lw=0.9),
            capprops=dict(color=s["color"], lw=0.9),
        )
        axb.scatter(
            i + rng.uniform(-0.2, 0.2, len(finite)),
            finite,
            s=3.5,
            color=s["color"],
            alpha=0.75,
            lw=0,
            zorder=3,
        )
        nq = int((u < 5).sum())
        axb.text(i, 195, str(nq), ha="center", va="bottom", fontsize=7, color="k", zorder=8)
    axb.set_xticks(range(len(ORDER)))
    axb.set_xticklabels([SHORT[a] for a in ORDER], fontsize=6.5)
    axb.set_xlim(-0.6, len(ORDER) - 0.4)
    axb.axhspan(160, 330, color="white", zorder=6)
    axb.axhline(160, color="0.6", lw=0.5, zorder=7)
    axb.yaxis.set_minor_formatter(mt.NullFormatter())
    # bottom row: unchanged
    axp.set_xscale("log")
    axp.set_yscale("log")
    axp.set_ylim(*ylim_t[case])
    axp.set_xlim(1, 100)
    axp.set_xticks([1, 5, 10, 100])
    axp.set_xticklabels(["1", "5", "10", "100"])
    axp.set_yticks(yt[case])
    axp.set_yticklabels([str(v) for v in yt[case]])
    nominor(axp)
    axp.axvline(5, color="k", lw=0.6, ls=":")
    for a in ORDER:
        p = d[
            (d.architecture == a)
            & (d.sampler == "pareto")
            & np.isfinite(d.worst_speed_median_percent)
            & np.isfinite(d.total_seconds)
        ].sort_values("total_seconds")
        s = STYLE[a]
        if p.empty:
            continue
        q = p[p.median_only_qualifies]
        if a != "MLP":
            axp.scatter(
                q.worst_speed_median_percent,
                q.total_seconds,
                s=3,
                color=s["color"],
                alpha=0.35,
                lw=0,
                zorder=2,
                rasterized=True,
            )
        pe = p.sort_values("worst_speed_median_percent")
        x = pe.worst_speed_median_percent.values
        y = pe.total_seconds.values
        env = np.minimum.accumulate(y)
        keep = np.r_[True, env[1:] < env[:-1]]
        xe = np.r_[x[keep], 100.0]
        ye = np.r_[env[keep], env[keep][-1]]
        axp.step(xe, ye, where="post", color=s["color"], ls=s["ls"], lw=s["lw"], zorder=3)
        if len(q) >= 10:
            t10 = q.sort_values("total_seconds").total_seconds.iloc[9]
            axp.plot(
                [5],
                [t10],
                marker="<",
                color=s["color"],
                mec="white",
                mew=0.4,
                ms=5,
                zorder=5,
                clip_on=False,
            )
    if j == 0:
        axb.set_ylabel("Worst-frame speed error (%)\nuniform search")
        axp.set_ylabel("Total simulation time (s)\nNSGA-II search")
    axp.set_xlabel("Worst-frame speed error (%)")


def _h(a):
    return Line2D(
        [0],
        [0],
        color=STYLE[a]["color"],
        ls=STYLE[a]["ls"],
        lw=STYLE[a]["lw"],
        label=STYLE[a]["label"],
    )


blank = Line2D([], [], color="none", label=" ")
# three columns (matplotlib fills the legend column by column): raw inputs | enriched inputs | markers
handles = [
    _h("CNN"),
    _h("FNO2"),
    _h("MLP"),
    _h("DahuNet_cnn_5f"),
    _h("DahuNet_fno_5f"),
    blank,
    Line2D([0], [0], color="k", ls=":", lw=0.6, label="5% criterion"),
    Line2D([0], [0], marker="<", color="k", lw=0, ms=5, label="tenth-cheapest qualifying trial"),
    blank,
]
fig.legend(
    handles=handles,
    loc="lower center",
    ncol=3,
    fontsize=6.8,
    bbox_to_anchor=(0.5, -0.005),
    handlelength=2.2,
    columnspacing=2.4,
    labelspacing=0.35,
)
fig.tight_layout(rect=(0, 0.115, 1, 1))
fig.savefig(HERE / "figure5_1_searches.pdf", dpi=300)

import sys

sys.path.insert(0, str(HERE.parents[1] / "_shared"))
from tables import render

render(HERE)
