import pandas as pd, numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from plotting import *
from pathlib import Path

HERE = Path(__file__).resolve().parent
import matplotlib.colors as mc

root = HERE / "data"


def tints(base):
    b = np.array(mc.to_rgb(base))
    w = np.ones(3)
    return {
        "raw": tuple(0.45 * b + 0.55 * w),
        "3f": tuple(0.75 * b + 0.25 * w),
        "5f": tuple(b),
        "scrambled": "#8c8c8c",
    }


inputs = [
    ("raw", "raw inputs", dict(ls="--", lw=1.2)),
    ("3f", "three slope features", dict(ls="-", lw=1.2)),
    ("5f", "five features", dict(ls="-", lw=1.8)),
    ("scrambled", "scrambled features", dict(ls="-.", lw=1.2)),
]
c = pd.read_csv(root / "correction_curves.csv")
c["input"] = c["input"].map(
    {"raw": "raw", "three_features": "3f", "five_features": "5f", "scrambled": "scrambled"}
)
c = c.rename(columns={"speed_median_percent": "err"})
for case in ["aletsch", "valais", "jakobshavn"]:
    subset = c[c.case == case]
    assert len(subset) == 1836 and subset.groupby(["input", "index", "seed"]).size().eq(51).all()
panels = [
    ("Aletsch, CNN", c[c.case == "aletsch"], "#1b4f99", (0.8, 8), [1, 2, 4, 8], (0, 5)),
    ("Valais, CNN", c[c.case == "valais"], "#1b4f99", (0.8, 8), [1, 2, 4, 8], (0, 7)),
    (
        "Sermeq Kujalleq, FNO",
        c[c.case == "jakobshavn"],
        "#c2410c",
        (0.1, 1.2),
        [0.1, 0.2, 0.5, 1],
        (0, 1.2),
    ),
]
fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.4))
for j, (title, c, base, ylim, yt, ylim2) in enumerate(panels):
    C = tints(base)
    ax = axes[0, j]
    ax.set_title(title, loc="left", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ax.set_yticks(yt)
    ax.set_yticklabels([str(v) for v in yt])
    nominor(ax)
    ax.set_xlabel("Adam updates at the new state")
    if j == 0:
        ax.set_ylabel("Median speed error (%)")
    ax2 = axes[1, j]
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(["raw", "3 feat.", "5 feat.", "scr."], fontsize=7)
    ax2.set_ylim(*ylim2)
    ax2.set_xlim(-0.6, 3.6)
    if j == 0:
        ax2.set_ylabel("Median speed error (%)")
    ax2.set_title("Before and after 50 updates", loc="left", fontsize=7.5, fontweight="normal")
    for key, lab, st in inputs:
        d = c[c.input == key]
        g = d.groupby("step").err
        med = g.median()
        q1 = g.quantile(0.25)
        q3 = g.quantile(0.75)
        ax.plot(med.index, med.values, color=C[key], label=lab, **st)
        ax.fill_between(med.index, q1.values, q3.values, color=C[key], alpha=0.15, lw=0)
    if j == 0:
        ax.legend(fontsize=6.3, loc="upper right", handlelength=2.2)
    for k, (key, lab, st) in enumerate(inputs):
        d = c[c.input == key]
        piv = d.pivot_table(index=["index", "seed"], columns="step", values="err")
        for _, r in piv.iterrows():
            ax2.plot([k - 0.18, k + 0.18], [r[0], r[50]], color=C[key], lw=0.7, alpha=0.6)
        ax2.plot(
            [k - 0.18, k + 0.18],
            [piv[0].median(), piv[50].median()],
            color="k",
            lw=1.6,
            marker="o",
            ms=3,
        )
fig.tight_layout()
fig.savefig(HERE / "figure5_2_input_ablation.pdf", dpi=300)

import sys

sys.path.insert(0, str(HERE.parents[1] / "_shared"))
from tables import render

render(HERE)
