import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as mt

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "legend.frameon": False,
        "axes.titleweight": "bold",
        "axes.titlesize": 9,
        "pdf.fonttype": 42,
    }
)
STYLE = {
    "MLP": dict(color="#8c8c8c", ls="-", lw=1.0, label="MLP, raw inputs"),
    "CNN": dict(color="#6aa6dc", ls="--", lw=1.2, label="CNN, raw inputs"),
    "DahuNet_cnn_5f": dict(color="#1b4f99", ls="-", lw=1.7, label="DahuNet-CNN (enriched inputs)"),
    "FNO2": dict(color="#f2a65a", ls="--", lw=1.2, label="FNO, raw inputs"),
    "DahuNet_fno_5f": dict(color="#c2410c", ls="-", lw=1.7, label="DahuNet-FNO (enriched inputs)"),
}
ORDER = ["MLP", "CNN", "DahuNet_cnn_5f", "FNO2", "DahuNet_fno_5f"]
CASES = [("aletsch", "Aletsch"), ("valais", "Valais"), ("jakobshavn", "Sermeq Kujalleq")]


def nominor(ax):
    ax.xaxis.set_minor_formatter(mt.NullFormatter())
    ax.yaxis.set_minor_formatter(mt.NullFormatter())
