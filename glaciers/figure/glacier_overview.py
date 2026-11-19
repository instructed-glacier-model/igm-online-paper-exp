"""Original Figure 2 rendering, adapted from Thomas Gregov's DahuNet script.
Only data loading is separated; styling and layout retain the published code.
Distributed under the repository's GPL-3.0-or-later license.
"""
from __future__ import annotations
import os
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.patheffects as mpe
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter

VMIN = 1.0
VMAX = 1000.0
CMAP = "turbo"
MARGIN_KM = 1.5
TOPO_STEP = {"aletsch": 500, "valais": 500, "jakob": 100}
TOPO_CLR = "#7a746e"
TOPO_LW = 0.25
TOPO_ALPHA = 0.4
STREAM_SUB = {"aletsch": 2, "valais": 5, "jakob": 2}
STREAM_DENSITY = {"aletsch": 2.0, "valais": 1.5, "jakob": 1.5}
STREAM_LW = 0.65
STREAM_ARROW = 0.9
STREAM_MIN_SPD = {"aletsch": 2.0, "valais": 2.0, "jakob": 1e-05}
SCALE_SIDE = {"aletsch": "left", "valais": "right", "jakob": "right"}
BG_LAND = "#cdc9c3"
BG_OCEAN = "#7aafc4"
OUTLINE_CLR = "black"
OUTLINE_LW = 0.65
PANEL_LABELS = ["(a)", "(b)", "(c)"]
LABEL_X = [0.03, 0.22, 0.97]
LABEL_HA = ["left", "left", "right"]
GLACIER_NAMES = ["Aletsch", "Valais", "Sermeq Kujalleq"]
COUNTRIES = ["Switzerland", "Switzerland", "Greenland"]
SCALE_KM = 5.0


def ice_bbox(data: dict) -> tuple[float, float, float, float]:
    mask = data["ice_mask"]
    x, y = (data["x_km"], data["y_km"])
    cols = np.any(mask, axis=0)
    rows = np.any(mask, axis=1)
    x0 = max(x[cols][0] - MARGIN_KM, x[0])
    x1 = min(x[cols][-1] + MARGIN_KM, x[-1])
    y0 = max(y[rows][0] - MARGIN_KM, y[0])
    y1 = min(y[rows][-1] + MARGIN_KM, y[-1])
    return (x0, x1, y0, y1)


def add_scale_bar(ax, length_km: float, side: str = "right", y_frac: float = 0.07):
    """Minimal scale bar: plain black line + end ticks + black label."""
    xl, yl = (ax.get_xlim(), ax.get_ylim())
    xr, yr = (xl[1] - xl[0], yl[1] - yl[0])
    margin = 0.06 * xr
    if side == "right":
        xe = xl[1] - margin
        xs = xe - length_km
    else:
        xs = xl[0] + margin
        xe = xs + length_km
    cx = (xs + xe) / 2
    yp = yl[0] + y_frac * yr
    th = 0.012 * yr
    kw = dict(color="black", lw=1.5, solid_capstyle="butt", zorder=10)
    ax.plot([xs, xe], [yp, yp], **kw)
    ax.plot([xs, xs], [yp - th, yp + th], **kw)
    ax.plot([xe, xe], [yp - th, yp + th], **kw)
    ax.text(
        cx,
        yp + 0.022 * yr,
        f"{int(length_km)} km",
        ha="center",
        va="bottom",
        fontsize=8,
        color="black",
        zorder=11,
    )


def make_figure(out_dir: str, datasets):
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica Neue", "DejaVu Sans"],
            "font.size": 11,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    keys = ["aletsch", "valais", "jakob"]
    datasets = [datasets[k] for k in keys]
    extents = [ice_bbox(d) for d in datasets]
    dx_list = [e[1] - e[0] for e in extents]
    dy_list = [e[3] - e[2] for e in extents]
    aspects = [w / h for w, h in zip(dx_list, dy_list)]
    norm = LogNorm(vmin=VMIN, vmax=VMAX)
    sm = mpl.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    fig_w = 7.2
    fig_h = fig_w * 0.56
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(
        1,
        3,
        figure=fig,
        width_ratios=aspects,
        wspace=0.06,
        left=0.01,
        right=0.88,
        top=0.8,
        bottom=0.2,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for ax, data, extent, lbl, name, country, lbl_x, lbl_ha in zip(
        axes, datasets, extents, PANEL_LABELS, GLACIER_NAMES, COUNTRIES, LABEL_X, LABEL_HA
    ):
        x_km, y_km = (data["x_km"], data["y_km"])
        x0, x1, y0, y1 = extent
        xi = (x_km >= x0) & (x_km <= x1)
        yi = (y_km >= y0) & (y_km <= y1)
        xc, yc = (x_km[xi], y_km[yi])
        ext_img = [xc[0], xc[-1], yc[0], yc[-1]]
        XX, YY = np.meshgrid(xc, yc)
        ice_c = data["ice_mask"][np.ix_(yi, xi)]
        ocean_c = data["ocean_mask"][np.ix_(yi, xi)]
        vel_c = data["vel_ice"][np.ix_(yi, xi)]
        land_rgba = np.full((*ice_c.shape, 4), 1.0)
        land_rgba[..., :3] = mpl.colors.to_rgb(BG_LAND)
        land_rgba[ice_c, 3] = 0.0
        ax.imshow(
            land_rgba,
            origin="lower",
            extent=ext_img,
            aspect="equal",
            interpolation="nearest",
            zorder=1,
        )
        if ocean_c.any():
            oc = np.zeros((*ice_c.shape, 4))
            oc[ocean_c, :3] = mpl.colors.to_rgb(BG_OCEAN)
            oc[ocean_c, 3] = 1.0
            ax.imshow(
                oc,
                origin="lower",
                extent=ext_img,
                aspect="equal",
                interpolation="nearest",
                zorder=2,
            )
        if data["topg"] is not None:
            topg_c = data["topg"][np.ix_(yi, xi)]
            hide = ice_c | ocean_c
            land_cells = topg_c[~hide]
            if land_cells.size > 0:
                z_min = max(0.0, float(land_cells.min()))
                z_max = float(land_cells.max())
                step = TOPO_STEP[data["key"]]
                levels = np.arange(
                    np.ceil(z_min / step) * step, np.floor(z_max / step) * step + 1, step
                )
                if len(levels) > 0:
                    topo_ma = ma.masked_where(hide, topg_c)
                    ax.contour(
                        XX,
                        YY,
                        topo_ma,
                        levels=levels,
                        colors=TOPO_CLR,
                        linewidths=TOPO_LW,
                        alpha=TOPO_ALPHA,
                        zorder=3,
                    )
        ax.imshow(
            vel_c,
            origin="lower",
            extent=ext_img,
            cmap=CMAP,
            norm=norm,
            aspect="equal",
            interpolation="bilinear",
            zorder=4,
        )
        ax.contour(
            XX,
            YY,
            ice_c.astype(float),
            levels=[0.5],
            colors=OUTLINE_CLR,
            linewidths=OUTLINE_LW,
            zorder=5,
        )
        if data["u"] is not None:
            u_c = data["u"][np.ix_(yi, xi)]
            v_c = data["v"][np.ix_(yi, xi)]
            spd = np.sqrt(u_c**2 + v_c**2)
            thr = STREAM_MIN_SPD[data["key"]]
            valid = (spd > thr) & ice_c
            u_n = np.where(valid, u_c / np.where(spd > 0, spd, 1.0), 0.0)
            v_n = np.where(valid, v_c / np.where(spd > 0, spd, 1.0), 0.0)
            sub = STREAM_SUB[data["key"]]
            ns_x = max(2, len(xc) // sub)
            ns_y = max(2, len(yc) // sub)
            xs = np.linspace(xc[0], xc[-1], ns_x)
            ys = np.linspace(yc[0], yc[-1], ns_y)
            xi_s = np.round(np.linspace(0, len(xc) - 1, ns_x)).astype(int)
            yi_s = np.round(np.linspace(0, len(yc) - 1, ns_y)).astype(int)
            u_s = u_n[np.ix_(yi_s, xi_s)]
            v_s = v_n[np.ix_(yi_s, xi_s)]
            strm = ax.streamplot(
                xs,
                ys,
                u_s,
                v_s,
                density=STREAM_DENSITY[data["key"]],
                color="white",
                linewidth=STREAM_LW,
                arrowsize=STREAM_ARROW,
                zorder=7,
            )
            strm.arrows.set_visible(False)
            for p in list(ax.patches):
                if isinstance(p, mpatches.FancyArrowPatch):
                    p.remove()
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
        add_scale_bar(ax, SCALE_KM, side=SCALE_SIDE[data["key"]])
        ax.text(
            lbl_x,
            0.97,
            lbl,
            transform=ax.transAxes,
            ha=lbl_ha,
            va="top",
            fontsize=9,
            fontweight="bold",
            color="black",
            zorder=8,
        )
        ax.text(
            0.5,
            1.13,
            name,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            clip_on=False,
        )
        ax.text(
            0.5,
            1.03,
            f"({country})",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=8,
            clip_on=False,
        )
        grid_info = f"{data['nx']} × {data['ny']},  Δx = {int(data['dx_m'])} m"
        ax.text(
            0.5,
            -0.08,
            grid_info,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7,
            color="0.35",
            clip_on=False,
        )
    fig.canvas.draw()
    pos = axes[-1].get_position()
    cbar_ax = fig.add_axes([0.91, pos.y0, 0.016, pos.height])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="vertical", extend="max")
    cbar.set_label("Depth-averaged speed (m yr$^{-1}$)", fontsize=8, labelpad=5)
    cbar.ax.tick_params(labelsize=8, which="both", length=3)
    cbar.set_ticks([1, 5, 10, 50, 100, 500, 1000])
    cbar.set_ticklabels(["1", "5", "10", "50", "100", "500", "1000"])
    for sp in cbar.ax.spines.values():
        sp.set_linewidth(0.5)
    os.makedirs(out_dir, exist_ok=True)
    for ext in ["pdf"]:
        out = os.path.join(out_dir, f"figure4_1_glacier_overview.{ext}")
        fig.savefig(out, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)
