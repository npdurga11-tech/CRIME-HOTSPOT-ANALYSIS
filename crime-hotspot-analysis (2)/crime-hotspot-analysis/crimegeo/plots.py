"""Static figures (PNG). Conventions: sequential blue for magnitude, blue<->red diverging for signed
correlation, categorical slots 1-2 for two series, one accent (orange) for emphasis, recessive grid,
2 px lines, text in ink colours (never series colours), light surface."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, PowerNorm  # noqa: E402

from .config import PALETTE as P  # noqa: E402
from .config import SOCIO_LABELS  # noqa: E402
from .temporal import DAYS, MONTHS  # noqa: E402

SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", P["seq"])
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10, "axes.edgecolor": P["axis"], "axes.labelcolor": P["ink2"],
    "xtick.color": P["muted"], "ytick.color": P["muted"], "text.color": P["ink"],
    "axes.facecolor": P["surface"], "figure.facecolor": P["surface"], "savefig.facecolor": P["surface"],
})


def _style(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(P["axis"])
    ax.spines["bottom"].set_color(P["axis"])
    if grid_axis:
        ax.grid(axis=grid_axis, color=P["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _title(ax, title, sub=None):
    ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", color=P["ink"], pad=22 if sub else 10)
    if sub:
        ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=9.5, color=P["ink2"], va="bottom")


def _tag(fig, text):
    """Footer note anchored under the lowest-left axes (so it never widens the saved canvas)."""
    if text:
        axes = [a for a in fig.axes if a.get_label() != "<colorbar>"]
        ax = min(axes, key=lambda a: (round(a.get_position().y0, 2), a.get_position().x0))
        ax.annotate(text, xy=(0, 0), xycoords="axes fraction", xytext=(0, -46), textcoords="offset points",
                    fontsize=8, color=P["muted"], ha="left", va="top", annotation_clip=False)


def _badge(ax, x_km, y_km, label, filled=False):
    """Rank badge offset from the cluster with a thin leader line, so it never hides the cluster."""
    ax.annotate(
        label, xy=(x_km, y_km), xytext=(15, 15), textcoords="offset points", ha="center", va="center",
        fontsize=8.5, fontweight="bold", color="#ffffff" if filled else P["ink"], zorder=6,
        arrowprops=dict(arrowstyle="-", color=P["ink2"], lw=0.8, shrinkA=0, shrinkB=0),
        bbox=dict(boxstyle="circle,pad=0.28", fc=P["orange"] if filled else P["surface"],
                  ec=P["surface"] if filled else P["ink2"], lw=1.2))


def _save(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return Path(path)


# ---------------------------------------------------------------------------------------
def fig_kdistance(kd, eps_m, k, path, tag=""):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(np.arange(len(kd)) / len(kd) * 100, kd, color=P["blue"], linewidth=2)
    ax.axhline(eps_m, color=P["muted"], linewidth=1, linestyle=(0, (4, 3)))
    ax.text(1, eps_m, f"  chosen eps = {eps_m:.0f} m", color=P["ink2"], va="bottom", fontsize=9)
    ax.set_xlabel("Incidents, sorted by distance (percentile)")
    ax.set_ylabel(f"Distance to {k}-th nearest neighbour (m)")
    ax.set_ylim(0, min(kd.max(), np.percentile(kd, 99.5) * 1.2))
    _style(ax)
    _title(ax, "How tight is the clustering?", "k-distance curve: a sharp elbow, if there is one, suggests a value for eps")
    _tag(fig, tag)
    return _save(fig, path)


def fig_hotspots(xy, rank, centroids_xy, path, tag=""):
    fig, ax = plt.subplots(figsize=(8, 8))
    noise = rank == 0
    ax.scatter(xy[noise, 0] / 1000, xy[noise, 1] / 1000, s=0.8, color=P["axis"], alpha=0.35, linewidths=0, rasterized=True)
    ax.scatter(xy[~noise, 0] / 1000, xy[~noise, 1] / 1000, s=1.2, color=P["orange"], alpha=0.6, linewidths=0, rasterized=True)
    for r, (cx, cy) in centroids_xy.items():
        if r <= 10:
            _badge(ax, cx / 1000, cy / 1000, str(r))
    ax.set_aspect("equal")
    ax.set_xlabel("km east of the data centroid")
    ax.set_ylabel("km north")
    _style(ax, grid_axis=None)
    n_h = int((~noise).sum())
    _title(ax, "DBSCAN hotspots",
           f"{n_h:,} of {len(xy):,} incidents ({100 * n_h / len(xy):.0f}%) fall inside a hotspot; top 10 numbered by severity-weighted risk")
    _tag(fig, tag)
    return _save(fig, path)


def fig_heatmap(xy, polygons_xy, centroids_xy, path, cell_m=200, tag=""):
    xe = np.arange(np.floor(xy[:, 0].min() / cell_m), np.ceil(xy[:, 0].max() / cell_m) + 1) * cell_m / 1000
    ye = np.arange(np.floor(xy[:, 1].min() / cell_m), np.ceil(xy[:, 1].max() / cell_m) + 1) * cell_m / 1000
    H, _, _ = np.histogram2d(xy[:, 0] / 1000, xy[:, 1] / 1000, bins=[xe, ye])
    H = np.ma.masked_equal(H, 0)
    fig, ax = plt.subplots(figsize=(9, 8))
    m = ax.pcolormesh(xe, ye, H.T, cmap=SEQ_CMAP, norm=PowerNorm(0.5, vmin=1), shading="flat", rasterized=True)
    for r, poly in polygons_xy.items():
        if poly is not None:
            ax.fill(poly[:, 0] / 1000, poly[:, 1] / 1000, facecolor="none", edgecolor=P["orange"], linewidth=1.4)
    for r, (cx, cy) in centroids_xy.items():
        if r <= 10:
            _badge(ax, cx / 1000, cy / 1000, str(r), filled=True)
    cb = fig.colorbar(m, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label(f"Incidents per {cell_m} m cell", color=P["ink2"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, colors=P["muted"])
    ax.set_aspect("equal")
    ax.set_xlabel("km east")
    ax.set_ylabel("km north")
    _style(ax, grid_axis=None)
    _title(ax, "Incident heatmap with DBSCAN hotspot outlines", "darker = more incidents; orange outlines = hotspots, top 10 numbered")
    _tag(fig, tag)
    return _save(fig, path)


def fig_time_series(monthly, dec, by_group, seasonal, path, tag=""):
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 10.5), gridspec_kw={"height_ratios": [3, 3, 2.3], "hspace": 0.55})
    ax = axes[0]
    ax.plot(dec.index, dec["observed"], color=P["blue"], linewidth=2, label="Monthly incidents")
    ax.plot(dec.index, dec["trend"], color=P["orange"], linewidth=2, label="Underlying trend (12-month centred average)")
    ax.set_ylabel("Incidents per month")
    ax.legend(frameon=False, loc="upper right", fontsize=9, labelcolor=P["ink2"])
    _style(ax)
    _title(ax, "Monthly crime volume", "all complete months, with the underlying trend separated from seasonal swings")

    ax = axes[1]
    for col, colr, lab in (("violent", P["blue"], "Violent"), ("property", P["orange"], "Property")):
        if col in by_group:
            ax.plot(by_group.index, by_group[col], color=colr, linewidth=2, label=lab)
            ax.text(by_group.index[-1], by_group[col].iloc[-1], f"  {lab}", color=P["ink2"], va="center", fontsize=9)
    ax.set_ylabel("Incidents per month")
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=P["ink2"], ncol=2)
    ax.set_xlim(right=by_group.index[-1] + pd.Timedelta(days=75))
    _style(ax)
    _title(ax, "Violent vs property crime", "both series on the same axis (incidents per month)")

    ax = axes[2]
    vals = seasonal.to_numpy()
    ax.bar(MONTHS, vals - 1, bottom=1, color=P["blue"], width=0.62)
    ax.axhline(1, color=P["axis"], linewidth=1)
    hi, lo = int(vals.argmax()), int(vals.argmin())
    for i in (hi, lo):
        ax.text(i, vals[i] + (0.012 if i == hi else -0.012), f"{vals[i]:.2f}", ha="center",
                va="bottom" if i == hi else "top", fontsize=9, color=P["ink"])
    ax.set_ylim(min(vals) - 0.07, max(vals) + 0.07)
    ax.set_ylabel("Seasonal index (1.00 = average)")
    _style(ax)
    _title(ax, "Seasonality", f"busiest month: {MONTHS[hi]} ({100 * (vals[hi] - 1):+.0f}% vs average); quietest: {MONTHS[lo]} ({100 * (vals[lo] - 1):+.0f}%)")
    _tag(fig, tag)
    return _save(fig, path)


def fig_hour_dow(matrix, path, tag=""):
    fig, ax = plt.subplots(figsize=(11, 4.4))
    im = ax.imshow(matrix.to_numpy(), aspect="auto", cmap=SEQ_CMAP, interpolation="nearest")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f"{h:02d}" for h in range(0, 24, 2)])
    ax.set_yticks(range(7))
    ax.set_yticklabels(DAYS)
    ax.set_xlabel("Hour of day")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, pad=0.015)
    cb.set_label("% of all incidents", color=P["ink2"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, colors=P["muted"])
    _title(ax, "When does crime happen?", "share of all incidents by weekday and hour; darker = more incidents")
    _tag(fig, tag)
    return _save(fig, path)


def fig_demographics(zones, outcome, outcome_label, corr, std_resid, path, tag=""):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [1, 1.1], "wspace": 0.55})
    ax = axes[0]
    c = corr.sort_values("spearman_rho")
    colors = [P["red"] if v > 0 else P["blue"] for v in c["spearman_rho"]]
    ax.barh(c["indicator"], c["spearman_rho"], color=colors, height=0.62)
    ax.axvline(0, color=P["axis"], linewidth=1)
    for y, (v, pv) in enumerate(zip(c["spearman_rho"], c["p_value"])):
        ax.text(v + (0.03 if v >= 0 else -0.03), y, f"{v:+.2f}" + ("" if pv < 0.05 else " (n.s.)"),
                va="center", ha="left" if v >= 0 else "right", fontsize=9, color=P["ink"])
    ax.set_xlim(-1.25, 1.25)
    ax.set_xlabel("Spearman rank correlation with crime rate")
    _style(ax, grid_axis="x")
    _title(ax, "Which neighbourhood factors travel with crime?", "red = higher crime where indicator is higher; blue = lower")

    ax = axes[1]
    xcol = "poverty_pct" if "poverty_pct" in zones else corr.iloc[0]["column"]
    x, y = zones[xcol].to_numpy(), zones[outcome].to_numpy()
    out = np.abs(std_resid) >= 1.5
    ax.scatter(x[~out], y[~out], s=46, color=P["axis"], edgecolors=P["surface"], linewidths=1.2, zorder=3)
    ax.scatter(x[out], y[out], s=60, color=P["orange"], edgecolors=P["surface"], linewidths=1.2, zorder=4)
    b1, b0 = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, b0 + b1 * xs, color=P["blue"], linewidth=2, zorder=2)
    for i in np.where(out)[0]:
        ax.annotate(str(zones.iloc[i]["zone_name"]), (x[i], y[i]), xytext=(7, 5), textcoords="offset points",
                    fontsize=9, color=P["ink"])
    ax.set_xlabel(SOCIO_LABELS.get(xcol, xcol))
    ax.set_ylabel(outcome_label.split(" (")[0].capitalize())
    _style(ax, grid_axis="both")
    _title(ax, "Zones vs poverty", "orange = zones whose crime is far from what socioeconomics predict (|std. residual| >= 1.5)")
    _tag(fig, tag)
    return _save(fig, path)


def fig_allocation(plan_hotspots, path, top_n=12, tag=""):
    h = plan_hotspots.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    y = np.arange(len(h))
    ax.barh(y + 0.19, h["risk_share_pct"], height=0.36, color=P["blue"], label="Share of hotspot risk")
    ax.barh(y - 0.19, h["directed_share_pct"], height=0.36, color=P["orange"], label="Recommended share of directed patrol hours")
    ax.set_yticks(y)
    ax.set_yticklabels([f"Hotspot #{int(r)}" for r in h["rank"]])
    for yi, a, b in zip(y, h["risk_share_pct"], h["directed_share_pct"]):
        ax.text(a + 0.1, yi + 0.19, f"{a:.1f}%", va="center", fontsize=8.5, color=P["ink2"])
        ax.text(b + 0.1, yi - 0.19, f"{b:.1f}%", va="center", fontsize=8.5, color=P["ink2"])
    ax.set_xlabel("% of total across all hotspots")
    ax.legend(frameon=False, loc="lower right", fontsize=9, labelcolor=P["ink2"])
    _style(ax, grid_axis="x")
    _title(ax, "Proposed split of hotspot-directed patrol hours", "sub-linear weighting (alpha < 1) flattens allocation relative to raw risk")
    _tag(fig, tag)
    return _save(fig, path)
