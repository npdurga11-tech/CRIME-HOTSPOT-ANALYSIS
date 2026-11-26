"""Spatial clustering of incidents into hotspots with DBSCAN.

Design notes
------------
* Coordinates are projected to local metres so ``eps`` has a physical meaning (a Chicago city
  block is roughly 200 m).
* Incidents are snapped to a 25 m grid and clustered as weighted points. This is mathematically
  the same as DBSCAN on the duplicated points (``sample_weight`` counts toward ``min_samples``)
  but keeps memory small when thousands of incidents pile onto a few street corners.
* ``min_samples`` is expressed relative to the citywide average density, so "a hotspot" means
  "an eps-radius neighbourhood holding at least N times the average number of incidents".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

EARTH_M = 6_371_008.8
SNAP_M = 25.0
FOOTPRINT_CELL_M = 100.0


class LocalProjection:
    """Equirectangular projection around a reference point; fine at city scale (<1 m error/km)."""

    def __init__(self, lat0: float, lon0: float):
        self.lat0, self.lon0 = lat0, lon0

    def forward(self, lat, lon):
        x = np.radians(np.asarray(lon) - self.lon0) * EARTH_M * np.cos(np.radians(self.lat0))
        y = np.radians(np.asarray(lat) - self.lat0) * EARTH_M
        return x, y

    def inverse(self, x, y):
        lat = self.lat0 + np.degrees(np.asarray(y) / EARTH_M)
        lon = self.lon0 + np.degrees(np.asarray(x) / (EARTH_M * np.cos(np.radians(self.lat0))))
        return lat, lon


def snap_to_grid(xy: np.ndarray, cell: float = SNAP_M):
    """Return (unique cell centres, counts, inverse index mapping incidents -> cells)."""
    keys = np.floor(xy / cell).astype(np.int64)
    uniq, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    centres = (uniq + 0.5) * cell
    return centres, counts.astype(float), inverse.ravel()


def occupied_area_km2(xy: np.ndarray, cell: float = FOOTPRINT_CELL_M) -> float:
    """Area of the grid cells that contain at least one incident (a robust 'footprint')."""
    keys = np.unique(np.floor(xy / cell).astype(np.int64), axis=0)
    return len(keys) * (cell / 1000.0) ** 2


def auto_min_samples(xy: np.ndarray, eps_m: float, multiple: float) -> int:
    """min_samples = multiple x (expected incidents in an eps-disc at average density)."""
    density = len(xy) / occupied_area_km2(xy)               # incidents per km^2
    expected = density * np.pi * (eps_m / 1000.0) ** 2
    return int(max(10, round(multiple * expected)))


def k_distance_curve(xy: np.ndarray, k: int, sample: int = 40_000, seed: int = 0) -> np.ndarray:
    """Sorted distance to the k-th nearest neighbour (classic DBSCAN eps diagnostic)."""
    rng = np.random.default_rng(seed)
    pts = xy if len(xy) <= sample else xy[rng.choice(len(xy), sample, replace=False)]
    # Scale k to the subsample so the curve is comparable with the full-data neighbourhood size.
    k_eff = max(2, int(round(k * len(pts) / len(xy))))
    d, _ = NearestNeighbors(n_neighbors=k_eff + 1).fit(pts).kneighbors(pts)
    return np.sort(d[:, -1])


def run_dbscan(xy: np.ndarray, eps_m: float, min_samples: int) -> np.ndarray:
    """Cluster incidents; returns one label per incident (-1 = noise)."""
    centres, counts, inverse = snap_to_grid(xy)
    labels_cell = DBSCAN(eps=eps_m, min_samples=min_samples, algorithm="kd_tree").fit(
        centres, sample_weight=counts).labels_
    return labels_cell[inverse]


def sensitivity_grid(xy: np.ndarray, eps_values=(150, 250, 350), multiples=(3, 4, 6)) -> pd.DataFrame:
    """How much do the headline numbers move with the two DBSCAN parameters?"""
    total_area = occupied_area_km2(xy)
    rows = []
    for eps in eps_values:
        for m in multiples:
            ms = auto_min_samples(xy, eps, m)
            lab = run_dbscan(xy, eps, ms)
            in_cl = lab >= 0
            area = sum(occupied_area_km2(xy[lab == c]) for c in np.unique(lab[in_cl]))
            rows.append({"eps_m": eps, "density_multiple": m, "min_samples": ms,
                         "n_hotspots": int(len(np.unique(lab[in_cl]))),
                         "pct_incidents_in_hotspots": round(100 * in_cl.mean(), 1),
                         "pct_area_in_hotspots": round(100 * area / total_area, 1)})
    return pd.DataFrame(rows)


def _best_window(hourly: np.ndarray, width: int = 6):
    """Contiguous (circular) window of `width` hours holding the largest share of the total."""
    sums = [sum(hourly[(s + i) % 24] for i in range(width)) for s in range(24)]
    s = int(np.argmax(sums))
    return s, float(sums[s] / hourly.sum()) if hourly.sum() else np.nan


def summarise_hotspots(df_geo: pd.DataFrame, labels: np.ndarray, proj: LocalProjection,
                       xy: np.ndarray, total_area_km2: float) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Rank clusters by severity-weighted volume and describe each one.

    Returns (summary table, per-incident rank array with 0 = not in a hotspot, polygon dict).
    """
    work = df_geo.assign(_label=labels, _x=xy[:, 0], _y=xy[:, 1])
    in_cl = work[work["_label"] >= 0]
    cutoff = work["timestamp"].max()
    last12_start = (cutoff - pd.DateOffset(months=12)).normalize()
    prev12_start = (cutoff - pd.DateOffset(months=24)).normalize()

    rows, hulls = [], {}
    for lab, g in in_cl.groupby("_label"):
        pts = g[["_x", "_y"]].to_numpy()
        hourly = np.bincount(g["hour"], weights=g["severity"], minlength=24)
        start_h, win_share = _best_window(hourly)
        top_types = g["primary_type"].value_counts(normalize=True).head(3)
        last12 = int((g["timestamp"] > last12_start).sum())
        prev12 = int(((g["timestamp"] > prev12_start) & (g["timestamp"] <= last12_start)).sum())
        cx, cy = pts.mean(axis=0)
        clat, clon = proj.inverse(cx, cy)
        footprint = occupied_area_km2(pts)
        hull_poly = None
        if len(pts) >= 3:
            try:
                h = ConvexHull(pts)
                hull_poly = pts[h.vertices]
            except Exception:  # degenerate (collinear) cluster
                hull_poly = None
        rows.append({
            "label": lab, "incidents": len(g), "risk_score": float(g["severity"].sum()),
            "violent_share_pct": round(100 * (g["group"] == "violent").mean(), 1),
            "centroid_lat": clat, "centroid_lon": clon, "footprint_km2": round(footprint, 3),
            "incidents_per_km2": round(len(g) / footprint) if footprint else np.nan,
            "peak_window": f"{start_h:02d}:00-{(start_h + 6) % 24:02d}:00",
            "peak_window_share_pct": round(100 * win_share, 1),
            "peak_dow": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][int(g["dow"].value_counts().idxmax())],
            "top_types": ", ".join(f"{t.title()} {v:.0%}" for t, v in top_types.items()),
            "last12_incidents": last12, "prev12_incidents": prev12,
            "yoy_change_pct": round(100 * (last12 - prev12) / prev12, 1) if prev12 >= 30 else np.nan,
        })
        hulls[lab] = hull_poly

    summ = pd.DataFrame(rows).sort_values("risk_score", ascending=False).reset_index(drop=True)
    summ.insert(0, "rank", np.arange(1, len(summ) + 1))
    summ["share_of_all_risk_pct"] = (100 * summ["risk_score"] / df_geo["severity"].sum()).round(1)

    label_to_rank = dict(zip(summ["label"], summ["rank"]))
    rank_arr = np.array([label_to_rank.get(l, 0) for l in labels], dtype=int)
    polygons = {int(label_to_rank[l]): (proj.inverse(h[:, 0], h[:, 1]) if h is not None else None)
                for l, h in hulls.items()}
    return summ, rank_arr, polygons
