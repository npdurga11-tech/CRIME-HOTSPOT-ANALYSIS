#!/usr/bin/env python3
"""End-to-end crime hotspot analysis.

    python run_analysis.py                    # synthetic demo data (works offline)
    python run_analysis.py --source chicago   # real City of Chicago open data (needs internet)
    python run_analysis.py --source csv --csv crimes.csv --demographics-csv zones.csv
"""
from __future__ import annotations

import argparse
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from crimegeo import allocation, clustering as C, mapping, plots, socioeconomic as S, temporal as T
from crimegeo.cleaning import clean_crimes
from crimegeo.config import CITY, SOCIO_COLS
from crimegeo.data_sources import load_dataset
from crimegeo.report import write_report


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["synthetic", "chicago", "csv"], default="synthetic")
    p.add_argument("--csv", help="crime CSV (source=csv): case_number,date,primary_type,latitude,longitude,community_area")
    p.add_argument("--demographics-csv", help="zone CSV (source=csv): zone_id,zone_name,population + socioeconomic columns")
    p.add_argument("--population-csv", help="zone_id,population (source=chicago; the portal dataset has no population)")
    p.add_argument("--app-token", help="optional Socrata app token (higher rate limits)")
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default=None, help="last date to include (default: 2026-10-31 for synthetic, today otherwise)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eps", type=float, default=250.0, help="DBSCAN radius in metres")
    p.add_argument("--density-multiple", type=float, default=10.0, help="min_samples as a multiple of average density")
    p.add_argument("--patrol-hours", type=float, default=1000.0, help="illustrative weekly deployable hours")
    p.add_argument("--directed-share", type=float, default=0.70)
    p.add_argument("--alpha", type=float, default=0.75)
    p.add_argument("--outdir", default="outputs")
    p.add_argument("--report", default="REPORT.md", help="where to write the Markdown report")
    return p.parse_args()


def main():
    a = parse_args()
    t0 = time.time()
    out = Path(a.outdir)
    for sub in ("figures", "maps", "tables"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    synthetic = a.source == "synthetic"
    end = a.end or ("2026-10-31" if synthetic else date.today().isoformat())

    # 1. Obtain ---------------------------------------------------------------------------
    print(f"[1/8] loading data (source={a.source}) ...")
    raw, demo = load_dataset(a.source, seed=a.seed, start=a.start, end=end, csv_path=a.csv,
                             demographics_csv=a.demographics_csv, population_csv=a.population_csv, app_token=a.app_token)
    raw.to_csv(f"data/crimes_raw_{a.source}.csv.gz", index=False)
    demo.to_csv(f"data/demographics_{a.source}.csv", index=False)

    # 2. Clean ----------------------------------------------------------------------------
    print("[2/8] cleaning ...")
    clean, audit = clean_crimes(raw, start=a.start, end=end)
    audit.to_csv(out / "tables/data_cleaning_audit.csv", index=False)
    geo = clean[clean["has_geo"]].reset_index(drop=True)
    print(f"      {len(clean):,} clean rows, {len(geo):,} with coordinates")

    # 3. Cluster --------------------------------------------------------------------------
    print("[3/8] DBSCAN hotspots ...")
    proj = C.LocalProjection(float(geo["latitude"].mean()), float(geo["longitude"].mean()))
    x, y = proj.forward(geo["latitude"], geo["longitude"])
    xy = np.column_stack([x, y])
    min_samples = C.auto_min_samples(xy, a.eps, a.density_multiple)
    labels = C.run_dbscan(xy, a.eps, min_samples)
    total_area = C.occupied_area_km2(xy)
    hotspots, rank, polygons = C.summarise_hotspots(geo, labels, proj, xy, total_area)
    hot_area = float(hotspots["footprint_km2"].sum())
    pct_in_hot = 100 * float((rank > 0).mean())
    pct_area_hot = 100 * hot_area / total_area
    # Sensitivity grid is centred on the chosen setting so the "used" row sits in the middle.
    sens = C.sensitivity_grid(xy, eps_values=tuple(round(a.eps * f) for f in (0.6, 1.0, 1.4)),
                              multiples=tuple(round(a.density_multiple * f, 1) for f in (0.6, 1.0, 1.5)))
    sens["used"] = np.where((sens["eps_m"] == round(a.eps)) & (sens["density_multiple"] == round(a.density_multiple, 1)), "used", "")
    kd = C.k_distance_curve(xy, min_samples)
    hotspots.drop(columns=["label"]).to_csv(out / "tables/hotspots.csv", index=False)
    sens.to_csv(out / "tables/dbscan_sensitivity.csv", index=False)
    clean["hotspot_rank"] = 0
    clean.loc[clean["has_geo"], "hotspot_rank"] = rank
    clean.to_csv(out / "tables/crimes_clean_with_hotspot_rank.csv.gz", index=False)
    print(f"      eps={a.eps:.0f} m, min_samples={min_samples:,}: {len(hotspots)} hotspots, "
          f"{pct_in_hot:.1f}% of incidents in {pct_area_hot:.1f}% of area")

    # 4. Time series ----------------------------------------------------------------------
    print("[4/8] time series ...")
    monthly = T.monthly_counts(clean)
    dec = T.decompose(monthly)
    trend = T.trend_summary(dec)
    seasonal = T.seasonal_profile(dec)
    by_group = T.monthly_counts(clean, by="group")
    hd = T.hour_dow_matrix(clean)
    ttypes = T.type_trends(clean)
    dec.to_csv(out / "tables/monthly_decomposition.csv")
    by_group.to_csv(out / "tables/monthly_by_group.csv")
    ttypes.to_csv(out / "tables/crime_type_trends.csv", index=False)
    hd.round(3).to_csv(out / "tables/hour_by_weekday_share_pct.csv")

    # 5. Socioeconomics -------------------------------------------------------------------
    print("[5/8] socioeconomic correlates ...")
    zones = S.zone_table(clean, demo, geo)
    outcome, outcome_label = S.outcome_column(zones)
    corr = S.spearman_table(zones, outcome)
    preds = [c for c in SOCIO_COLS if c in zones.columns]
    keep = S.select_by_vif(zones[preds])
    fit = S.ols(zones, outcome, keep)
    cz = zones.dropna(subset=["latitude"])
    zx, zy = proj.forward(cz["latitude"], cz["longitude"])
    zxy = np.column_stack([zx, zy])
    moran_rate = S.morans_i(np.log(cz[outcome].to_numpy()), zxy)
    resid_map = dict(zip(zones["zone_id"], fit["std_resid"]))
    moran_resid = S.morans_i(np.array([fit["resid"][i] for i in cz.index]), zxy)
    zones["std_resid"] = zones["zone_id"].map(resid_map)
    zones.to_csv(out / "tables/zone_table.csv", index=False)
    corr.to_csv(out / "tables/socioeconomic_correlations.csv", index=False)
    fit["coef"].to_csv(out / "tables/regression_coefficients.csv", index=False)
    S.vif_table(zones[preds]).rename("vif").to_csv(out / "tables/vif.csv")

    # 6. Allocation -----------------------------------------------------------------------
    print("[6/8] allocation plan ...")
    plan = allocation.build_plan(hotspots, zones, seasonal, zones, patrol_hours=a.patrol_hours,
                                 directed_share=a.directed_share, alpha=a.alpha)
    plan["hotspots"].drop(columns=["label"]).to_csv(out / "tables/allocation_hotspots.csv", index=False)
    plan["baseline"].to_csv(out / "tables/allocation_baseline_zones.csv", index=False)
    plan["seasonal"].to_csv(out / "tables/allocation_seasonal.csv", index=False)
    for k, v in plan["watch"].items():
        v.to_csv(out / f"tables/watchlist_{k}.csv", index=False)

    # 7. Maps -----------------------------------------------------------------------------
    print("[7/8] interactive maps ...")
    banner = ("SYNTHETIC DEMO DATA: not real crime figures." if synthetic else "")
    payload = mapping.build_payload(geo, rank, hotspots, polygons, zones, proj, banner,
                                    f"{CITY}-style crime hotspots ({a.start[:4]}-{end[:4]})" if synthetic
                                    else f"Crime hotspots ({a.start} to {end})")
    mapping.build_map(payload, out / "maps/hotspot_map.html")
    try:
        mapping.build_folium_map(payload, out / "maps/hotspot_map_folium.html")
        print("      Folium map written as well")
    except ImportError:
        print("      folium not installed: skipped hotspot_map_folium.html (pip install folium)")

    # 8. Figures + report -----------------------------------------------------------------
    print("[8/8] figures and report ...")
    tag = "SYNTHETIC DEMO DATA" if synthetic else ""
    cent_xy = {int(r["rank"]): proj.forward(r["centroid_lat"], r["centroid_lon"]) for _, r in hotspots.iterrows()}
    poly_xy = {r: (np.column_stack(proj.forward(p[0], p[1])) if p is not None else None) for r, p in polygons.items()}
    fd = out / "figures"
    plots.fig_kdistance(kd, a.eps, min_samples, fd / "k_distance.png", tag)
    plots.fig_hotspots(xy, rank, cent_xy, fd / "hotspot_map_static.png", tag)
    plots.fig_heatmap(xy, poly_xy, cent_xy, fd / "heatmap_static.png", tag=tag)
    plots.fig_time_series(monthly, dec, by_group, seasonal, fd / "time_series.png", tag)
    plots.fig_hour_dow(hd, fd / "hour_by_weekday.png", tag)
    plots.fig_demographics(zones, outcome, outcome_label, corr, zones["std_resid"].to_numpy(), fd / "demographics.png", tag)
    plots.fig_allocation(plan["hotspots"], fd / "allocation.png", tag=tag)

    ctx = dict(
        city_label=f"{CITY}-style synthetic dataset" if synthetic else CITY, synthetic=synthetic,
        banner=("SYNTHETIC DEMONSTRATION DATA. The pipeline is real; the numbers below describe a simulated city, not actual crime."
                if synthetic else f"Real data: {a.source}"),
        source_desc=("synthetic Chicago-like incident reports (fabricated, seed %d) with deliberately injected data-quality errors" % a.seed
                     if synthetic else a.source),
        hotspots=hotspots, plan=plan, trend=trend, hour_dow=hd, corr=corr, fit=fit, moran_rate=moran_rate,
        moran_resid=moran_resid, outcome_label=outcome_label, n_geo=len(geo), n_clean=len(clean), audit=audit,
        pct_in_hot=pct_in_hot, pct_area_hot=pct_area_hot, eps=a.eps, min_samples=min_samples,
        multiple=a.density_multiple, sensitivity=sens, seasonal=seasonal, type_trends=ttypes, n_zones=len(zones),
        vif=S.vif_table(zones[preds]),
    )
    write_report(a.report, ctx)
    print(f"done in {time.time() - t0:.0f}s -> {a.report}, {out}/")


if __name__ == "__main__":
    main()
