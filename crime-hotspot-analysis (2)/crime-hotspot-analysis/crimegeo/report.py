"""Build REPORT.md from the computed results so the text can never disagree with the tables."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .temporal import DAYS, MONTHS


def md_table(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.astype(object).iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if pd.isna(v):
                cells.append("n/a")
            elif c in fmt:
                cells.append(fmt[c](v) if callable(fmt[c]) else fmt[c].format(v))
            elif isinstance(v, (float, np.floating)):
                cells.append(f"{v:,.1f}")
            elif isinstance(v, (int, np.integer)):
                cells.append(f"{v:,}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _pcell(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def write_report(path, c: dict) -> None:
    hs, plan, tr, hd = c["hotspots"], c["plan"], c["trend"], c["hour_dow"]
    corr, fit, moran_y, moran_r = c["corr"], c["fit"], c["moran_rate"], c["moran_resid"]
    dem_label = c["outcome_label"]
    top = hs.iloc[0]
    n_geo, n_all = c["n_geo"], c["n_clean"]

    # Citywide rhythm
    flat = hd.to_numpy().ravel()
    hourly = hd.sum(axis=0).to_numpy()
    sums = [sum(hourly[(s + i) % 24] for i in range(6)) for s in range(24)]
    ws = int(np.argmax(sums))
    win = f"{ws:02d}:00-{(ws + 6) % 24:02d}:00"
    win_share = sums[ws] / hourly.sum() * 100
    dow_share = hd.sum(axis=1)
    peak_dow = dow_share.idxmax()

    seas = c["seasonal"]
    pk, lo = seas.idxmax(), seas.idxmin()

    single = corr[corr["column"] != "hardship_index"]      # the hardship index is a composite of the others
    strongest = single.iloc[single["spearman_rho"].abs().argmax()]
    vif = c["vif"]
    sel = ", ".join(fit["coef"]["term"].iloc[1:])
    nd = plan["watch"]["non_demographic_zones"]
    rising = plan["watch"]["rising_hotspots"]
    pri = plan["watch"]["prevention_priority_zones"]

    L: list[str] = []
    a = L.append
    a(f"# Crime Hotspot Analysis: {c['city_label']}\n")
    a(f"> **{c['banner']}**\n")
    a("## 1. Key findings\n")
    a(f"- **Crime is spatially concentrated.** {c['pct_in_hot']:.0f}% of geocoded incidents fall inside "
      f"{len(hs)} DBSCAN hotspots that cover only {c['pct_area_hot']:.1f}% of the occupied area. The top hotspot alone "
      f"holds {top['share_of_all_risk_pct']:.1f}% of citywide severity-weighted risk.")
    a(f"- **Volume trend:** the seasonally-adjusted trend is {tr['trend_pct_per_year']:+.1f}% per year ({_p(tr['trend_p_value'])}); "
      f"the last 12 complete months are {tr['yoy_change_pct']:+.1f}% versus the 12 before." if "yoy_change_pct" in tr else
      f"- **Volume trend:** {tr['trend_pct_per_year']:+.1f}% per year ({_p(tr['trend_p_value'])}).")
    a(f"- **Seasonality:** {pk} is the busiest month ({100 * (seas[pk] - 1):+.0f}% vs an average month); {lo} is the quietest ({100 * (seas[lo] - 1):+.0f}%).")
    a(f"- **Rhythm:** the busiest six-hour window citywide is {win} ({win_share:.0f}% of incidents); {peak_dow} has the most incidents of any weekday.")
    a(f"- **Neighbourhood context:** among the individual indicators, zone-level crime rates are most strongly related to *{strongest['indicator']}* "
      f"(Spearman rho = {strongest['spearman_rho']:+.2f}, {_p(strongest['p_value'])}, n = {int(strongest['n_zones'])} zones). "
      f"Crime rates are also spatially clustered (Moran's I = {moran_y['I']:.2f}, {_p(moran_y['p_value'])}).")
    if len(nd):
        names = ", ".join(nd["zone_name"].astype(str))
        a(f"- **Not everything is explained by disadvantage:** {names} show{'s' if len(nd) == 1 else ''} markedly higher crime than "
          f"the socioeconomic model predicts, which points to drivers such as commercial activity or foot traffic.")
    a("")

    a("## 2. Data and cleaning\n")
    a(f"Source: **{c['source_desc']}**. Study window: {tr['first_month']} to {tr['last_month']}. "
      f"{n_all:,} clean incident records, of which {n_geo:,} ({100 * n_geo / n_all:.1f}%) have usable coordinates. "
      "Records without coordinates stay in the time-series counts but are excluded from mapping and clustering.\n")
    a(md_table(c["audit"][["step", "rows_affected", "rows_remaining", "note"]], {"rows_affected": "{:,}", "rows_remaining": "{:,}"}))
    a("")

    a("## 3. Spatial hotspots (DBSCAN)\n")
    a(f"Coordinates were projected to local metres. DBSCAN used **eps = {c['eps']:.0f} m** (about one to two city blocks) and "
      f"**min_samples = {c['min_samples']:,}**, i.e. a core point needs at least {c['multiple']:g}x the number of incidents that a "
      f"{c['eps']:.0f} m radius would hold at the citywide average density. Incidents were snapped to a 25 m grid and clustered as weighted points "
      "(equivalent to DBSCAN on the raw points, but memory-safe). Clusters are ranked by severity-weighted risk, not raw count.\n")
    cols = ["rank", "incidents", "share_of_all_risk_pct", "violent_share_pct", "footprint_km2", "peak_window", "peak_dow", "top_types", "yoy_change_pct"]
    t = hs.head(10)[cols].rename(columns={"share_of_all_risk_pct": "citywide risk share %", "violent_share_pct": "violent %",
                                           "footprint_km2": "km²", "peak_window": "peak 6h window", "peak_dow": "peak day",
                                           "top_types": "top crime types", "yoy_change_pct": "last-12m change %"})
    a(f"**Top 10 of {len(hs)} hotspots**\n")
    a(md_table(t, {"km²": "{:.2f}", "citywide risk share %": "{:.1f}", "violent %": "{:.0f}", "last-12m change %": "{:+.0f}"}))
    a("\n**Parameter sensitivity.** The hotspot count and coverage move with eps and the density multiple. The row marked *used* is the setting "
      "behind every map and table here; the other rows show how much the headline numbers would change. Check that conclusions about "
      "*where* the largest hotspots are survive neighbouring settings:\n")
    sens = c["sensitivity"].rename(columns={"eps_m": "eps (m)", "density_multiple": "density multiple", "min_samples": "min_samples",
                                            "n_hotspots": "hotspots", "pct_incidents_in_hotspots": "% incidents in hotspots",
                                            "pct_area_in_hotspots": "% area in hotspots"})
    a(md_table(sens, {"eps (m)": "{:.0f}", "density multiple": "{:g}", "min_samples": "{:,}"}))
    a("\n![hotspots](outputs/figures/hotspot_map_static.png)\n\n![heatmap](outputs/figures/heatmap_static.png)\n")
    a("The k-distance curve below is the classic diagnostic for choosing eps: a sharp elbow would mark where neighbours stop being 'close'. "
      "In this data the curve rises smoothly with no clear elbow, so it cannot pin eps down on its own. eps was therefore set on physical grounds "
      "(block scale) and its influence is shown in the sensitivity table above.\n")
    a("![k-distance](outputs/figures/k_distance.png)\n")
    a("Interactive version with toggleable heatmaps (all, violent, night, daytime), hotspot popups and zone overlay: "
      "`outputs/maps/hotspot_map.html` (needs an internet connection for the basemap tiles).\n")

    a("## 4. Time-series analysis\n")
    a(f"Monthly counts over {tr['n_months']} complete months were decomposed into trend x seasonality x residual. "
      f"The log-linear trend of the seasonally-adjusted series is {tr['trend_pct_per_year']:+.1f}% per year (R² = {tr['trend_r2']:.2f}, {_p(tr['trend_p_value'])}).\n")
    a("**By crime type, last 12 complete months vs the 12 before:**\n")
    a(md_table(c["type_trends"], {"prev_12_months": "{:,}", "last_12_months": "{:,}", "change_pct": "{:+.1f}"}))
    a("\n![time series](outputs/figures/time_series.png)\n\n![hour by weekday](outputs/figures/hour_by_weekday.png)\n")

    a("## 5. Socioeconomic correlates\n")
    a(f"Unit of analysis: {c['n_zones']} zones. Outcome: {dem_label}.\n")
    a(md_table(corr[["indicator", "spearman_rho", "p_value", "n_zones"]], {"spearman_rho": "{:+.2f}", "p_value": _pcell, "n_zones": "{:d}"}))
    if vif.max() >= 5:
        collin = (f"\nThe indicators are collinear (largest variance inflation factor {vif.max():.1f}; above 5 is usually treated as a problem), so a regression "
                  f"with all of them gives unstable coefficients. Predictors were dropped one at a time (highest VIF first) until all VIFs were below 5, "
                  f"leaving **{sel}**. Which of several collinear indicators survives is somewhat arbitrary: read the model as 'neighbourhood "
                  f"disadvantage' jointly, not as a ranking of individual causes. ")
    else:
        collin = f"\nThe indicators are not strongly collinear (largest VIF {vif.max():.1f}), so all were kept: **{sel}**. "
    a(collin + f"OLS on log crime rate with standardised predictors gives R² = {fit['r2']:.2f} (adjusted {fit['adj_r2']:.2f}, n = {fit['n']}):\n")
    cf = fit["coef"].iloc[1:][["term", "coef_per_1sd", "std_err", "p_value", "pct_change_per_1sd"]]
    a(md_table(cf, {"coef_per_1sd": "{:+.3f}", "std_err": "{:.3f}", "p_value": _pcell, "pct_change_per_1sd": "{:+.0f}"}))
    a(f"\n`pct_change_per_1sd` is the change in the crime rate associated with a one-standard-deviation higher value of that indicator, holding the others fixed. "
      f"Spatial autocorrelation: Moran's I of the crime rate is {moran_y['I']:.2f} ({_p(moran_y['p_value'])}); "
      f"of the regression residuals {moran_r['I']:.2f} ({_p(moran_r['p_value'])}), "
      f"{'so little spatial structure is left unexplained' if moran_r['p_value'] > 0.05 else 'so spatial structure remains that the model does not capture'}.\n")
    a("![demographics](outputs/figures/demographics.png)\n")
    a("**Read this carefully:** these are correlations between *places*. They do not show that disadvantage causes crime, and they say nothing about "
      f"individual residents (ecological fallacy). With only {c['n_zones']} zones, statistical power is modest and individual coefficients are fragile.\n")

    p = plan["params"]
    a("## 6. Recommendations for police resource allocation\n")
    a(f"The proposal below is a transparent starting point, not an optimiser. Assumptions you should change with operational input: "
      f"a budget of **{p['patrol_hours']:,.0f} deployable patrol hours per week** (illustrative), **{100 * p['directed_share']:.0f}%** of it hotspot-directed, "
      f"directed hours shared in proportion to risk^**{p['alpha']}** (alpha < 1 spreads resources more than raw risk would, reflecting diminishing returns).\n")
    a("### 6.1 Hotspot-directed patrol\n")
    pt = plan["hotspots"].head(10)[["rank", "risk_share_pct", "directed_share_pct", "hours_per_week", "peak_window", "peak_window_share_pct", "status"]]
    pt = pt.rename(columns={"risk_share_pct": "share of hotspot risk %", "directed_share_pct": "share of directed hours %",
                            "hours_per_week": "hours / week", "peak_window": "emphasise window",
                            "peak_window_share_pct": "window holds % of hotspot risk"})
    a(md_table(pt, {"share of hotspot risk %": "{:.1f}", "share of directed hours %": "{:.1f}", "hours / week": "{:.0f}",
                    "window holds % of hotspot risk": "{:.0f}"}))
    a(f"\nWithin each hotspot, weight hours toward its own peak window (the 'emphasise window' column). The remaining {100 * (1 - p['directed_share']):.0f}% "
      "of hours stay as baseline zone patrol, shared by each zone's incident volume (full table: `outputs/tables/allocation_baseline_zones.csv`).\n")
    a("![allocation](outputs/figures/allocation.png)\n")
    a("### 6.2 Timing\n")
    a(f"- **Time of day:** citywide, {win} holds {win_share:.0f}% of incidents. Shift start times so that the heaviest staffing overlaps this window, and check each hotspot's own window in the table above because they differ.")
    a(f"- **Day of week:** {peak_dow} is the busiest day; weekends (Fri-Sun) carry {100 * dow_share[['Fri', 'Sat', 'Sun']].sum() / dow_share.sum():.0f}% of incidents versus {300 / 7:.0f}% if evenly spread.")
    a(f"- **Seasonal flex:** scale flexible/overtime hours by the seasonal index: {pk} runs {100 * (seas[pk] - 1):+.0f}% above average, {lo} {100 * (seas[lo] - 1):+.0f}%.\n")
    seas_tbl = pd.DataFrame({"metric": ["seasonal index", "flex hours vs average"]})
    for m, idx, pct in zip(plan["seasonal"]["month"], plan["seasonal"]["seasonal_index"],
                           plan["seasonal"]["flex_hours_vs_average_pct"]):
        seas_tbl[m] = [f"{idx:.2f}", f"{pct:+.0f}%"]
    a(md_table(seas_tbl))
    a("")
    a("### 6.3 Watch list and non-enforcement levers\n")
    if len(rising):
        a("- **Rising hotspots (last 12 months at least 10% above the previous 12):** "
          + "; ".join(f"#{int(r['rank'])} ({r['yoy_change_pct']:+.0f}%, peak {r['peak_window']})" for _, r in rising.iterrows())
          + ". Review these first at the next deployment cycle.")
    else:
        a("- **Rising hotspots:** none exceeded +10% year over year.")
    if len(nd):
        a("- **Crime above what disadvantage predicts:** " + ", ".join(nd["zone_name"].astype(str))
          + ". These look like activity-driven (commercial, transit, nightlife) rather than neighbourhood-driven problems, so pair patrol with place-management partners (business improvement districts, transit and licensing authorities, lighting and design reviews).")
    if len(pri):
        a("- **High hardship and high violent-crime zones:** " + ", ".join(pri["zone_name"].astype(str))
          + ". Policing alone is unlikely to change the underlying conditions; prioritise these zones for place-based and social investments (lighting, vacant-lot remediation, youth and employment programmes, violence-intervention partnerships) alongside any patrol.")
    a("- **Evaluate, don't assume:** run any new deployment as a pilot with comparison areas, track displacement to adjacent blocks, and re-run this pipeline quarterly so hotspots, windows and trends stay current.\n")

    a("## 7. Limitations and responsible use\n")
    a("- **Reported crime is not all crime.** Reporting rates and enforcement intensity shape the data; categories like narcotics mostly reflect where police look. Allocating more patrol to high-count areas can raise recorded counts there and reinforce the pattern (a feedback loop). Treat hotspots as a prompt for judgement and review, not an automatic order.")
    a("- **Ecological fallacy and small n.** Zone-level correlations describe places, not people, and are based on few zones.")
    a("- **Parameter dependence.** DBSCAN results depend on eps and min_samples (see sensitivity table). Severity weights are illustrative, not an official harm index.")
    a(f"- **Geocoding.** {100 - 100 * n_geo / n_all:.1f}% of records lacked usable coordinates and are absent from maps; if their absence is not random, hotspots may be biased.")
    a("- **Modifiable areal unit problem.** Zone boundaries are administrative; different boundaries could change the correlations.")
    if c["synthetic"]:
        a("- **SYNTHETIC DATA.** Every number above comes from a simulated dataset with built-in structure (including the link between disadvantage and crime). The *method* and *code* are the deliverable; the *findings* describe the simulation, not a real city. Re-run with `--source chicago` (or your own CSV) for real results.")
    a("")
    a("## 8. Reproduce\n")
    a("```bash\npip install -r requirements.txt\npython run_analysis.py                      # synthetic demo data\npython run_analysis.py --source chicago     # real Chicago open data (needs internet)\n```\n")
    a("All tables are in `outputs/tables/`, figures in `outputs/figures/`, interactive map in `outputs/maps/`.\n")
    from pathlib import Path
    Path(path).write_text("\n".join(L), encoding="utf-8")
