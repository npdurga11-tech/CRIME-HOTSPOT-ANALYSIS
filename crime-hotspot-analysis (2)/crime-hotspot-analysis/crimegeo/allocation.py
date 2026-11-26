"""Turn the analysis into a transparent, adjustable patrol-allocation proposal.

The method is deliberately simple so a reviewer can audit it:

1. Split deployable hours into *hotspot-directed* patrol and *baseline* (citywide / zone) patrol.
2. Share directed hours across hotspots in proportion to ``risk_score ** alpha``.
   alpha < 1 spreads resources more evenly than raw risk would, reflecting diminishing returns
   from piling every officer onto the single worst block. alpha = 1 is fully proportional.
3. Within each hotspot, tie the time-of-day emphasis to that hotspot's own peak six-hour window.
4. Scale flexible hours by the seasonal index.

alpha, the directed share and the hours budget are ASSUMPTIONS, not findings. Tune them with
operational input and evaluate any deployment (see REPORT limitations).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .temporal import MONTHS


def hotspot_status(yoy: float) -> str:
    if pd.isna(yoy):
        return "insufficient history"
    if yoy >= 10:
        return "rising"
    if yoy <= -10:
        return "falling"
    return "stable"


def build_plan(hotspots: pd.DataFrame, zones: pd.DataFrame, seasonal: pd.Series,
               resid_df: pd.DataFrame | None, patrol_hours: float = 1000.0, directed_share: float = 0.70,
               alpha: float = 0.75) -> dict:
    h = hotspots.copy()
    weight = h["risk_score"] ** alpha
    h["risk_share_pct"] = 100 * h["risk_score"] / h["risk_score"].sum()
    h["directed_share_pct"] = 100 * weight / weight.sum()
    h["hours_per_week"] = patrol_hours * directed_share * weight / weight.sum()
    h["peak_window_hours_per_week"] = h["hours_per_week"] * h["peak_window_share_pct"] / 100
    h["status"] = h["yoy_change_pct"].map(hotspot_status)

    base = None
    if zones is not None and len(zones):
        base = zones[["zone_id", "zone_name", "incidents_per_year"]].copy()
        base["baseline_share_pct"] = 100 * base["incidents_per_year"] / base["incidents_per_year"].sum()
        base["baseline_hours_per_week"] = patrol_hours * (1 - directed_share) * base["baseline_share_pct"] / 100
        base = base.sort_values("baseline_hours_per_week", ascending=False).reset_index(drop=True)

    seas = pd.DataFrame({"month": MONTHS, "seasonal_index": seasonal.to_numpy()})
    seas["flex_hours_vs_average_pct"] = (100 * (seas["seasonal_index"] - 1)).round(1)

    watch = {
        "rising_hotspots": h[(h["status"] == "rising")][["rank", "incidents", "yoy_change_pct", "peak_window"]],
        "non_demographic_zones": pd.DataFrame(),
        "prevention_priority_zones": pd.DataFrame(),
    }
    if resid_df is not None and "std_resid" in resid_df:
        watch["non_demographic_zones"] = resid_df[resid_df["std_resid"] >= 1.5].sort_values(
            "std_resid", ascending=False)[["zone_id", "zone_name", "std_resid"]
                                          + [c for c in ["rate_per_1000"] if c in resid_df]]
    if zones is not None and "hardship_index" in zones and "violent" in zones:
        z = zones.copy()
        vcol = "violent_rate_per_1000" if "violent_rate_per_1000" in z else "violent"
        pri = z[(z["hardship_index"] >= z["hardship_index"].quantile(0.75))
                & (z[vcol] >= z[vcol].quantile(0.75))]
        watch["prevention_priority_zones"] = pri.sort_values(vcol, ascending=False)[
            ["zone_id", "zone_name", "hardship_index", vcol]]
    return {"hotspots": h, "baseline": base, "seasonal": seas, "watch": watch,
            "params": {"patrol_hours": patrol_hours, "directed_share": directed_share, "alpha": alpha}}
