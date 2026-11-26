"""Time-series analysis: monthly trend, seasonality, year-over-year change, hour x weekday rhythm.

Classical multiplicative decomposition is implemented directly with numpy (no statsmodels needed):
    series = trend x seasonal x residual
with the trend taken from a centred 2x12 moving average.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def monthly_counts(df: pd.DataFrame, by: str | None = None) -> pd.DataFrame | pd.Series:
    """Complete calendar months only: a trailing partial month would fake a 'drop'."""
    last_ts = df["timestamp"].max()
    last_full = last_ts.to_period("M") if last_ts.is_month_end or last_ts.day == last_ts.days_in_month \
        else last_ts.to_period("M") - 1
    d = df[df["month_start"] <= last_full.to_timestamp()]
    if by is None:
        return d.groupby("month_start").size().rename("incidents")
    return d.groupby(["month_start", by]).size().unstack(fill_value=0)


def decompose(y: pd.Series, period: int = 12) -> pd.DataFrame:
    """Multiplicative decomposition of a monthly series."""
    vals = y.to_numpy(dtype=float)
    n = len(vals)
    if n < 2 * period:
        raise ValueError(f"need at least {2 * period} months for decomposition, got {n}")
    w = np.r_[0.5, np.ones(period - 1), 0.5] / period
    trend = np.full(n, np.nan)
    trend[period // 2: n - period // 2] = np.convolve(vals, w, mode="valid")
    ratio = vals / trend
    month_idx = y.index.month.to_numpy() - 1
    seas = np.array([np.nanmean(ratio[month_idx == m]) for m in range(12)])
    seas = seas / seas.mean()
    seasonal = seas[month_idx]
    return pd.DataFrame({
        "observed": vals, "trend": trend, "seasonal_index": seasonal,
        "seasonally_adjusted": vals / seasonal, "residual": vals / (trend * seasonal),
    }, index=y.index)


def seasonal_profile(dec: pd.DataFrame) -> pd.Series:
    s = dec.groupby(dec.index.month)["seasonal_index"].first()
    s.index = MONTHS
    return s


def trend_summary(dec: pd.DataFrame) -> dict:
    """Log-linear trend of the seasonally-adjusted series, plus last-12 vs previous-12 change."""
    sa = dec["seasonally_adjusted"]
    t_years = (np.arange(len(sa))) / 12.0
    res = stats.linregress(t_years, np.log(sa.to_numpy()))
    out = {
        "trend_pct_per_year": float(100 * (np.exp(res.slope) - 1)),
        "trend_p_value": float(res.pvalue),
        "trend_r2": float(res.rvalue ** 2),
        "n_months": int(len(sa)),
        "first_month": sa.index[0].strftime("%b %Y"),
        "last_month": sa.index[-1].strftime("%b %Y"),
    }
    obs = dec["observed"]
    if len(obs) >= 24:
        last12, prev12 = obs.iloc[-12:].sum(), obs.iloc[-24:-12].sum()
        out.update(last12=int(last12), prev12=int(prev12), yoy_change_pct=float(100 * (last12 - prev12) / prev12))
    return out


def type_trends(df: pd.DataFrame, min_prev: int = 200) -> pd.DataFrame:
    """Last-12 vs previous-12 complete months, by crime type."""
    mc = monthly_counts(df, by="primary_type")
    if len(mc) < 24:
        return pd.DataFrame()
    last12, prev12 = mc.iloc[-12:].sum(), mc.iloc[-24:-12].sum()
    out = pd.DataFrame({"prev_12_months": prev12, "last_12_months": last12})
    out["change_pct"] = (100 * (out["last_12_months"] - out["prev_12_months"]) / out["prev_12_months"]).round(1)
    out = out[out["prev_12_months"] >= min_prev].sort_values("last_12_months", ascending=False)
    return out.reset_index().rename(columns={"primary_type": "crime_type"})


def hour_dow_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Share of incidents (%) by weekday (rows) and hour (columns)."""
    m = df.groupby(["dow", "hour"]).size().unstack(fill_value=0).reindex(index=range(7), columns=range(24), fill_value=0)
    m = 100 * m / m.to_numpy().sum()
    m.index = DAYS
    return m


def hotspot_concentration_over_time(df_geo: pd.DataFrame, rank: np.ndarray) -> pd.Series:
    """Monthly share (%) of geocoded incidents that fall inside a hotspot."""
    d = df_geo.assign(_in=rank > 0)
    return (100 * d.groupby("month_start")["_in"].mean()).rename("pct_in_hotspots")
