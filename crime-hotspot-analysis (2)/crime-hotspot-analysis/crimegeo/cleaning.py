"""Cleaning raw crime reports into an analysis-ready table, with an audit trail of every fix."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BBOX, DEFAULT_SEVERITY, PROPERTY_TYPES, SEVERITY, VIOLENT_TYPES


def parse_dates(s: pd.Series) -> pd.Series:
    """Parse ISO-8601 timestamps first, then US-style 'MM/DD/YYYY hh:mm:ss AM'. Unparseable -> NaT."""
    s = s.astype("string").str.strip()
    out = pd.to_datetime(s, format="ISO8601", errors="coerce")
    left = out.isna() & s.notna() & (s != "")
    if left.any():
        us = pd.to_datetime(s[left], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
        out = out.where(~left, us.reindex(out.index))
    return out


def clean_crimes(raw: pd.DataFrame, start: str | None = None, end: str | None = None,
                 bbox: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(clean, audit)``.

    Rows without usable coordinates are KEPT (``has_geo`` = False) so time-series counts stay
    complete; they are simply excluded from the spatial steps.
    """
    bbox = bbox or BBOX
    df = raw.copy()
    audit: list[dict] = []

    def log(step: str, affected: int, note: str = ""):
        audit.append({"step": step, "rows_affected": int(affected), "rows_remaining": len(df), "note": note})

    log("raw rows loaded", 0, "starting point")

    # 1. Exact duplicates ------------------------------------------------------------------
    before = len(df)
    df = df.drop_duplicates(subset=["case_number", "date", "primary_type", "latitude", "longitude"])
    log("exact duplicate rows removed", before - len(df))

    # 2. Dates -----------------------------------------------------------------------------
    df["timestamp"] = parse_dates(df["date"])
    bad = df["timestamp"].isna()
    n_bad = int(bad.sum())
    df = df.loc[~bad].copy()
    log("unparseable / missing dates removed", n_bad)

    lo = pd.Timestamp(start) if start else pd.Timestamp.min
    hi = (pd.Timestamp(end) + pd.Timedelta(days=1)) if end else pd.Timestamp.now()
    out_of_window = (df["timestamp"] < lo) | (df["timestamp"] >= hi)
    n_out = int(out_of_window.sum())
    df = df.loc[~out_of_window].copy()
    log("dates outside the study window / in the future removed", n_out)

    # 3. Crime category --------------------------------------------------------------------
    pt = df["primary_type"].astype("string").str.strip().str.upper()
    n_text = int((pt.notna() & (pt != df["primary_type"].astype("string"))).sum())
    n_missing_type = int(pt.isna().sum())
    df["primary_type"] = pt.fillna("UNKNOWN").astype(str)
    log("category text normalised (whitespace / case)", n_text, "kept, not removed")
    log("missing category labelled UNKNOWN", n_missing_type, "kept, not removed")

    # 4. Coordinates -----------------------------------------------------------------------
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    n_missing = int((lat.isna() | lon.isna()).sum())

    zero = (lat == 0) | (lon == 0)
    n_zero = int(zero.sum())
    lat, lon = lat.mask(zero), lon.mask(zero)

    # latitude/longitude swapped (lat looks like a longitude)
    swapped = (lat < -80) & (lon > 0)
    n_swapped = int(swapped.sum())
    lat_fix, lon_fix = lat.where(~swapped, lon), lon.where(~swapped, lat)
    lat, lon = lat_fix, lon_fix

    outside = ~(lat.between(bbox["lat_min"], bbox["lat_max"]) & lon.between(bbox["lon_min"], bbox["lon_max"]))
    n_outside = int((outside & lat.notna() & lon.notna()).sum())
    lat, lon = lat.mask(outside), lon.mask(outside)

    df["latitude"], df["longitude"] = lat, lon
    df["has_geo"] = lat.notna() & lon.notna()
    log("missing coordinates flagged", n_missing, "kept for time series, excluded from mapping/clustering")
    log("zero coordinates blanked", n_zero, "kept, no geometry")
    log("swapped lat/lon repaired", n_swapped, "repaired in place")
    log("coordinates outside city bounding box blanked", n_outside, "kept, no geometry")

    # 5. Zone id + derived fields ----------------------------------------------------------
    df["zone_id"] = pd.to_numeric(df["community_area"], errors="coerce").astype("Int64")
    ts = df["timestamp"]
    df["year"] = ts.dt.year
    df["month_start"] = ts.dt.to_period("M").dt.to_timestamp()
    df["dow"] = ts.dt.dayofweek
    df["hour"] = ts.dt.hour
    df["group"] = np.where(df["primary_type"].isin(VIOLENT_TYPES), "violent",
                           np.where(df["primary_type"].isin(PROPERTY_TYPES), "property", "other"))
    df["severity"] = df["primary_type"].map(SEVERITY).fillna(DEFAULT_SEVERITY)

    keep = ["case_number", "timestamp", "primary_type", "group", "severity", "latitude", "longitude",
            "has_geo", "zone_id", "year", "month_start", "dow", "hour"]
    df = df[keep].sort_values("timestamp").reset_index(drop=True)
    log("clean rows ready for analysis", 0, f"{int(df['has_geo'].sum()):,} of them have usable coordinates")
    return df, pd.DataFrame(audit)
