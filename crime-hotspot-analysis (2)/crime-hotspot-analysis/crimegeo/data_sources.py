"""Where the data comes from.

Three sources are supported:

* ``synthetic``  - a realistic, *fabricated* Chicago-like dataset (default). It has the same
  columns as the real portal data, includes deliberate data-quality problems for the cleaning
  step to fix, and is fully reproducible from a seed. Findings on it demonstrate the method;
  they are NOT findings about the real city.
* ``chicago``    - downloads real data from the City of Chicago open-data (Socrata) portal.
  NOTE: this code path could not be run in the sandbox this project was built in (no network),
  so it is untested. See README for what to check on first run.
* ``csv``        - your own crime-report CSV with the columns listed in README.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

M_PER_DEG_LAT = 111_200.0
SOCRATA_BASE = "https://data.cityofchicago.org/resource"
CRIMES_DATASET = "ijzp-q8t2"          # Crimes - 2001 to Present
SOCIO_DATASET = "kn9c-c2s2"           # Census Data - Selected socioeconomic indicators, 2008-2012


# ---------------------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------------------
def _offset_to_latlon(lat0, lon0, dx_m, dy_m):
    lat = lat0 + dy_m / M_PER_DEG_LAT
    lon = lon0 + dx_m / (M_PER_DEG_LAT * np.cos(np.radians(lat0)))
    return lat, lon


def _make_zones(rng: np.random.Generator, n_zones: int) -> pd.DataFrame:
    """Create synthetic neighbourhood zones with spatially smooth socioeconomic disadvantage."""
    lat0, lon0 = 41.84, -87.68
    centres = [(41.882, -87.631)]  # zone 1 = central business district (the "hub")
    while len(centres) < n_zones:
        r, th = np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
        lat, lon = lat0 + 0.17 * r * np.sin(th), lon0 + 0.095 * r * np.cos(th)
        if all(np.hypot((lat - a) * 111.2, (lon - b) * 82.8) > 3.0 for a, b in centres):
            centres.append((lat, lon))
    lat = np.array([c[0] for c in centres])
    lon = np.array([c[1] for c in centres])

    # Latent disadvantage: higher to the south/west, plus noise; the hub is relatively affluent.
    z = -(lat - lat0) / 0.17 - 0.5 * (lon - lon0) / 0.095 + rng.normal(0, 0.6, n_zones)
    z = (z - z.mean()) / z.std()
    z[0] = -0.6

    clip = np.clip
    pop = clip(rng.lognormal(np.log(42_000), 0.45, n_zones), 8_000, 110_000).round(-2)
    pop[0] = 14_000
    df = pd.DataFrame({
        "zone_id": np.arange(1, n_zones + 1),
        "zone_name": [f"Zone {i:02d}" for i in range(1, n_zones + 1)],
        "population": pop.astype(int),
        "poverty_pct": clip(18 + 9.0 * z + rng.normal(0, 3, n_zones), 3, 55).round(1),
        "unemployment_pct": clip(12 + 5.5 * z + rng.normal(0, 2, n_zones), 2, 35).round(1),
        "no_hs_diploma_pct": clip(20 + 8.0 * z + rng.normal(0, 4, n_zones), 2, 55).round(1),
        "crowded_housing_pct": clip(4.5 + 2.5 * z + rng.normal(0, 1.2, n_zones), 0.5, 20).round(1),
        "dependency_pct": clip(38 + 3.0 * z + rng.normal(0, 4.5, n_zones), 20, 60).round(1),
        "per_capita_income": clip(np.exp(np.log(26_000) - 0.45 * z + rng.normal(0, 0.15, n_zones)),
                                  8_000, 90_000).round(-2),
    })
    # Hardship index = mean percentile rank of the six indicators (income reversed), scaled 1-100.
    ranks = pd.concat([
        df["crowded_housing_pct"].rank(pct=True), df["poverty_pct"].rank(pct=True),
        df["unemployment_pct"].rank(pct=True), df["no_hs_diploma_pct"].rank(pct=True),
        df["dependency_pct"].rank(pct=True), 1 - df["per_capita_income"].rank(pct=True) + 1 / n_zones,
    ], axis=1).mean(axis=1)
    df["hardship_index"] = (1 + 99 * (ranks - ranks.min()) / (ranks.max() - ranks.min())).round(0)
    df["is_hub"] = [True] + [False] * (n_zones - 1)  # kept for transparency; analysis ignores it
    df["_lat"], df["_lon"], df["_z"] = lat, lon, z
    return df


# type -> (base share, peak hour, hour sharpness, seasonal amplitude, weekday profile, trend/yr)
_VIOLENT_DOW = [0.90, 0.90, 0.92, 0.97, 1.12, 1.25, 1.20]
_PROPERTY_DOW = [1.00, 1.00, 1.00, 1.00, 1.05, 1.00, 0.90]
_BURGLARY_DOW = [1.08, 1.08, 1.05, 1.05, 1.00, 0.88, 0.85]
_PROFILES = {
    "THEFT":               (0.26, 15.0, 0.8, 0.10, _PROPERTY_DOW, -0.01),
    "BATTERY":             (0.16, 22.0, 0.9, 0.22, _VIOLENT_DOW, -0.05),
    "CRIMINAL DAMAGE":     (0.11, 23.0, 0.6, 0.12, _PROPERTY_DOW, -0.03),
    "ASSAULT":             (0.08, 21.5, 0.9, 0.22, _VIOLENT_DOW, -0.05),
    "BURGLARY":            (0.06, 11.0, 0.9, 0.08, _BURGLARY_DOW, -0.04),
    "MOTOR VEHICLE THEFT": (0.08, 1.0, 0.9, 0.03, _PROPERTY_DOW, 0.08),
    "ROBBERY":             (0.05, 21.0, 1.0, 0.12, _VIOLENT_DOW, -0.05),
    "NARCOTICS":           (0.07, 17.0, 0.7, 0.02, _PROPERTY_DOW, -0.06),
    "DECEPTIVE PRACTICE":  (0.12, 12.0, 0.5, 0.00, _PROPERTY_DOW, 0.03),
    "HOMICIDE":            (0.0015, 23.0, 0.8, 0.20, _VIOLENT_DOW, -0.05),
}
_VIOLENT_SET = {"BATTERY", "ASSAULT", "ROBBERY", "HOMICIDE", "NARCOTICS"}


def _type_probs(z: float, is_hub: bool) -> pd.Series:
    p = pd.Series({t: v[0] for t, v in _PROFILES.items()})
    for t in p.index:
        if t in _VIOLENT_SET:
            p[t] *= np.exp((0.6 if t == "HOMICIDE" else 0.35 if t != "NARCOTICS" else 0.3) * z)
        else:
            p[t] *= np.exp(-0.05 * z)
    if is_hub:
        p["THEFT"] *= 3.0
        p["ROBBERY"] *= 1.6
        p["DECEPTIVE PRACTICE"] *= 2.0
        p["MOTOR VEHICLE THEFT"] *= 0.8
    return p / p.sum()


def generate_synthetic(seed: int = 42, n_zones: int = 28,
                       start: str = "2023-01-01", end: str = "2026-10-31"):
    """Return ``(raw_crimes, demographics)``. Raw crimes intentionally contain dirty records."""
    rng = np.random.default_rng(seed)
    zones = _make_zones(rng, n_zones)
    days = pd.date_range(start, end, freq="D")
    t_years = (days - days[0]).days.to_numpy() / 365.25
    doy, dow = days.dayofyear.to_numpy(), days.dayofweek.to_numpy()
    season_phase = np.cos(2 * np.pi * (doy - 196) / 365.25)  # peaks mid-July

    # Daily weights per crime type: seasonality x weekday profile x long-run trend.
    day_w = {}
    for t, (_, _, _, amp, dow_prof, trend) in _PROFILES.items():
        w = (1 + amp * season_phase) * np.asarray(dow_prof)[dow] * np.clip(1 + trend * t_years, 0.2, None)
        day_w[t] = w / w.sum()

    years = len(days) / 365.25
    frames = []
    for _, zr in zones.iterrows():
        z, hub = zr["_z"], bool(zr["is_hub"])
        rate = 38 * np.exp(0.5 * z + rng.normal(0, 0.18)) * (2.8 if hub else 1.0)  # per 1,000 per year
        n = rng.poisson(rate * zr["population"] / 1000 * years)
        probs = _type_probs(z, hub)
        types = rng.choice(probs.index.to_numpy(), size=n, p=probs.to_numpy())

        # Spatial pattern: a few tight cores (street-level hotspots) plus a diffuse background.
        k = int(np.clip(1 + rng.poisson(1.2 + 0.6 * max(z, 0)), 1, 4)) + (2 if hub else 0)
        core_xy = rng.normal(0, 900, size=(k, 2))
        core_sd = rng.uniform(100, 250, size=k)
        core_w = rng.dirichlet(np.ones(k) * 2)
        in_core = rng.uniform(size=n) < (0.72 if hub else 0.55)
        which = rng.choice(k, size=n, p=core_w)
        dx = np.where(in_core, core_xy[which, 0] + rng.normal(0, 1, n) * core_sd[which],
                      rng.normal(0, 1500, n))
        dy = np.where(in_core, core_xy[which, 1] + rng.normal(0, 1, n) * core_sd[which],
                      rng.normal(0, 1500, n))
        lat, lon = _offset_to_latlon(zr["_lat"], zr["_lon"], dx, dy)

        # Timestamps per crime type.
        ts = np.empty(n, dtype="datetime64[s]")
        for t in np.unique(types):
            idx = np.where(types == t)[0]
            day_idx = rng.choice(len(days), size=len(idx), p=day_w[t])
            _, peak, kappa, *_ = _PROFILES[t]
            hw = np.exp(kappa * np.cos(2 * np.pi * (np.arange(24) - peak) / 24))
            hours = rng.choice(24, size=len(idx), p=hw / hw.sum())
            secs = hours * 3600 + rng.integers(0, 3600, len(idx))
            ts[idx] = days.to_numpy()[day_idx].astype("datetime64[s]") + secs.astype("timedelta64[s]")
        frames.append(pd.DataFrame({
            "date": ts, "primary_type": types, "latitude": lat, "longitude": lon,
            "community_area": int(zr["zone_id"]),
        }))

    crimes = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    crimes.insert(0, "case_number", [f"JA{i:06d}" for i in range(1, len(crimes) + 1)])
    crimes = _inject_dirt(crimes, rng)
    demo = zones.drop(columns=["_lat", "_lon", "_z"])
    return crimes, demo


def _inject_dirt(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add realistic data-quality problems so the cleaning step has something to do."""
    df = df.copy()
    n = len(df)
    pick = lambda frac: rng.choice(n, size=int(n * frac), replace=False)  # noqa: E731

    # Dates: half US-style strings, half ISO; a few junk and impossible values.
    us = rng.uniform(size=n) < 0.5
    dates = pd.Series(df["date"])
    df["date"] = np.where(us, dates.dt.strftime("%m/%d/%Y %I:%M:%S %p"), dates.dt.strftime("%Y-%m-%dT%H:%M:%S"))
    df.loc[pick(0.002), "date"] = rng.choice(["N/A", "UNKNOWN", ""], size=int(n * 0.002))
    df.loc[pick(0.002), "date"] = "2027-03-15T10:00:00"  # future-dated

    # Coordinates: missing, zero, swapped, wildly out of range.
    df["latitude"] = df["latitude"].astype(float)
    df["longitude"] = df["longitude"].astype(float)
    miss = pick(0.02)
    df.loc[miss, ["latitude", "longitude"]] = np.nan
    zero = pick(0.003)
    df.loc[zero, ["latitude", "longitude"]] = 0.0
    swap = pick(0.003)
    df.loc[swap, ["latitude", "longitude"]] = df.loc[swap, ["longitude", "latitude"]].to_numpy()
    far = pick(0.002)
    df.loc[far, "latitude"] = df.loc[far, "latitude"] + 1.5

    # Category text: stray whitespace / lower case / missing.
    messy = pick(0.01)
    df.loc[messy, "primary_type"] = [(" " + s.lower() + " ") if i % 2 else (s + " ")
                                     for i, s in enumerate(df.loc[messy, "primary_type"])]
    df.loc[pick(0.002), "primary_type"] = np.nan
    df["community_area"] = df["community_area"].astype("float")
    df.loc[pick(0.01), "community_area"] = np.nan

    # Exact duplicate rows.
    dup = df.iloc[pick(0.005)]
    return pd.concat([df, dup], ignore_index=True).sample(frac=1.0, random_state=int(rng.integers(1e6))) \
        .reset_index(drop=True)


# ---------------------------------------------------------------------------------------
# Real data: City of Chicago open-data portal (Socrata SODA API)  -- UNTESTED in sandbox
# ---------------------------------------------------------------------------------------
def _soda_get(dataset: str, params: dict, app_token: str | None = None, retries: int = 3) -> list[dict]:
    import requests

    headers = {"X-App-Token": app_token} if app_token else {}
    for attempt in range(retries):
        resp = requests.get(f"{SOCRATA_BASE}/{dataset}.json", params=params, headers=headers, timeout=120)
        if resp.status_code == 200:
            return resp.json()
        time.sleep(2 ** attempt)
    resp.raise_for_status()
    return []


def fetch_chicago_crimes(start: str = "2023-01-01", end: str | None = None, max_rows: int = 600_000,
                         app_token: str | None = None, page_size: int = 50_000) -> pd.DataFrame:
    """Download crime reports (with coordinates) from the Chicago data portal, paged by date."""
    where = f"date >= '{start}T00:00:00'"
    if end:
        where += f" AND date <= '{end}T23:59:59'"
    cols = "case_number,date,primary_type,latitude,longitude,community_area"
    rows: list[dict] = []
    offset = 0
    while offset < max_rows:
        batch = _soda_get(CRIMES_DATASET, {"$select": cols, "$where": where, "$order": "date",
                                           "$limit": page_size, "$offset": offset}, app_token)
        if not batch:
            break
        rows.extend(batch)
        offset += page_size
        if len(batch) < page_size:
            break
    return pd.DataFrame(rows)


_SOCIO_RENAMES = [  # (substring in source column, canonical name)
    ("crowded", "crowded_housing_pct"), ("poverty", "poverty_pct"), ("unemployed", "unemployment_pct"),
    ("high_school", "no_hs_diploma_pct"), ("under_18", "dependency_pct"),
    ("per_capita", "per_capita_income"), ("hardship", "hardship_index"),
    ("community_area_name", "zone_name"),
]


def fetch_chicago_socioeconomics(app_token: str | None = None, population_csv: str | None = None) -> pd.DataFrame:
    """Community-area socioeconomic indicators. Population is NOT in this dataset; pass a CSV with
    columns ``zone_id,population`` to get per-capita rates, otherwise rankings use raw counts."""
    df = pd.DataFrame(_soda_get(SOCIO_DATASET, {"$limit": 200}, app_token))
    rename = {}
    for col in df.columns:
        for key, canon in _SOCIO_RENAMES:
            if key in col.lower() and canon not in rename.values():
                rename[col] = canon
    df = df.rename(columns=rename)
    df["zone_id"] = pd.to_numeric(df.get("ca"), errors="coerce")
    df = df.dropna(subset=["zone_id"]).copy()  # drops the city-wide summary row
    df["zone_id"] = df["zone_id"].astype(int)
    for c in ["crowded_housing_pct", "poverty_pct", "unemployment_pct", "no_hs_diploma_pct",
              "dependency_pct", "per_capita_income", "hardship_index"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if population_csv:
        df = df.merge(pd.read_csv(population_csv)[["zone_id", "population"]], on="zone_id", how="left")
    else:
        df["population"] = np.nan
    return df


def load_dataset(source: str, seed: int = 42, start: str = "2023-01-01", end: str | None = None,
                 csv_path: str | None = None, demographics_csv: str | None = None,
                 population_csv: str | None = None, app_token: str | None = None):
    """Return ``(raw_crimes, demographics)`` for the chosen source."""
    if source == "synthetic":
        return generate_synthetic(seed=seed)
    if source == "chicago":
        return (fetch_chicago_crimes(start, end, app_token=app_token),
                fetch_chicago_socioeconomics(app_token, population_csv))
    if source == "csv":
        if not csv_path or not demographics_csv:
            raise SystemExit("--source csv needs --csv and --demographics-csv")
        return pd.read_csv(csv_path), pd.read_csv(demographics_csv)
    raise SystemExit(f"unknown source: {source}")
