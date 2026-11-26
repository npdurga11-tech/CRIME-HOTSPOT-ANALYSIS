"""Link crime to neighbourhood socioeconomics: rates, rank correlations, OLS, spatial autocorrelation.

Everything is implemented with numpy/scipy (statsmodels / pysal are not required).

Unit of analysis is the zone (community area). Correlations at this level describe *places*;
they do not show that any individual resident is more or less likely to offend (ecological fallacy).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import SOCIO_COLS, SOCIO_LABELS


def zone_table(df: pd.DataFrame, demo: pd.DataFrame, df_geo: pd.DataFrame | None = None) -> pd.DataFrame:
    """Annualised incident counts and rates per zone, joined to the demographics."""
    years = (df["timestamp"].max() - df["timestamp"].min()).days / 365.25
    d = df.dropna(subset=["zone_id"]).copy()
    d["zone_id"] = d["zone_id"].astype(int)
    agg = d.groupby("zone_id").agg(
        incidents=("case_number", "size"),
        violent=("group", lambda s: int((s == "violent").sum())),
        property_=("group", lambda s: int((s == "property").sum())),
        severity=("severity", "sum"),
    ).reset_index().rename(columns={"property_": "property"})
    out = demo.merge(agg, on="zone_id", how="inner")
    out["incidents_per_year"] = out["incidents"] / years
    if out["population"].notna().all():
        out["rate_per_1000"] = out["incidents_per_year"] / out["population"] * 1000
        out["violent_rate_per_1000"] = out["violent"] / years / out["population"] * 1000
        out["severity_rate_per_1000"] = out["severity"] / years / out["population"] * 1000
    if df_geo is not None:
        cent = df_geo.dropna(subset=["zone_id"]).groupby("zone_id")[["latitude", "longitude"]].mean()
        out = out.merge(cent.reset_index().assign(zone_id=lambda x: x["zone_id"].astype(int)), on="zone_id", how="left")
    return out


def outcome_column(zones: pd.DataFrame) -> tuple[str, str]:
    if "rate_per_1000" in zones.columns:
        return "rate_per_1000", "incidents per 1,000 residents per year"
    return "incidents_per_year", "incidents per year (no population data: raw counts, confounded by zone size)"


def spearman_table(zones: pd.DataFrame, outcome: str, predictors: list[str] | None = None) -> pd.DataFrame:
    predictors = predictors or [c for c in SOCIO_COLS + ["hardship_index"] if c in zones.columns]
    rows = []
    for p in predictors:
        rho, pv = stats.spearmanr(zones[p], zones[outcome])
        rows.append({"indicator": SOCIO_LABELS.get(p, p), "column": p, "spearman_rho": rho, "p_value": pv,
                     "n_zones": int(zones[[p, outcome]].dropna().shape[0])})
    return pd.DataFrame(rows).sort_values("spearman_rho", ascending=False).reset_index(drop=True)


def vif_table(X: pd.DataFrame) -> pd.Series:
    """Variance inflation factors via regressing each predictor on the others."""
    out = {}
    Xv = X.to_numpy(dtype=float)
    for j, c in enumerate(X.columns):
        y = Xv[:, j]
        A = np.column_stack([np.ones(len(y)), np.delete(Xv, j, axis=1)])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        ss_res = ((y - A @ beta) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        out[c] = np.inf if r2 >= 1 else 1 / (1 - r2)
    return pd.Series(out)


def select_by_vif(X: pd.DataFrame, threshold: float = 5.0) -> list[str]:
    """Drop the most collinear predictor until every VIF is below the threshold."""
    cols = list(X.columns)
    while len(cols) > 1:
        v = vif_table(X[cols])
        if v.max() < threshold:
            break
        cols.remove(v.idxmax())
    return cols


def ols(zones: pd.DataFrame, outcome: str, predictors: list[str], log_outcome: bool = True) -> dict:
    """OLS of (log) outcome on standardised predictors. Returns coefficient table, fit stats, residuals."""
    y = np.log(zones[outcome].to_numpy(dtype=float)) if log_outcome else zones[outcome].to_numpy(dtype=float)
    X = zones[predictors].to_numpy(dtype=float)
    mu, sd = X.mean(axis=0), X.std(axis=0, ddof=0)
    Xs = (X - mu) / sd
    A = np.column_stack([np.ones(len(y)), Xs])
    n, k = A.shape
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    dof = n - k
    sigma2 = (resid ** 2).sum() / dof
    cov = sigma2 * np.linalg.inv(A.T @ A)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), dof)
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid ** 2).sum() / ss_tot
    names = ["intercept"] + predictors
    coef = pd.DataFrame({"term": names, "coef_per_1sd": beta, "std_err": se, "t": t, "p_value": p})
    # For a log outcome, exp(coef) - 1 is the % change in the rate for a +1 SD move in the predictor.
    if log_outcome:
        coef["pct_change_per_1sd"] = 100 * (np.exp(coef["coef_per_1sd"]) - 1)
        coef.loc[0, "pct_change_per_1sd"] = np.nan
    std_resid = resid / np.sqrt(sigma2)
    return {"coef": coef, "r2": float(r2), "adj_r2": float(1 - (1 - r2) * (n - 1) / dof), "n": int(n),
            "resid": resid, "std_resid": std_resid, "dof": int(dof)}


def morans_i(values: np.ndarray, xy_m: np.ndarray, k: int = 5, n_perm: int = 999, seed: int = 0) -> dict:
    """Global Moran's I with k-nearest-neighbour, row-standardised weights and a permutation p-value."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    d = np.linalg.norm(xy_m[:, None, :] - xy_m[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    W = np.zeros((n, n))
    for i in range(n):
        W[i, np.argsort(d[i])[:k]] = 1.0 / k
    z = v - v.mean()

    def stat(zz):
        return (n / W.sum()) * (zz @ W @ zz) / (zz @ zz)

    obs = stat(z)
    rng = np.random.default_rng(seed)
    perms = np.array([stat(rng.permutation(z)) for _ in range(n_perm)])
    p = (1 + (perms >= obs).sum()) / (n_perm + 1)          # one-sided: clustering
    return {"I": float(obs), "expected_I": float(-1 / (n - 1)), "p_value": float(p), "k": k, "n": n}
