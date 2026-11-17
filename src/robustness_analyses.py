"""
Robustness analyses.

R1. USD conversion of Binance prices.
    Binance BTC and ETH are quoted in USDT. Here every Binance close is converted to USD with
    the same-day Kraken USDT/USD close (close_usd = close_usdt * P_USDT/USD). The paper's
    cross-venue spreads (Table 1), variance decomposition (Table 2) and volatility / VaR
    comparison (Table 3) are then recomputed. Each computation first reproduces the
    published, unconverted values as a check.

R2. Do cross-venue spreads predict USDT micro-depegs?
    A daily cross-venue dispersion index is added to the baseline predictors: the mean
    absolute relative spread across the three venue pairs and both assets, known at day t.
    Four logistic models are compared under the paper's expanding-window protocol:
      dispersion only | 30-day volatility only | baseline (10 predictors) | baseline + dispersion,
    using paired moving-block bootstrap intervals (L = 14, B = 2000) on the out-of-sample
    forecasts. Coefficients on the initial 70% window use the coefficient bootstrap
    (L = 14, B = 500).

Outputs: outputs/robustness/
"""
from __future__ import annotations

from itertools import permutations
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data" / "clean"
OUT_DIR = REPO / "outputs" / "robustness"
sys.path.insert(0, str(REPO))

from src.expanding_window_analysis import FEATURE_COLS, THRESH  # noqa: E402
from src.round2_analyses import (  # noqa: E402
    decompose, moving_block_bootstrap_indices, oos_bootstrap, BLOCK_LEN, B_COEF, SEED,
)

VENUES = ["binance", "kraken", "coinbase"]
PAIRS = [("binance", "kraken", "B-K"), ("binance", "coinbase", "B-C"), ("kraken", "coinbase", "K-C")]


# ---------------------------------------------------------------- R1
def nw_pvalue(x: pd.Series, lags: int = 6) -> float:
    res = sm.OLS(x.values, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(res.pvalues[0])


def spreads_table(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for asset, g in panel.groupby("asset"):
        for i, j, tag in PAIRS:
            s = g[f"close_{i}"] / g[f"close_{j}"] - 1
            rows.append({"asset": asset, "pair": tag, "mean": s.mean(), "std": s.std(),
                         "min": s.min(), "max": s.max(), "p_nw": nw_pvalue(s)})
    return pd.DataFrame(rows)


def decomposition_table(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for asset, g in panel.groupby("asset"):
        g = g.sort_values("date_utc").copy()
        for v in VENUES:
            g[f"ret_{v}"] = np.log(g[f"close_{v}"]).diff()
        full = g.dropna()
        rr = pd.DataFrame([decompose(full[f"ret_{i}"], full[f"ret_{j}"])
                           for i, j in permutations(VENUES, 2)])
        rows.append({"asset": asset, "n": len(full),
                     "var_delta_mean": rr.share_var_delta_pct.mean(),
                     "var_delta_max": rr.share_var_delta_pct.max(),
                     "two_cov_min": rr.share_two_cov_pct.min(),
                     "two_cov_max": rr.share_two_cov_pct.max(),
                     "net_abs_mean": rr.net_venue_pct.abs().mean(),
                     "net_abs_max": rr.net_venue_pct.abs().max(),
                     "corr_min": rr.corr_ij.min()})
    return pd.DataFrame(rows)


def risk_table(panel: pd.DataFrame, lam: float = 0.94) -> pd.DataFrame:
    rows = []
    for asset, g in panel.groupby("asset"):
        g = g.sort_values("date_utc")
        for v in VENUES:
            r = np.log(g[f"close_{v}"]).diff().dropna().values
            var = np.empty(len(r))
            var[0] = r[0] ** 2
            for t in range(1, len(r)):
                var[t] = lam * var[t - 1] + (1 - lam) * r[t - 1] ** 2
            rows.append({"asset": asset, "venue": v, "sigma": r.std(ddof=1),
                         "sigma_ewma_last": np.sqrt(var[-1]),
                         "var5": np.quantile(r, 0.05), "var1": np.quantile(r, 0.01)})
    return pd.DataFrame(rows)


def r1(panel: pd.DataFrame, usdt: pd.DataFrame) -> dict:
    conv = panel.merge(usdt[["date_utc", "close_px"]].rename(columns={"close_px": "usdt_usd"}),
                       on="date_utc", how="left")
    missing = int(conv["usdt_usd"].isna().sum())
    conv["close_binance"] = conv["close_binance"] * conv["usdt_usd"]
    out = {"missing_usdt_days": missing}
    for name, p in (("published", panel), ("usd_converted", conv.drop(columns="usdt_usd"))):
        out[f"spreads_{name}"] = spreads_table(p)
        out[f"decomp_{name}"] = decomposition_table(p)
        out[f"risk_{name}"] = risk_table(p)
    return out


# ---------------------------------------------------------------- R2
def dispersion_index(spreads: pd.DataFrame) -> pd.DataFrame:
    s = spreads.copy()
    s["abs_mean"] = s[["spread_rel_bk", "spread_rel_bc", "spread_rel_kc"]].abs().mean(axis=1)
    d = s.groupby("date_utc")["abs_mean"].mean().rename("dispersion").reset_index()
    d["date"] = d["date_utc"].dt.strftime("%Y-%m-%d")
    return d[["date", "dispersion"]]


def expanding_logit(X: np.ndarray, y: np.ndarray, T0: int) -> np.ndarray:
    p = np.zeros(len(y) - T0)
    for t in range(T0, len(y)):
        sc = StandardScaler().fit(X[:t])
        m = LogisticRegression(max_iter=2000, random_state=SEED).fit(sc.transform(X[:t]), y[:t])
        p[t - T0] = m.predict_proba(sc.transform(X[t:t + 1]))[0, 1]
    return p


def coef_bootstrap(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    def fit(Xb, yb):
        sc = StandardScaler().fit(Xb)
        return LogisticRegression(max_iter=4000).fit(sc.transform(Xb), yb).coef_[0]
    point = fit(X, y)
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(B_COEF):
        idx = moving_block_bootstrap_indices(len(X), BLOCK_LEN, rng)
        if y.iloc[idx].nunique() < 2:
            continue
        boot.append(fit(X.iloc[idx], y.iloc[idx]))
    boot = np.array(boot)
    return pd.DataFrame({"predictor": X.columns, "coef": point,
                         "ci95_lo": np.percentile(boot, 2.5, axis=0),
                         "ci95_hi": np.percentile(boot, 97.5, axis=0),
                         "p_boot": [2 * min((boot[:, k] <= 0).mean(), (boot[:, k] >= 0).mean())
                                    for k in range(boot.shape[1])],
                         "B_eff": len(boot)})


def r2(df: pd.DataFrame, disp: pd.DataFrame):
    df = df.merge(disp, on="date", how="left")
    n_missing = int(df["dispersion"].isna().sum())
    df = df.dropna(subset=["dispersion"]).reset_index(drop=True)
    y = df["depeg"].astype(int).values
    T0 = int(len(y) * 0.70)
    specs = {
        "Dispersion only": ["dispersion"],
        "Volatility (30-day) only": ["ret_vol_30"],
        "Baseline (10 predictors)": list(FEATURE_COLS),
        "Baseline + dispersion": list(FEATURE_COLS) + ["dispersion"],
    }
    preds = {k: expanding_logit(df[c].values, y, T0) for k, c in specs.items()}
    ci, diff, skipped = oos_bootstrap(y[T0:], preds, ref="Baseline (10 predictors)")
    train = df.iloc[:T0]
    coefs = coef_bootstrap(train[list(FEATURE_COLS) + ["dispersion"]], train["depeg"].astype(int))
    corr = df[["dispersion", "ret_vol_30", "ret_vol_7", "abs_dev"]].corr()
    info = {"n": len(y), "T0": T0, "test": len(y) - T0, "test_events": int(y[T0:].sum()),
            "missing_dispersion_days": n_missing, "skipped_draws": skipped}
    return ci, diff, coefs, corr, info


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(DATA_DIR / "panel_overlap_btc_eth_3venues.csv", parse_dates=["date_utc"])
    spreads = pd.read_csv(DATA_DIR / "spreads_overlap_btc_eth_3venues.csv", parse_dates=["date_utc"])
    usdt = pd.read_csv(DATA_DIR / "usdt_kraken.csv", parse_dates=["date_utc"])

    res = r1(panel, usdt)
    print("R1 missing USDT/USD days in panel:", res["missing_usdt_days"])
    for k, v in res.items():
        if isinstance(v, pd.DataFrame):
            v.to_csv(OUT_DIR / f"r1_{k}.csv", index=False)
            print(f"\n== R1 {k}\n{v.round(6).to_string(index=False)}")

    df = pd.read_csv(DATA_DIR / "usdt_depeg_dataset.csv").sort_values("date").reset_index(drop=True)
    df["depeg"] = (df["abs_dev"] > THRESH).astype(int).shift(-1)
    df = df.dropna(subset=["depeg"]).reset_index(drop=True)
    ci, diff, coefs, corr, info = r2(df, dispersion_index(spreads))
    print("\nR2", info)
    for name, t in (("oos_ci", ci), ("oos_diff", diff), ("coefs_70pct", coefs), ("corr", corr)):
        t.to_csv(OUT_DIR / f"r2_{name}.csv", index=(name == "corr"))
        print(f"\n== R2 {name}\n{t.round(3).to_string()}")


if __name__ == "__main__":
    main()
