"""
Interaction specification and USDC comparison re-estimated with the current deviation |d_t|
among the predictors (the full model of full_model_analyses.py).

1. Interaction: the three interaction terms of the paper are added to the full model.
   - Out-of-sample: full vs full + interactions (paired block bootstrap, B = 2000).
   - Coefficients and the conditional slope theta(sigma^30) of r_{t-1} on the 70% window
     (B = 500, L = 14), exactly as in round2_analyses.conditional_slope.
2. USDC at 2 bp: regime-only model (published 0.507) reproduced, plus the full model with |d_t|
   and |d_t| alone.
Outputs: outputs/full_model/interaction_*.csv, usdc_*.csv
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.expanding_window_analysis import FEATURE_COLS, SEED  # noqa: E402
from src.round2_analyses import (oos_bootstrap, moving_block_bootstrap_indices,  # noqa: E402
                                 INTERACTIONS, BLOCK_LEN, B_COEF)
from src.robustness_analyses import expanding_logit  # noqa: E402
from src.full_model_analyses import load, FULL  # noqa: E402

OUT = REPO / "outputs" / "full_model"


def interaction():
    df = load(0.001)
    cols = list(FULL)
    for a, b in INTERACTIONS:
        df[f"{a}_x_{b}"] = df[a] * df[b]
        cols.append(f"{a}_x_{b}")
    y = df["depeg"].astype(int).values
    T0 = int(len(y) * 0.70)
    preds = {"Full": expanding_logit(df[FULL].values, y, T0),
             "Full + interactions": expanding_logit(df[cols].values, y, T0)}
    ci, diff, _ = oos_bootstrap(y[T0:], preds, ref="Full")
    ci.to_csv(OUT / "interaction_oos_ci.csv", index=False)
    diff.to_csv(OUT / "interaction_oos_diff.csv", index=False)
    print(ci.round(3).to_string(index=False)); print(diff.round(3).to_string(index=False))

    train = df.iloc[:T0]
    X, yt = train[cols], train["depeg"].astype(int)
    k_r, k_x = cols.index("ret_lag1"), cols.index("ret_lag1_x_ret_vol_30")
    grid = {f"P{q}": train["ret_vol_30"].quantile(q / 100) for q in (10, 25, 50, 75, 90)}

    def fit(Xb, yb):
        sc = StandardScaler().fit(Xb)
        m = LogisticRegression(max_iter=4000).fit(sc.transform(Xb), yb)
        c = m.coef_[0]
        s_r, s_x = sc.scale_[k_r], sc.scale_[k_x]
        slopes = np.array([(c[k_r] / s_r + c[k_x] * s / s_x) * s_r for s in grid.values()])
        return c, slopes

    c0, s0 = fit(X, yt)
    rng = np.random.default_rng(SEED)
    bc, bs = [], []
    for _ in range(B_COEF):
        idx = moving_block_bootstrap_indices(len(X), BLOCK_LEN, rng)
        if yt.iloc[idx].nunique() < 2:
            continue
        c, s = fit(X.iloc[idx], yt.iloc[idx])
        bc.append(c); bs.append(s)
    bc, bs = np.array(bc), np.array(bs)
    pv = lambda a: 2 * min((a <= 0).mean(), (a >= 0).mean())
    coefs = pd.DataFrame({"predictor": cols, "coef": c0,
                          "ci95_lo": np.percentile(bc, 2.5, axis=0), "ci95_hi": np.percentile(bc, 97.5, axis=0),
                          "p_boot": [pv(bc[:, k]) for k in range(len(cols))]})
    slope = pd.DataFrame({"quantile": list(grid), "sigma30": list(grid.values()), "theta": s0,
                          "odds_ratio": np.exp(s0),
                          "ci95_lo": np.percentile(bs, 2.5, axis=0), "ci95_hi": np.percentile(bs, 97.5, axis=0),
                          "p_boot": [pv(bs[:, k]) for k in range(len(grid))], "B_eff": len(bs)})
    coefs.to_csv(OUT / "interaction_coefs.csv", index=False)
    slope.to_csv(OUT / "interaction_slope.csv", index=False)
    print(coefs.round(3).to_string(index=False)); print(slope.round(4).to_string(index=False))


def usdc():
    u = pd.read_csv(REPO / "data" / "clean" / "usdc_kraken.csv").rename(columns={"close": "close_px", "volume": "volume_base"})
    u = u[pd.to_datetime(u["date"]) <= "2026-02-17"]  # common end date, as in usdc_expanding_window.py
    u = u.sort_values("date").reset_index(drop=True)
    u["abs_dev"] = (u["close_px"] - 1).abs()
    u["depeg"] = (u["abs_dev"] > 0.0002).astype(int).shift(-1)
    u["ret"] = np.log(u["close_px"]).diff()
    for k in (1, 2, 3):
        u[f"ret_lag{k}"] = u["ret"].shift(k)
    for w in (7, 14, 30):
        u[f"ret_vol_{w}"] = u["ret"].rolling(w).std()
    u["volume_lag1"] = u["volume_base"].shift(1)
    u["volume_lag3"] = u["volume_base"].shift(3)
    for w in (7, 14):
        u[f"volume_vol_{w}"] = u["volume_base"].rolling(w).std()
    u = u.dropna().reset_index(drop=True)
    y = u["depeg"].astype(int).values
    T0 = int(len(y) * 0.70)
    preds = {"Regime": expanding_logit(u[FEATURE_COLS].values, y, T0),
             "Full": expanding_logit(u[FULL].values, y, T0),
             "|d_t| only": expanding_logit(u[["abs_dev"]].values, y, T0)}
    ci, _, _ = oos_bootstrap(y[T0:], preds, ref="Regime")
    ci.to_csv(OUT / "usdc_ci.csv", index=False)
    print(f"USDC n={len(y)} T0={T0} rate={y.mean():.3f} test events={y[T0:].sum()}")
    print(ci.round(3).to_string(index=False))


if __name__ == "__main__":
    interaction()
    usdc()
