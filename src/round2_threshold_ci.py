"""
Second-round revision: block-bootstrap intervals for the remaining
out-of-sample results, where feasible.

  - threshold sweep (manuscript Table 12): expanding-window logistic at
    5, 10, 15, 20 bps, as in revision2026/robustness_experiments.ipynb, D1;
  - USDC at 2 bps (manuscript Table 11), as in src/usdc_expanding_window.py.

Intervals: moving-block bootstrap of the out-of-sample forecast/outcome pairs
(block length 14, 2000 replications), as in src/round2_analyses.py. Draws
whose resampled outcomes contain a single class are skipped and counted;
an interval is reported only when at least 90% of draws are usable.

Produces outputs/round2/threshold_bootstrap_ci.csv
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.round2_analyses import moving_block_bootstrap_indices, BLOCK_LEN, B_OOS, SEED  # noqa: E402

DATA_DIR = REPO / "data" / "clean"
OUT_DIR = REPO / "outputs" / "round2"
FEATURES = ["ret_lag1", "ret_lag2", "ret_lag3", "ret_vol_7", "ret_vol_14", "ret_vol_30",
            "volume_lag1", "volume_lag3", "volume_vol_7", "volume_vol_14"]


def build(df, thresh, price, volume):
    df = df.sort_values("date").copy()
    df["depeg"] = ((df[price] - 1.0).abs() > thresh).astype(int).shift(-1)
    df["ret"] = np.log(df[price]).diff()
    for k in (1, 2, 3):
        df[f"ret_lag{k}"] = df["ret"].shift(k)
    for w in (7, 14, 30):
        df[f"ret_vol_{w}"] = df["ret"].rolling(w).std()
    df["volume_lag1"] = df[volume].shift(1)
    df["volume_lag3"] = df[volume].shift(3)
    for w in (7, 14):
        df[f"volume_vol_{w}"] = df[volume].rolling(w).std()
    return df.dropna().reset_index(drop=True)


def walk_forward(df):
    X, y = df[FEATURES].values, df["depeg"].astype(int).values
    T0 = int(len(y) * 0.70)
    p = np.zeros(len(y) - T0)
    for t in range(T0, len(y)):
        sc = StandardScaler().fit(X[:t])
        m = LogisticRegression(max_iter=2000, random_state=SEED).fit(sc.transform(X[:t]), y[:t])
        p[t - T0] = m.predict_proba(sc.transform(X[t:t + 1]))[0, 1]
    return y[T0:], p


def intervals(y, p):
    metrics = {"AUROC": roc_auc_score, "AUPRC": average_precision_score, "Brier": brier_score_loss}
    rng = np.random.default_rng(SEED)
    draws = {m: [] for m in metrics}
    for _ in range(B_OOS):
        idx = moving_block_bootstrap_indices(len(y), BLOCK_LEN, rng)
        if y[idx].min() == y[idx].max():
            continue
        for m, f in metrics.items():
            draws[m].append(f(y[idx], p[idx]))
    usable = len(draws["AUROC"]) / B_OOS
    row = {"events_test": int(y.sum()), "n_test": len(y), "usable_draws": usable}
    for m, f in metrics.items():
        row[m] = f(y, p) if y.min() != y.max() else np.nan
        ok = usable >= 0.9
        row[f"{m}_lo"] = np.percentile(draws[m], 2.5) if ok else np.nan
        row[f"{m}_hi"] = np.percentile(draws[m], 97.5) if ok else np.nan
    return row


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    usdt = (pd.read_csv(DATA_DIR / "usdt_kraken.csv", parse_dates=["date_utc"])
            .rename(columns={"date_utc": "date"})[["date", "close_px", "volume_base"]])
    for bps in (5, 10, 15, 20):
        d = build(usdt, bps / 1e4, "close_px", "volume_base")
        r = intervals(*walk_forward(d))
        r.update(series="USDT", threshold_bps=bps, event_rate=d["depeg"].mean())
        rows.append(r)

    usdc = pd.read_csv(DATA_DIR / "usdc_kraken.csv", parse_dates=["date"])
    usdc = usdc[usdc["date"] <= "2026-02-17"]
    d = build(usdc, 2 / 1e4, "close", "volume")
    r = intervals(*walk_forward(d))
    r.update(series="USDC", threshold_bps=2, event_rate=d["depeg"].mean())
    rows.append(r)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "threshold_bootstrap_ci.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
