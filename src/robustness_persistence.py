"""
Follow-up to robustness_analyses.py, R2.

The all-pair dispersion index correlates 0.91 with the same-day USDT deviation, because the
Binance pairs are USDT-quoted, so it partly measures the peg deviation itself. This script
separates the two channels:
  - K-C dispersion: mean |Kraken-Coinbase spread| over BTC and ETH (both USD-quoted, so free
    of USDT);
  - USD dispersion: mean |spread| over all three pairs after converting Binance to USD;
  - persistence: today's absolute USDT deviation |d_t| (a naive benchmark for d_{t+1} > tau).
All models use the paper's expanding-window logistic protocol and paired moving-block
bootstrap intervals against the 10-predictor baseline.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.expanding_window_analysis import FEATURE_COLS, THRESH  # noqa: E402
from src.round2_analyses import oos_bootstrap  # noqa: E402
from src.robustness_analyses import expanding_logit, coef_bootstrap  # noqa: E402

DATA_DIR = REPO / "data" / "clean"
OUT_DIR = REPO / "outputs" / "robustness"


def main():
    sp = pd.read_csv(DATA_DIR / "spreads_overlap_btc_eth_3venues.csv", parse_dates=["date_utc"])
    usdt = pd.read_csv(DATA_DIR / "usdt_kraken.csv", parse_dates=["date_utc"])
    sp = sp.merge(usdt[["date_utc", "close_px"]].rename(columns={"close_px": "u"}), on="date_utc")
    b_usd = sp["close_binance"] * sp["u"]
    sp["bk_usd"] = b_usd / sp["close_kraken"] - 1
    sp["bc_usd"] = b_usd / sp["close_coinbase"] - 1
    sp["kc_abs"] = sp["spread_rel_kc"].abs()
    sp["usd_abs"] = sp[["bk_usd", "bc_usd", "spread_rel_kc"]].abs().mean(axis=1)
    sp["raw_abs"] = sp[["spread_rel_bk", "spread_rel_bc", "spread_rel_kc"]].abs().mean(axis=1)
    d = sp.groupby("date_utc")[["kc_abs", "usd_abs", "raw_abs"]].mean().reset_index()
    d["date"] = d["date_utc"].dt.strftime("%Y-%m-%d")
    d = d.rename(columns={"kc_abs": "disp_kc", "usd_abs": "disp_usd", "raw_abs": "disp_raw"})

    df = pd.read_csv(DATA_DIR / "usdt_depeg_dataset.csv").sort_values("date").reset_index(drop=True)
    df["depeg"] = (df["abs_dev"] > THRESH).astype(int).shift(-1)
    df = df.dropna(subset=["depeg"]).merge(d[["date", "disp_kc", "disp_usd", "disp_raw"]],
                                           on="date", how="left").dropna().reset_index(drop=True)
    y = df["depeg"].astype(int).values
    T0 = int(len(y) * 0.70)
    base = list(FEATURE_COLS)
    specs = {
        "Baseline (10 predictors)": base,
        "Persistence |d_t| only": ["abs_dev"],
        "Baseline + |d_t|": base + ["abs_dev"],
        "K-C dispersion only": ["disp_kc"],
        "USD dispersion only": ["disp_usd"],
        "Baseline + K-C dispersion": base + ["disp_kc"],
        "Baseline + USD dispersion": base + ["disp_usd"],
        "Baseline + |d_t| + USD dispersion": base + ["abs_dev", "disp_usd"],
    }
    preds = {k: expanding_logit(df[c].values, y, T0) for k, c in specs.items()}
    # naive rule without any fitting: today's deviation itself as the score
    preds["Raw |d_t| (no model)"] = df["abs_dev"].values[T0:]
    ci, diff, _ = oos_bootstrap(y[T0:], preds, ref="Baseline (10 predictors)")
    ci.to_csv(OUT_DIR / "r3_oos_ci.csv", index=False)
    diff.to_csv(OUT_DIR / "r3_oos_diff.csv", index=False)
    corr = df[["abs_dev", "disp_raw", "disp_usd", "disp_kc", "ret_vol_30"]].corr()
    corr.to_csv(OUT_DIR / "r3_corr.csv")
    train = df.iloc[:T0]
    coefs = coef_bootstrap(train[base + ["abs_dev", "disp_usd"]], train["depeg"].astype(int))
    coefs.to_csv(OUT_DIR / "r3_coefs_70pct.csv", index=False)

    print(f"n={len(y)} T0={T0} test events={y[T0:].sum()}  train event rate={y[:T0].mean():.3f}")
    print("event persistence P(Y_t+1=1 | Y_t=1) =",
          round(df.loc[df.abs_dev > THRESH, "depeg"].mean(), 3),
          " P(Y_t+1=1 | Y_t=0) =", round(df.loc[df.abs_dev <= THRESH, "depeg"].mean(), 3))
    print(corr.round(3).to_string())
    print(ci[ci.metric != "Brier"].round(3).to_string(index=False))
    print(diff[diff.metric == "AUROC"][["comparison", "diff_point", "ci95_lo", "ci95_hi", "p_boot"]]
          .round(3).to_string(index=False))
    print(coefs.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
