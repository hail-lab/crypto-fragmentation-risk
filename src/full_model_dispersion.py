"""
Does cross-venue dispersion add to the full model (|d_t| + 10 regime predictors)?

Dispersion measures free of the USDT quote:
  USD dispersion: mean |spread| over B-K, B-C, K-C and both assets, after converting Binance to USD;
  K-C dispersion: mean |Kraken-Coinbase spread| over both assets.
Expanding-window logistic; paired moving-block bootstrap (L = 14, B = 2000) against the full model.
Coefficients of full + USD dispersion on the 70% window are in outputs/robustness/r3_coefs_70pct.csv.
Output: outputs/full_model/dispersion_vs_full_ci.csv, dispersion_vs_full_diff.csv
"""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.full_model_analyses import load, FULL  # noqa: E402
from src.robustness_analyses import expanding_logit  # noqa: E402
from src.round2_analyses import oos_bootstrap  # noqa: E402

DATA = REPO / "data" / "clean"
OUT = REPO / "outputs" / "full_model"


def main():
    sp = pd.read_csv(DATA / "spreads_overlap_btc_eth_3venues.csv", parse_dates=["date_utc"])
    u = pd.read_csv(DATA / "usdt_kraken.csv", parse_dates=["date_utc"])
    sp = sp.merge(u[["date_utc", "close_px"]].rename(columns={"close_px": "u"}), on="date_utc")
    b = sp["close_binance"] * sp["u"]
    sp["disp_usd"] = pd.concat([(b / sp["close_kraken"] - 1).abs(), (b / sp["close_coinbase"] - 1).abs(),
                                sp["spread_rel_kc"].abs()], axis=1).mean(axis=1)
    sp["disp_kc"] = sp["spread_rel_kc"].abs()
    d = sp.groupby("date_utc")[["disp_usd", "disp_kc"]].mean().reset_index()
    d["date"] = d["date_utc"].dt.strftime("%Y-%m-%d")
    df = load(0.001).merge(d[["date", "disp_usd", "disp_kc"]], on="date", how="left")
    assert df[["disp_usd", "disp_kc"]].notna().all().all()
    y = df["depeg"].astype(int).values
    T0 = int(len(y) * 0.70)
    preds = {"Full": expanding_logit(df[FULL].values, y, T0),
             "Full + USD dispersion": expanding_logit(df[FULL + ["disp_usd"]].values, y, T0),
             "Full + K-C dispersion": expanding_logit(df[FULL + ["disp_kc"]].values, y, T0)}
    ci, diff, _ = oos_bootstrap(y[T0:], preds, ref="Full")
    ci.to_csv(OUT / "dispersion_vs_full_ci.csv", index=False)
    diff.to_csv(OUT / "dispersion_vs_full_diff.csv", index=False)
    print(ci[ci.metric != "Brier"].round(3).to_string(index=False))
    print(diff[diff.metric != "Brier"][["comparison", "metric", "diff_point", "ci95_lo", "ci95_hi", "p_boot"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
