"""
Onset analysis: can the model anticipate NEW micro-depegs?

Persistence is strong (P(Y_{t+1}=1 | Y_t=1) = 0.57 vs 0.05), so the naive rule "today's
deviation" scores AUROC 0.94 on all days. The early-warning question is about days on which
the peg is still intact (d_t <= tau): does anything predict that tomorrow breaches tau?

The expanding-window forecasts are the same as in robustness_persistence.py (trained on all
days). They are evaluated on the out-of-sample days with d_t <= tau only, with
moving-block bootstrap intervals on that subset (B = 2000, L = 14).
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.expanding_window_analysis import FEATURE_COLS, THRESH  # noqa: E402
from src.round2_analyses import oos_bootstrap  # noqa: E402
from src.robustness_analyses import expanding_logit  # noqa: E402

DATA_DIR = REPO / "data" / "clean"
OUT_DIR = REPO / "outputs" / "robustness"


def main():
    df = pd.read_csv(DATA_DIR / "usdt_depeg_dataset.csv").sort_values("date").reset_index(drop=True)
    df["depeg"] = (df["abs_dev"] > THRESH).astype(int).shift(-1)
    df = df.dropna(subset=["depeg"]).reset_index(drop=True)
    y = df["depeg"].astype(int).values
    T0 = int(len(y) * 0.70)
    base = list(FEATURE_COLS)
    specs = {"Baseline (10 predictors)": base, "Persistence |d_t| only": ["abs_dev"],
             "Baseline + |d_t|": base + ["abs_dev"], "Volatility (30-day) only": ["ret_vol_30"]}
    preds = {k: expanding_logit(df[c].values, y, T0) for k, c in specs.items()}
    intact = (df["abs_dev"].values[T0:] <= THRESH)
    y_on = y[T0:][intact]
    p_on = {k: v[intact] for k, v in preds.items()}
    ci, diff, skipped = oos_bootstrap(y_on, p_on, ref="Baseline (10 predictors)")
    ci.to_csv(OUT_DIR / "r4_onset_ci.csv", index=False)
    diff.to_csv(OUT_DIR / "r4_onset_diff.csv", index=False)
    print(f"test days={len(y) - T0}, intact days={intact.sum()}, onset events={y_on.sum()}, "
          f"continuation events={(y[T0:][~intact]).sum()} of {(~intact).sum()} breached days; "
          f"skipped draws={skipped}")
    print(ci[ci.metric != "Brier"].round(3).to_string(index=False))
    print(diff[diff.metric == "AUROC"][["comparison", "diff_point", "ci95_lo", "ci95_hi", "p_boot"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
