"""
Cross-stablecoin comparison: USDC vs USDT with expanding-window evaluation.

Trims USDC data to end at the same date as USDT (Feb 17, 2026) so both
stablecoins share a common end-date.  Uses the same expanding-window
walk-forward protocol as the main USDT analysis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data" / "clean"
OUT_DIR  = REPO / "outputs"

SEED = 42
CUTOFF_DATE = "2026-02-17"  # align with USDT end-date

FEATURE_COLS = [
    "ret_lag1", "ret_lag2", "ret_lag3",
    "ret_vol_7", "ret_vol_14", "ret_vol_30",
    "volume_lag1", "volume_lag3",
    "volume_vol_7", "volume_vol_14",
]


def build_features(df: pd.DataFrame, thresh: float) -> pd.DataFrame:
    """Build features from raw price data."""
    df = df.sort_values("date").copy()
    df["abs_dev"] = (df["close"] - 1.0).abs()
    df["is_offpeg"] = (df["abs_dev"] > thresh).astype(int)
    df["depeg"] = df["is_offpeg"].shift(-1)

    df["ret"] = np.log(df["close"]).diff()
    df["ret_lag1"] = df["ret"].shift(1)
    df["ret_lag2"] = df["ret"].shift(2)
    df["ret_lag3"] = df["ret"].shift(3)

    for w in [7, 14, 30]:
        df[f"ret_vol_{w}"] = df["ret"].rolling(w).std()

    df["volume_lag1"] = df["volume"].shift(1)
    df["volume_lag3"] = df["volume"].shift(3)

    for w in [7, 14]:
        df[f"volume_vol_{w}"] = df["volume"].rolling(w).std()

    df = df.dropna().reset_index(drop=True)
    return df


def expanding_window_logistic(df: pd.DataFrame) -> dict:
    """Run expanding-window walk-forward logistic regression."""
    X = df[FEATURE_COLS].values
    y = df["depeg"].astype(int).values
    N = len(y)
    T0 = int(N * 0.70)

    probs = np.full(N, np.nan)

    for t in range(T0, N):
        X_train, y_train = X[:t], y[:t]
        X_test = X[t:t+1]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=2000, random_state=SEED)
        clf.fit(X_train_s, y_train)
        probs[t] = clf.predict_proba(X_test_s)[:, 1][0]

    mask = ~np.isnan(probs)
    y_test = y[mask]
    p_test = probs[mask]

    if len(np.unique(y_test)) < 2:
        return {"AUROC": np.nan, "AUPRC": np.nan, "Brier": np.nan,
                "N": int(mask.sum()), "T0": T0}

    return {
        "AUROC": round(roc_auc_score(y_test, p_test), 3),
        "AUPRC": round(average_precision_score(y_test, p_test), 3),
        "Brier": round(brier_score_loss(y_test, p_test), 3),
        "N": int(mask.sum()),
        "T0": T0,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load and trim USDC ----
    usdc_raw = pd.read_csv(DATA_DIR / "usdc_kraken.csv")
    usdc_raw["date"] = pd.to_datetime(usdc_raw["date"])
    usdc_raw = usdc_raw[usdc_raw["date"] <= CUTOFF_DATE].copy()
    usdc_raw = usdc_raw.sort_values("date").reset_index(drop=True)

    # ---- Load USDT (already ends at Feb 17) ----
    usdt_raw = pd.read_csv(DATA_DIR / "usdt_kraken.csv")
    usdt_raw = usdt_raw.rename(columns={"date_utc": "date",
                                         "close_px": "close",
                                         "volume_base": "volume"})
    usdt_raw = usdt_raw[["date", "close", "volume"]].copy()
    usdt_raw["date"] = pd.to_datetime(usdt_raw["date"])

    print("=" * 60)
    print("DATA PERIODS (ALIGNED)")
    print("=" * 60)
    print(f"  USDT: {usdt_raw['date'].min().date()} to {usdt_raw['date'].max().date()}  "
          f"({len(usdt_raw)} obs)")
    print(f"  USDC: {usdc_raw['date'].min().date()} to {usdc_raw['date'].max().date()}  "
          f"({len(usdc_raw)} obs)")

    # ---- Deviation statistics ----
    usdt_dev = (usdt_raw["close"] - 1.0).abs()
    usdc_dev = (usdc_raw["close"] - 1.0).abs()

    print("\n" + "=" * 60)
    print("DEVIATION COMPARISON (basis points)")
    print("=" * 60)
    print(f"  {'Metric':<20s} {'USDT':>10s} {'USDC':>10s}")
    print("  " + "-" * 42)
    print(f"  {'Observations':<20s} {len(usdt_raw):>10d} {len(usdc_raw):>10d}")
    print(f"  {'Mean |dev|':<20s} {usdt_dev.mean()*10_000:>10.2f} {usdc_dev.mean()*10_000:>10.2f}")
    print(f"  {'Std |dev|':<20s} {usdt_dev.std()*10_000:>10.2f} {usdc_dev.std()*10_000:>10.2f}")
    print(f"  {'Max |dev|':<20s} {usdt_dev.max()*10_000:>10.2f} {usdc_dev.max()*10_000:>10.2f}")
    print(f"  {'P95 |dev|':<20s} {usdt_dev.quantile(0.95)*10_000:>10.2f} {usdc_dev.quantile(0.95)*10_000:>10.2f}")
    print(f"  {'P99 |dev|':<20s} {usdt_dev.quantile(0.99)*10_000:>10.2f} {usdc_dev.quantile(0.99)*10_000:>10.2f}")

    for t_bps in [2, 3, 5, 10, 15, 20]:
        t = t_bps / 10_000
        usdt_rate = (usdt_dev > t).mean()
        usdc_rate = (usdc_dev > t).mean()
        print(f"  {'Events >' + str(t_bps) + ' bps':<20s} {usdt_rate:>10.3f} {usdc_rate:>10.3f}")

    # ---- USDC: expanding-window logistic at 2 bps ----
    usdc_feat = build_features(usdc_raw.copy(), thresh=0.0002)
    usdc_event_rate = usdc_feat["depeg"].mean()
    usdc_n_events = int(usdc_feat["depeg"].sum())

    print("\n" + "=" * 60)
    print("USDC EXPANDING-WINDOW LOGISTIC (2 bps threshold)")
    print("=" * 60)
    print(f"  Observations: {len(usdc_feat)}, Events: {usdc_n_events}, "
          f"Rate: {usdc_event_rate:.3f}")

    usdc_results = expanding_window_logistic(usdc_feat)
    print(f"  T0 (70%): {usdc_results['T0']}, Test steps: {usdc_results['N']}")
    print(f"  AUROC: {usdc_results['AUROC']}")
    print(f"  AUPRC: {usdc_results['AUPRC']}")
    print(f"  Brier: {usdc_results['Brier']}")

    # ---- USDT reference (already computed elsewhere, just show data stats) ----
    usdt_feat = build_features(usdt_raw.copy(), thresh=0.001)
    usdt_event_rate = usdt_feat["depeg"].mean()
    usdt_n_events = int(usdt_feat["depeg"].sum())

    print("\n" + "=" * 60)
    print("USDT EXPANDING-WINDOW LOGISTIC (10 bps threshold) — reference from main analysis")
    print("=" * 60)
    print(f"  Observations: {len(usdt_feat)}, Events: {usdt_n_events}, "
          f"Rate: {usdt_event_rate:.3f}")
    print("  AUROC: 0.832  (from expanding_window_analysis.py)")
    print("  AUPRC: 0.422")
    print("  Brier: 0.117")

    # ---- Save results ----
    results = pd.DataFrame([
        {"Stablecoin": "USDT", "Threshold (bps)": 10,
         "Event Rate": round(usdt_event_rate, 3), "N": len(usdt_feat),
         "AUROC": 0.832, "AUPRC": 0.422, "Brier": 0.117},
        {"Stablecoin": "USDC", "Threshold (bps)": 2,
         "Event Rate": round(usdc_event_rate, 3), "N": len(usdc_feat),
         "AUROC": usdc_results["AUROC"],
         "AUPRC": usdc_results["AUPRC"],
         "Brier": usdc_results["Brier"]},
    ])
    out_path = OUT_DIR / "cross_stablecoin_expanding.csv"
    results.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    # ---- Summary for paper ----
    print("\n" + "=" * 60)
    print("SUMMARY FOR PAPER.TEX")
    print("=" * 60)
    print(f"  USDT period: {usdt_raw['date'].min().date()} -- {usdt_raw['date'].max().date()}")
    print(f"  USDC period: {usdc_raw['date'].min().date()} -- {usdc_raw['date'].max().date()}")
    print(f"  USDT obs (raw): {len(usdt_raw)}")
    print(f"  USDC obs (raw): {len(usdc_raw)}")
    print(f"  USDC obs (usable after features): {len(usdc_feat)}")
    print(f"  USDC max deviation: {usdc_dev.max()*10_000:.1f} bps")
    print(f"  USDC AUROC @ 2 bps: {usdc_results['AUROC']}")


if __name__ == "__main__":
    main()
