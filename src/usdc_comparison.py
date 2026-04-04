"""
Cross-stablecoin comparison: USDT vs USDC micro-depeg prediction.

Uses threshold that produces comparable event rates:
  - USDT at 10 bps (~10.1% event rate)
  - USDC at 2 bps  (~16.9% event rate)

Also documents that USDC never exceeds 5 bps while USDT regularly exceeds 10 bps.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data" / "clean"
OUT_DIR  = REPO / "outputs"

TRAIN_FRAC = 0.70
SEED = 42

FEATURE_COLS = [
    "ret_lag1", "ret_lag2", "ret_lag3",
    "ret_vol_7", "ret_vol_14", "ret_vol_30",
    "volume_lag1", "volume_lag3",
    "volume_vol_7", "volume_vol_14",
]


def build_features_from_raw(df: pd.DataFrame, thresh: float) -> pd.DataFrame:
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


def run_logistic(df: pd.DataFrame) -> dict:
    """Run logistic regression and return metrics."""
    split = int(len(df) * TRAIN_FRAC)
    X_train = df.iloc[:split][FEATURE_COLS]
    y_train = df.iloc[:split]["depeg"].astype(int)
    X_test  = df.iloc[split:][FEATURE_COLS]
    y_test  = df.iloc[split:]["depeg"].astype(int)

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        return {"AUROC": np.nan, "AUPRC": np.nan, "Brier": np.nan}

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=2000, random_state=SEED)
    clf.fit(X_train_s, y_train)
    probs = clf.predict_proba(X_test_s)[:, 1]

    return {
        "AUROC": round(roc_auc_score(y_test, probs), 3),
        "AUPRC": round(average_precision_score(y_test, probs), 3),
        "Brier": round(brier_score_loss(y_test, probs), 3),
        "coefs": dict(zip(FEATURE_COLS, clf.coef_[0])),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load USDC raw
    usdc_raw = pd.read_csv(DATA_DIR / "usdc_kraken.csv")
    usdc_raw = usdc_raw.rename(columns={"close": "close", "volume": "volume"})

    # Load USDT raw
    usdt_raw = pd.read_csv(DATA_DIR / "usdt_kraken.csv")
    usdt_raw = usdt_raw.rename(columns={"date_utc": "date", "close_px": "close", "volume_base": "volume"})
    usdt_raw = usdt_raw[["date", "close", "volume"]].copy()

    # ---- Deviation comparison ----
    print("=" * 60)
    print("CROSS-STABLECOIN DEVIATION COMPARISON")
    print("=" * 60)

    usdt_dev = (usdt_raw["close"] - 1.0).abs()
    usdc_dev = (usdc_raw["close"] - 1.0).abs()

    print(f"\n{'Metric':<25s} {'USDT':>10s} {'USDC':>10s}")
    print("-" * 47)
    print(f"{'Observations':<25s} {len(usdt_raw):>10d} {len(usdc_raw):>10d}")
    print(f"{'Mean |dev| (bps)':<25s} {usdt_dev.mean()*10000:>10.2f} {usdc_dev.mean()*10000:>10.2f}")
    print(f"{'Std |dev| (bps)':<25s} {usdt_dev.std()*10000:>10.2f} {usdc_dev.std()*10000:>10.2f}")
    print(f"{'Max |dev| (bps)':<25s} {usdt_dev.max()*10000:>10.2f} {usdc_dev.max()*10000:>10.2f}")
    print(f"{'P95 |dev| (bps)':<25s} {usdt_dev.quantile(0.95)*10000:>10.2f} {usdc_dev.quantile(0.95)*10000:>10.2f}")
    print(f"{'P99 |dev| (bps)':<25s} {usdt_dev.quantile(0.99)*10000:>10.2f} {usdc_dev.quantile(0.99)*10000:>10.2f}")

    for t_bps in [2, 3, 5, 10, 15, 20]:
        t = t_bps / 10000
        usdt_rate = (usdt_dev > t).mean()
        usdc_rate = (usdc_dev > t).mean()
        print(f"{'Events >' + str(t_bps) + ' bps':<25s} {usdt_rate:>10.3f} {usdc_rate:>10.3f}")

    # ---- USDC at 2 bps threshold ----
    print("\n" + "=" * 60)
    print("USDC PREDICTION AT 2-BPS THRESHOLD")
    print("=" * 60)

    usdc_2bps = build_features_from_raw(usdc_raw.copy(), thresh=0.0002)
    event_rate = usdc_2bps["depeg"].mean()
    n_events = int(usdc_2bps["depeg"].sum())
    print(f"  Observations: {len(usdc_2bps)}, Events: {n_events}, Rate: {event_rate:.3f}")

    usdc_metrics = run_logistic(usdc_2bps)
    print(f"  Logistic AUROC: {usdc_metrics['AUROC']}")
    print(f"  Logistic AUPRC: {usdc_metrics['AUPRC']}")
    print(f"  Logistic Brier: {usdc_metrics['Brier']}")

    if "coefs" in usdc_metrics:
        print("\n  Logistic coefficients (USDC, 2 bps):")
        for feat, coef in sorted(usdc_metrics["coefs"].items(), key=lambda x: abs(x[1]), reverse=True):
            print(f"    {feat:20s}: {coef:+.4f}")

    # ---- USDT at 10 bps threshold (comparison) ----
    print("\n" + "=" * 60)
    print("USDT PREDICTION AT 10-BPS THRESHOLD (reference)")
    print("=" * 60)

    usdt_10bps = build_features_from_raw(usdt_raw.copy(), thresh=0.001)
    event_rate = usdt_10bps["depeg"].mean()
    n_events = int(usdt_10bps["depeg"].sum())
    print(f"  Observations: {len(usdt_10bps)}, Events: {n_events}, Rate: {event_rate:.3f}")

    usdt_metrics = run_logistic(usdt_10bps)
    print(f"  Logistic AUROC: {usdt_metrics['AUROC']}")
    print(f"  Logistic AUPRC: {usdt_metrics['AUPRC']}")
    print(f"  Logistic Brier: {usdt_metrics['Brier']}")

    # ---- Save comparison table ----
    comparison = pd.DataFrame([
        {"Stablecoin": "USDT", "Threshold (bps)": 10, "Event Rate": 0.101,
         "N": len(usdt_10bps), "AUROC": usdt_metrics["AUROC"],
         "AUPRC": usdt_metrics["AUPRC"], "Brier": usdt_metrics["Brier"]},
        {"Stablecoin": "USDC", "Threshold (bps)": 2, "Event Rate": round(usdc_2bps["depeg"].mean(), 3),
         "N": len(usdc_2bps), "AUROC": usdc_metrics["AUROC"],
         "AUPRC": usdc_metrics["AUPRC"], "Brier": usdc_metrics["Brier"]},
    ])
    comparison.to_csv(OUT_DIR / "cross_stablecoin_logistic.csv", index=False)
    print("\n--- Cross-Stablecoin Logistic Comparison ---")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
