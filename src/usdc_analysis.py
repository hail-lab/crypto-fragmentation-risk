"""
Pull USDC/USD daily data from Kraken and run the same micro-depeg
prediction pipeline used for USDT, enabling cross-stablecoin comparison.
"""
from __future__ import annotations

import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data" / "clean"
OUT_DIR  = REPO / "outputs"

KRAKEN_BASE = "https://api.kraken.com/0/public/OHLC"
INTERVAL_MIN = 1440  # daily

TRAIN_FRAC = 0.70
SEED = 42
THRESH = 0.001  # 10 bps

FEATURE_COLS = [
    "ret_lag1", "ret_lag2", "ret_lag3",
    "ret_vol_7", "ret_vol_14", "ret_vol_30",
    "volume_lag1", "volume_lag3",
    "volume_vol_7", "volume_vol_14",
]


def fetch_kraken_ohlc(pair: str, since_date: str = "2024-02-28") -> pd.DataFrame:
    """Pull daily OHLC from Kraken public API."""
    since_ts = int(datetime.fromisoformat(since_date).replace(tzinfo=timezone.utc).timestamp())

    all_rows = []
    since = since_ts

    for _ in range(20):   # safety limit
        params = {"pair": pair, "interval": INTERVAL_MIN, "since": since}
        r = requests.get(KRAKEN_BASE, params=params, timeout=60)
        r.raise_for_status()
        js = r.json()
        if js.get("error"):
            raise RuntimeError(f"Kraken API error: {js['error']}")

        result = js["result"]
        data_key = [k for k in result.keys() if k != "last"][0]
        rows = result[data_key]

        if not rows:
            break

        all_rows.extend(rows)
        last_ts = int(result.get("last", since))
        if last_ts <= since:
            break
        since = last_ts
        time.sleep(0.5)

    df = pd.DataFrame(all_rows, columns=[
        "time", "open", "high", "low", "close", "vwap", "volume", "count"
    ])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["date"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.date.astype(str)
    df = df.drop_duplicates(subset=["date"], keep="last")
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", "close", "volume"]]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the same feature set as for USDT."""
    df = df.sort_values("date").copy()
    df["abs_dev"] = (df["close"] - 1.0).abs()
    df["is_offpeg"] = (df["abs_dev"] > THRESH).astype(int)
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


def run_classifiers(df: pd.DataFrame, stablecoin: str) -> pd.DataFrame:
    """Run all 5 classifiers on the dataset."""
    split = int(len(df) * TRAIN_FRAC)
    X_train = df.iloc[:split][FEATURE_COLS]
    y_train = df.iloc[:split]["depeg"].astype(int)
    X_test  = df.iloc[split:][FEATURE_COLS]
    y_test  = df.iloc[split:]["depeg"].astype(int)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=5, random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.8, random_state=SEED),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=SEED),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                             max_iter=500, early_stopping=True, validation_fraction=0.15, random_state=SEED),
    }

    rows = []
    for name, clf in classifiers.items():
        use_scaled = name in ("Logistic Regression", "SVM (RBF)", "MLP")
        Xtr = X_train_s if use_scaled else X_train
        Xte = X_test_s  if use_scaled else X_test

        clf.fit(Xtr, y_train)
        probs = clf.predict_proba(Xte)[:, 1]

        auroc = roc_auc_score(y_test, probs)
        auprc = average_precision_score(y_test, probs)
        brier = brier_score_loss(y_test, probs)
        rows.append({
            "Stablecoin": stablecoin, "Classifier": name,
            "AUROC": round(auroc, 3), "AUPRC": round(auprc, 3), "Brier": round(brier, 3),
        })
        print(f"  {stablecoin} {name:25s}  AUROC={auroc:.3f}  AUPRC={auprc:.3f}  Brier={brier:.3f}")

    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Pull USDC/USD from Kraken ---
    print("Pulling USDC/USD from Kraken...")
    usdc_raw = fetch_kraken_ohlc("USDCUSD", since_date="2024-02-01")
    usdc_raw.to_csv(DATA_DIR / "usdc_kraken.csv", index=False)
    print(f"  Saved {len(usdc_raw)} rows to usdc_kraken.csv")
    print(f"  Date range: {usdc_raw['date'].iloc[0]} to {usdc_raw['date'].iloc[-1]}")
    print(f"  Close range: {usdc_raw['close'].min():.6f} to {usdc_raw['close'].max():.6f}")

    # --- 2. Build features ---
    usdc = build_features(usdc_raw)
    event_rate = usdc["depeg"].mean()
    print(f"  USDC depeg rate at {THRESH*10000:.0f}bps: {event_rate:.3f} ({int(usdc['depeg'].sum())}/{len(usdc)})")
    print(f"  Usable observations: {len(usdc)}")

    # --- 3. Run classifiers on USDC ---
    print("\nUSDC classifier results:")
    usdc_results = run_classifiers(usdc, "USDC")

    # --- 4. Also run on USDT for direct comparison ---
    print("\nUSDT classifier results (re-run for comparison):")
    usdt_raw = pd.read_csv(DATA_DIR / "usdt_depeg_dataset.csv")
    usdt_raw = usdt_raw.sort_values("date").reset_index(drop=True)
    usdt_raw["is_offpeg"] = (usdt_raw["abs_dev"] > THRESH).astype(int)
    usdt_raw["depeg"] = usdt_raw["is_offpeg"].shift(-1)
    usdt_raw = usdt_raw.dropna(subset=["depeg"]).reset_index(drop=True)
    usdt_results = run_classifiers(usdt_raw, "USDT")

    # --- 5. Combined comparison ---
    combined = pd.concat([usdt_results, usdc_results], ignore_index=True)
    combined.to_csv(OUT_DIR / "cross_stablecoin_comparison.csv", index=False)
    print("\n--- Combined Cross-Stablecoin Comparison ---")
    print(combined.to_string(index=False))

    # --- 6. Summary: Logistic only for both ---
    logistic_only = combined[combined["Classifier"] == "Logistic Regression"]
    print("\n--- Logistic Regression: USDT vs USDC ---")
    print(logistic_only.to_string(index=False))


if __name__ == "__main__":
    main()
