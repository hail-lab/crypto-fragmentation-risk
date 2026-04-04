"""
Expanding-window evaluation for:
  1. Threshold robustness (logistic only, 10 / 15 / 20 bps)
  2. Baseline vs interaction logistic specification (10 bps)

Uses the same true expanding-window protocol as expanding_window_analysis.py:
for each forecast origin t = T0 ... N-1, the model is re-estimated on
observations {0,...,t-1} and a one-step-ahead prediction is recorded for t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data" / "clean"
OUT_DIR  = REPO / "outputs"

SEED = 42
THRESHOLDS = [0.0010, 0.0015, 0.0020]  # 10, 15, 20 bps

FEATURE_COLS = [
    "ret_lag1", "ret_lag2", "ret_lag3",
    "ret_vol_7", "ret_vol_14", "ret_vol_30",
    "volume_lag1", "volume_lag3",
    "volume_vol_7", "volume_vol_14",
]

INTERACTION_FEATURES = [
    ("ret_lag1", "ret_vol_30"),   # lagged return × 30-day vol
    ("ret_lag1", "ret_vol_7"),    # lagged return × 7-day vol
    ("volume_lag3", "ret_vol_30"),  # lagged volume × 30-day vol
]


def expanding_window_logistic(X, y, T0):
    """Expanding-window predictions for logistic regression."""
    N = len(y)
    preds = np.zeros(N - T0)
    for t in range(T0, N):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[:t])
        X_pt = scaler.transform(X[t:t+1])
        clf = LogisticRegression(max_iter=2000, random_state=SEED)
        clf.fit(X_tr, y[:t])
        preds[t - T0] = clf.predict_proba(X_pt)[:, 1][0]
    return preds


def safe_metric(y_true, probs, metric_fn):
    if len(set(y_true)) < 2:
        return float("nan")
    return metric_fn(y_true, probs)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # Part 1: Threshold robustness with expanding window
    # ================================================================
    print("=" * 60)
    print("PART 1: Threshold Robustness (Expanding Window)")
    print("=" * 60)

    # Load raw USDT data to build datasets at various thresholds
    usdt_path = DATA_DIR / "usdt_kraken.csv"
    if not usdt_path.exists():
        # Try alternative location
        usdt_path = REPO / "data_clean" / "usdt_kraken.csv"
    usdt = pd.read_csv(usdt_path)
    usdt = usdt.rename(columns={"date_utc": "date"})
    usdt["date"] = pd.to_datetime(usdt["date"])
    usdt = usdt[["date", "close_px", "volume_base"]].copy()
    usdt = usdt.sort_values("date").reset_index(drop=True)

    thresh_rows = []

    for thresh in THRESHOLDS:
        # Build features from raw data
        df = usdt.copy()
        df["abs_dev"] = (df["close_px"] - 1.0).abs()
        df["is_offpeg"] = (df["abs_dev"] > thresh).astype(int)
        df["depeg"] = df["is_offpeg"].shift(-1)
        df["ret"] = np.log(df["close_px"]).diff()
        df["ret_lag1"] = df["ret"].shift(1)
        df["ret_lag2"] = df["ret"].shift(2)
        df["ret_lag3"] = df["ret"].shift(3)
        for w in [7, 14, 30]:
            df[f"ret_vol_{w}"] = df["ret"].rolling(w).std()
        df["volume_lag1"] = df["volume_base"].shift(1)
        df["volume_lag3"] = df["volume_base"].shift(3)
        for w in [7, 14]:
            df[f"volume_vol_{w}"] = df["volume_base"].rolling(w).std()
        df = df.dropna().reset_index(drop=True)

        X = df[FEATURE_COLS].values
        y = df["depeg"].astype(int).values
        N = len(y)
        T0 = int(N * 0.70)

        event_rate = y.mean()
        bps = int(round(thresh * 10000))
        print(f"\n  Threshold: {bps} bps  |  N={N}, T0={T0}, event_rate={event_rate:.3f}")

        preds = expanding_window_logistic(X, y, T0)
        y_test = y[T0:]

        auroc = safe_metric(y_test, preds, roc_auc_score)
        auprc = safe_metric(y_test, preds, average_precision_score)
        brier = brier_score_loss(y_test, preds)

        thresh_rows.append({
            "threshold_bps": bps,
            "event_rate": round(event_rate, 3),
            "AUROC": round(auroc, 3) if not np.isnan(auroc) else "---",
            "AUPRC": round(auprc, 3) if not np.isnan(auprc) else "---",
            "Brier": round(brier, 3),
        })
        print(f"  AUROC={auroc:.3f}  AUPRC={auprc:.3f}  Brier={brier:.3f}")

    thresh_df = pd.DataFrame(thresh_rows)
    thresh_df.to_csv(OUT_DIR / "threshold_robustness_expanding.csv", index=False)
    print("\n" + thresh_df.to_string(index=False))

    # ================================================================
    # Part 2: Baseline vs Interaction (Expanding Window, 10 bps)
    # ================================================================
    print("\n" + "=" * 60)
    print("PART 2: Baseline vs Interaction Model (Expanding Window)")
    print("=" * 60)

    # Use the already-processed 10 bps dataset
    df10 = pd.read_csv(DATA_DIR / "usdt_depeg_dataset.csv")
    df10 = df10.sort_values("date").reset_index(drop=True)
    df10["is_offpeg"] = (df10["abs_dev"] > 0.001).astype(int)
    df10["depeg"] = df10["is_offpeg"].shift(-1)
    df10 = df10.dropna(subset=["depeg"]).reset_index(drop=True)
    df10["depeg"] = df10["depeg"].astype(int)

    # Baseline features
    X_base = df10[FEATURE_COLS].values
    y10 = df10["depeg"].values
    N10 = len(y10)
    T0_10 = int(N10 * 0.70)

    # Interaction features (13 = 10 base + 3 interactions)
    interaction_cols = FEATURE_COLS.copy()
    for col_a, col_b in INTERACTION_FEATURES:
        iname = f"{col_a}_x_{col_b}"
        df10[iname] = df10[col_a] * df10[col_b]
        interaction_cols.append(iname)
    X_inter = df10[interaction_cols].values

    print(f"\n  Baseline features:    {len(FEATURE_COLS)}")
    print(f"  Interaction features: {len(interaction_cols)}")
    print(f"  N={N10}, T0={T0_10}")

    # Expanding-window for baseline
    print("\n  Running baseline expanding-window ...")
    preds_base = expanding_window_logistic(X_base, y10, T0_10)
    y_test_10 = y10[T0_10:]

    auroc_b = roc_auc_score(y_test_10, preds_base)
    auprc_b = average_precision_score(y_test_10, preds_base)
    brier_b = brier_score_loss(y_test_10, preds_base)
    print(f"  Baseline:    AUROC={auroc_b:.3f}  AUPRC={auprc_b:.3f}  Brier={brier_b:.3f}")

    # Expanding-window for interaction
    print("  Running interaction expanding-window ...")
    preds_inter = expanding_window_logistic(X_inter, y10, T0_10)

    auroc_i = roc_auc_score(y_test_10, preds_inter)
    auprc_i = average_precision_score(y_test_10, preds_inter)
    brier_i = brier_score_loss(y_test_10, preds_inter)
    print(f"  Interaction: AUROC={auroc_i:.3f}  AUPRC={auprc_i:.3f}  Brier={brier_i:.3f}")

    model_comp = pd.DataFrame([
        {"Specification": "Baseline (main effects)", "Features": len(FEATURE_COLS),
         "AUROC": round(auroc_b, 3), "AUPRC": round(auprc_b, 3), "Brier": round(brier_b, 3)},
        {"Specification": "Interaction", "Features": len(interaction_cols),
         "AUROC": round(auroc_i, 3), "AUPRC": round(auprc_i, 3), "Brier": round(brier_i, 3)},
    ])
    model_comp.to_csv(OUT_DIR / "model_comparison_expanding.csv", index=False)
    print("\n" + model_comp.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
