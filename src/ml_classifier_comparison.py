# src/ml_classifier_comparison.py
"""
Systematic comparison of classifiers for next-day USDT micro-depeg prediction.

Walk-forward evaluation (70/30 temporal split) with:
  - Logistic Regression (interpretable baseline)
  - Random Forest
  - Gradient Boosting (GBM)
  - Support Vector Machine (SVM, RBF kernel)

Outputs:
  - classifier_comparison.csv   (metrics table)
  - feature_importance_rf.csv   (RF permutation importance)
  - feature_importance_gbm.csv  (GBM feature importance)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

from src.config import OUTPUTS

from pathlib import Path

TRAIN_FRAC = 0.70
RANDOM_SEED = 42
THRESH = 0.001  # 10 bps, matching paper baseline

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "clean"

FEATURE_COLS = [
    "ret_lag1", "ret_lag2", "ret_lag3",
    "ret_vol_7", "ret_vol_14", "ret_vol_30",
    "volume_lag1", "volume_lag3",
    "volume_vol_7", "volume_vol_14",
]


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_DIR / "usdt_depeg_dataset.csv")
    df = df.sort_values("date").reset_index(drop=True)

    # Rebuild labels at the paper's 10-bps threshold
    df["is_offpeg"] = (df["abs_dev"] > THRESH).astype(int)
    df["depeg"] = df["is_offpeg"].shift(-1)
    df = df.dropna(subset=["depeg"]).reset_index(drop=True)
    df["depeg"] = df["depeg"].astype(int)

    split = int(len(df) * TRAIN_FRAC)
    train_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["depeg"].astype(int)
    X_test = test_df[FEATURE_COLS]
    y_test = test_df["depeg"].astype(int)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Define classifiers
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=5, random_state=RANDOM_SEED
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=RANDOM_SEED
        ),
        "SVM (RBF)": SVC(
            kernel="rbf", probability=True, random_state=RANDOM_SEED
        ),
    }

    rows = []
    fitted = {}

    for name, clf in classifiers.items():
        # SVM and Logistic need scaled data; tree methods don't but it doesn't hurt
        use_scaled = name in ("Logistic Regression", "SVM (RBF)")
        Xtr = X_train_s if use_scaled else X_train
        Xte = X_test_s if use_scaled else X_test

        clf.fit(Xtr, y_train)
        probs = clf.predict_proba(Xte)[:, 1]

        auroc = roc_auc_score(y_test, probs)
        auprc = average_precision_score(y_test, probs)
        brier = brier_score_loss(y_test, probs)

        rows.append({
            "Classifier": name,
            "Features": len(FEATURE_COLS),
            "AUROC": round(auroc, 3),
            "AUPRC": round(auprc, 3),
            "Brier": round(brier, 3),
        })
        fitted[name] = (clf, Xtr, Xte)

        print(f"{name:25s}  AUROC={auroc:.3f}  AUPRC={auprc:.3f}  Brier={brier:.3f}")

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUTS / "classifier_comparison.csv", index=False)
    print("\nSaved:", OUTPUTS / "classifier_comparison.csv")
    print(results.to_string(index=False))

    # Feature importance for tree methods
    # Random Forest: permutation importance on test set
    rf_clf, _, rf_Xte = fitted["Random Forest"]
    perm = permutation_importance(
        rf_clf, X_test, y_test,
        n_repeats=30, random_state=RANDOM_SEED, scoring="roc_auc"
    )
    rf_imp = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)
    rf_imp.to_csv(OUTPUTS / "feature_importance_rf.csv", index=False)
    print("\nRF Permutation Importance:")
    print(rf_imp.to_string(index=False))

    # Gradient Boosting: built-in feature importance
    gbm_clf = fitted["Gradient Boosting"][0]
    gbm_imp = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": gbm_clf.feature_importances_,
    }).sort_values("importance", ascending=False)
    gbm_imp.to_csv(OUTPUTS / "feature_importance_gbm.csv", index=False)
    print("\nGBM Feature Importance:")
    print(gbm_imp.to_string(index=False))


if __name__ == "__main__":
    main()
