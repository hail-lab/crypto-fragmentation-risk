# src/interaction_logit_bootstrap.py
"""
Logistic regression with interaction terms for micro-depeg prediction.

Adds interaction terms between return shocks and volatility regimes
to test for nonlinear amplification:
  - ret_lag1 × ret_vol_30   (recent return shock × medium-horizon vol regime)
  - ret_lag1 × ret_vol_7    (recent return shock × short-horizon vol regime)
  - volume_lag3 × ret_vol_30 (liquidity withdrawal × vol regime)

Compares baseline (main effects only) vs interaction specification
using AUROC, AUPRC on the walk-forward test set, plus bootstrap inference
on the interaction coefficients.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from src.config import DATA_CLEAN, OUTPUTS

THRESH = 0.001  # 10 bps
TRAIN_FRAC = 0.70

# Bootstrap settings
B = 500
BLOCK_LEN = 14
RANDOM_SEED = 42

# Baseline features (same as main specification)
BASE_FEATURES = [
    "ret_lag1", "ret_lag2", "ret_lag3",
    "ret_vol_7", "ret_vol_14", "ret_vol_30",
    "volume_lag1", "volume_lag3",
    "volume_vol_7", "volume_vol_14",
]

# Interaction terms to add
INTERACTIONS = [
    ("ret_lag1", "ret_vol_30"),    # return shock × medium-horizon vol
    ("ret_lag1", "ret_vol_7"),     # return shock × short-horizon vol
    ("volume_lag3", "ret_vol_30"), # liquidity × vol regime
]


def build_dataset(usdt: pd.DataFrame) -> pd.DataFrame:
    df = usdt.sort_values("date").copy()

    df["abs_dev"] = (df["close_px"] - 1.0).abs()
    df["is_offpeg"] = (df["abs_dev"] > THRESH).astype(int)
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

    # Interaction terms (constructed from lagged features — no leakage)
    for a, b in INTERACTIONS:
        df[f"{a}_x_{b}"] = df[a] * df[b]

    df = df.dropna().reset_index(drop=True)
    return df


def moving_block_bootstrap_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    if block_len <= 1:
        return rng.integers(0, n, size=n)
    starts = rng.integers(0, n - block_len + 1, size=int(np.ceil(n / block_len)))
    idx = []
    for s in starts:
        idx.extend(range(s, s + block_len))
    return np.array(idx[:n], dtype=int)


def fit_and_predict(X_train, y_train, X_test):
    scaler = StandardScaler()
    Xs_train = scaler.fit_transform(X_train)
    Xs_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=4000)
    model.fit(Xs_train, y_train)
    probs = model.predict_proba(Xs_test)[:, 1]
    return model.coef_[0].copy(), probs


def safe_auc(y_true, probs):
    if len(set(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, probs)


def safe_auprc(y_true, probs):
    if len(set(y_true)) < 2:
        return float("nan")
    return average_precision_score(y_true, probs)


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    usdt = pd.read_csv(DATA_CLEAN / "usdt_kraken.csv").rename(columns={"date_utc": "date"})
    usdt["date"] = pd.to_datetime(usdt["date"])
    usdt = usdt[["date", "close_px", "volume_base"]].copy()

    df = build_dataset(usdt)

    split = int(len(df) * TRAIN_FRAC)
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()

    y_train = train["depeg"].astype(int)
    y_test = test["depeg"].astype(int)

    interaction_cols = [f"{a}_x_{b}" for a, b in INTERACTIONS]
    interact_features = BASE_FEATURES + interaction_cols

    # ── 1. Baseline model (main effects only) ──
    X_train_base = train[BASE_FEATURES]
    X_test_base = test[BASE_FEATURES]
    _, probs_base = fit_and_predict(X_train_base, y_train, X_test_base)

    # ── 2. Interaction model ──
    X_train_int = train[interact_features]
    X_test_int = test[interact_features]
    point_coefs, probs_int = fit_and_predict(X_train_int, y_train, X_test_int)

    # ── 3. Model comparison ──
    comparison = pd.DataFrame([
        {
            "model": "Baseline (main effects)",
            "n_features": len(BASE_FEATURES),
            "AUROC": safe_auc(y_test, probs_base),
            "AUPRC": safe_auprc(y_test, probs_base),
            "Brier": brier_score_loss(y_test, probs_base),
        },
        {
            "model": "Interaction specification",
            "n_features": len(interact_features),
            "AUROC": safe_auc(y_test, probs_int),
            "AUPRC": safe_auprc(y_test, probs_int),
            "Brier": brier_score_loss(y_test, probs_int),
        },
    ])

    comp_path = OUTPUTS / "interaction_model_comparison.csv"
    comparison.to_csv(comp_path, index=False)
    print("\n=== Model Comparison ===")
    print(comparison.to_string(index=False))

    # ── 4. Bootstrap inference on interaction model ──
    print(f"\nRunning {B} bootstrap replications (block_len={BLOCK_LEN})...")
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(train)
    n_feat = len(interact_features)

    boot = np.zeros((B, n_feat), dtype=float)
    failed = 0

    for b in range(B):
        idx = moving_block_bootstrap_indices(n, BLOCK_LEN, rng)
        Xb = X_train_int.iloc[idx]
        yb = y_train.iloc[idx]

        if yb.nunique() < 2:
            failed += 1
            continue

        try:
            scaler = StandardScaler()
            Xs = scaler.fit_transform(Xb)
            model = LogisticRegression(max_iter=4000)
            model.fit(Xs, yb)
            boot[b] = model.coef_[0].copy()
        except Exception:
            failed += 1

    valid = ~(boot == 0).all(axis=1)
    boot = boot[valid]
    B_eff = boot.shape[0]

    ci_lo = np.percentile(boot, 2.5, axis=0)
    ci_hi = np.percentile(boot, 97.5, axis=0)

    pvals = []
    for j in range(n_feat):
        p_le = np.mean(boot[:, j] <= 0)
        p_ge = np.mean(boot[:, j] >= 0)
        pvals.append(2 * min(p_le, p_ge))

    coef_table = pd.DataFrame({
        "feature": interact_features,
        "coef_point": point_coefs,
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "p_boot": pvals,
    }).sort_values("coef_point", key=lambda s: s.abs(), ascending=False)

    coef_path = OUTPUTS / "interaction_coef_bootstrap_10bps.csv"
    coef_table.to_csv(coef_path, index=False)

    print(f"\nBootstrap done. Effective draws: {B_eff}/{B} (failed={failed})")
    print(f"\nSaved: {comp_path}")
    print(f"Saved: {coef_path}")

    print("\n=== Interaction Model Coefficients ===")
    print(coef_table.to_string(index=False))

    # Highlight just the interaction terms
    print("\n=== Interaction Terms Only ===")
    int_only = coef_table[coef_table["feature"].str.contains("_x_")]
    print(int_only.to_string(index=False))


if __name__ == "__main__":
    main()
