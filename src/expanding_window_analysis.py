"""
Expanding-window walk-forward evaluation for all 5 classifiers.

Implements the TRUE expanding-window protocol described in Algorithm 1 of the
paper: for each forecast origin t (starting from T0 = 70% of N), the model is
re-estimated on observations {1,...,t} and a one-step-ahead prediction is
recorded for t+1.  All classifiers see the same expanding sequence of
train/test pairs, ensuring a strict apples-to-apples comparison.

Produces:
  1. enhanced_classifier_comparison.csv   (Table VII numbers)
  2. roc_curves_comparison.png            (ROC overlay)
  3. pr_curves_comparison.png             (PR overlay)
  4. shap_gbm_beeswarm.png               (SHAP for GBM)
  5. delong_tests.csv                     (pairwise DeLong tests)
  6. confusion_matrices.csv               (Youden-J optimal threshold)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve, confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

from pathlib import Path
import os
import warnings
warnings.filterwarnings("ignore")

# ---------- paths ----------
REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data" / "clean"
OUT_DIR  = REPO / "outputs"
# Optional: copy figures to a submission directory (set SUBMISSION_DIR env var)
_sub = os.environ.get("SUBMISSION_DIR", "")
SUBMISSION_DIR = Path(_sub) if _sub else None

SEED = 42
THRESH = 0.001  # 10 bps

FEATURE_COLS = [
    "ret_lag1", "ret_lag2", "ret_lag3",
    "ret_vol_7", "ret_vol_14", "ret_vol_30",
    "volume_lag1", "volume_lag3",
    "volume_vol_7", "volume_vol_14",
]

FEATURE_LABELS = {
    "ret_lag1": r"$r_{t-1}$",
    "ret_lag2": r"$r_{t-2}$",
    "ret_lag3": r"$r_{t-3}$",
    "ret_vol_7": r"$\sigma^{7}$",
    "ret_vol_14": r"$\sigma^{14}$",
    "ret_vol_30": r"$\sigma^{30}$",
    "volume_lag1": r"$v_{t-1}$",
    "volume_lag3": r"$v_{t-3}$",
    "volume_vol_7": r"$\sigma^{v}_{7}$",
    "volume_vol_14": r"$\sigma^{v}_{14}$",
}


# ============================================================
# DeLong test (two-sided) for comparing two AUROCs
# ============================================================
def _auc_covariance(y_true, p1, p2):
    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]
    n1, n0 = len(pos), len(neg)

    V10_1 = np.array([np.mean(p1[pos[i]] > p1[neg]) + 0.5 * np.mean(p1[pos[i]] == p1[neg]) for i in range(n1)])
    V10_2 = np.array([np.mean(p2[pos[i]] > p2[neg]) + 0.5 * np.mean(p2[pos[i]] == p2[neg]) for i in range(n1)])
    V01_1 = np.array([np.mean(p1[pos] > p1[neg[j]]) + 0.5 * np.mean(p1[pos] == p1[neg[j]]) for j in range(n0)])
    V01_2 = np.array([np.mean(p2[pos] > p2[neg[j]]) + 0.5 * np.mean(p2[pos] == p2[neg[j]]) for j in range(n0)])

    S10 = np.cov(np.column_stack([V10_1, V10_2]), rowvar=False)
    S01 = np.cov(np.column_stack([V01_1, V01_2]), rowvar=False)

    S = S10 / n1 + S01 / n0
    return S


def delong_test(y_true, p1, p2):
    from scipy.stats import norm
    auc1 = roc_auc_score(y_true, p1)
    auc2 = roc_auc_score(y_true, p2)
    S = _auc_covariance(y_true, p1, p2)
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var <= 0:
        return 0.0, 1.0
    z = (auc1 - auc2) / np.sqrt(var)
    p = 2 * norm.sf(abs(z))
    return float(z), float(p)


# ============================================================
# Classifier factory — returns a fresh (un-fitted) instance
# ============================================================
def _make_classifiers():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=5, random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.8, random_state=SEED
        ),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=SEED),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
            max_iter=500, early_stopping=True, validation_fraction=0.15,
            random_state=SEED,
        ),
    }

NEEDS_SCALING = {"Logistic Regression", "SVM (RBF)", "MLP"}


# ============================================================
# Expanding-window walk-forward loop
# ============================================================
def expanding_window_predictions(X: np.ndarray, y: np.ndarray,
                                  feature_names: list[str],
                                  T0: int) -> dict[str, np.ndarray]:
    """
    For each forecast origin t = T0 ... N-1:
      - Train on {0,...,t-1}, predict for t.
    Returns dict  classifier_name -> array of predicted probabilities
    (length = N - T0, aligned with y[T0:]).
    """
    N = len(y)
    n_test = N - T0
    clf_names = list(_make_classifiers().keys())
    preds = {name: np.zeros(n_test) for name in clf_names}

    for t in range(T0, N):
        idx = t - T0  # position in output array
        X_train, y_train = X[:t], y[:t]
        X_point = X[t : t + 1]  # single test row

        # Standardise inside the loop (no leakage)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_point_s = scaler.transform(X_point)

        classifiers = _make_classifiers()  # fresh instances each iteration

        for name, clf in classifiers.items():
            use_scaled = name in NEEDS_SCALING
            Xtr = X_train_s if use_scaled else X_train
            Xpt = X_point_s if use_scaled else X_point
            clf.fit(Xtr, y_train)
            preds[name][idx] = clf.predict_proba(Xpt)[:, 1][0]

        if (t - T0) % 50 == 0:
            print(f"  expanding-window step {t - T0 + 1}/{n_test}")

    return preds


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- load data ----
    df = pd.read_csv(DATA_DIR / "usdt_depeg_dataset.csv")
    df = df.sort_values("date").reset_index(drop=True)
    df["is_offpeg"] = (df["abs_dev"] > THRESH).astype(int)
    df["depeg"] = df["is_offpeg"].shift(-1)
    df = df.dropna(subset=["depeg"]).reset_index(drop=True)
    df["depeg"] = df["depeg"].astype(int)

    X = df[FEATURE_COLS].values
    y = df["depeg"].values
    N = len(y)
    T0 = int(N * 0.70)  # initial window — first 70% for training

    print(f"Dataset: N={N}, T0={T0}, test steps={N - T0}")
    print(f"Positive rate: {y.mean():.3f}")
    print(f"Running expanding-window evaluation for 5 classifiers ...\n")

    # ---- expanding-window walk-forward ----
    preds = expanding_window_predictions(X, y, FEATURE_COLS, T0)

    y_test = y[T0:]  # true labels for the test portion
    assert len(y_test) == N - T0

    # ---- collect metrics ----
    results = []
    for name, probs in preds.items():
        auroc = roc_auc_score(y_test, probs)
        auprc = average_precision_score(y_test, probs)
        brier = brier_score_loss(y_test, probs)
        results.append({
            "Classifier": name, "Features": len(FEATURE_COLS),
            "AUROC": round(auroc, 3), "AUPRC": round(auprc, 3),
            "Brier": round(brier, 3),
        })
        print(f"{name:25s}  AUROC={auroc:.3f}  AUPRC={auprc:.3f}  Brier={brier:.3f}")

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT_DIR / "enhanced_classifier_comparison.csv", index=False)
    print("\n", res_df.to_string(index=False))

    # ---- 1. ROC Curve overlay ----
    fig, ax = plt.subplots(figsize=(5.5, 5))
    styles = {
        "Logistic Regression": ("-", "tab:blue"),
        "Random Forest": ("--", "tab:orange"),
        "Gradient Boosting": ("-.", "tab:green"),
        "SVM (RBF)": (":", "tab:red"),
        "MLP": ((0, (3, 1, 1, 1)), "tab:purple"),
    }
    for name, probs in preds.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc_val = roc_auc_score(y_test, probs)
        ls, c = styles[name]
        ax.plot(fpr, tpr, linestyle=ls, color=c, linewidth=1.5,
                label=f"{name} ({auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Expanding-Window Evaluation (10 bps)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    fig.tight_layout()
    fig.savefig(OUT_DIR / "roc_curves_comparison.png", dpi=300, bbox_inches="tight")
    if SUBMISSION_DIR and SUBMISSION_DIR.exists():
        fig.savefig(SUBMISSION_DIR / "roc_curves_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved ROC curve figure.")

    # ---- 2. PR Curve overlay ----
    fig, ax = plt.subplots(figsize=(5.5, 5))
    baseline_rate = y_test.mean()
    for name, probs in preds.items():
        prec, rec, _ = precision_recall_curve(y_test, probs)
        ap = average_precision_score(y_test, probs)
        ls, c = styles[name]
        ax.plot(rec, prec, linestyle=ls, color=c, linewidth=1.5,
                label=f"{name} ({ap:.3f})")
    ax.axhline(baseline_rate, color="gray", linestyle="--", alpha=0.5,
               label=f"Baseline ({baseline_rate:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curves — Expanding-Window (10 bps)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([0, 1.01])
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pr_curves_comparison.png", dpi=300, bbox_inches="tight")
    if SUBMISSION_DIR and SUBMISSION_DIR.exists():
        fig.savefig(SUBMISSION_DIR / "pr_curves_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved PR curve figure.")

    # ---- 3. SHAP for Gradient Boosting ----
    # Retrain GBM on full training window for SHAP interpretability
    try:
        import shap
        scaler_full = StandardScaler()
        X_full_train = scaler_full.fit_transform(X[:T0])
        X_full_test = scaler_full.transform(X[T0:])

        gbm_full = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=SEED
        )
        gbm_full.fit(X[:T0], y[:T0])  # raw features for tree model

        explainer = shap.TreeExplainer(gbm_full)
        shap_values = explainer.shap_values(
            pd.DataFrame(X[T0:], columns=FEATURE_COLS)
        )

        fig, ax = plt.subplots(figsize=(6, 5))
        shap.summary_plot(
            shap_values,
            pd.DataFrame(X[T0:], columns=FEATURE_COLS),
            feature_names=[FEATURE_LABELS.get(c, c) for c in FEATURE_COLS],
            show=False, plot_size=None,
        )
        plt.title("SHAP Feature Importance — Gradient Boosting", fontsize=11)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "shap_gbm_beeswarm.png", dpi=300, bbox_inches="tight")
        if SUBMISSION_DIR and SUBMISSION_DIR.exists():
            plt.savefig(SUBMISSION_DIR / "shap_gbm_beeswarm.png", dpi=300, bbox_inches="tight")
        plt.close("all")
        print("Saved SHAP beeswarm figure.")
    except Exception as e:
        print(f"SHAP failed: {e}")

    # ---- 4. DeLong pairwise tests ----
    ref = "Logistic Regression"
    delong_rows = []
    for name in preds:
        if name == ref:
            continue
        z, p = delong_test(y_test, preds[ref], preds[name])
        delong_rows.append({
            "Comparison": f"{ref} vs {name}",
            "z": round(z, 3),
            "p-value": round(p, 4),
        })
        print(f"DeLong {ref} vs {name}: z={z:.3f}, p={p:.4f}")
    dl_df = pd.DataFrame(delong_rows)
    dl_df.to_csv(OUT_DIR / "delong_tests.csv", index=False)

    # ---- 5. Confusion matrices (Youden J optimal threshold) ----
    print("\n--- Confusion Matrices (Youden J threshold) ---")
    cm_rows = []
    for name, probs in preds.items():
        fpr, tpr, thresholds = roc_curve(y_test, probs)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_thresh = thresholds[best_idx]
        pred_labels = (probs >= best_thresh).astype(int)
        cm = confusion_matrix(y_test, pred_labels)
        tn, fp, fn, tp = cm.ravel()
        cm_rows.append({
            "Classifier": name, "Threshold": round(best_thresh, 4),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Sensitivity": round(tp / (tp + fn) if (tp + fn) > 0 else 0, 3),
            "Specificity": round(tn / (tn + fp) if (tn + fp) > 0 else 0, 3),
        })
        print(f"{name:25s}  thresh={best_thresh:.4f}  TP={tp} FP={fp} FN={fn} TN={tn}")
    cm_df = pd.DataFrame(cm_rows)
    cm_df.to_csv(OUT_DIR / "confusion_matrices.csv", index=False)

    print(f"\nDone. All outputs saved to: {OUT_DIR}")
    print(f"Figures also copied to: {SUBMISSION_DIR}")


if __name__ == "__main__":
    main()
