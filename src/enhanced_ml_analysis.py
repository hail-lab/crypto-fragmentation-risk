"""
Enhanced ML analysis for IEEE Access paper.

Produces:
  1. 5-classifier comparison (adds MLP to existing 4)
  2. ROC curve overlay figure (all 5 classifiers)
  3. Precision-Recall curve overlay figure (all 5 classifiers)
  4. SHAP beeswarm plot for Gradient Boosting
  5. DeLong test: pairwise AUROC significance
  6. Confusion matrices at optimal threshold (Youden J)
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

TRAIN_FRAC = 0.70
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
    """Compute DeLong covariance matrix for two sets of predictions."""
    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]
    n1, n0 = len(pos), len(neg)

    # Structural components for each predictor
    V10_1 = np.array([np.mean(p1[pos[i]] > p1[neg]) + 0.5 * np.mean(p1[pos[i]] == p1[neg]) for i in range(n1)])
    V10_2 = np.array([np.mean(p2[pos[i]] > p2[neg]) + 0.5 * np.mean(p2[pos[i]] == p2[neg]) for i in range(n1)])
    V01_1 = np.array([np.mean(p1[pos] > p1[neg[j]]) + 0.5 * np.mean(p1[pos] == p1[neg[j]]) for j in range(n0)])
    V01_2 = np.array([np.mean(p2[pos] > p2[neg[j]]) + 0.5 * np.mean(p2[pos] == p2[neg[j]]) for j in range(n0)])

    S10 = np.cov(np.column_stack([V10_1, V10_2]), rowvar=False)
    S01 = np.cov(np.column_stack([V01_1, V01_2]), rowvar=False)

    S = S10 / n1 + S01 / n0
    return S


def delong_test(y_true, p1, p2):
    """Two-sided DeLong test. Returns z-stat and p-value."""
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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- load data ----
    df = pd.read_csv(DATA_DIR / "usdt_depeg_dataset.csv")
    df = df.sort_values("date").reset_index(drop=True)
    df["is_offpeg"] = (df["abs_dev"] > THRESH).astype(int)
    df["depeg"] = df["is_offpeg"].shift(-1)
    df = df.dropna(subset=["depeg"]).reset_index(drop=True)
    df["depeg"] = df["depeg"].astype(int)

    split = int(len(df) * TRAIN_FRAC)
    X_train = df.iloc[:split][FEATURE_COLS]
    y_train = df.iloc[:split]["depeg"]
    X_test  = df.iloc[split:][FEATURE_COLS]
    y_test  = df.iloc[split:]["depeg"]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ---- classifiers ----
    classifiers = {
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

    results = []
    probs_dict = {}

    for name, clf in classifiers.items():
        use_scaled = name in ("Logistic Regression", "SVM (RBF)", "MLP")
        Xtr = X_train_s if use_scaled else X_train
        Xte = X_test_s  if use_scaled else X_test

        clf.fit(Xtr, y_train)
        probs = clf.predict_proba(Xte)[:, 1]
        probs_dict[name] = probs

        auroc = roc_auc_score(y_test, probs)
        auprc = average_precision_score(y_test, probs)
        brier = brier_score_loss(y_test, probs)

        results.append({
            "Classifier": name, "Features": len(FEATURE_COLS),
            "AUROC": round(auroc, 3), "AUPRC": round(auprc, 3), "Brier": round(brier, 3),
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
        "MLP": ((0,(3,1,1,1)), "tab:purple"),
    }
    for name, probs in probs_dict.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc_val = roc_auc_score(y_test, probs)
        ls, c = styles[name]
        ax.plot(fpr, tpr, linestyle=ls, color=c, linewidth=1.5,
                label=f"{name} ({auc_val:.3f})")
    ax.plot([0,1], [0,1], "k--", alpha=0.3, linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Micro-Depeg Classifiers (10 bps)")
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
    for name, probs in probs_dict.items():
        prec, rec, _ = precision_recall_curve(y_test, probs)
        ap = average_precision_score(y_test, probs)
        ls, c = styles[name]
        ax.plot(rec, prec, linestyle=ls, color=c, linewidth=1.5,
                label=f"{name} ({ap:.3f})")
    ax.axhline(baseline_rate, color="gray", linestyle="--", alpha=0.5, label=f"Baseline ({baseline_rate:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curves — Micro-Depeg Classifiers (10 bps)")
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
    try:
        import shap
        gbm_clf = classifiers["Gradient Boosting"]
        explainer = shap.TreeExplainer(gbm_clf)
        shap_values = explainer.shap_values(X_test)

        fig, ax = plt.subplots(figsize=(6, 5))
        shap.summary_plot(
            shap_values, X_test,
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
    names = list(probs_dict.keys())
    ref = "Logistic Regression"
    delong_rows = []
    for name in names:
        if name == ref:
            continue
        z, p = delong_test(y_test.values, probs_dict[ref], probs_dict[name])
        delong_rows.append({"Comparison": f"{ref} vs {name}", "z": round(z, 3), "p-value": round(p, 4)})
        print(f"DeLong {ref} vs {name}: z={z:.3f}, p={p:.4f}")
    dl_df = pd.DataFrame(delong_rows)
    dl_df.to_csv(OUT_DIR / "delong_tests.csv", index=False)

    # ---- 5. Confusion matrices (Youden J optimal threshold) ----
    print("\n--- Confusion Matrices (Youden J threshold) ---")
    cm_rows = []
    for name, probs in probs_dict.items():
        fpr, tpr, thresholds = roc_curve(y_test, probs)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_thresh = thresholds[best_idx]
        preds = (probs >= best_thresh).astype(int)
        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()
        cm_rows.append({
            "Classifier": name, "Threshold": round(best_thresh, 4),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Sensitivity": round(tp/(tp+fn) if (tp+fn) > 0 else 0, 3),
            "Specificity": round(tn/(tn+fp) if (tn+fp) > 0 else 0, 3),
        })
        print(f"{name:25s}  thresh={best_thresh:.4f}  TP={tp} FP={fp} FN={fn} TN={tn}")
    cm_df = pd.DataFrame(cm_rows)
    cm_df.to_csv(OUT_DIR / "confusion_matrices.csv", index=False)

    print("\nDone. All outputs saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
