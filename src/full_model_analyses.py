"""
Micro-depeg prediction with the current peg deviation as a predictor ("full model").

Micro-depeg events are persistent: P(Y_{t+1}=1 | Y_t=1) = 0.57 against 0.05 otherwise. The
current absolute deviation |d_t| is therefore added to the ten regime predictors. Everything
below uses the paper's expanding-window protocol (initial window 70%, re-estimated daily) and
the paired moving-block bootstrap (L = 14; B = 2000 for out-of-sample metrics, B = 500 for
coefficients).

Outputs (outputs/full_model/):
  check_baseline.csv     published baseline logistic AUROC / AUPRC reproduced
  models_ci.csv          persistence, regime and full logistic models: metrics with CIs
  models_diff.csv        paired differences (full minus regime, persistence minus full, ...)
  classifiers_ci.csv     five classifiers on the full feature set
  classifiers_diff.csv   alternatives minus logistic (bootstrap + DeLong)
  coefs_full.csv         full logistic coefficients on the 70% window (L = 14, 7, 21)
  thresholds.csv         full logistic at 5 / 10 / 15 / 20 bp
  spec_robustness.csv    class-weighted and probit variants
  oos_predictions.csv
  figures: roc_full.(png|pdf), pr_full.(png|pdf), calibration_full.(png|pdf), shap_full.(png|pdf)
"""
from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             roc_curve, precision_recall_curve)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.expanding_window_analysis import FEATURE_COLS, expanding_window_predictions, SEED  # noqa: E402
from src.round2_analyses import oos_bootstrap, moving_block_bootstrap_indices  # noqa: E402
from src.robustness_analyses import expanding_logit  # noqa: E402

DATA_DIR = REPO / "data" / "clean"
OUT_DIR = REPO / "outputs" / "full_model"
FULL = list(FEATURE_COLS) + ["abs_dev"]
LABELS = {"ret_lag1": r"$r_{t-1}$", "ret_lag2": r"$r_{t-2}$", "ret_lag3": r"$r_{t-3}$",
          "ret_vol_7": r"$\sigma^{7}$", "ret_vol_14": r"$\sigma^{14}$", "ret_vol_30": r"$\sigma^{30}$",
          "volume_lag1": r"$v_{t-1}$", "volume_lag3": r"$v_{t-3}$",
          "volume_vol_7": r"$\sigma^{v}_{7}$", "volume_vol_14": r"$\sigma^{v}_{14}$",
          "abs_dev": r"$|d_t|$"}
STYLES = {"Logistic Regression": ("-", "tab:blue"), "Random Forest": ("--", "tab:orange"),
          "Gradient Boosting": ("-.", "tab:green"), "SVM (RBF)": (":", "tab:red"),
          "MLP": ((0, (3, 1, 1, 1)), "tab:purple")}
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.labelsize": 11,
                     "legend.fontsize": 8, "axes.grid": True, "grid.alpha": 0.3})


def load(thresh: float) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "usdt_depeg_dataset.csv").sort_values("date").reset_index(drop=True)
    df["depeg"] = (df["abs_dev"] > thresh).astype(int).shift(-1)
    return df.dropna(subset=["depeg"]).reset_index(drop=True)


def coef_table(train: pd.DataFrame, block_lens=(14, 7, 21), B=500) -> pd.DataFrame:
    X, y = train[FULL], train["depeg"].astype(int)

    def fit(Xb, yb):
        sc = StandardScaler().fit(Xb)
        return LogisticRegression(max_iter=4000).fit(sc.transform(Xb), yb).coef_[0]
    out = pd.DataFrame({"predictor": FULL, "coef": fit(X, y)})
    for L in block_lens:
        rng = np.random.default_rng(SEED)
        boot = []
        for _ in range(B):
            idx = moving_block_bootstrap_indices(len(X), L, rng)
            if y.iloc[idx].nunique() < 2:
                continue
            boot.append(fit(X.iloc[idx], y.iloc[idx]))
        boot = np.array(boot)
        out[f"ci_lo_L{L}"] = np.percentile(boot, 2.5, axis=0)
        out[f"ci_hi_L{L}"] = np.percentile(boot, 97.5, axis=0)
        out[f"p_L{L}"] = [2 * min((boot[:, k] <= 0).mean(), (boot[:, k] >= 0).mean())
                          for k in range(boot.shape[1])]
    return out


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def figures(y_test, preds, df, T0):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, p in preds.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        ls, c = STYLES[name]
        ax.plot(fpr, tpr, linestyle=ls, color=c, lw=1.5, label=f"{name} ({roc_auc_score(y_test, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, lw=0.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(-0.01, 1.01), ylim=(-0.01, 1.01))
    ax.legend(loc="lower right")
    save(fig, "roc_full")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, p in preds.items():
        prec, rec, _ = precision_recall_curve(y_test, p)
        ls, c = STYLES[name]
        ax.plot(rec, prec, linestyle=ls, color=c, lw=1.5,
                label=f"{name} ({average_precision_score(y_test, p):.3f})")
    ax.axhline(y_test.mean(), color="gray", ls="--", alpha=0.6, label=f"Event rate ({y_test.mean():.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(-0.01, 1.01), ylim=(0, 1.01))
    ax.legend(loc="upper right")
    save(fig, "pr_full")

    p = preds["Logistic Regression"]
    bins = np.linspace(0, 1, 6)
    cal = (pd.DataFrame({"p": p, "y": y_test, "bin": pd.cut(p, bins, include_lowest=True)})
           .groupby("bin", observed=False).agg(mp=("p", "mean"), ar=("y", "mean"), n=("y", "size"))
           .dropna())
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    ax.plot([0, 1], [0, 1], "--", color="#9E9E9E", lw=1.0, label="Perfect calibration")
    ax.plot(cal.mp, cal.ar, "o-", color="#1565C0", lw=1.8, ms=7, mec="white", mew=0.8,
            label="Logistic model")
    for _, r in cal.iterrows():
        ax.annotate(f"n={int(r.n)}", (r.mp, r.ar), textcoords="offset points", xytext=(6, 6),
                    fontsize=7, color="#616161")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed frequency", xlim=(-0.02, 1.02),
           ylim=(-0.02, 1.02), aspect="equal")
    ax.legend(loc="upper left")
    save(fig, "calibration_full")
    cal.to_csv(OUT_DIR / "calibration_bins.csv")

    import shap
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                                    subsample=0.8, random_state=SEED)
    gb.fit(df.iloc[:T0][FULL], df.iloc[:T0]["depeg"].astype(int))
    sv = shap.TreeExplainer(gb).shap_values(df.iloc[T0:][FULL])
    plt.figure(figsize=(6, 5))
    shap.summary_plot(sv, df.iloc[T0:][FULL], feature_names=[LABELS[c] for c in FULL],
                      show=False, plot_size=None)
    save(plt.gcf(), "shap_full")
    imp = pd.DataFrame({"predictor": FULL, "mean_abs_shap": np.abs(sv).mean(axis=0)})
    imp.sort_values("mean_abs_shap", ascending=False).to_csv(OUT_DIR / "shap_importance.csv", index=False)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load(0.001)
    y = df["depeg"].astype(int).values
    T0 = int(len(y) * 0.70)
    y_test = y[T0:]

    # 0. reproduce the published baseline
    p_reg = expanding_logit(df[FEATURE_COLS].values, y, T0)
    chk = pd.DataFrame([{"AUROC": roc_auc_score(y_test, p_reg), "AUPRC": average_precision_score(y_test, p_reg),
                         "Brier": brier_score_loss(y_test, p_reg), "published_AUROC": 0.832,
                         "published_AUPRC": 0.422, "published_Brier": 0.117}])
    chk.to_csv(OUT_DIR / "check_baseline.csv", index=False)
    print(chk.round(3).to_string(index=False))

    # 1. persistence vs regime vs full logistic
    models = {"Full (regime + |d_t|)": expanding_logit(df[FULL].values, y, T0),
              "Regime (10 predictors)": p_reg,
              "Persistence |d_t| only": expanding_logit(df[["abs_dev"]].values, y, T0)}
    ci, diff, _ = oos_bootstrap(y_test, models, ref="Full (regime + |d_t|)")
    ci.to_csv(OUT_DIR / "models_ci.csv", index=False)
    diff.to_csv(OUT_DIR / "models_diff.csv", index=False)
    print(ci.round(3).to_string(index=False)); print(diff.round(3).to_string(index=False))

    # 2. five classifiers on the full feature set
    preds = expanding_window_predictions(df[FULL].values, y, FULL, T0)
    ci5, diff5, _ = oos_bootstrap(y_test, preds, ref="Logistic Regression")
    ci5.to_csv(OUT_DIR / "classifiers_ci.csv", index=False)
    diff5.to_csv(OUT_DIR / "classifiers_diff.csv", index=False)
    pd.DataFrame({"date": df["date"].iloc[T0:].values, "y": y_test, **preds,
                  "Regime logistic": p_reg}).to_csv(OUT_DIR / "oos_predictions.csv", index=False)
    print(ci5.round(3).to_string(index=False)); print(diff5.round(3).to_string(index=False))

    # 3. coefficients on the initial 70% window
    coefs = coef_table(df.iloc[:T0])
    coefs.to_csv(OUT_DIR / "coefs_full.csv", index=False)
    print(coefs.round(3).to_string(index=False))

    # 4. thresholds
    rows = []
    for bps in (5, 10, 15, 20):
        d = load(bps / 10000)
        yy = d["depeg"].astype(int).values
        t0 = int(len(yy) * 0.70)
        pf = expanding_logit(d[FULL].values, yy, t0)
        pr = expanding_logit(d[FEATURE_COLS].values, yy, t0)
        row = {"bps": bps, "event_rate": yy.mean(), "test_events": int(yy[t0:].sum())}
        if yy[t0:].min() != yy[t0:].max():
            row.update(AUROC_full=roc_auc_score(yy[t0:], pf), AUPRC_full=average_precision_score(yy[t0:], pf),
                       Brier_full=brier_score_loss(yy[t0:], pf), AUROC_regime=roc_auc_score(yy[t0:], pr))
            if yy[t0:].sum() >= 2:
                c, _, _ = oos_bootstrap(yy[t0:], {"full": pf, "regime": pr}, ref="regime")
                for m in ("AUROC", "AUPRC"):
                    r = c[(c.classifier == "full") & (c.metric == m)].iloc[0]
                    row[f"{m}_full_lo"], row[f"{m}_full_hi"] = r.ci95_lo, r.ci95_hi
        rows.append(row)
    th = pd.DataFrame(rows)
    th.to_csv(OUT_DIR / "thresholds.csv", index=False)
    print(th.round(3).to_string(index=False))

    # 5. specification robustness
    pw = np.zeros(len(y) - T0); pp = np.zeros(len(y) - T0)
    X = df[FULL].values
    for t in range(T0, len(y)):
        sc = StandardScaler().fit(X[:t])
        Xt, Xp = sc.transform(X[:t]), sc.transform(X[t:t + 1])
        pw[t - T0] = LogisticRegression(max_iter=2000, class_weight="balanced",
                                        random_state=SEED).fit(Xt, y[:t]).predict_proba(Xp)[0, 1]
        pp[t - T0] = sm.Probit(y[:t], sm.add_constant(Xt)).fit(disp=0).predict(
            sm.add_constant(Xp, has_constant="add"))[0]
    spec = pd.DataFrame([{"variant": n, "AUROC": roc_auc_score(y_test, p),
                          "AUPRC": average_precision_score(y_test, p)}
                         for n, p in (("class-weighted logit", pw), ("probit", pp))])
    spec.to_csv(OUT_DIR / "spec_robustness.csv", index=False)
    print(spec.round(3).to_string(index=False))

    figures(y_test, preds, df, T0)
    print("figures saved")


if __name__ == "__main__":
    main()
