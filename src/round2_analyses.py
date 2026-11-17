"""
Second-round analyses (2026).

1. Variance decomposition with the covariance term (Eq. 6): for every
   ordered venue pair (i, j),
       Var(r_i) = Var(r_j) + Var(Delta) + 2 Cov(r_j, Delta),
   reported full-sample (Table 2) and by stress condition (Table 4).
2. Conditional return slope in the interaction specification: the
   log-odds effect of a one-standard-deviation r_{t-1} shock as a function
   of sigma^30, with moving-block bootstrap intervals.
3. Out-of-sample uncertainty for the expanding-window classifiers:
   moving-block bootstrap intervals (resampling the sequential forecast /
   outcome pairs in blocks) for AUROC, AUPRC, Brier and for the paired
   gradient boosting minus logistic regression differences.

Produces (outputs/round2/):
  variance_decomposition_table2.csv, variance_decomposition_pairs.csv,
  variance_decomposition_stress.csv, interaction_conditional_slope.csv,
  oos_bootstrap_ci.csv, oos_bootstrap_diff.csv, oos_predictions.csv,
  oos_bootstrap_ci_interaction.csv, oos_bootstrap_diff_interaction.csv
"""
from __future__ import annotations

from itertools import permutations
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data" / "clean"
OUT_DIR = REPO / "outputs" / "round2"
sys.path.insert(0, str(REPO))

from src.expanding_window_analysis import (  # noqa: E402
    FEATURE_COLS, THRESH, expanding_window_predictions, delong_test,
)
SEED = 42
BASE_FEATURES = FEATURE_COLS
INTERACTIONS = [("ret_lag1", "ret_vol_30"), ("ret_lag1", "ret_vol_7"), ("volume_lag3", "ret_vol_30")]


# Same construction as src/interaction_logit_bootstrap.py (copied rather than
# imported, because that module's src.config import creates directories).
def build_dataset(usdt: pd.DataFrame) -> pd.DataFrame:
    df = usdt.sort_values("date").copy()
    df["abs_dev"] = (df["close_px"] - 1.0).abs()
    df["depeg"] = (df["abs_dev"] > THRESH).astype(int).shift(-1)
    df["ret"] = np.log(df["close_px"]).diff()
    for k in (1, 2, 3):
        df[f"ret_lag{k}"] = df["ret"].shift(k)
    for w in (7, 14, 30):
        df[f"ret_vol_{w}"] = df["ret"].rolling(w).std()
    df["volume_lag1"] = df["volume_base"].shift(1)
    df["volume_lag3"] = df["volume_base"].shift(3)
    for w in (7, 14):
        df[f"volume_vol_{w}"] = df["volume_base"].rolling(w).std()
    for a, b in INTERACTIONS:
        df[f"{a}_x_{b}"] = df[a] * df[b]
    return df.dropna().reset_index(drop=True)


def moving_block_bootstrap_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, n - block_len + 1, size=int(np.ceil(n / block_len)))
    idx = [k for s in starts for k in range(s, s + block_len)]
    return np.array(idx[:n], dtype=int)


VENUES = ["binance", "kraken", "coinbase"]
PAIRS = [("binance", "kraken", "bk"), ("binance", "coinbase", "bc"), ("kraken", "coinbase", "kc")]
BLOCK_LEN = 14
B_COEF = 500
B_OOS = 2000


# ------------------------------------------------------------------
# 1. Variance decomposition
# ------------------------------------------------------------------
def decompose(ri: pd.Series, rj: pd.Series) -> dict:
    d = ri - rj
    v_i, v_j, v_d = ri.var(), rj.var(), d.var()
    cov2 = 2 * np.cov(rj, d, ddof=1)[0, 1]
    return {
        "var_ri": v_i, "var_rj": v_j, "var_delta": v_d, "two_cov": cov2,
        "share_var_rj_pct": 100 * v_j / v_i,
        "share_var_delta_pct": 100 * v_d / v_i,
        "share_two_cov_pct": 100 * cov2 / v_i,
        "net_venue_pct": 100 * (v_d + cov2) / v_i,   # = 100 (Var r_i - Var r_j) / Var r_i
        "identity_resid": v_i - (v_j + v_d + cov2),
        "corr_ij": np.corrcoef(ri, rj)[0, 1],
    }


def variance_decomposition(panel: pd.DataFrame, spreads: pd.DataFrame):
    pair_rows, t2_rows, stress_rows = [], [], []
    for asset, g in panel.groupby("asset"):
        g = g.sort_values("date_utc").copy()
        for v in VENUES:
            g[f"ret_{v}"] = np.log(g[f"close_{v}"]).diff()

        # Table 2: all six ordered pairs, full sample
        full = g.dropna()
        recs = []
        for i, j in permutations(VENUES, 2):
            r = decompose(full[f"ret_{i}"], full[f"ret_{j}"])
            r.update(asset=asset, i=i, j=j, n=len(full))
            recs.append(r)
        pair_rows += recs
        rr = pd.DataFrame(recs)
        t2_rows.append({
            "asset": asset, "n": len(full),
            "var_delta_mean": rr.share_var_delta_pct.mean(),
            "var_delta_max": rr.share_var_delta_pct.max(),
            "two_cov_mean": rr.share_two_cov_pct.mean(),
            "two_cov_min": rr.share_two_cov_pct.min(),
            "two_cov_max": rr.share_two_cov_pct.max(),
            "net_abs_mean": rr.net_venue_pct.abs().mean(),
            "net_abs_max": rr.net_venue_pct.abs().max(),
            "corr_min": rr.corr_ij.min(),
            "max_identity_resid": rr.identity_resid.abs().max(),
        })

        # Table 4: stress conditions (same construction as revision2026/robustness_experiments.ipynb, Section A)
        g["ret_avg"] = g[[f"ret_{v}" for v in VENUES]].mean(axis=1)
        g["rv30"] = g["ret_avg"].rolling(30).std()
        s = spreads.loc[spreads.asset == asset,
                        ["date_utc", "spread_rel_bk", "spread_rel_bc", "spread_rel_kc"]]
        g = g.merge(s, on="date_utc", how="left")
        for _, _, tag in PAIRS:
            col = f"spread_rel_{tag}"
            g[f"z_{tag}"] = (g[col] - g[col].rolling(30).mean()) / g[col].rolling(30).std()
        g = g.dropna().reset_index(drop=True)
        conds = {
            "Full sample": pd.Series(True, index=g.index),
            "Top-quartile volatility": g["rv30"] >= g["rv30"].quantile(0.75),
            "Top-decile volatility": g["rv30"] >= g["rv30"].quantile(0.90),
            "Worst-decile returns": g["ret_avg"] <= g["ret_avg"].quantile(0.10),
            "Anomaly days": (g[[f"z_{t}" for _, _, t in PAIRS]].abs() >= 3).any(axis=1),
        }
        for name, mask in conds.items():
            sub = g[mask]
            recs = [decompose(sub[f"ret_{vi}"], sub[f"ret_{vj}"]) for vi, vj, _ in PAIRS]
            rr = pd.DataFrame(recs)
            stress_rows.append({
                "asset": asset, "condition": name, "n": int(mask.sum()),
                "var_delta_mean": rr.share_var_delta_pct.mean(),
                "var_delta_max": rr.share_var_delta_pct.max(),
                "two_cov_mean": rr.share_two_cov_pct.mean(),
                "two_cov_min": rr.share_two_cov_pct.min(),
                "two_cov_max": rr.share_two_cov_pct.max(),
                "net_abs_mean": rr.net_venue_pct.abs().mean(),
                "net_abs_max": rr.net_venue_pct.abs().max(),
            })
    return pd.DataFrame(t2_rows), pd.DataFrame(pair_rows), pd.DataFrame(stress_rows)


# ------------------------------------------------------------------
# 2. Conditional slope of r_{t-1} in the interaction specification
# ------------------------------------------------------------------
def conditional_slope(usdt: pd.DataFrame) -> pd.DataFrame:
    df = build_dataset(usdt)
    train = df.iloc[: int(len(df) * 0.70)]
    cols = BASE_FEATURES + [f"{a}_x_{b}" for a, b in INTERACTIONS]
    X, y = train[cols], train["depeg"].astype(int)
    k_r, k_int = cols.index("ret_lag1"), cols.index("ret_lag1_x_ret_vol_30")
    sig_grid = {f"P{q}": train["ret_vol_30"].quantile(q / 100) for q in (10, 25, 50, 75, 90)}

    def slopes(Xb, yb):
        sc = StandardScaler().fit(Xb)
        m = LogisticRegression(max_iter=4000).fit(sc.transform(Xb), yb)
        b_r, b_x = m.coef_[0][k_r], m.coef_[0][k_int]
        s_r, s_x = sc.scale_[k_r], sc.scale_[k_int]
        # d eta / d r = b_r / s_r + b_x * sigma / s_x ; reported per one SD of r_{t-1}
        # (SD taken from the estimation sample) so that it is comparable with Table 5
        return np.array([(b_r / s_r + b_x * s / s_x) * s_r for s in sig_grid.values()]), b_r, b_x

    point, b_r, b_x = slopes(X, y)
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(B_COEF):
        idx = moving_block_bootstrap_indices(len(train), BLOCK_LEN, rng)
        yb = y.iloc[idx]
        if yb.nunique() < 2:
            continue
        boot.append(slopes(X.iloc[idx], yb)[0])
    boot = np.array(boot)
    rows = []
    for k, (name, s) in enumerate(sig_grid.items()):
        p = 2 * min((boot[:, k] <= 0).mean(), (boot[:, k] >= 0).mean())
        rows.append({"sigma30_quantile": name, "sigma30": s,
                     "slope_logodds_per_sd": point[k],
                     "odds_ratio_per_sd": np.exp(point[k]),
                     "ci95_lo": np.percentile(boot[:, k], 2.5),
                     "ci95_hi": np.percentile(boot[:, k], 97.5),
                     "p_boot": p, "B_eff": len(boot),
                     "b_ret_lag1_std": b_r, "b_interaction_std": b_x,
                     "sd_ret_lag1": train["ret_lag1"].std(),
                     "sd_sigma30": train["ret_vol_30"].std(),
                     "mean_sigma30": train["ret_vol_30"].mean()})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# 3. Out-of-sample bootstrap intervals
# ------------------------------------------------------------------
def oos_bootstrap(y: np.ndarray, preds: dict, ref: str = "Logistic Regression"):
    names = list(preds)
    metrics = {"AUROC": roc_auc_score, "AUPRC": average_precision_score,
               "Brier": brier_score_loss}
    n = len(y)
    rng = np.random.default_rng(SEED)
    draws = {(c, m): [] for c in names for m in metrics}
    diffs = {(c, m): [] for c in names if c != ref for m in metrics}
    skipped = 0
    for _ in range(B_OOS):
        idx = moving_block_bootstrap_indices(n, BLOCK_LEN, rng)
        yb = y[idx]
        if yb.min() == yb.max():
            skipped += 1
            continue
        vals = {(c, m): f(yb, preds[c][idx]) for c in names for m, f in metrics.items()}
        for key, v in vals.items():
            draws[key].append(v)
        for c, m in diffs:
            diffs[(c, m)].append(vals[(c, m)] - vals[(ref, m)])

    ci_rows = []
    for c in names:
        for m, f in metrics.items():
            d = np.array(draws[(c, m)])
            ci_rows.append({"classifier": c, "metric": m, "point": f(y, preds[c]),
                            "ci95_lo": np.percentile(d, 2.5), "ci95_hi": np.percentile(d, 97.5),
                            "B_eff": len(d)})
    diff_rows = []
    for (c, m), d in diffs.items():
        d = np.array(d)
        point = metrics[m](y, preds[c]) - metrics[m](y, preds[ref])
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        z, p_dl = delong_test(y, preds[ref], preds[c]) if m == "AUROC" else (np.nan, np.nan)
        diff_rows.append({"comparison": f"{c} - {ref}", "metric": m,
                          "diff_point": point,
                          "ci95_lo": np.percentile(d, 2.5), "ci95_hi": np.percentile(d, 97.5),
                          "ci90_lo": np.percentile(d, 5), "ci90_hi": np.percentile(d, 95),
                          "p_boot": p, "delong_z": z, "delong_p": p_dl, "B_eff": len(d)})
    return pd.DataFrame(ci_rows), pd.DataFrame(diff_rows), skipped


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(DATA_DIR / "panel_overlap_btc_eth_3venues.csv", parse_dates=["date_utc"])
    spreads = pd.read_csv(DATA_DIR / "spreads_overlap_btc_eth_3venues.csv", parse_dates=["date_utc"])
    t2, pairs, stress = variance_decomposition(panel, spreads)
    t2.to_csv(OUT_DIR / "variance_decomposition_table2.csv", index=False)
    pairs.to_csv(OUT_DIR / "variance_decomposition_pairs.csv", index=False)
    stress.to_csv(OUT_DIR / "variance_decomposition_stress.csv", index=False)
    print(t2.round(4).to_string(index=False))
    print(stress.round(4).to_string(index=False))

    usdt = (pd.read_csv(DATA_DIR / "usdt_kraken.csv", parse_dates=["date_utc"])
            .rename(columns={"date_utc": "date"})[["date", "close_px", "volume_base"]])
    cs = conditional_slope(usdt)
    cs.to_csv(OUT_DIR / "interaction_conditional_slope.csv", index=False)
    print(cs.round(4).to_string(index=False))

    df = pd.read_csv(DATA_DIR / "usdt_depeg_dataset.csv").sort_values("date").reset_index(drop=True)
    df["depeg"] = (df["abs_dev"] > THRESH).astype(int).shift(-1)
    df = df.dropna(subset=["depeg"]).reset_index(drop=True)
    X, y = df[FEATURE_COLS].values, df["depeg"].astype(int).values
    T0 = int(len(y) * 0.70)
    preds = expanding_window_predictions(X, y, FEATURE_COLS, T0)
    y_test = y[T0:]
    pd.DataFrame({"date": df["date"].iloc[T0:].values, "y": y_test, **preds}).to_csv(
        OUT_DIR / "oos_predictions.csv", index=False)
    ci, diff, skipped = oos_bootstrap(y_test, preds)
    ci.to_csv(OUT_DIR / "oos_bootstrap_ci.csv", index=False)
    diff.to_csv(OUT_DIR / "oos_bootstrap_diff.csv", index=False)
    print(f"N={len(y)}, T0={T0}, test={len(y_test)}, test events={y_test.sum()}, skipped draws={skipped}")
    print(ci.round(3).to_string(index=False))
    print(diff.round(3).to_string(index=False))

    # Table 8: baseline versus interaction specification, same expanding window
    # (as in src/expanding_window_supplementary.py, Part 2)
    cols = list(FEATURE_COLS)
    for a, b in INTERACTIONS:
        df[f"{a}_x_{b}"] = df[a] * df[b]
        cols.append(f"{a}_x_{b}")
    Xi = df[cols].values
    p_int = np.zeros(len(y) - T0)
    for t in range(T0, len(y)):
        sc = StandardScaler().fit(Xi[:t])
        m = LogisticRegression(max_iter=2000, random_state=SEED).fit(sc.transform(Xi[:t]), y[:t])
        p_int[t - T0] = m.predict_proba(sc.transform(Xi[t:t + 1]))[0, 1]
    ci_i, diff_i, _ = oos_bootstrap(y_test, {"Baseline": preds["Logistic Regression"],
                                             "Interaction": p_int}, ref="Baseline")
    ci_i.to_csv(OUT_DIR / "oos_bootstrap_ci_interaction.csv", index=False)
    diff_i.to_csv(OUT_DIR / "oos_bootstrap_diff_interaction.csv", index=False)
    print(ci_i.round(3).to_string(index=False))
    print(diff_i.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
