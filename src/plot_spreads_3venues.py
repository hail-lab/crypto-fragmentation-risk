# src/plot_spreads_3venues.py
"""Generate publication-quality triangular spread and Z-score plots for IEEE Access."""
from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA_CLEAN = Path(__file__).resolve().parents[1] / "data" / "clean"
OUTPUTS    = Path(__file__).resolve().parents[1] / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

ROLL_WINDOW = 30
Z_THRESH    = 3.0

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

PAIR_LABELS = {
    "spread_rel_bk": "Binance\u2013Kraken",
    "spread_rel_bc": "Binance\u2013Coinbase",
    "spread_rel_kc": "Kraken\u2013Coinbase",
}
COLORS = ["#1565C0", "#C62828", "#2E7D32"]


def plot_spreads_timeseries(df: pd.DataFrame, asset: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 2.8))
    for (col, label), c in zip(PAIR_LABELS.items(), COLORS):
        ax.plot(df["date"], df[col], label=label, linewidth=0.6, color=c, alpha=0.85)
    ax.axhline(0, color="grey", linewidth=0.4, linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Relative Spread")
    ax.set_title(f"{asset}: Triangular Cross-Venue Relative Spreads")
    # Legend below the axes so it never occludes the data
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=3, frameon=False, columnspacing=1.4)
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    out = OUTPUTS / f"spread_rel_timeseries_{asset}.png"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_zscore(df: pd.DataFrame, asset: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 2.8))
    for (col, label), c in zip(PAIR_LABELS.items(), COLORS):
        zcol = f"z_{col}"
        ax.plot(df["date"], df[zcol], label=label, linewidth=0.6, color=c, alpha=0.85)
    ax.axhline(Z_THRESH, color="#EF6C00", linewidth=1.0, linestyle="--", label=f"\u00b1{Z_THRESH} threshold")
    ax.axhline(-Z_THRESH, color="#EF6C00", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling Z-Score")
    ax.set_title(f"{asset}: Rolling Z-Scores of Triangular Spreads (window = {ROLL_WINDOW} days)")
    # Legend below the axes so it never occludes the data
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=4, frameon=False, columnspacing=1.2)
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    out = OUTPUTS / f"spread_rel_zscore_{asset}.png"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    raw = pd.read_csv(DATA_CLEAN / "spreads_overlap_btc_eth_3venues.csv")
    raw["date"] = pd.to_datetime(raw["date_utc"])
    raw = raw.sort_values(["asset", "date"])

    for asset, g in raw.groupby("asset"):
        g = g.copy().reset_index(drop=True)

        # Compute rolling Z-scores for each pair
        for col in PAIR_LABELS:
            rm = g[col].rolling(ROLL_WINDOW, min_periods=ROLL_WINDOW).mean()
            rs = g[col].rolling(ROLL_WINDOW, min_periods=ROLL_WINDOW).std()
            g[f"z_{col}"] = (g[col] - rm) / rs

        g = g.dropna().reset_index(drop=True)
        plot_spreads_timeseries(g, asset)
        plot_zscore(g, asset)


if __name__ == "__main__":
    main()
