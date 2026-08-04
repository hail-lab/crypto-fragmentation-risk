# src/calibration_plot_pub.py
"""Generate publication-quality calibration plot for IEEE Access."""
from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA_CLEAN = Path(__file__).resolve().parents[1] / "data" / "clean"
OUTPUTS    = Path(__file__).resolve().parents[1] / "outputs"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})


def main() -> None:
    df = pd.read_csv(DATA_CLEAN / "depeg_predictions.csv")
    df = df.sort_values("date")

    probs = df["logit_prob"]
    y     = df["depeg"]

    # 5 equal-width bins
    bins = np.linspace(0, 1, 6)
    df["bin"] = pd.cut(probs, bins, include_lowest=True)

    calib = (
        df.groupby("bin", observed=False)
          .agg(
              mean_prob=("logit_prob", "mean"),
              actual_rate=("depeg", "mean"),
              count=("depeg", "size"),
          )
          .dropna()
          .reset_index()
    )

    fig, ax = plt.subplots(figsize=(4.5, 4.0))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "--", color="#9E9E9E", linewidth=1.0, label="Perfect calibration")

    # Calibration curve
    ax.plot(calib["mean_prob"], calib["actual_rate"],
            "o-", color="#1565C0", linewidth=1.8, markersize=7,
            markeredgecolor="white", markeredgewidth=0.8,
            label="Logistic model")

    # Annotate bin counts
    for _, row in calib.iterrows():
        ax.annotate(f"n={int(row['count'])}",
                    (row["mean_prob"], row["actual_rate"]),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=7, color="#616161")

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Frequency")
    ax.set_title("Calibration: Logistic Micro-Depeg Model (5 Bins)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", framealpha=0.9)

    fig.tight_layout()
    out = OUTPUTS / "calibration_logistic.png"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
