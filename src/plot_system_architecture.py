# src/plot_system_architecture.py
"""Generate a system architecture diagram (Fig. 6) for IEEE Access paper."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
import os

OUTPUTS = Path(__file__).resolve().parents[1] / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

# Optional: also save to a submission directory (set SUBMISSION_DIR env var)
_sub = os.environ.get("SUBMISSION_DIR", "")
PAPER_DIR = Path(_sub) if _sub else None


def rounded_box(ax, xy, w, h, color, alpha=0.12):
    box = FancyBboxPatch(xy, w, h,
                         boxstyle="round,pad=0.006,rounding_size=0.015",
                         linewidth=1.4, edgecolor=color,
                         facecolor=color, alpha=alpha)
    ax.add_patch(box)
    border = FancyBboxPatch(xy, w, h,
                            boxstyle="round,pad=0.006,rounding_size=0.015",
                            linewidth=1.4, edgecolor=color, facecolor="none")
    ax.add_patch(border)


def arrow(ax, start, end, color="#546E7A", style="-|>"):
    a = FancyArrowPatch(start, end, arrowstyle=style,
                        mutation_scale=11, linewidth=1.4, color=color)
    ax.add_patch(a)


def label(ax, x, y, text, fs=8, color="#212121", **kw):
    ax.text(x, y, text, fontsize=fs, color=color, va="top",
            family="sans-serif", **kw)


def title(ax, x, y, text, color, fs=8.5):
    ax.text(x, y, text, fontsize=fs, fontweight="bold", color=color,
            va="top", family="sans-serif")


# Colours
C_DATA  = "#1565C0"
C_PROC  = "#00838F"
C_FRAG  = "#2E7D32"
C_PRED  = "#C62828"
C_OUT   = "#6A1B9A"
C_ARROW = "#546E7A"


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Title
    ax.text(0.50, 0.97,
            "Walk-Forward Risk Monitoring Pipeline",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color="#0b3c5d", family="serif")

    # ── Column 1: Data Ingestion ──
    bw = 0.17
    x1 = 0.02

    rounded_box(ax, (x1, 0.72), bw, 0.17, C_DATA)
    title(ax, x1 + 0.01, 0.88, "DATA INGESTION", C_DATA)
    label(ax, x1 + 0.012, 0.845, "Binance API\nKraken API\nCoinbase API\nUSDT/USD (Kraken)", fs=7)

    rounded_box(ax, (x1, 0.47), bw, 0.19, C_PROC)
    title(ax, x1 + 0.01, 0.65, "PREPROCESSING", C_PROC)
    label(ax, x1 + 0.012, 0.61, "Temporal alignment\nUTC normalisation\nAnomaly filtering\nLog returns + features", fs=7)

    # Arrow data -> preprocess
    arrow(ax, (x1 + bw/2, 0.72), (x1 + bw/2, 0.665))

    # ── Column 2: Two parallel modules ──
    x2 = 0.25
    mw = 0.24

    # MODULE A — Fragmentation
    rounded_box(ax, (x2, 0.62), mw, 0.28, C_FRAG)
    title(ax, x2 + 0.01, 0.89, "MODULE A", C_FRAG, fs=8)
    label(ax, x2 + 0.01, 0.855, "Cross-Venue Fragmentation", fs=7.5, fontweight="bold", color=C_FRAG)
    label(ax, x2 + 0.012, 0.81,
          "\u2022 Triangular pairwise spreads\n"
          "\u2022 Rolling Z-score anomalies\n"
          "\u2022 Variance decomposition\n"
          "\u2022 USD conversion of USDT quotes\n"
          "\u2022 EWMA & sample volatility\n"
          "\u2022 Historical VaR (1%, 5%)", fs=7)

    # MODULE B — Prediction
    rounded_box(ax, (x2, 0.12), mw, 0.42, C_PRED)
    title(ax, x2 + 0.01, 0.53, "MODULE B", C_PRED, fs=8)
    label(ax, x2 + 0.01, 0.495, "Micro-Depeg Classification", fs=7.5, fontweight="bold", color=C_PRED)
    label(ax, x2 + 0.012, 0.45,
          "\u2022 Features: current deviation |d\u209c|,\n"
          "  \u03c3\u2077, \u03c3\u00b9\u2074, \u03c3\u00b3\u2070, returns, volume\n"
          "\u2022 Expanding-window protocol\n"
          "\u2022 Persistence benchmark\n"
          "\u2022 5-classifier benchmark\n"
          "  (LR, RF, GB, SVM, MLP)\n"
          "\u2022 DeLong tests, block bootstrap\n"
          "\u2022 Spreads as depeg predictors\n"
          "\u2022 SHAP feature attribution\n"
          "\u2022 USDC cross-stablecoin test", fs=7)

    # Arrows preprocess -> modules
    arrow(ax, (x1 + bw, 0.56), (x2, 0.80))
    arrow(ax, (x1 + bw, 0.52), (x2, 0.40))

    # ── Column 3: Evaluation ──
    x3 = 0.55
    ew = 0.20

    # Evaluation A
    rounded_box(ax, (x3, 0.68), ew, 0.22, C_FRAG, alpha=0.06)
    title(ax, x3 + 0.01, 0.89, "DIAGNOSTICS", C_FRAG, fs=8)
    label(ax, x3 + 0.012, 0.845,
          "Net venue component:\n"
          "≤ 1.2% of variance (quoted)\n"
          "< 0.5% in U.S. dollar terms\n\n"
          "VaR invariant to\n"
          "venue choice", fs=7)

    # Evaluation B
    rounded_box(ax, (x3, 0.18), ew, 0.40, C_PRED, alpha=0.06)
    title(ax, x3 + 0.01, 0.57, "EVALUATION", C_PRED, fs=8)
    label(ax, x3 + 0.012, 0.525,
          "AUROC, AUPRC, Brier\n"
          "Calibration diagnostics\n"
          "Block-bootstrap CIs\n"
          "DeLong comparisons\n"
          "SHAP attribution\n\n"
          "Coefficient & SHAP\n"
          "interpretation for\n"
          "risk attribution", fs=7)

    arrow(ax, (x2 + mw, 0.76), (x3, 0.79))
    arrow(ax, (x2 + mw, 0.33), (x3, 0.38))

    # ── Column 4: Output ──
    x4 = 0.80
    ow = 0.18

    rounded_box(ax, (x4, 0.52), ow, 0.38, C_OUT)
    title(ax, x4 + 0.01, 0.89, "OUTPUTS", C_OUT, fs=8)
    label(ax, x4 + 0.012, 0.845,
          "\u25b6 Venue dispersion\n"
          "   alert flags\n\n"
          "\u25b6 Daily depeg\n"
          "   probability score\n\n"
          "\u25b6 Risk attribution\n"
          "   (interpretable\n"
          "    coefficients)\n\n"
          "\u25b6 Regime state\n"
          "   indicator", fs=7)

    arrow(ax, (x3 + ew, 0.79), (x4, 0.79))
    arrow(ax, (x3 + ew, 0.38), (x4, 0.62))

    # Bottom note
    ax.text(0.50, 0.04,
            "All predictions use strictly causal information (features at t, labels at t+1). "
            "Coefficients are auditable for regulatory compliance.",
            ha="center", va="center", fontsize=7, fontstyle="italic",
            color="#607D8B", family="serif")

    # Save
    fig.savefig(str(OUTPUTS / "system_architecture.png"),
                bbox_inches="tight", pad_inches=0.02, dpi=300)
    fig.savefig(str(OUTPUTS / "system_architecture.pdf"),
                bbox_inches="tight", pad_inches=0.02)
    print(f"Saved: {OUTPUTS / 'system_architecture.png'}")
    if PAPER_DIR and PAPER_DIR.exists():
        fig.savefig(str(PAPER_DIR / "system_architecture.png"),
                    bbox_inches="tight", pad_inches=0.02, dpi=300)
        fig.savefig(str(PAPER_DIR / "system_architecture.pdf"),
                    bbox_inches="tight", pad_inches=0.02)
        print(f"Saved: {PAPER_DIR / 'system_architecture.png'}")

    plt.close(fig)


if __name__ == "__main__":
    main()
