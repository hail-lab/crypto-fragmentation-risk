# src/vol_var_3venues.py
"""Generate cross-venue volatility and VaR comparison table for 3 venues."""
from __future__ import annotations

import pandas as pd
import numpy as np

from src.config import DATA_CLEAN, OUTPUTS

LAMBDA = 0.94  # EWMA decay


def compute_ewma_vol(r: pd.Series, lam: float = LAMBDA) -> float:
    var = 0.0
    for x in r:
        var = lam * var + (1 - lam) * x**2
    return np.sqrt(var)


def historical_var(r: pd.Series, alpha: float) -> float:
    return np.quantile(r, alpha)


def main() -> None:
    panel = pd.read_csv(DATA_CLEAN / "panel_overlap_btc_eth_3venues.csv")
    panel = panel.sort_values(["asset", "date_utc"])

    venues = ["binance", "kraken", "coinbase"]
    results = []

    for asset, g in panel.groupby("asset"):
        g = g.copy().reset_index(drop=True)

        for venue in venues:
            col = f"close_{venue}"
            g[f"ret_{venue}"] = np.log(g[col]).diff()

        for venue in venues:
            r = g[f"ret_{venue}"].dropna()
            sample_vol = r.std()
            ewma_vol = compute_ewma_vol(r)
            var_1 = historical_var(r, 0.01)
            var_5 = historical_var(r, 0.05)

            results.append({
                "asset": asset,
                "venue": venue.capitalize(),
                "sample_vol": round(sample_vol, 6),
                "ewma_vol": round(ewma_vol, 6),
                "VaR_1pct": round(var_1, 6),
                "VaR_5pct": round(var_5, 6),
            })

    out = pd.DataFrame(results)
    out_path = OUTPUTS / "vol_var_3venues.csv"
    out.to_csv(out_path, index=False)

    print("\nCross-Venue Volatility & VaR Comparison (3 venues):")
    print(out.to_string(index=False))
    print(f"\nSaved: {out_path}")

    # Also print observation counts for verification
    for asset, g in panel.groupby("asset"):
        print(f"\n{asset}: {len(g)} daily observations per venue")


if __name__ == "__main__":
    main()
