"""
Rebuild the three-venue BTC/ETH panel by retrieving Coinbase daily closes.

Coinbase market data are licensed for personal or research use and may not be
redistributed to third parties without Coinbase's written consent (Coinbase
Market Data Terms of Use; Coinbase Developer Platform Terms). This repository
therefore ships only the Binance and Kraken columns:

    data/clean/panel_overlap_btc_eth_3venues.public.csv
    data/clean/spreads_overlap_btc_eth_3venues.public.csv

This script downloads the Coinbase BTC-USD and ETH-USD daily closes from
Coinbase's public candles endpoint (Advanced Trade API, via ccxt, as in
src/pull_coinbase_daily.py), merges them on (date_utc, asset), recomputes the
Binance-Coinbase and Kraken-Coinbase spreads, and writes the full files that
the analysis scripts read:

    data/clean/panel_overlap_btc_eth_3venues.csv
    data/clean/spreads_overlap_btc_eth_3venues.csv

Those two files are listed in .gitignore and must not be committed.
By running this script you obtain the data from Coinbase under its terms.

Usage:  python -m src.restore_coinbase_columns
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
CLEAN = REPO / "data" / "clean"
PRODUCTS = {"BTC": "BTC/USD", "ETH": "ETH/USD"}
DAY_MS = 24 * 60 * 60 * 1000


def fetch_closes(start: str, end: str) -> pd.DataFrame:
    import ccxt
    ex = ccxt.coinbase({"enableRateLimit": True})
    ex.load_markets()
    t0 = int(datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp() * 1000)
    t1 = int(datetime.fromisoformat(end).replace(tzinfo=timezone.utc).timestamp() * 1000) + DAY_MS
    rows = []
    for asset, symbol in PRODUCTS.items():
        ms = t0
        while ms < t1:
            batch = ex.fetch_ohlcv(symbol, timeframe="1d", since=ms, limit=300)
            if not batch:
                break
            rows += [(asset, b[0], b[4]) for b in batch if b[0] < t1]
            nxt = batch[-1][0] + DAY_MS
            if nxt <= ms:
                break
            ms = nxt
            time.sleep(ex.rateLimit / 1000)
    df = pd.DataFrame(rows, columns=["asset", "ts", "close_coinbase"]).drop_duplicates(["asset", "ts"])
    df["date_utc"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    return df[["date_utc", "asset", "close_coinbase"]]


def main() -> None:
    panel = pd.read_csv(CLEAN / "panel_overlap_btc_eth_3venues.public.csv")
    spreads = pd.read_csv(CLEAN / "spreads_overlap_btc_eth_3venues.public.csv")
    cb = fetch_closes(panel["date_utc"].min(), panel["date_utc"].max())

    full = panel.merge(cb, on=["date_utc", "asset"], how="left")
    missing = full["close_coinbase"].isna().sum()
    if missing:
        raise RuntimeError(f"{missing} panel days have no Coinbase close; check connectivity or symbols")
    full = full[["date_utc", "asset", "close_binance", "close_kraken", "close_coinbase"]]
    full.to_csv(CLEAN / "panel_overlap_btc_eth_3venues.csv", index=False)

    s = spreads.merge(cb, on=["date_utc", "asset"], how="left")
    s["spread_rel_bc"] = s["close_binance"] / s["close_coinbase"] - 1.0
    s["spread_rel_kc"] = s["close_kraken"] / s["close_coinbase"] - 1.0
    s = s[["date_utc", "asset", "close_binance", "close_kraken", "close_coinbase",
           "spread_rel_bk", "spread_rel_bc", "spread_rel_kc"]]
    s.to_csv(CLEAN / "spreads_overlap_btc_eth_3venues.csv", index=False)
    print(f"restored {len(full)} panel rows and {len(s)} spread rows with Coinbase closes")


if __name__ == "__main__":
    main()
