"""
Fetch Polymarket recession-market price histories and Treasury yield curve data.

Sources:
  - Polymarket Gamma API (event metadata) — https://gamma-api.polymarket.com/events
  - Polymarket CLOB API (price-history time series) — https://clob.polymarket.com/prices-history
  - Yahoo Finance (Treasury yields ^IRX, ^FVX, ^TNX, ^TYX) via yfinance

Outputs (under data/):
  - polymarket_<slug>.csv    one row per timestamp with the YES-token price
  - treasury_yields.csv      daily yields and 10y-3m spread
  - markets_meta.json        event/market metadata for reproducibility
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"

# Recession markets we want, indexed by event slug.
# These are the seven recession-themed events with non-trivial volume that
# we located by paginating the closed-event endpoint sorted by startDate.
RECESSION_EVENTS = [
    "us-recession-in-2025",                          # ~$11.7M volume, main market
    "us-recession-by-end-of-2026",                   # ~$1.5M, active
    "us-recession-in-2024-1",                        # ~$876K
    "us-recession-announced-by-nber-before-june-2025",  # ~$859K
    "negative-gdp-growth-in-2026",                   # ~$27K (sanity check)
]

# Manifold markets for cross-platform comparison (binary YES/NO recession markets).
MANIFOLD_MARKETS = {
    "us-recession-end-2025-two-quarters": "f2i3zt28mm",   # resolved NO
    "us-recession-by-end-of-2024": "IMLre6PYIQgsesV6Wa6O", # resolved NO
}


def fetch_event(slug: str) -> dict:
    r = requests.get(f"{GAMMA}/events", params={"slug": slug}, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not data:
        raise RuntimeError(f"No event for slug={slug}")
    return data[0]


def fetch_manifold_bets(market_id: str) -> pd.DataFrame:
    """Fetch all bets for a Manifold market, returning ts + post-trade probability."""
    rows: list[dict] = []
    before: str | None = None
    while True:
        params = {"contractId": market_id, "limit": 1000}
        if before:
            params["before"] = before
        r = requests.get("https://api.manifold.markets/v0/bets", params=params, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < 1000:
            break
        before = batch[-1]["id"]
        time.sleep(0.1)
    if not rows:
        return pd.DataFrame(columns=["ts", "price"])
    df = pd.DataFrame(rows)
    df = df[["createdTime", "probAfter"]].rename(columns={"createdTime": "ts", "probAfter": "price"})
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.sort_values("ts").reset_index(drop=True)


def fetch_price_history(token_id: str, fidelity: int = 720) -> pd.DataFrame:
    """Fetch price-history for a CLOB token.

    Resolved markets only support fidelity >= 720 (12h). We use 720 to maximize
    granularity, then resample to daily for analysis.
    """
    r = requests.get(
        f"{CLOB}/prices-history",
        params={"market": token_id, "interval": "max", "fidelity": fidelity},
        timeout=30,
    )
    r.raise_for_status()
    history = r.json().get("history", [])
    if not history:
        return pd.DataFrame(columns=["ts", "price"])
    df = pd.DataFrame(history).rename(columns={"t": "ts", "p": "price"})
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    return df.sort_values("ts").reset_index(drop=True)


def main() -> None:
    meta: dict[str, dict] = {}

    for slug in RECESSION_EVENTS:
        print(f"[poly] {slug}")
        event = fetch_event(slug)
        markets = event.get("markets", [])
        # Most events have one binary YES/NO market; take the first.
        m = markets[0]
        token_ids = json.loads(m["clobTokenIds"]) if isinstance(m["clobTokenIds"], str) else m["clobTokenIds"]
        yes_token = token_ids[0]  # YES outcome
        prices = fetch_price_history(yes_token, fidelity=720)
        out = DATA_DIR / f"polymarket_{slug}.csv"
        prices.to_csv(out, index=False)
        print(f"  -> {len(prices)} rows -> {out.name}")
        outcome_prices = json.loads(m["outcomePrices"]) if isinstance(m["outcomePrices"], str) else m["outcomePrices"]
        meta[slug] = {
            "event_title": event.get("title"),
            "start_date": event.get("startDate"),
            "end_date": event.get("endDate"),
            "volume_usd": float(event.get("volume", 0) or 0),
            "market_id": m.get("id"),
            "market_question": m.get("question"),
            "outcomes": m.get("outcomes"),
            "outcome_prices_at_close": outcome_prices,
            "yes_token_id": yes_token,
        }
        time.sleep(0.25)

    # Manifold cross-platform comparison.
    for name, mid in MANIFOLD_MARKETS.items():
        print(f"[manifold] {name}")
        df = fetch_manifold_bets(mid)
        out = DATA_DIR / f"manifold_{name}.csv"
        df.to_csv(out, index=False)
        print(f"  -> {len(df)} bets -> {out.name}")
        time.sleep(0.25)

    with open(DATA_DIR / "markets_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Treasury yields — pull 2021-01 through latest to span all our markets.
    print("[treasury] downloading ^IRX ^FVX ^TNX ^TYX from Yahoo Finance")
    tickers = ["^IRX", "^FVX", "^TNX", "^TYX"]
    yt = yf.download(tickers, start="2021-01-01", end="2026-05-15", progress=False, auto_adjust=True)
    close = yt["Close"].copy()
    close.columns = [c.replace("^", "") for c in close.columns]
    # Spreads (in percentage points). ^IRX is the 3-month T-bill.
    close["spread_10y_3m"] = close["TNX"] - close["IRX"]
    close["spread_10y_5y"] = close["TNX"] - close["FVX"]
    close["spread_30y_10y"] = close["TYX"] - close["TNX"]
    close.index.name = "date"
    out = DATA_DIR / "treasury_yields.csv"
    close.to_csv(out)
    print(f"  -> {len(close)} rows -> {out.name}")
    print(f"  -> date range: {close.index.min().date()} to {close.index.max().date()}")


if __name__ == "__main__":
    main()
