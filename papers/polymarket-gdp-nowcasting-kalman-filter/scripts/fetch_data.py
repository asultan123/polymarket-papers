"""Fetch Polymarket bucket markets, price histories, and Yahoo Finance macro series.

All sources are public, no API keys required.

Outputs
-------
- data/polymarket_markets.json      list of relevant US GDP/recession markets (gamma snapshot)
- data/polymarket_prices/<slug>.csv daily price for each market (CLOB price-history)
- data/yfinance.csv                 wide table of daily ^GSPC, ^IRX (3m T-bill), ^TNX (10y),
                                    HYG (HY credit), DXY (^DXY) prices, all adj-close
- data/run_summary.json             metadata (counts, timestamps, dates)

Note: BEA quarterly GDP growth is reconstructed from the bucket that resolved to 1
inside each quarter's set of Polymarket markets (resolution mirrors the BEA release).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PRICES = DATA / "polymarket_prices"
DATA.mkdir(exist_ok=True)
PRICES.mkdir(exist_ok=True)

GAMMA = "https://gamma-api.polymarket.com/markets"
CLOB_HISTORY = "https://clob.polymarket.com/prices-history"

# Slugs we care about. We hand-curate the list of resolved US bucket markets so the
# implied "BEA truth" is unambiguous (one bucket resolved 1, others resolved 0).
US_GDP_BUCKETS: dict[str, list[str]] = {
    "Q1_2025": [
        "will-us-gdp-growth-be-greater-than-2-in-q1-2025",
        "will-us-gdp-growth-be-between-2-and-1-in-q1-2025",
        "will-us-gdp-growth-be-between-1-and-0-in-q1-2025",
        "will-us-gdp-growth-be-between-0-and-1-in-q1-2025",  # 0 to -1
        "will-us-gdp-growth-be-between-1-and-2-in-q1-2025",  # -1 to -2
        "will-us-gdp-growth-be-less-than-2-in-q1-2025",      # < -2
    ],
    "Q2_2025": [
        "will-us-gdp-growth-in-q2-2025-be-less-than-1pt5",
        "will-us-gdp-growth-in-q2-2025-be-between-1pt5-and-2pt0",
        "will-us-gdp-growth-in-q2-2025-be-between-2pt0-and-2pt5",
        "will-us-gdp-growth-in-q2-2025-be-between-2pt5-and-3pt0",
        "will-us-gdp-growth-in-q2-2025-be-between-3pt0-and-3pt5",
        "will-us-gdp-growth-in-q2-2025-be-greater-than-3pt5",
    ],
    "Q3_2025": [
        "will-us-gdp-growth-in-q2-2025-be-less-than-1pt0",
        "will-us-gdp-growth-in-q2-2025-be-between-1pt0-and-1pt5",
        "will-us-gdp-growth-in-q2-2025-be-between-1pt5-and-2pt0",
        "will-us-gdp-growth-in-q2-2025-be-between-2pt0-and-2pt5",
        "will-us-gdp-growth-in-q2-2025-be-between-2pt5-and-3pt0",
        "will-us-gdp-growth-in-q2-2025-be-between-3pt0-and-3pt5",
        "will-us-gdp-growth-in-q2-2025-be-greater-than-3pt5",
    ],
    "Q4_2025": [
        "will-us-gdp-growth-in-q4-2025-be-less-than-1pt0",
        "will-us-gdp-growth-in-q4-2025-be-between-1pt0-and-1pt5",
        "will-us-gdp-growth-in-q4-2025-be-between-1pt5-and-2pt0",
        "will-us-gdp-growth-in-q4-2025-be-between-2pt0-and-2pt5",
        "will-us-gdp-growth-in-q4-2025-be-between-2pt5-and-3pt0",
        "will-us-gdp-growth-in-q4-2025-be-between-3pt0-and-3pt5",
        "will-us-gdp-growth-in-q4-2025-be-greater-than-3pt5",
    ],
    "Q1_2026": [
        "will-us-gdp-growth-in-q1-2026-be-less-than-1pt0",
        "will-us-gdp-growth-in-q1-2026-be-between-1pt0-and-1pt5",
        "will-us-gdp-growth-in-q1-2026-be-between-1pt5-and-2pt0",
        "will-us-gdp-growth-in-q1-2026-be-between-2pt0-and-2pt5",
        "will-us-gdp-growth-in-q1-2026-be-between-2pt5-and-3pt0",
        "will-us-gdp-growth-in-q1-2026-be-between-3pt0-and-3pt5",
        "will-us-gdp-growth-in-q1-2026-be-greater-than-3pt5",
    ],
}

# Bucket midpoints (annualized real-GDP-growth %). Open-ended top/bottom buckets get a
# nominal value 0.5 pp beyond the cut.
BUCKET_MIDPOINTS_Q1_2025 = {
    "will-us-gdp-growth-be-greater-than-2-in-q1-2025": 3.0,
    "will-us-gdp-growth-be-between-2-and-1-in-q1-2025": 1.5,
    "will-us-gdp-growth-be-between-1-and-0-in-q1-2025": 0.5,
    "will-us-gdp-growth-be-between-0-and-1-in-q1-2025": -0.5,
    "will-us-gdp-growth-be-between-1-and-2-in-q1-2025": -1.5,
    "will-us-gdp-growth-be-less-than-2-in-q1-2025": -2.5,
}
BUCKET_MIDPOINTS_REG = {  # regular 0.5pp buckets
    "less-than-1pt0": 0.5,
    "less-than-1pt5": 1.0,
    "between-1pt0-and-1pt5": 1.25,
    "between-1pt5-and-2pt0": 1.75,
    "between-2pt0-and-2pt5": 2.25,
    "between-2pt5-and-3pt0": 2.75,
    "between-3pt0-and-3pt5": 3.25,
    "greater-than-3pt5": 4.0,
}

# Auxiliary markets (US recession + Fed cuts)
AUX_SLUGS = [
    "us-recession-in-2025",
    "us-recession-by-end-of-2026",
    "negative-gdp-growth-in-q1-2025",
    "negative-gdp-growth-in-q2-2025",
    "negative-gdp-growth-in-q3-2025",
    "negative-gdp-growth-in-q4-2025-295",
    "negative-gdp-growth-in-2025",
    "negative-gdp-growth-in-2026",
]


def fetch_market_meta(slug: str) -> dict[str, Any] | None:
    """Return the gamma metadata for a slug, or None if not found. Falls back to
    paging when slug filter returns wrong rows."""
    r = requests.get(GAMMA, params={"slug": slug}, timeout=30)
    r.raise_for_status()
    matches = [m for m in r.json() if m.get("slug") == slug]
    if matches:
        return matches[0]
    # Page through closed markets to find by slug
    for tag_id in [102000, 101800, 370, 100201]:
        for closed in ("true", "false"):
            offset = 0
            while offset < 2000:
                rr = requests.get(GAMMA, params={"limit": 500, "tag_id": tag_id,
                                                 "offset": offset, "closed": closed}, timeout=30)
                rows = rr.json()
                if not rows:
                    break
                for m in rows:
                    if m.get("slug") == slug:
                        return m
                if len(rows) < 500:
                    break
                offset += 500
    return None


def fetch_price_history(token_id: str, fidelity: int = 1440) -> pd.DataFrame:
    """Daily-fidelity (1440 minutes) price history for one CLOB token.
    Returns DataFrame with columns date, price."""
    r = requests.get(CLOB_HISTORY, params={"market": token_id, "interval": "all",
                                            "fidelity": fidelity}, timeout=60)
    r.raise_for_status()
    hist = r.json().get("history", [])
    if not hist:
        return pd.DataFrame(columns=["date", "price"])
    df = pd.DataFrame(hist)
    df["date"] = pd.to_datetime(df["t"], unit="s", utc=True).dt.tz_convert(None).dt.normalize()
    df = df.rename(columns={"p": "price"}).drop(columns=["t"])
    df = df.groupby("date", as_index=False)["price"].last()
    return df


def fetch_all_polymarket() -> dict[str, Any]:
    out: dict[str, Any] = {"buckets": {}, "aux": {}, "fetched_at": dt.datetime.utcnow().isoformat()}
    all_slugs = set()
    for slugs in US_GDP_BUCKETS.values():
        all_slugs.update(slugs)
    all_slugs.update(AUX_SLUGS)

    for slug in sorted(all_slugs):
        print(f"[polymarket] fetching {slug}")
        meta = fetch_market_meta(slug)
        if meta is None:
            print("  -> not found, skipping")
            continue
        ids = meta.get("clobTokenIds")
        if isinstance(ids, str):
            ids = json.loads(ids)
        if not ids:
            print("  -> no CLOB token ids, skipping")
            continue
        yes_token = ids[0]
        df = fetch_price_history(yes_token, fidelity=1440)
        if df.empty:
            print("  -> empty history, skipping")
            continue
        df.to_csv(PRICES / f"{slug}.csv", index=False)
        # Decide whether this slug is a bucket or aux
        record = {
            "slug": slug,
            "question": meta.get("question"),
            "conditionId": meta.get("conditionId"),
            "yes_token": yes_token,
            "n_obs": len(df),
            "start": df.date.min().date().isoformat(),
            "end": df.date.max().date().isoformat(),
            "outcomePrices": meta.get("outcomePrices"),
            "closed": meta.get("closed"),
            "volumeNum": meta.get("volumeNum"),
            "endDate": meta.get("endDate"),
            "startDate": meta.get("startDate"),
        }
        bucket_quarter = None
        for q, slugs in US_GDP_BUCKETS.items():
            if slug in slugs:
                bucket_quarter = q
                break
        if bucket_quarter:
            out["buckets"].setdefault(bucket_quarter, []).append(record)
        else:
            out["aux"][slug] = record
        time.sleep(0.4)  # be polite

    with (DATA / "polymarket_markets.json").open("w") as f:
        json.dump(out, f, indent=2)
    return out


def fetch_yahoo() -> pd.DataFrame:
    tickers = {
        "SP500": "^GSPC",
        "TBILL_3M": "^IRX",      # 13-week T-bill yield (annualized %)
        "TNX_10Y": "^TNX",       # 10y yield
        "FVX_5Y": "^FVX",        # 5y yield
        "HYG": "HYG",            # iShares HY corp bond ETF (credit)
        "TLT": "TLT",            # 20y+ Treasury ETF (duration)
        "DXY": "DX-Y.NYB",       # USD index
        "XLI": "XLI",            # Industrials sector ETF
        "XLY": "XLY",            # Consumer discretionary
        "XLU": "XLU",            # Utilities
    }
    df_all = []
    for name, sym in tickers.items():
        print(f"[yfinance] {name} -> {sym}")
        h = yf.Ticker(sym).history(start="2024-09-01", end="2026-05-17", auto_adjust=True)
        if h.empty:
            print("  -> empty")
            continue
        series = h["Close"].rename(name)
        series.index = series.index.tz_localize(None).normalize()
        df_all.append(series)
    df = pd.concat(df_all, axis=1).sort_index()
    df.index.name = "date"
    df.to_csv(DATA / "yfinance.csv")
    return df


def main() -> None:
    pm = fetch_all_polymarket()
    yf_df = fetch_yahoo()
    summary = {
        "fetched_at": dt.datetime.utcnow().isoformat(),
        "n_bucket_quarters": len(pm["buckets"]),
        "n_aux_markets": len(pm["aux"]),
        "bucket_counts": {q: len(rows) for q, rows in pm["buckets"].items()},
        "yfinance_date_range": [yf_df.index.min().date().isoformat(),
                                 yf_df.index.max().date().isoformat()],
        "yfinance_columns": list(yf_df.columns),
        "yfinance_rows": len(yf_df),
    }
    with (DATA / "run_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
