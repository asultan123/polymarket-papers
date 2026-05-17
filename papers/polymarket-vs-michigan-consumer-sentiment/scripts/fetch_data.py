"""
fetch_data.py
-------------
Pulls the raw data used in the paper:
  1. University of Michigan Index of Consumer Sentiment (monthly, 1978-).
  2. Polymarket macro-related markets metadata via Gamma API.
  3. Polymarket daily price history per market via CLOB.

Writes everything under data/ as CSV / JSON. No FRED, no auth.

Usage:
    python fetch_data.py            # default
    python fetch_data.py --refresh  # force re-fetch
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Tags on the Polymarket Gamma API that surface macro markets.
# Discovered by paging the /tags endpoint.
MACRO_TAGS = [
    100201,  # recession
    100328,  # Economy
    101249,  # Macro Inflation
    102000,  # Macro Indicators
    103339,  # Fed Chair
    131,     # interest rates
    370,     # GDP
    130,     # economy1
    101247,  # Macro Graph
    101250,  # Macro Single
    103753,  # initial jobless claims
]

# Keywords used as a secondary filter on slug. A market is included if either:
#   - it has a macro tag, OR
#   - its slug contains one of these keywords.
MACRO_KEYWORDS = [
    "recession", "inflation", "fed-", "-fed-", "gdp", "unemploy",
    "jobless", "jobs-report", "cpi", "rate-cut", "rate-hike",
    "interest-rate", "powell", "jpow", "federal-reserve",
    "treasury", "yield", "consumer-sentiment",
]

GAMMA = "https://gamma-api.polymarket.com/markets"
CLOB_PRICES = "https://clob.polymarket.com/prices-history"
MICHIGAN_URL = "http://www.sca.isr.umich.edu/files/tbmics.csv"

MIN_VOLUME = 5_000          # in $; below this the daily price is noise-dominated
MIN_DAYS_HISTORY = 10        # require at least 10 daily ticks


def fetch_michigan(path: Path) -> pd.DataFrame:
    """Download and normalize University of Michigan Index of Consumer Sentiment."""
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])
    raw = requests.get(MICHIGAN_URL, timeout=30).content
    out = path.parent / "michigan_raw.csv"
    out.write_bytes(raw)
    df = pd.read_csv(out)
    df.columns = [c.strip() for c in df.columns]
    # cols: Month, YYYY, ICS_ALL
    df["date"] = pd.to_datetime(df["Month"] + " " + df["YYYY"].astype(str),
                                format="%B %Y")
    df = df.rename(columns={"ICS_ALL": "ics"})[["date", "ics"]].sort_values("date")
    df.to_csv(path, index=False)
    return df


def fetch_markets_for_tag(tag_id: int) -> list[dict]:
    """Page through all closed markets for a given tag_id."""
    out, offset, fails = [], 0, 0
    while fails < 3:
        try:
            r = requests.get(GAMMA, params={
                "tag_id": tag_id,
                "closed": "true",
                "limit": 100,
                "offset": offset,
                "order": "startDate",
                "ascending": "true",
            }, timeout=30)
            if r.status_code >= 500:
                fails += 1
                time.sleep(2)
                offset += 100
                continue
            r.raise_for_status()
            batch = r.json()
        except requests.RequestException:
            fails += 1
            time.sleep(2)
            offset += 100
            continue
        if not batch:
            break
        out.extend(batch)
        if len(batch) < 100:
            break
        offset += 100
        fails = 0
        time.sleep(0.1)
    return out


def fetch_markets_by_date_window(start: str, end: str) -> list[dict]:
    """Page through closed markets within a start-date window for slug-keyword filtering.

    Gamma returns 500 on deep pagination; we cap and skip with backoff.
    """
    out, offset, fails = [], 0, 0
    while offset < 6000 and fails < 3:
        try:
            r = requests.get(GAMMA, params={
                "closed": "true",
                "limit": 100,
                "offset": offset,
                "order": "startDate",
                "ascending": "true",
                "start_date_min": start,
                "start_date_max": end,
            }, timeout=30)
            if r.status_code >= 500:
                fails += 1
                time.sleep(2)
                offset += 100
                continue
            r.raise_for_status()
            batch = r.json()
        except requests.RequestException:
            fails += 1
            time.sleep(2)
            offset += 100
            continue
        if not batch:
            break
        out.extend(batch)
        if len(batch) < 100:
            break
        offset += 100
        fails = 0
        time.sleep(0.1)
    return out


def gather_macro_markets(path: Path) -> pd.DataFrame:
    """Return one row per candidate macro market, deduped."""
    if path.exists():
        return pd.read_csv(path)
    seen: dict[str, dict] = {}

    for tag in MACRO_TAGS:
        for m in fetch_markets_for_tag(tag):
            mid = str(m.get("id"))
            seen.setdefault(mid, m)["_via_tag"] = tag

    for window in [("2022-01-01", "2023-06-30"),
                   ("2023-06-30", "2024-06-30"),
                   ("2024-06-30", "2025-06-30"),
                   ("2025-06-30", "2026-06-30")]:
        for m in fetch_markets_by_date_window(*window):
            slug = (m.get("slug") or "").lower()
            if not any(k in slug for k in MACRO_KEYWORDS):
                continue
            mid = str(m.get("id"))
            seen.setdefault(mid, m)

    rows = []
    for m in seen.values():
        vol = m.get("volumeNum") or 0
        if vol < MIN_VOLUME:
            continue
        try:
            tokens = json.loads(m.get("clobTokenIds") or "[]")
            outcomes = json.loads(m.get("outcomes") or "[]")
            prices_final = json.loads(m.get("outcomePrices") or "[]")
        except Exception:
            continue
        if len(tokens) < 2 or len(outcomes) < 2:
            continue
        rows.append({
            "id": m.get("id"),
            "slug": m.get("slug"),
            "question": m.get("question"),
            "volume": vol,
            "liquidity": m.get("liquidityNum"),
            "start_date": m.get("startDateIso"),
            "end_date": m.get("endDateIso"),
            "closed": m.get("closed"),
            "yes_token": tokens[0],
            "no_token": tokens[1] if len(tokens) > 1 else None,
            "yes_final": prices_final[0] if prices_final else None,
            "no_final": prices_final[1] if len(prices_final) > 1 else None,
            "outcomes": json.dumps(outcomes),
        })
    df = pd.DataFrame(rows).sort_values("start_date")
    df.to_csv(path, index=False)
    return df


def fetch_price_history(token_id: str) -> pd.DataFrame:
    """Daily price history for one CLOB token. Returns df with cols date, p."""
    r = requests.get(CLOB_PRICES, params={
        "market": token_id,
        "interval": "max",
        "fidelity": 1440,   # 1 day in minutes
    }, timeout=30)
    if r.status_code != 200:
        return pd.DataFrame()
    data = r.json().get("history", [])
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["t"], unit="s", utc=True).dt.tz_localize(None).dt.normalize()
    df = df[["date", "p"]].drop_duplicates("date").sort_values("date")
    return df


def gather_price_histories(markets: pd.DataFrame, path: Path,
                            max_markets: int | None = None,
                            workers: int = 8) -> pd.DataFrame:
    """Pull price history for every market in `markets`; long-format CSV.

    Uses a thread pool because the work is network-bound. Markets are
    pre-sorted by volume descending so the highest-impact ones come back
    first.
    """
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])

    from concurrent.futures import ThreadPoolExecutor, as_completed
    sub = markets.sort_values("volume", ascending=False).reset_index(drop=True)
    if max_markets:
        sub = sub.head(max_markets)
    log_path = path.parent / "fetch_progress.log"

    def fetch_one(row):
        token = str(row["yes_token"])
        df = fetch_price_history(token)
        if len(df) < MIN_DAYS_HISTORY:
            return None
        df["market_id"] = row["id"]
        df["slug"] = row["slug"]
        df["volume"] = row["volume"]
        return df

    rows = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_one, r): i for i, r in sub.iterrows()}
        for fut in as_completed(futs):
            done += 1
            try:
                df = fut.result()
                if df is not None:
                    rows.append(df)
            except Exception as e:
                with log_path.open("a") as f:
                    f.write(f"err {e}\n")
            if done % 50 == 0:
                msg = f"  [{done}/{len(sub)}] hits={len(rows)}\n"
                with log_path.open("a") as f:
                    f.write(msg)
                print(msg, end="", flush=True)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(path, index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    if args.refresh:
        for p in DATA_DIR.glob("*.csv"):
            p.unlink()

    print("[1/3] Michigan Consumer Sentiment ...")
    michigan = fetch_michigan(DATA_DIR / "michigan.csv")
    print(f"  rows: {len(michigan)}  span: {michigan['date'].min().date()} -> {michigan['date'].max().date()}")

    print("[2/3] Polymarket macro markets metadata ...")
    markets = gather_macro_markets(DATA_DIR / "macro_markets.csv")
    print(f"  candidate markets after volume filter: {len(markets)}")

    print("[3/3] Polymarket daily price histories ...")
    prices = gather_price_histories(markets, DATA_DIR / "macro_prices.csv",
                                     max_markets=1200, workers=10)
    print(f"  total daily price ticks: {len(prices)}")
    print(f"  markets with usable history: {prices['market_id'].nunique() if len(prices) else 0}")


if __name__ == "__main__":
    main()
