"""Fetch resolved Polymarket markets and their price histories.

Builds a dataset of binary YES/NO markets with:
  - resolution (1 if YES won, 0 if NO won)
  - market mid-price at several lookback horizons
    (T-1 day, T-7 days, T-30 days before resolution)
  - metadata (category, end date, volume)

Output: data/markets.parquet (or .csv if pyarrow unavailable).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

GAMMA = "https://gamma-api.polymarket.com/markets"
CLOB = "https://clob.polymarket.com/prices-history"

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)


def _get(url: str, params: dict[str, Any], retries: int = 5) -> Any:
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(2 ** i)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if i == retries - 1:
                raise
            time.sleep(2 ** i)
    raise RuntimeError("unreachable")


def list_resolved_markets(target_n: int = 2000, min_volume: float = 5000.0) -> list[dict]:
    """Page through closed markets sorted by volume desc."""
    out: list[dict] = []
    offset = 0
    page_size = 100
    while len(out) < target_n:
        batch = _get(
            GAMMA,
            {
                "closed": "true",
                "limit": page_size,
                "offset": offset,
                "order": "volumeNum",
                "ascending": "false",
            },
        )
        if not batch:
            break
        for m in batch:
            vol = m.get("volumeNum") or 0
            if vol < min_volume:
                # since we are sorted desc by volume, we can stop
                return out
            if m.get("umaResolutionStatus") != "resolved":
                continue
            outcomes = _safe_json(m.get("outcomes"))
            prices = _safe_json(m.get("outcomePrices"))
            tokens = _safe_json(m.get("clobTokenIds"))
            if not outcomes or not prices or not tokens:
                continue
            if len(outcomes) != 2 or len(prices) != 2 or len(tokens) != 2:
                continue
            # Require resolution to one side: one price 1, other 0
            try:
                p0, p1 = float(prices[0]), float(prices[1])
            except (TypeError, ValueError):
                continue
            if not ((p0 == 1.0 and p1 == 0.0) or (p0 == 0.0 and p1 == 1.0)):
                continue
            out.append(m)
        offset += page_size
        # politeness
        time.sleep(0.05)
    return out


def _safe_json(s: Any) -> Any:
    if isinstance(s, list):
        return s
    if isinstance(s, str):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return None
    return None


def fetch_price_history(token_id: str, start_ts: int) -> list[dict]:
    """Return raw history points for a CLOB token."""
    data = _get(CLOB, {"market": token_id, "fidelity": 60, "startTs": start_ts})
    return data.get("history", []) or []


def price_at_or_before(history: list[dict], ts: int) -> float | None:
    """Last price at or before timestamp ts. None if no data."""
    last = None
    for h in history:
        if h["t"] <= ts:
            last = h["p"]
        else:
            break
    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-n", type=int, default=2000,
                    help="max number of markets to scan from Gamma")
    ap.add_argument("--min-volume", type=float, default=5000.0,
                    help="min lifetime USD volume to include")
    ap.add_argument("--max-out", type=int, default=1500,
                    help="cap on number of markets we actually fetch CLOB data for")
    ap.add_argument("--output", default=str(DATA / "markets.csv"))
    args = ap.parse_args()

    print(f"[1/2] listing resolved markets (vol >= ${args.min_volume:,.0f})...")
    markets = list_resolved_markets(args.target_n, args.min_volume)
    print(f"   got {len(markets)} markets meeting filters")
    if args.max_out and len(markets) > args.max_out:
        markets = markets[: args.max_out]
        print(f"   capped to {len(markets)}")

    rows = []
    print("[2/2] fetching price histories...")
    for i, m in enumerate(markets):
        if (i + 1) % 25 == 0:
            print(f"   {i+1}/{len(markets)}")
        outcomes = _safe_json(m["outcomes"])
        prices = _safe_json(m["outcomePrices"])
        tokens = _safe_json(m["clobTokenIds"])
        # YES is conventionally index 0 for Yes/No markets, but for sports
        # markets it's just outcome[0]. We always treat outcome[0] as the
        # "positive" class, and the binary label is whether outcome[0] won.
        positive_outcome = outcomes[0]
        positive_token = tokens[0]
        label = 1 if float(prices[0]) == 1.0 else 0

        try:
            end_iso = m.get("endDate") or m.get("endDateIso")
            if not end_iso:
                continue
            end_ts = int(pd.Timestamp(end_iso).timestamp())
        except Exception:
            continue

        # fetch from 60 days before end
        start_ts = end_ts - 60 * 86400
        try:
            hist = fetch_price_history(positive_token, start_ts)
        except Exception as e:
            print(f"   skip {m['slug']}: {e}")
            continue
        if not hist:
            continue

        p_1d = price_at_or_before(hist, end_ts - 86400)
        p_7d = price_at_or_before(hist, end_ts - 7 * 86400)
        p_30d = price_at_or_before(hist, end_ts - 30 * 86400)
        # latest known price before close
        p_close = hist[-1]["p"] if hist else None

        rows.append(
            dict(
                market_id=m["id"],
                slug=m["slug"],
                question=m["question"],
                positive_outcome=positive_outcome,
                category=m.get("category"),
                volume_usd=m.get("volumeNum"),
                end_iso=end_iso,
                end_ts=end_ts,
                label=label,
                p_1d=p_1d,
                p_7d=p_7d,
                p_30d=p_30d,
                p_close=p_close,
                n_history_points=len(hist),
            )
        )
        # politeness toward CLOB
        time.sleep(0.02)

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"wrote {len(df)} rows -> {args.output}")


if __name__ == "__main__":
    main()
