"""Fetch Polymarket market metadata, executed trades, and coarse price history.

Two facts about the public endpoints determined our strategy (May 2026):

* `clob.polymarket.com/prices-history` returns an empty array for any resolved
  market when the requested granularity is finer than 12 hours.  We therefore
  use 12h fidelity, which is sufficient for calibration / Brier-score work but
  too coarse for cascade detection.
* `data-api.polymarket.com/trades` accepts limit<=500 and refuses offsets above
  ~3,500.  This caps us at the ~3,500 most-recent taker trades per market.  For
  high-volume markets these are concentrated near resolution -- the most
  cascade-prone region -- so the cap is not a binding loss for our purpose.

Outputs (under data/):
  markets.json         all binary markets with vol > $100k (metadata only)
  markets_sample.json  ~40 highest-volume markets we analyse in depth
  trades_<id>.csv      taker trades, newest-first (<=3500 rows per market)
  prices12h_<id>.csv   12h-fidelity midprice history for YES outcome
  summary.csv          one row per analysed market
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

GAMMA = "https://gamma-api.polymarket.com/markets"
CLOB_PRICES = "https://clob.polymarket.com/prices-history"
DATA_TRADES = "https://data-api.polymarket.com/trades"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "polymarket-cascade-research/0.1"})


def get(url: str, params: dict | None = None, retries: int = 3, timeout: int = 30) -> Any:
    last: Exception | None = None
    for i in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 400:
                return None
        except requests.RequestException as e:
            last = e
        time.sleep(0.5 * (i + 1))
    if last is not None:
        raise last
    return None


def fetch_top_closed_markets(limit_pages: int = 8, page_size: int = 100) -> list[dict]:
    out: list[dict] = []
    for offset in range(0, limit_pages * page_size, page_size):
        page = get(
            GAMMA,
            params={
                "limit": page_size,
                "offset": offset,
                "closed": "true",
                "order": "volumeNum",
                "ascending": "false",
            },
        )
        if not page:
            break
        out.extend(page)
    return out


def fetch_prices_12h(token_id: str) -> list[dict]:
    """Full-history midprice at 12h fidelity. The fine-grained endpoint is
    broken for resolved markets, so we use the coarse one."""
    out = get(
        CLOB_PRICES,
        params={"market": token_id, "interval": "max", "fidelity": 60 * 12},
    )
    return (out or {}).get("history", [])


def fetch_trades(market_condition_id: str, limit: int, offset: int) -> list[dict]:
    res = get(
        DATA_TRADES,
        params={
            "market": market_condition_id,
            "limit": limit,
            "offset": offset,
            "takerOnly": "true",
        },
    )
    return res or []


def paginate_trades(market_condition_id: str) -> list[dict]:
    """Walk to the offset cap (~3500) for the given market.  Newest-first."""
    out: list[dict] = []
    seen: set[str] = set()
    for offset in range(0, 4000, 500):
        batch = fetch_trades(market_condition_id, limit=500, offset=offset)
        if not batch or len(batch) < 5:
            break
        new = [t for t in batch if t.get("transactionHash") not in seen]
        for t in new:
            seen.add(t.get("transactionHash") or "")
        out.extend(new)
        if len(batch) < 500:
            break
        time.sleep(0.12)
    return out


def main() -> None:
    print("Fetching top closed Polymarket markets by volume ...")
    markets = fetch_top_closed_markets(limit_pages=6)
    keep = []
    for m in markets:
        try:
            outcomes = json.loads(m.get("outcomes") or "[]")
            tokens = json.loads(m.get("clobTokenIds") or "[]")
        except json.JSONDecodeError:
            continue
        if len(outcomes) == 2 and len(tokens) == 2 and (m.get("volumeNum") or 0) > 100_000:
            keep.append(
                {
                    "id": m["id"],
                    "question": m.get("question", ""),
                    "slug": m.get("slug", ""),
                    "condition_id": m.get("conditionId"),
                    "outcomes": outcomes,
                    "outcome_prices": m.get("outcomePrices"),
                    "token_ids": tokens,
                    "volume_usd": m.get("volumeNum") or 0,
                    "start_date": m.get("startDate"),
                    "end_date": m.get("endDate"),
                    "closed_time": m.get("closedTime"),
                    "category": m.get("category"),
                }
            )
    keep.sort(key=lambda r: -(r["volume_usd"] or 0))
    print(f"Kept {len(keep)} binary markets with volume > $100k")
    (DATA / "markets.json").write_text(json.dumps(keep, indent=2))

    # pick the largest 40 by volume for depth analysis
    sample = keep[:40]
    (DATA / "markets_sample.json").write_text(json.dumps(sample, indent=2))

    summary_rows = []
    for i, m in enumerate(sample, 1):
        print(f"[{i}/{len(sample)}] {m['question'][:80]}  vol=${m['volume_usd']:,.0f}")
        # 12-hour price history for YES outcome
        try:
            hist = fetch_prices_12h(m["token_ids"][0])
        except Exception as e:  # noqa: BLE001
            print(f"  prices fetch failed: {e}")
            hist = []
        n_price_points = len(hist)
        if hist:
            out = DATA / f"prices12h_{m['id']}.csv"
            with out.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["timestamp", "price"])
                for row in hist:
                    w.writerow([row["t"], row["p"]])

        # trades
        try:
            trades = paginate_trades(m["condition_id"])
        except Exception as e:  # noqa: BLE001
            print(f"  trades fetch failed: {e}")
            trades = []
        if trades:
            out = DATA / f"trades_{m['id']}.csv"
            keys = [
                "timestamp",
                "price",
                "size",
                "side",
                "outcome",
                "outcomeIndex",
                "transactionHash",
                "proxyWallet",
            ]
            # de-dup and sort ascending by timestamp
            trades_sorted = sorted(trades, key=lambda t: int(t.get("timestamp", 0)))
            with out.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for t in trades_sorted:
                    w.writerow({k: t.get(k) for k in keys})
            t_min = min(int(t["timestamp"]) for t in trades_sorted)
            t_max = max(int(t["timestamp"]) for t in trades_sorted)
            duration_h = (t_max - t_min) / 3600.0
        else:
            t_min = t_max = 0
            duration_h = 0.0

        summary_rows.append(
            {
                "id": m["id"],
                "question": m["question"][:120],
                "category": m["category"],
                "volume_usd": round(m["volume_usd"], 2),
                "n_price_points_12h": n_price_points,
                "n_trades": len(trades),
                "trades_t_min": t_min,
                "trades_t_max": t_max,
                "trades_span_hours": round(duration_h, 2),
                "end_date": m["end_date"],
            }
        )
        sys.stdout.flush()
        time.sleep(0.2)

    if summary_rows:
        with (DATA / "summary.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        print(f"\nWrote {DATA / 'summary.csv'}")
    print("Done.")


if __name__ == "__main__":
    main()
