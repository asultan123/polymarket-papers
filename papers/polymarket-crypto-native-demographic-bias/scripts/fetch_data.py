"""Fetch resolved binary markets from Polymarket and Manifold.

Outputs three CSVs into ./data/ (created if missing):
    polymarket_markets.csv  - one row per resolved binary Polymarket market
    polymarket_prices.csv   - per-market price snapshots at 1d, 7d, 30d before close
    manifold_markets.csv    - resolved binary Manifold markets

Usage: python fetch_data.py
"""
from __future__ import annotations
import json
import time
import csv
import os
from pathlib import Path

import requests

GAMMA = "https://gamma-api.polymarket.com/markets"
CLOB_HIST = "https://clob.polymarket.com/prices-history"
MANIFOLD = "https://api.manifold.markets/v0/markets"

OUT = Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "academic-research-polymarket-bias/0.1"})

# ---------------------------------------------------------------------------
# Polymarket: discover resolved binary markets via pagination on Gamma API
# ---------------------------------------------------------------------------

def fetch_polymarket_markets(max_markets: int = 4000,
                              min_volume: float = 500.0) -> list[dict]:
    """Pull closed Polymarket markets ordered by volume (descending).

    We focus on markets with non-trivial volume because price history on
    illiquid markets is dominated by the seed-AMM price and adds noise to
    any calibration analysis.
    """
    rows: list[dict] = []
    seen_ids: set[str] = set()
    offset = 0
    page = 500
    while len(rows) < max_markets:
        params = {
            "limit": page,
            "closed": "true",
            "order": "volumeNum",
            "ascending": "false",
            "offset": offset,
        }
        try:
            r = SESSION.get(GAMMA, params=params, timeout=60)
            r.raise_for_status()
            batch = r.json()
        except requests.RequestException as e:
            print(f"[gamma] error at offset {offset}: {e}; treating as end of pagination")
            break
        if not batch:
            print(f"[gamma] empty page at offset {offset}; stopping")
            break
        # Defensive: stop if response is shorter than requested AND offset moved
        progress = 0
        for m in batch:
            mid = m.get("id")
            if mid is None or mid in seen_ids:
                continue
            seen_ids.add(mid)
            vol = m.get("volumeNum") or 0
            if vol < min_volume:
                # Once volume falls below threshold, stop (we sorted by volume desc)
                print(f"[gamma] volume {vol:.0f} < threshold; stopping at {len(rows)} markets")
                return rows
            # binary YES/NO only
            try:
                outcomes = json.loads(m.get("outcomes") or "[]")
                prices = json.loads(m.get("outcomePrices") or "[]")
            except Exception:
                continue
            if len(outcomes) != 2 or len(prices) != 2:
                continue
            if sorted(map(str, outcomes)) != ["No", "Yes"]:
                continue
            try:
                p0, p1 = float(prices[0]), float(prices[1])
            except Exception:
                continue
            # Resolved unambiguously: one side = 1, other = 0
            yes_idx = outcomes.index("Yes")
            no_idx = outcomes.index("No")
            yes_payout = float(prices[yes_idx])
            no_payout = float(prices[no_idx])
            if not (
                (abs(yes_payout - 1) < 1e-6 and abs(no_payout) < 1e-6)
                or (abs(yes_payout) < 1e-6 and abs(no_payout - 1) < 1e-6)
            ):
                # Tied / canceled / partial - skip
                continue
            outcome_yes = int(abs(yes_payout - 1) < 1e-6)
            try:
                clob_tokens = json.loads(m.get("clobTokenIds") or "[]")
            except Exception:
                clob_tokens = []
            if len(clob_tokens) != 2:
                continue
            yes_token = clob_tokens[yes_idx]
            rows.append(
                {
                    "id": m.get("id"),
                    "condition_id": m.get("conditionId"),
                    "slug": m.get("slug"),
                    "question": m.get("question"),
                    "category": m.get("category"),
                    "yes_token": yes_token,
                    "outcome_yes": outcome_yes,
                    "volume_num": vol,
                    "liquidity_num": m.get("liquidityNum") or 0,
                    "start_date": m.get("startDate"),
                    "end_date": m.get("endDate"),
                    "closed_time": m.get("closedTime"),
                    "created_at": m.get("createdAt"),
                }
            )
            progress += 1
            if len(rows) >= max_markets:
                break
        print(f"[gamma] offset={offset} batch={len(batch)} new={progress} total={len(rows)}")
        offset += page
        if progress == 0:
            # Pagination stalled - bail
            break
        time.sleep(0.3)
    return rows


# ---------------------------------------------------------------------------
# Polymarket: fetch price history at multiple horizons
# ---------------------------------------------------------------------------

def fetch_price_history(token_id: str, fidelity: int = 1440) -> list[dict]:
    """fidelity in minutes. 1440 = daily."""
    params = {"market": token_id, "interval": "max", "fidelity": fidelity}
    try:
        r = SESSION.get(CLOB_HIST, params=params, timeout=30)
        r.raise_for_status()
    except requests.RequestException:
        return []
    js = r.json()
    return js.get("history", [])


def extract_horizon_prices(history: list[dict], end_ts: int) -> dict:
    """Given a list of {t, p} points and the close timestamp,
    extract the most recent price at <= (end - 1d, 7d, 30d)."""
    horizons = {"p_1d": 86400, "p_7d": 7 * 86400, "p_30d": 30 * 86400}
    out: dict = {}
    if not history:
        return {k: None for k in horizons}
    sorted_hist = sorted(history, key=lambda x: x["t"])
    for label, secs in horizons.items():
        target = end_ts - secs
        # Find last point with t <= target
        best = None
        for pt in sorted_hist:
            if pt["t"] <= target:
                best = pt
            else:
                break
        out[label] = best["p"] if best else None
    # Also record final pre-resolution price (last available point)
    out["p_final"] = sorted_hist[-1]["p"] if sorted_hist else None
    out["history_len"] = len(sorted_hist)
    out["first_ts"] = sorted_hist[0]["t"] if sorted_hist else None
    out["last_ts"] = sorted_hist[-1]["t"] if sorted_hist else None
    return out


def isoformat_to_ts(iso: str | None) -> int | None:
    if not iso:
        return None
    from datetime import datetime, timezone
    s = iso.replace("Z", "+00:00")
    # Some closedTime values are formatted like "2024-11-06 14:25:31+00"
    try:
        return int(datetime.fromisoformat(s).timestamp())
    except ValueError:
        # Try the alt format
        try:
            return int(datetime.strptime(s, "%Y-%m-%d %H:%M:%S%z").timestamp())
        except Exception:
            return None


def enrich_with_prices(markets: list[dict], pause: float = 0.05) -> list[dict]:
    out = []
    n = len(markets)
    for i, m in enumerate(markets):
        # Anchor at earliest of (scheduled end date, actual close time): this is the
        # earliest moment the resolution could plausibly have been known. Markets
        # often keep trading after the resolving event so closed_time alone overstates
        # how long uncertainty remained.
        ed = isoformat_to_ts(m.get("end_date"))
        ct = isoformat_to_ts(m.get("closed_time"))
        if ed and ct:
            end_ts = min(ed, ct)
        else:
            end_ts = ed or ct
        if end_ts is None:
            continue
        hist = fetch_price_history(m["yes_token"])
        prices = extract_horizon_prices(hist, end_ts)
        rec = {**m, **prices, "end_ts": end_ts}
        out.append(rec)
        if (i + 1) % 50 == 0:
            print(f"[clob] enriched {i+1}/{n}")
        time.sleep(pause)
    return out


# ---------------------------------------------------------------------------
# Manifold: pull resolved binary markets
# ---------------------------------------------------------------------------

def fetch_manifold_resolved(max_markets: int = 5000) -> list[dict]:
    rows: list[dict] = []
    before = None
    while len(rows) < max_markets:
        params = {"limit": 1000}
        if before:
            params["before"] = before
        r = SESSION.get(MANIFOLD, params=params, timeout=60)
        if r.status_code != 200:
            print(f"[manifold] HTTP {r.status_code}; stopping")
            break
        batch = r.json()
        if not batch:
            break
        for m in batch:
            if m.get("outcomeType") != "BINARY":
                continue
            if not m.get("isResolved"):
                continue
            res = m.get("resolution")
            if res not in ("YES", "NO"):
                continue
            rows.append({
                "id": m["id"],
                "slug": m.get("slug"),
                "question": m.get("question"),
                "creator": m.get("creatorUsername"),
                "created_time": m.get("createdTime"),
                "close_time": m.get("closeTime"),
                "is_resolved": m.get("isResolved"),
                "resolution": res,
                "outcome_yes": 1 if res == "YES" else 0,
                "probability": m.get("probability"),
                "volume": m.get("volume"),
                "unique_bettors": m.get("uniqueBettorCount"),
                "total_liquidity": m.get("totalLiquidity"),
            })
            if len(rows) >= max_markets:
                break
        before = batch[-1]["id"]
        print(f"[manifold] total resolved binary so far={len(rows)}")
        time.sleep(0.2)
    return rows


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        print(f"[save] no rows for {path.name}")
        return
    keys: list[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[save] wrote {len(rows):,} rows -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Polymarket pull ===")
    pm = fetch_polymarket_markets(max_markets=int(os.environ.get("PM_MAX", "2500")),
                                  min_volume=float(os.environ.get("PM_MINVOL", "1000")))
    save_csv(pm, OUT / "polymarket_markets_raw.csv")
    print(f"\nEnriching {len(pm)} markets with CLOB price history...")
    pm_priced = enrich_with_prices(pm)
    save_csv(pm_priced, OUT / "polymarket_prices.csv")

    print("\n=== Manifold pull ===")
    mf = fetch_manifold_resolved(max_markets=int(os.environ.get("MF_MAX", "8000")))
    save_csv(mf, OUT / "manifold_markets.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
