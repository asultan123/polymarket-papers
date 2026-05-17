"""Fetch raw data for Polymarket vs economists CPI forecast comparison.

Three sources:
  1. Polymarket Gamma API + CLOB API: monthly/annual US inflation markets
     with per-bracket implied probabilities sampled hourly.
  2. Cleveland Fed Inflation Nowcasting: daily MoM and YoY CPI nowcasts
     plus the eventual realized BLS value.
  3. (Optional) Yahoo Finance: 5-year breakeven inflation (TIP/IEF spread)
     as a market-implied benchmark we discuss qualitatively.

Outputs three JSON files to ../data/:
    polymarket_events.json   (list of events with per-market price histories)
    cleveland_fed_year.json  (raw YoY chart series)
    cleveland_fed_month.json (raw MoM chart series)

Run once; analysis.py consumes the cached files.
"""

import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import requests

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
CLEVELAND_BASE = (
    "https://www.clevelandfed.org/-/media/files/webcharts/inflationnowcasting"
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)


def get(url, params=None, retries=3, sleep=1.0):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30,
                             headers={"User-Agent": "polymarket-research/0.1"})
            r.raise_for_status()
            return r
        except Exception as e:
            if attempt + 1 == retries:
                raise
            time.sleep(sleep * (attempt + 1))


def fetch_cleveland_fed():
    """Download YoY and MoM nowcast time-series. Cached as raw JSON."""
    for chart in ("year", "month"):
        out = os.path.join(DATA_DIR, f"cleveland_fed_{chart}.json")
        if os.path.exists(out) and os.path.getsize(out) > 1000:
            print(f"  cached: cleveland_fed_{chart}.json")
            continue
        url = f"{CLEVELAND_BASE}/nowcast_{chart}.json"
        # The Cleveland Fed CDN occasionally appends ?sc_lang=en
        r = get(url, params={"sc_lang": "en"})
        with open(out, "w") as f:
            f.write(r.text)
        print(f"  downloaded: cleveland_fed_{chart}.json ({len(r.text)} bytes)")


def list_us_inflation_events():
    """Return list of event dicts for monthly + annual US CPI markets.

    Hits both `tag_slug=cpi` and `tag_slug=inflation` to maximise recall,
    dedupes on event id, then filters for US monthly/annual CPI prints.
    """
    keepers = []
    seen = set()
    for tag in ("cpi", "inflation"):
        r = get(f"{GAMMA}/events", params={
            "limit": 500, "tag_slug": tag, "closed": "true",
        })
        for e in r.json():
            if e["id"] in seen:
                continue
            seen.add(e["id"])
            title = e["title"].lower()
            # exclude non-US country prints and side markets
            if any(k in title for k in [
                "argentina", "canada", "eurozone", "china", "u.k.",
                "brazil", "egg", "powell", "how high", "treasury",
                "gold", "delays"
            ]):
                continue
            if any(k in title for k in [
                "inflation - monthly", "inflation - annual",
                "inflation us - monthly", "inflation us - annual",
                "us inflation"
            ]):
                keepers.append(e)
    keepers.sort(key=lambda e: e.get("endDate", ""))
    return keepers


def parse_bracket(question, kind):
    """Map a market question to (lower, upper) numeric bracket in percent.

    kind: 'monthly' (MoM%) or 'annual' (YoY%).

    Returns (lo, hi, mid) or None if the bracket cannot be parsed.

    Examples:
        "Will annual inflation increase by 2.3% or less in April?"
            -> (-inf, 2.3, 2.25)
        "Will annual inflation increase by 2.4% in April?"
            -> (2.35, 2.45, 2.4)
        "Will annual inflation increase by 2.7% or more in April?"
            -> (2.7, +inf, 2.75)
        "US inflation >0.2% from Feb to March 2024?"   (legacy binary form)
            -> (0.2, +inf, None)  # binary - handled separately
    """
    q = question.lower()
    # legacy binary form
    m = re.search(r"inflation\s*>\s*(-?\d+\.?\d*)", q)
    if m:
        thresh = float(m.group(1))
        return ("binary", thresh)

    # "or less" lower-tail bracket
    m = re.search(r"by (\-?\d+\.?\d*)\s*%?\s*or less", q)
    if m:
        v = float(m.group(1))
        return ("le", v)

    # "or more" upper-tail bracket
    m = re.search(r"by (\-?\d+\.?\d*)\s*%?\s*or more", q)
    if m:
        v = float(m.group(1))
        return ("ge", v)

    # exact "by X.X%"
    m = re.search(r"by (\-?\d+\.?\d*)\s*%", q)
    if m:
        v = float(m.group(1))
        return ("eq", v)
    return None


def fetch_event_with_prices(event):
    """Augment an event dict with per-market parsed brackets + price history.

    For each market we request the full price history at hourly fidelity
    over the period [event.startDate - 1d, event.endDate].  Polymarket
    returns YES probability for the first outcome token.
    """
    start = event.get("startDate") or event.get("creationDate")
    end = event.get("endDate")
    if not start or not end:
        return None
    try:
        # The CLOB caps (startTs,endTs) windows at ~14-15 days when
        # fidelity=60 minutes, so we anchor the window to the last
        # 14 days before resolution -- which covers the only segment
        # that matters for short-horizon CPI predictive accuracy.
        end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
        end_ts = int(end_dt.timestamp()) + 3600
        start_ts = end_ts - 14 * 86400
    except Exception:
        return None

    kind = ("monthly" if "monthly" in event["title"].lower()
            else "annual")
    out_markets = []
    for m in event.get("markets", []):
        bracket = parse_bracket(m["question"], kind)
        if bracket is None:
            continue
        # outcomePrices and outcomes come back as JSON-encoded strings
        outcome_prices = m.get("outcomePrices")
        if isinstance(outcome_prices, str):
            try:
                outcome_prices = json.loads(outcome_prices)
            except Exception:
                outcome_prices = None
        token_ids = m.get("clobTokenIds")
        if isinstance(token_ids, str):
            try:
                token_ids = json.loads(token_ids)
            except Exception:
                token_ids = []
        if not token_ids:
            continue
        yes_token = token_ids[0]

        # price history.  The CLOB requires the (startTs,endTs) span be
        # at most ~30 days, which is why we anchored the window above.
        # interval=max returns 0 rows for older markets, so the explicit
        # timestamps are the only reliable path.
        try:
            r = get(f"{CLOB}/prices-history", params={
                "market": yes_token,
                "startTs": start_ts,
                "endTs": end_ts,
                "fidelity": "60",
            })
            history = r.json().get("history", [])
        except Exception as e:
            print(f"    !! price fetch failed for {m['id']}: {e}", file=sys.stderr)
            history = []

        out_markets.append({
            "id": m["id"],
            "question": m["question"],
            "bracket": bracket,
            "outcomePrices_final": outcome_prices,
            "yes_token": yes_token,
            "history": history,  # list of {"t": unix, "p": price}
        })
    if not out_markets:
        return None

    return {
        "id": event["id"],
        "title": event["title"],
        "startDate": start,
        "endDate": end,
        "volume": event.get("volume", 0),
        "kind": kind,
        "markets": out_markets,
    }


def main():
    print("Fetching Cleveland Fed nowcasting data...")
    fetch_cleveland_fed()

    print("\nListing US inflation events from Polymarket...")
    events = list_us_inflation_events()
    print(f"  found {len(events)} candidate events")

    print("\nFetching per-bracket price histories...")
    enriched = []
    for i, e in enumerate(events):
        print(f"  [{i+1}/{len(events)}] {e['title']} (vol=${e.get('volume', 0):,.0f})")
        out = fetch_event_with_prices(e)
        if out is not None:
            enriched.append(out)

    out_path = os.path.join(DATA_DIR, "polymarket_events.json")
    with open(out_path, "w") as f:
        json.dump(enriched, f)
    sz = os.path.getsize(out_path)
    print(f"\nWrote {out_path} ({sz/1024:.1f} KB, {len(enriched)} events)")


if __name__ == "__main__":
    main()
