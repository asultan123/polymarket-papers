"""Fetch raw data for the Polymarket vs FiveThirtyEight 2024 election retrospective.

Outputs into ``data/`` (alongside this script):
  - ``fte_averages.csv``          : FiveThirtyEight 2024 general-election polling averages
                                    (Wayback snapshot taken on 2024-11-05, election day).
  - ``polymarket_markets.json``   : Metadata for each state-level "Will a Republican win X?"
                                    market plus the national Trump market on Polymarket.
  - ``polymarket_prices.csv``     : Long-format daily mid-price history for each market,
                                    pulled from the Polymarket CLOB API.
  - ``election_results.csv``      : Hand-coded actual 2024 state outcomes (Trump wins = 1).

The Wayback snapshot is needed because the upstream URL
``projects.fivethirtyeight.com/polls/data/presidential_general_averages.csv``
went offline when Disney shut down 538 in 2025. The September 2024 file that
ships in the ``fivethirtyeight/data`` GitHub repo stops 7 weeks before
election day and is explicitly marked "uncorrected"; we use the archived
production file because it covers the full cycle.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

FTE_WAYBACK_URL = (
    "https://web.archive.org/web/20241105000000if_/"
    "https://projects.fivethirtyeight.com/polls/data/"
    "presidential_general_averages.csv"
)

# Polymarket event slugs for the state-level "Who will win X?" markets and the
# top-line presidential-winner market. Each event hosts three child markets
# ("Will a Democrat win X?", "Will a Republican win X?", "Will a candidate
# from another party win X?"); we keep the Republican one because it is the
# direct analogue of Trump's win probability.
STATE_EVENTS = {
    "Arizona": "arizona-presidential-election-winner",
    "Georgia": "georgia-presidential-election-winner",
    "Michigan": "michigan-presidential-election-winner",
    "Nevada": "nevada-presidential-election-winner",
    "North Carolina": "north-carolina-presidential-election-winner",
    "Pennsylvania": "pennsylvania-presidential-election-winner",
    "Wisconsin": "wisconsin-presidential-election-winner",
    "Florida": "florida-presidential-election-winner",
    "Texas": "texas-presidential-election-winner",
    "Ohio": "ohio-presidential-election-winner",
    "Minnesota": "minnesota-presidential-election-winner",
    "New Hampshire": "new-hampshire-presidential-election-winner",
    "Virginia": "virginia-presidential-election-winner",
}

NATIONAL_EVENT_SLUG = "presidential-election-winner-2024"

# Official 2024 results: 1 if Trump (Republican) won the state, 0 otherwise.
# Cross-checked against AP, Cook Political Report and the FEC certified totals.
ACTUAL_RESULTS = {
    "Arizona": 1, "Georgia": 1, "Michigan": 1, "Nevada": 1,
    "North Carolina": 1, "Pennsylvania": 1, "Wisconsin": 1,
    "Florida": 1, "Texas": 1, "Ohio": 1,
    "Minnesota": 0, "New Hampshire": 0, "Virginia": 0,
    "National": 1,  # Trump won the electoral college
}


def fetch_fte_averages() -> None:
    out = DATA_DIR / "fte_averages.csv"
    print(f"[fte] downloading {FTE_WAYBACK_URL}")
    r = requests.get(FTE_WAYBACK_URL, timeout=120)
    r.raise_for_status()
    out.write_bytes(r.content)
    print(f"[fte] wrote {out} ({len(r.content):,} bytes)")


def fetch_event(slug: str) -> dict:
    url = f"https://gamma-api.polymarket.com/events?slug={slug}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    body = r.json()
    if not body:
        raise RuntimeError(f"no event for slug={slug}")
    return body[0]


def pick_republican_market(event: dict) -> dict:
    """Return the binary 'Will a Republican win X?' child market."""
    for m in event["markets"]:
        slug = m.get("slug", "")
        if "will-a-republican-win" in slug:
            return m
    raise RuntimeError(f"no Republican market for event {event['slug']}")


def pick_trump_national_market(event: dict) -> dict:
    """Return the 'Will Donald Trump win...' market on the top-level event."""
    for m in event["markets"]:
        if "donald-trump-win" in m.get("slug", ""):
            return m
    raise RuntimeError("no Trump market on national event")


def fetch_price_history(token_id: str) -> list[dict]:
    """Fetch the full daily-fidelity price history for a CLOB token."""
    url = (
        "https://clob.polymarket.com/prices-history"
        f"?market={token_id}&interval=all&fidelity=1440"
    )
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json().get("history", [])


def fetch_polymarket() -> None:
    """Resolve markets and fetch daily price history for each."""
    markets_out = {}
    prices_rows = []

    # Top-level: national Trump market.
    nat_event = fetch_event(NATIONAL_EVENT_SLUG)
    nat_market = pick_trump_national_market(nat_event)
    nat_yes_token = json.loads(nat_market["clobTokenIds"])[0]
    markets_out["National"] = {
        "event_slug": NATIONAL_EVENT_SLUG,
        "market_slug": nat_market["slug"],
        "question": nat_market["question"],
        "yes_token_id": nat_yes_token,
        "volume": float(nat_market.get("volumeNum", 0.0)),
    }
    hist = fetch_price_history(nat_yes_token)
    for pt in hist:
        prices_rows.append({"state": "National", "t": pt["t"], "p": pt["p"]})
    print(f"[poly] National: {len(hist)} daily points")

    for state, slug in STATE_EVENTS.items():
        try:
            ev = fetch_event(slug)
        except Exception as e:
            print(f"[poly] {state}: SKIP ({e})", file=sys.stderr)
            continue
        try:
            mkt = pick_republican_market(ev)
        except Exception as e:
            print(f"[poly] {state}: SKIP ({e})", file=sys.stderr)
            continue

        yes_token = json.loads(mkt["clobTokenIds"])[0]
        markets_out[state] = {
            "event_slug": slug,
            "market_slug": mkt["slug"],
            "question": mkt["question"],
            "yes_token_id": yes_token,
            "volume": float(mkt.get("volumeNum", 0.0)),
        }
        hist = fetch_price_history(yes_token)
        for pt in hist:
            prices_rows.append({"state": state, "t": pt["t"], "p": pt["p"]})
        print(f"[poly] {state}: {len(hist)} daily points (vol=${mkt.get('volumeNum',0):,.0f})")
        time.sleep(0.2)  # be nice to the API

    (DATA_DIR / "polymarket_markets.json").write_text(
        json.dumps(markets_out, indent=2)
    )
    print(f"[poly] wrote polymarket_markets.json ({len(markets_out)} markets)")

    # Long-format CSV of daily prices.
    import csv
    with (DATA_DIR / "polymarket_prices.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["state", "t", "p"])
        w.writeheader()
        w.writerows(prices_rows)
    print(f"[poly] wrote polymarket_prices.csv ({len(prices_rows):,} rows)")


def write_actual_results() -> None:
    import csv
    out = DATA_DIR / "election_results.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["state", "trump_won"])
        for state, won in ACTUAL_RESULTS.items():
            w.writerow([state, won])
    print(f"[res] wrote {out}")


def main() -> None:
    fetch_fte_averages()
    fetch_polymarket()
    write_actual_results()
    print("[done] all data in", DATA_DIR)


if __name__ == "__main__":
    main()
