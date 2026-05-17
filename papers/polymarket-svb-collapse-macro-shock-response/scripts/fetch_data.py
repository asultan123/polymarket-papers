"""Fetch Polymarket Gamma + CLOB data and Yahoo Finance baselines around the
March 2023 SVB collapse.

Outputs (all written next to this script, under ../data/):
  - candidates.json       Gamma metadata for each market we analyse
  - prices/<id>.csv       timestamped midprice history for each market's YES token
  - yahoo/<ticker>.csv    daily OHLCV for KRE, SPY, TLT, ^TNX, FRC, SIVBQ, SBNY

This is the only script that hits the network. analysis.py and figure_*.py
read the cached CSVs only, so the analysis is fully replayable offline once
fetch_data.py has been run once.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
PRICES = DATA / "prices"
YAHOO = DATA / "yahoo"
for d in (DATA, PRICES, YAHOO):
    d.mkdir(parents=True, exist_ok=True)

GAMMA = "https://gamma-api.polymarket.com/markets"
CLOB_HIST = "https://clob.polymarket.com/prices-history"

# Markets relevant to SVB / contagion / Fed pivot. Questions are matched exactly
# against the Gamma `question` field to avoid pulling unrelated markets.
TARGET_QUESTIONS = [
    # SVB-direct
    "Will SVB fail?",
    "Will SVB be acquired by Monday night?",
    "Will another US bank fail in March?",
    "Will uninsured SVB depositors get all their money back by June 30?",
    "Will uninsured SVB depositors get all their money back by EOY?",
    # Contagion candidates
    "Will Silvergate announce it is filing for bankruptcy by March 31, 2023?",
    "Will a third US bank fail by March 17?",
    "Will Credit Suisse fail by March 31?",
    "Will First Republic Bank fail by March 17?",
    "Will First Republic Bank fail by March 31?",
    "Will Bank of America fail by March 17?",
    # Fed-pivot markets that straddle the shock (key for diff-in-diff)
    "Will the Fed cut rates in 2023?",
    "Will the Fed cut rates in March?",
    "Will the Fed increase interest rates by 25 bps after its March meeting?",
    "Will the Fed increase interest rates by 50 bps after its March meeting?",
    "Will the Fed increase interest rates by 0 bps after its March meeting?",
    "Will the Fed decrease interest rates by 25 bps after its March meeting?",
    "Will the Fed increase interest rates by 25 bps after its May meeting?",
    "Will the Fed increase interest rates by 0 bps after its May meeting?",
    "Will the Fed decrease interest rates by 25 bps after its May meeting?",
]

# CLOB enforces a 14-day cap per request regardless of fidelity.
MAX_WINDOW_DAYS = 14


def find_candidates() -> list[dict]:
    """Page through Gamma /markets and return only the targets."""
    all_mkts: list[dict] = []
    offset = 0
    while True:
        r = requests.get(
            GAMMA,
            params={
                "limit": 100,
                "offset": offset,
                "end_date_min": "2023-03-08T00:00:00Z",
                "end_date_max": "2023-12-31T23:59:59Z",
                "closed": "true",
            },
            timeout=60,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        all_mkts.extend(batch)
        offset += 100
        if offset > 30000:
            break

    by_q = {m["question"]: m for m in all_mkts if m.get("question")}
    out = []
    for q in TARGET_QUESTIONS:
        if q in by_q:
            out.append(by_q[q])
        else:
            print(f"  WARN: {q!r} not found in Gamma")
    return out


def fetch_history(token_id: str, start: datetime, end: datetime,
                  fidelity: int = 60) -> pd.DataFrame:
    """Pull (t, p) pairs in 14-day chunks. Returns deduped DataFrame."""
    out: list[dict] = []
    cur = start
    chunk = timedelta(days=MAX_WINDOW_DAYS - 1)  # 13 days to be safe
    while cur < end:
        nxt = min(cur + chunk, end)
        r = requests.get(
            CLOB_HIST,
            params={
                "market": token_id,
                "startTs": int(cur.timestamp()),
                "endTs": int(nxt.timestamp()),
                "fidelity": fidelity,
            },
            timeout=30,
        )
        if r.ok:
            out.extend(r.json().get("history", []))
        else:
            print(f"    chunk {cur.date()}->{nxt.date()} ERR {r.status_code}: {r.text[:80]}")
        cur = nxt
        time.sleep(0.3)
    if not out:
        return pd.DataFrame(columns=["t", "p"])
    df = pd.DataFrame(out).drop_duplicates("t").sort_values("t").reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["t"], unit="s", utc=True)
    return df[["ts", "t", "p"]]


def fetch_all_polymarket() -> None:
    print("=== Polymarket: discovering candidate markets ===")
    cands = find_candidates()
    print(f"  {len(cands)} candidates")
    (DATA / "candidates.json").write_text(json.dumps(cands, indent=2))

    print("\n=== Polymarket: pulling price history (CLOB) ===")
    for m in cands:
        mid = m["id"]
        slug = m.get("slug", mid)
        out_path = PRICES / f"{mid}.csv"
        if out_path.exists():
            print(f"  [skip] {mid}  {slug}")
            continue

        outcomes = json.loads(m["outcomes"])
        tokens = json.loads(m["clobTokenIds"])
        yes_token = tokens[outcomes.index("Yes")]

        # Window: the day before the market opened through the close, capped at 90 days
        # to keep download size sane for long-running markets.
        s_iso = m.get("startDate") or m.get("createdAt")
        e_iso = m.get("endDate") or m.get("closedTime")
        if not s_iso or not e_iso:
            print(f"  [skip - no dates] {mid}")
            continue
        s = datetime.fromisoformat(s_iso.replace("Z", "+00:00")) - timedelta(days=1)
        e = datetime.fromisoformat(e_iso.replace("Z", "+00:00")) + timedelta(days=1)
        # For markets that briefly resolved same-day, widen to a 24h window.
        if (e - s).total_seconds() < 86400:
            e = s + timedelta(days=2)

        # Fidelity: minute for short markets (<7 days), hourly otherwise.
        fid = 1 if (e - s).days <= 7 else 60
        print(f"  {mid:>7} fid={fid:>2}  {s.date()}->{e.date()}  {m['question']}")
        df = fetch_history(yes_token, s, e, fidelity=fid)
        if df.empty:
            print(f"    no points returned")
            continue
        df.to_csv(out_path, index=False)
        print(f"    {len(df):>5} rows  saved {out_path.name}")


def fetch_svb_minute() -> None:
    """The 'Will SVB fail?' market opened and closed in a single afternoon,
    so we pull a dedicated minute-fidelity file alongside the hourly one."""
    print("\n=== Polymarket: SVB market at minute fidelity ===")
    cands_path = DATA / "candidates.json"
    if not cands_path.exists():
        print("  candidates.json missing, skipping")
        return
    cands = json.loads(cands_path.read_text())
    svb = next((c for c in cands if c.get("question") == "Will SVB fail?"), None)
    if svb is None:
        print("  SVB market not in candidates list")
        return
    out = PRICES / f"{svb['id']}_min.csv"
    if out.exists():
        print(f"  [skip] {out.name}")
        return
    tokens = json.loads(svb["clobTokenIds"])
    outcomes = json.loads(svb["outcomes"])
    yes_token = tokens[outcomes.index("Yes")]
    s = datetime(2023, 3, 10, 12, 0, tzinfo=timezone.utc)
    e = datetime(2023, 3, 11, 0, 0, tzinfo=timezone.utc)
    df = fetch_history(yes_token, s, e, fidelity=1)
    if df.empty:
        print("  no points returned")
        return
    df.to_csv(out, index=False)
    print(f"  {len(df)} minute observations saved to {out.name}")


def fetch_yahoo() -> None:
    print("\n=== Yahoo Finance baselines ===")
    # KRE = regional bank ETF, SPY = broad market, TLT = long Treasuries,
    # ^TNX = 10y yield, ^VIX = vol, ^IRX = 13w T-bill yield (proxy for Fed expectations).
    # SIVBQ = SVB's post-delisting OTC ticker (limited data); SBNY/FRC similarly limited.
    tickers = ["KRE", "SPY", "TLT", "^TNX", "^VIX", "^IRX"]
    start = "2023-01-01"
    end = "2023-06-30"
    for tk in tickers:
        path = YAHOO / f"{tk.replace('^','').replace('=','')}.csv"
        if path.exists():
            print(f"  [skip] {tk}")
            continue
        df = yf.download(tk, start=start, end=end, progress=False, auto_adjust=False)
        if df.empty:
            print(f"  {tk:>6} EMPTY")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df.to_csv(path)
        print(f"  {tk:>6} {len(df)} rows  {df.index.min().date()}->{df.index.max().date()}")


def main():
    fetch_all_polymarket()
    fetch_svb_minute()
    fetch_yahoo()
    print("\nDone.")


if __name__ == "__main__":
    main()
