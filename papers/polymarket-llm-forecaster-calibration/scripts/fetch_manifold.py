"""Fetch resolved binary Manifold Markets for cross-platform comparison.

Polls /v0/markets with pagination (lexicographic by id), filters to BINARY,
resolved YES/NO, with non-trivial volume. Records the market's last
probability before resolution (the analogue of Polymarket's p_close).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

API = "https://api.manifold.markets/v0/markets"
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=40,
                    help="number of 1000-market pages to scan")
    ap.add_argument("--min-volume", type=float, default=100.0)
    ap.add_argument("--output", default=str(DATA / "manifold.csv"))
    args = ap.parse_args()

    rows = []
    before = None
    for page in range(args.pages):
        params = {"limit": 1000}
        if before:
            params["before"] = before
        r = requests.get(API, params=params, timeout=30)
        r.raise_for_status()
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
            vol = m.get("volume") or 0
            if vol < args.min_volume:
                continue
            rows.append(
                dict(
                    id=m.get("id"),
                    slug=m.get("slug"),
                    question=m.get("question"),
                    resolution=res,
                    label=1 if res == "YES" else 0,
                    prob_resolution=m.get("resolutionProbability"),
                    prob_current=m.get("probability"),
                    resolution_time_ms=m.get("resolutionTime"),
                    close_time_ms=m.get("closeTime"),
                    volume=vol,
                    unique_bettors=m.get("uniqueBettorCount"),
                )
            )
        before = batch[-1]["id"]
        print(f"page {page+1}: scanned {len(batch)}, kept {len(rows)} total")
        time.sleep(0.1)

    df = pd.DataFrame(rows)
    # filter again to non-null probabilities
    df = df.dropna(subset=["prob_resolution"])
    df.to_csv(args.output, index=False)
    print(f"wrote {len(df)} rows -> {args.output}")


if __name__ == "__main__":
    main()
