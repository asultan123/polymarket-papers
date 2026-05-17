"""Fetch markets from Polymarket Gamma API and Manifold Markets API.

Outputs:
  data/polymarket_markets.json
  data/manifold_markets.json

These are intermediate raw dumps used by analysis.py. Re-running this script will
overwrite them. Run takes ~5-10 minutes depending on API responsiveness.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

POLYMARKET_BASE = "https://gamma-api.polymarket.com/markets"
MANIFOLD_BASE = "https://api.manifold.markets/v0/markets"

USER_AGENT = "polymarket-manifold-arb-research/1.0 (academic research)"


def fetch_polymarket(target: int = 6000) -> list[dict]:
    """Fetch Polymarket binary markets via Gamma API.

    Uses offset pagination. The Gamma API returns 422 past a certain offset
    (we observe ~10200), so we stop gracefully when that happens.
    """
    out: list[dict] = []
    offset = 0
    page_size = 100
    while len(out) < target:
        params = {
            "limit": page_size,
            "offset": offset,
            "order": "volumeNum",
            "ascending": "false",
        }
        try:
            r = requests.get(
                POLYMARKET_BASE,
                params=params,
                headers={"User-Agent": USER_AGENT},
                timeout=30,
            )
            r.raise_for_status()
        except requests.HTTPError as exc:
            print(f"  polymarket: stopping at offset={offset} ({exc})")
            break
        page = r.json()
        if not page:
            break
        out.extend(page)
        offset += page_size
        print(f"  polymarket: {len(out)} markets")
        time.sleep(0.25)
    return out


def fetch_manifold(target: int = 8000) -> list[dict]:
    """Fetch Manifold binary markets via the lite-markets endpoint.

    The API supports cursor pagination via the `before` parameter (market id).
    We page until we have enough markets or hit the end.
    """
    out: list[dict] = []
    before: str | None = None
    page_size = 1000
    while len(out) < target:
        params: dict[str, str | int] = {"limit": page_size}
        if before is not None:
            params["before"] = before
        r = requests.get(
            MANIFOLD_BASE,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=30,
        )
        r.raise_for_status()
        page = r.json()
        if not page:
            break
        out.extend(page)
        before = page[-1]["id"]
        print(f"  manifold: {len(out)} markets")
        time.sleep(0.25)
    return out


def main() -> None:
    print("[1/2] Fetching Polymarket markets…")
    poly = fetch_polymarket(target=6000)
    with (DATA_DIR / "polymarket_markets.json").open("w") as f:
        json.dump(poly, f)
    print(f"  → saved {len(poly)} Polymarket markets")

    print("[2/2] Fetching Manifold markets…")
    mani = fetch_manifold(target=8000)
    with (DATA_DIR / "manifold_markets.json").open("w") as f:
        json.dump(mani, f)
    print(f"  → saved {len(mani)} Manifold markets")


if __name__ == "__main__":
    main()
