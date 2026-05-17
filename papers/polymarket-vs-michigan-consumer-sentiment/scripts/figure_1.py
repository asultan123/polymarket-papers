"""
figure_1.py
-----------
Three-panel figure used in the paper:
  (a) Time series of Michigan ICS (left) and the daily liquidity-weighted
      Polymarket macro-pessimism index (right), 2023-02 to 2026-05.
  (b) Scatter of monthly first differences using the FULL index — null.
  (c) Scatter of monthly first differences using the RECESSION-ONLY subset
      — significant negative slope (R^2 = 0.46, p < 1e-8).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).parent))
from analysis import build_index, polarity  # noqa

DATA = Path(__file__).parent.parent / "data"
OUT = Path(__file__).parent.parent / "figure_1.png"


def main():
    daily = pd.read_csv(DATA / "pessimism_daily.csv", parse_dates=["date"],
                        index_col="date")
    ics = pd.read_csv(DATA / "michigan.csv", parse_dates=["date"])
    merged = pd.read_csv(DATA / "merged_monthly.csv", parse_dates=["date"],
                          index_col="date")

    # recession-only subset
    markets = pd.read_csv(DATA / "macro_markets.csv")
    prices = pd.read_csv(DATA / "macro_prices.csv", parse_dates=["date"])
    rec_markets = markets[markets["slug"].str.contains("recession", case=False, na=False)]
    rec_daily = build_index(rec_markets, prices)
    rec_monthly = rec_daily.resample("MS").mean(numeric_only=True)
    ics_m = ics.assign(date=pd.to_datetime(ics["date"])).set_index("date")
    rec_merged = rec_monthly.join(ics_m, how="inner")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.7))

    # --- (a) time series ---
    ax = axes[0]
    win = ics[(ics["date"] >= daily.index.min())
              & (ics["date"] <= daily.index.max())]
    ax.plot(win["date"], win["ics"], color="#1f77b4", lw=1.6,
            label="Michigan ICS")
    ax.set_ylabel("Michigan ICS", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax.set_xlabel("date")
    ax2 = ax.twinx()
    ax2.plot(daily.index, daily["pessimism"], color="#d62728", lw=0.7,
             alpha=0.8)
    ax2.set_ylabel("full pessimism index", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax.set_title("(a) Time series, full index")
    ax.grid(alpha=0.25)
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")

    # --- (b) scatter, full index ---
    ax = axes[1]
    dx = merged["pessimism"].diff().dropna()
    dy = merged["ics"].diff().reindex(dx.index)
    g = dx.notna() & dy.notna()
    x = dx[g].values
    y = dy[g].values
    ax.scatter(x, y, s=22, color="#444", alpha=0.7)
    X = np.vstack([np.ones_like(x), x]).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    xs = np.linspace(x.min(), x.max(), 50)
    r = np.corrcoef(x, y)[0, 1]
    ax.plot(xs, beta[0] + beta[1] * xs, color="#d62728", lw=1.5)
    ax.axhline(0, color="k", lw=0.4, alpha=0.4)
    ax.axvline(0, color="k", lw=0.4, alpha=0.4)
    ax.set_xlabel(r"$\Delta$ pessimism (full)")
    ax.set_ylabel(r"$\Delta$ Michigan ICS")
    ax.set_title(f"(b) Full index: r = {r:+.2f}, n = {len(x)}")
    ax.grid(alpha=0.25)

    # --- (c) scatter, recession only ---
    ax = axes[2]
    dxr = rec_merged["pessimism"].diff().dropna()
    dyr = rec_merged["ics"].diff().reindex(dxr.index)
    gr = dxr.notna() & dyr.notna()
    xr, yr = dxr[gr].values, dyr[gr].values
    ax.scatter(xr, yr, s=22, color="#444", alpha=0.7)
    if len(xr) >= 3:
        Xr = np.vstack([np.ones_like(xr), xr]).T
        br, *_ = np.linalg.lstsq(Xr, yr, rcond=None)
        xs = np.linspace(xr.min(), xr.max(), 50)
        rr = np.corrcoef(xr, yr)[0, 1]
        ax.plot(xs, br[0] + br[1] * xs, color="#d62728", lw=1.5)
        ax.set_title(f"(c) Recession only: r = {rr:+.2f}, n = {len(xr)}")
    ax.axhline(0, color="k", lw=0.4, alpha=0.4)
    ax.axvline(0, color="k", lw=0.4, alpha=0.4)
    ax.set_xlabel(r"$\Delta$ pessimism (recession only)")
    ax.set_ylabel(r"$\Delta$ Michigan ICS")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
