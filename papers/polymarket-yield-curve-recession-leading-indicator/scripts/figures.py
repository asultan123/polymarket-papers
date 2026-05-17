"""Generate the four figures referenced in the paper.

Reads cached CSVs from data/ and emits PNG files into figures/.

Figures
-------
1. figure_1_levels.png - Polymarket recession-2025 daily probability vs.
   Estrella-Mishkin probit P(rec) derived from 10y-3m Treasury spread, with the
   yield curve inversion shaded. Both 2024 and 2025 markets overlaid.
2. figure_2_spread_vs_market.png - Scatter of Polymarket P_yes versus 10y-3m
   spread for the four recession markets; linear fits per market.
3. figure_3_event_study.png - 30-day windows around the April 2025 US tariff
   announcement showing Polymarket reacting while EM probit moved sideways.
4. figure_4_volatility.png - Distribution of daily absolute changes for each
   signal.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True, parents=True)

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def estrella_mishkin_prob(spread):
    return norm.cdf(-0.5333 - 0.6330 * np.asarray(spread))


def load_poly(slug):
    df = pd.read_csv(DATA / f"polymarket_{slug}.csv")
    df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601")
    df["date"] = df["ts"].dt.tz_convert("UTC").dt.normalize()
    s = df.groupby("date")["price"].mean()
    s.index = s.index.tz_localize(None)
    return s


def load_manifold(name):
    df = pd.read_csv(DATA / f"manifold_{name}.csv")
    df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601")
    df["date"] = df["ts"].dt.tz_convert("UTC").dt.normalize()
    s = df.groupby("date")["price"].last()
    s.index = s.index.tz_localize(None)
    return s


def load_yields():
    return pd.read_csv(DATA / "treasury_yields.csv", index_col=0, parse_dates=True)


def figure_1_levels():
    y = load_yields()
    em_prob = pd.Series(estrella_mishkin_prob(y["spread_10y_3m"]), index=y.index)

    poly_2025 = load_poly("us-recession-in-2025")
    poly_2024 = load_poly("us-recession-in-2024-1")
    poly_nber = load_poly("us-recession-announced-by-nber-before-june-2025")

    inverted = (y["spread_10y_3m"] < 0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 5.2), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    # Top: probabilities
    ax1.plot(em_prob.index, em_prob.values, color="#7a7a7a", lw=1.4,
             label="EM probit P(rec | 10y-3m spread)")
    ax1.plot(poly_2024.index, poly_2024.values, color="#c4452f", lw=1.4,
             label="Polymarket: recession in 2024")
    ax1.plot(poly_nber.index, poly_nber.values, color="#e89540", lw=1.4,
             label="Polymarket: NBER recession by May 2025")
    ax1.plot(poly_2025.index, poly_2025.values, color="#2a6cb0", lw=1.7,
             label="Polymarket: recession in 2025 (main)")

    # Shade yield-curve-inverted periods.
    runs = inverted_runs(inverted)
    for s, e in runs:
        ax1.axvspan(s, e, color="#a0a0a0", alpha=0.13, linewidth=0)

    ax1.set_xlim(pd.Timestamp("2022-06-01"), pd.Timestamp("2026-05-15"))
    ax1.set_ylim(-0.02, 0.78)
    ax1.set_ylabel("P(US recession)")
    ax1.legend(loc="upper right", framealpha=0.9, ncol=1)
    ax1.set_title("Yield-curve probit vs. Polymarket recession markets (2022-2026)")
    ax1.grid(True, alpha=0.25, linewidth=0.5)

    # Bottom: spread
    ax2.plot(y.index, y["spread_10y_3m"], color="#404040", lw=1.2)
    ax2.axhline(0, color="#c4452f", lw=0.9, ls="--")
    for s, e in runs:
        ax2.axvspan(s, e, color="#a0a0a0", alpha=0.13, linewidth=0)
    ax2.set_ylabel("10y-3m (pp)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.25, linewidth=0.5)
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_minor_locator(mdates.MonthLocator((1, 4, 7, 10)))

    fig.tight_layout()
    fig.savefig(FIG / "figure_1_levels.png", bbox_inches="tight")
    plt.close(fig)


def inverted_runs(mask: pd.Series):
    runs = []
    in_run = False
    start = None
    prev = None
    for d, v in mask.items():
        if v and not in_run:
            in_run = True
            start = d
        elif not v and in_run:
            in_run = False
            runs.append((start, prev))
        prev = d
    if in_run:
        runs.append((start, prev))
    return runs


def figure_2_spread_vs_market():
    y = load_yields()
    spread = y["spread_10y_3m"]
    markets = {
        "rec-2025": ("us-recession-in-2025", "#2a6cb0"),
        "rec-2024": ("us-recession-in-2024-1", "#c4452f"),
        "NBER-by-May-2025": ("us-recession-announced-by-nber-before-june-2025", "#e89540"),
        "rec-by-2026": ("us-recession-by-end-of-2026", "#3f8a4f"),
    }
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for label, (slug, color) in markets.items():
        p = load_poly(slug)
        df = pd.concat([p.rename("p"), spread.rename("s")], axis=1, sort=True).dropna()
        if df.empty:
            continue
        ax.scatter(df["s"], df["p"], s=10, alpha=0.5, color=color, label=f"{label} (n={len(df)})", edgecolors="none")
        if len(df) >= 3 and df["s"].std() > 0:
            m, b = np.polyfit(df["s"], df["p"], 1)
            xs = np.linspace(df["s"].min(), df["s"].max(), 50)
            ax.plot(xs, m * xs + b, color=color, lw=1.0)

    # Overlay EM probit curve.
    xs = np.linspace(spread.min(), spread.max(), 200)
    ax.plot(xs, estrella_mishkin_prob(xs), color="black", lw=1.5, ls="--",
            label="EM probit (1998)")
    ax.axvline(0, color="grey", lw=0.7, ls=":")
    ax.set_xlabel("10y-3m Treasury spread (pp)")
    ax.set_ylabel("Polymarket P(YES) on recession question")
    ax.set_title("Recession probability vs. yield-curve slope")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=7)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(FIG / "figure_2_spread_vs_market.png", bbox_inches="tight")
    plt.close(fig)


def figure_3_event_study():
    y = load_yields()
    em_prob = pd.Series(estrella_mishkin_prob(y["spread_10y_3m"]), index=y.index)
    poly_2025 = load_poly("us-recession-in-2025")
    manifold_2025 = load_manifold("us-recession-end-2025-two-quarters")

    start = pd.Timestamp("2025-03-15")
    end = pd.Timestamp("2025-05-10")

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.plot(poly_2025.loc[start:end].index, poly_2025.loc[start:end].values,
            color="#2a6cb0", lw=1.6, marker="o", ms=2.5, label="Polymarket: recession in 2025")
    ax.plot(manifold_2025.loc[start:end].index, manifold_2025.loc[start:end].values,
            color="#9054b0", lw=1.4, marker="s", ms=2.5, label="Manifold: recession in 2025 (2Q)")
    ax.plot(em_prob.loc[start:end].index, em_prob.loc[start:end].values,
            color="#7a7a7a", lw=1.4, label="EM probit P(rec)")
    ax.axvline(pd.Timestamp("2025-04-02"), color="#c4452f", lw=0.9, ls="--", label='"Liberation Day" tariffs (Apr 2 2025)')
    ax.axvline(pd.Timestamp("2025-04-09"), color="#3f8a4f", lw=0.9, ls=":", label="Tariff 90-day pause (Apr 9)")
    ax.set_ylim(0, 0.78)
    ax.set_xlabel("Date")
    ax.set_ylabel("P(US recession)")
    ax.set_title("Event study: Polymarket vs. yield curve, April 2025 tariff shock")
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG / "figure_3_event_study.png", bbox_inches="tight")
    plt.close(fig)


def figure_4_volatility():
    y = load_yields()
    em_prob = pd.Series(estrella_mishkin_prob(y["spread_10y_3m"]), index=y.index)
    poly_2025 = load_poly("us-recession-in-2025")
    manifold_2025 = load_manifold("us-recession-end-2025-two-quarters")

    # Restrict to the 2025-recession overlap window.
    start = poly_2025.index.min()
    end = poly_2025.index.max()
    em_win = em_prob.loc[start:end]
    pol_win = poly_2025.loc[start:end]
    man_win = manifold_2025.reindex(pol_win.index).dropna()

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    bins = np.linspace(0, 0.30, 31)
    ax.hist(em_win.diff().abs().dropna(), bins=bins, alpha=0.55, color="#7a7a7a",
            label=f"EM probit (σ={em_win.diff().std():.4f})", density=True)
    ax.hist(pol_win.diff().abs().dropna(), bins=bins, alpha=0.55, color="#2a6cb0",
            label=f"Polymarket rec-2025 (σ={pol_win.diff().std():.4f})", density=True)
    if len(man_win) > 2:
        ax.hist(man_win.diff().abs().dropna(), bins=bins, alpha=0.55, color="#9054b0",
                label=f"Manifold rec-2025 (σ={man_win.diff().std():.4f})", density=True)
    ax.set_xlabel("|daily change in P(recession)|")
    ax.set_ylabel("Density")
    ax.set_title("Daily-change distribution, Jan 2025 - Dec 2025")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(FIG / "figure_4_volatility.png", bbox_inches="tight")
    plt.close(fig)


def main():
    figure_1_levels()
    figure_2_spread_vs_market()
    figure_3_event_study()
    figure_4_volatility()
    print(f"wrote figures to {FIG}")


if __name__ == "__main__":
    main()
