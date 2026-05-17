"""Generate the paper's figures from the merged daily panel.

Reads ``data/polymarket_prices.csv``, ``data/fte_averages.csv`` and
``results.json``; writes:
  - figures/fig1_panel.png   : 6-panel time series, market vs poll-implied
                               win probability, for the 6 closest swing
                               states (AZ, GA, MI, NV, NC, PA, WI).
  - figures/fig2_scatter.png : final pre-election market price vs poll-implied
                               win probability, colored by outcome.
  - figures/fig3_calibration.png : reliability diagram comparing the two
                                   forecast streams.
  - figures/fig4_national_gap.png : daily national time series with the
                                    gap between market and poll-implied
                                    Trump win probability.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

ELECTION_DAY = pd.Timestamp("2024-11-05")
STATE_MAP = {
    "Arizona": "Arizona", "Georgia": "Georgia", "Michigan": "Michigan",
    "Nevada": "Nevada", "North Carolina": "North Carolina",
    "Pennsylvania": "Pennsylvania", "Wisconsin": "Wisconsin",
    "Florida": "Florida", "Texas": "Texas", "Ohio": "Ohio",
    "Minnesota": "Minnesota", "New Hampshire": "New Hampshire",
    "Virginia": "Virginia", "National": "National",
}


def state_sigma(days_out: float) -> float:
    d = max(0.0, min(180.0, days_out))
    return 3.0 + (5.0 - 3.0) * (d / 180.0)


def load_merged():
    poly = pd.read_csv(DATA / "polymarket_prices.csv")
    poly["date"] = pd.to_datetime(poly["t"], unit="s", utc=True).dt.tz_localize(None).dt.normalize()
    poly = poly.groupby(["state", "date"], as_index=False)["p"].mean()

    fte = pd.read_csv(DATA / "fte_averages.csv")
    fte = fte[fte["cycle"].astype(str) == "2024"].copy()
    fte["date"] = pd.to_datetime(fte["date"])
    fte["pct"] = fte["pct_trend_adjusted"].fillna(fte["pct_estimate"])
    fte = fte[fte["candidate"].isin(["Trump", "Harris"])]
    wide = fte.pivot_table(
        index=["state", "date"], columns="candidate", values="pct"
    ).reset_index()
    wide = wide.dropna(subset=["Trump", "Harris"])
    wide["margin"] = wide["Trump"] - wide["Harris"]
    wide["days_out"] = (ELECTION_DAY - wide["date"]).dt.days.astype(float)
    wide["sigma"] = wide["days_out"].apply(state_sigma)
    wide["winprob"] = stats.norm.cdf(wide["margin"] / wide["sigma"])
    wide = wide[wide["state"].isin(STATE_MAP.keys())]
    wide["state"] = wide["state"].map(STATE_MAP)

    merged = pd.merge(poly, wide[["state", "date", "margin", "winprob"]],
                      on=["state", "date"], how="inner")
    return merged


def fig1_panel(merged: pd.DataFrame):
    states = ["Pennsylvania", "Michigan", "Wisconsin",
              "Arizona", "Georgia", "Nevada"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True, sharey=True)
    for ax, st in zip(axes.flat, states):
        sub = merged[merged["state"] == st].sort_values("date")
        ax.plot(sub["date"], sub["p"], color="#1f77b4", lw=1.5, label="Polymarket P(Trump wins)")
        ax.plot(sub["date"], sub["winprob"], color="#d62728", lw=1.5, ls="--",
                label="538 poll-implied P(Trump wins)")
        ax.axhline(0.5, color="gray", lw=0.5, alpha=0.5)
        ax.axvline(ELECTION_DAY, color="black", lw=0.6, alpha=0.6)
        ax.set_title(st, fontsize=10)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
    axes[0, 0].set_ylabel("Win probability", fontsize=9)
    axes[1, 0].set_ylabel("Win probability", fontsize=9)
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Daily Trump win probability: Polymarket vs FiveThirtyEight (2024)",
                 fontsize=11)
    fig.tight_layout()
    out = FIG / "fig1_panel.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def fig2_scatter():
    res = json.loads((ROOT / "results.json").read_text())
    rows = res["per_state"]
    xs = [r["final_market_price"] for r in rows]
    ys = [r["final_poll_winprob"] for r in rows]
    won = [r["actual_trump_won"] for r in rows]
    names = [r["state"] for r in rows]

    fig, ax = plt.subplots(figsize=(6, 5.5))
    for x, y, w, n in zip(xs, ys, won, names):
        c = "#d62728" if w == 1 else "#1f77b4"
        marker = "^" if w == 1 else "v"
        ax.scatter([x], [y], color=c, s=70, marker=marker, edgecolor="black",
                   linewidth=0.5, zorder=3)
        ax.annotate(n, (x, y), fontsize=7, xytext=(4, 4),
                    textcoords="offset points")
    ax.plot([0, 1], [0, 1], color="gray", lw=0.7, ls=":")
    ax.set_xlabel("Final Polymarket price (P(Trump wins))")
    ax.set_ylabel("Final 538 poll-implied P(Trump wins)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Final pre-election forecasts (Nov 4, 2024)")
    from matplotlib.lines import Line2D
    leg = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#d62728",
               markeredgecolor="black", markersize=9, label="Trump won"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="#1f77b4",
               markeredgecolor="black", markersize=9, label="Harris won"),
        Line2D([0], [0], color="gray", ls=":", label="y = x"),
    ]
    ax.legend(handles=leg, loc="upper left", fontsize=8)
    fig.tight_layout()
    out = FIG / "fig2_scatter.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def fig3_calibration():
    res = json.loads((ROOT / "results.json").read_text())
    cm = res["calibration_market"]
    cp = res["calibration_poll"]
    fig, ax = plt.subplots(figsize=(6, 5))
    if cm:
        ax.plot([b["mean_pred"] for b in cm], [b["frac_won"] for b in cm],
                marker="o", color="#1f77b4", label="Polymarket", lw=1.5)
        for b in cm:
            ax.annotate(f"n={b['n']}", (b["mean_pred"], b["frac_won"]),
                        fontsize=7, xytext=(4, -10), textcoords="offset points",
                        color="#1f77b4")
    if cp:
        ax.plot([b["mean_pred"] for b in cp], [b["frac_won"] for b in cp],
                marker="s", color="#d62728", label="538 poll-implied",
                lw=1.5, ls="--")
        for b in cp:
            ax.annotate(f"n={b['n']}", (b["mean_pred"], b["frac_won"]),
                        fontsize=7, xytext=(4, 6), textcoords="offset points",
                        color="#d62728")
    ax.plot([0, 1], [0, 1], color="gray", lw=0.7, ls=":")
    ax.set_xlabel("Mean predicted Trump win probability (bucket)")
    ax.set_ylabel("Empirical Trump win fraction (bucket)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Reliability diagram (n = {res['n_states']} states)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    out = FIG / "fig3_calibration.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def fig4_national_gap(merged: pd.DataFrame):
    nat = merged[merged["state"] == "National"].sort_values("date").copy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(nat["date"], nat["p"], color="#1f77b4", lw=1.5,
             label="Polymarket P(Trump wins)")
    ax1.plot(nat["date"], nat["winprob"], color="#d62728", lw=1.5, ls="--",
             label="538 poll-implied P(Trump wins)")
    ax1.axhline(0.5, color="gray", lw=0.5, alpha=0.5)
    ax1.axvline(ELECTION_DAY, color="black", lw=0.6, alpha=0.6)
    # Mark Biden withdrawal (2024-07-21) and Harris nomination (2024-08-22).
    ax1.axvline(pd.Timestamp("2024-07-21"), color="purple", lw=0.6, ls=":")
    ax1.text(pd.Timestamp("2024-07-21"), 0.05, "Biden out", fontsize=7,
             rotation=90, color="purple", va="bottom")
    ax1.axvline(pd.Timestamp("2024-09-10"), color="green", lw=0.6, ls=":")
    ax1.text(pd.Timestamp("2024-09-10"), 0.05, "Debate", fontsize=7,
             rotation=90, color="green", va="bottom")
    ax1.set_ylabel("Trump win probability")
    ax1.set_title("National Trump win probability, Polymarket vs 538-implied")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_ylim(0, 1)

    gap = nat["p"] - nat["winprob"]
    ax2.fill_between(nat["date"], 0, gap, where=(gap >= 0), color="#d62728",
                     alpha=0.5, label="Market more Trump-favorable")
    ax2.fill_between(nat["date"], 0, gap, where=(gap < 0), color="#1f77b4",
                     alpha=0.5, label="Polls more Trump-favorable")
    ax2.axhline(0, color="black", lw=0.7)
    ax2.set_ylabel("Market – polls\n(probability)")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    out = FIG / "fig4_national_gap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def main():
    merged = load_merged()
    fig1_panel(merged)
    fig2_scatter()
    fig3_calibration()
    fig4_national_gap(merged)


if __name__ == "__main__":
    main()
