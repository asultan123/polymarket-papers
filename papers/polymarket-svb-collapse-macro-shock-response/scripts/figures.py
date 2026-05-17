"""Build figures referenced by main.tex.

Each figure reads from ../data/ only -- no network calls. Outputs go to
../figures/ as PNG.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
PRICES = DATA / "prices"
YAHOO = DATA / "yahoo"
FIGS = HERE.parent / "figures"
FIGS.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def load_market(mid: str, minute: bool = False) -> pd.DataFrame:
    fname = f"{mid}_min.csv" if minute else f"{mid}.csv"
    df = pd.read_csv(PRICES / fname)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def load_yahoo(ticker: str) -> pd.DataFrame:
    df = pd.read_csv(YAHOO / f"{ticker}.csv", skiprows=3,
                     names=["Date", "Adj Close", "Close", "High", "Low", "Open", "Volume"])
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date").sort_index()


def fig_svb_intraday():
    """Minute-level price discovery in 'Will SVB fail?' on 2023-03-10."""
    svb = load_market("249094", minute=True)
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.plot(svb["ts"], svb["p"], color="#1f77b4", lw=1.2)
    ax.axvline(pd.Timestamp("2023-03-10 14:21", tz="UTC"), color="gray", ls="--", lw=0.8, alpha=0.7)
    ax.axvline(pd.Timestamp("2023-03-10 17:00", tz="UTC"), color="firebrick", ls="--", lw=0.8, alpha=0.7)
    ax.axvline(pd.Timestamp("2023-03-10 19:17", tz="UTC"), color="gray", ls=":", lw=0.8, alpha=0.7)
    ax.text(pd.Timestamp("2023-03-10 14:24", tz="UTC"), 0.05, "market\nopened", fontsize=7, color="gray")
    ax.text(pd.Timestamp("2023-03-10 17:03", tz="UTC"), 0.10, "FDIC named\nreceiver (~12:00 ET)", fontsize=7, color="firebrick")
    ax.text(pd.Timestamp("2023-03-10 19:20", tz="UTC"), 0.05, "market\nresolved", fontsize=7, color="gray")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("P(SVB fails)")
    ax.set_title("Polymarket 'Will SVB fail?' — minute fidelity, 2023-03-10 UTC")
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    out = FIGS / "fig_svb_intraday.pdf"
    fig.savefig(out, dpi=180)
    fig.savefig(FIGS / "fig_svb_intraday.png", dpi=180)
    print("wrote", out)


def fig_fed_pivot():
    """Path of P(0/25/50 bp hike) for March meeting, Feb 14 -> Mar 22."""
    series = {
        "P(no change)":   "248564",
        "P(+25 bp hike)": "248566",
        "P(+50 bp hike)": "248565",
        "P(-25 bp cut)":  "248563",
    }
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]
    for (label, mid), color in zip(series.items(), colors):
        df = load_market(mid)
        df = df[(df["ts"] >= "2023-02-14") & (df["ts"] <= "2023-03-22 18:00")]
        # Smooth a bit
        ax.plot(df["ts"], df["p"], label=label, color=color, lw=1.0)
    # Event lines
    for t, lab, c in [
        ("2023-03-08 21:00", "Silvergate / SVB 8-K", "gray"),
        ("2023-03-10 16:45", "SVB closed", "firebrick"),
        ("2023-03-12 22:00", "Signature / BTFP", "orange"),
        ("2023-03-22 18:00", "FOMC +25 bp", "black"),
    ]:
        ax.axvline(pd.Timestamp(t, tz="UTC"), color=c, ls="--", lw=0.7, alpha=0.65)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Polymarket implied probability")
    ax.set_title("March 2023 FOMC outcome — Polymarket probability paths")
    ax.legend(loc="upper left", ncol=2, fontsize=7)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    # Annotate events
    ax.text(pd.Timestamp("2023-03-08 22:00", tz="UTC"), 0.97, "Silvergate\nSVB 8-K", fontsize=6.5, color="gray", ha="left", va="top")
    ax.text(pd.Timestamp("2023-03-10 17:30", tz="UTC"), 0.85, "SVB\nclosed", fontsize=6.5, color="firebrick", ha="left", va="top")
    ax.text(pd.Timestamp("2023-03-12 23:00", tz="UTC"), 0.97, "Signature /\nBTFP", fontsize=6.5, color="orange", ha="left", va="top")
    ax.text(pd.Timestamp("2023-03-22 18:30", tz="UTC"), 0.20, "FOMC", fontsize=6.5, color="black", ha="left", va="top")
    fig.tight_layout()
    out = FIGS / "fig_fed_pivot.pdf"
    fig.savefig(out, dpi=180)
    fig.savefig(FIGS / "fig_fed_pivot.png", dpi=180)
    print("wrote", out)


def fig_cross_asset():
    """Daily series: P(Fed cut 2023), KRE close, 10y yield --- March 2023."""
    # Polymarket daily mean
    fc = load_market("248836")
    fc["date"] = fc["ts"].dt.tz_convert(None).dt.normalize()
    daily = fc.groupby("date")["p"].mean()

    kre = load_yahoo("KRE")["Close"]
    tnx = load_yahoo("TNX")["Close"]

    start = pd.Timestamp("2023-02-20")
    end = pd.Timestamp("2023-04-15")

    fig, axes = plt.subplots(3, 1, figsize=(6.0, 5.4), sharex=True)
    ax1, ax2, ax3 = axes

    ax1.plot(daily.loc[start:end].index, daily.loc[start:end].values, color="#1f77b4", lw=1.2)
    ax1.set_ylabel("P(Fed cuts in 2023)")
    ax1.set_ylim(0, 0.75)

    ax2.plot(kre.loc[start:end].index, kre.loc[start:end].values, color="#d62728", lw=1.2)
    ax2.set_ylabel("KRE close ($)")

    ax3.plot(tnx.loc[start:end].index, tnx.loc[start:end].values, color="#2ca02c", lw=1.2)
    ax3.set_ylabel("10y Treasury yield (%)")
    ax3.set_xlabel("Date (UTC)")

    # Event lines on all 3
    events = [
        ("2023-03-08", "Silvergate"),
        ("2023-03-10", "SVB closed"),
        ("2023-03-12", "Signature / BTFP"),
        ("2023-03-19", "CS / UBS"),
        ("2023-03-22", "FOMC"),
    ]
    for t, lab in events:
        for ax in (ax1, ax2, ax3):
            ax.axvline(pd.Timestamp(t), color="gray", ls="--", lw=0.6, alpha=0.5)
        ax1.text(pd.Timestamp(t), 0.70, lab, rotation=90, fontsize=6, color="gray", ha="right", va="top")

    ax3.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.suptitle("Polymarket vs. cash markets, Feb–Apr 2023", y=0.995)
    fig.tight_layout()
    out = FIGS / "fig_cross_asset.pdf"
    fig.savefig(out, dpi=180)
    fig.savefig(FIGS / "fig_cross_asset.png", dpi=180)
    print("wrote", out)


def fig_contagion():
    """Trajectories of named-bank failure markets + the catch-all third-bank market."""
    items = [
        ("Third US bank fail by Mar 17", "249162", "#1f77b4"),
        ("First Republic fail by Mar 17", "249167", "#2ca02c"),
        ("Bank of America fail by Mar 17", "249173", "#d62728"),
        ("Credit Suisse fail by Mar 31", "249185", "#9467bd"),
    ]
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    for label, mid, color in items:
        df = load_market(mid)
        # For CS, trim to before UBS deal so flatline doesn't dominate
        if mid == "249185":
            df = df[df["ts"] <= "2023-03-19 18:00"]
        ax.plot(df["ts"], df["p"], label=label, color=color, lw=1.0)
    for t, lab, c in [
        ("2023-03-10 16:45", "SVB closed", "firebrick"),
        ("2023-03-12 22:00", "Signature / BTFP", "orange"),
        ("2023-03-19 18:00", "UBS-CS", "black"),
    ]:
        ax.axvline(pd.Timestamp(t, tz="UTC"), color=c, ls="--", lw=0.7, alpha=0.6)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Polymarket implied probability of failure")
    ax.set_title("Contagion markets — Mar 10–31 2023")
    ax.legend(loc="upper right", fontsize=7)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.tight_layout()
    out = FIGS / "fig_contagion.pdf"
    fig.savefig(out, dpi=180)
    fig.savefig(FIGS / "fig_contagion.png", dpi=180)
    print("wrote", out)


def main():
    fig_svb_intraday()
    fig_fed_pivot()
    fig_cross_asset()
    fig_contagion()


if __name__ == "__main__":
    main()
