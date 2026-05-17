"""Analyze Polymarket prices vs FiveThirtyEight polling averages for 2024.

Pipeline:
  1. Load FTE polling averages (Wayback snapshot) and reshape to a daily
     state x candidate panel.
  2. Build a "polls-implied" Trump win probability for each state-day by
     mapping the daily two-party margin through a Gaussian CDF with a
     state-level standard deviation tuned from the literature on poll
     forecast error (~5pp at three months out, shrinking to ~3pp on
     election eve; see Silver, 2014; Heidemanns et al., 2020).
  3. Load Polymarket daily mid prices, align on dates, restrict to states
     for which we have both series.
  4. Compute, per state:
        - Pearson correlation between margin and price
        - Final pre-election price and polls-implied probability
        - Brier score against the certified result
        - Daily mean absolute deviation between the two forecast series
  5. Compute panel-wide metrics: aggregate Brier, calibration histogram,
     and a paired comparison (sign test) of who was closer to the truth
     on the eve of the election.

All numbers written to results.json so that the LaTeX paper can quote them
directly. Console output is a human-readable summary.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "results.json"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

ELECTION_DAY = pd.Timestamp("2024-11-05")

# Mapping from FTE state names to our state keys.
STATE_MAP = {
    "Arizona": "Arizona", "Georgia": "Georgia", "Michigan": "Michigan",
    "Nevada": "Nevada", "North Carolina": "North Carolina",
    "Pennsylvania": "Pennsylvania", "Wisconsin": "Wisconsin",
    "Florida": "Florida", "Texas": "Texas", "Ohio": "Ohio",
    "Minnesota": "Minnesota", "New Hampshire": "New Hampshire",
    "Virginia": "Virginia", "National": "National",
}


def days_to_election(d: pd.Timestamp) -> float:
    return (ELECTION_DAY - d).days


def state_sigma(days_out: float) -> float:
    """State-level forecast error standard deviation, in percentage points.

    A linear interpolation between 5.0pp at >=180 days out and 3.0pp on
    election eve. Endpoints derived from the post-mortem in Heidemanns,
    Gelman & Morris (2020), Table 2, which reports state RMSE shrinking
    from ~5pp early in the cycle to ~3pp in the final week.
    """
    d = max(0.0, min(180.0, days_out))
    return 3.0 + (5.0 - 3.0) * (d / 180.0)


def margin_to_winprob(margin_pp: float, days_out: float) -> float:
    """Two-party margin (Trump - Harris, in pp) -> P(Trump wins)."""
    sigma = state_sigma(days_out)
    return float(stats.norm.cdf(margin_pp / sigma))


def load_fte() -> pd.DataFrame:
    df = pd.read_csv(DATA / "fte_averages.csv")
    df = df[df["cycle"].astype(str) == "2024"].copy()
    df["date"] = pd.to_datetime(df["date"])
    # Use the trend-adjusted estimate when available, else raw estimate.
    df["pct"] = df["pct_trend_adjusted"].fillna(df["pct_estimate"])
    df = df[df["candidate"].isin(["Trump", "Harris"])]
    # Pivot to wide: one row per (state, date).
    wide = df.pivot_table(
        index=["state", "date"], columns="candidate", values="pct"
    ).reset_index()
    wide = wide.dropna(subset=["Trump", "Harris"])
    wide["margin"] = wide["Trump"] - wide["Harris"]
    wide["days_out"] = wide["date"].apply(days_to_election)
    wide["winprob"] = [
        margin_to_winprob(m, d)
        for m, d in zip(wide["margin"], wide["days_out"])
    ]
    return wide.rename(columns={"state": "state_fte"})


def load_polymarket() -> pd.DataFrame:
    df = pd.read_csv(DATA / "polymarket_prices.csv")
    df["date"] = pd.to_datetime(df["t"], unit="s", utc=True).dt.tz_localize(None).dt.normalize()
    df = df.groupby(["state", "date"], as_index=False)["p"].mean()  # average if duplicates
    return df


def load_results() -> dict:
    df = pd.read_csv(DATA / "election_results.csv")
    return dict(zip(df["state"], df["trump_won"]))


@dataclass
class StateMetrics:
    state: str
    n_overlap_days: int
    first_overlap: str
    last_overlap: str
    pearson_r: float
    pearson_p: float
    final_market_price: float
    final_poll_margin_pp: float
    final_poll_winprob: float
    actual_trump_won: int
    market_brier: float
    poll_brier: float
    mean_abs_diff_winprob: float


def per_state(merged: pd.DataFrame, state: str, won: int) -> StateMetrics:
    sub = merged[merged["state"] == state].sort_values("date")
    if len(sub) < 5:
        return None
    r, p = stats.pearsonr(sub["p"], sub["winprob"])
    # "Final" = the latest observation strictly before election day.
    pre = sub[sub["date"] < ELECTION_DAY]
    last = pre.iloc[-1]
    market_brier = (last["p"] - won) ** 2
    poll_brier = (last["winprob"] - won) ** 2
    mad = float((sub["p"] - sub["winprob"]).abs().mean())
    return StateMetrics(
        state=state,
        n_overlap_days=int(len(sub)),
        first_overlap=str(sub["date"].iloc[0].date()),
        last_overlap=str(sub["date"].iloc[-1].date()),
        pearson_r=float(r),
        pearson_p=float(p),
        final_market_price=float(last["p"]),
        final_poll_margin_pp=float(last["margin"]),
        final_poll_winprob=float(last["winprob"]),
        actual_trump_won=int(won),
        market_brier=float(market_brier),
        poll_brier=float(poll_brier),
        mean_abs_diff_winprob=mad,
    )


def lead_lag_correlation(merged: pd.DataFrame, state: str, max_lag: int = 14):
    """Cross-correlation between daily changes in market price and poll margin.

    Returns (lag, r). Positive lag means market changes lead poll changes by
    `lag` days. Uses daily first differences to avoid spurious correlation
    from shared trend.
    """
    sub = merged[merged["state"] == state].sort_values("date").set_index("date")
    sub = sub[["p", "margin"]].astype(float)
    sub = sub.reindex(pd.date_range(sub.index.min(), sub.index.max())).interpolate()
    dp = sub["p"].diff()
    dm = sub["margin"].diff()
    rs = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x = dp.iloc[lag:]
            y = dm.iloc[: len(dm) - lag]
        else:
            x = dp.iloc[: lag]
            y = dm.iloc[-lag:]
        x = np.asarray(x); y = np.asarray(y)
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() < 30:
            continue
        if np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
            continue
        r, _ = stats.pearsonr(x[mask], y[mask])
        rs[lag] = float(r)
    return rs


def calibration_buckets(probs, outcomes, n_bins: int = 5):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi if hi < 1.0 else probs <= hi)
        if mask.sum() == 0:
            continue
        out.append({
            "lo": float(lo), "hi": float(hi),
            "n": int(mask.sum()),
            "mean_pred": float(np.mean(np.asarray(probs)[mask])),
            "frac_won": float(np.mean(np.asarray(outcomes)[mask])),
        })
    return out


def main():
    fte = load_fte()
    poly = load_polymarket()
    results = load_results()

    # Restrict to states we care about.
    fte = fte[fte["state_fte"].isin(STATE_MAP.keys())]
    fte["state"] = fte["state_fte"].map(STATE_MAP)

    merged = pd.merge(
        poly, fte[["state", "date", "margin", "winprob"]],
        on=["state", "date"], how="inner",
    )

    per_state_metrics = []
    panel_market_briers = []
    panel_poll_briers = []
    panel_outcomes = []
    panel_market_probs = []
    panel_poll_probs = []

    print(f"{'State':<18} {'N':>4} {'r':>6} {'pMkt':>6} {'pPoll':>6} "
          f"{'actual':>6} {'B_mkt':>7} {'B_poll':>7}")
    print("-" * 70)
    for state, won in sorted(results.items()):
        m = per_state(merged, state, won)
        if m is None:
            print(f"{state:<18} (insufficient overlap)")
            continue
        per_state_metrics.append(asdict(m))
        panel_market_briers.append(m.market_brier)
        panel_poll_briers.append(m.poll_brier)
        panel_outcomes.append(won)
        panel_market_probs.append(m.final_market_price)
        panel_poll_probs.append(m.final_poll_winprob)
        print(
            f"{m.state:<18} {m.n_overlap_days:>4d} {m.pearson_r:>6.3f} "
            f"{m.final_market_price:>6.2f} {m.final_poll_winprob:>6.2f} "
            f"{m.actual_trump_won:>6d} {m.market_brier:>7.4f} {m.poll_brier:>7.4f}"
        )

    panel_market_brier = float(np.mean(panel_market_briers))
    panel_poll_brier = float(np.mean(panel_poll_briers))
    print("-" * 70)
    print(f"Panel mean Brier (market): {panel_market_brier:.4f}")
    print(f"Panel mean Brier (polls) : {panel_poll_brier:.4f}")

    # Paired sign test: how often was the market closer than polls?
    diffs = np.array(panel_market_briers) - np.array(panel_poll_briers)
    market_better = int((diffs < 0).sum())
    poll_better = int((diffs > 0).sum())
    ties = int((diffs == 0).sum())
    # Two-sided sign test using binomial.
    n = market_better + poll_better
    if n > 0:
        p_sign = float(stats.binomtest(market_better, n, p=0.5,
                                       alternative="two-sided").pvalue)
    else:
        p_sign = float("nan")
    # Paired Wilcoxon signed-rank test on Brier differences.
    if len(diffs) >= 1 and not np.allclose(diffs, 0):
        wilc = stats.wilcoxon(panel_market_briers, panel_poll_briers,
                              zero_method="wilcox", alternative="two-sided")
        wilcoxon_stat = float(wilc.statistic)
        wilcoxon_p = float(wilc.pvalue)
    else:
        wilcoxon_stat = float("nan")
        wilcoxon_p = float("nan")

    # Lead-lag analysis (pooled over states).
    pooled_lags: dict[int, list[float]] = {}
    for state in results.keys():
        if state == "National":
            continue
        lags = lead_lag_correlation(merged, state)
        for k, v in lags.items():
            pooled_lags.setdefault(k, []).append(v)
    lag_summary = {
        str(k): {"mean": float(np.mean(v)), "n_states": len(v)}
        for k, v in sorted(pooled_lags.items())
    }
    # Best mean lag (most positive correlation).
    if pooled_lags:
        best_lag, best_mean = max(
            ((k, np.mean(v)) for k, v in pooled_lags.items()),
            key=lambda kv: kv[1],
        )
    else:
        best_lag, best_mean = 0, float("nan")

    # Calibration on the panel of final pre-election forecasts.
    calib_market = calibration_buckets(np.array(panel_market_probs),
                                       np.array(panel_outcomes))
    calib_poll = calibration_buckets(np.array(panel_poll_probs),
                                     np.array(panel_outcomes))

    # National-only deep dive: time series of |price - winprob|.
    nat = merged[merged["state"] == "National"].sort_values("date")
    if len(nat) > 0:
        nat_mean_gap = float((nat["p"] - nat["winprob"]).mean())
        nat_max_gap = float((nat["p"] - nat["winprob"]).abs().max())
        nat_first_date = str(nat["date"].iloc[0].date())
        nat_last_date = str(nat["date"].iloc[-1].date())
        nat_n = int(len(nat))
    else:
        nat_mean_gap = nat_max_gap = float("nan")
        nat_first_date = nat_last_date = ""
        nat_n = 0

    # Volume-weighted Brier as a robustness check (weight each state by log
    # of trade volume so the small Ohio market doesn't equal Pennsylvania).
    vol_map = json.loads((DATA / "polymarket_markets.json").read_text())
    vols = np.array([
        max(1.0, vol_map.get(m["state"], {}).get("volume", 1.0))
        for m in per_state_metrics
    ])
    log_w = np.log(vols)
    log_w /= log_w.sum()
    weighted_market_brier = float(np.sum(log_w * np.array(panel_market_briers)))
    weighted_poll_brier = float(np.sum(log_w * np.array(panel_poll_briers)))

    out = {
        "n_states": len(per_state_metrics),
        "election_day": str(ELECTION_DAY.date()),
        "per_state": per_state_metrics,
        "panel_market_brier": panel_market_brier,
        "panel_poll_brier": panel_poll_brier,
        "volume_weighted_market_brier": weighted_market_brier,
        "volume_weighted_poll_brier": weighted_poll_brier,
        "sign_test": {
            "market_closer": market_better,
            "poll_closer": poll_better,
            "ties": ties,
            "p_value": p_sign,
        },
        "wilcoxon_signed_rank": {
            "statistic": wilcoxon_stat,
            "p_value": wilcoxon_p,
        },
        "lead_lag_pooled": lag_summary,
        "best_lag_days": int(best_lag),
        "best_lag_mean_r": float(best_mean),
        "calibration_market": calib_market,
        "calibration_poll": calib_poll,
        "national": {
            "mean_signed_gap_market_minus_polls": nat_mean_gap,
            "max_abs_gap": nat_max_gap,
            "first_overlap": nat_first_date,
            "last_overlap": nat_last_date,
            "n_overlap_days": nat_n,
        },
        "data_provenance": {
            "fte_url": (
                "https://web.archive.org/web/20241105000000if_/"
                "https://projects.fivethirtyeight.com/polls/data/"
                "presidential_general_averages.csv"
            ),
            "polymarket_gamma": "https://gamma-api.polymarket.com/events",
            "polymarket_clob": "https://clob.polymarket.com/prices-history",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[ok] wrote {OUT}")
    print(f"Best lead/lag: market change leads poll change by "
          f"{best_lag} day(s) (mean r = {best_mean:.3f})")
    print(f"Sign test: market closer on {market_better}/{n} states, "
          f"p = {p_sign:.3f}")
    print(f"Wilcoxon signed-rank: W = {wilcoxon_stat}, p = {wilcoxon_p:.4f}")


if __name__ == "__main__":
    main()
