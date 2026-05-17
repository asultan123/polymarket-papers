"""
analysis.py
-----------
Builds a daily liquidity-weighted Polymarket macro-pessimism index from the
raw price ticks pulled by fetch_data.py and compares it to the University of
Michigan Index of Consumer Sentiment (ICS).

Outputs everything we cite in the paper into data/results.json and prints to
stdout for review.

Steps:
  1. Load macro_markets.csv + macro_prices.csv + michigan.csv.
  2. Label each market with a sign:
        +1 if "Yes" outcome corresponds to a *negative* macro event
        -1 if "Yes" outcome corresponds to a *positive* macro event
        0  if ambiguous / outside scope (dropped).
  3. For each (market, day) compute a pessimism contribution p_i(t).
  4. Aggregate: pessimism_index(t) = sum_i w_i(t) * p_i(t),
       w_i(t) = volume_i / sum_active(volume_j),
       where "active" means market i has a price for day t.
  5. Monthly mean -> compare with Michigan ICS (Pearson, Spearman, OLS,
     Granger, lead-lag).
  6. Sensitivity: re-run with subsets (recession only, Fed only, equal weights,
     restricted to >$50k volume).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests, adfuller

DATA = Path(__file__).parent.parent / "data"
RESULTS = DATA / "results.json"


# -----------------------------------------------------------------------------
# 1. Label markets by macro polarity
# -----------------------------------------------------------------------------

# Slug substrings that indicate the YES outcome is BAD for the economy.
NEG_KEYWORDS = [
    "recession",                          # P(recession) up -> pessimism
    "rate-hike", "rates-above", "increase-interest-rates", "raise-interest-rates",
    "raise-rates", "fed-set-interest-rates-above", "rates-above",
    "rate-increase",
    "inflation-above", "inflation-be-above",
    "exceed-7", "exceed-6", "exceed-5", "exceed-4", "exceed-3",
    "be-7p", "be-6p", "be-5p", "be-4p", "be-above",
    "unemployment-above", "unemployment-rate-above", "u-3-unemployment-rate-above",
    "u-3-unemployment-rate-in-january-be-above", "u-3-unemployment-rate-in-january-be-above-3pt8",
    "u-3-unemployment-rate-in-january-be-above-4",
    "jobless-claims-exceed", "jobless-claims-above", "jobless-claims-be-above",
    "bank-failure", "bank-failures",
    "gdp-declines", "gdp-decreases",
    "credit-downgrade", "downgrade",
    "vix-above",
    "circuit-breaker", "emergency-rate-cut",
]

# Slug substrings that indicate the YES outcome is GOOD for the economy.
POS_KEYWORDS = [
    "fed-cut", "rate-cut", "decrease-interest-rates",
    "fed-decreases-interest-rates", "cut-rates",
    "gdp-growth-be-greater-than", "gdp-growth-greater-than",
    "gdp-be-greater-than", "gdp-increases",
    "inflation-below", "inflation-be-below",
    "be-less-than-2", "be-less-than-3",
    "unemployment-below", "jobless-claims-below",
    "spx-above", "sp500-above", "stocks-above",
]


def polarity(slug: str) -> int:
    """+1 if YES is a bad-for-economy event, -1 if good, 0 if ambiguous."""
    s = (slug or "").lower()
    pos = any(k in s for k in POS_KEYWORDS)
    neg = any(k in s for k in NEG_KEYWORDS)
    if neg and not pos:
        return +1
    if pos and not neg:
        return -1

    # Inflation patterns.
    # Conventions for these markets:
    #   - "or-more"  -> YES means the realized level is AT LEAST the threshold
    #     -> hawkish, BAD for consumer sentiment if threshold is non-trivial
    #   - "or-less"  -> YES means realized level is AT MOST the threshold
    #     -> dovish, GOOD for consumer sentiment if threshold is non-trivial
    #   - exact-equals at threshold T -> YES means realized ~ T, treat as
    #     bad-if-T-high, good-if-T-low; this is noisy but symmetric.
    has_or_more  = ("or-more" in s) or ("more-than" in s) or ("exceed" in s) or ("above" in s)
    has_or_less  = ("or-less" in s) or ("below" in s)
    is_annual = "annual-inflation" in s or ("inflation" in s and "annual" in s) or "inflation-reach" in s
    is_monthly = "monthly-inflation" in s or "from-" in s  # "from-may-to-june" is monthly

    m = re.search(r"inflation[-a-z0-9]*?(\d+)pt(\d+)", s)
    if m:
        level = int(m.group(1)) + int(m.group(2)) / 10
        if has_or_more:
            return +1
        if has_or_less:
            return -1
        # bare exact match
        if is_annual:
            return +1 if level >= 3.0 else -1
        if is_monthly:
            return +1 if level >= 0.3 else -1
    m = re.search(r"inflation-reach-more-than-(\d+)", s)
    if m:
        return +1
    return 0


# -----------------------------------------------------------------------------
# 2. Build the pessimism index
# -----------------------------------------------------------------------------

def build_index(markets: pd.DataFrame,
                prices: pd.DataFrame,
                min_volume: float = 5_000) -> pd.DataFrame:
    """Return a daily DataFrame indexed by date with cols pessimism, n_active, total_vol."""
    markets = markets.copy()
    markets["polarity"] = markets["slug"].map(polarity)
    keep = markets[(markets["polarity"] != 0) & (markets["volume"] >= min_volume)]
    print(f"[build_index] markets kept after polarity+volume: {len(keep)}/{len(markets)}")
    print(f"[build_index]   neg (YES=bad): {(keep['polarity']==1).sum()}")
    print(f"[build_index]   pos (YES=good): {(keep['polarity']==-1).sum()}")

    pol_lookup = dict(zip(keep["id"].astype(str), keep["polarity"]))
    vol_lookup = dict(zip(keep["id"].astype(str), keep["volume"]))
    p = prices.copy()
    p["market_id"] = p["market_id"].astype(str)
    p = p[p["market_id"].isin(pol_lookup)]
    p["polarity"] = p["market_id"].map(pol_lookup)
    p["volume"] = p["market_id"].map(vol_lookup)
    # Each market contributes p_yes if polarity=+1, else 1-p_yes
    p["pessimism_raw"] = np.where(p["polarity"] == 1, p["p"], 1 - p["p"])
    # cap volume at 95th pct so a few mega-markets don't dominate
    cap = np.percentile(list(vol_lookup.values()), 95)
    p["w_volume"] = np.minimum(p["volume"], cap)
    # daily weighted mean
    grp = p.groupby(p["date"].dt.date)
    daily = grp.apply(
        lambda g: pd.Series({
            "pessimism": np.average(g["pessimism_raw"], weights=g["w_volume"]),
            "n_active": g["market_id"].nunique(),
            "total_vol": g["w_volume"].sum(),
        })
    )
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()
    return daily


# -----------------------------------------------------------------------------
# 3. Statistical tests vs. Michigan ICS
# -----------------------------------------------------------------------------

def to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    monthly = daily.resample("MS").agg({
        "pessimism": "mean",
        "n_active": "mean",
        "total_vol": "sum",
    })
    return monthly


def merge_with_ics(monthly_idx: pd.DataFrame, ics: pd.DataFrame) -> pd.DataFrame:
    ics = ics.copy()
    ics["date"] = pd.to_datetime(ics["date"])
    ics = ics.set_index("date")
    df = monthly_idx.join(ics, how="inner")
    return df


def run_stats(merged: pd.DataFrame, label: str) -> dict:
    """Return correlations + OLS + Granger + lead-lag + ADF summaries."""
    out = {"label": label, "n_months": int(len(merged))}
    if len(merged) < 6:
        out["note"] = "too few months for inference"
        return out

    x = merged["pessimism"].values
    y = merged["ics"].values
    # Levels
    out["pearson_levels_r"], out["pearson_levels_p"] = map(float, st.pearsonr(x, y))
    out["spearman_levels_r"], out["spearman_levels_p"] = map(float, st.spearmanr(x, y))

    # First differences (more honest given likely non-stationarity)
    dx = np.diff(x)
    dy = np.diff(y)
    if len(dx) > 2:
        out["pearson_diff_r"], out["pearson_diff_p"] = map(float, st.pearsonr(dx, dy))
        out["spearman_diff_r"], out["spearman_diff_p"] = map(float, st.spearmanr(dx, dy))
    else:
        out["pearson_diff_r"] = None

    # ADF stationarity for both
    try:
        out["adf_pessimism_p"] = float(adfuller(x, maxlag=4)[1])
        out["adf_ics_p"] = float(adfuller(y, maxlag=4)[1])
    except Exception as e:
        out["adf_error"] = str(e)

    # OLS: ICS_t = a + b * pessimism_t (Newey-West s.e., 3 lags)
    X = sm.add_constant(x)
    try:
        mod = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        out["ols_levels_beta"] = float(mod.params[1])
        out["ols_levels_se"] = float(mod.bse[1])
        out["ols_levels_p"] = float(mod.pvalues[1])
        out["ols_levels_r2"] = float(mod.rsquared)
    except Exception as e:
        out["ols_error"] = str(e)

    # OLS on first differences
    if len(dx) > 2:
        Xd = sm.add_constant(dx)
        try:
            md = sm.OLS(dy, Xd).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
            out["ols_diff_beta"] = float(md.params[1])
            out["ols_diff_se"] = float(md.bse[1])
            out["ols_diff_p"] = float(md.pvalues[1])
            out["ols_diff_r2"] = float(md.rsquared)
        except Exception as e:
            out["ols_diff_error"] = str(e)

    # Granger: does pessimism Granger-cause ICS?  use first-difference series
    if len(dx) > 6:
        gdf = pd.DataFrame({"ics_d": dy, "pess_d": dx})
        try:
            res = grangercausalitytests(gdf[["ics_d", "pess_d"]],
                                        maxlag=min(3, len(dx)//4),
                                        verbose=False)
            out["granger_pess_to_ics_p"] = {
                str(l): float(v[0]["ssr_ftest"][1]) for l, v in res.items()
            }
            res2 = grangercausalitytests(gdf[["pess_d", "ics_d"]],
                                         maxlag=min(3, len(dx)//4),
                                         verbose=False)
            out["granger_ics_to_pess_p"] = {
                str(l): float(v[0]["ssr_ftest"][1]) for l, v in res2.items()
            }
        except Exception as e:
            out["granger_error"] = str(e)

    # Lead-lag cross-correlation up to +-3 months on first differences
    if len(dx) > 6:
        lags = range(-3, 4)
        leadlag = {}
        for L in lags:
            if L > 0:
                a, b = dx[:-L], dy[L:]
            elif L < 0:
                a, b = dx[-L:], dy[:L]
            else:
                a, b = dx, dy
            if len(a) > 2:
                r, p = st.pearsonr(a, b)
                leadlag[str(L)] = {"r": float(r), "p": float(p), "n": int(len(a))}
        out["leadlag"] = leadlag

    return out


# -----------------------------------------------------------------------------
# 4. Subset analyses
# -----------------------------------------------------------------------------

def subset(markets: pd.DataFrame, slug_filter: str) -> pd.DataFrame:
    return markets[markets["slug"].str.contains(slug_filter, case=False, na=False)]


def main():
    print("loading data...")
    markets = pd.read_csv(DATA / "macro_markets.csv")
    prices = pd.read_csv(DATA / "macro_prices.csv", parse_dates=["date"])
    ics = pd.read_csv(DATA / "michigan.csv", parse_dates=["date"])
    print(f"  markets: {len(markets)}  price ticks: {len(prices)}  ics rows: {len(ics)}")
    print(f"  price span: {prices['date'].min().date()} -> {prices['date'].max().date()}")

    all_results = {}

    # ---- baseline: full index ----
    daily = build_index(markets, prices)
    daily.to_csv(DATA / "pessimism_daily.csv")
    monthly = to_monthly(daily)
    monthly.to_csv(DATA / "pessimism_monthly.csv")
    merged = merge_with_ics(monthly, ics)
    merged.to_csv(DATA / "merged_monthly.csv")
    print(f"merged monthly span: {merged.index.min().date()} -> {merged.index.max().date()} (n={len(merged)})")
    all_results["full"] = run_stats(merged, "full liquidity-weighted index")

    # ---- baseline: equal-weight ----
    daily_eq = build_index_equal(markets, prices)
    monthly_eq = to_monthly(daily_eq)
    merged_eq = merge_with_ics(monthly_eq, ics)
    all_results["equal_weight"] = run_stats(merged_eq, "equal-weighted index")

    # ---- subset: recession markets only ----
    rec = subset(markets, "recession")
    daily_r = build_index(rec, prices)
    if len(daily_r):
        merged_r = merge_with_ics(to_monthly(daily_r), ics)
        all_results["recession_only"] = run_stats(merged_r, "recession markets only")
        all_results["recession_only"]["n_markets"] = int(len(rec))

    # ---- subset: Fed rate markets ----
    fed = markets[markets["slug"].str.contains(r"fed-|interest-rates|powell|fomc", case=False, na=False, regex=True)]
    daily_f = build_index(fed, prices)
    if len(daily_f):
        merged_f = merge_with_ics(to_monthly(daily_f), ics)
        all_results["fed_only"] = run_stats(merged_f, "Fed rate markets only")
        all_results["fed_only"]["n_markets"] = int(len(fed))

    # ---- subset: inflation markets ----
    infl = markets[markets["slug"].str.contains("inflation|cpi", case=False, na=False, regex=True)]
    daily_i = build_index(infl, prices)
    if len(daily_i):
        merged_i = merge_with_ics(to_monthly(daily_i), ics)
        all_results["inflation_only"] = run_stats(merged_i, "inflation markets only")
        all_results["inflation_only"]["n_markets"] = int(len(infl))

    # ---- restrict to high-liquidity markets ----
    daily_high = build_index(markets, prices, min_volume=50_000)
    if len(daily_high):
        merged_h = merge_with_ics(to_monthly(daily_high), ics)
        all_results["high_volume"] = run_stats(merged_h, "markets with vol>$50k")

    # ---- a sanity / placebo: replace Polymarket index with random walk ----
    rng = np.random.default_rng(42)
    placebo = monthly.copy()
    placebo["pessimism"] = np.cumsum(rng.standard_normal(len(monthly))) / np.sqrt(len(monthly))
    placebo_merged = merge_with_ics(placebo, ics)
    all_results["placebo_random_walk"] = run_stats(placebo_merged, "placebo: random walk")

    # ---- coverage diagnostics ----
    all_results["coverage"] = {
        "n_candidate_markets": int(len(markets)),
        "n_markets_with_polarity_label": int((markets["slug"].map(polarity) != 0).sum()),
        "n_markets_in_index": int(daily.attrs.get("n_markets", 0)),
        "n_daily_ticks": int(len(prices)),
        "daily_span": [str(prices["date"].min().date()), str(prices["date"].max().date())],
        "monthly_overlap_with_ics": int(len(merged)),
    }

    with open(RESULTS, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"wrote {RESULTS}")

    # quick console summary
    for k, v in all_results.items():
        if k == "coverage":
            continue
        print(f"--- {k} ---")
        for kk in ("n_months", "pearson_levels_r", "pearson_levels_p",
                   "pearson_diff_r", "pearson_diff_p",
                   "ols_levels_beta", "ols_levels_p", "ols_levels_r2",
                   "ols_diff_beta", "ols_diff_p", "ols_diff_r2"):
            if kk in v:
                print(f"  {kk}: {v[kk]}")


def build_index_equal(markets: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    markets = markets.copy()
    markets["polarity"] = markets["slug"].map(polarity)
    keep = markets[markets["polarity"] != 0]
    pol_lookup = dict(zip(keep["id"].astype(str), keep["polarity"]))
    p = prices.copy()
    p["market_id"] = p["market_id"].astype(str)
    p = p[p["market_id"].isin(pol_lookup)]
    p["polarity"] = p["market_id"].map(pol_lookup)
    p["pessimism_raw"] = np.where(p["polarity"] == 1, p["p"], 1 - p["p"])
    grp = p.groupby(p["date"].dt.date)
    daily = grp.apply(
        lambda g: pd.Series({
            "pessimism": g["pessimism_raw"].mean(),
            "n_active": g["market_id"].nunique(),
            "total_vol": 0.0,
        })
    )
    daily.index = pd.to_datetime(daily.index)
    return daily.sort_index()


if __name__ == "__main__":
    main()
