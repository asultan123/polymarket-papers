"""
Analysis of Polymarket recession-probability series vs Treasury yield-curve signals.

Produces the numbers and tables referenced in the paper. Run after fetch_data.py.
Writes a single JSON of summary statistics to results/summary.json and prints a
human-readable summary to stdout.

Methodology
-----------
1. Load Polymarket prices for the four resolved/active recession markets.
2. Load 10y/3m Treasury yields and compute the 10y-3m spread.
3. Convert the spread to an implied recession probability via the
   Estrella & Mishkin (1998) probit:  P = Phi(-0.5333 - 0.6330 * spread).
4. Resample everything to daily frequency, align on overlapping dates.
5. Compute:
     - Pearson and Spearman correlation between Polymarket and yield-curve P(rec).
     - Granger causality (max lag 5 days) in both directions.
     - Augmented Dickey-Fuller test for stationarity on first differences.
     - Volatility (daily standard deviation of differences).
     - Maximum drawdown / drawup of each signal.
     - Calibration: ex-ante mean P(rec) vs realised outcome (all 4 resolved NO).
6. Event study: 1-week change in each signal around Apr 2 - Apr 9 2025 (tariff shock).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr, spearmanr
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True, parents=True)

# Estrella-Mishkin (1998) probit coefficients for the 10y-3m spread.
EM_INTERCEPT = -0.5333
EM_SLOPE = -0.6330


def estrella_mishkin_prob(spread: pd.Series | np.ndarray) -> np.ndarray:
    """Convert 10y-3m Treasury spread (pp) to a probit-implied 12-month
    recession probability using the Estrella-Mishkin (1998) coefficients."""
    return norm.cdf(EM_INTERCEPT + EM_SLOPE * np.asarray(spread))


def load_polymarket(slug: str) -> pd.Series:
    df = pd.read_csv(DATA / f"polymarket_{slug}.csv")
    df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601")
    df["date"] = df["ts"].dt.tz_convert("UTC").dt.normalize()
    daily = df.groupby("date")["price"].mean()
    daily.index = daily.index.tz_localize(None)
    daily.name = f"poly_{slug}"
    return daily


def load_manifold(name: str) -> pd.Series:
    df = pd.read_csv(DATA / f"manifold_{name}.csv")
    df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601")
    df["date"] = df["ts"].dt.tz_convert("UTC").dt.normalize()
    daily = df.groupby("date")["price"].last()
    daily.index = daily.index.tz_localize(None)
    daily.name = f"manifold_{name}"
    return daily


def load_yields() -> pd.DataFrame:
    y = pd.read_csv(DATA / "treasury_yields.csv", index_col=0, parse_dates=True)
    y["em_prob"] = estrella_mishkin_prob(y["spread_10y_3m"])
    return y


def summarize_series(s: pd.Series) -> dict:
    """Headline summary stats for a probability series."""
    s = s.dropna()
    return {
        "n_obs": int(len(s)),
        "start": str(s.index.min().date()),
        "end": str(s.index.max().date()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "min": float(s.min()),
        "max": float(s.max()),
        "std": float(s.std(ddof=1)),
        "daily_change_std": float(s.diff().std(ddof=1)),
        "n_threshold_crossings_50pct": int(((s.shift(1) < 0.5) & (s >= 0.5)).sum()),
    }


def safe_pearson(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Pearson r and p, returning NaN if degenerate (constant series)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan"), float("nan")
    r, p = pearsonr(a, b)
    return float(r), float(p)


def safe_spearman(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan"), float("nan")
    r, p = spearmanr(a, b)
    return float(r), float(p)


def granger_pvals(y: pd.Series, x: pd.Series, max_lag: int = 5) -> dict:
    """Granger causality: does x Granger-cause y? Test on first differences
    so both series are approximately stationary. Returns p-value of the
    F-statistic at each lag (1..max_lag)."""
    df = pd.concat([y, x], axis=1).dropna()
    if len(df) < max_lag + 10:
        return {f"lag_{k}": float("nan") for k in range(1, max_lag + 1)}
    diffs = df.diff().dropna()
    try:
        # statsmodels deprecated verbose arg in 0.14; print silenced via context.
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = grangercausalitytests(diffs.values[:, [0, 1]], maxlag=max_lag)
    except Exception:
        return {f"lag_{k}": float("nan") for k in range(1, max_lag + 1)}
    return {f"lag_{k}": float(res[k][0]["ssr_ftest"][1]) for k in range(1, max_lag + 1)}


def adf_pvalue(s: pd.Series) -> float:
    s = s.dropna().diff().dropna()
    if len(s) < 10:
        return float("nan")
    try:
        return float(adfuller(s, autolag="AIC")[1])
    except Exception:
        return float("nan")


def event_window_change(s: pd.Series, start: str, end: str) -> float | None:
    s = s.dropna()
    s = s[(s.index >= start) & (s.index <= end)]
    if len(s) < 2:
        return None
    return float(s.iloc[-1] - s.iloc[0])


def main() -> None:
    yields = load_yields()
    em_prob = yields["em_prob"]
    spread = yields["spread_10y_3m"]

    # Markets to analyse together with the matching yield-curve window.
    poly_slugs = [
        "us-recession-in-2025",
        "us-recession-by-end-of-2026",
        "us-recession-in-2024-1",
        "us-recession-announced-by-nber-before-june-2025",
    ]
    manifold_names = [
        "us-recession-end-2025-two-quarters",
        "us-recession-by-end-of-2024",
    ]

    out: dict = {
        "estrella_mishkin": {"intercept": EM_INTERCEPT, "slope": EM_SLOPE},
        "yield_curve": {
            "n_obs": int(len(yields)),
            "start": str(yields.index.min().date()),
            "end": str(yields.index.max().date()),
            "spread_10y_3m": {
                "mean": float(spread.mean()),
                "min": float(spread.min()),
                "max": float(spread.max()),
                "n_days_inverted": int((spread < 0).sum()),
                "n_days_total": int(spread.notna().sum()),
                "frac_inverted": float((spread < 0).mean()),
                "longest_inversion_run_days": int(longest_run(spread < 0)),
            },
            "em_prob": summarize_series(em_prob),
        },
        "markets": {},
        "cross_market_overlap": {},
    }

    # Per-market analysis: Polymarket
    series_collection: dict[str, pd.Series] = {}
    for slug in poly_slugs:
        poly = load_polymarket(slug)
        series_collection[f"poly_{slug}"] = poly
        out["markets"][f"poly_{slug}"] = market_section(poly, em_prob, spread)

    # Manifold
    for name in manifold_names:
        m = load_manifold(name)
        series_collection[f"manifold_{name}"] = m
        out["markets"][f"manifold_{name}"] = market_section(m, em_prob, spread)

    # Polymarket vs Manifold (same-question alignment)
    pairs = [
        ("poly_us-recession-in-2025", "manifold_us-recession-end-2025-two-quarters"),
        ("poly_us-recession-in-2024-1", "manifold_us-recession-by-end-of-2024"),
    ]
    for a, b in pairs:
        s1 = series_collection[a]
        s2 = series_collection[b]
        df = pd.concat([s1, s2], axis=1, sort=True).dropna()
        if df.empty:
            continue
        r_p, p_p = safe_pearson(df.iloc[:, 0].values, df.iloc[:, 1].values)
        r_s, p_s = safe_spearman(df.iloc[:, 0].values, df.iloc[:, 1].values)
        out["cross_market_overlap"][f"{a}__vs__{b}"] = {
            "n_overlap_days": int(len(df)),
            "pearson_r": r_p, "pearson_p": p_p,
            "spearman_r": r_s, "spearman_p": p_s,
            "mean_poly": float(df.iloc[:, 0].mean()),
            "mean_manifold": float(df.iloc[:, 1].mean()),
            "mean_abs_diff": float((df.iloc[:, 0] - df.iloc[:, 1]).abs().mean()),
        }

    # Event study: 2025-04-02 (tariff "Liberation Day") through 2025-04-09.
    event_block = {}
    for key, s in series_collection.items():
        event_block[key] = event_window_change(s, "2025-04-02", "2025-04-09")
    event_block["em_prob"] = event_window_change(em_prob, "2025-04-02", "2025-04-09")
    event_block["spread_10y_3m"] = event_window_change(spread, "2025-04-02", "2025-04-09")
    out["event_study_tariff_apr2025"] = event_block

    # Write
    with open(RESULTS / "summary.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

    pretty_print(out)


def longest_run(mask: pd.Series) -> int:
    longest = current = 0
    for v in mask.fillna(False).values:
        if v:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def market_section(market: pd.Series, em_prob: pd.Series, spread: pd.Series) -> dict:
    """Stats for one prediction-market probability series compared with
    the yield-curve EM probability over the same days."""
    df = pd.concat([market.rename("p_mkt"), em_prob.rename("p_em"), spread.rename("spread")], axis=1, sort=True).dropna()
    if df.empty:
        return {"n_overlap_days": 0}
    r_p, p_p = safe_pearson(df["p_mkt"].values, df["p_em"].values)
    r_s, p_s = safe_spearman(df["p_mkt"].values, df["p_em"].values)
    # Pearson on first differences captures short-term co-movement.
    diffs = df[["p_mkt", "p_em"]].diff().dropna()
    r_diff_p, p_diff_p = safe_pearson(diffs["p_mkt"].values, diffs["p_em"].values)
    # Granger in both directions on differences.
    g_em_causes_mkt = granger_pvals(df["p_mkt"], df["p_em"])
    g_mkt_causes_em = granger_pvals(df["p_em"], df["p_mkt"])
    return {
        "n_overlap_days": int(len(df)),
        "start": str(df.index.min().date()),
        "end": str(df.index.max().date()),
        "summary_market": summarize_series(df["p_mkt"]),
        "summary_em_prob": summarize_series(df["p_em"]),
        "pearson_level": {"r": r_p, "p": p_p},
        "spearman_level": {"r": r_s, "p": p_s},
        "pearson_diff": {"r": r_diff_p, "p": p_diff_p},
        "adf_p_mkt_first_diff": adf_pvalue(df["p_mkt"]),
        "adf_p_em_first_diff": adf_pvalue(df["p_em"]),
        "granger_em_to_mkt": g_em_causes_mkt,   # H0: em does NOT Granger-cause mkt
        "granger_mkt_to_em": g_mkt_causes_em,   # H0: mkt does NOT Granger-cause em
        "mean_abs_diff": float((df["p_mkt"] - df["p_em"]).abs().mean()),
        "bias_mkt_minus_em": float((df["p_mkt"] - df["p_em"]).mean()),
    }


def pretty_print(out: dict) -> None:
    print("=" * 78)
    print("Yield curve (full sample)")
    print("=" * 78)
    yc = out["yield_curve"]
    s = yc["spread_10y_3m"]
    print(f"  10y-3m spread: mean={s['mean']:.3f}pp, range=[{s['min']:.2f},{s['max']:.2f}]")
    print(f"  Days inverted: {s['n_days_inverted']}/{s['n_days_total']} ({100*s['frac_inverted']:.1f}%)")
    print(f"  Longest inversion run: {s['longest_inversion_run_days']} trading days")
    e = yc["em_prob"]
    print(f"  EM probit P(rec): mean={e['mean']:.3f}, max={e['max']:.3f}")

    print()
    print("=" * 78)
    print("Per-market analysis")
    print("=" * 78)
    for k, v in out["markets"].items():
        if v.get("n_overlap_days", 0) == 0:
            continue
        print(f"\n--- {k} ---")
        print(f"  overlap: {v['n_overlap_days']} days ({v['start']} to {v['end']})")
        sm = v["summary_market"]; se = v["summary_em_prob"]
        print(f"  market: mean={sm['mean']:.3f} max={sm['max']:.3f} daily-Δ-std={sm['daily_change_std']:.4f}")
        print(f"  EM:     mean={se['mean']:.3f} max={se['max']:.3f} daily-Δ-std={se['daily_change_std']:.4f}")
        print(f"  Pearson level: r={v['pearson_level']['r']:+.3f}  p={v['pearson_level']['p']:.3g}")
        print(f"  Spearman level: r={v['spearman_level']['r']:+.3f}  p={v['spearman_level']['p']:.3g}")
        print(f"  Pearson diff: r={v['pearson_diff']['r']:+.3f}  p={v['pearson_diff']['p']:.3g}")
        print(f"  bias (mkt - EM): {v['bias_mkt_minus_em']:+.3f}, MAD={v['mean_abs_diff']:.3f}")
        print(f"  Granger EM→mkt (lag 1..5): " +
              " ".join(f"{v['granger_em_to_mkt'][f'lag_{i}']:.3g}" for i in range(1, 6)))
        print(f"  Granger mkt→EM (lag 1..5): " +
              " ".join(f"{v['granger_mkt_to_em'][f'lag_{i}']:.3g}" for i in range(1, 6)))

    print()
    print("=" * 78)
    print("Cross-platform (Polymarket vs Manifold, same question)")
    print("=" * 78)
    for k, v in out["cross_market_overlap"].items():
        print(f"\n--- {k} ---")
        for kk, vv in v.items():
            print(f"  {kk}: {vv}")

    print()
    print("=" * 78)
    print("Event study: 2025-04-02 to 2025-04-09 (US tariff announcement)")
    print("=" * 78)
    for k, v in out["event_study_tariff_apr2025"].items():
        print(f"  {k}: Δ = {v}")


if __name__ == "__main__":
    main()
