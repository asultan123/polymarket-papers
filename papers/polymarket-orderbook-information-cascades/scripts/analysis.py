"""Cascade / self-excitement diagnostics on Polymarket trade arrivals.

For every market in markets_sample.json we compute, from the trade timeseries:

  * Basic point-process descriptives -- count, span, intensity, coefficient of
    variation of inter-arrival times (Cox 1962), Kolmogorov-Smirnov test of
    the exponentialised inter-arrivals against the unit exponential.

  * Fano factor F(tau) = Var[N(tau)]/E[N(tau)] over a logarithmic grid of
    window lengths.  For a Hawkes process with exponential kernel and
    branching ratio eta, F(tau) -> 1/(1-eta)^2 as tau grows; we report both
    eta(tau_max) and a scale-invariant median estimate.  This follows the
    moment construction in Hardiman, Bercot & Bouchaud (2013, "Critical
    reflexivity in financial markets") and the survey in Bacry,
    Mastromatteo & Muzy (2015).

  * Trade-sign autocorrelation rho(ell) -- the canonical microstructure
    signature of herding documented by Bouchaud, Gefen, Potters & Wyart
    (2004) for equities; applied here to BUY/SELL signed indicators.

  * Cross-section: how branching ratio relates to log-volume, trade count
    and category.  Pearson correlations + scatter.

  * Calibration: 12h-fidelity midprice path; report Brier score over the
    final week and final 24h for resolved markets whose resolution is in
    `outcomePrices`.

Everything reads from data/ and writes results to results/.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RES = ROOT / "results"
FIG = ROOT / "figures"
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)


# ---------- I/O ----------


def load_markets() -> list[dict]:
    return json.loads((DATA / "markets_sample.json").read_text())


def load_trades(mid: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (timestamps, prices, sizes, signs) sorted ascending by ts.
    sign = +1 for BUY taker (taker pays ask), -1 for SELL taker."""
    path = DATA / f"trades_{mid}.csv"
    if not path.exists():
        return (np.array([]), np.array([]), np.array([]), np.array([]))
    ts, px, sz, sg = [], [], [], []
    with path.open() as f:
        for row in csv.DictReader(f):
            try:
                ts.append(int(row["timestamp"]))
                px.append(float(row["price"]))
                sz.append(float(row["size"]))
                sg.append(1 if (row["side"] or "").upper() == "BUY" else -1)
            except (ValueError, TypeError):
                continue
    order = np.argsort(ts)
    return (np.asarray(ts)[order], np.asarray(px)[order], np.asarray(sz)[order], np.asarray(sg)[order])


def load_prices12h(mid: str) -> tuple[np.ndarray, np.ndarray]:
    path = DATA / f"prices12h_{mid}.csv"
    if not path.exists():
        return (np.array([]), np.array([]))
    ts, p = [], []
    with path.open() as f:
        for row in csv.DictReader(f):
            try:
                ts.append(int(row["timestamp"]))
                p.append(float(row["price"]))
            except (ValueError, TypeError):
                continue
    return (np.asarray(ts), np.asarray(p))


# ---------- Point-process diagnostics ----------


def interarrival_stats(ts: np.ndarray) -> dict:
    """CV of inter-arrival times + KS test vs unit exponential."""
    if len(ts) < 20:
        return {"n": int(len(ts)), "cv": float("nan"), "ks_stat": float("nan"), "ks_p": float("nan")}
    dt = np.diff(ts).astype(float)
    dt = dt[dt > 0]
    if len(dt) < 10:
        return {"n": int(len(ts)), "cv": float("nan"), "ks_stat": float("nan"), "ks_p": float("nan")}
    mean = dt.mean()
    cv = dt.std(ddof=1) / mean if mean > 0 else float("nan")
    # standardize: a homogeneous Poisson would give dt/mean ~ Exp(1)
    x = dt / mean
    ks = stats.kstest(x, "expon")
    return {"n": int(len(ts)), "cv": float(cv), "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue)}


def fano_curve(ts: np.ndarray, taus: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute F(tau) = Var[N(tau)] / E[N(tau)] for each window length in taus.
    `ts` are integer-second timestamps."""
    if len(ts) < 50:
        return (taus, np.full_like(taus, np.nan, dtype=float))
    T_min, T_max = ts.min(), ts.max()
    fs = []
    for tau in taus:
        if tau <= 0:
            fs.append(np.nan)
            continue
        n_bins = max(int((T_max - T_min) // tau), 1)
        if n_bins < 4:
            fs.append(np.nan)
            continue
        edges = T_min + np.arange(n_bins + 1) * tau
        counts, _ = np.histogram(ts, bins=edges)
        if counts.mean() <= 0:
            fs.append(np.nan)
            continue
        f = counts.var(ddof=1) / counts.mean()
        fs.append(float(f))
    return (taus, np.asarray(fs))


def eta_from_fano(F: float) -> float:
    """Hawkes branching ratio satisfying F = 1/(1-eta)^2 -> eta = 1 - 1/sqrt(F).
    Returns nan when F<1 (sub-Poisson) or F is non-finite."""
    if not np.isfinite(F) or F <= 0:
        return float("nan")
    if F < 1.0:
        return 0.0  # not over-dispersed; effectively no self-excitement
    return float(1.0 - 1.0 / math.sqrt(F))


def sign_autocorr(signs: np.ndarray, max_lag: int = 100) -> np.ndarray:
    """Sample autocorrelation of the BUY/SELL sign series at lags 1..max_lag."""
    s = signs.astype(float)
    if len(s) < max_lag + 10:
        return np.full(max_lag, np.nan)
    s = s - s.mean()
    denom = (s * s).sum()
    if denom == 0:
        return np.full(max_lag, np.nan)
    out = np.empty(max_lag)
    for k in range(1, max_lag + 1):
        out[k - 1] = (s[: -k] * s[k:]).sum() / denom
    return out


def response_function(ts: np.ndarray, px: np.ndarray, sg: np.ndarray, max_lag: int = 50) -> np.ndarray:
    """E[ p_{i+l} - p_i | sg_i = +1 ] - E[ p_{i+l} - p_i | sg_i = -1 ].
    A persistent positive response indicates information-bearing flow rather
    than mean-reversion (Bouchaud et al. 2004)."""
    if len(px) < max_lag + 10:
        return np.full(max_lag, np.nan)
    out = np.empty(max_lag)
    for ell in range(1, max_lag + 1):
        dp = px[ell:] - px[: -ell]
        s = sg[: -ell]
        m_plus = dp[s > 0].mean() if (s > 0).any() else np.nan
        m_minus = dp[s < 0].mean() if (s < 0).any() else np.nan
        out[ell - 1] = m_plus - m_minus if np.isfinite(m_plus) and np.isfinite(m_minus) else np.nan
    return out


# ---------- Calibration ----------


def parse_outcome_prices(s: str | None) -> tuple[float, float] | None:
    if not s:
        return None
    try:
        v = json.loads(s)
        a, b = float(v[0]), float(v[1])
        return (a, b)
    except Exception:  # noqa: BLE001
        return None


def brier_for_horizon(ts: np.ndarray, p: np.ndarray, resolution: float, horizon_h: float) -> float:
    """Mean Brier score over the last `horizon_h` hours of midprice path."""
    if len(ts) == 0:
        return float("nan")
    cut = ts.max() - int(horizon_h * 3600)
    mask = ts >= cut
    if mask.sum() < 2:
        return float("nan")
    p_h = p[mask]
    return float(np.mean((p_h - resolution) ** 2))


# ---------- Main pipeline ----------


def classify(question: str) -> str:
    q = question.lower()
    election_kw = ("election", "president", "vp", "vice president", "mayor", "popular vote", "candidate", "haley", "rfk", "kennedy", "obama", "harris", "trump be inaugurated", "republican politician", "democratic politician")
    sports_kw = ("super bowl", "nba", "champions league", "uefa", "finals", "wizards", "raiders", "panthers", "titans", "giants", "browns", "hornets", "jazz", "raptors", "kings", "stanley cup", "world series", "world cup", "nhl", "mls")
    macro_kw = ("fed", "interest rate", "inflation", "cpi", "ppi", "recession", "gdp", "rate cut", "rate hike", "bps", "tariff")
    geo_kw = ("ceasefire", "iran", "ukraine", "russia", "israel", "gaza", "shutdown", "khamenei", "zelenskyy", "tiktok", "kim jong")
    if any(k in q for k in election_kw):
        return "election"
    if any(k in q for k in sports_kw):
        return "sports"
    if any(k in q for k in macro_kw):
        return "macro"
    if any(k in q for k in geo_kw):
        return "geopolitics"
    return "other"


def burst_count(ts: np.ndarray, window_s: float = 60.0, k_sigma: float = 4.0) -> int:
    """Count windows whose trade count is > mean + k_sigma * sqrt(mean), i.e.
    rate-spikes far above a homogeneous-Poisson expectation."""
    if len(ts) < 50:
        return 0
    T_min, T_max = ts.min(), ts.max()
    n_bins = max(int((T_max - T_min) // window_s), 1)
    if n_bins < 5:
        return 0
    edges = T_min + np.arange(n_bins + 1) * window_s
    counts, _ = np.histogram(ts, bins=edges)
    mu = counts.mean()
    if mu <= 0:
        return 0
    threshold = mu + k_sigma * math.sqrt(mu)
    return int((counts > threshold).sum())


def analyse_one(m: dict) -> dict:
    mid = m["id"]
    ts, px, sz, sg = load_trades(mid)
    pts, pp = load_prices12h(mid)

    out: dict = {
        "id": mid,
        "question": m["question"],
        "category": classify(m["question"]),
        "volume_usd": m.get("volume_usd") or 0,
        "n_trades": int(len(ts)),
    }
    out["n_bursts_60s_4sigma"] = burst_count(ts, window_s=60.0, k_sigma=4.0)
    out["n_bursts_60s_6sigma"] = burst_count(ts, window_s=60.0, k_sigma=6.0)
    out["span_h"] = float((ts.max() - ts.min()) / 3600.0) if len(ts) > 1 else 0.0
    out["intensity_per_min"] = float(len(ts) / (out["span_h"] * 60.0)) if out["span_h"] > 0 else float("nan")

    ia = interarrival_stats(ts)
    out.update({"cv_dt": ia["cv"], "ks_stat": ia["ks_stat"], "ks_p": ia["ks_p"]})

    # Fano on a log grid of window sizes from 5s to 1h (capped at span/10)
    if out["span_h"] >= 0.1:
        max_tau = max(60.0, out["span_h"] * 3600.0 / 10.0)
        taus = np.unique(np.round(np.geomspace(5.0, max_tau, 24)).astype(int))
        _, F = fano_curve(ts, taus)
        # eta at the largest tau where F is finite
        F_finite = F[np.isfinite(F)]
        out["fano_max"] = float(F_finite.max()) if F_finite.size else float("nan")
        out["fano_med"] = float(np.nanmedian(F)) if F_finite.size else float("nan")
        out["eta_hat"] = eta_from_fano(out["fano_max"])
        # store the full curve for plotting later
        out["_fano_taus"] = taus.tolist()
        out["_fano_F"] = F.tolist()
    else:
        out["fano_max"] = float("nan")
        out["fano_med"] = float("nan")
        out["eta_hat"] = float("nan")
        out["_fano_taus"] = []
        out["_fano_F"] = []

    # sign autocorrelation
    rho = sign_autocorr(sg, max_lag=100)
    out["rho_1"] = float(rho[0]) if rho.size else float("nan")
    out["rho_5"] = float(rho[4]) if rho.size > 4 else float("nan")
    out["rho_25"] = float(rho[24]) if rho.size > 24 else float("nan")
    out["_rho"] = rho.tolist()

    # response function
    R = response_function(ts, px, sg, max_lag=50)
    out["_R"] = R.tolist()
    out["R_1"] = float(R[0]) if R.size else float("nan")
    out["R_10"] = float(R[9]) if R.size > 9 else float("nan")
    out["R_50"] = float(R[49]) if R.size > 49 else float("nan")

    # calibration: outcomePrices is [yes_price, no_price] for a binary market; 1/0 after resolution
    op = parse_outcome_prices(m.get("outcome_prices"))
    if op is not None and len(pts) > 5:
        resolution = float(op[0])  # YES probability after resolution
        out["resolution_yes"] = resolution
        out["brier_24h"] = brier_for_horizon(pts, pp, resolution, 24.0)
        out["brier_1wk"] = brier_for_horizon(pts, pp, resolution, 24.0 * 7)
        out["brier_1mo"] = brier_for_horizon(pts, pp, resolution, 24.0 * 30)
    else:
        out["resolution_yes"] = float("nan")
        out["brier_24h"] = float("nan")
        out["brier_1wk"] = float("nan")
        out["brier_1mo"] = float("nan")

    return out


def cross_section(results: list[dict]) -> dict:
    """Aggregate across markets: correlations and group means."""
    rows = [r for r in results if r["n_trades"] >= 200 and np.isfinite(r.get("eta_hat", float("nan")))]
    if not rows:
        return {"n_used": 0}
    eta = np.array([r["eta_hat"] for r in rows])
    vol = np.array([r["volume_usd"] for r in rows])
    nt = np.array([r["n_trades"] for r in rows])
    intens = np.array([r["intensity_per_min"] for r in rows])
    rho1 = np.array([r["rho_1"] for r in rows])

    def corr(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 5:
            return (float("nan"), float("nan"))
        c = stats.pearsonr(a[m], b[m])
        return (float(c[0]), float(c[1]))

    # Spearman corrs are more robust for skewed volume
    def scorr(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 5:
            return (float("nan"), float("nan"))
        c = stats.spearmanr(a[m], b[m])
        return (float(c.statistic), float(c.pvalue))

    summary = {
        "n_used": len(rows),
        "eta_mean": float(np.nanmean(eta)),
        "eta_median": float(np.nanmedian(eta)),
        "eta_p25": float(np.nanpercentile(eta, 25)),
        "eta_p75": float(np.nanpercentile(eta, 75)),
        "rho1_mean": float(np.nanmean(rho1)),
        "rho1_median": float(np.nanmedian(rho1)),
        "corr_eta_log_vol_pearson": corr(eta, np.log10(vol + 1)),
        "corr_eta_log_intensity_pearson": corr(eta, np.log10(intens + 1e-6)),
        "corr_eta_log_vol_spearman": scorr(eta, np.log10(vol + 1)),
        "corr_eta_log_intensity_spearman": scorr(eta, np.log10(intens + 1e-6)),
    }

    # By category
    by_cat: dict = defaultdict(list)
    for r in rows:
        by_cat[r["category"] or "(uncategorized)"].append(r["eta_hat"])
    summary["by_category"] = {
        k: {"n": len(v), "eta_mean": float(np.mean(v)), "eta_median": float(np.median(v))}
        for k, v in by_cat.items()
    }
    return summary


def write_results(results: list[dict], summary: dict) -> None:
    # market-level CSV (strip the heavy per-curve arrays)
    keys = [
        "id", "question", "category", "volume_usd", "n_trades", "span_h", "intensity_per_min",
        "n_bursts_60s_4sigma", "n_bursts_60s_6sigma",
        "cv_dt", "ks_stat", "ks_p",
        "fano_max", "fano_med", "eta_hat",
        "rho_1", "rho_5", "rho_25",
        "R_1", "R_10", "R_50",
        "resolution_yes", "brier_24h", "brier_1wk", "brier_1mo",
    ]
    with (RES / "per_market.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k) for k in keys})

    # full per-market dump for figure-making (kept JSON; <1MB)
    light = []
    for r in results:
        light.append({k: r[k] for k in r if not k.startswith("_") or k in ("_fano_taus", "_fano_F", "_rho", "_R")})
    (RES / "per_market_full.json").write_text(json.dumps(light, indent=1))

    (RES / "cross_section.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    markets = load_markets()
    print(f"Analysing {len(markets)} markets ...")
    results = []
    for i, m in enumerate(markets, 1):
        try:
            r = analyse_one(m)
        except Exception as e:  # noqa: BLE001
            print(f"[{i}/{len(markets)}] {m['id']} FAILED: {e}")
            continue
        print(
            f"[{i}/{len(markets)}] n={r['n_trades']:>5}  "
            f"span={r['span_h']:>6.1f}h  "
            f"eta_hat={r['eta_hat']:>.3f}  rho1={r['rho_1']:>+.3f}  "
            f"{r['question'][:70]}"
        )
        results.append(r)
    summary = cross_section(results)
    write_results(results, summary)
    print("\nCross-section summary:")
    for k, v in summary.items():
        if k == "by_category":
            print("  by_category:")
            for cat, d in sorted(v.items(), key=lambda kv: -kv[1]["n"])[:10]:
                print(f"    {cat:>30}  n={d['n']:>3}  eta_mean={d['eta_mean']:.3f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
