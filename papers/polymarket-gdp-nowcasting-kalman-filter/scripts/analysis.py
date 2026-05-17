"""Build the Polymarket-implied GDP nowcast, fit a Kalman filter, and emit all
numbers that go into the paper.

The script consumes the artefacts written by ``fetch_data.py`` and writes:

- ``analysis_outputs/results.json``  -- every number cited in the paper
- ``figure_panel.png`` -- 3-quarter daily nowcast (Q1 2025, Q4 2025, Q1 2026)
- ``figure_recession.png`` -- US-recession-2025 market vs. yield-curve
- ``figure_correlations.png`` -- Polymarket-implied GDP vs. financial signals
- ``figure_kalman.png`` -- raw Polymarket vs. Kalman-filtered nowcast

Every number reported in the paper must be replayable from these scripts.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PRICES = DATA / "polymarket_prices"
ANALYSIS = ROOT / "analysis_outputs"
ANALYSIS.mkdir(exist_ok=True)

# Bucket midpoints (annualized real-GDP-growth %).
Q1_2025_MIDS = {
    "will-us-gdp-growth-be-greater-than-2-in-q1-2025": 3.0,
    "will-us-gdp-growth-be-between-2-and-1-in-q1-2025": 1.5,
    "will-us-gdp-growth-be-between-1-and-0-in-q1-2025": 0.5,
    "will-us-gdp-growth-be-between-0-and-1-in-q1-2025": -0.5,
    "will-us-gdp-growth-be-between-1-and-2-in-q1-2025": -1.5,
    "will-us-gdp-growth-be-less-than-2-in-q1-2025": -2.5,
}
Q4_2025_MIDS = {
    "will-us-gdp-growth-in-q4-2025-be-less-than-1pt0": 0.5,
    "will-us-gdp-growth-in-q4-2025-be-between-1pt0-and-1pt5": 1.25,
    "will-us-gdp-growth-in-q4-2025-be-between-1pt5-and-2pt0": 1.75,
    "will-us-gdp-growth-in-q4-2025-be-between-2pt0-and-2pt5": 2.25,
    "will-us-gdp-growth-in-q4-2025-be-between-2pt5-and-3pt0": 2.75,
    "will-us-gdp-growth-in-q4-2025-be-between-3pt0-and-3pt5": 3.25,
    "will-us-gdp-growth-in-q4-2025-be-greater-than-3pt5": 4.0,
}
Q1_2026_MIDS = {
    "will-us-gdp-growth-in-q1-2026-be-less-than-1pt0": 0.5,
    "will-us-gdp-growth-in-q1-2026-be-between-1pt0-and-1pt5": 1.25,
    "will-us-gdp-growth-in-q1-2026-be-between-1pt5-and-2pt0": 1.75,
    "will-us-gdp-growth-in-q1-2026-be-between-2pt0-and-2pt5": 2.25,
    "will-us-gdp-growth-in-q1-2026-be-between-2pt5-and-3pt0": 2.75,
    "will-us-gdp-growth-in-q1-2026-be-between-3pt0-and-3pt5": 3.25,
    "will-us-gdp-growth-in-q1-2026-be-greater-than-3pt5": 4.0,
}

# Realized GDP (midpoint of bucket whose YES resolved to 1 per gamma API).  All
# are annualized real-GDP-growth %, BEA "Advance Estimate" basis.
REALIZED = {"Q1_2025": -0.5, "Q4_2025": 1.25, "Q1_2026": 2.25}

# BEA Advance Estimate release dates, taken from each market's description text.
# Q4 2025's release was delayed past its scheduled 2026-01-29 date; the market
# closedTime (used as the effective resolution timestamp) is 2026-02-20.
SCHEDULED_RELEASE = {
    "Q1_2025": pd.Timestamp("2025-04-30"),
    "Q4_2025": pd.Timestamp("2026-01-29"),
    "Q1_2026": pd.Timestamp("2026-04-30"),
}
EFFECTIVE_RESOLUTION = {  # actual market close (from gamma closedTime)
    "Q1_2025": pd.Timestamp("2025-04-30"),
    "Q4_2025": pd.Timestamp("2026-02-20"),
    "Q1_2026": pd.Timestamp("2026-04-30"),
}
QUARTER_START = {
    "Q1_2025": pd.Timestamp("2025-01-01"),
    "Q4_2025": pd.Timestamp("2025-10-01"),
    "Q1_2026": pd.Timestamp("2026-01-01"),
}


def load_bucket_panel(mids: dict[str, float]) -> pd.DataFrame:
    """Wide DataFrame indexed by date with the daily YES-price for each bucket."""
    panels = []
    for slug in mids:
        path = PRICES / f"{slug}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, parse_dates=["date"]).set_index("date")["price"]
        panels.append(df.rename(slug))
    panel = pd.concat(panels, axis=1, sort=True).sort_index()
    return panel


def implied_moments(panel: pd.DataFrame, mids: dict[str, float]) -> pd.DataFrame:
    """Daily renormalized expected GDP and its std (across buckets)."""
    use = panel.copy().clip(lower=0)
    use = use.fillna(0.0)
    s = use.sum(axis=1)
    use_norm = use.divide(s, axis=0)
    mid = np.array([mids[c] for c in use_norm.columns])
    e = use_norm.values @ mid
    var = use_norm.values @ (mid ** 2) - e ** 2
    out = pd.DataFrame({"E_gdp": e, "SD_gdp": np.sqrt(np.maximum(var, 0)),
                         "raw_sum": s.values},
                        index=use_norm.index)
    return out


def fin_signals(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Daily financial signals used as alternative observables."""
    yf = pd.read_csv(DATA / "yfinance.csv", parse_dates=["date"]).set_index("date")
    yf = yf.loc[(yf.index >= start - pd.Timedelta(days=120)) & (yf.index <= end)]
    sig = pd.DataFrame(index=yf.index)
    sig["sp500_ret90"] = yf["SP500"].pct_change(periods=63)
    sig["term_spread"] = yf["TNX_10Y"] - yf["TBILL_3M"]
    sig["hyg_ret30"] = yf["HYG"].pct_change(periods=21)
    sig["cyc_def"] = (yf["XLY"] / yf["XLU"]).pct_change(periods=42)
    sig = sig.loc[(sig.index >= start) & (sig.index <= end)]
    return sig


def kalman_filter(y: np.ndarray, H: np.ndarray, R: np.ndarray,
                  F: float, Q: float, x0: float, P0: float
                  ) -> tuple[np.ndarray, np.ndarray]:
    """1-D-state Kalman filter, handles NaN observations."""
    T, m = y.shape
    x_hat = np.empty(T)
    P_hat = np.empty(T)
    x_pred = x0
    P_pred = P0
    for t in range(T):
        for k in range(m):
            if not np.isnan(y[t, k]):
                S = H[k] * P_pred * H[k] + R[k]
                if S <= 0:
                    continue
                K = P_pred * H[k] / S
                x_pred = x_pred + K * (y[t, k] - H[k] * x_pred)
                P_pred = (1.0 - K * H[k]) * P_pred
        x_hat[t] = x_pred
        P_hat[t] = max(P_pred, 1e-8)
        x_pred = F * x_pred
        P_pred = F * P_pred * F + Q
    return x_hat, P_hat


def calibrate_HR(truth_per_q: dict[str, float],
                 polym_E_per_q: dict[str, pd.Series],
                 fin_per_q: dict[str, pd.DataFrame]
                 ) -> tuple[dict[str, float], dict[str, float]]:
    rows = []
    for q, truth in truth_per_q.items():
        e = polym_E_per_q[q].rename("polym_E")
        sig = fin_per_q[q]
        df = pd.concat([e, sig], axis=1, sort=True).dropna(how="all")
        df["target"] = truth
        rows.append(df.reset_index(drop=True))
    big = pd.concat(rows, ignore_index=True)
    H, R = {}, {}
    for col in ["polym_E", "sp500_ret90", "term_spread", "hyg_ret30", "cyc_def"]:
        sub = big[[col, "target"]].dropna()
        if len(sub) < 5:
            continue
        x = sub[col].values
        t = sub["target"].values
        # observable = H * target + e  (with intercept absorbed into mean).
        # We center by mean to be robust.
        tc = t - t.mean()
        xc = x - x.mean()
        denom = (tc ** 2).sum()
        if denom < 1e-9:
            continue
        H_k = float((tc * xc).sum() / denom)
        if abs(H_k) < 1e-6:
            continue
        resid = x - (H_k * t)
        H[col] = H_k
        R[col] = float(max(np.var(resid), 1e-4))
    return H, R


def main() -> None:
    results: dict[str, Any] = {"meta": {
        "as_of": pd.Timestamp.utcnow().isoformat(),
        "realized": REALIZED,
        "scheduled_release": {k: v.date().isoformat() for k, v in SCHEDULED_RELEASE.items()},
        "effective_resolution": {k: v.date().isoformat() for k, v in EFFECTIVE_RESOLUTION.items()},
    }, "quarters": {}}

    polym_E: dict[str, pd.Series] = {}
    fin_per_q: dict[str, pd.DataFrame] = {}

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), sharey=False)
    qs = [("Q1_2025", Q1_2025_MIDS), ("Q4_2025", Q4_2025_MIDS),
          ("Q1_2026", Q1_2026_MIDS)]

    for ax, (q, mids) in zip(axes, qs):
        panel = load_bucket_panel(mids).dropna(how="all")
        # Restrict to the quarter window (from quarter start through scheduled release).
        panel_q = panel.loc[(panel.index >= QUARTER_START[q]) &
                              (panel.index <= EFFECTIVE_RESOLUTION[q])]
        moments = implied_moments(panel_q, mids)
        truth = REALIZED[q]
        sched = SCHEDULED_RELEASE[q]
        # Restrict signals to overlapping window
        fin = fin_signals(panel_q.index.min(), panel_q.index.max())

        # Track per-quarter quantities
        meta = {
            "n_days": int(len(moments)),
            "first_date": moments.index.min().date().isoformat(),
            "last_date": moments.index.max().date().isoformat(),
            "scheduled_release": sched.date().isoformat(),
            "effective_resolution": EFFECTIVE_RESOLUTION[q].date().isoformat(),
            "realized_gdp_pct": float(truth),
            "polym_first_E": float(moments["E_gdp"].iloc[0]),
            "polym_mean_E": float(moments["E_gdp"].mean()),
            "polym_min_E": float(moments["E_gdp"].min()),
            "polym_max_E": float(moments["E_gdp"].max()),
            "polym_mean_SD": float(moments["SD_gdp"].mean()),
            "panel_raw_sum_mean": float(moments["raw_sum"].mean()),
        }
        # Lead-time errors (relative to scheduled release date)
        for lead in [60, 30, 14, 7, 1]:
            target_date = sched - pd.Timedelta(days=lead)
            sub = moments[moments.index <= target_date]
            if len(sub):
                v = float(sub["E_gdp"].iloc[-1])
                meta[f"E_T_minus_{lead}"] = v
                meta[f"abs_err_T_minus_{lead}"] = float(abs(v - truth))
        results["quarters"][q] = meta
        polym_E[q] = moments["E_gdp"]
        fin_per_q[q] = fin

        # Plot
        ax.fill_between(moments.index,
                        moments["E_gdp"] - moments["SD_gdp"],
                        moments["E_gdp"] + moments["SD_gdp"],
                        alpha=0.18, color="#1f77b4", label="±1σ implied")
        ax.plot(moments.index, moments["E_gdp"], color="#1f77b4", linewidth=1.6,
                label="Polymarket E[GDP]")
        ax.axhline(truth, color="#d62728", linestyle="--", linewidth=1.4,
                   label=f"BEA realized {truth:+.2f}%")
        ax.axvline(sched, color="grey", linestyle=":", alpha=0.8,
                   label="scheduled release")
        if EFFECTIVE_RESOLUTION[q] != sched:
            ax.axvline(EFFECTIVE_RESOLUTION[q], color="purple",
                       linestyle=":", alpha=0.7, label="effective resolution")
        ax.set_title(q.replace("_", " "))
        ax.set_ylabel("Real GDP growth (annualized %)")
        ax.set_xlabel("")
        for label in ax.get_xticklabels():
            label.set_rotation(25)
        ax.legend(loc="best", fontsize=7)

    fig.suptitle("Polymarket-implied US real GDP nowcasts (daily)", y=1.04)
    fig.tight_layout()
    fig.savefig(ROOT / "figure_panel.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # -------------------- calibrate H, R --------------------
    H, R = calibrate_HR(REALIZED, polym_E, fin_per_q)
    results["loadings_H"] = H
    results["obs_noise_R"] = R

    # -------------------- Kalman fusion --------------------
    fig_k, axes_k = plt.subplots(1, 3, figsize=(13.5, 3.8))
    kalman_results: dict[str, dict[str, Any]] = {}
    for ax, (q, _) in zip(axes_k, qs):
        e = polym_E[q]
        sig = fin_per_q[q]
        idx = e.index.union(sig.index).sort_values()
        e_full = e.reindex(idx)
        sig_full = sig.reindex(idx)
        cols = ["polym_E", "sp500_ret90", "term_spread", "hyg_ret30", "cyc_def"]
        # Only include columns we managed to calibrate
        cols = [c for c in cols if c in H]
        if not cols:
            continue
        Y = np.column_stack([
            e_full.values if "polym_E" in cols else None,
            *[sig_full[c].values for c in cols if c != "polym_E"]
        ]) if "polym_E" in cols else np.column_stack([sig_full[c].values for c in cols])
        # Re-build Y robustly
        Ycols = []
        for c in cols:
            if c == "polym_E":
                Ycols.append(e_full.values)
            else:
                Ycols.append(sig_full[c].values)
        Y = np.column_stack(Ycols)
        H_vec = np.array([H[c] for c in cols])
        R_vec = np.array([R[c] for c in cols])
        Q_var = float(np.var(list(REALIZED.values())))
        Q_daily = Q_var / 63.0
        x0 = float(np.mean(list(REALIZED.values())))
        P0 = Q_var
        x_hat, P_hat = kalman_filter(Y, H_vec, R_vec, 1.0, Q_daily, x0, P0)
        truth = REALIZED[q]
        sched = SCHEDULED_RELEASE[q]
        df_k = pd.DataFrame({"x": x_hat, "P": P_hat}, index=idx)
        meta = {}
        for lead in [60, 30, 14, 7, 1]:
            tgt = sched - pd.Timedelta(days=lead)
            sub = df_k[df_k.index <= tgt]
            if len(sub):
                meta[f"x_T_minus_{lead}"] = float(sub["x"].iloc[-1])
                meta[f"abs_err_T_minus_{lead}"] = float(abs(sub["x"].iloc[-1] - truth))
        kalman_results[q] = meta

        # Plot Kalman vs raw Polymarket
        ax.plot(idx, x_hat, color="#9467bd", linewidth=1.7, label="Kalman fused")
        ax.plot(idx, e_full.values, color="#1f77b4", linewidth=1.0, alpha=0.7,
                 label="Raw Polymarket E[GDP]")
        ax.axhline(truth, color="#d62728", linestyle="--", linewidth=1.4,
                    label=f"BEA realized {truth:+.2f}%")
        ax.axvline(sched, color="grey", linestyle=":", alpha=0.7)
        ax.set_title(q.replace("_", " "))
        ax.set_ylabel("Real GDP growth (annualized %)")
        ax.legend(loc="best", fontsize=7)
        for label in ax.get_xticklabels():
            label.set_rotation(25)

    fig_k.suptitle("Kalman-fused nowcast vs. raw Polymarket implied expectation",
                    y=1.04)
    fig_k.tight_layout()
    fig_k.savefig(ROOT / "figure_kalman.png", dpi=160, bbox_inches="tight")
    plt.close(fig_k)

    results["kalman"] = kalman_results

    # -------------------- Financial-only baseline --------------------
    fin_only: dict[str, dict[str, Any]] = {}
    for q, _ in qs:
        sig = fin_per_q[q]
        cols = [c for c in ["sp500_ret90", "term_spread", "hyg_ret30", "cyc_def"]
                if c in H]
        if not cols:
            continue
        Y = np.column_stack([sig[c].values for c in cols])
        H_vec = np.array([H[c] for c in cols])
        R_vec = np.array([R[c] for c in cols])
        Q_var = float(np.var(list(REALIZED.values())))
        Q_daily = Q_var / 63.0
        x_hat, _ = kalman_filter(Y, H_vec, R_vec, 1.0, Q_daily,
                                  float(np.mean(list(REALIZED.values()))), Q_var)
        idx = sig.index
        df_f = pd.DataFrame({"x": x_hat}, index=idx)
        truth = REALIZED[q]
        meta = {}
        for lead in [60, 30, 14, 7, 1]:
            tgt = SCHEDULED_RELEASE[q] - pd.Timedelta(days=lead)
            sub = df_f[df_f.index <= tgt]
            if len(sub):
                meta[f"x_T_minus_{lead}"] = float(sub["x"].iloc[-1])
                meta[f"abs_err_T_minus_{lead}"] = float(abs(sub["x"].iloc[-1] - truth))
        fin_only[q] = meta
    results["financial_only"] = fin_only

    # -------------------- Headline metrics --------------------
    def rmse(xs):
        xs = [v for v in xs if v is not None]
        if not xs:
            return None
        return float(math.sqrt(np.mean(np.square(xs))))

    def mae(xs):
        xs = [v for v in xs if v is not None]
        if not xs:
            return None
        return float(np.mean(np.abs(xs)))

    headline = {}
    for lead in [60, 30, 14, 7, 1]:
        polym_e = [results["quarters"][q].get(f"abs_err_T_minus_{lead}") for q in REALIZED]
        kalman_e = [kalman_results.get(q, {}).get(f"abs_err_T_minus_{lead}") for q in REALIZED]
        fin_e = [fin_only.get(q, {}).get(f"abs_err_T_minus_{lead}") for q in REALIZED]
        headline[f"T_minus_{lead}"] = {
            "polymarket_RMSE": rmse(polym_e),
            "polymarket_MAE": mae(polym_e),
            "kalman_RMSE": rmse(kalman_e),
            "kalman_MAE": mae(kalman_e),
            "financial_only_RMSE": rmse(fin_e),
            "financial_only_MAE": mae(fin_e),
            "polymarket_errors_per_q": polym_e,
            "kalman_errors_per_q": kalman_e,
            "financial_only_errors_per_q": fin_e,
        }
    results["headline"] = headline

    # Pearson correlation at T-30 across 3 quarters (tiny N, but real)
    ts = []
    pe = []
    for q in REALIZED:
        v = results["quarters"][q].get("E_T_minus_30")
        if v is not None and np.isfinite(v):
            pe.append(v)
            ts.append(REALIZED[q])
    if len(ts) >= 3:
        x = np.array(pe)
        y = np.array(ts)
        # Centred correlation
        xb = x - x.mean(); yb = y - y.mean()
        denom = math.sqrt((xb ** 2).sum() * (yb ** 2).sum())
        r = float((xb * yb).sum() / denom) if denom > 0 else float("nan")
        results["headline"]["pearson_T_minus_30_r"] = r
        results["headline"]["pearson_T_minus_30_n"] = int(len(x))
        results["headline"]["pearson_T_minus_30_polym_E"] = x.tolist()
        results["headline"]["pearson_T_minus_30_realized"] = y.tolist()

    # -------------------- Recession-market analysis --------------------
    rec25 = pd.read_csv(PRICES / "us-recession-in-2025.csv", parse_dates=["date"])\
              .set_index("date")["price"]
    rec26 = pd.read_csv(PRICES / "us-recession-by-end-of-2026.csv",
                         parse_dates=["date"]).set_index("date")["price"]

    nber_2025 = 0  # NBER did not declare any 2025 recession peak
    results["recession"] = {
        "us_rec_2025": {
            "n_days": int(len(rec25)),
            "first_date": rec25.index.min().date().isoformat(),
            "last_date": rec25.index.max().date().isoformat(),
            "mean_prob": float(rec25.mean()),
            "max_prob": float(rec25.max()),
            "max_prob_date": rec25.idxmax().date().isoformat(),
            "final_prob": float(rec25.iloc[-1]),
            "nber_realized": int(nber_2025),
            "brier_score_vs_nber": float(((rec25 - nber_2025) ** 2).mean()),
        },
        "us_rec_2026": {
            "n_days": int(len(rec26)),
            "first_date": rec26.index.min().date().isoformat(),
            "last_date": rec26.index.max().date().isoformat(),
            "mean_prob": float(rec26.mean()),
            "current_prob": float(rec26.iloc[-1]),
            "max_prob": float(rec26.max()),
            "max_prob_date": rec26.idxmax().date().isoformat(),
        },
    }

    yf_macro = pd.read_csv(DATA / "yfinance.csv", parse_dates=["date"]).set_index("date")
    spread = (yf_macro["TNX_10Y"] - yf_macro["TBILL_3M"]).rename("term_spread")
    fig2, ax2 = plt.subplots(figsize=(10, 3.8))
    ax2.plot(rec25.index, rec25 * 100, color="#1f77b4", linewidth=1.5,
              label='Polymarket: "US recession in 2025?"')
    ax2.plot(rec26.index, rec26 * 100, color="#2ca02c", linewidth=1.5,
              label='Polymarket: "US recession by end of 2026?"')
    ax2.set_ylabel("Polymarket implied probability (%)")
    ax2.set_xlabel("Date")
    ax2b = ax2.twinx()
    ax2b.plot(spread.index, spread.values, color="#ff7f0e", linewidth=1.0,
              alpha=0.7, label="10y – 3m yield spread (%)")
    ax2b.set_ylabel("Yield spread (pp)")
    ax2.legend(loc="upper left", fontsize=8)
    ax2b.legend(loc="upper right", fontsize=8)
    ax2.set_title("US-recession Polymarket prices vs. Treasury yield spread")
    fig2.tight_layout()
    fig2.savefig(ROOT / "figure_recession.png", dpi=160, bbox_inches="tight")
    plt.close(fig2)

    # Daily correlation: rec25 vs. yield spread / SP500 1mo return
    rec_yf = pd.concat([
        rec25.rename("rec25_prob"),
        spread.rename("term_spread"),
        yf_macro["HYG"].pct_change(21).rename("hyg_ret30"),
        yf_macro["SP500"].pct_change(63).rename("sp500_ret90"),
    ], axis=1, sort=True).dropna()
    if len(rec_yf) >= 30:
        corrs = rec_yf.corr().round(3).loc["rec25_prob"].to_dict()
        results["recession"]["us_rec_2025"]["daily_corr_with"] = corrs
        results["recession"]["us_rec_2025"]["n_days_for_corr"] = int(len(rec_yf))

    # -------------------- Correlations heat-map --------------------
    rows = []
    for q in REALIZED:
        sig = fin_per_q[q].copy()
        sig["polym_E"] = polym_E[q]
        rows.append(sig.dropna(how="all"))
    panel_all = pd.concat(rows)
    corr_cols = ["polym_E", "term_spread", "sp500_ret90", "hyg_ret30", "cyc_def"]
    corrs = panel_all[corr_cols].corr().round(3)
    results["correlations"] = corrs.to_dict()

    fig3, ax3 = plt.subplots(figsize=(5.2, 4.4))
    im = ax3.imshow(corrs.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax3.set_xticks(range(len(corrs.columns)))
    ax3.set_yticks(range(len(corrs.index)))
    ax3.set_xticklabels(corrs.columns, rotation=35, ha="right")
    ax3.set_yticklabels(corrs.index)
    for i in range(len(corrs.index)):
        for j in range(len(corrs.columns)):
            ax3.text(j, i, f"{corrs.values[i,j]:+.2f}",
                      ha="center", va="center",
                      color="white" if abs(corrs.values[i, j]) > 0.5 else "black",
                      fontsize=9)
    fig3.colorbar(im, ax=ax3, fraction=0.045)
    ax3.set_title(f"Daily cross-signal correlations\n(N = {len(panel_all.dropna())} pooled obs)")
    fig3.tight_layout()
    fig3.savefig(ROOT / "figure_correlations.png", dpi=160, bbox_inches="tight")
    plt.close(fig3)

    # -------------------- Persist --------------------
    with (ANALYSIS / "results.json").open("w") as f:
        json.dump(results, f, indent=2, default=str)
    print("=== headline ===")
    print(json.dumps(results["headline"], indent=2))
    print("=== per-quarter ===")
    print(json.dumps(results["quarters"], indent=2))
    print("=== recession ===")
    print(json.dumps(results["recession"], indent=2))


if __name__ == "__main__":
    main()
