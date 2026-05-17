"""Compute calibration statistics on Polymarket markets.

Inputs:
    data/markets.csv (produced by fetch_data.py)
    data/manifold.csv (optional, produced by fetch_manifold.py)

Outputs:
    data/results.json - all numbers referenced in the paper
    figure_1.png      - reliability diagram at T-1 day
    figure_2.png      - Brier vs horizon, plus volume-stratified Brier

Determinism: matplotlib uses deterministic Agg backend; np.random.seed(0)
where bootstrapping is used.
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


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-9) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability(p: np.ndarray, y: np.ndarray, n_bins: int = 10):
    """Equal-width binning. Returns bin centers, empirical freq, counts."""
    edges = np.linspace(0, 1, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    freq = np.full(n_bins, np.nan)
    count = np.zeros(n_bins, dtype=int)
    mean_p = np.full(n_bins, np.nan)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        count[i] = mask.sum()
        if count[i] > 0:
            freq[i] = y[mask].mean()
            mean_p[i] = p[mask].mean()
    return centers, mean_p, freq, count


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error: |mean(p) - mean(y)| weighted by bin counts."""
    _, mean_p, freq, count = reliability(p, y, n_bins)
    n = count.sum()
    if n == 0:
        return float("nan")
    ok = ~np.isnan(freq)
    return float(np.sum(count[ok] / n * np.abs(mean_p[ok] - freq[ok])))


def bootstrap_ci(metric_fn, p, y, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(p)
    stats_ = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats_[b] = metric_fn(p[idx], y[idx])
    lo = float(np.quantile(stats_, alpha / 2))
    hi = float(np.quantile(stats_, 1 - alpha / 2))
    return lo, hi


def reliability_slope(p, y):
    """Slope of empirical freq vs predicted prob across non-empty bins.

    Slope = 1 => perfect calibration. <1 => overconfident.
    """
    centers, mean_p, freq, count = reliability(p, y, n_bins=10)
    ok = (count >= 5) & ~np.isnan(freq)
    if ok.sum() < 3:
        return None
    res = stats.linregress(mean_p[ok], freq[ok])
    return dict(slope=float(res.slope), intercept=float(res.intercept),
                r=float(res.rvalue), p=float(res.pvalue),
                slope_se=float(res.stderr))


def all_metrics(p, y, label: str):
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    mask = ~np.isnan(p)
    p = p[mask]
    y = y[mask]
    out = dict(
        label=label,
        n=int(len(p)),
        base_rate=float(y.mean()) if len(y) else None,
        brier=brier(p, y) if len(p) else None,
        log_loss=log_loss(p, y) if len(p) else None,
        ece=ece(p, y) if len(p) else None,
    )
    if len(p) >= 50:
        lo, hi = bootstrap_ci(brier, p, y)
        out["brier_ci"] = [lo, hi]
        lo, hi = bootstrap_ci(log_loss, p, y)
        out["log_loss_ci"] = [lo, hi]
        out["slope_fit"] = reliability_slope(p, y)
    return out


def figure_reliability(df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    horizons = [("p_1d", "T - 1 day", "tab:blue"),
                ("p_7d", "T - 7 days", "tab:orange"),
                ("p_30d", "T - 30 days", "tab:green")]
    for col, lbl, c in horizons:
        sub = df[df[col].notna()]
        p = sub[col].values
        y = sub["label"].values
        _, mean_p, freq, count = reliability(p, y, n_bins=10)
        ok = count >= 5
        ax.plot(mean_p[ok], freq[ok], "o-", color=c, label=f"{lbl} (n={len(sub)})")
    ax.set_xlabel("Predicted probability (market price)")
    ax.set_ylabel("Empirical frequency of YES")
    ax.set_title("Polymarket reliability diagram")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def figure_brier_vs_horizon(df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    horizons = [("p_30d", 30), ("p_7d", 7), ("p_1d", 1)]
    # Brier vs horizon
    xs, ys, lo, hi = [], [], [], []
    for col, h in horizons:
        sub = df[df[col].notna()]
        if len(sub) < 20:
            continue
        p = sub[col].values
        y = sub["label"].values
        ys.append(brier(p, y))
        l, hi_ = bootstrap_ci(brier, p, y)
        lo.append(l); hi.append(hi_)
        xs.append(h)
    axes[0].errorbar(xs, ys, yerr=[np.array(ys)-np.array(lo), np.array(hi)-np.array(ys)],
                     fmt="o-", capsize=4)
    axes[0].set_xlabel("Days before resolution")
    axes[0].set_ylabel("Brier score")
    axes[0].set_title("Brier score by forecasting horizon")
    axes[0].invert_xaxis()
    axes[0].grid(alpha=0.3)

    # Brier vs volume bucket (using p_1d)
    sub = df[df["p_1d"].notna() & df["volume_usd"].notna()].copy()
    if len(sub) >= 50:
        edges = np.quantile(sub["volume_usd"], [0, 0.25, 0.5, 0.75, 1.0])
        buckets = []
        bx, by, blo, bhi = [], [], [], []
        for i in range(4):
            lo_v, hi_v = edges[i], edges[i+1]
            if i == 3:
                mask = (sub["volume_usd"] >= lo_v) & (sub["volume_usd"] <= hi_v)
            else:
                mask = (sub["volume_usd"] >= lo_v) & (sub["volume_usd"] < hi_v)
            if mask.sum() < 10:
                continue
            p = sub.loc[mask, "p_1d"].values
            y = sub.loc[mask, "label"].values
            b = brier(p, y)
            l, h = bootstrap_ci(brier, p, y)
            bx.append(f"Q{i+1}\nn={int(mask.sum())}")
            by.append(b); blo.append(l); bhi.append(h)
        axes[1].errorbar(range(len(bx)), by,
                         yerr=[np.array(by)-np.array(blo), np.array(bhi)-np.array(by)],
                         fmt="s-", capsize=4, color="tab:purple")
        axes[1].set_xticks(range(len(bx)))
        axes[1].set_xticklabels(bx)
        axes[1].set_xlabel("Volume quartile (low -> high)")
        axes[1].set_ylabel("Brier score at T-1d")
        axes[1].set_title("Brier vs market volume")
        axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    df = pd.read_csv(DATA / "markets.csv")
    print(f"loaded {len(df)} polymarket rows")

    # Drop any with no label
    df = df[df["label"].isin([0, 1])].copy()

    results = dict(
        polymarket=dict(
            total_markets=int(len(df)),
            base_rate=float(df["label"].mean()),
            horizons={
                "p_1d": all_metrics(df["p_1d"], df["label"], "T-1d"),
                "p_7d": all_metrics(df["p_7d"], df["label"], "T-7d"),
                "p_30d": all_metrics(df["p_30d"], df["label"], "T-30d"),
                "p_close": all_metrics(df["p_close"], df["label"], "close"),
            },
        )
    )

    # Stratified analysis: volume tertiles using p_1d
    sub = df[df["p_1d"].notna()].copy()
    sub = sub[sub["volume_usd"].notna()]
    edges = np.quantile(sub["volume_usd"], [0, 1/3, 2/3, 1.0])
    strata = {}
    for name, lo, hi in [("low", edges[0], edges[1]),
                         ("mid", edges[1], edges[2]),
                         ("high", edges[2], edges[3])]:
        if name == "high":
            mask = (sub["volume_usd"] >= lo) & (sub["volume_usd"] <= hi)
        else:
            mask = (sub["volume_usd"] >= lo) & (sub["volume_usd"] < hi)
        s = sub[mask]
        strata[name] = all_metrics(s["p_1d"], s["label"], f"volume_{name}")
        strata[name]["vol_lo"] = float(lo)
        strata[name]["vol_hi"] = float(hi)
    results["polymarket"]["by_volume_p1d"] = strata

    # Comparison to "always base rate" baseline & 50/50 baseline
    p1d_sub = df[df["p_1d"].notna()]
    y_arr = p1d_sub["label"].values
    n = len(p1d_sub)
    base_rate = float(y_arr.mean()) if n else 0.5
    p_baseline = np.full(n, base_rate)
    p_uniform = np.full(n, 0.5)
    results["baselines_at_t1d"] = dict(
        n=n,
        base_rate_brier=brier(p_baseline, y_arr) if n else None,
        uniform_brier=brier(p_uniform, y_arr) if n else None,
        base_rate_log_loss=log_loss(p_baseline, y_arr) if n else None,
        uniform_log_loss=log_loss(p_uniform, y_arr) if n else None,
    )

    # Bin counts for the reliability table
    p = p1d_sub["p_1d"].values
    y = p1d_sub["label"].values
    centers, mean_p, freq, count = reliability(p, y, n_bins=10)
    results["reliability_t1d"] = dict(
        bin_centers=centers.tolist(),
        mean_p=mean_p.tolist(),
        empirical_freq=freq.tolist(),
        count=count.tolist(),
    )

    # Manifold (optional)
    manifold_path = DATA / "manifold.csv"
    if manifold_path.exists():
        mdf = pd.read_csv(manifold_path)
        print(f"loaded {len(mdf)} manifold rows")
        # Use resolution-time probability as forecast
        mdf = mdf[mdf["prob_resolution"].notna()]
        results["manifold"] = dict(
            n=int(len(mdf)),
            base_rate=float(mdf["label"].mean()),
            metric_at_close=all_metrics(mdf["prob_resolution"], mdf["label"], "manifold_close"),
        )

    # Figures
    figure_reliability(df, ROOT / "figure_1.png")
    figure_brier_vs_horizon(df, ROOT / "figure_2.png")

    # Categorical breakdown if present (sport-like vs others using question text)
    cat = df.copy()
    cat["is_crypto"] = cat["question"].str.contains(
        r"(?i)bitcoin|btc|ethereum|eth|solana|crypto|token|coin", regex=True
    )
    cat["is_politics"] = cat["question"].str.contains(
        r"(?i)trump|biden|harris|election|congress|senate|president|prime minister|win the.*primary",
        regex=True,
    )
    cat["is_sports"] = cat["question"].str.contains(
        r"(?i)NBA|NFL|MLB|UEFA|Premier League|Champions League|world series|super bowl|tennis|grand slam|finals\b|vs\.|game \d",
        regex=True,
    )
    cat["is_macro"] = cat["question"].str.contains(
        r"(?i)\binflation\b|\bunemployment\b|\bfed\b|\brate cut\b|\binterest rate\b|cpi\b|gdp\b|jobs report",
        regex=True,
    )
    by_topic = {}
    for col in ["is_crypto", "is_politics", "is_sports", "is_macro"]:
        s = cat[cat[col] & cat["p_1d"].notna()]
        if len(s) >= 30:
            by_topic[col] = all_metrics(s["p_1d"], s["label"], col)
    results["polymarket"]["by_topic_p1d"] = by_topic

    with open(DATA / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("wrote", DATA / "results.json")

    # human summary
    print("\n=== Summary ===")
    h = results["polymarket"]["horizons"]
    for k in ["p_30d", "p_7d", "p_1d", "p_close"]:
        m = h[k]
        if m["brier"] is not None:
            print(f"{k:>8} n={m['n']:>4}  Brier={m['brier']:.4f}  "
                  f"LogLoss={m['log_loss']:.4f}  ECE={m['ece']:.4f}")
    print(f"Baseline (base rate): Brier={results['baselines_at_t1d']['base_rate_brier']:.4f}")
    print(f"Baseline (uniform 0.5): Brier={results['baselines_at_t1d']['uniform_brier']:.4f}")


if __name__ == "__main__":
    main()
