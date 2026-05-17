"""Generate the paper's four figures from the analysis output.

  figure_1_fano_curves.png   Fano factor vs window size for individual
                             markets, with the Hawkes prediction overlay.
  figure_2_eta_distribution.png   distribution of branching-ratio estimates,
                             with marginal violins by category.
  figure_3_sign_autocorr.png  mean +/- IQR of trade-sign autocorrelation
                             across markets, with category breakdowns.
  figure_4_intensity_vs_eta.png   scatter of branching ratio against trade
                             intensity, with Spearman regression line.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
})


def load() -> tuple[list[dict], list[dict]]:
    summary = list(csv.DictReader((RES / "per_market.csv").open()))
    full = json.loads((RES / "per_market_full.json").read_text())
    return summary, full


CATEGORY_COLORS = {
    "election": "#d62728",
    "sports": "#1f77b4",
    "macro": "#2ca02c",
    "geopolitics": "#9467bd",
    "other": "#8c564b",
}


def fig1_fano(full: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    # plot Fano vs tau curves for every market, color by category, light alpha
    for r in full:
        taus = np.asarray(r.get("_fano_taus", []))
        F = np.asarray(r.get("_fano_F", []))
        if len(taus) < 4:
            continue
        col = CATEGORY_COLORS.get(r["category"], "#bbbbbb")
        ax.plot(taus, F, color=col, alpha=0.35, linewidth=0.9)
    # reference: F = 1 (Poisson)
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.0, label="Poisson ($F=1$)")
    # reference: F = 1/(1-eta)^2 horizontal asymptotes for eta=0.5, 0.8, 0.95
    for eta, lbl in [(0.5, r"$\eta=0.5$"), (0.8, r"$\eta=0.8$"), (0.95, r"$\eta=0.95$")]:
        F_asym = 1.0 / (1.0 - eta) ** 2
        ax.axhline(F_asym, color="black", linestyle="--", linewidth=0.6, alpha=0.5)
        ax.text(ax.get_xlim()[1] * 0.92, F_asym * 1.05, lbl, fontsize=7, color="black", alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Window length $\tau$ (s)")
    ax.set_ylabel(r"Fano factor $F(\tau)=\mathrm{Var}[N]/\mathbb{E}[N]$")
    ax.set_title("Trade-arrival over-dispersion vs. window size (40 markets)")
    # category legend
    handles = [plt.Line2D([0], [0], color=c, label=k) for k, c in CATEGORY_COLORS.items() if k != "other"]
    handles.append(plt.Line2D([0], [0], color="black", linestyle=":", label="Poisson"))
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "figure_1_fano_curves.png", bbox_inches="tight")
    plt.close(fig)


def fig2_eta(summary: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2), gridspec_kw={"width_ratios": [1.0, 1.4]})

    # left: histogram
    eta = np.array([float(r["eta_hat"]) for r in summary if r["eta_hat"] not in ("nan", "")])
    eta = eta[np.isfinite(eta)]
    ax = axes[0]
    ax.hist(eta, bins=np.linspace(0.5, 1.0, 16), color="#444444", alpha=0.85)
    ax.axvline(np.median(eta), color="#d62728", linestyle="--", linewidth=1.0, label=f"median = {np.median(eta):.3f}")
    ax.set_xlabel(r"Branching-ratio estimate $\hat\eta$")
    ax.set_ylabel("markets")
    ax.set_xlim(0.5, 1.0)
    ax.legend(frameon=False)
    ax.set_title(r"(a) Distribution of $\hat\eta$")

    # right: box plots by category
    ax = axes[1]
    by_cat = {}
    for r in summary:
        cat = r["category"]
        v = float(r["eta_hat"])
        if not np.isfinite(v):
            continue
        by_cat.setdefault(cat, []).append(v)
    order = ["election", "sports", "macro", "geopolitics", "other"]
    order = [c for c in order if c in by_cat]
    positions = np.arange(len(order))
    data = [by_cat[c] for c in order]
    bp = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True, showfliers=True, medianprops={"color": "black"})
    for patch, c in zip(bp["boxes"], order):
        patch.set_facecolor(CATEGORY_COLORS.get(c, "#bbbbbb"))
        patch.set_alpha(0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{c}\n(n={len(by_cat[c])})" for c in order])
    ax.set_ylabel(r"$\hat\eta$")
    ax.set_ylim(0.4, 1.0)
    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.set_title(r"(b) $\hat\eta$ by topic")

    fig.tight_layout()
    fig.savefig(FIG / "figure_2_eta_distribution.png", bbox_inches="tight")
    plt.close(fig)


def fig3_sign_acf(full: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    M = np.array([r["_rho"] for r in full if r.get("_rho") and len(r["_rho"]) >= 100])
    M = np.where(np.isfinite(M), M, np.nan)
    lags = np.arange(1, M.shape[1] + 1)
    median = np.nanmedian(M, axis=0)
    p25 = np.nanpercentile(M, 25, axis=0)
    p75 = np.nanpercentile(M, 75, axis=0)
    ax.fill_between(lags, p25, p75, color="#aaaaaa", alpha=0.4, label="25-75% interquartile band")
    ax.plot(lags, median, color="black", linewidth=1.5, label="median across 40 markets")
    # per-category median
    by_cat = {}
    for r in full:
        if not r.get("_rho"):
            continue
        by_cat.setdefault(r["category"], []).append(r["_rho"])
    for cat in ["election", "sports", "macro", "geopolitics"]:
        if cat not in by_cat:
            continue
        m = np.nanmedian(np.array(by_cat[cat]), axis=0)
        ax.plot(lags, m, color=CATEGORY_COLORS[cat], linewidth=1.0, alpha=0.95, label=f"{cat} median (n={len(by_cat[cat])})")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("lag $\\ell$ (trades)")
    ax.set_ylabel(r"$\rho(\ell)$ of trade sign")
    ax.set_xlim(0, 100)
    ax.set_title("Trade-sign autocorrelation across markets")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "figure_3_sign_autocorr.png", bbox_inches="tight")
    plt.close(fig)


def fig4_intensity_vs_eta(summary: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    eta = np.array([float(r["eta_hat"]) for r in summary])
    intens = np.array([float(r["intensity_per_min"]) for r in summary])
    cats = [r["category"] for r in summary]
    mask = np.isfinite(eta) & np.isfinite(intens) & (intens > 0)
    for cat in set(cats):
        idx = [i for i, c in enumerate(cats) if c == cat and mask[i]]
        if not idx:
            continue
        ax.scatter(intens[idx], eta[idx], s=35, color=CATEGORY_COLORS.get(cat, "#bbb"),
                   alpha=0.85, label=f"{cat} (n={len(idx)})", edgecolor="white", linewidth=0.5)
    x = intens[mask]
    y = eta[mask]
    if mask.sum() >= 5:
        lx = np.log10(x)
        slope, intercept, r, p, _ = stats.linregress(lx, y)
        xs = np.geomspace(x.min(), x.max(), 50)
        ax.plot(xs, intercept + slope * np.log10(xs), color="black", linestyle="--", linewidth=1.0,
                label=f"OLS on $\\log_{{10}}x$: $r={r:.2f}$, $p={p:.1e}$")
        sp = stats.spearmanr(x, y)
        ax.text(0.97, 0.05,
                f"Spearman $\\rho={sp.statistic:.2f}$\n$p={sp.pvalue:.1e}$",
                transform=ax.transAxes, fontsize=8, ha="right",
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="none", alpha=0.85))
    ax.set_xscale("log")
    ax.set_xlabel("trade intensity (trades / minute)")
    ax.set_ylabel(r"$\hat\eta$")
    ax.set_ylim(0.7, 1.02)
    ax.set_title("Branching ratio vs. trade intensity")
    ax.legend(frameon=False, loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "figure_4_intensity_vs_eta.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    summary, full = load()
    fig1_fano(full)
    fig2_eta(summary)
    fig3_sign_acf(full)
    fig4_intensity_vs_eta(summary)
    print(f"Wrote figures to {FIG}")


if __name__ == "__main__":
    main()
