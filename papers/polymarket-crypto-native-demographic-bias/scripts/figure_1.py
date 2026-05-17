"""Figure 1: Polymarket calibration curves at 1d / 7d / 30d horizons.
Figure 2: Brier components by topic.
Figure 3: Cross-platform calibration comparison (PM vs Manifold).

Reads ../data/results_summary.json and writes figure_*.png to ../.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

with (DATA / "results_summary.json").open() as fh:
    R = json.load(fh)

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "savefig.bbox": "tight",
    "savefig.dpi": 160,
})

# -------------------- Figure 1: PM calibration at three horizons ------------

fig, ax = plt.subplots(figsize=(4.5, 4.0))
colors = {"p_1d": "#1f77b4", "p_7d": "#ff7f0e", "p_30d": "#2ca02c"}
labels = {"p_1d": "1 day pre-close",
          "p_7d": "7 days pre-close",
          "p_30d": "30 days pre-close"}

for col, color in colors.items():
    bins = R["polymarket_calibration_bins"].get(col, [])
    xs, ys, ns = [], [], []
    for b in bins:
        if b["n"] and not np.isnan(b["mean_pred"]):
            xs.append(b["mean_pred"])
            ys.append(b["mean_actual"])
            ns.append(b["n"])
    if not xs:
        continue
    sizes = np.array(ns)
    sizes = 20 + 280 * (sizes - sizes.min()) / max(1, sizes.max() - sizes.min())
    ax.plot(xs, ys, "-", color=color, alpha=0.7,
            label=f"{labels[col]} (N={R['polymarket_calibration'][col]['n']})")
    ax.scatter(xs, ys, s=sizes, color=color, alpha=0.7, edgecolor="white",
               linewidth=0.6)

ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect calibration")
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_xlabel("Predicted probability (Polymarket Yes-token price)")
ax.set_ylabel("Empirical YES resolution rate")
ax.set_title("Polymarket calibration across forecast horizons")
ax.legend(loc="upper left", framealpha=0.9)
ax.grid(True, alpha=0.3)
fig.savefig(ROOT / "figure_1.png")
plt.close(fig)
print("wrote figure_1.png")

# -------------------- Figure 2: by topic (Brier + reliability bars) ----------

bt = R["polymarket_by_topic"]
topics = sorted(bt.keys(), key=lambda t: bt[t]["n"], reverse=True)
brier = [bt[t]["brier"] for t in topics]
brier_lo = [bt[t]["brier_ci_lo"] for t in topics]
brier_hi = [bt[t]["brier_ci_hi"] for t in topics]
ns = [bt[t]["n"] for t in topics]

fig, ax = plt.subplots(figsize=(6.2, 3.8))
x = np.arange(len(topics))
yerr = np.array([
    [bs - lo for bs, lo in zip(brier, brier_lo)],
    [hi - bs for bs, hi in zip(brier, brier_hi)],
])
ax.bar(x, brier, color="#4c72b0", alpha=0.85, edgecolor="white")
ax.errorbar(x, brier, yerr=yerr, fmt="none", ecolor="black",
             elinewidth=0.8, capsize=2.5)
ax.set_xticks(x)
ax.set_xticklabels([f"{t}\n(N={n})" for t, n in zip(topics, ns)],
                    rotation=35, ha="right", fontsize=8)
ax.set_ylabel("Brier score (lower = better, 95\\% bootstrap CI)")
ax.set_title("Polymarket 7-day calibration by topic")
ax.grid(True, axis="y", alpha=0.3)

# overlay base-rate Brier as a baseline (always predict base rate)
br = R["polymarket_calibration"]["p_7d"]["base_rate"]
br_brier = br * (1 - br)
ax.axhline(br_brier, color="red", ls="--", lw=1,
            label=f"climatology BS = p̄(1-p̄) = {br_brier:.3f}")
ax.legend(loc="upper left")
fig.savefig(ROOT / "figure_2.png")
plt.close(fig)
print("wrote figure_2.png")

# -------------------- Figure 3: PM vs Manifold calibration overlay ----------

fig, ax = plt.subplots(figsize=(4.5, 4.0))

# PM 7d
bins = R["polymarket_calibration_bins"]["p_7d"]
xs_p, ys_p, ns_p = [], [], []
for b in bins:
    if b["n"]:
        xs_p.append(b["mean_pred"])
        ys_p.append(b["mean_actual"])
        ns_p.append(b["n"])

# Manifold
bins_m = R["manifold_calibration_bins"]
xs_m, ys_m, ns_m = [], [], []
for b in bins_m:
    if b["n"]:
        xs_m.append(b["mean_pred"])
        ys_m.append(b["mean_actual"])
        ns_m.append(b["n"])

n_pm = R["polymarket_calibration"]["p_7d"]["n"]
n_mf = R["manifold_calibration_final_prob"]["n"]
ax.plot(xs_p, ys_p, "-o", color="#ff7f0e",
         label=f"Polymarket 7d (N={n_pm:,})", markersize=4)
ax.plot(xs_m, ys_m, "-s", color="#2ca02c",
         label=f"Manifold (N={n_mf:,})", markersize=4)
ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_xlabel("Predicted probability")
ax.set_ylabel("Empirical YES resolution rate")
ax.set_title("Cross-platform calibration: Polymarket vs Manifold")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left", framealpha=0.9)
fig.savefig(ROOT / "figure_3.png")
plt.close(fig)
print("wrote figure_3.png")
