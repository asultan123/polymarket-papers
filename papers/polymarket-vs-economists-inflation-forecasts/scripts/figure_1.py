"""Generate figure_1.png used in main.tex.

Two-panel scatter:
  (a) Polymarket implied point forecast vs. realised CPI
  (b) Cleveland Fed nowcast vs. realised CPI

Markers coloured by kind (annual vs monthly).  The 45-degree line is the
ideal predictor.  Reads ../data/analysis_results.json.
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "figure_1.png")


def main():
    res = json.load(open(os.path.join(DATA_DIR, "analysis_results.json")))
    rows = [r for r in res["rows"]
            if r.get("polymarket_mean") is not None
            and r.get("cleveland_fed_nowcast") is not None
            and r.get("realized") is not None]

    poly = [r["polymarket_mean"] for r in rows]
    cf = [r["cleveland_fed_nowcast"] for r in rows]
    real = [r["realized"] for r in rows]
    is_annual = [r["kind"] == "annual" for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6), sharey=False)
    for ax, vals, label in [(axes[0], poly, "Polymarket implied"),
                             (axes[1], cf, "Cleveland Fed nowcast")]:
        ann_x = [v for v, a in zip(vals, is_annual) if a]
        ann_y = [v for v, a in zip(real, is_annual) if a]
        mon_x = [v for v, a in zip(vals, is_annual) if not a]
        mon_y = [v for v, a in zip(real, is_annual) if not a]
        ax.scatter(ann_x, ann_y, s=42, edgecolor="black",
                   facecolor="#bd5b66", alpha=0.85,
                   label=f"Annual (YoY, n={len(ann_x)})")
        ax.scatter(mon_x, mon_y, s=42, edgecolor="black",
                   facecolor="#3a8ec8", alpha=0.85, marker="s",
                   label=f"Monthly (MoM, n={len(mon_x)})")
        # 45-degree line spanning observed range
        all_v = vals + real
        lo, hi = min(all_v) - 0.1, max(all_v) + 0.1
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6,
                label="perfect predictor")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"{label} forecast (% change)")
        ax.set_ylabel("Realised BLS CPI (% change)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=7.5)

    axes[0].set_title("(a) Polymarket bracket-implied mean",
                      fontsize=10)
    axes[1].set_title("(b) Cleveland Fed nowcast",
                      fontsize=10)

    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=160)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
