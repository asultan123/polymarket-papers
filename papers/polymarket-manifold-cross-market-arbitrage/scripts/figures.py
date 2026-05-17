"""Generate paper figures from matched_pairs.csv and summary.json.

Outputs PNG files next to scripts/ (figure_1.png, figure_2.png, figure_3.png).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def load_pairs() -> list[dict]:
    with (RESULTS / "matched_pairs.csv").open() as f:
        return list(csv.DictReader(f))


def is_mirror(q: str) -> bool:
    return "[Polymarket]" in q or "Polymarket]" in q or "Polymarket?" in q


def figure_1_scatter(pairs: list[dict], out: Path) -> None:
    """Polymarket Yes price vs Manifold probability, colored by similarity strata."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    def scatter_group(filter_fn, **kwargs):
        xs, ys = [], []
        for p in pairs:
            if filter_fn(p):
                xs.append(float(p["poly_yes"]))
                ys.append(float(p["mani_prob"]))
        ax.scatter(xs, ys, **kwargs)
        return len(xs)

    n_low = scatter_group(
        lambda p: float(p["similarity"]) < 0.70 and not is_mirror(p["mani_question"]),
        s=14, alpha=0.35, color="#bbbbbb", label="low-sim"
    )
    n_hi = scatter_group(
        lambda p: float(p["similarity"]) >= 0.70 and not is_mirror(p["mani_question"]),
        s=28, alpha=0.85, color="#1f77b4", label="high-sim"
    )
    n_mir = scatter_group(
        lambda p: is_mirror(p["mani_question"]),
        s=42, alpha=0.95, color="#d62728", marker="^", label="mirror"
    )

    handles, labels = ax.get_legend_handles_labels()
    labels[0] = f"low-sim (<0.70), n={n_low}"
    labels[1] = f"high-sim (>=0.70), n={n_hi}"
    labels[2] = f"mirror (explicit), n={n_mir}"

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="y=x")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Polymarket Yes price")
    ax.set_ylabel("Manifold probability")
    ax.set_title("Matched markets: Polymarket vs. Manifold")
    ax.legend(handles, labels, loc="upper left", fontsize=8, frameon=True)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def figure_2_gap_histogram(pairs: list[dict], out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.5))

    # All pairs
    g_all = np.array([float(p["gap_poly_minus_mani"]) for p in pairs])
    ax1.hist(g_all, bins=30, color="#888888", edgecolor="white")
    ax1.axvline(0, color="k", lw=1)
    ax1.set_title(f"All matched pairs (n={len(g_all)})")
    ax1.set_xlabel("Gap = Poly $-$ Manifold")
    ax1.set_ylabel("Count")
    ax1.set_xlim(-1, 1)
    ax1.grid(alpha=0.3)

    # High-similarity, independent, liquid
    g_hi = np.array([
        float(p["gap_poly_minus_mani"]) for p in pairs
        if float(p["similarity"]) >= 0.70
        and not is_mirror(p["mani_question"])
        and (float(p["mani_volume"]) >= 100 or int(p["mani_bettors"]) >= 5)
    ])
    ax2.hist(g_hi, bins=20, color="#1f77b4", edgecolor="white")
    ax2.axvline(0, color="k", lw=1)
    ax2.axvline(g_hi.mean(), color="r", lw=1, linestyle="--",
                label=f"mean={g_hi.mean():+.3f}")
    ax2.set_title(f"High-sim, independent, liquid (n={len(g_hi)})")
    ax2.set_xlabel("Gap = Poly $-$ Manifold")
    ax2.set_xlim(-1, 1)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def figure_3_similarity_vs_gap(pairs: list[dict], buckets: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    centers = [(b["lo"] + b["hi"]) / 2 for b in buckets]
    abs_gaps = [b["abs_mean_gap"] for b in buckets]
    ns = [b["n"] for b in buckets]
    widths = [b["hi"] - b["lo"] for b in buckets]

    bars = ax.bar(centers, abs_gaps, width=[w * 0.9 for w in widths],
                  color="#1f77b4", edgecolor="white", alpha=0.85)
    for bar, n in zip(bars, ns):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"n={n}", ha="center", fontsize=8)
    ax.set_xlabel("TF-IDF cosine similarity bucket")
    ax.set_ylabel("Mean $|$Polymarket $-$ Manifold$|$")
    ax.set_title("Disagreement shrinks as question-text similarity rises")
    ax.set_xticks([0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
    ax.set_xticklabels(["0.4–0.5", "0.5–0.6", "0.6–0.7", "0.7–0.8", "0.8–0.9", "0.9–1.0"],
                       rotation=0, fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    pairs = load_pairs()
    summary = json.load((RESULTS / "summary.json").open())

    figure_1_scatter(pairs, ROOT / "figure_1.png")
    figure_2_gap_histogram(pairs, ROOT / "figure_2.png")
    figure_3_similarity_vs_gap(pairs, summary["similarity_buckets"], ROOT / "figure_3.png")
    print("Wrote figure_1.png, figure_2.png, figure_3.png")


if __name__ == "__main__":
    main()
