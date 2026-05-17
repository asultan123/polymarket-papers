"""Analyse Polymarket vs Manifold calibration and category-level biases.

Reads CSVs produced by fetch_data.py from ../data/ and writes:
    ../data/results_summary.json - every number that appears in the paper
    ../figure_1.png  - calibration curves (PM 1d / 7d / 30d before close)
    ../figure_2.png  - calibration by Polymarket topic category
    ../figure_3.png  - Polymarket vs Manifold calibration overlay

The single source of truth for paper numbers is results_summary.json - never
re-type a number from a plot.
"""
from __future__ import annotations
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# ---------------------------------------------------------------------------
# Topic classification via keyword matching (transparent + reproducible).
# Rules applied in order; first match wins. We deliberately keep this simple
# rather than using a learned classifier because we want every result in the
# paper to be inspectable by hand.
# ---------------------------------------------------------------------------

TOPIC_RULES: list[tuple[str, re.Pattern]] = [
    ("Crypto", re.compile(
        r"\b(bitcoin|btc|ethereum|eth|solana|sol|crypto|dogecoin|doge|"
        r"xrp|usdc|usdt|stablecoin|tether|coinbase|binance|polymarket|"
        r"polygon|chainlink|cardano|avalanche|nft|memecoin|altcoin|"
        r"defi|web3|ledger|wallet)\b", re.I)),
    ("Finance/Macro", re.compile(
        r"\b(fed|federal reserve|fomc|interest rate|cpi|inflation|s&p|"
        r"sp500|nasdaq|recession|gdp|unemployment|jobs report|nfp|"
        r"yield|bond|treasury|powell|yellen|stock market|dow jones)\b",
        re.I)),
    ("Politics", re.compile(
        r"\b(trump|biden|harris|kamala|desantis|haley|vance|"
        r"president|presidential|election|senate|senator|governor|"
        r"congress|congressional|house race|primary|caucus|impeach|"
        r"republican|democrat|gop|electoral|swing state|nominee|"
        r"prime minister|chancellor|parliament|cabinet|secretary of)\b",
        re.I)),
    ("Geopolitics/War", re.compile(
        r"\b(ukraine|russia|putin|zelens?kyy|israel|gaza|hamas|"
        r"hezbollah|iran|iraq|syria|yemen|houthi|nato|china|taiwan|"
        r"north korea|war|ceasefire|invasion|missile|strike|nuclear|"
        r"airstrike|sanction|opec)\b", re.I)),
    ("Sports", re.compile(
        r"\b(nba|nfl|mlb|nhl|champion|playoff|super bowl|"
        r"world series|world cup|finals|fifa|uefa|premier league|"
        r"la liga|bundesliga|formula 1|f1|tennis|wimbledon|"
        r"masters|pga|us open|olympics|olympic|boxing|ufc|mma|"
        r"goal|vs\.|game [0-9]|cricket|t20|ipl|nrl|afl|dota|csgo|"
        r"valorant|lol|league of legends|esports|grand slam)\b", re.I)),
    ("Tech/AI", re.compile(
        r"\b(openai|chatgpt|gpt-?[0-9]|gpt|claude|anthropic|gemini|"
        r"google|deepmind|llm|ai model|artificial intelligence|"
        r"tesla|spacex|musk|apple|microsoft|nvidia|amazon|meta|"
        r"facebook|tiktok|twitter|x platform|instagram)\b", re.I)),
    ("Culture/Entertainment", re.compile(
        r"\b(oscar|emmy|grammy|movie|film|box office|album|song|"
        r"netflix|disney|marvel|kardashian|swift|taylor|kanye|"
        r"drake|tiktok|youtube|spotify|streamer|celebrity|wedding|"
        r"divorce|baby|engaged)\b", re.I)),
    ("Weather/Climate", re.compile(
        r"\b(hurricane|tornado|earthquake|storm|tsunami|"
        r"flood|wildfire|heatwave|temperature|climate)\b", re.I)),
]


def classify_topic(question: str) -> str:
    q = question or ""
    for label, pat in TOPIC_RULES:
        if pat.search(q):
            return label
    return "Other"


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------


@dataclass
class CalibrationBin:
    p_lo: float
    p_hi: float
    mean_pred: float
    mean_actual: float
    n: int


def calibration_curve(preds: np.ndarray, actuals: np.ndarray,
                      bin_edges: list[float] | None = None) -> list[CalibrationBin]:
    """Bin predictions and compute empirical fraction of YES outcomes."""
    if bin_edges is None:
        bin_edges = [0.0, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0]
    out: list[CalibrationBin] = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        if hi == bin_edges[-1]:
            mask = (preds >= lo) & (preds <= hi)
        else:
            mask = (preds >= lo) & (preds < hi)
        n = int(mask.sum())
        if n == 0:
            out.append(CalibrationBin(lo, hi, float("nan"), float("nan"), 0))
            continue
        out.append(CalibrationBin(lo, hi,
                                  float(preds[mask].mean()),
                                  float(actuals[mask].mean()),
                                  n))
    return out


def brier_score(preds: np.ndarray, actuals: np.ndarray) -> float:
    return float(((preds - actuals) ** 2).mean())


def brier_decomposition(preds: np.ndarray, actuals: np.ndarray,
                         bin_edges: list[float] | None = None) -> dict:
    """Murphy (1973) decomposition: BS = reliability - resolution + uncertainty."""
    if bin_edges is None:
        bin_edges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.001]
    N = len(preds)
    base = float(actuals.mean())
    uncertainty = base * (1 - base)
    reliability = 0.0
    resolution = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (preds >= lo) & (preds < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        f_k = float(preds[mask].mean())
        o_k = float(actuals[mask].mean())
        reliability += n / N * (f_k - o_k) ** 2
        resolution += n / N * (o_k - base) ** 2
    return {
        "brier": brier_score(preds, actuals),
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "base_rate": base,
    }


def log_loss(preds: np.ndarray, actuals: np.ndarray, eps: float = 1e-6) -> float:
    p = np.clip(preds, eps, 1 - eps)
    return float(-(actuals * np.log(p) + (1 - actuals) * np.log(1 - p)).mean())


# ---------------------------------------------------------------------------
# Bootstrap CI for calibration error
# ---------------------------------------------------------------------------


def bootstrap_brier_ci(preds: np.ndarray, actuals: np.ndarray,
                        n_boot: int = 2000, alpha: float = 0.05,
                        seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    N = len(preds)
    scores = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, N, N)
        scores[i] = brier_score(preds[idx], actuals[idx])
    lo, hi = np.quantile(scores, [alpha / 2, 1 - alpha / 2])
    return float(brier_score(preds, actuals)), float(lo), float(hi)


# ---------------------------------------------------------------------------
# Match Polymarket and Manifold markets via fuzzy slug/question matching
# ---------------------------------------------------------------------------


STOPWORDS = set("the a an of in on for to and or by with at is are was were be been "
                "will did does do has have had this that these those who whom which "
                "what when where why how before after over under between against "
                "vs vs. v. than as also from any get got make made get got than then "
                "into onto out about up down".split())


def tokens(s: str) -> set[str]:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    toks = {t for t in s.split() if t and t not in STOPWORDS and len(t) > 2}
    return toks


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def match_markets(pm_rows: list[dict], mf_rows: list[dict],
                   min_score: float = 0.55) -> list[dict]:
    """Greedy nearest-neighbour matching between Polymarket and Manifold."""
    pm_tok = [(i, tokens(r["question"])) for i, r in enumerate(pm_rows)]
    mf_tok = [(j, tokens(r["question"])) for j, r in enumerate(mf_rows)]
    matched: list[dict] = []
    used_mf: set[int] = set()
    for i, ti in pm_tok:
        if len(ti) < 3:
            continue
        best_j = -1
        best_s = 0.0
        for j, tj in mf_tok:
            if j in used_mf:
                continue
            if len(tj) < 3:
                continue
            s = jaccard(ti, tj)
            if s > best_s:
                best_s = s
                best_j = j
        if best_s >= min_score and best_j >= 0:
            used_mf.add(best_j)
            matched.append({
                "pm_idx": i,
                "mf_idx": best_j,
                "score": best_s,
                "pm_question": pm_rows[i]["question"],
                "mf_question": mf_rows[best_j]["question"],
            })
    return matched


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def _to_float(x):
    try:
        if x is None or x == "" or x == "None":
            return None
        return float(x)
    except (ValueError, TypeError):
        return None


def load_polymarket() -> pd.DataFrame:
    df = pd.read_csv(DATA / "polymarket_prices.csv")
    for c in ("p_1d", "p_7d", "p_30d", "p_final"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["outcome_yes"] = df["outcome_yes"].astype(int)
    df["topic"] = df["question"].fillna("").map(classify_topic)
    return df


def load_manifold() -> pd.DataFrame:
    df = pd.read_csv(DATA / "manifold_markets.csv")
    df["probability"] = pd.to_numeric(df["probability"], errors="coerce")
    df["outcome_yes"] = df["outcome_yes"].astype(int)
    df["topic"] = df["question"].fillna("").map(classify_topic)
    return df


def analyse() -> dict:
    pm = load_polymarket()
    mf = load_manifold()
    print(f"Loaded Polymarket: {len(pm)} rows, Manifold: {len(mf)} rows")
    results: dict = {
        "n_polymarket_markets": int(len(pm)),
        "n_manifold_markets": int(len(mf)),
    }

    # ------------------------------------------------------------- coverage
    pm_with_price = pm.dropna(subset=["p_7d"])
    results["n_polymarket_with_7d_price"] = int(len(pm_with_price))
    results["polymarket_volume_total_usd"] = float(pm_with_price["volume_num"].sum())
    results["polymarket_yes_rate"] = float(pm["outcome_yes"].mean())
    results["manifold_yes_rate"] = float(mf["outcome_yes"].mean())

    # ------------------------------------- main calibration: PM at 1d, 7d, 30d
    horizons_summary: dict = {}
    for col in ("p_1d", "p_7d", "p_30d"):
        sub = pm.dropna(subset=[col])
        p = sub[col].to_numpy()
        a = sub["outcome_yes"].to_numpy()
        if len(p) == 0:
            continue
        bs, lo, hi = bootstrap_brier_ci(p, a)
        dec = brier_decomposition(p, a)
        horizons_summary[col] = {
            "n": int(len(p)),
            "brier": bs,
            "brier_ci_lo": lo,
            "brier_ci_hi": hi,
            "reliability": dec["reliability"],
            "resolution": dec["resolution"],
            "uncertainty": dec["uncertainty"],
            "log_loss": log_loss(p, a),
            "base_rate": dec["base_rate"],
            "mean_pred": float(p.mean()),
        }
    results["polymarket_calibration"] = horizons_summary

    # calibration bins (for figure)
    calib_pm = {}
    for col in ("p_1d", "p_7d", "p_30d"):
        sub = pm.dropna(subset=[col])
        p = sub[col].to_numpy()
        a = sub["outcome_yes"].to_numpy()
        bins = calibration_curve(p, a)
        calib_pm[col] = [b.__dict__ for b in bins]
    results["polymarket_calibration_bins"] = calib_pm

    # ------------------------------------ by topic / category
    by_topic = {}
    for topic, sub in pm.dropna(subset=["p_7d"]).groupby("topic"):
        if len(sub) < 20:
            continue
        p = sub["p_7d"].to_numpy()
        a = sub["outcome_yes"].to_numpy()
        bs, lo, hi = bootstrap_brier_ci(p, a)
        dec = brier_decomposition(p, a)
        by_topic[topic] = {
            "n": int(len(p)),
            "brier": bs,
            "brier_ci_lo": lo,
            "brier_ci_hi": hi,
            "reliability": dec["reliability"],
            "resolution": dec["resolution"],
            "base_rate": dec["base_rate"],
            "mean_pred": float(p.mean()),
            "volume_mean": float(sub["volume_num"].mean()),
            "volume_median": float(sub["volume_num"].median()),
        }
    results["polymarket_by_topic"] = by_topic

    # ----------- volume-stratified calibration (high vs low volume markets)
    pm7 = pm.dropna(subset=["p_7d"]).copy()
    vol_q = pm7["volume_num"].quantile([0.33, 0.67]).tolist()
    pm7["vol_bucket"] = pd.cut(pm7["volume_num"],
                                bins=[-1, vol_q[0], vol_q[1], float("inf")],
                                labels=["low", "mid", "high"])
    by_vol = {}
    for bucket, sub in pm7.groupby("vol_bucket"):
        p = sub["p_7d"].to_numpy()
        a = sub["outcome_yes"].to_numpy()
        bs, lo, hi = bootstrap_brier_ci(p, a)
        by_vol[str(bucket)] = {
            "n": int(len(p)),
            "brier": bs,
            "brier_ci_lo": lo,
            "brier_ci_hi": hi,
            "volume_thresh_lo": float(sub["volume_num"].min()),
            "volume_thresh_hi": float(sub["volume_num"].max()),
        }
    results["polymarket_by_volume"] = by_vol
    results["polymarket_volume_quantiles"] = {"q33": float(vol_q[0]),
                                              "q67": float(vol_q[1])}

    # ------------------------------------ Manifold baseline calibration
    mf_clean = mf.dropna(subset=["probability"])
    p = mf_clean["probability"].to_numpy()
    a = mf_clean["outcome_yes"].to_numpy()
    bs, lo, hi = bootstrap_brier_ci(p, a)
    dec = brier_decomposition(p, a)
    results["manifold_calibration_final_prob"] = {
        "n": int(len(p)),
        "brier": bs,
        "brier_ci_lo": lo,
        "brier_ci_hi": hi,
        "reliability": dec["reliability"],
        "resolution": dec["resolution"],
        "log_loss": log_loss(p, a),
        "base_rate": dec["base_rate"],
        "mean_pred": float(p.mean()),
    }
    results["manifold_calibration_bins"] = [
        b.__dict__ for b in calibration_curve(p, a)
    ]

    # ------------------------------------ matched cross-platform sample
    pm_records = pm_with_price.to_dict("records")
    mf_records = mf_clean.to_dict("records")
    matches = match_markets(pm_records, mf_records, min_score=0.55)
    results["n_cross_platform_matches"] = len(matches)
    if matches:
        diffs = []
        diffs_signed = []
        for m in matches:
            pm_r = pm_records[m["pm_idx"]]
            mf_r = mf_records[m["mf_idx"]]
            pm_p = pm_r.get("p_7d")
            mf_p = mf_r.get("probability")
            if pm_p is None or mf_p is None:
                continue
            diff = abs(pm_p - mf_p)
            diffs.append(diff)
            diffs_signed.append(pm_p - mf_p)
            m["pm_p_7d"] = pm_p
            m["mf_prob"] = mf_p
            m["outcome_yes"] = pm_r["outcome_yes"]
        results["cross_platform_mean_abs_diff"] = float(np.mean(diffs)) if diffs else None
        results["cross_platform_mean_signed_diff"] = float(np.mean(diffs_signed)) if diffs_signed else None
        results["cross_platform_pairwise_brier_pm"] = float(
            np.mean([(m["pm_p_7d"] - m["outcome_yes"]) ** 2 for m in matches if "pm_p_7d" in m])
        )
        results["cross_platform_pairwise_brier_mf"] = float(
            np.mean([(m["mf_prob"] - m["outcome_yes"]) ** 2 for m in matches if "mf_prob" in m])
        )
        results["cross_platform_examples"] = [
            {
                "pm_q": m["pm_question"][:120],
                "mf_q": m["mf_question"][:120],
                "pm_p_7d": m.get("pm_p_7d"),
                "mf_prob": m.get("mf_prob"),
                "outcome_yes": m.get("outcomes_yes") or m.get("outcome_yes"),
                "jaccard": round(m["score"], 3),
            }
            for m in matches[:10]
        ]

    # ------------------------------------ topic distribution
    pm_topic_counts = pm["topic"].value_counts().to_dict()
    mf_topic_counts = mf["topic"].value_counts().to_dict()
    pm_topic_volume = pm.groupby("topic")["volume_num"].sum().to_dict()
    results["polymarket_topic_counts"] = {k: int(v) for k, v in pm_topic_counts.items()}
    results["manifold_topic_counts"] = {k: int(v) for k, v in mf_topic_counts.items()}
    results["polymarket_topic_volume_usd"] = {k: float(v) for k, v in pm_topic_volume.items()}

    # ------------------------------------ "Crypto-native preference test":
    # do PM Crypto markets vs Politics markets show different bias signs?
    bias_by_topic = {}
    for topic, sub in pm7.groupby("topic"):
        if len(sub) < 20:
            continue
        # Signed mean error: average(pred - actual)
        p = sub["p_7d"].to_numpy()
        a = sub["outcome_yes"].to_numpy()
        bias_by_topic[topic] = {
            "n": int(len(p)),
            "signed_mean_error": float((p - a).mean()),
            "mean_pred": float(p.mean()),
            "mean_actual": float(a.mean()),
        }
    results["polymarket_bias_by_topic"] = bias_by_topic

    # write
    out_path = DATA / "results_summary.json"
    with out_path.open("w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"Wrote {out_path}")
    return results


if __name__ == "__main__":
    analyse()
