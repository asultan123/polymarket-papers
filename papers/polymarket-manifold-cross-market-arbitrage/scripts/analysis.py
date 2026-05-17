"""Cross-market arbitrage analysis between Polymarket and Manifold.

Pipeline:
  1) Load raw market dumps produced by fetch_data.py.
  2) Restrict to active binary markets with current prices.
  3) Use TF-IDF + cosine similarity on normalized questions to surface
     candidate matches.
  4) Apply rule-based filters to reduce false positives:
       - both end-dates within 365 days of each other,
       - shared named entities (overlap of capitalized tokens >= 1),
       - both questions contain at least one of a small set of polarity
         keywords (will/can/by) so we're comparing real binary forecasts.
  5) Compute per-pair price gap = poly_yes_price - manifold_probability.
  6) Report summary statistics, run a sign test, fit a least-squares
     regression of Manifold probability on Polymarket Yes price, and
     estimate the fraction of pairs whose absolute gap exceeds plausible
     round-trip Polymarket trading costs (2% maker+taker baseline).

All summary numbers and the matched-pair CSV are emitted to ../results/.
Random seed fixed for reproducibility.
"""
from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

SEED = 20260517
RNG = np.random.default_rng(SEED)

STOP = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "by",
    "with", "and", "or", "not", "be", "is", "are", "was", "were",
    "will", "would", "should", "could", "can", "do", "does", "did",
    "have", "has", "had", "this", "that", "these", "those",
    "before", "after",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[\.'-][A-Za-z0-9]+)*")


def normalize(q: str) -> str:
    tokens = TOKEN_RE.findall(q.lower())
    return " ".join(t for t in tokens if t not in STOP and len(t) > 1)


def cap_tokens(q: str) -> set[str]:
    return {t.lower() for t in TOKEN_RE.findall(q) if t[:1].isupper() and len(t) > 2}


def load_polymarket() -> list[dict]:
    raw = json.load((DATA / "polymarket_markets.json").open())
    out = []
    for m in raw:
        if not m.get("active") or m.get("closed"):
            continue
        outcomes = m.get("outcomes")
        prices = m.get("outcomePrices")
        if not outcomes or not prices:
            continue
        try:
            outcomes = json.loads(outcomes) if isinstance(outcomes, str) else outcomes
            prices = json.loads(prices) if isinstance(prices, str) else prices
        except Exception:
            continue
        if outcomes != ["Yes", "No"] or len(prices) != 2:
            continue
        try:
            yes = float(prices[0])
        except (TypeError, ValueError):
            continue
        if not (0.0 < yes < 1.0):
            continue
        out.append({
            "platform": "polymarket",
            "id": m["id"],
            "question": m["question"],
            "yes_price": yes,
            "volume": float(m.get("volumeNum", 0.0)),
            "liquidity": float(m.get("liquidityNum", 0.0)),
            "best_bid": _safe_float(m.get("bestBid")),
            "best_ask": _safe_float(m.get("bestAsk")),
            "end_date": m.get("endDate", ""),
            "slug": m.get("slug", ""),
        })
    return out


def _safe_float(x) -> float | None:
    try:
        return float(x) if x is not None else None
    except (TypeError, ValueError):
        return None


def load_manifold() -> list[dict]:
    raw = json.load((DATA / "manifold_markets.json").open())
    out = []
    for m in raw:
        if m.get("outcomeType") != "BINARY":
            continue
        if m.get("isResolved"):
            continue
        prob = m.get("probability")
        if prob is None or not (0.0 < prob < 1.0):
            continue
        out.append({
            "platform": "manifold",
            "id": m["id"],
            "question": m["question"],
            "probability": float(prob),
            "volume": float(m.get("volume", 0.0)),
            "total_liquidity": float(m.get("totalLiquidity", 0.0)),
            "unique_bettors": int(m.get("uniqueBettorCount", 0)),
            "close_time_ms": m.get("closeTime", 0),
            "url": m.get("url", ""),
        })
    return out


def close_time_iso(close_time_ms: int) -> str:
    if not close_time_ms:
        return ""
    from datetime import datetime, timezone
    try:
        return datetime.fromtimestamp(close_time_ms / 1000.0, tz=timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        return ""


def days_between(iso_a: str, iso_b: str) -> float | None:
    if not iso_a or not iso_b:
        return None
    from datetime import datetime
    try:
        a = datetime.fromisoformat(iso_a.replace("Z", "+00:00"))
        b = datetime.fromisoformat(iso_b.replace("Z", "+00:00"))
    except ValueError:
        return None
    return abs((a - b).days)


def find_candidate_pairs(poly: list[dict], mani: list[dict], top_k: int = 5,
                          sim_threshold: float = 0.40) -> list[dict]:
    """For each Polymarket question, find top-k similar Manifold questions and
    keep those above sim_threshold. We further require capitalized-token
    overlap >= 1 to filter generic matches (`will X happen` style)."""
    poly_text = [normalize(p["question"]) for p in poly]
    mani_text = [normalize(m["question"]) for m in mani]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        norm="l2",
    )
    vectorizer.fit(poly_text + mani_text)

    A = vectorizer.transform(poly_text)
    B = vectorizer.transform(mani_text)

    # Process in chunks so we don't materialise an N x M matrix at once.
    chunk = 500
    candidates: list[dict] = []
    for start in range(0, A.shape[0], chunk):
        sim = cosine_similarity(A[start:start + chunk], B)
        for i_local in range(sim.shape[0]):
            i = start + i_local
            row = sim[i_local]
            order = np.argpartition(-row, top_k)[:top_k]
            for j in order:
                s = float(row[j])
                if s < sim_threshold:
                    continue
                p_caps = cap_tokens(poly[i]["question"])
                m_caps = cap_tokens(mani[j]["question"])
                shared = p_caps & m_caps
                if not shared:
                    continue
                candidates.append({
                    "poly_idx": int(i),
                    "mani_idx": int(j),
                    "similarity": s,
                    "shared_entities": sorted(shared),
                })
    return candidates


def attach_pair_features(candidates: list[dict], poly: list[dict], mani: list[dict]) -> list[dict]:
    """Compute gap, end-date alignment, and dedupe (keep best match per Polymarket question)."""
    enriched = []
    for c in candidates:
        p = poly[c["poly_idx"]]
        m = mani[c["mani_idx"]]
        m_close_iso = close_time_iso(m.get("close_time_ms", 0))
        delta_days = days_between(p["end_date"], m_close_iso)
        enriched.append({
            **c,
            "poly_question": p["question"],
            "mani_question": m["question"],
            "poly_yes": p["yes_price"],
            "mani_prob": m["probability"],
            "gap_poly_minus_mani": p["yes_price"] - m["probability"],
            "poly_volume": p["volume"],
            "poly_liquidity": p["liquidity"],
            "poly_best_bid": p["best_bid"],
            "poly_best_ask": p["best_ask"],
            "poly_spread": (p["best_ask"] - p["best_bid"])
                if (p["best_ask"] is not None and p["best_bid"] is not None) else None,
            "mani_volume": m["volume"],
            "mani_liquidity": m["total_liquidity"],
            "mani_bettors": m["unique_bettors"],
            "poly_end_date": p["end_date"],
            "mani_close_date": m_close_iso,
            "delta_close_days": delta_days,
            "poly_slug": p["slug"],
            "mani_url": m["url"],
        })

    enriched.sort(key=lambda r: (r["poly_idx"], -r["similarity"]))
    best_for_poly: dict[int, dict] = {}
    for r in enriched:
        if r["poly_idx"] not in best_for_poly:
            best_for_poly[r["poly_idx"]] = r
    pairs = list(best_for_poly.values())

    # Also dedupe so a Manifold market is matched at most once.
    pairs.sort(key=lambda r: -r["similarity"])
    used_mani: set[int] = set()
    deduped = []
    for r in pairs:
        if r["mani_idx"] in used_mani:
            continue
        used_mani.add(r["mani_idx"])
        deduped.append(r)
    return deduped


def stats_summary(pairs: list[dict], label: str = "all") -> dict:
    pairs.sort(key=lambda r: -r["similarity"])
    if not pairs:
        return {"n": 0, "label": label}

    gaps = np.array([p["gap_poly_minus_mani"] for p in pairs])
    poly = np.array([p["poly_yes"] for p in pairs])
    mani = np.array([p["mani_prob"] for p in pairs])

    n = len(pairs)
    mean_gap = float(gaps.mean())
    median_gap = float(np.median(gaps))
    abs_mean_gap = float(np.abs(gaps).mean())
    sd_gap = float(gaps.std(ddof=1)) if n > 1 else float("nan")

    # Paired t-test of poly vs manifold
    t_stat, t_p = stats.ttest_rel(poly, mani)

    # Wilcoxon signed-rank (non-parametric)
    nonzero = gaps[gaps != 0]
    w_stat, w_p = stats.wilcoxon(nonzero) if len(nonzero) >= 5 else (float("nan"), float("nan"))

    # Sign test: how many gaps are positive?
    n_pos = int((gaps > 0).sum())
    n_neg = int((gaps < 0).sum())
    sign_p = stats.binomtest(n_pos, n_pos + n_neg, p=0.5).pvalue if (n_pos + n_neg) > 0 else float("nan")

    # OLS regression: mani = a + b * poly
    if len(poly) >= 5:
        reg = stats.linregress(poly, mani)
        slope = float(reg.slope)
        intercept = float(reg.intercept)
        r2 = float(reg.rvalue ** 2)
        slope_p = float(reg.pvalue)
        slope_se = float(reg.stderr)
    else:
        slope = intercept = r2 = slope_p = slope_se = float("nan")

    # Pearson and Spearman correlations
    pear_r, pear_p = stats.pearsonr(poly, mani)
    spear_r, spear_p = stats.spearmanr(poly, mani)

    # Fraction with |gap| > 5%, > 10%, > 20%
    frac_gt05 = float((np.abs(gaps) > 0.05).mean())
    frac_gt10 = float((np.abs(gaps) > 0.10).mean())
    frac_gt20 = float((np.abs(gaps) > 0.20).mean())

    # "Arbitrage exceeding 2% round-trip cost" — a stand-in for whether the
    # discrepancy clears plausible fees. Real Polymarket trading fees are 0
    # currently but withdraw/USDC bridge friction is non-trivial; 2% is a
    # conservative round-trip threshold; 5% covers more illiquid markets.
    frac_gt2 = float((np.abs(gaps) > 0.02).mean())

    return {
        "label": label,
        "n": n,
        "mean_gap": mean_gap,
        "median_gap": median_gap,
        "abs_mean_gap": abs_mean_gap,
        "sd_gap": sd_gap,
        "ttest_t": float(t_stat),
        "ttest_p": float(t_p),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_p": float(w_p),
        "sign_test_p": float(sign_p),
        "n_poly_above": n_pos,
        "n_mani_above": n_neg,
        "slope": slope,
        "intercept": intercept,
        "r_squared": r2,
        "slope_p": slope_p,
        "slope_se": slope_se,
        "pearson_r": float(pear_r),
        "pearson_p": float(pear_p),
        "spearman_r": float(spear_r),
        "spearman_p": float(spear_p),
        "frac_abs_gap_gt_2pct": frac_gt2,
        "frac_abs_gap_gt_5pct": frac_gt05,
        "frac_abs_gap_gt_10pct": frac_gt10,
        "frac_abs_gap_gt_20pct": frac_gt20,
    }


def categorical_breakdown(pairs: list[dict]) -> dict:
    """Bucket pairs by best-guess category from keyword matching."""
    cats = {
        "politics_election": ["election", "primary", "president", "senate", "congress",
                              "vote", "ballot", "trump", "biden", "harris", "vance", "republican",
                              "democrat", "gop"],
        "crypto": ["bitcoin", "btc", "ethereum", "eth", "solana", "doge", "crypto", "coin"],
        "sports": ["nba", "nfl", "mlb", "champions", "super bowl", "world cup", "match",
                   "tournament", "playoff", "olympic", "ufc", "boxing", "win against",
                   "championship"],
        "ai_tech": ["openai", "anthropic", "claude", "gpt", "ai ", "model", "agi", "llm",
                    "tesla", "musk", "spacex"],
        "economy_macro": ["fed", "rate", "inflation", "cpi", "gdp", "recession", "unemployment",
                          "stock", "s&p", "nasdaq"],
    }
    counts = Counter()
    for p in pairs:
        text = (p["poly_question"] + " " + p["mani_question"]).lower()
        assigned = False
        for cat, kws in cats.items():
            if any(k in text for k in kws):
                counts[cat] += 1
                assigned = True
                break
        if not assigned:
            counts["other"] += 1
    return dict(counts)


def write_csv(pairs: list[dict], path: Path) -> None:
    if not pairs:
        return
    cols = [
        "similarity", "shared_entities",
        "poly_question", "mani_question",
        "poly_yes", "mani_prob", "gap_poly_minus_mani",
        "poly_volume", "poly_liquidity", "poly_best_bid", "poly_best_ask", "poly_spread",
        "mani_volume", "mani_liquidity", "mani_bettors",
        "poly_end_date", "mani_close_date", "delta_close_days",
        "poly_slug", "mani_url",
    ]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in pairs:
            row = []
            for c in cols:
                v = r.get(c)
                if isinstance(v, list):
                    v = "|".join(v)
                row.append(v)
            w.writerow(row)


def tag_mirror(pair: dict) -> bool:
    q = pair["mani_question"]
    return "[Polymarket]" in q or "Polymarket]" in q or "Polymarket?" in q


def main() -> None:
    poly = load_polymarket()
    mani = load_manifold()
    print(f"Loaded {len(poly)} Polymarket binary markets, {len(mani)} Manifold binary markets")

    cands = find_candidate_pairs(poly, mani, top_k=5, sim_threshold=0.40)
    print(f"Initial candidate pairs (sim>=0.40, shared entity): {len(cands)}")

    pairs = attach_pair_features(cands, poly, mani)
    print(f"After dedupe (1:1 best match): {len(pairs)}")

    # Apply ±365 day end-date alignment filter
    aligned = [p for p in pairs if p["delta_close_days"] is not None and p["delta_close_days"] <= 365]
    print(f"After end-date alignment (≤365d): {len(aligned)}")

    for p in aligned:
        p["is_mirror"] = tag_mirror(p)
        p["mani_liquid"] = p["mani_volume"] >= 100 or p["mani_bettors"] >= 5

    write_csv(aligned, RESULTS / "matched_pairs.csv")
    print(f"Wrote {len(aligned)} pairs to {RESULTS/'matched_pairs.csv'}")

    # Strata
    all_pairs = aligned
    hi_sim = [p for p in aligned if p["similarity"] >= 0.70]
    hi_sim_indep = [p for p in hi_sim if not p["is_mirror"]]
    hi_sim_indep_liquid = [p for p in hi_sim_indep if p["mani_liquid"]]
    mirrors = [p for p in aligned if p["is_mirror"]]

    strata = {
        "all_pairs": all_pairs,
        "high_similarity": hi_sim,
        "high_similarity_independent": hi_sim_indep,
        "high_similarity_independent_liquid": hi_sim_indep_liquid,
        "mirror_markets": mirrors,
    }
    out: dict = {}
    for name, ps in strata.items():
        out[name] = stats_summary(ps, label=name)
        out[name]["category_counts"] = categorical_breakdown(ps)

    out["counts"] = {
        "polymarket_active_binary": len(poly),
        "manifold_active_binary": len(mani),
        "candidates_before_dedupe": len(cands),
        "after_dedupe": len(pairs),
        "after_date_filter": len(aligned),
        "high_similarity_sim_geq_0_70": len(hi_sim),
        "high_similarity_independent": len(hi_sim_indep),
        "high_similarity_independent_liquid": len(hi_sim_indep_liquid),
        "mirror_markets": len(mirrors),
    }

    # Similarity bucket analysis
    buckets = []
    for lo, hi in [(0.40, 0.50), (0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]:
        sub = [p for p in aligned if lo <= p["similarity"] < hi]
        if not sub:
            continue
        g = np.array([p["gap_poly_minus_mani"] for p in sub])
        buckets.append({
            "lo": lo, "hi": hi, "n": len(sub),
            "mean_gap": float(g.mean()),
            "abs_mean_gap": float(np.abs(g).mean()),
            "frac_abs_gt_5pct": float((np.abs(g) > 0.05).mean()),
        })
    out["similarity_buckets"] = buckets

    with (RESULTS / "summary.json").open("w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
