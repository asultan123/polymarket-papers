"""Head-to-head accuracy analysis: Polymarket vs Cleveland Fed nowcast.

Inputs (produced by fetch_data.py):
    ../data/polymarket_events.json
    ../data/cleveland_fed_year.json
    ../data/cleveland_fed_month.json

Outputs:
    ../data/analysis_results.json  - all metrics + per-event rows
    ../data/per_event_table.csv    - human-readable per-event comparison
    ../figure_1.png                - calibration / accuracy figure
    Prints a Markdown-flavoured summary to stdout.

Methodology, briefly:
  * For each Polymarket event (a set of bracket markets for a single
    monthly CPI release) we recover a probability distribution over
    bracket midpoints from the YES prices.  The implied point forecast
    is sum(p_i * mid_i) -- the expected value under that distribution.
  * "T - 1 day" forecast: last price within 24h before resolution.
  * Cleveland Fed nowcast is the last published value before the BLS
    release for the same target month.  We match by target month
    (e.g. 2025-3) and by chart kind (`year` = YoY, `month` = MoM).
  * Realized value is the "Actual" series in the Cleveland Fed JSON
    (sourced from the BLS CPI press release).
  * We compute MAE, RMSE, and signed bias for both forecasters, plus
    bracket-level Brier score and a 4-bin calibration check.
"""

import csv
import json
import math
import os
import re
import statistics
import sys
from datetime import datetime, timezone

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..")

MONTH_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\b",
    re.IGNORECASE,
)
MONTHS = ["january", "february", "march", "april", "may", "june",
          "july", "august", "september", "october", "november", "december"]


def event_target_month(title, end_date_iso):
    """Resolve which CPI report month the event resolves on.

    Polymarket conventionally uses the month name in the title:
        "March Inflation - Annual"     -> March of the year that
                                          contains the end date.
        "US inflation >0.2% from Feb to March 2024?"
                                       -> March 2024 (the "to" month).
    For events whose title contains an explicit year, we use it;
    otherwise we infer the year from end_date (which is the BLS release
    date for the target month, ~10-15 days after month end).
    """
    title_lower = title.lower()
    matches = MONTH_RE.findall(title_lower)
    if not matches:
        return None
    # "Feb to March" pattern -> take last match
    if "from " in title_lower and " to " in title_lower:
        month_name = matches[-1]
    else:
        month_name = matches[0]
    month_num = MONTHS.index(month_name) + 1

    # year: parse from title or fall back to endDate
    year_match = re.search(r"\b(20\d\d)\b", title)
    if year_match:
        year = int(year_match.group(1))
    else:
        # endDate is the BLS release date - the target month is the
        # *previous* calendar month in nearly all cases.
        end_dt = datetime.fromisoformat(end_date_iso.replace("Z", "+00:00"))
        # If we're trying to match "March", we want year such that
        # March + ~30-45 days ~= end_dt.
        candidate_years = [end_dt.year, end_dt.year - 1, end_dt.year + 1]
        best = None
        best_gap = float("inf")
        for y in candidate_years:
            target_dt = datetime(y, month_num, 15, tzinfo=timezone.utc)
            gap = abs((end_dt - target_dt).days)
            if gap < best_gap and target_dt < end_dt + (end_dt - end_dt):
                best_gap = gap
                best = y
        year = best
    return (year, month_num)


def polymarket_point_forecast(event, ts_cutoff):
    """Return (mean_forecast, normalised_prob_dict, last_ts, has_prices).

    For each bracket market in the event, take its last price <= ts_cutoff.
    Bracket midpoints:
        ("le", v)      -> midpoint = v - 0.05
        ("eq", v)      -> midpoint = v
        ("ge", v)      -> midpoint = v + 0.05
        ("binary", v)  -> handled separately (single-threshold market)

    We normalise the YES probabilities so they sum to 1 (markets are
    usually mutually exclusive but quoted independently; sum can drift
    a few percent from 1 due to spread).
    """
    bracket_probs = []  # list of (label, midpoint, yes_prob)
    binary_marker = None
    last_ts_overall = 0
    for m in event["markets"]:
        b = m["bracket"]
        # find last price <= cutoff
        history = m.get("history") or []
        prev_prices = [(h["t"], h["p"]) for h in history if h["t"] <= ts_cutoff]
        if not prev_prices:
            # if no history before cutoff, skip
            continue
        ts, p = prev_prices[-1]
        last_ts_overall = max(last_ts_overall, ts)
        kind, v = b[0], b[1]
        if kind == "le":
            mid = v - 0.05
        elif kind == "ge":
            mid = v + 0.05
        elif kind == "eq":
            mid = v
        elif kind == "binary":
            binary_marker = (v, p)
            continue
        else:
            continue
        bracket_probs.append((m["question"], mid, p))

    if binary_marker is not None and not bracket_probs:
        # legacy yes/no market; return only the directional probability
        return None, {"binary_threshold": binary_marker[0],
                      "yes_prob": binary_marker[1]}, last_ts_overall, True
    if not bracket_probs:
        return None, {}, 0, False

    total = sum(p for _, _, p in bracket_probs)
    if total <= 0:
        return None, {}, last_ts_overall, False

    mean = sum(mid * (p / total) for _, mid, p in bracket_probs)
    return mean, {label: p / total for label, _, p in bracket_probs}, last_ts_overall, True


def cleveland_fed_lookup(charts, target_year, target_month):
    """Find the matching chart for (year, month) and return:
       (last_nowcast_value, last_nowcast_date, actual_value, actual_date)
    Values are floats; dates are 'MM/DD' (Cleveland uses MM/DD labels).
    Returns None for any unmatched field.
    """
    subcap = f"{target_year}-{target_month}"
    chart = None
    for c in charts:
        if c["chart"]["subcaption"] == subcap:
            chart = c
            break
    if chart is None:
        return None, None, None, None
    cats = [c["label"] for c in chart["categories"][0]["category"]]
    cpi_series = None
    actual_series = None
    for ds in chart["dataset"]:
        if ds["seriesname"] == "CPI Inflation":
            cpi_series = ds["data"]
        elif ds["seriesname"] == "Actual CPI Inflation":
            actual_series = ds["data"]
    if cpi_series is None:
        return None, None, None, None

    last_now = None
    last_date = None
    for i, pt in enumerate(cpi_series):
        v = pt.get("value", "")
        if v not in ("", None):
            last_now = float(v)
            last_date = cats[i]
    actual = None
    actual_date = None
    if actual_series is not None:
        for i, pt in enumerate(actual_series):
            v = pt.get("value", "")
            if v not in ("", None):
                actual = float(v)
                actual_date = cats[i]
                break
    return last_now, last_date, actual, actual_date


def cleveland_fed_at_cutoff(charts, target_year, target_month, cutoff_date_label):
    """Like cleveland_fed_lookup but returns the nowcast value at the
    last date <= cutoff_date_label (format MM/DD)."""
    subcap = f"{target_year}-{target_month}"
    chart = None
    for c in charts:
        if c["chart"]["subcaption"] == subcap:
            chart = c
            break
    if chart is None:
        return None, None
    cats = [c["label"] for c in chart["categories"][0]["category"]]
    cpi_series = None
    for ds in chart["dataset"]:
        if ds["seriesname"] == "CPI Inflation":
            cpi_series = ds["data"]
            break
    if cpi_series is None:
        return None, None
    # iterate left-to-right keeping the last value seen at a date <= cutoff
    last = None
    last_d = None
    for i, pt in enumerate(cpi_series):
        cd = cats[i]
        if cd > cutoff_date_label:  # string compare works for MM/DD when month matches
            break
        v = pt.get("value", "")
        if v not in ("", None):
            last = float(v)
            last_d = cd
    return last, last_d


def brier_score_distribution(prob_dict, realized, bracket_lookup):
    """Brier score for a multi-class bracketed forecast.

    bracket_lookup maps question -> (kind, value).  We determine which
    bracket the realized value fell in, set its true probability to 1
    and others to 0, then compute mean squared error.
    """
    true_label = None
    for q, (kind, v) in bracket_lookup.items():
        if kind == "le" and realized <= v + 1e-9:
            true_label = q
            break
        if kind == "ge" and realized >= v - 1e-9:
            true_label = q
            break
        if kind == "eq" and abs(realized - v) <= 0.05:
            true_label = q
            break
    if true_label is None:
        return None
    score = 0.0
    for q, p in prob_dict.items():
        truth = 1.0 if q == true_label else 0.0
        score += (p - truth) ** 2
    return score


def main():
    events = json.load(open(os.path.join(DATA_DIR, "polymarket_events.json")))
    cf_year = json.load(open(os.path.join(DATA_DIR, "cleveland_fed_year.json")))
    cf_month = json.load(open(os.path.join(DATA_DIR, "cleveland_fed_month.json")))

    rows = []
    skipped = []
    for e in events:
        tgt = event_target_month(e["title"], e["endDate"])
        if tgt is None:
            skipped.append((e["title"], "no target month"))
            continue
        year, month = tgt
        end_dt = datetime.fromisoformat(e["endDate"].replace("Z", "+00:00"))
        # T-1 day cutoff
        cutoff_ts = int(end_dt.timestamp()) - 86400

        mean_fc, prob_dict, last_ts, has_prices = polymarket_point_forecast(e, cutoff_ts)
        if not has_prices:
            skipped.append((e["title"], "no Polymarket prices"))
            continue

        charts = cf_year if e["kind"] == "annual" else cf_month
        # Cleveland Fed cutoff: date label MM/DD <= last_ts day-of-month
        cutoff_dt = datetime.fromtimestamp(cutoff_ts, tz=timezone.utc)
        cutoff_label = cutoff_dt.strftime("%m/%d")
        cf_now, cf_now_date = cleveland_fed_at_cutoff(
            charts, year, month, cutoff_label)
        # actual
        _, _, actual, actual_date = cleveland_fed_lookup(charts, year, month)
        if actual is None:
            skipped.append((e["title"], "no realized value from Cleveland Fed"))
            continue

        bracket_lookup = {m["question"]: tuple(m["bracket"])
                          for m in e["markets"]
                          if isinstance(m["bracket"], list)}
        brier = (brier_score_distribution(prob_dict, actual, bracket_lookup)
                 if mean_fc is not None else None)

        if mean_fc is None and "yes_prob" in prob_dict:
            # binary legacy market
            yes_prob = prob_dict["yes_prob"]
            thresh = prob_dict["binary_threshold"]
            realized_yes = 1.0 if actual > thresh + 1e-9 else 0.0
            row = {
                "event_id": e["id"],
                "title": e["title"],
                "kind": e["kind"],
                "target_year": year,
                "target_month": month,
                "volume": e.get("volume", 0),
                "polymarket_mean": None,
                "polymarket_yes_prob": yes_prob,
                "polymarket_threshold": thresh,
                "polymarket_binary_brier": (yes_prob - realized_yes) ** 2,
                "cleveland_fed_nowcast": cf_now,
                "cleveland_fed_date": cf_now_date,
                "realized": actual,
                "realized_date": actual_date,
            }
        else:
            row = {
                "event_id": e["id"],
                "title": e["title"],
                "kind": e["kind"],
                "target_year": year,
                "target_month": month,
                "volume": e.get("volume", 0),
                "polymarket_mean": mean_fc,
                "polymarket_brier": brier,
                "cleveland_fed_nowcast": cf_now,
                "cleveland_fed_date": cf_now_date,
                "realized": actual,
                "realized_date": actual_date,
                "polymarket_err": (mean_fc - actual) if mean_fc is not None else None,
                "cleveland_fed_err": ((cf_now - actual)
                                     if (cf_now is not None) else None),
            }
        rows.append(row)

    # Compute summary stats on rows with both forecasts
    paired = [r for r in rows
              if r.get("polymarket_mean") is not None
              and r.get("cleveland_fed_nowcast") is not None
              and r.get("realized") is not None]

    def stats(errs):
        if not errs:
            return {"n": 0}
        abs_e = [abs(x) for x in errs]
        sq_e = [x * x for x in errs]
        return {
            "n": len(errs),
            "mean_err": statistics.mean(errs),
            "mae": statistics.mean(abs_e),
            "rmse": math.sqrt(statistics.mean(sq_e)),
            "max_abs_err": max(abs_e),
        }

    poly_errs = [r["polymarket_err"] for r in paired]
    cf_errs = [r["cleveland_fed_err"] for r in paired]

    summary = {
        "n_events_total": len(events),
        "n_rows_with_forecasts": len(rows),
        "n_paired_comparisons": len(paired),
        "polymarket": stats(poly_errs),
        "cleveland_fed": stats(cf_errs),
    }

    # paired difference test (Polymarket - CleveFed of abs errors)
    if len(paired) > 1:
        diffs = [abs(p) - abs(c) for p, c in zip(poly_errs, cf_errs)]
        from statistics import mean, stdev
        d_mean = mean(diffs)
        d_sd = stdev(diffs) if len(diffs) > 1 else 0
        d_se = d_sd / math.sqrt(len(diffs)) if d_sd > 0 else 0
        t_stat = d_mean / d_se if d_se > 0 else 0.0
        # crude two-sided p approx via Student's t with n-1 df using scipy
        try:
            from scipy import stats as scs
            p_val = float(2 * (1 - scs.t.cdf(abs(t_stat), df=len(diffs) - 1)))
        except Exception:
            p_val = None
        summary["paired_test"] = {
            "n": len(diffs),
            "mean_diff_abs_err_poly_minus_cf": d_mean,
            "sd": d_sd,
            "t_stat": t_stat,
            "p_value": p_val,
        }
        # correlation between Polymarket point forecast and realised
        poly_vals = [r["polymarket_mean"] for r in paired]
        real_vals = [r["realized"] for r in paired]
        cf_vals = [r["cleveland_fed_nowcast"] for r in paired]
        try:
            from scipy.stats import pearsonr, spearmanr
            pr_p, pp_p = pearsonr(poly_vals, real_vals)
            pr_cf, pp_cf = pearsonr(cf_vals, real_vals)
            summary["pearson_poly_vs_realised"] = {"r": float(pr_p), "p": float(pp_p)}
            summary["pearson_cf_vs_realised"] = {"r": float(pr_cf), "p": float(pp_cf)}
        except Exception as ex:
            print(f"pearson fail: {ex}", file=sys.stderr)

    # Split by kind
    summary["by_kind"] = {}
    for kind in ("annual", "monthly"):
        sub = [r for r in paired if r["kind"] == kind]
        if not sub:
            continue
        poly_sub = [r["polymarket_err"] for r in sub]
        cf_sub = [r["cleveland_fed_err"] for r in sub]
        summary["by_kind"][kind] = {
            "n": len(sub),
            "polymarket": stats(poly_sub),
            "cleveland_fed": stats(cf_sub),
        }

    # Brier scores for Polymarket bracket distributions
    brier_vals = [r["polymarket_brier"] for r in rows
                  if r.get("polymarket_brier") is not None]
    if brier_vals:
        summary["polymarket_brier_mean"] = statistics.mean(brier_vals)
        summary["polymarket_brier_n"] = len(brier_vals)
        summary["polymarket_brier_max"] = max(brier_vals)
        summary["polymarket_brier_min"] = min(brier_vals)

    # ---- Binary calibration on the legacy "inflation >X%" markets ----
    binary_rows = [r for r in rows
                   if r.get("polymarket_yes_prob") is not None
                   and r.get("realized") is not None
                   and r.get("polymarket_threshold") is not None]
    if binary_rows:
        bin_briers = []
        for r in binary_rows:
            realized_yes = 1.0 if r["realized"] > r["polymarket_threshold"] + 1e-9 else 0.0
            bin_briers.append((r["polymarket_yes_prob"] - realized_yes) ** 2)
        summary["binary_market_brier_mean"] = statistics.mean(bin_briers)
        summary["binary_market_n"] = len(binary_rows)

    # ---- Robustness: drop the single largest |poly-cf| difference ----
    if len(paired) > 5:
        sorted_paired = sorted(paired,
            key=lambda r: abs(abs(r["polymarket_err"]) - abs(r["cleveland_fed_err"])),
            reverse=True)
        trimmed = sorted_paired[1:]
        ptrim = [r["polymarket_err"] for r in trimmed]
        ctrim = [r["cleveland_fed_err"] for r in trimmed]
        summary["robust_drop_top1"] = {
            "n": len(trimmed),
            "polymarket": stats(ptrim),
            "cleveland_fed": stats(ctrim),
            "dropped": sorted_paired[0]["title"],
        }

    # ---- Dedupe: when Polymarket ran multiple events for the same
    # target month (e.g. original + "Higher Brackets") prefer the
    # "Higher/Lower Brackets" addendum, since that one was created
    # specifically to cover a bracket range the original missed.
    # This avoids penalising Polymarket for an obvious market-design
    # artefact (truncated bracket grid) the operator itself fixed.
    def _is_addendum(title):
        return "(Higher Brackets)" in title or "(Lower Brackets)" in title

    by_key = {}
    for r in paired:
        key = (r["target_year"], r["target_month"], r["kind"])
        if key not in by_key:
            by_key[key] = r
        else:
            # prefer addendum if exists, else higher-volume
            cur = by_key[key]
            if _is_addendum(r["title"]) and not _is_addendum(cur["title"]):
                by_key[key] = r
            elif _is_addendum(r["title"]) == _is_addendum(cur["title"]):
                if r["volume"] > cur["volume"]:
                    by_key[key] = r
    dedup = list(by_key.values())
    if len(dedup) != len(paired):
        pdv = [r["polymarket_err"] for r in dedup]
        cdv = [r["cleveland_fed_err"] for r in dedup]
        ddv = [abs(p) - abs(c) for p, c in zip(pdv, cdv)]
        d_mean = statistics.mean(ddv)
        d_sd = statistics.stdev(ddv) if len(ddv) > 1 else 0
        d_se = d_sd / math.sqrt(len(ddv)) if d_sd > 0 else 0
        t_d = d_mean / d_se if d_se > 0 else 0.0
        try:
            from scipy import stats as scs
            p_d = float(2 * (1 - scs.t.cdf(abs(t_d), df=len(ddv) - 1)))
        except Exception:
            p_d = None
        summary["dedup_highest_volume"] = {
            "n": len(dedup),
            "polymarket": stats(pdv),
            "cleveland_fed": stats(cdv),
            "mean_diff_abs_err": d_mean,
            "t_stat": t_d,
            "p_value": p_d,
        }

    # ---- Volume-tier breakdown ----
    if len(paired) > 10:
        med_vol = sorted([r["volume"] for r in paired])[len(paired) // 2]
        hi = [r for r in paired if r["volume"] >= med_vol]
        lo = [r for r in paired if r["volume"] < med_vol]
        for label, sub in (("high_volume", hi), ("low_volume", lo)):
            if not sub: continue
            summary[label] = {
                "n": len(sub),
                "median_volume": statistics.median([r["volume"] for r in sub]),
                "polymarket": stats([r["polymarket_err"] for r in sub]),
                "cleveland_fed": stats([r["cleveland_fed_err"] for r in sub]),
            }

    # Save
    out = {"summary": summary, "rows": rows, "skipped": skipped}
    with open(os.path.join(DATA_DIR, "analysis_results.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)

    # CSV table
    csv_path = os.path.join(DATA_DIR, "per_event_table.csv")
    with open(csv_path, "w", newline="") as f:
        cols = ["target_year", "target_month", "kind", "title",
                "polymarket_mean", "cleveland_fed_nowcast", "realized",
                "polymarket_err", "cleveland_fed_err",
                "polymarket_brier", "volume"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Print summary
    print("\n========== ANALYSIS SUMMARY ==========")
    print(f"Events fetched           : {summary['n_events_total']}")
    print(f"Rows with any forecast   : {summary['n_rows_with_forecasts']}")
    print(f"Paired (both + realised) : {summary['n_paired_comparisons']}")
    if summary["n_paired_comparisons"] > 0:
        p = summary["polymarket"]
        c = summary["cleveland_fed"]
        print(f"\nPolymarket  : n={p['n']} MAE={p['mae']:.3f}pp RMSE={p['rmse']:.3f}pp bias={p['mean_err']:+.3f}pp")
        print(f"Cleveland Fed: n={c['n']} MAE={c['mae']:.3f}pp RMSE={c['rmse']:.3f}pp bias={c['mean_err']:+.3f}pp")
        if "paired_test" in summary:
            pt = summary["paired_test"]
            print(f"Paired t-test (|poly|-|cf|): mean_diff={pt['mean_diff_abs_err_poly_minus_cf']:+.3f}pp  t={pt['t_stat']:.2f}  p={pt['p_value']}")
        if "pearson_poly_vs_realised" in summary:
            print(f"Pearson(Polymarket, realised)   r={summary['pearson_poly_vs_realised']['r']:.3f}  p={summary['pearson_poly_vs_realised']['p']:.4f}")
            print(f"Pearson(CleveFed,   realised)   r={summary['pearson_cf_vs_realised']['r']:.3f}  p={summary['pearson_cf_vs_realised']['p']:.4f}")
    if "polymarket_brier_mean" in summary:
        print(f"\nPolymarket Brier (bracket form): mean={summary['polymarket_brier_mean']:.3f}  n={summary['polymarket_brier_n']}")
    for kind, k in summary.get("by_kind", {}).items():
        print(f"\n-- {kind} (n={k['n']}) --")
        print(f"  Polymarket   MAE={k['polymarket']['mae']:.3f}pp RMSE={k['polymarket']['rmse']:.3f}pp bias={k['polymarket']['mean_err']:+.3f}pp")
        print(f"  CleveFed     MAE={k['cleveland_fed']['mae']:.3f}pp RMSE={k['cleveland_fed']['rmse']:.3f}pp bias={k['cleveland_fed']['mean_err']:+.3f}pp")

    print(f"\nSkipped events ({len(skipped)}):")
    for t, r in skipped[:10]:
        print(f"  - {t}: {r}")

    print("\nPer-event table written to data/per_event_table.csv")


if __name__ == "__main__":
    main()
