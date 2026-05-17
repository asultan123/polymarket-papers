"""Compute every numeric value referenced in the paper.

Reads CSVs produced by fetch_data.py and writes numbers.json + numbers.txt
into ../data/. The paper text quotes from numbers.txt verbatim; if a number
is not in numbers.txt, it is not in the paper.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
PRICES = DATA / "prices"
YAHOO = DATA / "yahoo"

# Event timeline (all UTC).
EVT = {
    "silvergate_close": pd.Timestamp("2023-03-08 21:00", tz="UTC"),  # 8 Mar press release
    "svb_8k": pd.Timestamp("2023-03-08 21:15", tz="UTC"),            # 8-K after-hours
    "svb_run_begins": pd.Timestamp("2023-03-09 14:30", tz="UTC"),    # 9 Mar open
    "svb_fdic_takeover": pd.Timestamp("2023-03-10 16:30", tz="UTC"), # 10 Mar 12:30 ET
    "signature_close": pd.Timestamp("2023-03-12 22:00", tz="UTC"),   # 12 Mar evening
    "btfp_announce": pd.Timestamp("2023-03-12 22:15", tz="UTC"),     # Sun evening
    "cs_ubs_deal": pd.Timestamp("2023-03-19 18:00", tz="UTC"),       # 19 Mar evening
    "fomc_march": pd.Timestamp("2023-03-22 18:00", tz="UTC"),        # 22 Mar 2 PM ET
}


def load_market(mid: str) -> pd.DataFrame:
    df = pd.read_csv(PRICES / f"{mid}.csv")
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def load_yahoo(ticker: str) -> pd.DataFrame:
    # yfinance CSVs have 3 header lines we want to skip; index col is Date.
    df = pd.read_csv(YAHOO / f"{ticker}.csv", skiprows=3,
                     names=["Date", "Adj Close", "Close", "High", "Low", "Open", "Volume"])
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date").sort_index()


def first_crossing(df: pd.DataFrame, level: float, direction: str) -> pd.Timestamp | None:
    if direction == "up":
        m = df["p"] >= level
    else:
        m = df["p"] <= level
    if not m.any():
        return None
    return df.loc[m, "ts"].iloc[0]


def main():
    cands = json.loads((DATA / "candidates.json").read_text())
    by_q = {c["question"]: c for c in cands}
    nums: dict = {}

    # ---- 1. Coverage / volume summary
    total_vol = sum((c.get("volumeNum") or 0) for c in cands)
    svb_direct_qs = [
        "Will SVB fail?",
        "Will SVB be acquired by Monday night?",
        "Will another US bank fail in March?",
        "Will uninsured SVB depositors get all their money back by June 30?",
        "Will uninsured SVB depositors get all their money back by EOY?",
    ]
    contagion_qs = [
        "Will Silvergate announce it is filing for bankruptcy by March 31, 2023?",
        "Will a third US bank fail by March 17?",
        "Will Credit Suisse fail by March 31?",
        "Will First Republic Bank fail by March 17?",
        "Will First Republic Bank fail by March 31?",
        "Will Bank of America fail by March 17?",
    ]
    fed_qs = [q for q in by_q if q.startswith("Will the Fed")]
    nums["n_markets_total"] = len(cands)
    nums["n_markets_svb_direct"] = len(svb_direct_qs)
    nums["n_markets_contagion"] = len(contagion_qs)
    nums["n_markets_fed"] = len(fed_qs)
    nums["volume_total_usd"] = float(total_vol)
    nums["volume_svb_direct_usd"] = float(sum((by_q[q].get("volumeNum") or 0) for q in svb_direct_qs))
    nums["volume_contagion_usd"] = float(sum((by_q[q].get("volumeNum") or 0) for q in contagion_qs))
    nums["volume_fed_usd"] = float(sum((by_q[q].get("volumeNum") or 0) for q in fed_qs))

    # ---- 2. SVB failure market: speed of price discovery (minute fidelity)
    svb_min_path = PRICES / "249094_min.csv"
    if svb_min_path.exists():
        svb = pd.read_csv(svb_min_path)
        svb["ts"] = pd.to_datetime(svb["ts"], utc=True)
    else:
        svb = load_market(by_q["Will SVB fail?"]["id"])
    nums["svb_n_obs_minute"] = int(len(svb))
    nums["svb_first_seen"] = str(svb["ts"].iloc[0])
    nums["svb_creation_price"] = float(svb["p"].iloc[0])
    nums["svb_max_price"] = float(svb["p"].max())
    # Crossings
    for thr in (0.55, 0.70, 0.90, 0.95, 0.99):
        t = first_crossing(svb, thr, "up")
        nums[f"svb_cross_{thr:.2f}_ts"] = str(t) if t is not None else None
        if t is not None:
            nums[f"svb_minutes_creation_to_{thr:.2f}"] = (t - svb["ts"].iloc[0]).total_seconds() / 60
    # Fast discovery phase: time between first crossing 0.55 and first crossing 0.95
    c55 = first_crossing(svb, 0.55, "up")
    c95 = first_crossing(svb, 0.95, "up")
    if c55 is not None and c95 is not None:
        nums["svb_minutes_0.55_to_0.95"] = (c95 - c55).total_seconds() / 60

    # ---- 3. SVB acquisition market: dynamics
    acq = load_market(by_q["Will SVB be acquired by Monday night?"]["id"])
    nums["svb_acq_n_obs"] = int(len(acq))
    nums["svb_acq_first_price"] = float(acq["p"].iloc[0])
    nums["svb_acq_last_price"] = float(acq["p"].iloc[-1])
    nums["svb_acq_max_price"] = float(acq["p"].max())
    nums["svb_acq_max_ts"] = str(acq.loc[acq["p"].idxmax(), "ts"])

    # ---- 4. Contagion: third-bank-fail market peak
    third = load_market(by_q["Will a third US bank fail by March 17?"]["id"])
    nums["third_bank_n_obs"] = int(len(third))
    nums["third_bank_max_price"] = float(third["p"].max())
    nums["third_bank_max_ts"] = str(third.loc[third["p"].idxmax(), "ts"])
    nums["third_bank_last_price"] = float(third["p"].iloc[-1])
    # FRC and BoA contagion
    frc = load_market(by_q["Will First Republic Bank fail by March 17?"]["id"])
    nums["frc_max_price"] = float(frc["p"].max())
    nums["frc_n_obs"] = int(len(frc))
    boa = load_market(by_q["Will Bank of America fail by March 17?"]["id"])
    nums["boa_max_price"] = float(boa["p"].max())
    nums["boa_n_obs"] = int(len(boa))
    # Credit Suisse
    cs = load_market(by_q["Will Credit Suisse fail by March 31?"]["id"])
    nums["cs_first_price"] = float(cs["p"].iloc[0])
    nums["cs_max_price"] = float(cs["p"].max())
    nums["cs_max_ts"] = str(cs.loc[cs["p"].idxmax(), "ts"])
    # Trim CS to before UBS deal so we don't include the post-resolution flatline
    cs_pre = cs[cs["ts"] <= EVT["cs_ubs_deal"]]
    nums["cs_n_obs_pre_ubs"] = int(len(cs_pre))
    nums["cs_max_price_pre_ubs"] = float(cs_pre["p"].max())

    # ---- 5. Fed-pivot diff-in-diff
    # For each "Will the Fed ... March meeting" market, compute mean price
    # over pre-SVB window (Feb 14 - Mar 7) and post-SVB / pre-FOMC window (Mar 10 - Mar 22).
    pre0 = pd.Timestamp("2023-02-14", tz="UTC")
    pre1 = pd.Timestamp("2023-03-07 23:59", tz="UTC")
    post0 = pd.Timestamp("2023-03-10", tz="UTC")
    post1 = pd.Timestamp("2023-03-22 12:00", tz="UTC")  # before FOMC announce
    fed_pivot = {}
    for q_short, q_full in [
        ("hike_0bp_march", "Will the Fed increase interest rates by 0 bps after its March meeting?"),
        ("hike_25bp_march", "Will the Fed increase interest rates by 25 bps after its March meeting?"),
        ("hike_50bp_march", "Will the Fed increase interest rates by 50 bps after its March meeting?"),
        ("cut_25bp_march", "Will the Fed decrease interest rates by 25 bps after its March meeting?"),
    ]:
        df = load_market(by_q[q_full]["id"])
        pre_mask = (df["ts"] >= pre0) & (df["ts"] <= pre1)
        post_mask = (df["ts"] >= post0) & (df["ts"] <= post1)
        fed_pivot[q_short] = {
            "pre_mean": float(df.loc[pre_mask, "p"].mean()),
            "post_mean": float(df.loc[post_mask, "p"].mean()),
            "pre_last": float(df.loc[df["ts"] <= pre1, "p"].iloc[-1]) if (df["ts"] <= pre1).any() else None,
            "post_first": float(df.loc[df["ts"] >= post0, "p"].iloc[0]) if (df["ts"] >= post0).any() else None,
            "n_pre": int(pre_mask.sum()),
            "n_post": int(post_mask.sum()),
        }
    nums["fed_pivot"] = fed_pivot

    # Pre-SVB hawkish drift summary (start vs end vs extreme)
    extremes = {}
    for q_short, q_full, agg in [
        ("hike_50bp_march_pre_max", "Will the Fed increase interest rates by 50 bps after its March meeting?", "max"),
        ("hike_25bp_march_pre_min", "Will the Fed increase interest rates by 25 bps after its March meeting?", "min"),
    ]:
        df = load_market(by_q[q_full]["id"])
        pre = df[(df["ts"] >= pre0) & (df["ts"] <= pre1)]
        v = pre["p"].max() if agg == "max" else pre["p"].min()
        idx = pre["p"].idxmax() if agg == "max" else pre["p"].idxmin()
        extremes[q_short] = {"value": float(v),
                              "ts": str(pre.loc[idx, "ts"]),
                              "first_pre": float(pre["p"].iloc[0]),
                              "last_pre": float(pre["p"].iloc[-1])}
    nums["fed_pivot_pre_extremes"] = extremes

    # ---- 6. "Will the Fed cut rates in 2023?" path
    fc23 = load_market(by_q["Will the Fed cut rates in 2023?"]["id"])
    # Window stats
    pre_fc = fc23[(fc23["ts"] >= pre0) & (fc23["ts"] <= pre1)]
    post_fc = fc23[(fc23["ts"] >= post0) & (fc23["ts"] <= post1)]
    nums["fed_cut_2023_pre_svb_mean"] = float(pre_fc["p"].mean())
    nums["fed_cut_2023_post_svb_mean"] = float(post_fc["p"].mean())
    nums["fed_cut_2023_pre_svb_last"] = float(pre_fc["p"].iloc[-1]) if len(pre_fc) else None
    nums["fed_cut_2023_post_svb_first"] = float(post_fc["p"].iloc[0]) if len(post_fc) else None
    nums["fed_cut_2023_peak"] = float(fc23["p"].max())
    nums["fed_cut_2023_peak_ts"] = str(fc23.loc[fc23["p"].idxmax(), "ts"])
    nums["fed_cut_2023_final"] = float(fc23["p"].iloc[-1])

    # Within-day price changes around the shock
    # Average daily price by UTC date over March 6-15, to show the path
    fc23["date"] = fc23["ts"].dt.tz_convert("US/Eastern").dt.date
    daily_fc = fc23.groupby("date")["p"].agg(["first", "last", "mean", "min", "max"]).reset_index()
    daily_fc_window = daily_fc[(daily_fc["date"] >= pd.Timestamp("2023-03-06").date()) &
                                (daily_fc["date"] <= pd.Timestamp("2023-03-22").date())]
    nums["fed_cut_2023_daily_path"] = daily_fc_window.assign(date=daily_fc_window["date"].astype(str)).to_dict(orient="records")

    # Peak intraday jump in fc23 — compute for each day the (max - min) on Mar 8-15
    jumps = []
    for d, sub in fc23.groupby("date"):
        jumps.append({"date": str(d), "intraday_range": float(sub["p"].max() - sub["p"].min()),
                      "open": float(sub["p"].iloc[0]), "close": float(sub["p"].iloc[-1]),
                      "n": int(len(sub))})
    jumps_df = pd.DataFrame(jumps)
    big = jumps_df.sort_values("intraday_range", ascending=False).head(5)
    nums["fed_cut_2023_top5_intraday_jumps"] = big.to_dict(orient="records")

    # ---- 7. Daily-resolution correlation: Polymarket Fed-cut prob vs KRE
    kre = load_yahoo("KRE")
    spy = load_yahoo("SPY")
    tlt = load_yahoo("TLT")
    tnx = load_yahoo("TNX")
    vix = load_yahoo("VIX")

    # Make a daily series of fc23 (UTC date) and join
    fc_daily = (fc23.set_index("ts")["p"].resample("1D").mean().to_frame("pm_fed_cut_2023"))
    fc_daily.index = fc_daily.index.tz_convert(None)
    merged = fc_daily.join(kre[["Close"]].rename(columns={"Close": "KRE_close"}), how="inner")
    merged = merged.join(spy[["Close"]].rename(columns={"Close": "SPY_close"}), how="inner")
    merged = merged.join(tlt[["Close"]].rename(columns={"Close": "TLT_close"}), how="inner")
    merged = merged.join(tnx[["Close"]].rename(columns={"Close": "TNX_close"}), how="inner")
    merged = merged.join(vix[["Close"]].rename(columns={"Close": "VIX_close"}), how="inner")

    # Daily returns / changes
    for col in ["KRE_close", "SPY_close", "TLT_close", "TNX_close", "VIX_close", "pm_fed_cut_2023"]:
        merged[col + "_d"] = merged[col].diff()

    # Restrict to March-April window
    win = merged[(merged.index >= "2023-03-01") & (merged.index <= "2023-04-30")].dropna()
    nums["corr_window_start"] = str(win.index.min().date())
    nums["corr_window_end"] = str(win.index.max().date())
    nums["corr_n_days"] = int(len(win))

    # Pearson on day-over-day changes
    corrs = {}
    for tk in ["KRE", "SPY", "TLT", "TNX", "VIX"]:
        r, p = stats.pearsonr(win["pm_fed_cut_2023_d"], win[f"{tk}_close_d"])
        corrs[tk] = {"pearson_r": float(r), "p_value": float(p), "n": int(len(win))}
    nums["daily_corr_fed_cut_vs"] = corrs

    # Also same in levels (not changes) over the SVB shock window
    shock = merged[(merged.index >= "2023-03-08") & (merged.index <= "2023-03-31")].dropna()
    shock_corrs = {}
    for tk in ["KRE", "SPY", "TLT", "TNX", "VIX"]:
        r, p = stats.pearsonr(shock["pm_fed_cut_2023"], shock[f"{tk}_close"])
        shock_corrs[tk] = {"pearson_r": float(r), "p_value": float(p), "n": int(len(shock))}
    nums["shock_window_corr_fed_cut_vs"] = shock_corrs
    nums["shock_window_n"] = int(len(shock))
    nums["shock_window"] = "2023-03-08 to 2023-03-31"

    # ---- 8. KRE / SPY / TLT moves around SVB
    def pct(a, b):
        return float((b - a) / a * 100)
    mar8 = kre.loc["2023-03-08"]["Close"]
    mar13 = kre.loc["2023-03-13"]["Close"]
    mar17 = kre.loc["2023-03-17"]["Close"]
    nums["KRE_mar08_close"] = float(mar8)
    nums["KRE_mar13_close"] = float(mar13)
    nums["KRE_mar17_close"] = float(mar17)
    nums["KRE_pct_mar8_to_mar13"] = pct(mar8, mar13)
    nums["KRE_pct_mar8_to_mar17"] = pct(mar8, mar17)

    tlt_mar8 = tlt.loc["2023-03-08"]["Close"]
    tlt_mar13 = tlt.loc["2023-03-13"]["Close"]
    nums["TLT_pct_mar8_to_mar13"] = pct(tlt_mar8, tlt_mar13)

    tnx_mar8 = tnx.loc["2023-03-08"]["Close"]
    tnx_mar13 = tnx.loc["2023-03-13"]["Close"]
    nums["TNX_mar8"] = float(tnx_mar8)
    nums["TNX_mar13"] = float(tnx_mar13)
    nums["TNX_change_mar8_to_mar13_bps"] = float((tnx_mar13 - tnx_mar8) * 100)

    # ---- 9. Outcome reality-check (did markets resolve correctly?)
    # Look at the "outcomePrices" field which stores [yes, no] at resolution.
    outcomes = {}
    for q, m in by_q.items():
        prices = json.loads(m["outcomePrices"])
        outcomes[q] = {"yes": float(prices[0]), "no": float(prices[1])}
    nums["resolutions"] = outcomes

    # ---- 10. Pre-trend test: was the Fed-pivot already underway pre-Silvergate?
    # Compare avg P(0bp hike) over Feb 14-28 vs Mar 1-7
    h0_march = load_market(by_q["Will the Fed increase interest rates by 0 bps after its March meeting?"]["id"])
    w1 = h0_march[(h0_march["ts"] >= pd.Timestamp("2023-02-14", tz="UTC")) &
                  (h0_march["ts"] <= pd.Timestamp("2023-02-28 23:59", tz="UTC"))]["p"]
    w2 = h0_march[(h0_march["ts"] >= pd.Timestamp("2023-03-01", tz="UTC")) &
                  (h0_march["ts"] <= pd.Timestamp("2023-03-07 23:59", tz="UTC"))]["p"]
    w3 = h0_march[(h0_march["ts"] >= pd.Timestamp("2023-03-08", tz="UTC")) &
                  (h0_march["ts"] <= pd.Timestamp("2023-03-14 23:59", tz="UTC"))]["p"]
    nums["hike0_feb_14_28_mean"] = float(w1.mean())
    nums["hike0_mar_01_07_mean"] = float(w2.mean())
    nums["hike0_mar_08_14_mean"] = float(w3.mean())

    # ---- Save numbers
    (DATA / "numbers.json").write_text(json.dumps(nums, indent=2, default=str))

    # Pretty text dump
    lines = []
    def dump(d, indent=0):
        pad = "  " * indent
        for k, v in d.items():
            if isinstance(v, dict):
                lines.append(f"{pad}{k}:")
                dump(v, indent + 1)
            elif isinstance(v, list):
                lines.append(f"{pad}{k}: [{len(v)} items]")
                for i, item in enumerate(v[:10]):
                    if isinstance(item, dict):
                        s = ", ".join(f"{kk}={vv}" for kk, vv in item.items())
                        lines.append(f"{pad}  [{i}] {s}")
                    else:
                        lines.append(f"{pad}  [{i}] {item}")
            elif isinstance(v, float):
                lines.append(f"{pad}{k}: {v:.6g}")
            else:
                lines.append(f"{pad}{k}: {v}")
    dump(nums)
    (DATA / "numbers.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
