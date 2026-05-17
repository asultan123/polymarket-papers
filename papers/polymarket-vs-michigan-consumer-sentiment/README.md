# Liquidity-weighted Polymarket vs Michigan Consumer Sentiment: a 2022-2025 backtest

Workshop-format paper testing whether a liquidity-weighted index built from
Polymarket macro contracts (recession, inflation, Fed-rate, jobs) tracks the
University of Michigan Index of Consumer Sentiment over Feb 2023 - Apr 2026
(the period for which we have overlapping data). Headline result: full index
is uncorrelated; a recession-only sub-index co-moves contemporaneously
(`r_diff = -0.68`, `R^2 = 0.46`, `p < 1e-8`, `n = 18`); Granger tests are
null. See `main.tex` for the writeup.

## Reproducing the numbers

```bash
cd papers/polymarket-vs-michigan-consumer-sentiment
pip install -r scripts/requirements.txt           # 6 deps, ~30 s
python -B scripts/fetch_data.py                   # ~4 min: ICS + Polymarket pull
python -B scripts/analysis.py                     # ~30 s: index + tests
python -B scripts/figure_1.py                     # writes figure_1.png
```

`fetch_data.py` writes `data/michigan.csv`, `data/macro_markets.csv`, and
`data/macro_prices.csv` (~9 MB combined; gitignored). `analysis.py` writes
`data/results.json` and the merged monthly file. Every number in the paper
(coverage counts, correlations, OLS coefficients, Granger p-values, lead-lag
profile) lives in `data/results.json`. Use `-B` to avoid stale bytecode caches
that can mask edits to `analysis.py:polarity`.

## File map

- `main.tex` -- paper source (NeurIPS 2024 style, 8-page workshop format)
- `references.bib` -- 16 references, all verified at the publisher's URL
- `neurips_2024.sty` -- NeurIPS 2024 style file
- `figure_1.png` -- three-panel figure (time series, full scatter, recession scatter)
- `scripts/fetch_data.py` -- Michigan SRC + Polymarket Gamma/CLOB pull
- `scripts/analysis.py` -- index construction + all statistical tests
- `scripts/figure_1.py` -- figure generator
- `scripts/requirements.txt` -- pinned versions used to produce the paper numbers

No PDF is committed; the paper is `.tex` only.
