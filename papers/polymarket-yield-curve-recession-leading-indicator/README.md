# Polymarket as a leading indicator for US recessions vs. the Treasury yield curve

LaTeX source plus the data-fetching and analysis pipeline that produces every
number in the paper.

## Reproducing the numbers

```bash
cd papers/polymarket-yield-curve-recession-leading-indicator
python3 -m venv .venv && source .venv/bin/activate    # optional
pip install -r scripts/requirements.txt
python3 scripts/fetch_data.py        # ~30s, writes data/*.csv
python3 scripts/analysis.py          # writes results/summary.json + prints to stdout
python3 scripts/figures.py           # writes figures/figure_{1..4}*.png
```

`fetch_data.py` hits three public endpoints: the Polymarket Gamma API
(`gamma-api.polymarket.com/events`), the Polymarket CLOB price-history endpoint
(`clob.polymarket.com/prices-history`), and the Manifold bets endpoint
(`api.manifold.markets/v0/bets`). Treasury yields come from Yahoo Finance via
the `yfinance` package. No API keys are required.

`analysis.py` reproduces the per-market statistics, Granger tests, ADF tests,
event-study window, and cross-platform correlations that appear in
Table 1, Table 2, and Section 5 of the paper. `figures.py` reproduces the four
figures.

To rebuild the PDF (requires a TeX distribution; none was available in the
environment used to write this paper):

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```
