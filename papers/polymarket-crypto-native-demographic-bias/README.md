# Wisdom of the Crypto-Native Crowd: Demographic Bias in Polymarket Aggregations

NeurIPS-style workshop paper auditing the calibration of Polymarket prediction-market prices, by topic and against a Manifold Markets baseline.

## Reproduce the numbers

From this directory:

```
pip install -r scripts/requirements.txt
python scripts/fetch_data.py    # pulls Polymarket + Manifold via public APIs (~25 min)
python scripts/analysis.py      # writes data/results_summary.json
python scripts/figure_1.py      # regenerates figure_1.png, figure_2.png, figure_3.png
```

Every number in `main.tex` is either pulled directly from `data/results_summary.json` or, where stated in the paper, computed from the raw CSVs that `fetch_data.py` writes into `data/`. The fetch hits two public, unauthenticated endpoints (Polymarket Gamma + CLOB, Manifold v0) and requires no API key. The raw CSVs are gitignored to keep the directory under the 50 MB budget; rerun `fetch_data.py` to regenerate them. To build the PDF, run `pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex` against a TeX Live install that includes `natbib` and `microtype`.
