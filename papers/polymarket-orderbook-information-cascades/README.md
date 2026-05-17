# Detecting Information Cascades in Polymarket Order Books

NeurIPS-format workshop paper. All numbers and figures are reproducible from two scripts and the public Polymarket APIs (no auth, no API key).

## How to reproduce

```bash
cd scripts
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python fetch_data.py     # ~5 min, writes ../data/ (~21 MB)
python analysis.py       # ~30 sec, writes ../results/
python make_figures.py   # ~5 sec, writes ../figures/
```

`fetch_data.py` pulls the 600 highest-volume binary markets from `gamma-api.polymarket.com`, keeps the top 40 by USD volume, and for each one pulls the public CLOB 12-hour midprice history plus up to the most-recent 3,500 taker trades from `data-api.polymarket.com`. `analysis.py` computes per-market diagnostics (Fano factor, Hawkes branching ratio via the Hardiman--Bercot--Bouchaud moment estimator, trade-sign autocorrelation, burst counts, calibration Brier scores). `make_figures.py` regenerates the four figures cited in `main.tex`.

Re-running on a different date will produce slightly different numbers because Polymarket's market list is dynamic. The fetched data is *not* committed; the scripts are the source of truth.

## Files

- `main.tex`, `references.bib`, `neurips_2024.sty` — paper source
- `figures/` — paper figures (PNG, kept in-repo)
- `scripts/fetch_data.py`, `scripts/analysis.py`, `scripts/make_figures.py`
- `scripts/requirements.txt` — pinned versions

No PDF is committed; the container has no LaTeX. Compile locally with `pdflatex main && bibtex main && pdflatex main && pdflatex main`.
