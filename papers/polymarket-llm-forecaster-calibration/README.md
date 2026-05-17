# Polymarket as a calibration benchmark for LLM-based forecasting agents

This directory contains the LaTeX source, bibliography, NeurIPS 2024
style file, fetch and analysis scripts, and generated figures for the
paper. To reproduce every number that appears in `main.tex`:

```bash
cd scripts
pip install -r requirements.txt
python fetch_data.py --target-n 3000 --min-volume 1000 --max-out 1500
python fetch_manifold.py --pages 15
python analysis.py
```

`fetch_data.py` calls Polymarket's public Gamma + CLOB APIs (no auth)
and produces `data/markets.csv` with 1,200--1,300 resolved binary
markets and their historical mid-prices at the relevant horizons.
`fetch_manifold.py` similarly pulls `data/manifold.csv` from the
Manifold v0 API. `analysis.py` consumes both CSVs and writes
`data/results.json` plus `figure_1.png` and `figure_2.png`; every
Brier score, log loss, ECE, and reliability slope cited in the paper
appears in `results.json`. Total wall-clock runtime is roughly 15
minutes (most of it is rate-limited CLOB calls). The exact numbers
will drift slightly between runs as new markets resolve and the CLOB
samples timestamps non-deterministically; the headline numbers should
not change by more than 0.005 in Brier.
