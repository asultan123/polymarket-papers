# GDP nowcasting via Polymarket aggregations: a Kalman filter approach

## Reproducing the numbers in this paper

```bash
pip install -r scripts/requirements.txt
python scripts/fetch_data.py    # ~30 s; writes data/* (public APIs, no key)
python scripts/analysis.py      # ~5 s;  writes analysis_outputs/results.json + figures
```

Every number in `main.tex` is loaded from `analysis_outputs/results.json`. Data sources are
the public Polymarket Gamma API and the `yfinance` library, both of which require no API
keys. Re-running `fetch_data.py` reproduces `data/` byte-for-byte (the underlying historical
prices are immutable).

## Layout
- `main.tex` / `references.bib` / `neurips_2024.sty` — paper source (NeurIPS 2024 format)
- `scripts/fetch_data.py` — pulls Polymarket bucket markets + Yahoo Finance series
- `scripts/analysis.py` — implied-expected-GDP aggregation, Kalman filter, figures
- `scripts/requirements.txt` — pinned package versions
- `data/` — fetched panels (~236 KB)
- `analysis_outputs/results.json` — every reported number with units
- `figure_*.png` — plots referenced by `main.tex`
