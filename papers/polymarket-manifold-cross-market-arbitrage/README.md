# Cross-Market Price Gaps Between Polymarket and Manifold Markets

Snapshot study comparing Polymarket (real-money) and Manifold (play-money) binary market prices. The snapshot used in the paper was taken 17 May 2026.

## Reproducing the numbers

```bash
cd scripts
pip install -r requirements.txt
python fetch_data.py     # ~5-10 min, writes data/{polymarket,manifold}_markets.json (~49 MB; gitignored)
python analysis.py       # writes results/matched_pairs.csv and results/summary.json
python figures.py        # writes ../figure_{1,2,3}.png
```

`fetch_data.py` hits the public unauthenticated Polymarket Gamma and Manifold v0 APIs. The data is not redistributed because (a) the dumps are ~50 MB and (b) the snapshot drifts within minutes. Re-running on a fresh day will produce different paired markets but the same pipeline; the numbers in the paper will not replicate exactly.

`analysis.py` is deterministic given a fixed input (random seed `20260517`). All numbers in `results/summary.json` are the same numbers cited in `main.tex`.

## Files

- `main.tex` — paper source (NeurIPS 2024 style, `preprint` option). No LaTeX is included here; compile with `pdflatex main && bibtex main && pdflatex main && pdflatex main` on a machine that has TeX.
- `references.bib` — bibliography (manually verified entries; the `main.tex` also includes a `thebibliography` block so it builds without `bibtex` if needed).
- `neurips_2024.sty` — official NeurIPS 2024 style file (Roman Garnett, March 2024 revision).
- `scripts/` — three Python scripts and pinned `requirements.txt`.
- `figure_{1,2,3}.png` — figures included by `main.tex`.
- `results/matched_pairs.csv`, `results/summary.json` — analysis outputs.

The `data/` directory (raw API dumps) is gitignored.
