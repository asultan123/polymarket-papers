# Polymarket vs. economists on inflation forecasts

NeurIPS-formatted workshop paper comparing Polymarket's bracket-implied
CPI forecasts to the Federal Reserve Bank of Cleveland's daily inflation
nowcasting model, on US CPI prints from January 2025 through April 2026.

## How to reproduce the numbers in this paper

```bash
cd scripts
pip install -r requirements.txt
python3 fetch_data.py        # downloads Polymarket events + price histories
                              # and Cleveland Fed nowcast JSON (~3 min)
python3 analysis.py           # computes MAE / RMSE / Brier / paired t-test
                              # -> ../data/analysis_results.json
                              # -> ../data/per_event_table.csv
python3 figure_1.py           # -> ../figure_1.png
```

Both API calls (Polymarket Gamma + CLOB, Cleveland Fed webcharts) are
public and require no auth.  The cached `data/` directory is excluded
from version control via `.gitignore`; re-run `fetch_data.py` to
regenerate it.

The LaTeX source is in `main.tex`; the bibliography in `references.bib`;
the NeurIPS 2024 style file in `neurips_2024.sty`.
