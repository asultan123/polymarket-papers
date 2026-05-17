# polymarket-papers

10 NeurIPS-format research papers exploring Polymarket prediction-market dynamics. Generated May 2026 by Claude Opus 4.7 sub-agents tasked to use real data + verify citations + produce reproducible methodology scripts.

Each `papers/<slug>/` directory contains:
- `main.tex` — paper source
- `references.bib` — citations (verified against Scholar / arXiv)
- `scripts/` — Python scripts that produced the numbers in the paper
- Figure source files

To compile any paper to PDF:
```
https://latexonline.cc/compile?git=https://github.com/asultan123/polymarket-papers&target=papers/<slug>/main.tex&command=pdflatex&force=true
```
