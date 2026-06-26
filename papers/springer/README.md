# Springer Manuscript Package

This directory stores the Springer Nature version of the manuscript.

Main files:

- `aej-sn-article.tex`: main Springer LaTeX manuscript.
- `aej-sn-article.pdf`: compiled Springer PDF.
- `aej-refs.bib`: bibliography database.
- `sn-jnl.cls` and `sn-mathphys-num.bst`: Springer template/style files.

The figure path points to `../cs/figures/`, so keep the `papers/cs/figures/`
directory available when compiling this version.

Build from this directory:

```bash
pdflatex aej-sn-article
bibtex aej-sn-article
pdflatex aej-sn-article
pdflatex aej-sn-article
```

 